"""`golden export`: forward-pass goldens for the fixture decision model (laya:004 US2, FR-006).

The per-stage tensors come from running `DecisionModel.forward` stage by stage (encoder with hooks, then the head
layers, `scorer`, `act_head`) exactly as `laya/common.py` does; the final outputs are asserted equal to a real
`model(...)` call so the decomposition cannot drift from the source of truth.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import common

#: Fast-path on vs off must agree this closely or export fails (raya:002 acceptance 2).
FASTPATH_TOL = 1e-6

STATE_SHORT = "the customer asked for a refund on a late delivery"
Q_CHOICE = {"type": "choice", "instructions": "which option best matches the document",
            "criteria": {"continue": "proceed with the request", "review": "open a review", "stop": "halt the request"}}
Q_SCORE = {"type": "score", "instructions": "how risky is the request",
           "criteria": ["zero risk", "low risk", "high risk"]}
Q_NOUL = {"type": "noul", "instructions": "this document needs review",
          "criteria": {"false": "no review is needed", "true": "a review is needed"}}

CASES = {
    # one local window of the fixture encoder (local_attention=16) is crossed only slightly
    "short": dict(hidden=64, question=Q_CHOICE, state=STATE_SHORT, max_len=64, head_max_len=48, pad_to=None),
    # several blocks, crossing global (layers 0, 2) and local (layer 1) layers
    "medium": dict(hidden=64, question=Q_SCORE, state=(STATE_SHORT + " ") * 14, max_len=256, head_max_len=96,
                   pad_to=None),
    # batch of one, right-padded with [PAD] and a zero attention mask
    "padded": dict(hidden=64, question=Q_NOUL, state=STATE_SHORT * 2, max_len=128, head_max_len=64, pad_to=160),
    # hidden=128 gives the head 2 attention heads, so the fast path is a genuinely different code path
    "fastpath_off": dict(hidden=128, question=Q_CHOICE, state=(STATE_SHORT + " ") * 6, max_len=128,
                         head_max_len=64, pad_to=None, fastpath_off=True),
}


def config_dir_name(hidden: int) -> str:
    """Checkpoint-shaped directory beside the weights: encoder/config.json and rl_agent_config.json."""
    return "fixture-h%d" % hidden


def _tensor_of(x):
    return x[0] if isinstance(x, (tuple, list)) else x


def staged_forward(model, batch) -> Dict[str, Any]:
    """Run `DecisionModel.forward` stage by stage, returning every intermediate (fp32, eval, no grad)."""
    import torch
    out: Dict[str, Any] = {}
    enc = model.encoder
    hooks = []

    def keep(name):
        def hook(_m, _inp, res):
            out[name] = _tensor_of(res).detach().clone()
        return hook
    hooks.append(enc.embeddings.register_forward_hook(keep("embedding_output")))
    for i, layer in enumerate(enc.layers):
        hooks.append(layer.register_forward_hook(keep("encoder_layer_%02d" % i)))
    hooks.append(enc.final_norm.register_forward_hook(keep("encoder_final_norm")))
    ids, att = batch["input_ids"], batch["attention_mask"]
    try:
        with torch.no_grad():
            h = enc(input_ids=ids, attention_mask=att).last_hidden_state
    finally:
        for hk in hooks:
            hk.remove()
    out["encoder_output"] = h.detach().clone()
    with torch.no_grad():
        add = model.type_emb(batch["qtype"])[:, None, :]
        out["type_emb_add"] = add.detach().clone()
        h = h + add
        out["head_input"] = h.detach().clone()
        if model.head is not None:
            pad = ~att.bool()
            for i, layer in enumerate(model.head.layers):
                h = layer(h, src_key_padding_mask=pad)
                out["head_layer_%02d" % i] = h.detach().clone()
        idx = batch["marker_pos"].clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        out["marker_states"] = m.detach().clone()
        logits = model.scorer(m).squeeze(-1).float().masked_fill(~batch["marker_mask"], -1e4)
        out["logits"] = logits
        p = torch.softmax(logits, -1)
        out["probabilities"] = p
        k = batch["marker_mask"].sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        if p.size(-1) >= 2:
            top2 = p.topk(2, -1).values
        else:
            top1 = p.topk(1, -1).values
            top2 = torch.cat([top1, torch.zeros_like(top1)], dim=-1)
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        out["act_features"] = feats
        out["pooled"] = h[:, 0].float()
        out["act_logits"] = model.act_head(torch.cat([out["pooled"], feats], -1))
    return out


def _case_batch(agent, spec) -> Dict[str, Any]:
    """Input ids etc. built by Laya's own `_encode_state` + `collate_items` (no reimplementation)."""
    import torch
    from laya.common import collate_items
    q = agent._to_internal(spec["question"])
    items = agent._encode_state(spec["state"], ["q"], {"q": q}, spec["max_len"], spec["head_max_len"])
    b = collate_items([items], agent.tok.pad_token_id)
    unpadded = int(b["input_ids"].shape[1])
    if spec.get("pad_to") and spec["pad_to"] > unpadded:
        extra = spec["pad_to"] - unpadded
        b["input_ids"] = torch.cat([b["input_ids"], torch.full((1, extra), agent.tok.pad_token_id)], 1)
        b["attention_mask"] = torch.cat([b["attention_mask"], torch.zeros(1, extra, dtype=torch.long)], 1)
    b["unpadded_length"] = unpadded
    return b


def export_case(name: str, spec: Dict[str, Any], out: Path, weights_written: set) -> Dict[str, Any]:
    import torch
    out = Path(out)
    with common.FixtureAgent(spec["hidden"]) as fx:
        agent, model = fx.agent, fx.agent.model
        model.eval()
        batch = _case_batch(agent, spec)
        fast_off = bool(spec.get("fastpath_off"))
        prev = torch.backends.mha.get_fastpath_enabled()
        diff_vs_fastpath: Optional[float] = None
        try:
            torch.backends.mha.set_fastpath_enabled(not fast_off)
            staged = staged_forward(model, batch)
            with torch.no_grad():
                logits, act = model(batch["input_ids"], batch["attention_mask"], batch["marker_pos"],
                                    batch["marker_mask"], batch["qtype"])
            if not (torch.equal(logits, staged["logits"]) and torch.equal(act, staged["act_logits"])):
                raise RuntimeError("case %s: staged forward differs from DecisionModel.forward" % name)
            if fast_off:
                torch.backends.mha.set_fastpath_enabled(True)
                on = staged_forward(model, batch)
                diff_vs_fastpath = max(float((on[k] - staged[k]).abs().max())
                                       for k in ("logits", "probabilities", "act_logits", "marker_states"))
                if not diff_vs_fastpath <= FASTPATH_TOL:
                    raise RuntimeError("case %s: fast path on vs off differ by %.3g > %g"
                                       % (name, diff_vs_fastpath, FASTPATH_TOL))
        finally:
            torch.backends.mha.set_fastpath_enabled(prev)

        tensors = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"],
                   "position_ids": torch.arange(batch["input_ids"].shape[1])[None, :],
                   "marker_pos": batch["marker_pos"], "marker_mask": batch["marker_mask"], "qtype": batch["qtype"]}
        tensors.update(staged)
        case_dir = out / name
        common.save_tensors(case_dir / "tensors.safetensors", tensors)
        wname = "weights-h%d.safetensors" % spec["hidden"]
        if wname not in weights_written:
            common.save_tensors(out / wname, dict(model.state_dict()))
            weights_written.add(wname)
            # The checkpoint's own config files, byte for byte: what laya.Agent loaded these weights with (laya:006).
            cfg_dir = out / config_dir_name(spec["hidden"])
            (cfg_dir / "encoder").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(fx.path) / "encoder" / "config.json", cfg_dir / "encoder" / "config.json")
            shutil.copyfile(Path(fx.path) / "rl_agent_config.json", cfg_dir / "rl_agent_config.json")
        ecfg = model.encoder.config
        meta = common.base_meta(
            "forward-golden", case=name, weights_file="../" + wname, config_dir="../" + config_dir_name(spec["hidden"]),
            fixture_hidden=spec["hidden"],
            fixture_checkpoint_seed=1234, agent_config=agent.cfg, encoder_config=common.encoder_config(model),
            layer_types=list(getattr(ecfg, "layer_types", []) or []),
            input={"question": spec["question"], "state": spec["state"], "max_len": spec["max_len"],
                   "head_max_len": spec["head_max_len"], "pad_to": spec.get("pad_to"),
                   "sequence_length": int(batch["input_ids"].shape[1]), "unpadded_length": batch["unpadded_length"]},
            mha_fastpath=not fast_off, head_heads=model.head.layers[0].self_attn.num_heads,
            fastpath_off_max_abs_diff_vs_on=diff_vs_fastpath, fastpath_tolerance=FASTPATH_TOL,
            tensors={k: {"shape": list(v.shape), "dtype": str(v.dtype).replace("torch.", "")}
                     for k, v in sorted(tensors.items())})
        common.write_json(case_dir / "meta.json", meta)
        return meta


def export_all(out: Path, cases: Optional[List[str]] = None) -> Dict[str, Any]:
    out = Path(out)
    common.make_deterministic()
    written: set = set()
    return {name: export_case(name, CASES[name], out, written) for name in (cases or list(CASES))}
