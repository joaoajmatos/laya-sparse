"""Tier F candidates A1 and B1 (laya:008, plan v2 section 3), applied outside ``laya/``.

* **A1** the encoder's global-attention layers become block-local (block 128, three-block span) with the
  question header's global tokens kept global. ``a1`` is the CPU path (``kernels/block_global.py``);
  ``a1_mask`` is the mask-only reference (dense SDPA with the same pattern as a mask), a quality diagnostic
  that also runs on a GPU agent. Rescue twin: A2, block 256.
* **B1** the decision head's attention layers attend only to keys in S = G plus every position within
  ``window`` (128) of a position in G. ``b1`` computes the head only on S (marker and CLS outputs are the
  only ones read, and both are in G, so the result is the reference's); ``b1_mask`` runs the dense head with
  the key restriction as a mask. Rescue twin: B2, window 256.

Plan v2.0 section 3 (literal wording, confirmed by the Research Lead): G = [CLS], the question tokens (through the first
[SEP]) and the option [MASK] markers. Option description tokens, the closing [SEP] and the state are NOT global.
(The ``header`` mode, the whole span through the second [SEP], is kept only as a tested alternative.)
G (the global tokens) is a function of the input ids only (``global_tokens``): never labels, never the state.
The input layout is ``[CLS] question [SEP] ([MASK] option)* [SEP] state [SEP]`` (laya/common.py).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

GLOBAL_MODES = ("header", "markers")
DEFAULT_GLOBAL_MODE = "markers"    # plan v2.0 s.3 (Research Lead, 2026-10-07): [CLS] + question tokens + option [MASK] markers

from .variants import CANDIDATES, _key_valid     # noqa: E402  (the table lives with the variant names; one key-validity reader)


class CandidateError(RuntimeError):
    """A candidate cannot be applied as asked."""


# --------------------------------------------------------------------------- the global-token set

def global_tokens(ids: torch.Tensor, valid: torch.Tensor, cls_id: int, sep_id: int, mask_id: int,
                  mode: str = DEFAULT_GLOBAL_MODE) -> torch.Tensor:
    """Boolean ``[B, L]``: the global tokens G of each row.

    ``markers`` (plan v2.0 section 3): [CLS] (position 0), the question tokens strictly between [CLS] and the first
    [SEP], and each option [MASK] marker (a [MASK] before the second [SEP]). Not global: the [SEP] tokens, the option
    description tokens after each marker, and all state tokens. A row with no [SEP] cannot be localized and has every
    valid token global (the candidate layers are then native for that row).
    ``header`` (alternative, not the gate's): positions 0 through the second [SEP], separators and option texts
    included; a row without a second [SEP] has every valid token global.
    Padding is never global.
    """
    if mode not in GLOBAL_MODES:
        raise CandidateError("global-token mode must be one of %s, got %r" % (GLOBAL_MODES, mode))
    valid = valid.to(torch.bool)
    is_sep = (ids == sep_id) & valid
    c = torch.cumsum(is_sep.to(torch.long), dim=-1)
    before = c - is_sep.to(torch.long)                                   # separators strictly before each position
    if mode == "header":
        g = before < 2                                                   # up to and including the second [SEP]
        localizable = c[:, -1:] >= 2
    else:
        pos0 = torch.zeros_like(is_sep)
        pos0[:, 0] = True
        g = pos0 | ((before < 1) & ~is_sep) | ((ids == mask_id) & (before < 2))
        localizable = c[:, -1:] >= 1
    return torch.where(localizable, g, torch.ones_like(g)) & valid


class _Context:
    """The G of the forward in flight, written by an encoder pre-hook and read by the replaced layers."""

    def __init__(self):
        self.g: Optional[torch.Tensor] = None
        self.valid: Optional[torch.Tensor] = None


def _special_ids(agent) -> Tuple[int, int, int]:
    tok = agent.tok
    ids = (tok.cls_token_id, tok.sep_token_id, tok.mask_token_id)
    if any(i is None for i in ids):
        raise CandidateError("the tokenizer lacks a [CLS], [SEP] or [MASK] id; global tokens cannot be located")
    return int(ids[0]), int(ids[1]), int(ids[2])


def _install_context(agent, mode: str) -> Tuple[_Context, Any]:
    ctx = _Context()
    cls_id, sep_id, mask_id = _special_ids(agent)

    def pre(_module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        att = kwargs.get("attention_mask")
        if ids is None:
            return None
        valid = torch.ones_like(ids, dtype=torch.bool) if att is None else att.to(torch.bool)
        ctx.valid = valid
        ctx.g = global_tokens(ids, valid, cls_id, sep_id, mask_id, mode)
        return None
    handle = agent.model.encoder.register_forward_pre_hook(pre, with_kwargs=True)
    return ctx, handle


# --------------------------------------------------------------------------- A1

def _global_layers(encoder):
    from .audit import _layer_type
    return [(i, layer) for i, layer in enumerate(encoder.layers) if _layer_type(i, layer, encoder.config)[0] == "global"]


def _a_forward(attn, ctx: _Context, block: int, mask_only: bool):
    from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb
    from .kernels.block_global import block_global_attention, block_global_reference
    kernel = block_global_reference if mask_only else block_global_attention

    def forward(hidden_states, position_embeddings=None, attention_mask=None, **kwargs):
        if ctx.g is None:
            raise CandidateError("global-token context missing: the layer ran outside the encoder forward")
        input_shape = hidden_states.shape[:-1]
        B, L = hidden_states.shape[0], hidden_states.shape[1]
        qkv = attn.Wqkv(hidden_states).view(*input_shape, 3, -1, attn.head_dim)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(dim=-3))
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
        kv = _key_valid(attention_mask, B, L)
        valid = ctx.valid if kv is None else kv
        out = kernel(q, k, v, block, valid, ctx.g)
        out = out.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return attn.out_drop(attn.Wo(out)), None
    return forward


def apply_a(agent, block: int, mask_only: bool, mode: str = DEFAULT_GLOBAL_MODE) -> Dict[str, Any]:
    encoder = agent.model.encoder
    layers = _global_layers(encoder)
    if not layers:
        raise CandidateError("the loaded encoder has no global-attention layers; candidate A does not apply")
    ctx, handle = _install_context(agent, mode)
    for _, layer in layers:
        layer.attn.forward = _a_forward(layer.attn, ctx, block, mask_only)

    def restore() -> None:
        handle.remove()
        for _, layer in layers:
            layer.attn.__dict__.pop("forward", None)
    return {"candidate": "A", "layers": [i for i, _ in layers], "block": block, "global_mode": mode,
            "mask_only": mask_only, "restore": restore}


# --------------------------------------------------------------------------- B1

def b_keep(g: torch.Tensor, valid: torch.Tensor, window: int) -> torch.Tensor:
    """S: valid positions within `window` of any global token (global tokens included). ``[B, L]`` bool."""
    near = F.max_pool1d(g.to(torch.float32)[:, None, :], kernel_size=2 * window + 1, stride=1, padding=window)[:, 0] > 0
    return near & valid.to(torch.bool)


def head_layer(layer, x: torch.Tensor, key_ok: torch.Tensor) -> torch.Tensor:
    """One pre-norm ``nn.TransformerEncoderLayer`` (eval) with the keys restricted by ``key_ok`` ``[B, T]`` bool."""
    if layer.training:
        raise CandidateError("the B1 head layer function is for inference (eval mode)")
    if not layer.norm_first:
        raise CandidateError("expected a pre-norm head layer (norm_first=True)")
    sa = layer.self_attn
    B, T, E = x.shape
    H = sa.num_heads
    a = layer.norm1(x)
    q, k, v = (t.unflatten(-1, (H, E // H)).transpose(1, 2)
               for t in F.linear(a, sa.in_proj_weight, sa.in_proj_bias).chunk(3, dim=-1))
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=key_ok[:, None, None, :])
    x = x + sa.out_proj(o.transpose(1, 2).reshape(B, T, E))
    return x + layer.linear2(layer.activation(layer.linear1(layer.norm2(x))))


def _decision_tail(model, h_markers, pooled_src, marker_mask):
    """The part of ``DecisionModel.forward`` after the head: scorer, entropy features, act head."""
    logits = model.scorer(h_markers).squeeze(-1).float()
    logits = logits.masked_fill(~marker_mask, -1e4)
    p = torch.softmax(logits.detach(), -1)
    k = marker_mask.sum(-1).clamp(min=2).float()
    ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
    if p.size(-1) >= 2:
        top2 = p.topk(2, -1).values
    else:
        top1 = p.topk(1, -1).values
        top2 = torch.cat([top1, torch.zeros_like(top1)], dim=-1)
    feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
    return logits, model.act_head(torch.cat([pooled_src.float(), feats], -1))


def _b_forward(model, cls_id, sep_id, mask_id, mode: str, window: int, mask_only: bool, info: Dict[str, Any]):
    def forward(input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder: bool = False):
        h = model.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = h + model.type_emb(qtype)[:, None, :]
        valid = attention_mask.to(torch.bool)
        g = global_tokens(input_ids, valid, cls_id, sep_id, mask_id, mode)
        keep = b_keep(g, valid, window)
        info["last_keep_fraction"] = float(keep.sum()) / max(1, int(valid.sum()))
        idx_m = marker_pos.clamp(min=0)
        if mask_only:
            for layer in model.head.layers:
                h = head_layer(layer, h, keep)
            m = torch.gather(h, 1, idx_m[:, :, None].expand(-1, -1, h.size(-1)))
            return _decision_tail(model, m, h[:, 0], marker_mask)
        B, L, d = h.shape
        counts = keep.sum(-1)
        smax = int(counts.max())
        order = torch.argsort((~keep).to(torch.int8), dim=-1, stable=True)[:, :smax]       # S in position order, padded
        sv = torch.arange(smax, device=h.device)[None, :] < counts[:, None]
        hc = h.gather(1, order[:, :, None].expand(B, smax, d))
        for layer in model.head.layers:
            hc = head_layer(layer, hc, sv)
        rank = torch.cumsum(keep.to(torch.long), dim=-1) - 1                                # position in the compact sequence
        pos_m = rank.gather(1, idx_m)
        m = torch.gather(hc, 1, pos_m[:, :, None].expand(-1, -1, d))
        return _decision_tail(model, m, hc[:, 0], marker_mask)
    return forward


def apply_b(agent, window: int, mask_only: bool, mode: str = DEFAULT_GLOBAL_MODE) -> Dict[str, Any]:
    model = agent.model
    if getattr(model, "head", None) is None:
        raise CandidateError("the loaded model has no decision-head attention layers; candidate B does not apply")
    cls_id, sep_id, mask_id = _special_ids(agent)
    info: Dict[str, Any] = {}
    model.forward = _b_forward(model, cls_id, sep_id, mask_id, mode, window, mask_only, info)

    def restore() -> None:
        model.__dict__.pop("forward", None)
    return {"candidate": "B", "layers": ["head.%d" % i for i in range(len(model.head.layers))], "window": window,
            "global_mode": mode, "mask_only": mask_only, "restore": restore, "info": info}


# --------------------------------------------------------------------------- dispatch

def apply_candidate(agent, name: str, mode: str = DEFAULT_GLOBAL_MODE) -> Dict[str, Any]:
    """Apply candidate variant `name` (see ``CANDIDATES``); returns its record with a ``restore`` callable."""
    if name not in CANDIDATES:
        raise CandidateError("unknown candidate %r; choose from %s" % (name, sorted(CANDIDATES)))
    kind, value, mask_only = CANDIDATES[name]
    rec = apply_a(agent, value, mask_only, mode) if kind == "A" else apply_b(agent, value, mask_only, mode)
    rec["variant"] = name
    return rec
