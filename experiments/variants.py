"""Optimized-system variants: the decision head's fast-path switch and CPU int8 quantization (T042).

Specs: specs/002-decision-benchmark-baselines (research.md R14; FR-017). These are reported apart
from the architectural baselines and applied outside ``laya/``:

* ``fastpath_off``   PyTorch's `TransformerEncoderLayer` inference fast path disabled (Phase 1 R17).
* ``local_exact``    exact |i-j| <= half-window kernel in the encoder's local (sliding-window) layers
                     instead of dense masked attention (laya:007); the head fast path stays native.
* ``local_exact_fastpath_off``  the same plus ``fastpath_off``: the **optimized native** that
                     laya:008 and the compression gate use as the cost reference.
* ``a1``, ``a2`` ... Tier F candidates (laya:008, experiments/candidates.py): A = global layers made block-local with
                     the header tokens global, B = decision head restricted to header windows. ``<name>_mask`` is the
                     mask-only reference (a quality diagnostic that also runs on a GPU agent); the plain name is the CPU
                     path with real skipped work and is CPU-only. a2/b2 are the declared rescue configurations.
* ``int8_encoder``   dynamic int8 quantization of the encoder's `Linear` layers only; the decision
                     head and its fast path stay native.
* ``int8_all_nofast`` dynamic int8 quantization of every `Linear` layer, with the fast path off.
                     With the fast path on, the head's fast-path check reads ``weight`` from a
                     quantized layer, which is a method, and fails (``'function' object has no
                     attribute 'device'``, 2026-09-29 spike), so that combination is refused.

Quantization is CPU-only. On another device the variant is ``unsupported`` with that reason.
"""
from __future__ import annotations

import contextlib
import copy
import warnings
from typing import Any, Dict, Optional

#: Tier F candidate variants -> (candidate, config value, mask_only); the code is in candidates.py (imported lazily).
CANDIDATES = {
    "a1": ("A", 128, False), "a1_mask": ("A", 128, True),      # block size
    "a2": ("A", 256, False), "a2_mask": ("A", 256, True),      # rescue
    "b1": ("B", 128, False), "b1_mask": ("B", 128, True),      # window half-width
    "b2": ("B", 256, False), "b2_mask": ("B", 256, True),      # rescue
}
CANDIDATE_VARIANTS = tuple(sorted(CANDIDATES))
VARIANTS = ("none", "fastpath_off", "local_exact", "local_exact_fastpath_off", "int8_encoder",
            "int8_all_nofast") + CANDIDATE_VARIANTS
LOCAL_EXACT_VARIANTS = ("local_exact", "local_exact_fastpath_off")
QUANT_API = "torch.ao.quantization.quantize_dynamic(qint8)"


class VariantError(RuntimeError):
    """A variant cannot be applied as asked."""


def unsupported_reason(variant: str, device: str) -> Optional[str]:
    """Why `variant` cannot run on `device`, or None when it can."""
    if variant not in VARIANTS:
        return "unknown variant %r" % variant
    if variant == "none":
        return None
    if variant in CANDIDATE_VARIANTS and variant.endswith("_mask"):
        return None                       # mask-only references run on any device (quality diagnostics)
    if device != "cpu":
        what = ("int8 dynamic quantization" if variant.startswith("int8")
                else "the exact local-attention kernel" if variant.startswith("local_exact")
                else "the candidate CPU path" if variant in CANDIDATE_VARIANTS else "the mha fast-path switch")
        return "%s is a CPU-only variant; not run on %s" % (what, device)
    return None


def _quantize(module) -> None:
    import torch
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")        # PyTorch's notice that quantize_dynamic is moving to torchao
        torch.ao.quantization.quantize_dynamic(module, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)


def count_quantized(module) -> int:
    """Number of dynamically quantized Linear layers in `module`."""
    import torch.ao.nn.quantized.dynamic as qdyn
    return sum(1 for m in module.modules() if isinstance(m, qdyn.Linear))


def _local_layers(encoder):
    """``[(index, layer)]`` of the encoder layers the audit calls local (sliding-window attention)."""
    from .audit import _layer_type
    ecfg = encoder.config
    return [(i, layer) for i, layer in enumerate(encoder.layers) if _layer_type(i, layer, ecfg)[0] == "local"]


def _half_width(attn, ecfg) -> int:
    """The window half-width as the loaded model states it (never hard-coded)."""
    sw = getattr(attn, "sliding_window", None)
    if isinstance(sw, int) and sw > 0:
        return sw - 1               # transformers stores config.sliding_window + 1 on the module
    cfg = getattr(ecfg, "sliding_window", None)
    if isinstance(cfg, int) and cfg > 0:
        return cfg
    loc = getattr(ecfg, "local_attention", None)
    if isinstance(loc, int) and loc > 1:
        return loc // 2
    raise VariantError("cannot read the local-attention half-width from the loaded encoder")


def _key_valid(attention_mask, B: int, L: int):
    """Per-key validity ``[B, L]`` from the 4-D mask transformers hands the layer.

    Key j is a real token exactly when the mask lets query j see itself (the diagonal), because the
    sliding-window mask always contains the diagonal for real tokens and the padding mask removes padded
    keys. None means no padding.
    """
    import torch
    if attention_mask is None:
        return None
    m = attention_mask
    if m.dim() == 4:
        diag = m[:, 0].diagonal(dim1=-2, dim2=-1)
    elif m.dim() == 2:
        diag = m
    else:
        raise VariantError("unexpected attention mask rank %d" % m.dim())
    if diag.dtype != torch.bool:
        diag = diag == 0 if diag.is_floating_point() and bool((diag <= 0).all()) else diag > 0
    return diag.expand(B, L) if diag.shape[0] == 1 and B > 1 else diag


def _exact_forward(attn, half_width: int):
    """A replacement ``forward`` for a ModernBERT attention module that runs the exact band kernel."""
    from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb
    from .kernels.local_exact import local_exact_attention

    def forward(hidden_states, position_embeddings=None, attention_mask=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        qkv = attn.Wqkv(hidden_states).view(*input_shape, 3, -1, attn.head_dim)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(dim=-3))
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
        B, L = hidden_states.shape[0], hidden_states.shape[1]
        out = local_exact_attention(q, k, v, half_width, key_valid=_key_valid(attention_mask, B, L))
        out = out.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return attn.out_drop(attn.Wo(out)), None
    return forward


def apply_local_exact(agent) -> Dict[str, Any]:
    """Replace the local layers' attention with the exact kernel; returns ``{layers, half_width, restore}``."""
    encoder = agent.model.encoder
    layers = _local_layers(encoder)
    if not layers:
        raise VariantError("the loaded encoder has no local (sliding-window) layers; local_exact does not apply")
    widths = {_half_width(layer.attn, encoder.config) for _, layer in layers}
    if len(widths) != 1:
        raise VariantError("local layers disagree on the window half-width: %s" % sorted(widths))
    hw = widths.pop()
    for _, layer in layers:
        layer.attn.forward = _exact_forward(layer.attn, hw)    # instance attribute shadows the class method

    def restore() -> None:
        for _, layer in layers:
            layer.attn.__dict__.pop("forward", None)
    return {"layers": [i for i, _ in layers], "half_width": hw, "restore": restore}


def apply_variant(agent, variant: str, reversible: bool = False) -> Dict[str, Any]:
    """Apply `variant` to the loaded agent; returns a record (and ``restore`` when `reversible`).

    Non-reversible application (the default) is for per-condition subprocesses. `reversible` keeps a copy
    of the model so tests can undo it.
    """
    import torch
    if variant not in VARIANTS:
        raise VariantError("unknown variant %r; choose from %s" % (variant, VARIANTS))
    prev_fast = bool(torch.backends.mha.get_fastpath_enabled())
    record: Dict[str, Any] = {"variant": variant, "quantization_api": None, "quantized_layers": 0,
                              "fastpath_before": prev_fast}
    saved = copy.deepcopy(agent.model) if reversible and variant.startswith("int8") else None

    def restore() -> None:
        torch.backends.mha.set_fastpath_enabled(prev_fast)
        if saved is not None:
            agent.model = saved

    if variant == "none":
        pass
    elif variant == "fastpath_off":
        torch.backends.mha.set_fastpath_enabled(False)
    elif variant in LOCAL_EXACT_VARIANTS:
        if variant == "local_exact_fastpath_off":
            torch.backends.mha.set_fastpath_enabled(False)
        rec = apply_local_exact(agent)
        inner_restore = rec.pop("restore")
        record.update(local_exact_layers=rec["layers"], local_exact_half_width=rec["half_width"])
        _restore = restore

        def restore() -> None:        # noqa: F811
            inner_restore()
            _restore()
    elif variant in CANDIDATE_VARIANTS:
        from . import candidates
        composed = None
        if not variant.endswith("_mask"):
            # The CPU path runs on top of optimized native (laya:007's exact local kernel; the loader turns the head
            # fast path off), so F4 isolates the candidate's saving. Mask-only references stay on plain native.
            composed = apply_local_exact(agent)
            record.update(composed_with="local_exact", local_exact_layers=composed["layers"],
                          local_exact_half_width=composed["half_width"])
        try:
            rec = candidates.apply_candidate(agent, variant)
        except Exception:
            if composed is not None:
                composed["restore"]()
            raise
        inner_restore = rec.pop("restore")
        if composed is not None:
            _first = inner_restore

            def inner_restore() -> None:        # noqa: F811
                _first()
                composed["restore"]()
        record.update({k: v for k, v in rec.items() if k not in ("info", "layers")}, candidate_layers=rec["layers"])
        _restore = restore

        def restore() -> None:        # noqa: F811
            inner_restore()
            _restore()
    elif variant == "int8_encoder":
        _quantize(agent.model.encoder)
        record.update(quantization_api=QUANT_API, quantized_layers=count_quantized(agent.model.encoder))
    else:   # int8_all_nofast
        if prev_fast:
            raise VariantError("int8_all_nofast needs the mha fast path off: quantized weights break the "
                               "TransformerEncoderLayer fast-path check (research.md R14); "
                               "run with --mha-fastpath off or use int8_encoder")
        _quantize(agent.model)
        record.update(quantization_api=QUANT_API, quantized_layers=count_quantized(agent.model))
    record["fastpath_after"] = bool(torch.backends.mha.get_fastpath_enabled())
    if reversible:
        record["restore"] = restore
    return record


@contextlib.contextmanager
def fastpath(enabled: bool):
    """Set PyTorch's mha fast path for a block and restore it after."""
    import torch
    prev = bool(torch.backends.mha.get_fastpath_enabled())
    torch.backends.mha.set_fastpath_enabled(bool(enabled))
    try:
        yield
    finally:
        torch.backends.mha.set_fastpath_enabled(prev)


def load_for_variant(model: str, revision: Optional[str], threads: Optional[int], variant: str):
    """Load the agent on CPU with `variant` applied; returns ``(agent, load_info, variant_record)``.

    ``fastpath_off``, ``local_exact_fastpath_off`` and ``int8_all_nofast`` load with the fast path off (a labelled runtime setting, as in
    Phase 1). ``int8_encoder`` keeps the native fast path.
    """
    from .runner import load_agent
    off = ("fastpath_off", "local_exact_fastpath_off", "int8_all_nofast") + CANDIDATE_VARIANTS
    agent, info = load_agent(model, revision, threads, mha_fastpath=variant not in off)
    rec = apply_variant(agent, variant if variant != "fastpath_off" else "none")
    rec["variant"] = variant
    rec["fastpath_after"] = info["mha_fastpath"]
    return agent, info, rec
