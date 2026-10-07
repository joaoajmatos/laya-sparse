"""T036: fast-path switch and int8 quantization variants (experiments/variants.py)."""
import laya
import pytest
import torch

from experiments import variants as V


@pytest.fixture()
def agent(fastpath_checkpoint):
    a = laya.Agent(fastpath_checkpoint, device="cpu")
    yield a


Q = {"q": {"type": "choice", "instructions": "which option best matches", "criteria": {"a": "x", "b": "y", "c": "z"}}}
STATE = "the small red city is quiet in the morning and the harbor is warm " * 3


def probs(agent):
    return agent.predict(STATE, Q, max_len=200)["answers"]["q"]["probabilities"]


def test_fastpath_context_sets_and_restores():
    before = torch.backends.mha.get_fastpath_enabled()
    with V.fastpath(False):
        assert torch.backends.mha.get_fastpath_enabled() is False
    assert torch.backends.mha.get_fastpath_enabled() == before


def test_fastpath_off_agrees_with_native_within_1e4(agent):
    native = probs(agent)
    with V.fastpath(False):
        off = probs(agent)
    for k in native:
        assert abs(native[k] - off[k]) < 1e-4


def test_fastpath_off_variant_is_recorded_and_restorable(agent):
    before = torch.backends.mha.get_fastpath_enabled()
    rec = V.apply_variant(agent, "fastpath_off", reversible=True)
    try:
        assert rec["variant"] == "fastpath_off" and rec["fastpath_after"] is False and rec["fastpath_before"] == before
    finally:
        rec["restore"]()
    assert torch.backends.mha.get_fastpath_enabled() == before


def test_int8_encoder_quantizes_only_the_encoder_and_keeps_the_head_fastpath(agent):
    rec = V.apply_variant(agent, "int8_encoder", reversible=True)
    try:
        assert rec["quantization_api"] == V.QUANT_API and rec["quantized_layers"] > 0
        assert V.count_quantized(agent.model.encoder) == rec["quantized_layers"]
        assert V.count_quantized(agent.model.head) == 0                  # the decision head stays native
        assert rec["fastpath_after"] is True
        p = probs(agent)                                                  # runs, with the head on its fast path
        assert abs(sum(p.values()) - 1.0) < 1e-3
    finally:
        rec["restore"]()
    assert V.count_quantized(agent.model) == 0                            # restored


def test_int8_all_needs_the_fast_path_off(agent):
    with pytest.raises(V.VariantError) as err:
        V.apply_variant(agent, "int8_all_nofast")
    assert "fast" in str(err.value)
    with V.fastpath(False):
        rec = V.apply_variant(agent, "int8_all_nofast", reversible=True)
        try:
            assert rec["quantized_layers"] > V.count_quantized(agent.model.encoder) - 1
            assert V.count_quantized(agent.model.head) > 0
            p = probs(agent)
            assert abs(sum(p.values()) - 1.0) < 1e-3
        finally:
            rec["restore"]()


def test_quantized_output_is_close_to_native_but_not_identical(agent):
    native = probs(agent)
    rec = V.apply_variant(agent, "int8_encoder", reversible=True)
    try:
        q = probs(agent)
    finally:
        rec["restore"]()
    assert max(abs(native[k] - q[k]) for k in native) < 0.2              # random tiny weights: a loose bound


def test_variants_are_cpu_only_and_unknown_ones_are_refused():
    assert V.unsupported_reason("none", "gpu") is None
    for v in ("fastpath_off", "int8_encoder", "int8_all_nofast"):
        assert V.unsupported_reason(v, "cpu") is None
        assert "CPU-only" in V.unsupported_reason(v, "gpu")
    assert "unknown" in V.unsupported_reason("fp16", "cpu")
    with pytest.raises(V.VariantError):
        V.apply_variant(object(), "fp16")


# --------------------------------------------------------------------------- laya:007: exact local attention

def _encode(agent, lengths, L=96, seed=0):
    g = torch.Generator().manual_seed(seed)
    vocab = agent.model.encoder.config.vocab_size
    ids = torch.randint(5, vocab, (len(lengths), L), generator=g)
    att = (torch.arange(L)[None, :] < torch.tensor(lengths)[:, None]).long()
    ids = ids * att
    with torch.no_grad():
        return agent.model.encoder(input_ids=ids, attention_mask=att).last_hidden_state, att


def test_local_exact_replaces_exactly_the_audit_local_layers_and_restores(agent):
    enc = agent.model.encoder
    expected = [i for i, layer in enumerate(enc.layers) if layer.attn.sliding_window is not None]
    assert expected, "the fixture must have a local layer"
    rec = V.apply_variant(agent, "local_exact", reversible=True)
    try:
        assert rec["local_exact_layers"] == expected
        assert rec["local_exact_half_width"] == enc.config.sliding_window == enc.config.local_attention // 2
        for i, layer in enumerate(enc.layers):
            assert ("forward" in layer.attn.__dict__) == (i in expected)      # global layers untouched
    finally:
        rec["restore"]()
    assert not any("forward" in layer.attn.__dict__ for layer in enc.layers)


def test_local_exact_encoder_output_matches_native_with_padding(agent):
    lengths = [96, 80, 33]                               # window is 8: every row exceeds it
    native, att = _encode(agent, lengths)
    rec = V.apply_variant(agent, "local_exact", reversible=True)
    try:
        exact, _ = _encode(agent, lengths)
    finally:
        rec["restore"]()
    valid = att.bool()
    assert float((native - exact).abs()[valid].max()) < 1e-5
    assert float((native - exact).abs().max()) < 1e-4    # padded rows too (they feed nothing)


def test_local_exact_probabilities_match_native(agent):
    native = probs(agent)
    rec = V.apply_variant(agent, "local_exact", reversible=True)
    try:
        exact = probs(agent)
    finally:
        rec["restore"]()
    for k in native:
        assert abs(native[k] - exact[k]) < 1e-5


def test_local_exact_is_not_a_noop(agent):
    """The replaced forward really runs (a mask that is wider than the window would also change outputs)."""
    calls = []
    import experiments.kernels.local_exact as K
    orig = K.local_exact_attention
    K.local_exact_attention = lambda *a, **kw: (calls.append(1), orig(*a, **kw))[1]
    rec = V.apply_variant(agent, "local_exact", reversible=True)
    try:
        probs(agent)
    finally:
        rec["restore"]()
        K.local_exact_attention = orig
    assert calls


def test_local_exact_fastpath_off_is_optimized_native_and_cpu_only(agent):
    before = torch.backends.mha.get_fastpath_enabled()
    rec = V.apply_variant(agent, "local_exact_fastpath_off", reversible=True)
    try:
        assert rec["fastpath_after"] is False and rec["local_exact_layers"]
    finally:
        rec["restore"]()
    assert torch.backends.mha.get_fastpath_enabled() == before
    assert not any("forward" in layer.attn.__dict__ for layer in agent.model.encoder.layers)
    for v in V.LOCAL_EXACT_VARIANTS:
        assert "CPU-only" in V.unsupported_reason(v, "gpu") and V.unsupported_reason(v, "cpu") is None


def test_local_exact_refuses_an_encoder_without_local_layers(agent):
    for layer in agent.model.encoder.layers:
        layer.attn.sliding_window = None
        layer.attention_type = "full_attention"
    with pytest.raises(V.VariantError, match="no local"):
        V.apply_variant(agent, "local_exact")


def test_key_validity_from_the_4d_mask_diagonal_equals_the_padding_mask():
    L = 20
    att = torch.tensor([[1] * 20, [1] * 13 + [0] * 7])
    pos = torch.arange(L)
    mask = ((pos[:, None] - pos[None, :]).abs() <= 4)[None, None] & att.bool()[:, None, None, :]
    assert torch.equal(V._key_valid(mask, 2, L), att.bool())
    assert V._key_valid(None, 2, L) is None
