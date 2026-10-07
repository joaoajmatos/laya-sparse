"""laya:008: candidates A1 and B1 (experiments/candidates.py): global tokens, mask-only vs CPU path, restore."""
import laya
import pytest
import torch

from experiments import candidates as C
from experiments import variants as V
from laya.common import build_sequence, collate_items

Q = {"t": "choice", "ins": "which option best matches", "options": ["x one", "y two", "z three"]}
QP = {"q": {"type": "choice", "instructions": "which option best matches", "criteria": {"a": "x", "b": "y", "c": "z"}}}
STATE = "the small red city is quiet in the morning and the harbor is warm " * 3
CLS, SEP, MASK = 2, 3, 4


@pytest.fixture()
def agent(fastpath_checkpoint):
    return laya.Agent(fastpath_checkpoint, device="cpu")


def probs(agent):
    return agent.predict(STATE, QP, max_len=200)["answers"]["q"]["probabilities"]


def _batch(agent, states, max_len=200):
    items = []
    for s in states:
        ids, markers = build_sequence(agent.tok, s, {"t": "choice", "ins": "pick one", "crit": {"a": "x one", "b": "y two", "c": "z"}}, max_len)
        items.append({"ids": ids, "markers": markers, "qtype": 0})
    return collate_items([items], agent.tok.pad_token_id)


def _logits(agent, b):
    with torch.no_grad():
        return agent.model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])


def test_global_tokens_header_markers_missing_and_padding():
    ids = torch.tensor([[2, 10, 11, 3, 4, 20, 21, 4, 22, 3, 30, 31, 3, 0, 0],
                        [2, 10, 3, 4, 20, 4, 21, 30, 31, 32, 0, 0, 0, 0, 0],      # no second [SEP]: header cut
                        [2, 10, 3, 4, 20, 3, 5, 6, 7, 3, 0, 0, 0, 0, 0]])
    valid = torch.tensor([[1] * 13 + [0] * 2, [1] * 10 + [0] * 5, [1] * 10 + [0] * 5], dtype=torch.bool)
    h = C.global_tokens(ids, valid, CLS, SEP, MASK, "header")
    assert h[0].nonzero().flatten().tolist() == list(range(10))
    assert h[1].nonzero().flatten().tolist() == list(range(10))                    # missing second [SEP]: every valid token
    assert h[2].nonzero().flatten().tolist() == list(range(6))
    m = C.global_tokens(ids, valid, CLS, SEP, MASK, "markers")
    assert m[0].nonzero().flatten().tolist() == [0, 1, 2, 4, 7]                    # CLS, question tokens, [MASK]s; no [SEP], no option text
    assert m[1].nonzero().flatten().tolist() == [0, 1, 3, 5]                       # no second [SEP] is fine: only the first is needed
    assert m[2].nonzero().flatten().tolist() == [0, 1, 3]
    no_sep = torch.tensor([[2, 10, 11, 12, 13, 0]])
    assert C.global_tokens(no_sep, torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.bool), CLS, SEP, MASK).sum() == 5
    assert not h[:, 13:].any() and not m[:, 13:].any()                              # padding is never global
    with pytest.raises(C.CandidateError):
        C.global_tokens(ids, valid, CLS, SEP, MASK, "bogus")


def test_b_keep_dilates_by_the_window_and_clips_at_the_ends():
    g = torch.zeros(2, 20, dtype=torch.bool)
    g[0, 2:4] = True
    g[1, 0] = True
    valid = torch.ones(2, 20, dtype=torch.bool)
    valid[1, 15:] = False
    k = C.b_keep(g, valid, 3)
    assert k[0].nonzero().flatten().tolist() == list(range(0, 7))
    assert k[1].nonzero().flatten().tolist() == [0, 1, 2, 3]


def test_head_layer_function_equals_the_native_layer(agent):
    layer = agent.model.head.layers[0].eval()
    x = torch.randn(2, 12, layer.linear1.in_features)
    pad = torch.zeros(2, 12, dtype=torch.bool)
    pad[1, 9:] = True
    with torch.no_grad():
        native = layer(x, src_key_padding_mask=pad)
        mine = C.head_layer(layer, x, ~pad)
    assert float((native - mine).abs()[~pad].max()) < 1e-5


@pytest.mark.parametrize("name", ["a1", "a1_mask", "b1", "b1_mask"])
def test_variants_are_registered_and_cpu_only_unless_mask_only(name):
    assert name in V.VARIANTS
    if name.endswith("_mask"):
        assert V.unsupported_reason(name, "gpu") is None
    else:
        assert "CPU-only" in V.unsupported_reason(name, "gpu")
    assert V.unsupported_reason(name, "cpu") is None


def test_a_replaces_exactly_the_global_layers_and_restores(agent):
    enc = agent.model.encoder
    expected = [i for i, l in enumerate(enc.layers) if l.attn.sliding_window is None]
    assert expected
    rec = C.apply_a(agent, 8, False)
    try:
        assert rec["layers"] == expected and rec["block"] == 8 and rec["global_mode"] == "markers"
        for i, l in enumerate(enc.layers):
            assert ("forward" in l.attn.__dict__) == (i in expected)             # local layers untouched
    finally:
        rec["restore"]()
    assert not any("forward" in l.attn.__dict__ for l in enc.layers) and not enc._forward_pre_hooks


def test_a_cpu_path_equals_mask_only_and_differs_from_native(agent):
    b = _batch(agent, [STATE, STATE[:90]])
    agent.model.eval()
    nl, _ = _logits(agent, b)
    out = {}
    for mask_only in (False, True):
        rec = C.apply_a(agent, 8, mask_only)
        try:
            out[mask_only] = _logits(agent, b)[0]
        finally:
            rec["restore"]()
    assert float((out[False] - out[True]).abs().max()) < 1e-5
    assert float((out[True] - nl).abs().max()) > 1e-5                              # a real change, not a no-op
    assert float((_logits(agent, b)[0] - nl).abs().max()) < 1e-6                   # restored


def test_a_with_a_block_covering_the_input_is_native(agent):
    b = _batch(agent, [STATE, STATE[:90]])
    agent.model.eval()
    nl, na = _logits(agent, b)
    rec = C.apply_a(agent, 1024, False)
    try:
        l, a = _logits(agent, b)
    finally:
        rec["restore"]()
    assert float((l - nl).abs().max()) < 1e-5 and float((a - na).abs().max()) < 1e-5


def test_b_with_everything_kept_equals_native_forward(agent):
    b = _batch(agent, [STATE, STATE[:90]])
    agent.model.eval()
    nl, na = _logits(agent, b)
    for mask_only in (False, True):
        rec = C.apply_b(agent, 10_000, mask_only)
        try:
            l, a = _logits(agent, b)
        finally:
            rec["restore"]()
        assert float((l - nl).abs().max()) < 1e-5 and float((a - na).abs().max()) < 1e-5


def test_b_compact_equals_mask_only_with_padding_and_differs_from_native(agent):
    b = _batch(agent, [STATE, STATE[:90], STATE[:40]])
    agent.model.eval()
    nl, na = _logits(agent, b)
    res = {}
    for mask_only in (False, True):
        rec = C.apply_b(agent, 8, mask_only)
        try:
            res[mask_only] = _logits(agent, b)
            kept = rec["info"]["last_keep_fraction"]
        finally:
            rec["restore"]()
        assert 0.0 < kept < 1.0                                                      # the window really removes keys
    assert float((res[False][0] - res[True][0]).abs().max()) < 1e-5
    assert float((res[False][1] - res[True][1]).abs().max()) < 1e-5
    assert float((res[True][0] - nl).abs().max()) > 1e-6                              # a real change
    assert not any("forward" in m.__dict__ for m in [agent.model])


def test_b_window_clips_and_end_to_end_predict_runs(agent):
    base = probs(agent)
    rec = V.apply_variant(agent, "b1", reversible=True)
    try:
        p = probs(agent)
        assert rec["variant"] == "b1" and rec["window"] == 128 and rec["candidate_layers"] == ["head.0"]
    finally:
        rec["restore"]()
    assert abs(sum(p.values()) - 1.0) < 1e-3
    assert probs(agent) == base


def test_candidates_refuse_models_without_their_layers(agent):
    for l in agent.model.encoder.layers:
        l.attention_type = "sliding_attention"
    with pytest.raises(C.CandidateError, match="no global"):
        C.apply_a(agent, 8, False)
    agent.model.head = None
    with pytest.raises(C.CandidateError, match="no decision-head"):
        C.apply_b(agent, 8, False)


@pytest.mark.parametrize("name,composed", [("a1", True), ("b1", True), ("a1_mask", False), ("b1_mask", False)])
def test_cpu_candidates_run_on_top_of_optimized_native_and_mask_only_ones_do_not(agent, name, composed):
    enc = agent.model.encoder
    local = [i for i, l in enumerate(enc.layers) if l.attn.sliding_window is not None]
    rec = V.apply_variant(agent, name, reversible=True)
    try:
        assert rec.get("composed_with") == ("local_exact" if composed else None)
        for i, l in enumerate(enc.layers):
            if i in local:
                assert ("forward" in l.attn.__dict__) == composed          # the exact local kernel is in place only when composed
        p = probs(agent)
        assert abs(sum(p.values()) - 1.0) < 1e-3
    finally:
        rec["restore"]()
    assert not any("forward" in l.attn.__dict__ for l in enc.layers) and not enc._forward_pre_hooks
    assert "forward" not in agent.model.__dict__


@pytest.mark.parametrize("name,fast", [("a1", False), ("b1", False), ("a1_mask", True), ("b1_mask", True), ("none", True)])
def test_loader_fast_path_setting_cpu_paths_off_mask_only_on_plain_native(name, fast, monkeypatch, agent):
    from experiments import runner
    seen = {}

    def fake_load(model, revision, threads, compile=False, mha_fastpath=True):
        seen["fast"] = mha_fastpath
        return agent, {"mha_fastpath": mha_fastpath}
    monkeypatch.setattr(runner, "load_agent", fake_load)
    _, info, rec = V.load_for_variant("m", None, 1, name)
    try:
        assert seen["fast"] is fast
    finally:
        pass
