"""laya:007: Tier E drivers (experiments/tier_e.py) on the offline fixture."""
import laya
import pytest

from experiments import tier_e as T
from experiments.baselines import build_runner

Q = {"type": "choice", "instructions": "which option best matches", "criteria": {"a": "x", "b": "y", "c": "z"}}
STATE = "the small red city is quiet in the morning and the harbor is warm " * 3


@pytest.fixture()
def agent(fastpath_checkpoint):
    return laya.Agent(fastpath_checkpoint, device="cpu")


def _item(n=0, length=200):
    return {"item_id": "c%d|q|distractor@mid|%d" % (n, length), "case_id": "c%d" % n, "question_id": "q",
            "split": "dev", "variant": "distractor@mid", "length": length, "state_text": STATE, "question": Q}


def test_e1_items_pick_the_first_question_of_each_variant_sample_case_sorted():
    items = []
    for c in ("c2", "c1", "c3", "c9"):
        for q in ("b", "a"):
            for v in ("distractor@mid", "neutral@mid"):
                items.append({"case_id": c, "question_id": q, "variant": v, "length": 512, "split": "dev", "item_id": c + q + v})
    items.append({"case_id": "c1", "question_id": "a", "variant": "distractor@mid", "length": 1024, "split": "dev", "item_id": "x"})
    items.append({"case_id": "c1", "question_id": "0", "variant": "distractor@mid", "length": 512, "split": "final", "item_id": "f"})
    got = T.e1_items(items, ["c1", "c2", "c3"], 512, n=2)
    assert [(i["case_id"], i["question_id"], i["variant"]) for i in got] == [("c1", "a", "distractor@mid"), ("c2", "a", "distractor@mid")]
    assert all(i["split"] == "dev" for i in got)


def test_measure_item_exact_kernel_is_within_tolerance_and_reports_every_local_layer(agent):
    rec = T.measure_item(agent, build_runner("native"), _item())
    assert rec["max_abs_prob_diff"] <= T.E1_TOLERANCE and not rec["prediction_changed"]
    assert list(rec["local_layer_max_abs_diff"]) == ["1"]                       # the fixture has one local layer
    assert 0 <= rec["local_layer_max_abs_diff"]["1"] <= 1e-4
    assert not any("forward" in l.attn.__dict__ for l in agent.model.encoder.layers)     # restored


def test_a_block3_kernel_fails_e1(agent, monkeypatch):
    """A superset window must be caught: swap the kernel for the block-granular one."""
    import experiments.kernels.local_exact as K
    from experiments.kernels.local import local_attention
    from experiments.kernels.reference import MaskSpec

    def block3(q, k, v, half_width, lengths=None, block=None, key_valid=None):
        L = q.shape[-2]
        return local_attention(q, k, v, MaskSpec([L] * q.shape[0], "block_local", block=2 * half_width))
    monkeypatch.setattr(K, "local_exact_attention", block3)
    rec = T.measure_item(agent, build_runner("native"), _item())
    assert rec["local_layer_max_abs_diff"]["1"] > 1e-3
    assert rec["max_abs_prob_diff"] > 0


def _rows(diff, n=T.E1_ITEMS_PER_LENGTH, changed=False):
    return {"rows": [{"item_id": "i%d" % k, "max_abs_prob_diff": diff, "prediction_changed": changed,
                      "local_layer_max_abs_diff": {"1": diff, "3": diff / 2}} for k in range(n)]}


def test_e1_verdict_needs_all_lengths_and_the_unloosened_tolerance():
    ok = {L: _rows(2e-6) for L in T.E1_LENGTHS}
    s = T.summarize_e1(ok)
    assert s["passed"] and s["observed_max"] == 2e-6 and s["lengths"][0]["per_local_layer_max_abs_diff"]["3"] == 1e-6
    bad = dict(ok)
    bad[8192] = _rows(1.2e-5)                         # just above 1e-5: reported with the observed maximum, not passed
    s = T.summarize_e1(bad)
    assert not s["passed"] and s["observed_max"] == 1.2e-5 and not s["lengths"][-1]["passed"]
    assert not T.summarize_e1({L: ok[L] for L in T.E1_LENGTHS[:3]})["passed"]          # missing lengths
    assert not T.summarize_e1({L: _rows(1e-7, n=19) for L in T.E1_LENGTHS})["passed"]  # fewer than 20 items


def _preds(changed_at=None, drop=None):
    out = {}
    for L, n in ((512, 95), (2048, 100), (8192, 100)):
        for k in range(n):
            iid = "c%d|q|d|%d" % (k, L)
            out[iid] = {"item_id": iid, "length": L, "predicted": "a", "probabilities": {"a": 0.5, "b": 0.45, "c": 0.05}}
    if changed_at:
        out[changed_at]["predicted"] = "b"
    if drop:
        del out[drop]
    return out


def test_e2_counts_flips_and_missing_items_as_failures():
    sizes = {512: 95, 2048: 100, 8192: 100}
    ok = T.compare_predictions(_preds(), _preds(), sizes)
    assert ok["passed"] and ok["n"] == 295 and ok["changed"] == 0
    flip = T.compare_predictions(_preds(changed_at="c3|q|d|2048"), _preds(), sizes)
    assert not flip["passed"] and flip["changed"] == 1
    assert flip["lengths"][1]["flips"][0]["reference_top2_margin"] == pytest.approx(0.05)
    miss = T.compare_predictions(_preds(drop="c0|q|d|8192"), _preds(), sizes)
    assert not miss["passed"] and miss["missing"] == 1
    assert not T.compare_predictions(_preds(), _preds(), {512: 96})["passed"]       # unexpected size


def _lat(opt_factor):
    return {L: {"none": {"p50": 100.0 * (L / 512), "p95": 110.0}, "fastpath_off": {"p50": 90.0 * (L / 512), "p95": 95.0},
                "local_exact_fastpath_off": {"p50": 90.0 * (L / 512) * opt_factor, "p95": 99.0}} for L in T.E1_LENGTHS}


def test_e3_passes_only_when_optimized_native_is_not_slower_at_every_length():
    assert T.summarize_e3(_lat(0.7))["passed"]
    assert T.summarize_e3(_lat(1.0))["passed"]                       # not slower (equal) passes
    slow = _lat(0.7)
    slow[512]["local_exact_fastpath_off"]["p50"] = 91.0
    s = T.summarize_e3(slow)
    assert not s["passed"] and not s["lengths"][0]["passed"] and s["lengths"][0]["optimized_vs_fastpath_off_p50"] > 1
    partial = {L: v for L, v in _lat(0.7).items() if L != 4096}
    assert not T.summarize_e3(partial)["passed"]


def test_report_verdicts_and_not_run(tmp_path):
    rep = T.build_report(tmp_path, None, {"passed": True}, {"passed": False}, {"git_dirty": False})
    assert rep["verdicts"] == {"E1": "not_run", "E2": "passed", "E3": "failed"}
    assert not rep["tier_e_passed"] and rep["final_scored_items"] == 0 and rep["device"] == "cpu" and rep["dtype"] == "fp32"
    assert T.build_report(tmp_path, {"passed": True}, {"passed": True}, {"passed": True}, {})["tier_e_passed"]


def test_gate_runs_refuse_a_dirty_tree_or_a_changed_laya(monkeypatch):
    import experiments.manifest as M
    monkeypatch.setattr(M, "code_info", lambda *a, **k: {"git_dirty": True, "laya_diff_empty": True, "dirty_paths": ["x"]})
    with pytest.raises(T.TierEError, match="clean git tree"):
        T.require_clean_tree()
    assert T.require_clean_tree(allow_dirty=True)["git_dirty"] is True
    monkeypatch.setattr(M, "code_info", lambda *a, **k: {"git_dirty": False, "laya_diff_empty": False})
    with pytest.raises(T.TierEError, match="laya/"):
        T.require_clean_tree()
    monkeypatch.setattr(M, "code_info", lambda *a, **k: {"git_dirty": False, "laya_diff_empty": True})
    assert T.require_clean_tree()["git_dirty"] is False


def test_run_probs_writes_one_file_per_length_and_resumes(fastpath_checkpoint, tmp_path):
    items = {200: [_item(0), _item(1)]}
    paths = T.run_probs(tmp_path, fastpath_checkpoint, None, 1, items, {"git_dirty": True}, log=lambda *_: None)
    assert paths[0].name == "probs.L200.json"
    import json
    body = json.loads(paths[0].read_text(encoding="utf-8"))
    assert body["device"] == "cpu" and body["dtype"] == "fp32" and body["final_scored_items"] == 0 and len(body["rows"]) == 2
    again = T.run_probs(tmp_path, "does-not-exist", None, 1, items, {}, log=lambda *_: None)
    assert again == paths                                             # finished lengths are not rerun (no model load)
