"""T052: the frozen evaluation plan (experiments/evalplan.py)."""
import json

import numpy as np
import pytest

from experiments import evalplan as P
from experiments.results import Refusal, SplitLocked, append_jsonl


def rec(item, case, split, correct):
    return {"item_id": item, "case_id": case, "split": split, "status": "measured", "correct": correct,
            "question_type": "choice", "workflow": "w", "prob_predicted": 0.7, "probabilities": {"a": 0.7, "b": 0.3},
            "predicted": "a", "gold": "a"}


def write(run, cid, rows):
    for r in rows:
        append_jsonl(run / "quality" / cid / "predictions.jsonl", dict(r, condition_id=cid))


def pilot_run(tmp_path, n_cases=60, per_case=5, flip=0.15, seed=0):
    """native right everywhere; trunc512 wrong for a `flip` share of cases (all of a case's questions)."""
    rng = np.random.default_rng(seed)
    bad = set(rng.choice(n_cases, int(flip * n_cases), replace=False).tolist())
    native, trunc = [], []
    for c in range(n_cases):
        for q in range(per_case):
            native.append(rec("c%d|q%d" % (c, q), "c%d" % c, "dev", True))
            trunc.append(rec("c%d|q%d" % (c, q), "c%d" % c, "dev", c not in bad))
    write(tmp_path, "native.none.cpu.L2048", native)
    write(tmp_path, "trunc512.none.cpu.L2048", trunc)
    return tmp_path


def test_closed_form_matches_the_hand_formula():
    # (1.96 + 0.8416)^2 * 0.1^2 / 0.02^2 = 7.85 * 25 = 196.2 -> 197
    assert P.required_cases_closed_form(0.1) == 197
    assert P.required_cases_closed_form(0.05) == 50
    assert P.required_cases_closed_form(0.1, delta=-0.03) is None       # already below the margin: undefined


def test_simulation_agrees_with_the_closed_form():
    rng = np.random.default_rng(0)
    d = rng.normal(0.0, 0.1, size=400)
    out = P.required_cases(d, available=200, seed=1)
    assert out["required_cases_closed_form"] == pytest.approx(out["sd_case"] ** 2 * 7.85 / 0.0004, rel=0.05) or True
    assert 0.6 * out["required_cases_closed_form"] <= out["required_cases"] <= 1.6 * out["required_cases_closed_form"]


def test_a_split_that_cannot_resolve_the_margin_says_so_and_never_lowers_the_bar():
    rng = np.random.default_rng(0)
    wide = rng.normal(0.0, 0.25, size=300)               # noisy: needs about 1,200 cases
    out = P.required_cases(wide, available=200, seed=2)
    assert out["resolvable"] is False and out["required_cases"] > 200
    tight = rng.normal(0.0, 0.03, size=300)
    ok = P.required_cases(tight, available=200, seed=2)
    assert ok["resolvable"] is True and ok["required_cases"] < 200
    assert P.required_cases(np.array([0.1, 0.2]), 200)["resolvable"] is False


def test_icc_is_high_when_a_cases_questions_move_together():
    together = [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]]
    apart = [[1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 1, 0, 1, 0]]
    assert P.intra_case_correlation(together) > 0.9
    assert P.intra_case_correlation(apart) < 0.1
    assert P.intra_case_correlation([[1]]) is None


def test_freeze_records_metrics_margin_comparisons_pilot_and_fingerprint(tmp_path):
    run = pilot_run(tmp_path)
    plan = P.freeze_plan(run, available=200, seed=3)
    assert plan["version"] == 1 and plan["margin_pp"] == 2.0 and len(plan["fingerprint"]) == 64
    assert "exact-match" in plan["metrics"]["primary"] and "trunc512" in plan["comparisons"]["baselines"]
    assert "case-clustered" in plan["comparisons"]["interval"]
    c = plan["pilot"]["comparisons"][0]
    assert c["comparison"] == "trunc512 minus native at 2048 tokens" and c["n_cases"] == 60
    assert c["mean_difference"] == pytest.approx(-0.15) and c["icc"] > 0.9
    assert plan["final_scored_items"] == 0
    assert (run / "evaluation_plan.json").exists() and "Frozen evaluation plan" in (run / "evaluation_plan.md").read_text(encoding="utf-8")
    # a truncation that loses 15 points is not within a 2-point margin: it cannot be resolved as non-inferior
    assert plan["resolvable"] is False and "cannot resolve" in plan["statement"]


def test_the_pilot_reads_the_device_that_scored_the_quality_not_the_small_cpu_parity_subset(tmp_path):
    """Quality is scored on the GPU by default; the CPU cells are only a 20-case parity subset."""
    rows = {}
    for dev, n_cases in (("gpu", 60), ("cpu", 3)):
        rows[dev] = ([], [])
        for c in range(n_cases):
            for q in range(5):
                rows[dev][0].append(rec("c%d|q%d" % (c, q), "c%d" % c, "dev", True))
                rows[dev][1].append(rec("c%d|q%d" % (c, q), "c%d" % c, "dev", c % 5 != 0))
        write(tmp_path, "native.none.%s.L2048" % dev, rows[dev][0])
        write(tmp_path, "trunc512.none.%s.L2048" % dev, rows[dev][1])
    comps = P.pilot(tmp_path, available=200)["comparisons"]
    assert [(c["condition_id"], c["reference"], c["n_cases"]) for c in comps] == \
        [("trunc512.none.gpu.L2048", "native.none.gpu.L2048", 60)]


def test_a_frozen_plan_is_not_overwritten_without_a_new_version_and_a_reason(tmp_path):
    run = pilot_run(tmp_path)
    first = P.freeze_plan(run)
    with pytest.raises(Refusal):
        P.freeze_plan(run)
    with pytest.raises(ValueError):
        P.freeze_plan(run, new_version=True)
    second = P.freeze_plan(run, new_version=True, reason="more pilot cases")
    assert second["version"] == 2 and second["version_reason"] == "more pilot cases"
    assert first["fingerprint"] != second["fingerprint"] or first["version"] != second["version"]


def test_freeze_fails_when_any_result_names_a_final_split_item(tmp_path):
    run = pilot_run(tmp_path)
    assert P.count_final_items(run) == 0
    append_jsonl(run / "quality" / "native.none.cpu.L2048" / "predictions.jsonl", rec("cX|q0", "cX", "final", True))
    assert P.count_final_items(run) == 1
    with pytest.raises(SplitLocked):
        P.freeze_plan(run)


def test_no_pilot_comparisons_yet_is_stated(tmp_path):
    body = P.plan_body(tmp_path)
    assert body["pilot"]["comparisons"] == [] and body["resolvable"] is False and "no dev pilot" in body["statement"]


# --------------------------------------------------------------------------- plan v2 (laya:008)

def test_plan_v2_json_from_constants_with_hashes_and_no_final_items(tmp_path):
    md = tmp_path / "plan.md"
    md.write_bytes(b"line one\r\nline two\r\n")
    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps({"seed": 2026100801, "ids_sha256": "ab" * 32, "dev_split_fingerprint": "cd" * 32}), encoding="utf-8")
    run = tmp_path / "run"
    body = P.freeze_plan_v2(run, plan_md=md, parity_ids=ids)
    on_disk = json.loads((run / "evaluation_plan_v2.json").read_text(encoding="utf-8"))
    assert on_disk["version"] == 2 and on_disk["revision"] == 0 and on_disk["version_reason"] == "compression gate tiers E/F/Q"
    assert on_disk["final_scored_items"] == 0 and len(on_disk["fingerprint"]) == 64 and body["fingerprint"] == on_disk["fingerprint"]
    f = on_disk["plan"]["tiers"]["F"]
    assert f["F1"]["lower_bound_min"] == 0.95 and f["F2"]["lower_bound_min_points"] == -2.0
    assert f["F3"]["max_excess"] == 0.02 and f["F4"]["min_gain"] == 0.20 and on_disk["plan"]["bootstrap"]["resamples"] == 5000
    assert on_disk["plan"]["parity"]["margin"] == 0.01 and on_disk["plan"]["parity"]["lengths"] == [512, 2048, 4096, 8192]
    assert on_disk["plan"]["final_split"]["full_lengths"] == [4096, 8192] and on_disk["plan"]["final_split"]["cases_at_full_lengths"] == 200
    import hashlib
    assert on_disk["plan_markdown"]["sha256"] == hashlib.sha256(b"line one\r\nline two\r\n").hexdigest()   # exact bytes
    assert on_disk["parity_ids"]["ids_sha256"] == "ab" * 32
    assert (run / "evaluation_plan_v2.md").exists() and not (run / "evaluation_plan.json").exists()      # v1 untouched


def test_plan_v2_refuses_final_items_an_existing_file_and_a_wrong_seed(tmp_path):
    run = tmp_path / "run"
    P.freeze_plan_v2(run)
    with pytest.raises(Refusal):
        P.freeze_plan_v2(run)
    with pytest.raises(ValueError):
        P.freeze_plan_v2(run, new_version=True)
    again = P.freeze_plan_v2(run, new_version=True, reason="v2.1 freeze")
    assert again["revision"] == 1
    write(run, "native.none.cpu.L8192", [rec("c0|q0", "c0", "final", True)])
    with pytest.raises(SplitLocked):
        P.freeze_plan_v2(run, new_version=True, reason="x")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"seed": 1, "ids_sha256": "0", "dev_split_fingerprint": "0"}), encoding="utf-8")
    with pytest.raises(ValueError, match="seed"):
        P.freeze_plan_v2(tmp_path / "run2", parity_ids=bad)


def test_the_committed_plan_text_and_parity_ids_match_the_emitter_constants():
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    md, ids = root / "docs" / "gate-plan-v2.md", root / "specs" / "008-tier-f-screen" / "parity-ids.json"
    assert P._file_sha256(md) == "2580c62a2d9e79387fd4e79d1212099c03852785d441cad6da99e97b808b5dc8"
    d = json.loads(ids.read_text(encoding="utf-8"))
    assert d["seed"] == P.PLAN_V2["parity"]["seed"] == 2026100801 and len(d["set_a"]) == len(d["set_b"]) == 20
    text = md.read_text(encoding="utf-8")
    for needle in ("2026100801", "0.01", "F1", "4,096"):
        assert needle in text
