"""T038: calibration and evaluation summary (experiments/summary.py)."""
import numpy as np
import pytest

from experiments import metrics
from experiments import summary as S
from experiments.results import append_jsonl, read_json


def rec(item, case, split, correct, qtype="choice", p=0.9, cid="x", wf="customer_service", vis="full", **kw):
    probs = {"a": p, "b": 1 - p} if qtype != "noul" else {"false": 1 - p, "true": p}
    pred = "a" if qtype != "noul" else "true"
    gold = pred if correct else ("b" if qtype != "noul" else "false")
    r = {"item_id": item, "case_id": case, "split": split, "workflow": wf, "question_type": qtype,
         "status": "measured", "correct": correct, "predicted": pred, "gold": gold, "probabilities": probs,
         "prob_predicted": p, "level_error": None, "evidence_visible": vis, "low_confidence": False,
         "argmax_agree": True, "tv_top_quartile": False, "condition_id": cid}
    r.update(kw)
    return r


def write(run, cid, rows):
    for r in rows:
        r = dict(r, condition_id=cid)
        append_jsonl(run / "quality" / cid / "predictions.jsonl", r)


def test_condition_id_round_trips():
    assert S.parse_condition_id("native.none.cpu.L512") == {"name": "native", "params": "", "variant": "none",
                                                            "device": "cpu", "length": 512}
    p = S.parse_condition_id("window.size512.none.cpu.L2048")
    assert p["name"] == "window" and p["params"] == "size512" and p["length"] == 2048
    assert S.parse_condition_id("native.int8_encoder.cpu.Loriginal")["length"] == "original"
    with pytest.raises(ValueError):
        S.parse_condition_id("nonsense")


def test_temperatures_are_fitted_on_calibration_records_only():
    cal = [rec("i%d" % i, "c%d" % (i // 5), "calibration", i % 5 != 0, p=0.99) for i in range(100)]
    fit = S.fit_condition(cal)
    assert fit["choice"]["n"] == 100 and fit["choice"]["temperature"] > 1.5          # overconfident: T above one
    with pytest.raises(ValueError) as err:
        S.fit_condition(cal + [rec("d", "cd", "dev", True)])
    assert "calibration split only" in str(err.value)


def test_calibrate_writes_only_from_calibration_split_and_summary_applies_it(tmp_path):
    cid = "native.none.cpu.L512"
    cal = [rec("cal%d" % i, "cc%d" % (i // 5), "calibration", i % 5 != 0, p=0.99) for i in range(100)]
    dev = [rec("dev%d" % i, "dc%d" % (i // 5), "dev", i % 5 != 0, p=0.99) for i in range(100)]
    write(tmp_path, cid, cal + dev)
    body = S.calibrate(tmp_path)
    assert list(body["conditions"]) == [cid] and body["fitted_on"] == "calibration"
    assert body["conditions"][cid]["choice"]["n"] == 100                              # the 100 calibration items, not 200
    summ = S.build_summary(tmp_path)
    c = summ["conditions"][cid]
    assert c["ece_raw"]["overall"] == pytest.approx(0.19, abs=1e-6)
    assert c["ece_scaled"]["overall"] < c["ece_raw"]["overall"]                       # raw and scaled are both reported
    assert c["accuracy"]["n"] == 100                                                  # dev only
    assert c["temperatures"]["choice"]["temperature"] > 1.5


def test_window_scaled_ece_is_labeled_non_comparable(tmp_path):
    cid = "window.size256.none.cpu.L1024"
    write(tmp_path, cid, [rec("w%d" % i, "c%d" % (i // 5), "calibration", i % 2 == 0) for i in range(20)]
          + [rec("v%d" % i, "d%d" % (i // 5), "dev", i % 2 == 0) for i in range(20)])
    S.calibrate(tmp_path)
    c = S.build_summary(tmp_path)["conditions"][cid]
    assert "non-comparable" in c["ece_scaled_note"]


def test_summary_has_strata_counts_by_status_and_visibility(tmp_path):
    cid = "trunc512.none.cpu.L2048"
    rows = [rec("a%d" % i, "c%d" % (i // 5), "dev", i % 2 == 0, vis="full" if i % 3 else "none",
                low_confidence=(i % 4 == 0)) for i in range(40)]
    rows.append(dict(rec("z", "cz", "dev", True), status="failed", reason="oom"))
    rows.append(dict(rec("y", "cy", "dev", True), status="unsupported", reason="target_exceeds_length"))
    write(tmp_path, cid, rows)
    c = S.build_summary(tmp_path)["conditions"][cid]
    assert c["counts_by_status"] == {"failed": 1, "measured": 40, "unsupported": 1}
    assert c["n_measured"] == 40 and c["accuracy"]["n"] == 40                        # denominator over measured only
    vis = c["accuracy"]["by_evidence_visible"]
    assert set(vis) == {"full", "none"} and vis["full"]["n"] + vis["none"]["n"] == 40
    assert c["accuracy"]["by_stratum"]["low_confidence"]["n"] == 10
    cell = next(x for x in S.build_summary(tmp_path)["cells"] if x["condition_id"] == cid)
    assert cell["status"] == "partial" and cell["n_measured"] == 40


def test_paired_comparison_against_references_uses_the_same_items_and_case_clusters(tmp_path):
    n = 120
    native = [rec("i%d" % i, "c%d" % (i // 5), "dev", True) for i in range(n)]
    trunc = [rec("i%d" % i, "c%d" % (i // 5), "dev", i % 2 == 0) for i in range(n)]       # 50% right vs 100%
    write(tmp_path, "native.none.cpu.L2048", native)
    write(tmp_path, "trunc512.none.cpu.L2048", trunc)
    write(tmp_path, "oracle.none.cpu.L2048", native)
    summ = S.build_summary(tmp_path, n_boot=500)
    p = summ["conditions"]["native.none.cpu.L2048"]["paired"]["trunc512.none.cpu.L2048"]
    assert p["diff"] == pytest.approx(0.5) and p["verdict"] == "better" and p["n_cases"] == 24
    q = summ["conditions"]["native.none.cpu.L2048"]["paired"]["oracle.none.cpu.L2048"]
    assert q["diff"] == 0.0 and q["verdict"] == "equal"
    back = summ["conditions"]["trunc512.none.cpu.L2048"]["paired"]["native.none.cpu.L2048"]
    assert back["verdict"] == "worse"


def test_underpowered_comparisons_are_inconclusive(tmp_path):
    rng = np.random.default_rng(1)
    a = rng.random(10) < 0.6
    b = rng.random(10) < 0.5
    write(tmp_path, "native.none.cpu.L512", [rec("i%d" % i, "c%d" % i, "dev", bool(a[i])) for i in range(10)])
    write(tmp_path, "trunc512.none.cpu.L512", [rec("i%d" % i, "c%d" % i, "dev", bool(b[i])) for i in range(10)])
    p = S.build_summary(tmp_path, n_boot=500)["conditions"]["native.none.cpu.L512"]["paired"]["trunc512.none.cpu.L512"]
    assert p["verdict"] == "inconclusive" and p["n_cases"] == 10


def test_missing_expected_cells_are_listed_and_optimized_variants_stay_out_of_the_pairing(tmp_path):
    write(tmp_path, "native.none.cpu.L512", [rec("i%d" % i, "c%d" % i, "dev", True) for i in range(6)])
    write(tmp_path, "native.int8_encoder.cpu.L512", [rec("i%d" % i, "c%d" % i, "dev", True) for i in range(6)])
    summ = S.build_summary(tmp_path, expected=["native.none.cpu.L512", "window.sizedefault.none.cpu.L512"], n_boot=100)
    status = {c["condition_id"]: c["status"] for c in summ["cells"]}
    assert status["window.sizedefault.none.cpu.L512"] == "missing" and status["native.none.cpu.L512"] == "measured"
    assert summ["conditions"]["native.int8_encoder.cpu.L512"]["paired"] == {}
    assert read_json(tmp_path, "summary.json")["split"] == "dev"


def test_retrieval_budget_selection_uses_dev_accuracy(tmp_path):
    for budget, acc_n in (("retrieve512", 2), ("retrieve1024", 5), ("retrieve2048", 5)):
        rows = [rec("i%d" % i, "c%d" % i, "dev", i < acc_n) for i in range(6)]
        write(tmp_path, "%s.none.cpu.L2048" % budget, rows)
    sel = S.build_summary(tmp_path, n_boot=50)["retrieval_budget"]
    assert sel["selected"] == 1024 and sel["tuned_on"] == "dev"          # tie with 2048: the smaller budget
    assert sel["mean_accuracy"]["512"] == pytest.approx(2 / 6)


def test_retrieval_budget_selection_never_mixes_devices(tmp_path):
    """The CPU parity subset (retrieve1024 only, on the 20-case sample) must not lift one budget's mean."""
    for budget, n_right in (("retrieve512", 4), ("retrieve1024", 3)):
        write(tmp_path, "%s.none.gpu.L2048" % budget, [rec("i%d" % i, "c%d" % i, "dev", i < n_right) for i in range(6)])
    write(tmp_path, "retrieve1024.none.cpu.L2048", [rec("i%d" % i, "c%d" % i, "dev", True) for i in range(6)])
    write(tmp_path, "native.none.gpu.L2048", [rec("i%d" % i, "c%d" % i, "dev", True) for i in range(6)])
    sel = S.build_summary(tmp_path, n_boot=50)["retrieval_budget"]
    assert sel["selected"] == 512                    # GPU-scored: 4/6 against 3/6
    assert sel["mean_accuracy"]["1024"] == pytest.approx(3 / 6)


# --------------------------------------------------------------------------- CPU/GPU parity (FR-024, 2026-09-30)

def _pair(tmp_path, n=100, flips=0, acc_shift=0):
    """The same items scored on CPU and GPU; `flips` of them change their predicted answer on GPU."""
    cpu = [rec("i%d" % i, "c%d" % (i // 5), "dev", i % 2 == 0, p=0.8) for i in range(n)]
    gpu = []
    for i, r in enumerate(cpu):
        g = dict(r)
        if i < flips:
            g["predicted"] = "b" if r["predicted"] == "a" else "a"
            g["correct"] = not r["correct"]
            g["probabilities"] = {"a": 0.4, "b": 0.6} if g["predicted"] == "b" else {"a": 0.6, "b": 0.4}
        else:
            g["probabilities"] = {"a": 0.8005, "b": 0.1995}
        gpu.append(g)
    write(tmp_path, "native.none.cpu.L512", cpu)
    write(tmp_path, "native.none.gpu.L512", gpu)


def test_parity_passes_when_predictions_agree_and_accuracy_matches(tmp_path):
    _pair(tmp_path, flips=0)
    par = S.parity(tmp_path)
    r = par["rows"][0]
    assert par["passed"] is True and r["n_items"] == 100 and r["same_prediction_share"] == 1.0
    assert r["accuracy_difference"] == 0.0 and r["max_abs_probability_difference"] == pytest.approx(0.0005, abs=1e-6)
    assert r["condition_id"] == "native.none.cpu.L512" and r["gpu_condition_id"] == "native.none.gpu.L512"
    keys = ("min_same_prediction_share", "max_accuracy_difference", "near_tie_margin")
    assert {k: par["criteria"][k] for k in keys} == {"min_same_prediction_share": 0.98, "max_accuracy_difference": 0.005,
                                                      "near_tie_margin": 0.01}
    assert r["strict_passed"] is True and r["tolerated_flips"] == [] and r["failing_flips"] == []


def test_parity_fails_when_too_many_answers_flip(tmp_path):
    _pair(tmp_path, flips=5)                       # 95% identical: below the 98% criterion
    par = S.parity(tmp_path)
    assert par["passed"] is False and par["rows"][0]["same_prediction_share"] == 0.95
    assert par["rows"][0]["passed"] is False


def test_parity_fails_on_an_accuracy_gap_even_with_98_percent_identical_predictions(tmp_path):
    cpu = [rec("i%d" % i, "c%d" % (i // 5), "dev", True) for i in range(100)]
    gpu = [dict(r) for r in cpu]
    for g in gpu[:2]:                              # two right answers become wrong on the GPU
        g["predicted"], g["correct"] = "b", False
    write(tmp_path, "native.none.cpu.L512", cpu)
    write(tmp_path, "native.none.gpu.L512", gpu)
    r = S.parity(tmp_path)["rows"][0]
    assert r["same_prediction_share"] == 0.98 and r["accuracy_difference"] == pytest.approx(-0.02)
    assert r["passed"] is False                    # identical share is enough, but accuracy moved 2 points (limit 0.5)


def test_parity_is_undetermined_without_a_cpu_twin_or_items(tmp_path):
    write(tmp_path, "native.none.gpu.L512", [rec("i%d" % i, "c%d" % i, "dev", True) for i in range(10)])
    par = S.parity(tmp_path)
    assert par["rows"] == [] and par["passed"] is None and par["strict_passed"] is None
    assert par["criteria"]["near_tie_margin"] == S.PARITY_NEAR_TIE_MARGIN == 0.01


def test_summary_carries_the_parity_block_and_both_devices_as_separate_cells(tmp_path):
    _pair(tmp_path)
    summ = S.build_summary(tmp_path, n_boot=100)
    devices = {c["condition_id"]: c["device"] for c in summ["cells"]}
    assert devices == {"native.none.cpu.L512": "cpu", "native.none.gpu.L512": "gpu"}
    assert summ["parity"]["passed"] is True


# --------------------------------------------------------------------------- margin-aware parity (parity-margin.md)

def _near_tie_pair(run, n=100, tie_flips=2, hard_flips=0, margin_probs=(0.504, 0.496), cid="native.none.%s.L8192"):
    """CPU items of which `tie_flips` are near-ties that the GPU answers differently, `hard_flips` are confident ones."""
    cpu = [rec("i%d" % i, "c%d" % (i // 5), "dev", True, p=0.9) for i in range(n)]
    for r in cpu[:tie_flips]:
        r["probabilities"] = {"a": margin_probs[0], "b": margin_probs[1]}
    gpu = [dict(r) for r in cpu]
    for r in gpu[:tie_flips + hard_flips]:
        r["predicted"], r["correct"] = "b", False
    write(run, cid % "cpu", cpu)
    write(run, cid % "gpu", gpu)
    return cpu, gpu


def test_near_tie_flips_are_tolerated_and_listed_with_margins_while_strict_is_reported(tmp_path):
    _near_tie_pair(tmp_path, tie_flips=2)                      # 2 flips: 98% identical, accuracy -2 points
    par = S.parity(tmp_path)
    r = par["rows"][0]
    assert par["passed"] is True and r["passed"] is True and r["failing_flips"] == []
    assert par["strict_passed"] is False and r["strict_passed"] is False          # the old rule still fails
    assert r["n_disagreements"] == 2 and r["accuracy_difference"] == pytest.approx(-0.02)
    assert [f["item_id"] for f in r["tolerated_flips"]] == ["i0", "i1"]
    assert r["tolerated_flips"][0]["cpu_margin"] == pytest.approx(0.008)
    assert r["tolerated_flips"][0]["cpu_predicted"] == "a" and r["tolerated_flips"][0]["gpu_predicted"] == "b"


def test_a_flip_above_the_margin_still_fails_parity(tmp_path):
    _near_tie_pair(tmp_path, tie_flips=2, hard_flips=1)        # the third flip is on a 0.8-margin item
    par = S.parity(tmp_path)
    r = par["rows"][0]
    assert par["passed"] is False and r["passed"] is False
    assert [f["item_id"] for f in r["failing_flips"]] == ["i2"] and r["failing_flips"][0]["cpu_margin"] == pytest.approx(0.8)
    assert len(r["tolerated_flips"]) == 2


def test_the_margin_is_a_parameter_and_the_bound_is_inclusive(tmp_path):
    _near_tie_pair(tmp_path, tie_flips=1, margin_probs=(0.75, 0.25))
    assert S.parity(tmp_path)["passed"] is False                                    # margin 0.5 > default 0.01
    assert S.parity(tmp_path, margin=0.5)["passed"] is True                         # exactly at the bound: tolerated
    assert S.parity(tmp_path, margin=0.49)["passed"] is False
    assert S.parity(tmp_path, margin=0.5)["criteria"]["near_tie_margin"] == 0.5


def test_margin_uses_the_cpu_probabilities_and_handles_one_option():
    assert S.top_two_margin({"a": 0.5, "b": 0.3, "c": 0.2}) == pytest.approx(0.2)
    assert S.top_two_margin({"only": 1.0}) == 1.0


def test_gpu_predictions_can_come_from_another_run_with_labels(tmp_path):
    cpu_run, other = tmp_path / "cpu_run", tmp_path / "fp16_run"
    cpu, gpu = _near_tie_pair(cpu_run, tie_flips=1)
    write(other, "native.none.gpu.L8192", [dict(r, predicted="a", correct=True) for r in gpu])      # the rerun agrees with CPU
    par = S.parity(cpu_run, gpu_run_path=other, cpu_label="CPU i7-8700 fp32", gpu_label="RTX 4060 fp16 autocast")
    r = par["rows"][0]
    assert r["same_prediction_share"] == 1.0 and r["n_disagreements"] == 0 and par["passed"] is True and par["strict_passed"] is True
    assert par["labels"] == {"cpu": "CPU i7-8700 fp32", "gpu": "RTX 4060 fp16 autocast"}
    assert S.parity(cpu_run)["rows"][0]["n_disagreements"] == 1                      # the run's own GPU twin still disagrees


def test_summary_and_report_carry_the_margin_aware_verdict(tmp_path):
    _near_tie_pair(tmp_path, tie_flips=2)
    summ = S.build_summary(tmp_path, n_boot=50)
    assert summ["parity"]["passed"] is True and summ["parity"]["strict_passed"] is False
    assert S.build_summary(tmp_path, n_boot=50, parity_margin=0.001)["parity"]["passed"] is False
    from experiments import report2 as R
    text = " ".join(x["text"] for x in R.section_parity(S.build_summary(tmp_path, n_boot=50), "gpu")["statements"])
    assert "near-tie flip on i0" in text and "stands in for CPU quality" in text and "missed" in text


def test_eval_summary_takes_a_parity_margin_option():
    from experiments import cli
    cli._load_commands()
    ns = cli.build_parser().parse_args(["eval-summary", "--parity-margin", "0.02"])
    assert ns.parity_margin == 0.02
    assert cli.build_parser().parse_args(["eval-summary"]).parity_margin is None


def test_the_accuracy_difference_is_fully_accounted_for_by_the_listed_flips(tmp_path):
    """The accuracy rule is replaced by the margin rule: accuracy moves only through disagreeing items."""
    _near_tie_pair(tmp_path, tie_flips=3, hard_flips=1)
    r = S.parity(tmp_path)["rows"][0]
    assert r["accuracy_difference"] == pytest.approx(-0.04) == pytest.approx(r["accuracy_difference_from_flips"])
    run2 = tmp_path / "agree"
    _near_tie_pair(run2, tie_flips=0)
    r2 = S.parity(run2)["rows"][0]
    assert r2["n_disagreements"] == 0 and r2["accuracy_difference"] == 0.0 and r2["accuracy_difference_from_flips"] == 0.0


def test_tolerated_flips_can_move_accuracy_past_the_old_limit_by_design(tmp_path):
    _near_tie_pair(tmp_path, tie_flips=2)                       # 2 near-tie flips: -2 points
    par = S.parity(tmp_path)
    r = par["rows"][0]
    assert r["accuracy_difference"] == pytest.approx(-0.02) and r["passed"] is True and r["strict_passed"] is False


def test_a_negative_or_non_finite_margin_is_refused(tmp_path):
    _near_tie_pair(tmp_path, tie_flips=1)
    for bad in (-0.01, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite number >= 0"):
            S.parity(tmp_path, margin=bad)
    assert S.parity(tmp_path, margin=0.0)["rows"][0]["failing_flips"][0]["cpu_margin"] == pytest.approx(0.008)   # 0 is allowed: strict ties only


def test_the_cli_refuses_a_negative_or_non_finite_parity_margin(capsys):
    from experiments import cli
    cli._load_commands()
    for bad in ("-0.01", "nan", "inf", "abc"):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["eval-summary", "--parity-margin", bad])
    assert cli.build_parser().parse_args(["eval-summary", "--parity-margin", "0"]).parity_margin == 0.0


def test_margin_zero_tolerates_only_an_exact_tie_through_parity(tmp_path):
    _near_tie_pair(tmp_path, tie_flips=1, margin_probs=(0.5, 0.5))                  # CPU margin exactly 0
    assert S.parity(tmp_path, margin=0.0)["passed"] is True
    assert S.parity(tmp_path, margin=0.0)["rows"][0]["tolerated_flips"][0]["cpu_margin"] == 0.0
    _near_tie_pair(tmp_path / "other", tie_flips=1, margin_probs=(0.504, 0.496))     # a 0.008 margin is not a tie
    assert S.parity(tmp_path / "other", margin=0.0)["passed"] is False


def test_single_entry_and_empty_probabilities_go_through_parity(tmp_path):
    cpu = [rec("i%d" % i, "c%d" % i, "dev", True) for i in range(4)]
    for r in cpu:
        r["probabilities"] = {"only": 1.0}
    gpu = [dict(r) for r in cpu]
    gpu[0]["predicted"], gpu[0]["correct"] = "b", False
    write(tmp_path, "native.none.cpu.L512", cpu)
    write(tmp_path, "native.none.gpu.L512", gpu)
    r = S.parity(tmp_path)["rows"][0]
    assert r["failing_flips"][0]["cpu_margin"] == 1.0 and r["passed"] is False      # one option: never a near-tie
    empty = tmp_path / "empty"
    cpu2 = [dict(r, probabilities={}) for r in cpu]
    write(empty, "native.none.cpu.L512", cpu2)
    write(empty, "native.none.gpu.L512", [dict(r, probabilities={}) for r in gpu])
    r2 = S.parity(empty)["rows"][0]
    assert r2["max_abs_probability_difference"] == 0.0 and r2["failing_flips"][0]["cpu_margin"] == 1.0
    assert S.top_two_margin({}) == 1.0


def test_a_tiny_cell_is_listed_but_not_counted_in_the_overall_verdict(tmp_path):
    """Two items scored through the CPU fallback must not decide parity (the fp32 8,192-token cell of p2-parity-fp32)."""
    _near_tie_pair(tmp_path, n=100, tie_flips=0, cid="native.none.%s.L2048")
    cpu = [rec("t%d" % i, "tc%d" % i, "dev", True) for i in range(2)]
    gpu = [dict(r, predicted="b", correct=False) for r in cpu]              # 100% wrong flips, but only 2 items
    write(tmp_path, "native.none.cpu.L8192", cpu)
    write(tmp_path, "native.none.gpu.L8192", gpu)
    par = S.parity(tmp_path)
    by = {r["condition_id"]: r for r in par["rows"]}
    assert by["native.none.cpu.L8192"]["counted"] is False and by["native.none.cpu.L8192"]["passed"] is False
    assert by["native.none.cpu.L2048"]["counted"] is True
    assert par["passed"] is True and par["strict_passed"] is True                    # decided by the counted row only
    assert par["criteria"]["min_items"] == S.PARITY_MIN_ITEMS == 20
    only_tiny = tmp_path / "tiny_only"
    write(only_tiny, "native.none.cpu.L8192", cpu)
    write(only_tiny, "native.none.gpu.L8192", gpu)
    assert S.parity(only_tiny)["passed"] is None                                     # nothing countable: undetermined
    assert S.parity(only_tiny, min_items=1)["passed"] is False


def test_the_report_says_when_a_row_is_not_counted(tmp_path):
    _near_tie_pair(tmp_path, n=100, tie_flips=0, cid="native.none.%s.L2048")
    cpu = [rec("t%d" % i, "tc%d" % i, "dev", True) for i in range(2)]
    write(tmp_path, "native.none.cpu.L8192", cpu)
    write(tmp_path, "native.none.gpu.L8192", [dict(r) for r in cpu])
    from experiments import report2 as R
    text = " ".join(x["text"] for x in R.section_parity(S.build_summary(tmp_path, n_boot=50), "gpu")["statements"])
    assert "NOT COUNTED: fewer than 20 items" in text
