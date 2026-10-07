"""T012 (and later T024, T037): quality runs, scoring and solvability (experiments/evalrun.py)."""
import pytest

from experiments import data as D
from experiments import evalrun as E
from experiments.results import Refusal, SplitLocked, read_jsonl


@pytest.fixture(scope="module")
def dev_items(tiny_upstream, tmp_path_factory):
    root = tmp_path_factory.mktemp("evalrun-data")
    D.import_dataset(root=root, download=tiny_upstream.download, with_checkpoints=False)
    D.make_splits(seed=3, root=root)
    return root, D.original_items("dev", root=root)


def cond(name="native"):
    return {"condition_id": "%s.none.cpu.Loriginal" % name, "name": name}


# --------------------------------------------------------------------------- scoring

def _item(qtype, label, criteria):
    return {"question": {"type": qtype, "criteria": criteria}, "gold": {"label": label}}


def test_choice_scoring_is_exact_match_on_the_label():
    it = _item("choice", "b", {"a": "x", "b": "y", "c": "z"})
    out = E.score_item(it, {"choice": "b", "probabilities": {"a": 0.2, "b": 0.5, "c": 0.3}})
    assert out["correct"] and out["prob_predicted"] == 0.5 and out["level_error"] is None
    out = E.score_item(it, {"choice": "a", "probabilities": {"a": 0.6, "b": 0.3, "c": 0.1}})
    assert not out["correct"] and out["prob_predicted"] == 0.6


def test_score_question_is_exact_level_match_with_separate_level_error():
    it = _item("score", "1", ["l0", "l1", "l2", "l3"])
    out = E.score_item(it, {"score": 2.2, "probabilities": {"0": 0.1, "1": 0.2, "2": 0.6, "3": 0.1}})
    assert out["predicted"] == "2" and not out["correct"]      # exact match only, not "within one"
    assert out["level_error"] == 1 and out["prob_predicted"] == 0.6
    right = E.score_item(it, {"score": 1.0, "probabilities": {"0": 0.1, "1": 0.7, "2": 0.1, "3": 0.1}})
    assert right["correct"] and right["level_error"] == 0


def test_noul_predicted_answer_probability_is_the_larger_side():
    it = _item("noul", "false", {"false": "n", "true": "y"})
    out = E.score_item(it, {"noul": 0.3})
    assert out["predicted"] == "false" and out["correct"] and out["prob_predicted"] == pytest.approx(0.7)
    out = E.score_item(it, {"noul": 0.9})
    assert out["predicted"] == "true" and not out["correct"] and out["prob_predicted"] == pytest.approx(0.9)


def test_choice_probability_map_is_keyed_by_label_so_option_order_cannot_matter():
    # The same gold label, options listed in another order: scoring reads the label key, not a position.
    a = _item("choice", "c", {"a": "x", "b": "y", "c": "z"})
    b = _item("choice", "c", {"c": "z", "a": "x", "b": "y"})
    ans = {"choice": "c", "probabilities": {"c": 0.6, "a": 0.3, "b": 0.1}}
    assert E.score_item(a, ans)["correct"] and E.score_item(b, ans)["correct"]
    assert E.option_labels(b) == ["c", "a", "b"]


# --------------------------------------------------------------------------- evidence visibility

def test_evidence_fraction_and_visibility():
    span = {"state_start": 100, "state_end": 200}
    assert E.evidence_fraction(span, 500, 900) == 1.0
    assert E.evidence_fraction(span, 150, 900) == 0.5
    assert E.evidence_fraction(span, 100, 900) == 0.0
    assert E.evidence_fraction(None, 300, 300) == 1.0 and E.evidence_fraction(None, 150, 300) == 0.5
    assert [E.visibility(x) for x in (1.0, 0.5, 0.0)] == ["full", "partial", "none"]


# --------------------------------------------------------------------------- running items

def test_run_items_makes_one_call_per_question_row(tiny_agent, dev_items, tmp_path):
    _, items = dev_items
    sample = items[:5]
    calls = []
    real = tiny_agent.predict

    def counting(state, questions, **kw):
        calls.append(list(questions))
        return real(state, questions, **kw)

    tiny_agent.predict = counting
    try:
        recs = E.run_items(tiny_agent, sample, cond(), E.NativeRunner(), out_path=tmp_path / "p.jsonl")
    finally:
        del tiny_agent.predict
    assert len(calls) == 5 and all(len(c) == 1 for c in calls)        # FR-023: one question per pass
    for r in recs:
        assert r["status"] == "measured" and r["device"] == "cpu"
        assert set(r) >= {"predicted", "probabilities", "prob_predicted", "correct", "tokens_seen",
                          "evidence_visible", "evidence_fraction", "gold", "level_error"}
        assert r["evidence_visible"] == "full" and r["tokens_seen"] > 0
        assert (r["level_error"] is not None) == (r["question_type"] == "score")
    assert len(read_jsonl(tmp_path / "p.jsonl")) == 5


def test_run_items_resumes_without_redoing_finished_items(tiny_agent, dev_items, tmp_path):
    _, items = dev_items
    path = tmp_path / "p.jsonl"
    E.run_items(tiny_agent, items[:3], cond(), E.NativeRunner(), out_path=path)
    first = path.read_text(encoding="utf-8")
    again = E.run_items(tiny_agent, items[:6], cond(), E.NativeRunner(), out_path=path)
    assert [r["item_id"] for r in again] == [i["item_id"] for i in items[3:6]]    # only the new ones ran
    assert path.read_text(encoding="utf-8").startswith(first)                     # earlier lines untouched
    assert len(read_jsonl(path)) == 6


def test_a_failure_is_recorded_and_never_replaced(tiny_agent, dev_items):
    _, items = dev_items

    def boom(agent, item):
        raise RuntimeError("DefaultCPUAllocator: not enough memory")

    recs = E.run_items(tiny_agent, items[:2], cond(), boom)
    assert [r["status"] for r in recs] == ["failed", "failed"]
    assert recs[0]["cause"] == "oom" and "not enough memory" in recs[0]["reason"]
    bad = dict(items[0], status="unsupported", reason="target_exceeds_length")
    rec = E.run_items(tiny_agent, [bad], cond(), E.NativeRunner())[0]
    assert rec["status"] == "unsupported" and rec["reason"] == "target_exceeds_length"


def test_the_final_split_and_non_cpu_devices_are_refused(tiny_agent, dev_items):
    _, items = dev_items
    with pytest.raises(SplitLocked):
        E.run_items(tiny_agent, [dict(items[0], split="final")], cond(), E.NativeRunner())

    class OnGpu:
        device = type("D", (), {"type": "cuda"})()

    with pytest.raises(Refusal) as err:
        E.run_items(OnGpu(), items[:1], cond(), E.NativeRunner())
    assert err.value.reason == "device_mismatch"


def test_summary_shows_the_denominator_and_strata(tiny_agent, dev_items):
    _, items = dev_items
    recs = E.run_items(tiny_agent, items, cond(), E.NativeRunner())
    s = E.summarize_records(recs)
    assert s["n_items"] == len(items) == s["n_measured"] and s["counts_by_status"] == {"measured": len(items)}
    assert set(s["accuracy"]["by_question_type"]) == {"choice", "noul", "score"}
    assert set(s["accuracy"]["by_workflow"]) == set(D.WORKFLOWS)
    assert {"low_confidence", "not_low_confidence", "argmax_disagree", "tv_top_quartile"} <= set(s["accuracy"]["by_stratum"])
    assert s["ece_raw"]["overall"] is not None and s["mean_abs_level_error"] is not None
    with_fail = recs + [dict(recs[0], status="failed", reason="x", item_id="zzz")]
    assert E.summarize_records(with_fail)["accuracy"]["n"] == len(items)      # accuracy over measured items only


# --------------------------------------------------------------------------- solvability

def test_majority_labels_use_the_train_split(tiny_upstream, tmp_path):
    root = tmp_path / "d"
    D.import_dataset(root=root, download=tiny_upstream.download, with_checkpoints=False)
    maj = E.majority_labels(D.load_cases("train", root))
    assert set(maj) == {"%s|%s" % (wf, q) for wf in D.WORKFLOWS
                        for q in ("action", "outcome", "needs_review", "risk", "urgency")}


def _records(items, correct_of):
    return [{"item_id": it["item_id"], "case_id": it["case_id"], "workflow": it["workflow"],
             "question_id": it["question_id"], "status": "measured", "correct": correct_of(it),
             "question_type": it["question_type"], "prob_predicted": 0.6} for it in items]


def test_solvability_rule_and_reference(dev_items):
    _, items = dev_items
    # A majority reference that is wrong for every item (it predicts a label no item has).
    majority = {"%s|%s" % (i["workflow"], i["question_id"]): "__never__" for i in items}
    good = E.solvability_for(_records(items, lambda i: True), items, majority)
    # A model that is never right cannot beat any reference.
    bad = E.solvability_for(_records(items, lambda i: False), items, majority)
    assert good["solves"] is True and bad["solves"] is False
    assert good["paired_vs_majority"]["lo"] > 0 and bad["paired_vs_majority"]["hi"] <= 0
    rep = E.solvability_report({"fine_tuned": good, "base": bad}, ["fine_tuned", "base"])
    assert rep["reference_checkpoint"] == "fine_tuned" and rep["note"] is None
    none = E.solvability_report({"fine_tuned": bad, "base": bad}, ["fine_tuned", "base"])
    assert none["reference_checkpoint"] == "none" and "valid reference" in none["note"]


# --------------------------------------------------------------------------- T037: the evaluation driver

from experiments import audit_items as A  # noqa: E402
from experiments import families as F  # noqa: E402

FAM_VARIANTS = ("distractor@mid", "neutral@mid")
FAM_LENGTHS = (256, 512)


@pytest.fixture(scope="module")
def fam(tiny_agent, tiny_upstream, tiny_checkpoint, tmp_path_factory):
    root = tmp_path_factory.mktemp("driver-data")
    D.import_dataset(root=root, download=tiny_upstream.download, with_checkpoints=False)
    D.make_splits(seed=3, root=root)
    meta = None
    for split in D.EVAL_SPLITS:
        meta = F.build_split(tiny_agent, split, lengths=FAM_LENGTHS, variants=FAM_VARIANTS, seed=0, root=root,
                             tokenizer_name="tiny")
    return {"root": root, "fid": meta["families_id"], "model": tiny_checkpoint}


def in_process(function, spec, time_cap=None):
    """Stand-in for `runner.run_condition`: same child function, no subprocess."""
    assert function == "experiments.evalrun:eval_condition"
    return E.eval_condition(dict(spec, time_cap=spec.get("time_cap")))


def cnd(name, length, **kw):
    return E.make_condition(name, length, **kw)


def test_condition_id_is_stable_and_marks_a_non_default_checkpoint():
    a = cnd("native", 512)
    assert a["condition_id"] == "native.none.cpu.L512" == cnd("native", 512)["condition_id"]
    assert cnd("native", 512, model=E.CHECKPOINTS["base"])["condition_id"] == "native.ckptbase.none.cpu.L512"
    assert cnd("window", 2048, params={"size": 256})["condition_id"] == "window.size256.none.cpu.L2048"
    assert cnd("native", 512, variant="int8_encoder")["condition_id"] == "native.int8_encoder.cpu.L512"


def test_tier_rule_runs_long_lengths_on_the_half_sample(fam):
    splits = D.read_data_json("splits.json", fam["root"])
    items = [{"case_id": c, "length": L, "variant": "distractor@mid", "split": "dev", "item_id": "%s|%d" % (c, L)}
             for c in splits["splits"]["dev"]["case_ids"] for L in (512, 2048, 4096, 8192)]
    for L in (512, 2048):
        assert len(E.select_items(items, "dev", splits, L, ["distractor@mid"])) == 12          # every dev case
    for L in (4096, 8192):
        got = {i["case_id"] for i in E.select_items(items, "dev", splits, L, ["distractor@mid"])}
        assert got == set(splits["half_sample"]["dev"]) and len(got) == 8                   # the recorded half-sample
    sub = E.select_items(items, "dev", splits, 512, ["distractor@mid"], case_ids=splits["variant_sample"])
    assert {i["case_id"] for i in sub} == set(splits["variant_sample"])
    assert len(E.select_items(items, "dev", splits, 512, ["distractor@mid"], max_cases=3)) == 3
    assert E.select_items(items, "dev", splits, 512, ["neutral@mid"]) == []


def test_run_eval_refuses_the_final_split_and_a_missing_or_stale_audit(fam, tmp_path):
    kw = dict(model=fam["model"], families_id=fam["fid"], variants=FAM_VARIANTS, revision=None,
              data_root=fam["root"], runner_fn=in_process)
    with pytest.raises(SplitLocked) as err:
        E.run_eval(tmp_path, "final", [cnd("native", 256)], require_audit=False, **kw)
    assert err.value.reason == "split_locked" and "FR-013" in str(err.value)
    import os
    result = fam["root"] / "audit_result.json"
    if result.exists():
        os.remove(result)
    with pytest.raises(Refusal) as err:
        E.run_eval(tmp_path, "dev", [cnd("native", 256)], **kw)
    assert err.value.reason == "audit_required"
    result.write_text('{"families_id": "other", "passed": true}', encoding="utf-8")
    with pytest.raises(Refusal) as err:
        E.run_eval(tmp_path, "dev", [cnd("native", 256)], **kw)
    assert err.value.reason == "audit_stale"
    result.write_text('{"families_id": "%s", "passed": true}' % fam["fid"], encoding="utf-8")
    out = E.run_eval(tmp_path, "dev", [cnd("native", 256)], max_cases=2, **kw)
    assert out[0]["status"] == "measured"                                # a passing audit of these families opens the gate
    os.remove(result)


def test_run_eval_scores_every_item_once_resumes_and_tags_the_condition(fam, tmp_path):
    kw = dict(model=fam["model"], families_id=fam["fid"], variants=["distractor@mid"], revision=None,
              data_root=fam["root"], runner_fn=in_process, require_audit=False)
    conds = [cnd("native", 512), cnd("trunc512", 512), cnd("oracle", 512)]
    out = E.run_eval(tmp_path, "dev", conds, **kw)
    assert [o["status"] for o in out] == ["measured"] * 3
    for c in conds:
        recs = read_jsonl(tmp_path / "quality" / c["condition_id"] / "predictions.jsonl")
        assert len(recs) == 12 * 5 and all(r["split"] == "dev" and r["device"] == "cpu" for r in recs)
        assert len({r["item_id"] for r in recs}) == len(recs)
    oracle = read_jsonl(tmp_path / "quality" / "oracle.none.cpu.L512" / "predictions.jsonl")
    assert all(r["deployable"] is False and r["evidence_visible"] == "full" for r in oracle)
    path = tmp_path / "quality" / conds[0]["condition_id"] / "predictions.jsonl"
    before = path.read_text(encoding="utf-8")
    again = E.run_eval(tmp_path, "dev", conds[:1], **kw)
    assert again[0]["n_done_now"] == 0 and again[0]["n_done_before"] == 60          # resumed: nothing redone
    assert path.read_text(encoding="utf-8") == before


def test_a_time_cap_is_relaunched_until_done(fam, tmp_path):
    launches = []

    def capped(function, spec, time_cap=None):
        launches.append(1)
        return in_process(function, dict(spec, time_cap=0.001 if len(launches) < 3 else None))

    out = E.run_eval(tmp_path, "dev", [cnd("native", 256)], model=fam["model"], families_id=fam["fid"],
                     variants=["distractor@mid"], revision=None, data_root=fam["root"], runner_fn=capped,
                     require_audit=False)
    assert out[0]["status"] == "measured" and len(launches) >= 3
    recs = read_jsonl(tmp_path / "quality" / "native.none.cpu.L256" / "predictions.jsonl")
    assert len({r["item_id"] for r in recs}) == len(recs) == 60


def test_a_capped_run_is_not_reported_complete_because_another_split_shares_the_predictions_file(fam, tmp_path):
    """Dev and calibration results of one condition share a predictions file; progress counts this split's items."""
    kw = dict(model=fam["model"], families_id=fam["fid"], variants=["distractor@mid"], revision=None,
              data_root=fam["root"], require_audit=False)
    E.run_eval(tmp_path, "dev", [cnd("native", 256)], runner_fn=in_process, **kw)
    launches = []

    def capped(function, spec, time_cap=None):
        launches.append(1)
        return in_process(function, dict(spec, time_cap=1e-9 if len(launches) == 1 else None))

    out = E.run_eval(tmp_path, "calibration", [cnd("native", 256)], runner_fn=capped, **kw)
    recs = read_jsonl(tmp_path / "quality" / "native.none.cpu.L256" / "predictions.jsonl")
    assert out[0]["status"] == "measured" and len(launches) >= 2
    assert len([r for r in recs if r["split"] == "calibration"]) == 40


def test_predictions_of_other_families_are_never_resumed_as_current(fam, tmp_path, tiny_agent):
    """Item ids do not change when the families are rebuilt (seed, rule), so stale results must be refused."""
    kw = dict(model=fam["model"], variants=["distractor@mid"], revision=None, data_root=fam["root"],
              runner_fn=in_process, require_audit=False)
    E.run_eval(tmp_path, "dev", [cnd("native", 256)], families_id=fam["fid"], max_cases=2, **kw)
    recs = read_jsonl(tmp_path / "quality" / "native.none.cpu.L256" / "predictions.jsonl")
    assert recs and {r["families_id"] for r in recs} == {fam["fid"]}
    other = F.build_split(tiny_agent, "dev", lengths=FAM_LENGTHS, variants=FAM_VARIANTS, seed=1, root=fam["root"],
                          tokenizer_name="tiny")["families_id"]
    assert other != fam["fid"]
    with pytest.raises(Refusal) as err:
        E.run_eval(tmp_path, "dev", [cnd("native", 256)], families_id=other, max_cases=2, **kw)
    assert err.value.reason == "fingerprint_mismatch"
    assert len(read_jsonl(tmp_path / "quality" / "native.none.cpu.L256" / "predictions.jsonl")) == len(recs)


def test_a_dead_child_leaves_its_items_failed_with_the_cause_and_the_run_continues(fam, tmp_path):
    def dies(function, spec, time_cap=None):
        return {"status": "failed", "cause": "oom", "reason": "MemoryError: out of memory", "exit_code": 3221225495}

    out = E.run_eval(tmp_path, "dev", [cnd("native", 256), cnd("native", 512)], model=fam["model"],
                     families_id=fam["fid"], variants=["distractor@mid"], revision=None, data_root=fam["root"],
                     runner_fn=dies, require_audit=False)
    assert [o["status"] for o in out] == ["failed", "failed"] and out[0]["n_marked_failed"] == 60
    recs = read_jsonl(tmp_path / "quality" / "native.none.cpu.L256" / "predictions.jsonl")
    assert len(recs) == 60 and all(r["status"] == "failed" and r["cause"] == "oom" for r in recs)
    assert "ended before this item finished" in recs[0]["reason"]
    # nothing was substituted: no measured record for these items exists at another length
    assert not (tmp_path / "quality" / "trunc512.none.cpu.L256").exists()


def test_unsupported_items_keep_their_place_with_a_reason(fam, tmp_path):
    E.run_eval(tmp_path, "dev", [cnd("native", 256)], model=fam["model"], families_id=fam["fid"],
               variants=["distractor@mid"], revision=None, data_root=fam["root"], runner_fn=in_process,
               require_audit=False)
    recs = read_jsonl(tmp_path / "quality" / "native.none.cpu.L256" / "predictions.jsonl")
    bad = [r for r in recs if r["status"] == "unsupported"]
    assert bad and all(r["reason"] for r in bad)                     # the long workflow does not fit 256 tokens
    assert {r["workflow"] for r in bad} == {"security_incidents"}


def test_tuning_uses_dev_only_and_writes_conditions_before_any_calibration_run(fam, tmp_path):
    body = E.tune_window(tmp_path, fam["model"], fam["fid"], revision=None, data_root=fam["root"],
                         sizes=("default", 60, 100), lengths=(512,), require_audit=False, runner_fn=in_process)
    assert body["tuned_on"] == "dev" and body["window"]["size"] in ("default", 60, 100)
    assert set(body["grid"]) == {"default", "60", "100"} and all(g["n"] > 0 for g in body["grid"].values())
    splits = D.read_data_json("splits.json", fam["root"])
    assert set(body["tuning_set"]["case_ids"]) == set(splits["variant_sample"])
    for s in ("default", 60, 100):
        cid = E.make_condition("window", 512, params={"size": s}, model=fam["model"])["condition_id"]
        recs = read_jsonl(tmp_path / "tuning" / cid / "predictions.jsonl")
        assert recs and {r["split"] for r in recs} == {"dev"} and {r["variant"] for r in recs} == {"distractor@mid"}
    assert not (tmp_path / "quality").exists()                       # tuning never writes evaluation results
    assert E.load_tuned_params(tmp_path)["size"] == body["window"]["size"]
    assert E.load_tuned_params(tmp_path / "nowhere")["size"] == "default"


def test_one_real_subprocess_runs_a_condition(fam, tmp_path):
    from experiments.runner import run_condition
    out = E.run_eval(tmp_path, "dev", [cnd("native", 256)], model=fam["model"], families_id=fam["fid"],
                     variants=["distractor@mid"], revision=None, data_root=fam["root"], max_cases=1,
                     require_audit=False, runner_fn=run_condition)
    assert out[0]["status"] == "measured" and out[0]["n_total"] == 5
    assert out[0].get("peak_rss_bytes")                                # the runner adds per-condition memory


# --------------------------------------------------------------------------- GPU-scored quality (FR-024, 2026-09-30)

class _AsDevice:
    """Wraps the tiny CPU agent and reports another device type; everything else is the real agent."""

    def __init__(self, agent, kind):
        self._agent, self.device = agent, type("D", (), {"type": kind})()

    def __getattr__(self, name):
        return getattr(self._agent, name)


def test_a_gpu_condition_labels_its_records_and_a_device_mismatch_is_refused(tiny_agent, dev_items):
    _, items = dev_items
    gpu_cond = dict(cond(), device="gpu", condition_id="native.none.gpu.Loriginal")
    recs = E.run_items(_AsDevice(tiny_agent, "cuda"), items[:3], gpu_cond, E.NativeRunner())
    assert [r["device"] for r in recs] == ["gpu"] * 3 and all(r["status"] == "measured" for r in recs)
    with pytest.raises(Refusal) as err:
        E.run_items(tiny_agent, items[:1], gpu_cond, E.NativeRunner())            # a CPU agent for a GPU condition
    assert err.value.reason == "device_mismatch"
    with pytest.raises(Refusal):
        E.run_items(_AsDevice(tiny_agent, "cuda"), items[:1], cond(), E.NativeRunner())   # and the other way round
    assert E.run_items(tiny_agent, items[:1], cond(), E.NativeRunner())[0]["device"] == "cpu"


def test_run_eval_refuses_gpu_conditions_without_cuda_and_writes_nothing(fam, tmp_path):
    import torch
    if torch.cuda.is_available():
        pytest.skip("this environment has CUDA")
    with pytest.raises(Refusal) as err:
        E.run_eval(tmp_path, "dev", [cnd("native", 256, device="gpu")], model=fam["model"], families_id=fam["fid"],
                   variants=["distractor@mid"], revision=None, data_root=fam["root"], runner_fn=in_process,
                   require_audit=False)
    assert err.value.reason == "device_mismatch" and "never scored on another device" in str(err.value)
    assert not (tmp_path / "quality").exists()


def test_variants_are_unsupported_on_the_gpu_without_loading_a_model(fam):
    for v in ("int8_encoder", "int8_all_nofast", "fastpath_off"):
        out = E.eval_condition({"condition": cnd("native", 256, variant=v, device="gpu"), "split": "dev",
                                "families_id": fam["fid"], "variants": ["distractor@mid"], "model": fam["model"],
                                "revision": None, "threads": 1, "out_dir": "unused", "data_root": str(fam["root"])})
        assert out["status"] == "unsupported" and "CPU-only" in out["reason"]


def test_gpu_condition_ids_carry_the_device(fam):
    c = cnd("native", 2048, device="gpu")
    assert c["condition_id"] == "native.none.gpu.L2048" and c["device"] == "gpu"
    t = E.tune_window.__defaults__          # tuning takes a device too
    assert "cpu" in t


def test_final_split_runs_all_cases_at_4k_and_8k_while_dev_and_calibration_keep_the_half_sample():
    """plan v2 (laya:008): 200 final cases at 4,096 and 8,192; no final item is scored here (only selected)."""
    cases = {"dev": ["d%d" % i for i in range(6)], "calibration": ["c%d" % i for i in range(4)],
             "final": ["f%d" % i for i in range(10)]}
    splits = {"splits": {s: {"case_ids": ids} for s, ids in cases.items()},
              "half_sample": {"dev": cases["dev"][:3], "calibration": cases["calibration"][:2], "final": cases["final"][:5]}}
    items = [{"case_id": c, "length": L, "variant": "distractor@mid", "split": s, "item_id": "%s|%d" % (c, L)}
             for s, ids in cases.items() for c in ids for L in (512, 2048, 4096, 8192)]
    for L in (4096, 8192):
        assert {i["case_id"] for i in E.select_items(items, "final", splits, L, ["distractor@mid"])} == set(cases["final"])
        assert len(E.select_items(items, "dev", splits, L, ["distractor@mid"])) == 3
        assert len(E.select_items(items, "calibration", splits, L, ["distractor@mid"])) == 2
    for L in (512, 2048):
        assert len(E.select_items(items, "final", splits, L, ["distractor@mid"])) == 10
    # only the length rule changed: a case-id restriction still applies to the final split
    assert len(E.select_items(items, "final", splits, 4096, ["distractor@mid"], case_ids=["f1", "f9"])) == 2


def test_mask_only_candidates_apply_to_a_gpu_agent_but_cpu_paths_do_not(fastpath_checkpoint):
    import laya
    from experiments.results import Refusal
    agent = laya.Agent(fastpath_checkpoint, device="cpu")          # stands in for a CUDA agent: the helper is device-blind
    rec = E.apply_mask_only_variant(agent, "b1_mask")
    assert rec["variant"] == "b1_mask" and rec["mask_only"] is True and rec.get("composed_with") is None
    agent.model.__dict__.pop("forward", None)
    for bad in ("a1", "b1", "local_exact", "fastpath_off", "none"):
        with pytest.raises(Refusal):
            E.apply_mask_only_variant(agent, bad)
