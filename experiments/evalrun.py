"""Quality runs: score evaluation items with a checkpoint, one forward pass per question row.

Specs: specs/002-decision-benchmark-baselines (research.md R7, R10, R11, R14; FR-006, FR-007,
FR-013, FR-016, FR-018, FR-021, FR-023, FR-025).

Every item is one question row (clarified 2026-09-29), so each is scored, and later timed, on its
own. Results are appended to ``quality/<condition_id>/predictions.jsonl`` as they finish; a restart
skips finished items. Quality is scored on CPU only (FR-024): `run_items` refuses another device.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from . import data, metrics, tokens
from .results import Refusal, append_jsonl, assert_no_final, read_jsonl

from .baselines import (NativeRunner, PROB_SCOPE_FULL as PROB_SCOPE_NATIVE, evidence_fraction,  # noqa: F401  (re-exported)
                        visibility)


# --------------------------------------------------------------------------- answers and scoring

def gold_label(item: Dict[str, Any]) -> str:
    return str(item["gold"]["label"])


def option_labels(item: Dict[str, Any]) -> List[str]:
    """Answer labels of the question in the order the model saw them."""
    q = item["question"]
    crit = q.get("criteria")
    if q["type"] == "noul":
        return ["false", "true"]
    if q["type"] == "score":
        return [str(i) for i in range(len(crit))]
    return [str(k) for k in crit]


def parse_answer(item: Dict[str, Any], answer: Dict[str, Any]) -> Dict[str, Any]:
    """Predicted label, the full probability map, and the *predicted answer's* probability."""
    t = item["question"]["type"]
    if t == "choice":
        probs = {str(k): float(v) for k, v in answer["probabilities"].items()}
        pred = str(answer["choice"])
    elif t == "score":
        probs = {str(k): float(v) for k, v in answer["probabilities"].items()}
        pred = max(probs, key=lambda k: (probs[k], -int(k)))     # most probable level; ties go to the lower level
    elif t == "noul":
        p_true = float(answer["noul"])
        probs = {"false": 1.0 - p_true, "true": p_true}
        pred = "true" if p_true > 0.5 else "false"
    else:
        raise ValueError("unknown question type %r" % t)
    return {"predicted": pred, "probabilities": probs, "prob_predicted": probs[pred]}


def score_item(item: Dict[str, Any], answer: Dict[str, Any]) -> Dict[str, Any]:
    """Exact match with the gold label for every type; absolute level error for ordinal questions."""
    out = parse_answer(item, answer)
    gold = gold_label(item)
    out["gold"] = gold
    out["correct"] = out["predicted"] == gold
    out["level_error"] = (abs(int(out["predicted"]) - int(gold))
                          if item["question"]["type"] == "score" else None)
    return out


# --------------------------------------------------------------------------- running items

def make_record(item: Dict[str, Any], condition: Dict[str, Any], **fields: Any) -> Dict[str, Any]:
    g = item["gold"]
    rec = {"item_id": item["item_id"], "condition_id": condition["condition_id"], "case_id": item["case_id"],
           "question_id": item["question_id"], "split": item["split"], "workflow": item["workflow"],
           "question_type": item["question_type"], "variant": item["variant"], "length": item["length"],
           "low_confidence": g.get("low_confidence"), "argmax_agree": g.get("argmax_agree"),
           "tv_top_quartile": g.get("tv_top_quartile"), "device": condition.get("device", "cpu")}
    if condition.get("families_id"):
        rec["families_id"] = condition["families_id"]
    rec.update(fields)
    return rec


def run_items(agent, items: Iterable[Dict[str, Any]], condition: Dict[str, Any], runner: Callable,
              out_path: Optional[Path] = None, time_cap: Optional[float] = None,
              on_record: Optional[Callable[[Dict[str, Any]], None]] = None) -> List[Dict[str, Any]]:
    """Score `items` with `runner`, appending each result to `out_path` as it finishes.

    Refuses the final split (FR-013) and an agent on a device other than the condition's (``cpu`` by
    default, ``gpu`` for GPU-scored quality; FR-024 labels every result with its device). Items already in `out_path` are
    skipped, so an interrupted run resumes (R11). An item that cannot run keeps its place with
    ``status`` ``unsupported`` or ``failed`` and a ``reason``; it is never replaced by another
    length, device or configuration (FR-025). Stops with ``partial`` items left unwritten when
    `time_cap` seconds have passed, so a rerun continues from there.
    """
    items = list(items)
    assert_no_final(items)
    want = "cuda" if condition.get("device", "cpu") == "gpu" else "cpu"
    if agent.device.type != want:
        raise Refusal("device_mismatch", "this condition is scored on %s but the agent is on %s; results are never "
                                         "attributed to another device (FR-024)" % (condition.get("device", "cpu"), agent.device))
    done = {r["item_id"] for r in read_jsonl(out_path)} if out_path else set()
    started = time.perf_counter()
    records: List[Dict[str, Any]] = []
    for item in items:
        if item["item_id"] in done:
            continue
        if time_cap is not None and time.perf_counter() - started > time_cap:
            break
        if item.get("status") != "ok":
            rec = make_record(item, condition, status="unsupported",
                              reason=item.get("reason") or "item is not runnable")
        else:
            try:
                out = runner(agent, item)
                scored = score_item(item, out["answer"])
                rec = make_record(item, condition, status="measured", **scored,
                                  **{k: v for k, v in out.items() if k != "answer"})
            except Exception as exc:               # recorded, never substituted
                cause = "oom" if isinstance(exc, MemoryError) or "not enough memory" in str(exc).lower() else "exception"
                rec = make_record(item, condition, status="failed", cause=cause,
                                  reason="%s: %s" % (type(exc).__name__, exc))
        if out_path:
            append_jsonl(out_path, rec)
        records.append(rec)
        if on_record:
            on_record(rec)
    return records


# --------------------------------------------------------------------------- summaries

def _acc(recs: Sequence[Dict[str, Any]]) -> Optional[float]:
    return metrics.accuracy([r["correct"] for r in recs])


def summarize_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Accuracy per question type, workflow and stratum, ordinal level error and raw ECE.

    Accuracy is computed over ``measured`` items only, and the denominator is shown (contracts rule 2).
    """
    counts: Dict[str, int] = {}
    for r in records:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    m = [r for r in records if r["status"] == "measured"]

    def group(key: Callable[[Dict[str, Any]], Any]) -> Dict[str, Any]:
        keys = sorted({key(r) for r in m}, key=str)
        return {str(k): {"accuracy": _acc([r for r in m if key(r) == k]), "n": sum(1 for r in m if key(r) == k)}
                for k in keys}

    lev = [r for r in m if r.get("level_error") is not None]
    strata = {
        "low_confidence": {"accuracy": _acc([r for r in m if r.get("low_confidence")]),
                           "n": sum(1 for r in m if r.get("low_confidence"))},
        "not_low_confidence": {"accuracy": _acc([r for r in m if r.get("low_confidence") is False]),
                               "n": sum(1 for r in m if r.get("low_confidence") is False)},
        "argmax_disagree": {"accuracy": _acc([r for r in m if r.get("argmax_agree") is False]),
                            "n": sum(1 for r in m if r.get("argmax_agree") is False)},
        "tv_top_quartile": {"accuracy": _acc([r for r in m if r.get("tv_top_quartile")]),
                            "n": sum(1 for r in m if r.get("tv_top_quartile"))},
    }
    by_vis = group(lambda r: r.get("evidence_visible"))
    return {
        "n_items": len(records), "n_measured": len(m), "counts_by_status": dict(sorted(counts.items())),
        "accuracy": {"overall": _acc(m), "n": len(m), "by_question_type": group(lambda r: r["question_type"]),
                     "by_workflow": group(lambda r: r["workflow"]), "by_stratum": strata,
                     "by_evidence_visible": by_vis},
        "mean_abs_level_error": float(np.mean([r["level_error"] for r in lev])) if lev else None,
        "ece_raw": {"overall": metrics.ece([r["prob_predicted"] for r in m], [r["correct"] for r in m]) if m else None,
                    "by_question_type": {t: metrics.ece([r["prob_predicted"] for r in m if r["question_type"] == t],
                                                        [r["correct"] for r in m if r["question_type"] == t])
                                         for t in sorted({r["question_type"] for r in m})}},
    }


# --------------------------------------------------------------------------- solvability (R14)

def majority_labels(train_cases: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    """Most frequent train label per (workflow, question); the ``per-question majority class`` reference."""
    tally: Dict[str, Dict[str, int]] = {}
    for c in train_cases:
        for qid, g in c["gold"].items():
            key = "%s|%s" % (c["workflow"], qid)
            tally.setdefault(key, {})
            tally[key][str(g["label"])] = tally[key].get(str(g["label"]), 0) + 1
    return {k: max(v, key=lambda lab: (v[lab], lab)) for k, v in tally.items()}


def solvability_for(records: Sequence[Dict[str, Any]], items: Sequence[Dict[str, Any]],
                    majority: Dict[str, str], seed: int = 0, n_boot: int = 2000) -> Dict[str, Any]:
    """One checkpoint's original-length result against the majority-class reference.

    ``solves`` is true only when its accuracy is above the majority accuracy with a case-clustered
    paired interval that excludes zero (research.md R14).
    """
    by_id = {it["item_id"]: it for it in items}
    m = [r for r in records if r["status"] == "measured"]
    maj_correct = [str(majority.get("%s|%s" % (r["workflow"], r["question_id"]))) == str(by_id[r["item_id"]]["gold"]["label"])
                   for r in m]
    paired = metrics.paired_cluster_bootstrap([r["correct"] for r in m], maj_correct, [r["case_id"] for r in m],
                                              n_boot=n_boot, seed=seed)
    summary = summarize_records(records)
    return {"summary": summary, "majority_accuracy": metrics.accuracy(maj_correct),
            "paired_vs_majority": paired,
            "solves": bool(paired["lo"] is not None and paired["lo"] > 0)}


def solvability_report(per_model: Dict[str, Dict[str, Any]], reference_order: Sequence[str]) -> Dict[str, Any]:
    """Combine per-checkpoint results and name the quality reference (FR-006, FR-007).

    `reference_order` lists the checkpoint names in preference order (the fine-tuned one first); the
    first that solves the short cases becomes the reference, or ``none`` when none does.
    """
    reference = "none"
    for name in reference_order:
        if per_model.get(name, {}).get("solves"):
            reference = name
            break
    note = None
    if reference == "none":
        note = ("no checkpoint solves the original-length development cases; later quality comparisons "
                "lack a valid reference and must be labeled so (FR-006)")
    return {"models": per_model, "reference_checkpoint": reference, "note": note}


# --------------------------------------------------------------------------- checkpoints and the solvability run

#: Short names of the two native checkpoints, in preference order for the quality reference.
CHECKPOINTS = {"fine_tuned": "convaiinnovations/laya-typed-decisions", "base": "convaiinnovations/laya"}


def native_condition(ckpt: str, length: Any = "original", variant: str = "none", device: str = "cpu") -> Dict[str, Any]:
    from .results import condition_id
    return {"name": "native", "params": {"ckpt": ckpt}, "variant": variant, "device": device, "length": length,
            "condition_id": condition_id("native", {"ckpt": ckpt}, variant, device, length)}


def solvability_condition(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Child-side entry point: run one checkpoint on the original-length items of a split (CPU)."""
    from .manifest import build_manifest
    from .runner import load_agent

    agent, info = load_agent(spec["model"], spec.get("revision"), spec.get("threads"), mha_fastpath=True)
    root = Path(spec["data_root"]) if spec.get("data_root") else None
    items = data.original_items(spec["split"], root=root)
    cond = native_condition(spec["ckpt"])
    out = Path(spec["out_dir"]) / "quality" / cond["condition_id"] / "predictions.jsonl"
    started = time.perf_counter()
    recs = run_items(agent, items, cond, NativeRunner(), out_path=out, time_cap=spec.get("time_cap"))
    man = build_manifest(agent, info, run_id=Path(spec["out_dir"]).name, seed=int(spec.get("seed", 0)),
                         threads_source=spec.get("threads_source", "default"),
                         data=data.manifest_data_block(root))
    return {"status": "measured", "condition_id": cond["condition_id"], "n_run_now": len(recs),
            "n_items": len(items), "elapsed_s": time.perf_counter() - started,
            "model": {"id": info["model"], "revision": info["revision"]},
            "configured_max_len": int(agent.cfg.get("max_len", 512)), "manifest": man}


def run_solvability(run_path: Path, models: Sequence[str] = ("fine_tuned", "base"), split: str = "dev",
                    threads: Optional[int] = None, threads_source: str = "default", seed: int = 0,
                    data_root: Optional[Path] = None, time_cap: Optional[float] = None,
                    checkpoints: Optional[Dict[str, str]] = None, revision: Optional[str] = "reviewed",
                    runner_fn: Optional[Callable] = None) -> Dict[str, Any]:
    """Run each checkpoint on original-length items of `split`, then decide the quality reference.

    One subprocess per checkpoint. Writes ``solvability.json`` (and ``manifest.json`` if absent).
    """
    from . import results
    from .runner import run_condition

    if split not in ("dev", "calibration"):
        raise results.SplitLocked("solvability runs on dev or calibration items only (FR-013)")
    checkpoints = checkpoints or CHECKPOINTS
    run_fn = runner_fn or run_condition
    items = data.original_items(split, root=data_root)
    train = data.load_cases("train", data_root)
    majority = majority_labels(train)
    per_model: Dict[str, Any] = {}
    manifest = None
    for ckpt in models:
        spec = {"ckpt": ckpt, "model": checkpoints[ckpt], "revision": revision, "threads": threads,
                "threads_source": threads_source, "seed": seed, "split": split, "out_dir": str(run_path),
                "data_root": str(data_root) if data_root else None, "time_cap": time_cap}
        item = run_fn("experiments.evalrun:solvability_condition", spec, time_cap=time_cap)
        if item.get("status") == "failed":
            per_model[ckpt] = {"status": "failed", "reason": item.get("reason"), "solves": False}
            continue
        manifest = manifest or item.pop("manifest", None)
        cid = item["condition_id"]
        records = read_jsonl(Path(run_path) / "quality" / cid / "predictions.jsonl")
        res = solvability_for(records, items, majority, seed=seed)
        res.update({"status": item.get("status", "measured"), "condition_id": cid, "model": item.get("model"),
                    "configured_max_len": item.get("configured_max_len"), "n_items": len(items)})
        per_model[ckpt] = res
    report = solvability_report(per_model, [m for m in models])
    report.update({"split": split, "majority_reference": "per (workflow, question) majority label of the train split"})
    results.write_json(run_path, "solvability.json", report)
    if manifest is not None and not (Path(run_path) / "manifest.json").exists():
        results.write_json(run_path, "manifest.json", manifest)
    return report


# --------------------------------------------------------------------------- the evaluation driver (T043)
#
# One (condition, length) pair per subprocess; each appends to quality/<condition_id>/predictions.jsonl
# and resumes from it. The parent relaunches a child that stopped at its time cap, and records what a
# dead child could not finish as `failed` (FR-025). The final split is refused (FR-013) and no run
# starts before the audit of the current families has passed (research.md R12).

TIER2_LENGTHS = (4096, 8192)          # run on the recorded half-sample only (research.md R11), except the final split
FINAL_FULL_LENGTHS = TIER2_LENGTHS    # plan v2 (docs/gate-plan-v2.md section 7): the final split runs all 200 cases there
VARIANT_LENGTHS = (512, 2048, 8192)   # optimized variants: fixed 20-case dev sample at these lengths
TUNE_LENGTHS = (1024, 2048)
DEFAULT_CHUNK_SECONDS = 1800.0
MAX_CHILD_LAUNCHES = 10_000


def ckpt_short(model: str) -> str:
    for k, v in CHECKPOINTS.items():
        if v == model:
            return k
    return str(model).replace("\\", "/").rsplit("/", 1)[-1]


def make_condition(name: str, length: int, variant: str = "none", device: str = "cpu",
                   params: Optional[Dict[str, Any]] = None, model: Optional[str] = None) -> Dict[str, Any]:
    """A baseline condition at one length; `params` carries the tuned window size."""
    from .results import condition_id
    p = dict(params or {})
    if model and ckpt_short(model) != "fine_tuned":
        p["ckpt"] = ckpt_short(model)
    return {"name": name, "params": p, "variant": variant, "device": device, "length": int(length),
            "condition_id": condition_id(name, p, variant, device, int(length))}


def select_items(items: Iterable[Dict[str, Any]], split: str, splits: Dict[str, Any], length: int,
                 variants: Sequence[str], case_ids: Optional[Sequence[str]] = None,
                 max_cases: Optional[int] = None) -> List[Dict[str, Any]]:
    """The items one condition runs: one length, the chosen variants, and the tier rule.

    Lengths of 4,096 and 8,192 tokens run on the split's recorded half-sample (research.md R11), except that the
    final split runs all of its cases there (plan v2; nothing scores a final item before the plan is frozen). `case_ids`
    restricts further (the variant sample); `max_cases` keeps the first N cases in sorted order (a pilot,
    which the caller must record).
    """
    allowed = set(splits["splits"][split]["case_ids"])
    if length in TIER2_LENGTHS and not (split == "final" and length in FINAL_FULL_LENGTHS):
        allowed &= set(splits["half_sample"][split])
    if case_ids is not None:
        allowed &= set(case_ids)
    if max_cases is not None:
        allowed = set(sorted(allowed)[:max_cases])
    return [it for it in items if it["length"] == length and it["variant"] in variants
            and it["case_id"] in allowed and it["split"] == split]


def _iter_split_items(fid: str, split: str, root: Optional[Path]):
    from . import families
    return families.read_items(families.items_dir(fid, root) / ("%s.jsonl.gz" % split))


def apply_mask_only_variant(agent, variant: str) -> Dict[str, Any]:
    """Apply a mask-only candidate variant (``a1_mask`` ...) to a GPU agent; a quality diagnostic, never a cost number."""
    from . import variants as V
    if not variant.endswith("_mask") or variant not in V.CANDIDATE_VARIANTS:
        raise Refusal("device_mismatch", "variant %r is CPU-only; only mask-only candidate variants run on a GPU agent" % variant)
    rec = V.apply_variant(agent, variant)
    return {k: v for k, v in rec.items() if k != "restore"}


def eval_condition(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Child-side entry point: run one (condition, length) pair on its device, resuming from its predictions file."""
    from . import variants as V
    from .baselines import build_runner

    cond = spec["condition"]
    reason = V.unsupported_reason(cond["variant"], cond.get("device", "cpu"))
    if reason:
        return {"status": "unsupported", "condition_id": cond["condition_id"], "reason": reason, "n_total": 0,
                "n_done_before": 0, "n_done_now": 0, "n_remaining": 0}
    root = Path(spec["data_root"]) if spec.get("data_root") else None
    splits = data.read_data_json("splits.json", root)
    items = select_items(_iter_split_items(spec["families_id"], spec["split"], root), spec["split"], splits,
                         cond["length"], spec["variants"], spec.get("case_ids"), spec.get("max_cases"))
    out = Path(spec["out_dir"]) / spec.get("subdir", "quality") / cond["condition_id"] / "predictions.jsonl"
    # The file also holds this condition's results on the other splits; only this run's items count.
    done_before = len({it["item_id"] for it in items} & {r["item_id"] for r in read_jsonl(out)})
    started = time.perf_counter()
    if cond.get("device", "cpu") == "gpu":
        from .latency import _load_gpu_agent
        agent, vrec = _load_gpu_agent(spec["model"], spec.get("revision")), {"variant": "none"}
        if cond["variant"] != "none":
            vrec = apply_mask_only_variant(agent, cond["variant"])
    else:
        agent, info, vrec = V.load_for_variant(spec["model"], spec.get("revision"), spec.get("threads"), cond["variant"])
    runner = build_runner(cond["name"], cond["params"])
    recs = run_items(agent, items, cond, runner, out_path=out, time_cap=spec.get("time_cap"))
    remaining = len(items) - done_before - len(recs)
    return {"status": "measured" if remaining <= 0 else "partial", "condition_id": cond["condition_id"],
            "n_total": len(items), "n_done_before": done_before, "n_done_now": len(recs),
            "n_remaining": max(0, remaining), "elapsed_s": time.perf_counter() - started,
            "variant": {k: v for k, v in vrec.items() if k != "restore"},
            "reason": None if remaining <= 0 else "time cap reached; rerun to continue"}


def _mark_failed(items: Sequence[Dict[str, Any]], cond: Dict[str, Any], out: Path, why: Dict[str, Any]) -> int:
    done = {r["item_id"] for r in read_jsonl(out)}
    n = 0
    for it in items:
        if it["item_id"] in done:
            continue
        append_jsonl(out, make_record(it, cond, status="failed", cause=why.get("cause", "crash"),
                                      reason="the measuring process ended before this item finished: %s"
                                             % (why.get("reason") or why.get("cause"))))
        n += 1
    return n


def run_eval(run_path: Path, split: str, conditions: Sequence[Dict[str, Any]], model: str, families_id: str,
             variants: Sequence[str], revision: Optional[str] = "reviewed", threads: Optional[int] = None,
             data_root: Optional[Path] = None, chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
             case_ids: Optional[Sequence[str]] = None, max_cases: Optional[int] = None,
             require_audit: bool = True, runner_fn: Optional[Callable] = None,
             subdir: str = "quality", log: Optional[Callable[[str], None]] = None,
             require_cuda: bool = True) -> List[Dict[str, Any]]:
    """Run every condition on `split`, one subprocess per (condition, length), relaunching to finish.

    Refuses ``final`` (FR-013), a missing or stale audit (unless `require_audit` is false), and GPU conditions
    without CUDA (nothing is quietly scored on another device, FR-024). A child that dies with no progress
    leaves its unfinished items `failed` with the cause, and the run continues.
    """
    from . import audit_items
    from . import variants as V
    from .results import Refusal, SplitLocked
    from .runner import run_condition

    if split == "final":
        raise SplitLocked("eval does not accept --split final: the final test split is not scored in this phase (FR-013)")
    if split not in ("dev", "calibration"):
        raise ValueError("split must be dev or calibration, got %r" % split)
    if require_audit:
        audit_items.require_audit(families_id, data_root)
    if require_cuda and any(c.get("device") == "gpu" and not V.unsupported_reason(c["variant"], "gpu") for c in conditions):
        import torch
        if not torch.cuda.is_available():
            raise Refusal("device_mismatch", "GPU-scored quality needs CUDA; run from the .venv-gpu environment "
                                             "(experiments/setup_gpu.ps1). Results are never scored on another device (FR-024)")
    run_fn = runner_fn or run_condition
    say = log or (lambda m: None)
    outcomes: List[Dict[str, Any]] = []
    for cond in conditions:
        out = Path(run_path) / subdir / cond["condition_id"] / "predictions.jsonl"
        # Item ids do not change when the families are rebuilt, so resuming would reuse results scored on other items.
        other = {r.get("families_id") for r in read_jsonl(out)} - {None, families_id}
        if other:
            raise Refusal("fingerprint_mismatch", "%s holds results scored on families %s, not the current %s; use "
                                                  "another --run-id for rebuilt families" % (out, sorted(other), families_id))
        cond = dict(cond, families_id=families_id)              # stamped into every record of this run
        spec = {"condition": cond, "split": split, "families_id": families_id, "variants": list(variants),
                "model": model, "revision": revision, "threads": threads, "out_dir": str(run_path),
                "data_root": str(data_root) if data_root else None, "case_ids": list(case_ids) if case_ids else None,
                "max_cases": max_cases, "time_cap": chunk_seconds, "subdir": subdir}
        last = None
        stalled = 0
        for _ in range(MAX_CHILD_LAUNCHES):
            before = len(read_jsonl(out))
            item = run_fn("experiments.evalrun:eval_condition", spec, time_cap=chunk_seconds + 600)
            if item.get("status") == "failed":            # the process died (oom, crash, timeout)
                progressed = len(read_jsonl(out)) > before
                stalled = 0 if progressed else stalled + 1
                say("%s: child %s (%s)" % (cond["condition_id"], item.get("cause"), "progress" if progressed else "no progress"))
                if stalled >= 2:
                    root = Path(data_root) if data_root else None
                    splits = data.read_data_json("splits.json", root)
                    todo = select_items(_iter_split_items(families_id, split, root), split, splits, cond["length"],
                                        variants, case_ids, max_cases)
                    n = _mark_failed(todo, cond, out, item)
                    last = {"status": "failed", "condition_id": cond["condition_id"], "n_marked_failed": n,
                            "cause": item.get("cause"), "reason": item.get("reason")}
                    break
                continue
            last = item
            say("%s: %s (%d/%d)" % (cond["condition_id"], item.get("status"),
                                    item.get("n_done_before", 0) + item.get("n_done_now", 0), item.get("n_total", 0)))
            if item.get("status") != "partial":
                break
            if item.get("n_done_now", 0) == 0 and chunk_seconds is not None:
                stalled += 1
                if stalled >= 2:
                    break
        outcomes.append(last or {"status": "failed", "condition_id": cond["condition_id"], "reason": "no result"})
    return outcomes


def tune_window(run_path: Path, model: str, families_id: str, revision: Optional[str] = "reviewed",
                threads: Optional[int] = None, data_root: Optional[Path] = None,
                sizes: Sequence[Any] = ("default", 256, 512), chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
                lengths: Sequence[int] = TUNE_LENGTHS, require_audit: bool = True,
                runner_fn: Optional[Callable] = None, log: Optional[Callable[[str], None]] = None,
                device: str = "cpu") -> Dict[str, Any]:
    """Choose the window size on the **dev** split only (FR-019) and write ``conditions.json``.

    `device` is where quality is scored (``gpu`` is fast; the choice is recorded).

    The tuning set is the fixed 20-case dev variant sample, `distractor@mid` items at `lengths`. The size with the
    highest accuracy wins; ties go to the one that processed fewer tokens. Retrieval budgets are not tuned here: each
    of 512, 1,024 and 2,048 runs as its own condition and `eval-summary` selects the best on dev.
    """
    from . import results
    splits = data.read_data_json("splits.json", data_root)
    case_ids = splits["variant_sample"]
    conds = [make_condition("window", L, params={"size": s}, model=model, device=device) for s in sizes for L in lengths]
    run_eval(run_path, "dev", conds, model, families_id, ["distractor@mid"], revision=revision, threads=threads,
             data_root=data_root, chunk_seconds=chunk_seconds, case_ids=case_ids, require_audit=require_audit,
             runner_fn=runner_fn, subdir="tuning", log=log)
    grid: Dict[str, Any] = {}
    for s in sizes:
        recs: List[Dict[str, Any]] = []
        for L in lengths:
            cid = make_condition("window", L, params={"size": s}, model=model, device=device)["condition_id"]
            recs += [r for r in read_jsonl(Path(run_path) / "tuning" / cid / "predictions.jsonl")]
        m = [r for r in recs if r["status"] == "measured"]
        grid[str(s)] = {"n": len(m), "accuracy": metrics.accuracy([r["correct"] for r in m]),
                        "mean_tokens_seen": float(np.mean([r["tokens_seen"] for r in m])) if m else None}
    scored = [(k, g) for k, g in grid.items() if g["n"]]
    if not scored:
        raise ValueError("no tuning results to choose from")
    best = min(scored, key=lambda kg: (-kg[1]["accuracy"], kg[1]["mean_tokens_seen"]))[0]
    body = {"tuned_on": "dev", "scored_on": device, "tuning_set": {"variant": "distractor@mid", "lengths": list(lengths),
                                              "case_ids": list(case_ids), "n_cases": len(case_ids)},
            "window": {"size": "default" if best == "default" else int(best), "stride": None},
            "grid": grid,
            "retrieve": {"tuned": False, "note": "each budget runs as its own condition; eval-summary selects the best on dev"}}
    results.write_json(run_path, "conditions.json", body)
    return body


def load_tuned_params(run_path: Path) -> Dict[str, Any]:
    """The tuned window parameters from ``conditions.json`` (default window when not tuned yet)."""
    try:
        from . import results
        return results.read_json(run_path, "conditions.json")["window"]
    except (OSError, KeyError, ValueError):
        return {"size": "default", "stride": None}
