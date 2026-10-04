"""Calibration on the calibration split, and the per-condition summary with paired comparisons (T044).

Specs: specs/002-decision-benchmark-baselines (research.md R10; FR-019, FR-021, FR-022, FR-027, FR-028).

* ``calibrate`` fits one temperature per (condition, question type) on **calibration-split** results
  only, and refuses anything else (FR-019).
* ``build_summary`` reads the **dev** results of every condition, applies those temperatures, and
  compares each condition with the native run at the matching length, the two truncations and the oracle,
  on the same items, with a case-clustered bootstrap (questions of one case are dependent, FR-022).
* Windowed probabilities are the deciding window's, not calibrated for the whole document, so their
  scaled calibration error is labeled non-comparable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import metrics
from .evalrun import summarize_records
from .results import read_json, read_jsonl, write_json

REFERENCE_CONDITIONS = ("native", "trunc512", "truncCap", "oracle")
WINDOW_NAME = "window"


def parse_condition_id(cid: str) -> Dict[str, Any]:
    """``<name>[.<params>].<variant>.<device>.L<length>`` back into its parts."""
    parts = cid.split(".")
    if len(parts) < 4 or not parts[-1].startswith("L"):
        raise ValueError("not a condition id: %r" % cid)
    length = parts[-1][1:]
    return {"name": parts[0], "params": ".".join(parts[1:-3]), "variant": parts[-3], "device": parts[-2],
            "length": int(length) if length.isdigit() else length}


def _vector(rec: Dict[str, Any]):
    labels = list(rec["probabilities"])
    return [float(rec["probabilities"][k]) for k in labels], labels.index(str(rec["gold"]))


def fit_condition(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """One temperature per question type from **calibration-split** records; refuses any other split."""
    bad = [r for r in records if r.get("split") != "calibration"]
    if bad:
        raise ValueError("temperatures are fitted on the calibration split only; got a %r record (FR-019)"
                         % bad[0].get("split"))
    out: Dict[str, Any] = {}
    for t in sorted({r["question_type"] for r in records if r["status"] == "measured"}):
        rows = [r for r in records if r["status"] == "measured" and r["question_type"] == t]
        vecs, gold = zip(*[_vector(r) for r in rows])
        out[t] = {"temperature": metrics.fit_temperature(list(vecs), list(gold)), "n": len(rows)}
    return out


def quality_dir(run_path: Path) -> Path:
    return Path(run_path) / "quality"


def condition_ids(run_path: Path) -> List[str]:
    d = quality_dir(run_path)
    return sorted(p.name for p in d.iterdir() if (p / "predictions.jsonl").exists()) if d.exists() else []


def calibrate(run_path: Path) -> Dict[str, Any]:
    """Fit temperatures for every condition that has calibration-split results; write ``calibration.json``."""
    body: Dict[str, Any] = {"fitted_on": "calibration", "conditions": {}}
    for cid in condition_ids(run_path):
        cal = [r for r in read_jsonl(quality_dir(run_path) / cid / "predictions.jsonl") if r.get("split") == "calibration"]
        if cal:
            body["conditions"][cid] = fit_condition(cal)
    write_json(run_path, "calibration.json", body)
    return body


def scaled_ece(records: Sequence[Dict[str, Any]], temps: Dict[str, Any]) -> Dict[str, Any]:
    """Classification ECE after applying each question type's calibration-split temperature."""
    conf, hit, by_type = [], [], {}
    for r in records:
        if r["status"] != "measured":
            continue
        t = temps.get(r["question_type"], {}).get("temperature", 1.0)
        vec, _ = _vector(r)
        labels = list(r["probabilities"])
        p = metrics.scale_probs(vec, t)[labels.index(r["predicted"])]
        conf.append(float(p))
        hit.append(bool(r["correct"]))
        by_type.setdefault(r["question_type"], ([], []))
        by_type[r["question_type"]][0].append(float(p))
        by_type[r["question_type"]][1].append(bool(r["correct"]))
    return {"overall": metrics.ece(conf, hit) if conf else None,
            "by_question_type": {t: metrics.ece(c, h) for t, (c, h) in sorted(by_type.items())}}


def paired(records: Sequence[Dict[str, Any]], ref_records: Sequence[Dict[str, Any]], n_boot: int = 5000,
           seed: int = 0) -> Optional[Dict[str, Any]]:
    """Case-clustered paired accuracy difference (condition minus reference) on the items both measured."""
    ref = {r["item_id"]: r for r in ref_records if r["status"] == "measured"}
    rows = [(r, ref[r["item_id"]]) for r in records if r["status"] == "measured" and r["item_id"] in ref]
    if not rows:
        return None
    out = metrics.paired_cluster_bootstrap([a["correct"] for a, _ in rows], [b["correct"] for _, b in rows],
                                           [a["case_id"] for a, _ in rows], n_boot=n_boot, seed=seed)
    out["verdict"] = metrics.verdict(out["lo"], out["hi"])
    return out


PARITY_MIN_SAME_PREDICTION = 0.98
PARITY_MAX_ACCURACY_DIFF = 0.005
#: A CPU/GPU label disagreement is tolerated as a near-tie flip when the CPU top-two probability margin on that item
#: is at most this (specs/002-decision-benchmark-baselines/parity-margin.md). The value is circular: chosen from the very flips it
#: judges, on 20 cases (5 distinct flipped items), with no held-out data. Basis: the 5 distinct bf16 flips of run p2-dev sat at CPU margins 0.0002, 0.0010, 0.0018, 0.0025 and 0.0092.
PARITY_NEAR_TIE_MARGIN = 0.01
#: A cell with fewer items than this is listed but not counted in the overall verdict (the fp32 8,192-token cell of
#: run p2-parity-fp32 has 2 items, scored through the CPU out-of-memory fallback).
PARITY_MIN_ITEMS = 20


def top_two_margin(probabilities: Dict[str, Any]) -> float:
    """Top-1 minus top-2 probability; 1.0 when there is no second option (a one-option or empty answer cannot tie)."""
    p = sorted((float(v) for v in probabilities.values()), reverse=True)
    return p[0] - p[1] if len(p) > 1 else 1.0


def parity(run_path: Path, split: str = "dev", margin: float = PARITY_NEAR_TIE_MARGIN,
           gpu_run_path: Optional[Path] = None, cpu_label: Optional[str] = None,
           gpu_label: Optional[str] = None, min_items: int = PARITY_MIN_ITEMS) -> Dict[str, Any]:
    """CPU against GPU on the items both scored (FR-024): the check that lets GPU-scored quality stand in for CPU.

    For every condition scored on both devices it reports the share of identical predicted answers, the accuracy on
    each device, and the largest probability difference. Two verdicts per row:

    * ``strict_passed``: at least 98% identical predictions and an accuracy difference of at most 0.5 points
      (the original rule; kept for comparison).
    * ``passed`` (the criterion): no label disagreement whose CPU top-two margin exceeds `margin`. Disagreements at or
      below the margin are tolerated near-tie flips, listed with their margins in ``tolerated_flips``; the others are
      listed in ``failing_flips``. This explicitly replaces the 0.5-point accuracy rule: both devices score the same
      items against the same gold, so accuracy can differ only through items whose predicted answer differs. With no
      disagreement the accuracy difference is exactly 0, and otherwise it is fully accounted for by the listed flips
      (``accuracy_difference_from_flips`` equals ``accuracy_difference``). A cell whose accuracy moved by more than
      0.5 points because of tolerated flips passes ``passed`` and misses ``strict_passed``, by design.

    A row with fewer than `min_items` shared items is still listed, with ``counted: false``, but is left out of the overall
    ``passed`` / ``strict_passed`` (a handful of items says nothing about parity).

    `gpu_run_path` reads the GPU predictions from another run directory (for example a rerun at another autocast
    dtype); `cpu_label` and `gpu_label` are free-text hardware and dtype labels carried into the result.
    """
    if not (np.isfinite(margin) and margin >= 0):
        raise ValueError("the parity margin must be a finite number >= 0, got %r" % (margin,))
    run_path = Path(run_path)
    gpu_root = Path(gpu_run_path) if gpu_run_path is not None else run_path
    ids = condition_ids(run_path)
    gpu_ids = condition_ids(gpu_root)
    rows: List[Dict[str, Any]] = []
    for cid in ids:
        info = parse_condition_id(cid)
        if info["device"] != "cpu" or info["variant"] != "none":
            continue
        twin = cid.replace(".cpu.", ".gpu.")
        if twin not in gpu_ids:
            continue
        cpu = {r["item_id"]: r for r in read_jsonl(quality_dir(run_path) / cid / "predictions.jsonl")
               if r.get("split") == split and r["status"] == "measured"}
        gpu = {r["item_id"]: r for r in read_jsonl(quality_dir(gpu_root) / twin / "predictions.jsonl")
               if r.get("split") == split and r["status"] == "measured"}
        shared = sorted(set(cpu) & set(gpu))
        if not shared:
            continue
        flips = [{"item_id": i, "cpu_margin": top_two_margin(cpu[i]["probabilities"]),
                  "cpu_predicted": cpu[i]["predicted"], "gpu_predicted": gpu[i]["predicted"],
                  "cpu_correct": bool(cpu[i]["correct"]), "gpu_correct": bool(gpu[i]["correct"])}
                 for i in shared if cpu[i]["predicted"] != gpu[i]["predicted"]]
        tolerated = [f for f in flips if f["cpu_margin"] <= margin]
        failing = [f for f in flips if f["cpu_margin"] > margin]
        same = len(shared) - len(flips)
        acc_c = float(np.mean([bool(cpu[i]["correct"]) for i in shared]))
        acc_g = float(np.mean([bool(gpu[i]["correct"]) for i in shared]))
        pdiff = max((abs(float(cpu[i]["probabilities"][k]) - float(gpu[i]["probabilities"].get(k, 0.0)))
                     for i in shared for k in cpu[i]["probabilities"]), default=0.0)
        from_flips = (sum(f["gpu_correct"] for f in flips) - sum(f["cpu_correct"] for f in flips)) / len(shared)
        strict = same / len(shared) >= PARITY_MIN_SAME_PREDICTION and abs(acc_g - acc_c) <= PARITY_MAX_ACCURACY_DIFF
        rows.append({"condition_id": cid, "gpu_condition_id": twin, "n_items": len(shared),
                     "same_prediction_share": same / len(shared), "accuracy_cpu": acc_c, "accuracy_gpu": acc_g,
                     "accuracy_difference": acc_g - acc_c, "accuracy_difference_from_flips": from_flips,
                     "max_abs_probability_difference": pdiff, "counted": len(shared) >= min_items,
                     "strict_passed": bool(strict), "n_disagreements": len(flips),
                     "tolerated_flips": tolerated, "failing_flips": failing, "passed": not failing})
    counted = [r for r in rows if r["counted"]]
    return {"criteria": {"min_same_prediction_share": PARITY_MIN_SAME_PREDICTION,
                         "max_accuracy_difference": PARITY_MAX_ACCURACY_DIFF, "near_tie_margin": margin, "min_items": min_items,
                         "rule": "pass = no CPU/GPU disagreement with a CPU top-two margin above near_tie_margin; "
                                 "strict_passed keeps the original 98% / 0.5-point rule"},
            "labels": {"cpu": cpu_label, "gpu": gpu_label},
            "rows": rows, "passed": (all(r["passed"] for r in counted) if counted else None),
            "strict_passed": (all(r["strict_passed"] for r in counted) if counted else None)}


def _cell_status(counts: Dict[str, int]) -> str:
    total = sum(counts.values())
    if total == 0:
        return "missing"
    if counts.get("measured", 0) == total:
        return "measured"
    if counts.get("unsupported", 0) == total:
        return "unsupported"
    if counts.get("failed", 0) == total:
        return "failed"
    return "partial"


def build_summary(run_path: Path, split: str = "dev", expected: Optional[Sequence[str]] = None,
                  n_boot: int = 5000, seed: int = 0, parity_margin: float = PARITY_NEAR_TIE_MARGIN) -> Dict[str, Any]:
    """Summaries of every condition's results on `split`, with paired comparisons and a cell table."""
    run_path = Path(run_path)
    try:
        temps_all = read_json(run_path, "calibration.json").get("conditions", {})
    except (OSError, ValueError):
        temps_all = {}
    recs = {cid: [r for r in read_jsonl(quality_dir(run_path) / cid / "predictions.jsonl") if r.get("split") == split]
            for cid in condition_ids(run_path)}
    conditions: Dict[str, Any] = {}
    cells: List[Dict[str, Any]] = []
    for cid, rows in recs.items():
        info = parse_condition_id(cid)
        s = summarize_records(rows)
        temps = temps_all.get(cid, {})
        s["ece_scaled"] = scaled_ece(rows, temps) if temps else None
        s["temperatures"] = temps or None
        if info["name"] == WINDOW_NAME:
            s["ece_scaled_note"] = "non-comparable: windowed probabilities are the deciding window's, not calibrated for the whole document"
        s["condition"] = info
        s["paired"] = {}
        if info["variant"] == "none" or info["name"] == "native":
            for ref in REFERENCE_CONDITIONS:
                if ref == info["name"]:
                    continue
                ref_cid = ".".join([ref] + ([info["params"]] if info["params"] and ref != WINDOW_NAME and "ckpt" in info["params"] else [])
                                   + ["none", info["device"], "L%s" % info["length"]])
                if ref_cid in recs and info["variant"] == "none":
                    p = paired(rows, recs[ref_cid], n_boot=n_boot, seed=seed)
                    if p:
                        s["paired"][ref_cid] = p
        conditions[cid] = s
        cells.append({"condition_id": cid, **{k: info[k] for k in ("name", "params", "variant", "device", "length")},
                      "status": _cell_status(s["counts_by_status"]), "n_items": s["n_items"],
                      "n_measured": s["n_measured"], "accuracy": s["accuracy"]["overall"],
                      "ece_raw": s["ece_raw"]["overall"],
                      "ece_scaled": (s["ece_scaled"] or {}).get("overall"),
                      "mean_abs_level_error": s["mean_abs_level_error"]})
    for cid in expected or []:
        if cid not in conditions:
            info = parse_condition_id(cid)
            cells.append({"condition_id": cid, **{k: info[k] for k in ("name", "params", "variant", "device", "length")},
                          "status": "missing", "n_items": 0, "n_measured": 0, "accuracy": None, "ece_raw": None,
                          "ece_scaled": None, "mean_abs_level_error": None})
    body = {"split": split, "conditions": conditions, "cells": sorted(cells, key=lambda c: c["condition_id"]),
            "retrieval_budget": select_retrieval_budget(cells), "parity": parity(run_path, split, margin=parity_margin)}
    write_json(run_path, "summary.json", body)
    return body


def select_retrieval_budget(cells: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The retrieval budget with the best dev accuracy over the lengths it was measured at (ties: the smaller).

    Only cells of the device that scored the headline quality count (GPU when any architectural cell was
    GPU-scored, as in the report); the CPU parity subset covers one budget on a smaller sample and would
    bias the comparison.
    """
    device = "gpu" if any(c["device"] == "gpu" and c["variant"] == "none" for c in cells) else "cpu"
    by_budget: Dict[int, List[float]] = {}
    for c in cells:
        if (c["name"].startswith("retrieve") and c["variant"] == "none" and c["device"] == device
                and c["accuracy"] is not None):
            by_budget.setdefault(int(c["name"][len("retrieve"):]), []).append(c["accuracy"])
    if not by_budget:
        return {"selected": None, "tuned_on": "dev", "mean_accuracy": {}}
    mean = {b: float(np.mean(v)) for b, v in by_budget.items()}
    best = min(mean, key=lambda b: (-mean[b], b))
    return {"selected": best, "tuned_on": "dev", "mean_accuracy": {str(b): mean[b] for b in sorted(mean)}}
