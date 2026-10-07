"""The frozen final-test evaluation plan and its power statement (T054).

Specs: specs/002-decision-benchmark-baselines (research.md R16; FR-013, FR-030; SC-008).

The plan fixes the final-test metrics, the quality margin, the comparisons, the decision rule and the
required number of cases *before* any final-test item is scored. It is derived from the pilot: the
paired case-level differences on the dev results. When the final split (200 cases) cannot resolve the
margin, the plan says so and gives the number of cases it would take; it never lowers the bar.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import metrics, summary
from .results import Refusal, SplitLocked, read_jsonl, write_json

MARGIN_PP = 2.0
POWER_TARGET = 0.8
Z_LOWER = 1.96          # 95% interval: lower bound is mean - 1.96 * se
Z_POWER = 0.8416        # z for 80% power
N_SIM = 4000
GRID_STEP = 10
GRID_MAX = 4000
BASELINES = ("trunc512", "truncCap", "window", "retrieve512", "retrieve1024", "retrieve2048")


def count_final_items(run_path: Path) -> int:
    """Items of the final split named in any result file under the run directory (must be 0, SC-008)."""
    n = 0
    for sub in ("quality", "tuning", "gpu_quality"):
        d = Path(run_path) / sub
        if d.exists():
            for f in d.glob("*/predictions.jsonl"):
                n += sum(1 for r in read_jsonl(f) if r.get("split") == "final")
    return n


def case_differences(a: Sequence[Dict[str, Any]], b: Sequence[Dict[str, Any]]) -> np.ndarray:
    """Case-level mean of the paired correctness difference (a minus b) on items both measured."""
    ref = {r["item_id"]: r for r in b if r["status"] == "measured"}
    by_case: Dict[str, List[float]] = {}
    for r in a:
        if r["status"] == "measured" and r["item_id"] in ref:
            by_case.setdefault(r["case_id"], []).append(float(r["correct"]) - float(ref[r["item_id"]]["correct"]))
    return np.array([float(np.mean(v)) for v in by_case.values()])


def intra_case_correlation(item_diffs_by_case: Sequence[Sequence[float]]) -> Optional[float]:
    """One-way ICC of the item-level paired differences within a case (questions of a case are dependent)."""
    groups = [np.asarray(g, dtype=float) for g in item_diffs_by_case if len(g) > 1]
    if len(groups) < 2:
        return None
    k = float(np.mean([len(g) for g in groups]))
    grand = np.concatenate(groups).mean()
    msb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (len(groups) - 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / max(1, sum(len(g) - 1 for g in groups))
    denom = msb + (k - 1) * msw
    return float((msb - msw) / denom) if denom > 0 else None


def required_cases_closed_form(sd_case: float, delta: float = 0.0, margin: float = MARGIN_PP / 100.0) -> Optional[int]:
    """Cases needed so the 95% lower bound of the paired difference clears -margin with 80% power.

    ``n = (z_lower + z_power)^2 * sd^2 / (delta + margin)^2``; undefined when the true difference is already
    below -margin.
    """
    gap = delta + margin
    if gap <= 0:
        return None
    return int(math.ceil(((Z_LOWER + Z_POWER) ** 2) * sd_case ** 2 / gap ** 2))


def simulated_power(d_cases: np.ndarray, n: int, delta: float, margin: float, seed: int, n_sim: int = N_SIM) -> float:
    """Share of simulated final splits of `n` cases whose 95% lower bound clears -margin.

    Cases are drawn with replacement from the pilot's case-level differences, shifted to a true mean `delta`.
    """
    rng = np.random.default_rng(seed)
    d = d_cases - d_cases.mean() + delta
    draws = d[rng.integers(0, len(d), size=(n_sim, n))]
    lower = draws.mean(axis=1) - Z_LOWER * draws.std(axis=1, ddof=1) / math.sqrt(n)
    return float((lower > -margin).mean())


def required_cases(d_cases: np.ndarray, available: int, delta: float = 0.0, margin: float = MARGIN_PP / 100.0,
                   seed: int = 0) -> Dict[str, Any]:
    """Required final cases from a seeded case-clustered simulation, with the closed form as a cross-check."""
    if len(d_cases) < 5:
        return {"required_cases": None, "power_at_available": None, "resolvable": False,
                "reason": "the pilot has fewer than 5 cases with paired results"}
    sd = float(d_cases.std(ddof=1))
    closed = required_cases_closed_form(sd, delta, margin)
    power_avail = simulated_power(d_cases, available, delta, margin, seed)
    # Power rises with n, so bisect over the grid of sample sizes instead of scanning it.
    lo, hi = 1, GRID_MAX // GRID_STEP
    need = None
    if simulated_power(d_cases, hi * GRID_STEP, delta, margin, seed + hi, n_sim=800) >= POWER_TARGET:
        while lo < hi:
            mid = (lo + hi) // 2
            if simulated_power(d_cases, mid * GRID_STEP, delta, margin, seed + mid, n_sim=800) >= POWER_TARGET:
                hi = mid
            else:
                lo = mid + 1
        need = lo * GRID_STEP
    return {"sd_case": sd, "delta_assumed": delta, "margin": margin, "required_cases": need,
            "required_cases_closed_form": closed, "power_at_available": power_avail,
            "resolvable": bool(power_avail >= POWER_TARGET), "available_cases": available}


def pilot(run_path: Path, available: int, seed: int = 0) -> Dict[str, Any]:
    """Pilot comparisons: every baseline against native at the matching length, from the dev results."""
    run_path = Path(run_path)
    recs = {cid: [r for r in read_jsonl(summary.quality_dir(run_path) / cid / "predictions.jsonl") if r.get("split") == "dev"]
            for cid in summary.condition_ids(run_path)}
    out: List[Dict[str, Any]] = []
    # The device that scored the headline quality (as in the report); the CPU cells of a GPU-scored run are only a parity subset.
    infos = {cid: summary.parse_condition_id(cid) for cid in recs}
    qdev = "gpu" if any(i["device"] == "gpu" and i["variant"] == "none" for i in infos.values()) else "cpu"
    for cid, rows in recs.items():
        info = infos[cid]
        if info["name"] not in BASELINES or info["variant"] != "none" or info["device"] != qdev:
            continue
        ref_cid = "native.none.%s.L%s" % (qdev, info["length"])
        if ref_cid not in recs:
            continue
        d = case_differences(rows, recs[ref_cid])
        if len(d) < 5:
            continue
        by_case: Dict[str, List[float]] = {}
        refmap = {r["item_id"]: r for r in recs[ref_cid] if r["status"] == "measured"}
        for r in rows:
            if r["status"] == "measured" and r["item_id"] in refmap:
                by_case.setdefault(r["case_id"], []).append(float(r["correct"]) - float(refmap[r["item_id"]]["correct"]))
        mean = float(d.mean())
        delta = min(0.0, mean) if mean >= -MARGIN_PP / 100.0 else mean
        out.append({"comparison": "%s minus native at %s tokens" % (info["name"], info["length"]),
                    "condition_id": cid, "reference": ref_cid, "n_cases": int(len(d)), "mean_difference": mean,
                    "sd_case": float(d.std(ddof=1)), "icc": intra_case_correlation(list(by_case.values())),
                    "sizing": required_cases(d, available, delta=0.0 if mean >= -MARGIN_PP / 100.0 else mean, seed=seed)})
    return {"comparisons": out}


def plan_body(run_path: Path, available: int = 200, seed: int = 0, version: int = 1, reason: str = "") -> Dict[str, Any]:
    p = pilot(run_path, available, seed)
    sized = [c["sizing"] for c in p["comparisons"] if c["sizing"].get("required_cases")]
    worst = max(sized, key=lambda s: s["required_cases"]) if sized else None
    any_unresolved = [c for c in p["comparisons"] if not c["sizing"].get("resolvable")]
    resolvable = bool(p["comparisons"]) and not any_unresolved
    if not p["comparisons"]:
        statement = ("There are no dev pilot comparisons yet, so the required sample size cannot be computed. "
                     "Run the baselines on dev before freezing the plan.")
    elif resolvable:
        statement = ("The %d-case final split can resolve the %.0f-point margin for every pilot comparison "
                     "(simulated power at least %.0f%%)." % (available, MARGIN_PP, 100 * POWER_TARGET))
    else:
        need = worst["required_cases"] if worst else None
        statement = ("The %d-case upstream final split cannot resolve the %.0f-point margin for %d of %d pilot "
                     "comparisons at %.0f%% power%s. The bar is not lowered: those comparisons will be reported as "
                     "inconclusive unless more cases are added." % (
                         available, MARGIN_PP, len(any_unresolved), len(p["comparisons"]), 100 * POWER_TARGET,
                         "; the most demanding needs about %d cases" % need if need else ""))
    body = {
        "version": version, "version_reason": reason or "initial plan", "margin_pp": MARGIN_PP,
        "metrics": {"primary": "exact-match accuracy per question (predicted label equals the gold label)",
                    "secondary": ["classification calibration error from the predicted answer's probability (raw and "
                                  "temperature-scaled with calibration-split temperatures)",
                                  "mean absolute level error for ordinal questions", "CPU p50 and p95 latency, peak memory"]},
        "comparisons": {"pairs": "each baseline against native Laya at the matching length, on identical final-split items",
                        "baselines": list(BASELINES), "lengths": [512, 1024, 2048, 4096, 8192],
                        "interval": "case-clustered percentile bootstrap, 5000 resamples, 95%",
                        "verdict_rule": "better if lower bound > 0; worse if upper bound < 0; equal if the interval lies "
                                        "within +/- %.0f points; otherwise inconclusive" % MARGIN_PP,
                        "within_margin_rule": "a baseline reaches native quality when the lower bound of (baseline - native) "
                                              "is above -%.0f points, and it costs less CPU p50 latency" % MARGIN_PP},
        "pilot": p, "power_target": POWER_TARGET, "available_cases": available,
        "required_cases": worst["required_cases"] if worst else None,
        "power": {"simulation": "seeded case-clustered resampling of the pilot's case-level differences (%d draws)" % N_SIM,
                  "worst_case_sizing": worst},
        "resolvable": resolvable, "statement": statement,
        "final_scored_items": count_final_items(run_path),
    }
    return body


def freeze_plan(run_path: Path, available: int = 200, seed: int = 0, new_version: bool = False, reason: str = "") -> Dict[str, Any]:
    """Write ``evaluation_plan.json`` and ``.md``. Refuses when a final item was scored or a frozen plan exists."""
    run_path = Path(run_path)
    n_final = count_final_items(run_path)
    if n_final:
        raise SplitLocked("%d final-split items appear in the results; the plan must be frozen before any final scoring (FR-030)" % n_final)
    existing = run_path / "evaluation_plan.json"
    version = 1
    if existing.exists():
        old = json.loads(existing.read_text(encoding="utf-8"))
        if not new_version:
            raise Refusal("fingerprint_mismatch", "evaluation_plan.json is frozen (version %s); pass --new-version with a "
                                                  "reason to create version %s" % (old.get("version"), old.get("version", 1) + 1))
        if not reason.strip():
            raise ValueError("--new-version needs a stated reason")
        version = int(old.get("version", 1)) + 1
    body = plan_body(run_path, available, seed, version, reason)
    canonical = json.dumps(body, sort_keys=True, default=str)
    body["fingerprint"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    body["frozen_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    write_json(run_path, "evaluation_plan.json", body)
    (run_path / "evaluation_plan.md").write_text(render_plan(body), encoding="utf-8")
    return body


def render_plan(b: Dict[str, Any]) -> str:
    lines = ["# Frozen evaluation plan (version %s)" % b["version"], "",
             "Frozen %s, fingerprint `%s`. Final-split items scored so far: **%d**." % (b["frozen_at"], b["fingerprint"][:16], b["final_scored_items"]),
             "", "- [estimated] %s" % b["statement"], "", "## Metrics", "- primary: %s" % b["metrics"]["primary"]]
    lines += ["- secondary: %s" % m for m in b["metrics"]["secondary"]]
    c = b["comparisons"]
    lines += ["", "## Comparisons", "- %s" % c["pairs"], "- interval: %s" % c["interval"], "- verdicts: %s" % c["verdict_rule"],
              "- within margin: %s" % c["within_margin_rule"], "", "## Sample size",
              "- available final cases: %d; required for the most demanding comparison: %s" % (b["available_cases"], b["required_cases"]),
              "- margin: %.0f percentage points; power target %.0f%%" % (b["margin_pp"], 100 * b["power_target"]), "",
              "## Pilot comparisons (dev)"]
    for p in b["pilot"]["comparisons"]:
        s = p["sizing"]
        lines.append("- %s: %d cases, mean difference %+.3f, case sd %.3f, ICC %s, required %s, power at %d cases %s, resolvable %s"
                     % (p["comparison"], p["n_cases"], p["mean_difference"], p["sd_case"],
                        "n/a" if p["icc"] is None else "%.2f" % p["icc"], s.get("required_cases"), b["available_cases"],
                        "n/a" if s.get("power_at_available") is None else "%.2f" % s["power_at_available"], s.get("resolvable")))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- plan v2 (compression gate, laya:008)

PLAN_V2_REASON = "compression gate tiers E/F/Q"
PLAN_V2_FILE = "evaluation_plan_v2.json"

#: The constants of docs/gate-plan-v2.md (sections 2 to 6). The JSON is built from these, never parsed from the markdown;
#: the markdown is tied to the JSON only by its sha256.
PLAN_V2 = {
    "lengths": [512, 1024, 2048, 4096, 8192],
    "tiers": {
        "E": {"E1": "per local layer and end to end, probabilities within 1e-5 absolute of native fp32 on CPU at 512, 1024, 2048, 4096, 8192",
              "E2": "100% identical predicted answers on the CPU parity subset (20 dev cases, distractor@mid, 512/2048/8192; 95/100/100 items)",
              "E3": "i7-8700 p50 of optimized native not slower than fastpath_off native at any length (reported, not gated beyond that)"},
        "F": {"F1": {"metric": "top-label agreement with optimized native, same device and dtype", "lower_bound_min": 0.95,
                     "interval": "case-clustered percentile bootstrap, 95%"},
              "F2": {"metric": "accuracy difference (candidate - native)", "lower_bound_min_points": -2.0},
              "F3": {"metric": "temperature-scaled classification ECE above native's", "max_excess": 0.02,
                     "note": "temperatures fitted on the calibration split, one per condition; cannot rescue a failed F1 or F2"},
              "F4": {"metric": "i7-8700 p50 below optimized-native p50 at the same length", "min_gain": 0.20, "p95": "not worse",
                     "note": "a CPU implementation that really skips work; a mask-only prototype is a quality diagnostic"}},
        "Q": {"status": "defined, not funded (S2 deferred)", "margin_points": 5.0, "claim_points": 2.0},
    },
    "bootstrap": {"resamples": 5000, "interval": 0.95, "unit": "case"},
    "candidate_verdict": "passes at a length if F1, F2, F3 and F4 hold; Raya support only if it passes at 4096 and 8192 and does not "
                         "fail F1 or F2 at 512, 1024, 2048 or original length",
    "candidates": {
        "A1": {"change": "the 10 global-attention layers become block-local (block 128, three-block span)",
               "global_tokens": ["[CLS]", "question tokens (between [CLS] and the first [SEP])", "option [MASK] markers"],
               "not_global": ["option description tokens", "[SEP] tokens", "state tokens"], "block": 128,
               "variants": {"cpu": "a1", "mask_only": "a1_mask"}},
        "B1": {"change": "decision-head self-attention keys restricted to the same global tokens plus a +/-128 window around each",
               "window": 128, "variants": {"cpu": "b1", "mask_only": "b1_mask"}},
        "rescue": {"A2": {"block": 256}, "B2": {"window": 256},
                   "rule": "only if not dropped and the primary's dev agreement point estimate is in [90%, 95%) at 4K or at 8K; "
                           "a point estimate below 90% at any of 2K, 4K, 8K drops the candidate (dropping wins)"},
    },
    "comparisons": ["C-E", "C-A", "C-B", "C-lat", "C-par", "C-ref"],
    "parity": {"margin": 0.01, "margin_rule": "a CPU/GPU disagreement counts only if the CPU top-two margin exceeds 0.01",
               "seed": 2026100801, "cases_per_set": 20, "sets": 2, "item_variant": "distractor@mid",
               "lengths": [512, 2048, 4096, 8192], "min_counted_items_per_cell": 20},
    "dtypes": {"quality": "GPU: fp32 at 512/1024/2048, fp16 autocast at 4096/8192", "latency": "i7-8700 fp32, 6 threads"},
    "final_split": {"full_lengths": [4096, 8192], "cases_at_full_lengths": 200,
                    "rule": "all 200 final cases at 4,096 and 8,192 tokens; dev and calibration keep the recorded half-sample there"},
    "stop_conditions": ["training-free stop", "candidate stop (dropping wins over rescue)", "speed stop", "parity stop",
                        "undecidable", "selection wins", "no change after any final item is scored"],
}


def _file_sha256(path: Path) -> str:
    """sha256 of the file's exact bytes (docs/gate-plan-v2.md is marked ``-text`` in .gitattributes, so no checkout changes them)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan_v2_body(run_path: Path, plan_md: Optional[Path] = None, parity_ids: Optional[Path] = None,
                 revision: int = 0, reason: str = "") -> Dict[str, Any]:
    body: Dict[str, Any] = {"version": 2, "revision": revision, "version_reason": PLAN_V2_REASON,
                            "revision_reason": reason or "pre-registration emission",
                            "supersedes": "none: plan v1 (evaluation_plan.json, fingerprint 58ea2c38f100d4f6) stays as recorded",
                            "plan": PLAN_V2, "final_scored_items": count_final_items(run_path)}
    if plan_md is not None:
        body["plan_markdown"] = {"path": Path(plan_md).as_posix(), "sha256": _file_sha256(plan_md)}
    if parity_ids is not None:
        ids = json.loads(Path(parity_ids).read_text(encoding="utf-8"))
        if ids.get("seed") != PLAN_V2["parity"]["seed"]:
            raise ValueError("the parity ids were drawn with seed %s, the plan fixes %s" % (ids.get("seed"), PLAN_V2["parity"]["seed"]))
        body["parity_ids"] = {"path": Path(parity_ids).as_posix(), "ids_sha256": ids["ids_sha256"],
                              "dev_split_fingerprint": ids["dev_split_fingerprint"]}
    return body


def freeze_plan_v2(run_path: Path, plan_md: Optional[Path] = None, parity_ids: Optional[Path] = None,
                   new_version: bool = False, reason: str = "") -> Dict[str, Any]:
    """Write ``evaluation_plan_v2.json`` and ``.md``. Never touches the v1 file; refuses when a final item was scored,
    or when a v2 file exists unless ``new_version`` and a reason are given (the revision then increases)."""
    run_path = Path(run_path)
    n_final = count_final_items(run_path)
    if n_final:
        raise SplitLocked("%d final-split items appear in the results; the plan must be frozen before any final scoring (FR-030)" % n_final)
    existing = run_path / PLAN_V2_FILE
    revision = 0
    if existing.exists():
        old = json.loads(existing.read_text(encoding="utf-8"))
        if not new_version:
            raise Refusal("fingerprint_mismatch", "%s exists (revision %s); pass --new-version with a reason to create revision %s"
                          % (PLAN_V2_FILE, old.get("revision"), old.get("revision", 0) + 1))
        if not reason.strip():
            raise ValueError("--new-version needs a stated reason")
        revision = int(old.get("revision", 0)) + 1
    body = plan_v2_body(run_path, plan_md, parity_ids, revision, reason)
    body["fingerprint"] = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    body["frozen_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    write_json(run_path, PLAN_V2_FILE, body)
    (run_path / "evaluation_plan_v2.md").write_text(render_plan_v2(body), encoding="utf-8")
    return body


def render_plan_v2(b: Dict[str, Any]) -> str:
    p = b["plan"]
    lines = ["# Compression gate plan v2 (revision %s)" % b["revision"], "",
             "Fingerprint `%s`, frozen %s. Final-split items scored so far: **%d**." % (b["fingerprint"][:16], b["frozen_at"], b["final_scored_items"]),
             "", "- reason: %s (%s)" % (b["version_reason"], b["revision_reason"])]
    if "plan_markdown" in b:
        lines.append("- plan text: `%s`, sha256 `%s`" % (b["plan_markdown"]["path"], b["plan_markdown"]["sha256"]))
    if "parity_ids" in b:
        lines.append("- parity ids: `%s`, sha256 `%s`" % (b["parity_ids"]["path"], b["parity_ids"]["ids_sha256"]))
    f = p["tiers"]["F"]
    lines += ["", "## Tier F thresholds",
              "- F1: agreement lower bound >= %.0f%%" % (100 * f["F1"]["lower_bound_min"]),
              "- F2: (candidate - native) accuracy lower bound >= %+.0f points" % f["F2"]["lower_bound_min_points"],
              "- F3: scaled ECE at most %.2f above native" % f["F3"]["max_excess"],
              "- F4: i7-8700 p50 at least %.0f%% below optimized native, p95 not worse" % (100 * f["F4"]["min_gain"]),
              "", "## Parity", "- margin %.2f, seed %d, %d cases per set, lengths %s" % (
                  p["parity"]["margin"], p["parity"]["seed"], p["parity"]["cases_per_set"], p["parity"]["lengths"]),
              "", "## Final split", "- %s" % p["final_split"]["rule"], ""]
    return "\n".join(lines)
