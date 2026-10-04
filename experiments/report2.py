"""The Phase 2 report: quality-against-cost curves, verdicts, the Phase 3 evidence table and limits (T055, T056).

Specs: specs/002-decision-benchmark-baselines (FR-026 to FR-030; SC-002, SC-007, SC-009).

Everything is read from result files; nothing is recomputed from memory. Each statement carries a label,
``[measured]``, ``[estimated]`` or ``[hypothesized]``, and the files it comes from. CPU tables read only
``latency/``; GPU numbers appear only in the GPU-labeled table, read from ``gpu_latency/``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import data, evalplan, summary
from .results import read_json, write_json

LABELS = ("measured", "estimated", "hypothesized")
LENGTHS = (512, 1024, 2048, 4096, 8192)
VERDICT_LENGTHS = (2048, 4096, 8192)
MARGIN = 0.02
BASELINES = ("trunc512", "truncCap", "window", "retrieve512", "retrieve1024", "retrieve2048")


class ReportError(RuntimeError):
    """The report cannot be assembled from this run directory."""


def _try(fn, *a, **k):
    try:
        return fn(*a, **k)
    except (OSError, ValueError, KeyError, FileNotFoundError):
        return None


def stmt(label: str, text: str, *sources: str) -> Dict[str, Any]:
    if label not in LABELS:
        raise ValueError("label must be one of %s" % (LABELS,))
    return {"label": label, "text": text, "sources": [s for s in sources if s]}


def pct(x: Optional[float], digits: int = 1) -> str:
    return "n/a" if x is None else "%.*f%%" % (digits, 100 * x)


def pts(x: Optional[float]) -> str:
    return "n/a" if x is None else "%+.1f points" % (100 * x)


def _latency(run_path: Path, cid: str, device: str = "cpu") -> Optional[Dict[str, Any]]:
    sub = "gpu_latency" if device == "gpu" else "latency"
    cid = cid.replace(".gpu.", ".cpu.").replace(".cpu.", ".%s." % device)     # a quality cell may be GPU-scored
    body = _try(read_json, Path(run_path) / sub, "%s.json" % cid)
    return body if body and body.get("timings_ms") else None


def _p50(run_path: Path, cid: str, device: str = "cpu") -> Optional[float]:
    b = _latency(run_path, cid, device)
    return b["timings_ms"]["p50"] if b else None


# --------------------------------------------------------------------------- sections

def section_data(run_path: Path, root: Optional[Path]) -> Dict[str, Any]:
    man = _try(data.read_data_json, "manifest.json", root)
    prof = _try(data.read_data_json, "length_profile.json", root)
    splits = _try(data.read_data_json, "splits.json", root)
    out: Dict[str, Any] = {"statements": [], "length_share": {}}
    if not man:
        out["statements"].append(stmt("measured", "The upstream dataset has not been imported (no data manifest)."))
        return out
    test = man["splits"]["test"]
    out["statements"].append(stmt(
        "measured", "Upstream %s at %s (%s): %d train and %d test cases; the test split has %d questions %s; fingerprint %s."
        % (man["source"], man["revision"][:12], man["license"], man["splits"]["train"]["cases"], test["cases"],
           test["questions"], test["questions_by_type"], man["fingerprint"]["combined"][:16]), "data/manifest.json"))
    lc = man["low_confidence"]
    out["statements"].append(stmt("measured", "Low-confidence labels are the bottom quartile of train label confidence (cutoff %.4f, %d train questions)."
                                  % (lc["cutoff"], lc["n_questions"]), "data/manifest.json"))
    if prof:
        t = prof["by_split"]["test"]
        shares = t["overall"]["cases"]["fit_share"]
        out["length_share"]["overall"] = shares
        r = t["overall"]["rows"]
        out["statements"].append(stmt(
            "measured", "Original test rows under Laya's tokenizer: min %d, median %d, p90 %d, p99 %d, max %d tokens; the share of test cases "
            "that fit within 512 / 1,024 / 2,048 / 4,096 / 8,192 tokens is %s."
            % (r["min"], r["median"], r["p90"], r["p99"], r["max"], " / ".join(pct(shares[str(n)]) for n in LENGTHS)),
            "data/length_profile.json"))
        for wf, v in t["by_workflow"].items():
            sh = v["cases"]["fit_share"]
            out["length_share"][wf] = sh
            out["statements"].append(stmt("measured", "%s: median row %d, max %d tokens; %s of cases fit within 512, %s within 1,024."
                                          % (wf, v["rows"]["median"], v["rows"]["max"], pct(sh["512"]), pct(sh["1024"])),
                                          "data/length_profile.json"))
        if t["over_head_budget"]:
            out["statements"].append(stmt("measured", "%d test questions have options that overflow the head budget and are reported apart."
                                          % len(t["over_head_budget"]), "data/length_profile.json"))
    if splits:
        n = {s: splits["splits"][s]["n_cases"] for s in data.EVAL_SPLITS}
        out["statements"].append(stmt(
            "measured", "Splits (seed %s, stratified by workflow): development %d, calibration %d, final %d cases. Lengths 512 to 2,048 run on every "
            "case of a split; 4,096 and 8,192 run on a recorded half-sample (development %d cases)."
            % (splits["seed"], n["dev"], n["calibration"], n["final"], len(splits["half_sample"]["dev"])), "data/splits.json"))
    return out


def section_solvability(run_path: Path) -> Dict[str, Any]:
    sol = _try(read_json, run_path, "solvability.json")
    out: Dict[str, Any] = {"statements": [], "reference": None}
    if not sol:
        out["statements"].append(stmt("measured", "The solvability check has not been run; there is no validated quality reference."))
        return out
    out["reference"] = sol["reference_checkpoint"]
    for name, m in sol["models"].items():
        if m.get("status") == "failed":
            out["statements"].append(stmt("measured", "%s failed to run: %s" % (name, m.get("reason")), "solvability.json"))
            continue
        acc = m["summary"]["accuracy"]["overall"]
        pv = m["paired_vs_majority"]
        out["statements"].append(stmt(
            "measured", "%s on original-length dev rows: accuracy %s over %d items (per-question majority class %s; paired difference %s, 95%% [%s, %s]); solves the short cases: %s."
            % (name, pct(acc), m["summary"]["accuracy"]["n"], pct(m["majority_accuracy"]), pts(pv["diff"]), pts(pv["lo"]), pts(pv["hi"]),
               "yes" if m["solves"] else "no"), "solvability.json"))
    if sol["reference_checkpoint"] == "none":
        out["statements"].append(stmt("measured", "No checkpoint solves the original-length development cases, so every later quality comparison lacks a valid reference (FR-006).", "solvability.json"))
    else:
        out["statements"].append(stmt("measured", "The quality reference is the %s checkpoint." % sol["reference_checkpoint"], "solvability.json"))
    return out


def section_audit(root: Optional[Path]) -> Dict[str, Any]:
    a = _try(data.read_data_json, "audit_result.json", root)
    if not a:
        return {"statements": [stmt("measured", "The hand audit has not been recorded (no audit_result.json); baseline comparisons are gated on it.")]}
    auditor = a.get("auditor", "researcher")
    out = [stmt(
        "measured", "Audit of %d items on families %s by %s: %s answer changes, %s ambiguous, evidence intact %s; passed: %s."
        % (a["n_audited"], a["families_id"], auditor, pct(a["share_answer_changed"]), pct(a["share_ambiguous"]), a["all_evidence_intact"], a["passed"]),
        "data/audit_result.json")]
    if auditor != "researcher":
        out.append(stmt("measured", "The audit (FR-014 asks for a hand audit) was performed by %s at the researcher's request, not by a human reviewer%s. "
                                    "It is weaker evidence than a human audit and should be repeated by a person before the result is relied on."
                        % (auditor, ": " + a["method"] if a.get("method") else ""), "data/audit_result.json"))
    return {"statements": out}


def quality_device(summ: Optional[Dict[str, Any]]) -> str:
    """The device that scored the headline quality: ``gpu`` when any architectural cell was GPU-scored, else ``cpu``."""
    cells = (summ or {}).get("cells") or []
    return "gpu" if any(c["device"] == "gpu" and c["variant"] == "none" for c in cells) else "cpu"


def section_curves(run_path: Path, summ: Optional[Dict[str, Any]], qdev: str = "cpu") -> Dict[str, Any]:
    out: Dict[str, Any] = {"rows": [], "curves": {"cpu": {}, "gpu": {}}, "statements": [], "quality_device": qdev}
    if not summ:
        out["statements"].append(stmt("measured", "No evaluation summary yet (run eval-summary)."))
        return out
    out["statements"].append(stmt("measured", "Quality (accuracy and calibration) in this section was scored on the %s; latency and memory are CPU "
                                              "measurements, and GPU latency is in its own table." % qdev.upper(), "summary.json"))
    for c in summ["cells"]:
        if c["variant"] != "none" or c["device"] != qdev:
            continue
        cpu = _latency(run_path, c["condition_id"], "cpu")
        gpu = _latency(run_path, c["condition_id"], "gpu")
        row = {"condition_id": c["condition_id"], "name": c["name"], "length": c["length"], "status": c["status"],
               "n_measured": c["n_measured"], "n_items": c["n_items"], "accuracy": c["accuracy"], "ece_raw": c["ece_raw"],
               "ece_scaled": c["ece_scaled"], "cpu_p50_ms": cpu["timings_ms"]["p50"] if cpu else None,
               "cpu_p95_ms": cpu["timings_ms"]["p95"] if cpu else None,
               "cpu_peak_rss_bytes": cpu.get("peak_rss_bytes") if cpu else None,
               "gpu_p50_ms": gpu["timings_ms"]["p50"] if gpu else None, "deployable": c["name"] != "oracle",
               "quality_device": qdev,
               "sources": ["summary.json"] + (["latency/%s.json" % c["condition_id"].replace(".gpu.", ".cpu.")] if cpu else [])}
        out["rows"].append(row)
        pt = {"length": c["length"], "accuracy": c["accuracy"], "ece_raw": c["ece_raw"]}
        if cpu:
            out["curves"]["cpu"].setdefault(c["name"], []).append(dict(pt, latency_p50_ms=cpu["timings_ms"]["p50"]))
        if gpu:
            out["curves"]["gpu"].setdefault(c["name"], []).append(dict(pt, latency_p50_ms=gpu["timings_ms"]["p50"]))
    missing = [r["condition_id"] for r in out["rows"] if r["status"] in ("missing", "partial", "failed")]
    out["statements"].append(stmt("measured", "%d condition-length cells; %d are complete, and %d are partial, failed or missing (each with its reason in summary.json)."
                                  % (len(out["rows"]), sum(1 for r in out["rows"] if r["status"] == "measured"), len(missing)), "summary.json"))
    return out


def section_verdicts(summ: Optional[Dict[str, Any]], qdev: str = "cpu") -> Dict[str, Any]:
    out: Dict[str, Any] = {"statements": [], "table": []}
    for L in VERDICT_LENGTHS:
        native = ((summ or {}).get("conditions") or {}).get("native.none.%s.L%d" % (qdev, L))
        for ref in ("trunc512", "truncCap"):
            p = ((native or {}).get("paired") or {}).get("%s.none.%s.L%d" % (ref, qdev, L))
            label = "truncation to 512 tokens" if ref == "trunc512" else "truncation to the configured cap"
            if not p:
                out["statements"].append(stmt("measured", "At %d tokens the native-versus-%s comparison was not measured." % (L, label)))
                continue
            word = {"better": "better than", "worse": "worse than", "equal": "equal to (within 2 points)", "inconclusive": "inconclusive against"}[p["verdict"]]
            out["table"].append({"length": L, "reference": ref, "difference": p["diff"], "lo": p["lo"], "hi": p["hi"], "verdict": p["verdict"], "n_cases": p["n_cases"]})
            out["statements"].append(stmt(
                "measured", "At %d tokens native Laya is %s %s: difference %s, 95%% case-clustered interval [%s, %s] over %d cases (%d items)."
                % (L, word, label, pts(p["diff"]), pts(p["lo"]), pts(p["hi"]), p["n_cases"], p["n_items"]), "summary.json"))
    return out


def section_margin(run_path: Path, summ: Optional[Dict[str, Any]], qdev: str = "cpu") -> Dict[str, Any]:
    """Does any baseline reach native quality within -2 points at lower CPU cost (FR-028)?

    Quality comes from the scoring device `qdev`; the cost compared is always CPU p50 latency."""
    out: Dict[str, Any] = {"statements": [], "qualifying": [], "inconclusive": []}
    conds = (summ or {}).get("conditions") or {}
    for cid, c in conds.items():
        info = c["condition"]
        if info["name"] not in BASELINES or info["variant"] != "none" or info["device"] != qdev:
            continue
        ref = "native.none.%s.L%s" % (qdev, info["length"])
        p = c["paired"].get(ref)
        if not p:
            continue
        cost, ref_cost = _p50(run_path, cid), _p50(run_path, ref)
        entry = {"condition_id": cid, "length": info["length"], "difference": p["diff"], "lo": p["lo"],
                 "cpu_p50_ms": cost, "native_cpu_p50_ms": ref_cost}
        if p["lo"] is not None and p["lo"] > -MARGIN and cost is not None and ref_cost is not None and cost < ref_cost:
            out["qualifying"].append(entry)
        elif p["hi"] is not None and p["hi"] > -MARGIN and (p["lo"] is None or p["lo"] <= -MARGIN):
            out["inconclusive"].append(entry)
    if out["qualifying"]:
        names = ["%s (%s vs native, native %.0f ms, baseline %.0f ms)" % (e["condition_id"], pts(e["difference"]), e["native_cpu_p50_ms"], e["cpu_p50_ms"])
                 for e in sorted(out["qualifying"], key=lambda e: (e["length"], e["condition_id"]))]
        out["statements"].append(stmt("measured", "Baselines within -2 points of native at the matching length, at lower CPU p50 latency: %s." % "; ".join(names), "summary.json"))
    else:
        extra = " %d comparison(s) were inconclusive at this sample size." % len(out["inconclusive"]) if out["inconclusive"] else ""
        out["statements"].append(stmt("measured", "No baseline was shown to reach native quality within -2 points at lower CPU cost on this development sample.%s" % extra, "summary.json"))
    return out


def section_variants(run_path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"rows": [], "statements": []}
    cids = summary.condition_ids(run_path)
    for cid in cids:
        info = summary.parse_condition_id(cid)
        if info["variant"] == "none" or info["name"] != "native":
            continue
        recs = [r for r in summary.read_jsonl(summary.quality_dir(run_path) / cid / "predictions.jsonl") if r.get("split") == "dev"]
        ref_cid = "native.none.cpu.L%s" % info["length"]
        if ref_cid not in cids:
            continue
        ref = [r for r in summary.read_jsonl(summary.quality_dir(run_path) / ref_cid / "predictions.jsonl") if r.get("split") == "dev"]
        p = summary.paired(recs, ref, n_boot=2000)
        if not p:
            continue
        ids = {r["item_id"] for r in recs if r["status"] == "measured"}
        shared = [r for r in ref if r["item_id"] in ids and r["status"] == "measured"]
        var = [r for r in recs if r["status"] == "measured"]
        ece_v = summary.metrics.ece([r["prob_predicted"] for r in var], [r["correct"] for r in var]) if var else None
        ece_r = summary.metrics.ece([r["prob_predicted"] for r in shared], [r["correct"] for r in shared]) if shared else None
        agree = sum(1 for r in var for s in shared if r["item_id"] == s["item_id"] and r["predicted"] == s["predicted"])
        cost, ref_cost = _p50(run_path, cid), _p50(run_path, ref_cid)
        row = {"condition_id": cid, "variant": info["variant"], "length": info["length"], "n_items": len(var),
               "accuracy_change": p["diff"], "lo": p["lo"], "hi": p["hi"], "ece_change": None if ece_v is None or ece_r is None else ece_v - ece_r,
               "same_prediction_share": agree / len(var) if var else None, "cpu_p50_ms": cost, "native_cpu_p50_ms": ref_cost}
        out["rows"].append(row)
        out["statements"].append(stmt(
            "measured", "%s at %s tokens (optimized-system variant, %d items): accuracy change against native %s (95%% [%s, %s]); raw calibration error change %s; same predicted answer on %s of items%s."
            % (info["variant"], info["length"], len(var), pts(p["diff"]), pts(p["lo"]), pts(p["hi"]),
               "n/a" if row["ece_change"] is None else "%+.3f" % row["ece_change"], pct(row["same_prediction_share"]),
               "" if cost is None or ref_cost is None else "; CPU p50 %.0f ms against %.0f ms native" % (cost, ref_cost)),
            "quality/%s/predictions.jsonl" % cid))
    if not out["rows"]:
        out["statements"].append(stmt("measured", "No optimized-system variant has results yet."))
    return out


def section_gpu(run_path: Path, summ: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    rows = []
    d = Path(run_path) / "gpu_latency"
    for f in sorted(d.glob("*.json")) if d.exists() else []:
        b = _try(read_json, d, f.name)
        if b and b.get("timings_ms"):
            rows.append({"condition_id": f.stem, "p50_ms": b["timings_ms"]["p50"], "p95_ms": b["timings_ms"]["p95"],
                         "peak_gpu_bytes": b.get("peak_gpu_bytes")})
    st = [stmt("measured", "GPU latency (RTX 4060 class, labeled GPU; never substituted for CPU) exists for %d condition-lengths." % len(rows), "gpu_latency/")]
    return {"rows": rows, "statements": st}


def section_phase3(verdicts: Dict[str, Any], margin: Dict[str, Any], variants: Dict[str, Any], sol: Dict[str, Any]) -> Dict[str, Any]:
    """Which Phase 3 directions the evidence supports, weakens or leaves open (SC-009)."""
    table = verdicts["table"]
    native_better = [t for t in table if t["verdict"] == "better"]
    native_not_better = [t for t in table if t["verdict"] in ("equal", "worse")]
    have = bool(table)
    subs = margin["qualifying"]
    rows = []
    if not have:
        rows.append({"direction": "all", "status": "open", "evidence": "No native-versus-truncation comparison has been measured yet.", "label": "measured"})
    else:
        if native_better:
            long_ctx = "Long context improves quality over truncation at %s tokens." % ", ".join(str(t["length"]) for t in native_better)
        else:
            long_ctx = "Native long context was not shown to beat truncation (%d equal or worse, %d inconclusive)." % (len(native_not_better), len(table) - len(native_not_better))
        sparse = ("supported" if native_better and not subs else "weakened" if subs else "weakened" if native_not_better and not native_better else "open")
        rows.append({"direction": "sparse attention", "status": sparse, "label": "measured",
                     "evidence": long_ctx + (" A simple baseline reaches native within the margin at lower cost, which weakens the case for a new attention pattern." if subs
                                             else " No simple baseline substitutes for native long context in this sample." if native_better else "")})
        sel = ("supported" if subs else "open")
        rows.append({"direction": "selection", "status": sel, "label": "measured",
                     "evidence": ("Baselines within the margin at lower CPU cost: %s." % ", ".join(sorted({e["condition_id"] for e in subs}))) if subs
                     else "No selection or windowing baseline was shown to match native within 2 points at lower cost here."})
        rows.append({"direction": "compression", "status": "open", "label": "hypothesized",
                     "evidence": "Compression was not measured in this phase; nothing here supports or weakens it."})
        rows.append({"direction": "none (truncate)", "status": "supported" if native_not_better and not native_better else "weakened" if native_better else "open",
                     "label": "measured", "evidence": long_ctx})
    if sol.get("reference") == "none":
        for r in rows:
            r["evidence"] += " (No validated quality reference: these rows are provisional.)"
    return {"rows": rows, "statements": [stmt(r["label"], "%s: %s. %s" % (r["direction"], r["status"], r["evidence"]), "summary.json") for r in rows]}


LIMITS = (
    ("measured", "Upstream data is synthetic, English and short (a structured record of a few hundred characters); every long-context conclusion holds for the constructed families only (neutral padding, near-matching distractors, position)."),
    ("measured", "The upstream test split is public and upstream has reported results on it, so it is not a never-inspected confirmatory set."),
    ("measured", "The fine-tuned checkpoint was trained on the upstream training split, which also supplies the distractor records; it may treat them as familiar."),
    ("hypothesized", "Conclusions may not transfer to realistic long documents, which this phase did not include (a documented deviation from constitution principle VII); a realistic long-document set remains a gap."),
    ("measured", "The decoder baseline is deferred: this fork has no decoder scoring path. Multilingual data and training are out of scope."),
    ("measured", "Distractor variants degrade toward padding at short lengths when no whole record fits; their actual target position is recorded per item."),
    ("estimated", "Compute-time figures in the plan are estimates; measured times are in the latency files."),
)


def section_parity(summ: Optional[Dict[str, Any]], qdev: str) -> Dict[str, Any]:
    """CPU against GPU scoring of the same items (FR-024): may GPU-scored quality stand in for CPU quality?"""
    par = (summ or {}).get("parity") or {}
    rows = par.get("rows") or []
    out: Dict[str, Any] = {"rows": rows, "criteria": par.get("criteria"), "passed": par.get("passed"), "statements": []}
    if qdev != "gpu":
        out["statements"].append(stmt("measured", "All quality in this report was scored on CPU; no GPU parity check applies.", "summary.json"))
        return out
    if not rows:
        out["statements"].append(stmt("measured", "Quality was scored on the GPU but no CPU parity subset exists yet, so GPU-scored quality is unverified (FR-024).", "summary.json"))
        return out
    crit = par["criteria"]
    margin = crit.get("near_tie_margin")
    for r in rows:
        out["statements"].append(stmt(
            "measured", "%s: on %d items CPU and GPU give the same predicted answer for %s, accuracy %s on CPU and %s on GPU (difference %s), largest probability difference %.4f; "
                        "%d disagreement(s), %d tolerated near-tie flip(s) (CPU top-two margin at most %s), %d above it; parity %s (strict 98%% / 0.5-point rule: %s)%s."
            % (r["condition_id"], r["n_items"], pct(r["same_prediction_share"]), pct(r["accuracy_cpu"]), pct(r["accuracy_gpu"]),
               pts(r["accuracy_difference"]), r["max_abs_probability_difference"], r.get("n_disagreements", 0),
               len(r.get("tolerated_flips", [])), margin, len(r.get("failing_flips", [])),
               "passed" if r["passed"] else "FAILED", "passed" if r.get("strict_passed") else "missed",
               "" if r.get("counted", True) else "; NOT COUNTED: fewer than %s items" % crit.get("min_items")), "summary.json"))
        for f in r.get("tolerated_flips", []):
            out["statements"].append(stmt(
                "measured", "%s: near-tie flip on %s, CPU margin %.4f (CPU answered %s, GPU %s)."
                % (r["condition_id"], f["item_id"], f["cpu_margin"], f["cpu_predicted"], f["gpu_predicted"]), "summary.json"))
    if par["passed"]:
        verdict = ("GPU-scored quality stands in for CPU quality on this parity subset: every CPU/GPU disagreement is a near-tie flip "
                   "(CPU top-two margin at most %s, a researcher-chosen margin set from these same flips, on a small sample). "
                   "Strict 98%% / 0.5-point rule: %s." % (margin, "passed" if par.get("strict_passed") else "missed"))
    elif par["passed"] is None:
        verdict = ("GPU-scored quality is UNVERIFIED: parity is undetermined, because no compared condition has at least %s shared items, "
                   "so its numbers are labeled GPU and are not called CPU-equivalent." % crit.get("min_items"))
    else:
        verdict = ("GPU-scored quality is UNVERIFIED: at least one compared condition has a CPU/GPU disagreement above the near-tie margin, "
                   "so its numbers are labeled GPU and are not called CPU-equivalent.")
    out["statements"].append(stmt("measured", verdict, "summary.json"))
    return out


def build_report(run_path: Path, data_root: Optional[Path] = None) -> Dict[str, Any]:
    run_path = Path(run_path)
    if not run_path.exists():
        raise ReportError("run directory %s does not exist" % run_path)
    summ = _try(read_json, run_path, "summary.json")
    sections: Dict[str, Any] = {}
    sections["data"] = section_data(run_path, data_root)
    sections["solvability"] = section_solvability(run_path)
    sections["audit"] = section_audit(data_root)
    qdev = quality_device(summ)
    sections["parity"] = section_parity(summ, qdev)
    sections["curves"] = section_curves(run_path, summ, qdev)
    sections["verdicts"] = section_verdicts(summ, qdev)
    sections["margin"] = section_margin(run_path, summ, qdev)
    sections["variants"] = section_variants(run_path)
    sections["gpu"] = section_gpu(run_path, summ)
    sections["phase3"] = section_phase3(sections["verdicts"], sections["margin"], sections["variants"], sections["solvability"])
    plan = _try(read_json, run_path, "evaluation_plan.json")
    n_final = evalplan.count_final_items(run_path)
    plan_stmts = [stmt("measured", "Final-split items scored in this phase: %d." % n_final)]
    if plan:
        plan_stmts.append(stmt("estimated", plan["statement"], "evaluation_plan.json"))
        plan_stmts.append(stmt("measured", "The final-test protocol (metrics, %.0f-point margin, comparisons, required %s cases) was frozen at %s (version %s, fingerprint %s)."
                               % (plan["margin_pp"], plan["required_cases"], plan["frozen_at"], plan["version"], plan["fingerprint"][:16]), "evaluation_plan.json"))
    else:
        plan_stmts.append(stmt("measured", "The final-test protocol has not been frozen yet (run freeze-plan)."))
    sections["plan"] = {"statements": plan_stmts, "final_scored_items": n_final}
    sections["limits"] = {"statements": [stmt(l, t) for l, t in LIMITS]}
    cmds = []
    for base in (run_path, Path(run_path).parent / "data"):
        f = base / "commands.txt"
        if f.exists():
            cmds += [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
    sections["reproduce"] = {"commands": cmds}
    body = {"run_id": run_path.name, "sections": sections}
    for name, sec in sections.items():
        for s in sec.get("statements", []):
            assert s["label"] in LABELS
    return body


def _table(headers: List[str], rows: List[List[str]]) -> List[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    return out + ["| " + " | ".join(r) + " |" for r in rows] + [""]


def _fmt_ms(x: Optional[float]) -> str:
    return "n/a" if x is None else "%.0f" % x


def render_markdown(body: Dict[str, Any]) -> str:
    s = body["sections"]
    lines = ["# Phase 2 report: decision benchmark and practical baselines", "", "Run `%s`. Every statement is labeled "
             "`[measured]`, `[estimated]` or `[hypothesized]` and names the file it comes from." % body["run_id"], ""]

    def add(title, key):
        lines.append("## " + title)
        for st in s[key]["statements"]:
            src = (" (%s)" % ", ".join("`%s`" % x for x in st["sources"])) if st["sources"] else ""
            lines.append("- [%s] %s%s" % (st["label"], st["text"], src))
        lines.append("")

    add("Data and length profile", "data")
    add("Solvability and the quality reference", "solvability")
    add("Hand audit", "audit")
    add("CPU/GPU parity of quality scoring", "parity")
    add("Quality against cost (latency on CPU)", "curves")
    rows = [[r["name"] + (" (diagnostic control, not deployable)" if r["name"] == "oracle" else ""), str(r["length"]), r["status"], "%d/%d" % (r["n_measured"], r["n_items"]),
             pct(r["accuracy"]), "n/a" if r["ece_raw"] is None else "%.3f" % r["ece_raw"],
             "n/a" if r["ece_scaled"] is None else "%.3f" % r["ece_scaled"],
             _fmt_ms(r["cpu_p50_ms"]), _fmt_ms(r["cpu_p95_ms"])] for r in s["curves"]["rows"]]
    lines += _table(["baseline", "tokens", "status", "measured/items", "accuracy", "ECE raw", "ECE scaled", "CPU p50 ms", "CPU p95 ms"], rows)
    add("Native versus truncation (2K, 4K, 8K)", "verdicts")
    add("Simple baselines within the 2-point margin (FR-028)", "margin")
    add("Optimized-system variants (apart from the architectural baselines)", "variants")
    lines.append("## GPU latency (GPU only; never substituted for CPU)")
    for st in s["gpu"]["statements"]:
        lines.append("- [%s] %s" % (st["label"], st["text"]))
    lines += [""] + _table(["condition", "GPU p50 ms", "GPU p95 ms"], [[r["condition_id"], _fmt_ms(r["p50_ms"]), _fmt_ms(r["p95_ms"])] for r in s["gpu"]["rows"]])
    add("Phase 3 directions: what the evidence supports", "phase3")
    add("Final-test protocol", "plan")
    add("Limits", "limits")
    lines += ["## Reproduce", ""] + ["    " + c for c in s["reproduce"]["commands"]] + [""]
    return "\n".join(lines)


def write_report(run_path: Path, data_root: Optional[Path] = None):
    body = build_report(run_path, data_root)
    j = write_json(run_path, "report2.json", body)
    m = Path(run_path) / "report2.md"
    m.write_text(render_markdown(body), encoding="utf-8")
    return j, m
