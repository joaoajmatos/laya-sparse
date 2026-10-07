"""Tier E check of the compression gate: the exact local-attention kernel (laya:007, plan v2 section 2).

* ``probs``    E1: option probabilities of native vs the exact kernel on the same items (CPU fp32), plus the
               max abs difference of every local layer's attention output (end to end, so later layers also
               carry the earlier layers' differences). Resumable per length.
* ``compare``  E2: predicted answers of a candidate run against reference predictions (p2-dev native, CPU).
* ``report``   E1/E2/E3 verdicts from the files above and the latency runs. Nothing is loosened: a failure
               is reported with the observed numbers.

Dev items only: the final split is never read here (``final_scored_items`` is 0). Every record carries
device and dtype.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from . import results

E1_LENGTHS = (512, 1024, 2048, 4096, 8192)
E1_TOLERANCE = 1e-5
E1_ITEMS_PER_LENGTH = 20
E1_VARIANT = "distractor@mid"
DEVICE, DTYPE = "cpu", "fp32"
SUBDIR = "tier_e"
PARITY_LENGTHS = (512, 2048, 8192)


class TierEError(RuntimeError):
    """The check cannot run as asked."""


def require_clean_tree(allow_dirty: bool = False) -> Dict[str, Any]:
    """Code info of the working tree; a gate run refuses a dirty tree or a changed `laya/` (plan v2 section 1)."""
    from .manifest import code_info
    info = code_info()
    if not allow_dirty:
        if info.get("git_dirty") is not False:
            raise TierEError("a gate run needs a clean git tree (git_dirty=%s; paths: %s); commit or exclude them"
                             % (info.get("git_dirty"), info.get("dirty_paths")))
    if info.get("laya_diff_empty") is False:
        raise TierEError("laya/ differs from the recorded commit; the inference contract must stay unchanged")
    return info


# --------------------------------------------------------------------------- E1: items

def e1_items(items: Iterable[Dict[str, Any]], variant_sample: Sequence[str], length: int,
             n: int = E1_ITEMS_PER_LENGTH) -> List[Dict[str, Any]]:
    """The first question (sorted by question id) of each variant-sample case at `length`, ``distractor@mid``, dev.

    One item per case, cases in sorted order, at most `n`. The caller names the ids in the report.
    """
    cases = set(variant_sample)
    by_case: Dict[str, Dict[str, Any]] = {}
    for it in items:
        if (it["length"] == length and it["variant"] == E1_VARIANT and it["split"] == "dev"
                and it["case_id"] in cases and it.get("status", "ok") == "ok"):
            cur = by_case.get(it["case_id"])
            if cur is None or it["question_id"] < cur["question_id"]:
                by_case[it["case_id"]] = it
    return [by_case[c] for c in sorted(by_case)][:n]


# --------------------------------------------------------------------------- E1: measuring

class _LayerTap:
    """Forward hooks on the attention modules of the local layers; records or compares their outputs."""

    def __init__(self, layers: Sequence[int], modules: Sequence[Any]):
        self.layers = list(layers)
        self.store: Dict[int, Any] = {}
        self.diffs: Dict[int, float] = {}
        self.reference: Optional[Dict[int, Any]] = None
        self._handles = [m.register_forward_hook(self._hook(i)) for i, m in zip(layers, modules)]

    def _hook(self, i: int):
        def hook(_module, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if self.reference is None:
                self.store[i] = t.detach().clone()
            else:
                d = float((t.detach() - self.reference[i]).abs().max())
                self.diffs[i] = max(self.diffs.get(i, 0.0), d)
        return hook

    def close(self) -> None:
        for h in self._handles:
            h.remove()


def measure_item(agent, runner, item: Dict[str, Any], restore_after: bool = True) -> Dict[str, Any]:
    """Native then exact-kernel forward on one item (same process, device, dtype); returns the differences."""
    import torch
    from . import variants as V
    from .evalrun import parse_answer
    encoder = agent.model.encoder
    local = V._local_layers(encoder)
    tap = _LayerTap([i for i, _ in local], [layer.attn for _, layer in local])
    try:
        with torch.no_grad():
            native = parse_answer(item, runner(agent, item)["answer"])
            tap.reference = dict(tap.store)
            rec = V.apply_local_exact(agent)
            try:
                exact = parse_answer(item, runner(agent, item)["answer"])
            finally:
                rec["restore"]()
    finally:
        tap.close()
    keys = sorted(set(native["probabilities"]) | set(exact["probabilities"]))
    pdiff = max(abs(native["probabilities"].get(k, 0.0) - exact["probabilities"].get(k, 0.0)) for k in keys)
    return {"item_id": item["item_id"], "case_id": item["case_id"], "question_id": item["question_id"],
            "length": item["length"], "max_abs_prob_diff": pdiff,
            "native_predicted": native["predicted"], "exact_predicted": exact["predicted"],
            "prediction_changed": native["predicted"] != exact["predicted"],
            "local_layer_max_abs_diff": {str(i): tap.diffs.get(i) for i in tap.layers},
            "native_probabilities": native["probabilities"], "exact_probabilities": exact["probabilities"]}


def run_probs(run_path: Path, model: str, revision: Optional[str], threads: Optional[int], items_by_length: Dict[int, List[Dict[str, Any]]],
              code: Dict[str, Any], log=print) -> List[Path]:
    """E1 over `items_by_length`; one file per length under ``<run>/tier_e/``; finished lengths are skipped."""
    from . import variants as V
    from .baselines import build_runner
    out_paths: List[Path] = []
    todo = {L: its for L, its in items_by_length.items()
            if not (Path(run_path) / SUBDIR / ("probs.L%d.json" % L)).exists()}
    for L, its in items_by_length.items():
        if L not in todo:
            out_paths.append(Path(run_path) / SUBDIR / ("probs.L%d.json" % L))
    if todo:
        agent, info, _ = V.load_for_variant(model, revision, threads, "fastpath_off")   # native reference, fast path off
        runner = build_runner("native")
        for L, its in todo.items():
            started = time.perf_counter()
            rows = []
            for k, it in enumerate(its):
                rows.append(measure_item(agent, runner, it))
                log("L%d item %d/%d %s max_prob_diff %.3g" % (L, k + 1, len(its), it["item_id"], rows[-1]["max_abs_prob_diff"]))
            payload = {"length": L, "device": DEVICE, "dtype": DTYPE, "mha_fastpath": False,
                       "reference": "native, mha fast path off (fastpath_off, within 5e-7 of native: research.md R17)",
                       "candidate": "local_exact", "tolerance": E1_TOLERANCE, "threads": threads, "load_info": info,
                       "elapsed_s": time.perf_counter() - started, "rows": rows, "code": code,
                       "final_scored_items": 0}
            out_paths.append(results.write_json(run_path, "%s/probs.L%d.json" % (SUBDIR, L), payload))
    return out_paths


def summarize_e1(per_length: Dict[int, Dict[str, Any]], tolerance: float = E1_TOLERANCE) -> Dict[str, Any]:
    """E1 verdict: max abs probability difference per length and per local layer; pass only if all lengths pass."""
    rows = []
    for L in sorted(per_length):
        its = per_length[L]["rows"]
        layer_max: Dict[str, float] = {}
        for it in its:
            for lay, d in it["local_layer_max_abs_diff"].items():
                if d is not None:
                    layer_max[lay] = max(layer_max.get(lay, 0.0), d)
        pmax = max((it["max_abs_prob_diff"] for it in its), default=None)
        rows.append({"length": L, "n_items": len(its), "max_abs_prob_diff": pmax,
                     "passed": pmax is not None and pmax <= tolerance, "predictions_changed":
                     sum(1 for it in its if it["prediction_changed"]), "per_local_layer_max_abs_diff": layer_max,
                     "item_ids": [it["item_id"] for it in its]})
    missing = [L for L in E1_LENGTHS if L not in per_length]
    return {"criterion": "max abs probability difference <= %g at every length, %d items per length"
                         % (tolerance, E1_ITEMS_PER_LENGTH),
            "device": DEVICE, "dtype": DTYPE, "lengths": rows, "missing_lengths": missing,
            "passed": bool(rows) and not missing and all(r["passed"] and r["n_items"] == E1_ITEMS_PER_LENGTH for r in rows),
            "observed_max": max((r["max_abs_prob_diff"] for r in rows if r["max_abs_prob_diff"] is not None), default=None)}


# --------------------------------------------------------------------------- E2: predictions

def _load_predictions(run_path: Path, subdir: str, condition_id: str) -> Dict[str, Dict[str, Any]]:
    path = Path(run_path) / subdir / condition_id / "predictions.jsonl"
    return {r["item_id"]: r for r in results.read_jsonl(path) if r.get("status") == "measured"}


def compare_predictions(candidate: Dict[str, Dict[str, Any]], reference: Dict[str, Dict[str, Any]],
                        expected: Optional[Dict[int, int]] = None) -> Dict[str, Any]:
    """E2: identical predicted answers per length. Items missing on either side count as failures."""
    ids = sorted(set(reference) | set(candidate))
    per: Dict[int, Dict[str, Any]] = {}
    for iid in ids:
        r, c = reference.get(iid), candidate.get(iid)
        L = (r or c)["length"]
        b = per.setdefault(L, {"length": L, "n": 0, "changed": 0, "missing": 0, "flips": []})
        b["n"] += 1
        if r is None or c is None:
            b["missing"] += 1
        elif r["predicted"] != c["predicted"]:
            b["changed"] += 1
            top = sorted(r["probabilities"].values(), reverse=True)
            b["flips"].append({"item_id": iid, "reference": r["predicted"], "candidate": c["predicted"],
                               "reference_top2_margin": (top[0] - top[1]) if len(top) > 1 else None})
    rows = [per[L] for L in sorted(per)]
    size_ok = all(per.get(L, {"n": -1})["n"] == n for L, n in (expected or {}).items())
    return {"criterion": "100% identical predicted answers on the CPU parity subset", "lengths": rows,
            "n": sum(r["n"] for r in rows), "changed": sum(r["changed"] for r in rows),
            "missing": sum(r["missing"] for r in rows), "expected_sizes_ok": size_ok,
            "passed": bool(rows) and size_ok and all(r["changed"] == 0 and r["missing"] == 0 for r in rows),
            "device": DEVICE, "dtype": DTYPE}


# --------------------------------------------------------------------------- E3: latency

def summarize_e3(latency: Dict[int, Dict[str, Dict[str, float]]]) -> Dict[str, Any]:
    """E3 from ``{length: {variant: {p50, p95}}}`` in ms: optimized native p50 not slower than fastpath_off at any length."""
    rows = []
    for L in sorted(latency):
        d = latency[L]
        opt, off, raw = d.get("local_exact_fastpath_off"), d.get("fastpath_off"), d.get("none")
        rows.append({"length": L, "raw_native": raw, "fastpath_off": off, "optimized_native": opt,
                     "optimized_vs_fastpath_off_p50": (opt["p50"] / off["p50"]) if opt and off else None,
                     "optimized_vs_raw_p50": (opt["p50"] / raw["p50"]) if opt and raw else None,
                     "passed": bool(opt and off and opt["p50"] <= off["p50"])})
    return {"criterion": "optimized native p50 <= fastpath_off native p50 at every length (speed reported, p95 reported)",
            "device": DEVICE, "dtype": DTYPE, "lengths": rows,
            "missing_lengths": [L for L in E1_LENGTHS if L not in latency],
            "passed": bool(rows) and all(r["passed"] for r in rows) and all(L in latency for L in E1_LENGTHS)}


def read_latency(run_path: Path, variants: Sequence[str] = ("none", "fastpath_off", "local_exact_fastpath_off"),
                 subdir: str = "latency") -> Dict[int, Dict[str, Dict[str, float]]]:
    out: Dict[int, Dict[str, Dict[str, float]]] = {}
    for L in E1_LENGTHS:
        for v in variants:
            p = Path(run_path) / subdir / ("native.%s.cpu.L%d.json" % (v, L))
            if not p.exists():
                continue
            d = results.read_json(run_path, "%s/%s" % (subdir, p.name))
            t = d.get("timings_ms")
            if d.get("status") == "measured" and t:
                out.setdefault(L, {})[v] = {"p50": t["p50"], "p95": t["p95"], "n": t["n"],
                                            "peak_rss_bytes": d.get("peak_rss_bytes")}
    return out


def build_report(run_path: Path, e1: Optional[Dict[str, Any]], e2: Optional[Dict[str, Any]],
                 e3: Optional[Dict[str, Any]], code: Dict[str, Any]) -> Dict[str, Any]:
    """The Tier E report: each criterion resolves to passed, failed or not_run, with the observed numbers."""
    def verdict(x):
        return "not_run" if x is None else ("passed" if x["passed"] else "failed")
    verdicts = {"E1": verdict(e1), "E2": verdict(e2), "E3": verdict(e3)}
    return {"tier": "E", "spec": "laya:007", "verdicts": verdicts,
            "tier_e_passed": all(v == "passed" for v in verdicts.values()),
            "E1": e1, "E2": e2, "E3": e3, "device": DEVICE, "dtype": DTYPE, "code": code,
            "final_scored_items": 0}
