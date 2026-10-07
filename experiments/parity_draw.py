"""Fresh parity case sets for the Tier F screen (laya:008; docs/gate-plan-v2.md section 4).

Deterministic, offline, no model and no measured result involved. Pool = ``half_sample.dev`` minus
``variant_sample`` (40 cases, 10 per workflow on the real splits). Within each workflow the pool is ordered by
``sha256("<seed>:" + case_id)``; the first ``per_workflow`` cases go to set A and the next ``per_workflow`` to set B.
A workflow with fewer than ``2 x per_workflow`` pool cases is an error: the rule is never relaxed.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence

PARITY_SEED = 2026100801
PER_WORKFLOW = 5
LENGTHS = (512, 2048, 4096, 8192)
ITEM_VARIANT = "distractor@mid"


class ParityDrawError(ValueError):
    """The draw cannot meet the plan's rule on these splits."""


def _key(seed: int, case_id: str) -> str:
    return hashlib.sha256(("%d:%s" % (seed, case_id)).encode("utf-8")).hexdigest()


def _workflow_of(splits: Dict[str, Any], workflows: Optional[Dict[str, str]]) -> Dict[str, str]:
    if workflows is not None:
        return workflows
    # case ids are "<workflow>_<number>" (e.g. agent_trace_observability_000032)
    return {cid: cid.rsplit("_", 1)[0] for cid in splits["splits"]["dev"]["case_ids"]}


def draw_fresh_sets(splits: Dict[str, Any], seed: int = PARITY_SEED, per_workflow: int = PER_WORKFLOW,
                    workflows: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    dev = set(splits["splits"]["dev"]["case_ids"])
    half = set(splits["half_sample"]["dev"])
    variant = set(splits["variant_sample"])
    if not half <= dev:
        raise ParityDrawError("the dev half-sample is not inside the dev split")
    pool = sorted(half - variant)
    wf = _workflow_of(splits, workflows)
    by_wf: Dict[str, List[str]] = {}
    for cid in pool:
        by_wf.setdefault(wf[cid], []).append(cid)
    a: List[str] = []
    b: List[str] = []
    per: Dict[str, Dict[str, int]] = {}
    for w in sorted(by_wf):
        order = sorted(by_wf[w], key=lambda c: _key(seed, c))
        if len(order) < 2 * per_workflow:
            raise ParityDrawError("workflow %s has %d pool cases; plan section 4 needs %d (5 per set). Reported, rule not changed."
                                  % (w, len(order), 2 * per_workflow))
        a += order[:per_workflow]
        b += order[per_workflow:2 * per_workflow]
        per[w] = {"pool": len(order), "set_a": per_workflow, "set_b": per_workflow}
    if not by_wf:
        raise ParityDrawError("the pool (half-sample minus variant sample) is empty")
    a, b = sorted(a), sorted(b)
    body = {"seed": seed, "procedure": "pool = half_sample.dev minus variant_sample; per workflow order by "
                                       "sha256(\"<seed>:\" + case_id); first %d to A, next %d to B" % (per_workflow, per_workflow),
            "dev_split_fingerprint": splits["splits"]["dev"]["fingerprint"], "pool_size": len(pool),
            "per_workflow": per, "item_variant": ITEM_VARIANT, "lengths": list(LENGTHS),
            "set_a": a, "set_b": b}
    body["ids_sha256"] = hashlib.sha256(json.dumps({"set_a": a, "set_b": b}, sort_keys=True).encode("utf-8")).hexdigest()
    return body


def check_sets(body: Dict[str, Any], splits: Dict[str, Any]) -> List[str]:
    """Problems with a drawn body (empty list when it meets the rule)."""
    problems: List[str] = []
    a, b = set(body["set_a"]), set(body["set_b"])
    if a & b:
        problems.append("sets A and B overlap")
    if (a | b) & set(splits["variant_sample"]):
        problems.append("a case is in the original variant sample")
    if not (a | b) <= set(splits["splits"]["dev"]["case_ids"]):
        problems.append("a case is outside dev")
    if not (a | b) <= set(splits["half_sample"]["dev"]):
        problems.append("a case is outside the dev half-sample (4K/8K would not be in the dev grid)")
    if len(a) != 20 or len(b) != 20:
        problems.append("a set does not have 20 cases")
    return problems
