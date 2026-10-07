"""laya:008: fresh parity sets (experiments/parity_draw.py)."""
import hashlib
import json

import pytest

from experiments import parity_draw as P

WFS = ("wf_a", "wf_b", "wf_c", "wf_d")


def _splits(per_wf_dev=30, half=15, variant=5, wfs=WFS):
    dev, halfs, var = [], [], []
    for w in wfs:
        ids = ["%s_%06d" % (w, i) for i in range(per_wf_dev)]
        dev += ids
        halfs += ids[:half]
        var += ids[10:10 + variant]            # 5 of the half-sample's 15 per workflow
    return {"splits": {"dev": {"case_ids": sorted(dev), "fingerprint": "f" * 64}},
            "half_sample": {"dev": sorted(halfs)}, "variant_sample": sorted(var)}


def test_same_seed_same_sets_and_rule_holds():
    sp = _splits()
    one, two = P.draw_fresh_sets(sp), P.draw_fresh_sets(sp)
    assert one == two and one["seed"] == 2026100801
    assert len(one["set_a"]) == len(one["set_b"]) == 20 and one["pool_size"] == 40
    assert P.check_sets(one, sp) == []
    for w in WFS:
        assert sum(c.startswith(w) for c in one["set_a"]) == 5 and sum(c.startswith(w) for c in one["set_b"]) == 5
    assert all(v == {"pool": 10, "set_a": 5, "set_b": 5} for v in one["per_workflow"].values())


def test_order_is_the_sha256_of_seed_and_id_and_the_seed_matters():
    sp = _splits()
    body = P.draw_fresh_sets(sp, seed=2026100801)
    pool = [c for c in sp["half_sample"]["dev"] if c not in sp["variant_sample"] and c.startswith("wf_a")]
    want = sorted(pool, key=lambda c: hashlib.sha256(("2026100801:" + c).encode()).hexdigest())
    assert sorted(want[:5]) == [c for c in body["set_a"] if c.startswith("wf_a")]
    assert sorted(want[5:10]) == [c for c in body["set_b"] if c.startswith("wf_a")]
    assert P.draw_fresh_sets(sp, seed=1)["set_a"] != body["set_a"]
    assert body["ids_sha256"] == hashlib.sha256(json.dumps({"set_a": body["set_a"], "set_b": body["set_b"]}, sort_keys=True).encode()).hexdigest()


def test_a_short_workflow_is_an_error_and_the_rule_is_not_relaxed():
    sp = _splits()
    sp["half_sample"]["dev"] = [c for c in sp["half_sample"]["dev"] if c != "wf_b_000003"]
    with pytest.raises(P.ParityDrawError, match="wf_b"):
        P.draw_fresh_sets(sp)


def test_check_sets_catches_overlap_and_variant_leak():
    sp = _splits()
    body = P.draw_fresh_sets(sp)
    bad = dict(body, set_b=body["set_a"])
    assert "sets A and B overlap" in P.check_sets(bad, sp)
    leak = dict(body, set_a=body["set_a"][:-1] + [sp["variant_sample"][0]])
    assert any("variant sample" in p for p in P.check_sets(leak, sp))
