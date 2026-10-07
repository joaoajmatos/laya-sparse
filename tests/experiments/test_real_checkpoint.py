"""T020 (slow): audit of the pinned English checkpoint. Needs network once, then the HF cache."""
import pytest

from experiments import audit as A
from experiments import cli, results

pytestmark = pytest.mark.slow


def test_audit_of_pinned_checkpoint(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(results, "RESULTS_ROOT", tmp_path)
    code = cli.main(["audit", "--run-id", "real", "--revision", "reviewed",
                     "--lengths", "512,4096,8192,16384"])
    assert code == 0
    run = tmp_path / "real"
    man = results.read_json(run, "manifest.json")
    aud = results.read_json(run, "audit.json")
    from laya.revisions import PINNED_REVISIONS
    assert man["model"]["revision"] == PINNED_REVISIONS["convaiinnovations/laya"]
    assert man["device"]["effective"] == "cpu" and man["fixture_model"] is False
    assert aud["layers"] and {l["attention_type"] for l in aud["layers"]} <= {"local", "global"}
    # Differences from docs/research-plan.md are recorded, never a failure.
    assert set(aud["research_plan_comparison"]) == set(A.RESEARCH_PLAN_EXPECTED)
    assert set(aud["discrepancies"]) == {k for k, v in aud["research_plan_comparison"].items() if not v["matches"]}
    over = [l for l in aud["lengths"] if l["length"] > aud["positional"]["max_position_embeddings"]]
    assert all(l["status"] == "unsupported" for l in over)
    print(A.format_schedule(aud))



# --------------------------------------------------------------------------- Phase 2 (specs/002): real data and checkpoints

@pytest.fixture(scope="module")
def real_data(tmp_path_factory):
    """The pinned upstream dataset, imported once into a temp data root (network on first use)."""
    from experiments import data
    root = tmp_path_factory.mktemp("real-data")
    man = data.import_dataset(root=root)
    data.make_splits(seed=data.DEFAULT_SPLIT_SEED, root=root)
    return root, man


@pytest.fixture(scope="module")
def typed_agent():
    from experiments.runner import load_agent
    agent, _ = load_agent("convaiinnovations/laya-typed-decisions", revision="reviewed", threads=None)
    return agent


def test_real_dataset_counts_splits_and_fingerprint(real_data):
    from experiments import data
    root, man = real_data
    assert man["revision"] == data.DATASET_REVISION and man["license"] == "apache-2.0"
    assert man["splits"]["train"]["cases"] == 1200 and man["splits"]["test"]["cases"] == 400
    assert man["splits"]["test"]["questions"] == 2000
    assert man["splits"]["test"]["questions_by_type"] == {"choice": 600, "noul": 600, "score": 800}
    assert man["workflows"]["test"] == {wf: 100 for wf in data.WORKFLOWS}
    again = data.import_dataset(root=root)                     # a re-import reproduces the fingerprint
    assert again["fingerprint"] == man["fingerprint"]
    sp = data.read_data_json("splits.json", root)
    assert [sp["splits"][s]["n_cases"] for s in data.EVAL_SPLITS] == [120, 80, 200]
    assert sp["splits"]["dev"]["n_per_workflow"] == {wf: 30 for wf in data.WORKFLOWS}
    assert len(sp["latency_sample"]) == 12 and len(sp["variant_sample"]) == 20
    assert man["low_confidence"]["basis"] == "train" and man["low_confidence"]["n_questions"] == 6000


def test_real_checkpoint_caps_and_original_rows_run_on_cpu(real_data, typed_agent):
    from experiments import data, evalrun
    root, _ = real_data
    assert int(typed_agent.cfg["max_len"]) == 1024 and int(typed_agent.cfg["head_max_len"]) == 256
    assert typed_agent.device.type == "cpu"
    items = data.original_items("dev", root=root)[:10]
    recs = evalrun.run_items(typed_agent, items, evalrun.native_condition("fine_tuned"), evalrun.NativeRunner())
    assert len(recs) == 10 and all(r["status"] == "measured" and r["device"] == "cpu" for r in recs)


def test_real_families_reach_exact_lengths_including_8192(real_data, typed_agent):
    from experiments import data, families
    root, _ = real_data
    pool = families.ReferencePool(data.load_cases("train", root))
    words = families.neutral_words(typed_agent)
    cases = data.split_cases("dev", root)
    short = next(c for c in cases if c["workflow"] == "customer_service")
    refs = pool.ordered(short, 0)
    for length in (512, 2048, 8192):
        for variant in ("neutral@mid", "distractor@mid", "distractor@end"):
            it = families.build_item(typed_agent, short, "action" if "action" in short["questions"] else list(short["questions"])[0],
                                     variant, length, "dev", refs, words, 0)
            if it["status"] != "ok":
                assert length == 512 and it["reason"].startswith(("target_exceeds_length", "no_room_for_context"))
                continue
            assert it["accounting"]["rows"][0]["final_length"] == length and not it["accounting"]["rows"][0]["truncated"]
            assert it["evidence_span"]["char_end"] > it["evidence_span"]["char_start"]
    # A customer-service case whose longest row is over 512 tokens is unsupported at 512 and only there.
    big = max((c for c in cases if c["workflow"] == "customer_service"),
              key=lambda c: len(data.serialize_state(c["state"])))
    at512 = families.build_item(typed_agent, big, list(big["questions"])[0], "distractor@mid", 512, "dev",
                                pool.ordered(big, 0), words, 0)
    at1024 = families.build_item(typed_agent, big, list(big["questions"])[0], "distractor@mid", 1024, "dev",
                                 pool.ordered(big, 0), words, 0)
    assert at1024["status"] == "ok"
    assert at512["status"] in ("ok", "unsupported")


def test_real_variants_run_and_their_shift_is_recorded_not_asserted_small(real_data, typed_agent):
    import copy
    from experiments import data, evalrun, variants
    root, _ = real_data
    items = data.original_items("dev", root=root)[:6]
    cond = evalrun.native_condition("fine_tuned")
    native = evalrun.run_items(typed_agent, items, cond, evalrun.NativeRunner())
    with variants.fastpath(False):
        off = evalrun.run_items(typed_agent, items, dict(cond, condition_id="off"), evalrun.NativeRunner())
    for a, b in zip(native, off):
        for k in a["probabilities"]:
            assert abs(a["probabilities"][k] - b["probabilities"][k]) < 1e-3       # fast-path off agrees with native
    saved = copy.deepcopy(typed_agent.model.encoder)
    rec = variants.apply_variant(typed_agent, "int8_encoder")
    try:
        assert rec["quantized_layers"] > 0
        q = evalrun.run_items(typed_agent, items, dict(cond, condition_id="int8"), evalrun.NativeRunner())
        shifts = [abs(a["probabilities"][k] - b["probabilities"][k]) for a, b in zip(native, q) for k in a["probabilities"]]
        print("int8_encoder max probability shift on 6 rows: %.4f" % max(shifts))
        assert all(r["status"] == "measured" for r in q)
    finally:
        typed_agent.model.encoder = saved


# --------------------------------------------------------------------------- laya:007: exact local attention, layer level

@pytest.mark.parametrize("length", [512, 1024, 2048, 4096, 8192])
def test_real_local_layers_exact_kernel_matches_native_layer_on_identical_inputs(typed_agent, length):
    """Every local layer: the exact kernel and the native dense-masked layer, same hidden states, padded tail, <= 1e-5."""
    import torch
    from experiments import variants as V
    enc = typed_agent.model.encoder
    local = V._local_layers(enc)
    assert len(local) == 18 and {V._half_width(l.attn, enc.config) for _, l in local} == {64}
    captured = {}

    def grab(i):
        def pre(_m, args, kwargs):
            captured[i] = (args, dict(kwargs))
        return pre
    handles = [layer.attn.register_forward_pre_hook(grab(i), with_kwargs=True) for i, layer in local]
    g = torch.Generator().manual_seed(7)
    ids = torch.randint(1000, 20000, (2, length), generator=g)
    att = (torch.arange(length)[None, :] < torch.tensor([length, length - 37])[:, None]).long()
    try:
        with torch.no_grad():
            enc(input_ids=ids * att, attention_mask=att)
    finally:
        for h in handles:
            h.remove()
    valid = att.bool()
    worst = {}
    with torch.no_grad():
        for i, layer in local:
            args, kwargs = captured[i]
            native = type(layer.attn).forward(layer.attn, *args, **kwargs)[0]
            exact = V._exact_forward(layer.attn, 64)(*args, **kwargs)[0]
            worst[i] = float((native - exact).abs()[valid].max())
    assert max(worst.values()) <= 1e-5, worst
