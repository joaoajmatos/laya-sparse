"""laya:004 (implements raya:002): golden fixtures. Offline, CPU, no timings."""
import json
import re

import pytest
import torch
from safetensors.torch import load_file

from experiments import cli
from experiments.golden import common, forward, kernels, weights

CASES = ("short", "medium", "padded", "fastpath_off")
STAGE_KEYS = {"input_ids", "attention_mask", "position_ids", "marker_pos", "marker_mask", "qtype",
              "embedding_output", "encoder_layer_00", "encoder_layer_01", "encoder_layer_02", "encoder_final_norm",
              "encoder_output", "type_emb_add", "head_input", "head_layer_00", "marker_states", "logits",
              "probabilities", "act_features", "pooled", "act_logits"}


def _files(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in common.iter_files(root)}


@pytest.fixture(scope="module")
def exports(tmp_path_factory):
    a, b = tmp_path_factory.mktemp("golden-a"), tmp_path_factory.mktemp("golden-b")
    forward.export_all(a)
    forward.export_all(b)
    return a, b


def test_export_has_every_case_and_tensor(exports):
    root, _ = exports
    for case in CASES:
        tensors = load_file(str(root / case / "tensors.safetensors"))
        assert STAGE_KEYS <= set(tensors), STAGE_KEYS - set(tensors)
        assert (root / case / "meta.json").is_file()
        meta = json.loads((root / case / "meta.json").read_text(encoding="utf-8"))
        L = meta["input"]["sequence_length"]
        hidden = meta["fixture_hidden"]
        assert tensors["input_ids"].shape == (1, L) and tensors["attention_mask"].shape == (1, L)
        for name in ("embedding_output", "encoder_output", "head_input", "head_layer_00", "encoder_layer_02"):
            assert tensors[name].shape == (1, L, hidden) and tensors[name].dtype == torch.float32
        k = tensors["marker_pos"].shape[1]
        assert tensors["marker_states"].shape == (1, k, hidden)
        assert tensors["logits"].shape == (1, k) and tensors["act_logits"].shape == (1, 2)
        assert tensors["type_emb_add"].shape == (1, 1, hidden)
        assert torch.allclose(tensors["probabilities"].sum(-1), torch.ones(1), atol=1e-6)
    assert (root / "weights-h64.safetensors").is_file() and (root / "weights-h128.safetensors").is_file()


def test_meta_records_commit_seed_versions_and_config(exports):
    root, _ = exports
    meta = json.loads((root / "short" / "meta.json").read_text(encoding="utf-8"))
    assert re.fullmatch(r"[0-9a-f]{40}", meta["laya_sparse_commit"])
    assert meta["seed"] == 0 and meta["dtype"] == "float32" and meta["mode"] == "eval" and meta["threads"] == 1
    assert meta["versions"]["torch"] == torch.__version__
    assert meta["encoder_config"]["num_hidden_layers"] == 3 and meta["agent_config"]["head_layers"] == 1
    assert "_name_or_path" not in meta["encoder_config"]          # a temp path would break byte identity
    assert meta["tensors"]["logits"]["dtype"] == "float32"


def test_padded_case_masks_the_padding(exports):
    root, _ = exports
    t = load_file(str(root / "padded" / "tensors.safetensors"))
    meta = json.loads((root / "padded" / "meta.json").read_text(encoding="utf-8"))
    n = meta["input"]["unpadded_length"]
    assert t["input_ids"].shape[1] == 160 > n
    assert t["attention_mask"][0, :n].all() and not t["attention_mask"][0, n:].any()
    assert (t["input_ids"][0, n:] == 0).all()                     # [PAD] id


def test_weights_cover_the_model(exports):
    root, _ = exports
    w = load_file(str(root / "weights-h64.safetensors"))
    with common.FixtureAgent(64) as fx:
        sd = fx.agent.model.state_dict()
        assert set(w) == set(sd)
        for k, v in sd.items():
            assert torch.equal(w[k], v)


def test_fastpath_case_is_a_real_fastpath_difference(exports):
    root, _ = exports
    meta = json.loads((root / "fastpath_off" / "meta.json").read_text(encoding="utf-8"))
    assert meta["mha_fastpath"] is False and meta["head_heads"] == 2
    assert meta["fastpath_off_max_abs_diff_vs_on"] <= forward.FASTPATH_TOL


def test_fastpath_mismatch_fails_the_export(tmp_path, monkeypatch):
    monkeypatch.setattr(forward, "FASTPATH_TOL", -1.0)
    with pytest.raises(RuntimeError, match="fast path"):
        forward.export_all(tmp_path, cases=["fastpath_off"])


def test_exports_are_byte_identical(exports):
    a, b = exports
    fa, fb = _files(a), _files(b)
    assert fa.keys() == fb.keys()
    assert all(fa[k] == fb[k] for k in fa), [k for k in fa if fa[k] != fb[k]]


def test_vendorable_set_is_small(exports):
    root, _ = exports
    vend = [p for p in common.iter_files(root)
            if "h128" not in str(p.relative_to(root)) and "fastpath_off" not in str(p.relative_to(root))]
    assert sum(p.stat().st_size for p in vend) < 5 * 1024 * 1024


def test_inventory_matches_state_dict(tmp_path):
    path = weights.write_inventory(tmp_path)
    body = json.loads(path.read_text(encoding="utf-8"))
    with common.FixtureAgent(64) as fx:
        sd = fx.agent.model.state_dict()
    assert set(body["keys"]) == set(sd) and body["n_keys"] == len(sd)
    for k, v in sd.items():
        assert body["keys"][k] == {"shape": list(v.shape), "dtype": "float32"}
    assert body["meta"]["model"] == "fixture"
    assert path.read_bytes() == weights.write_inventory(tmp_path / "again").read_bytes()


def test_kernel_goldens_record_inputs_outputs_and_diff(tmp_path):
    records = kernels.export_kernels(tmp_path, lengths=(256,))
    assert set(records) == {"block_local_L256", "gas_L256"}
    for name, meta in records.items():
        t = load_file(str(tmp_path / "kernels" / name / "tensors.safetensors"))
        assert set(t) == {"q", "k", "v", "reference", "kernel", "max_abs_diff"}
        assert t["q"].shape == (1, 16, 256, 64)
        assert float(t["max_abs_diff"]) == meta["max_abs_diff"] <= kernels.TOLERANCE
    again = tmp_path / "again"
    kernels.export_kernels(again, lengths=(256,))
    assert _files(tmp_path / "kernels") == _files(again / "kernels")


def test_kernel_diff_above_tolerance_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(kernels, "TOLERANCE", 0.0)
    with pytest.raises(RuntimeError, match="max abs diff"):
        kernels.export_kernels(tmp_path, lengths=(256,))


def test_cli_golden_command(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["golden", "weights-inventory", "--out", str(tmp_path / "o")]) == 0
    assert (tmp_path / "o" / "inventory.json").is_file()


# --------------------------------------------------------------------------- laya:006: config beside the weights

def test_config_files_agree_with_the_weights_and_the_tokenizer_goldens(exports, tmp_path):
    root, _ = exports
    from experiments.golden import tokenizer
    tokenizer.export_tokenizer(tmp_path)
    tok_meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    for hidden in (64, 128):
        cfg = json.loads((root / forward.config_dir_name(hidden) / "encoder" / "config.json").read_text(encoding="utf-8"))
        w = load_file(str(root / ("weights-h%d.safetensors" % hidden)))
        emb = next(v for k, v in w.items() if k.endswith("embeddings.tok_embeddings.weight"))
        assert cfg["vocab_size"] == emb.shape[0] and cfg["hidden_size"] == emb.shape[1] == hidden
        assert cfg["vocab_size"] == tok_meta["vocab_size"]                       # config, weights and tokenizer agree
        assert cfg["num_hidden_layers"] == 3 and cfg["num_attention_heads"] == 2
        layers = {k.split(".")[2] for k in w if k.startswith("encoder.layers.")}
        assert len(layers) == cfg["num_hidden_layers"]
        assert w["encoder.layers.0.mlp.Wi.weight"].shape[0] == 2 * cfg["intermediate_size"]  # gated MLP: 2 x intermediate
        assert cfg["intermediate_size"] == 2 * hidden


def test_agent_config_beside_the_weights_matches_the_case_meta(exports):
    root, _ = exports
    for case, hidden in (("short", 64), ("fastpath_off", 128)):
        meta = json.loads((root / case / "meta.json").read_text(encoding="utf-8"))
        assert meta["config_dir"] == "../" + forward.config_dir_name(hidden)
        agent_cfg = json.loads((root / forward.config_dir_name(hidden) / "rl_agent_config.json").read_text(encoding="utf-8"))
        assert agent_cfg == meta["agent_config"] and agent_cfg["laya_sparse_fixture"] is True
        assert (root / meta["config_dir"][3:] / "encoder" / "config.json").is_file()

