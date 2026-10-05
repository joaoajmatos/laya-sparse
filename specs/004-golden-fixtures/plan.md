# Implementation Plan: Golden Fixtures (laya:004, implements raya:002)

**Spec**: [spec.md](spec.md). Python 3.11 CPU torch (`.venv\Scripts\python`), transformers, safetensors; all already in `.[experiments]`.

## Design

- `experiments/golden/common.py`: determinism (`make_deterministic`: 1 thread, seed 0, fp32, deterministic algorithms), the fixture agent
  (conftest's `build_fixture_checkpoint`, seeded 1234, with the tokenizer swapped for a deterministic one, see below), byte-stable
  `write_json` (sorted keys, LF) and `save_tensors` (safetensors, sorted names), `base_meta` (commit SHA, versions, seed, threads).
- `weights.py`: `inventory.json` from `state_dict()`.
- `forward.py`: four cases (`short`, `medium`, `padded` on hidden 64; `fastpath_off` on hidden 128, 2 head heads). Inputs from Laya's
  `Agent._encode_state` + `collate_items`; the stages from hooks on the encoder plus a hand-run head mirroring `DecisionModel.forward`,
  asserted equal to a real forward call. `fastpath_off` also runs fast path on and fails above 1e-6.
- `kernels.py`: `reference_attention` vs `local_attention` / `gas_attention` at 256, 1000, 2048 (16 heads, dim 64, block 128, 4 selection blocks); fail above 1e-5.
- `cli.py`: one command `golden <action>` (`weights-inventory|export|kernels|tokenizer`), `--out`, `--checkpoint`, `--lengths`.

## Decisions

- **Deterministic tokenizer.** The conftest tokenizer is trained and the WordPiece trainer breaks frequency ties differently per run
  (vocabularies differed in about 9% of entries between two builds), so ids and the embedding shape were not reproducible. The goldens use the same
  word list, specials, normalizer and pre-tokenizer with an explicit vocabulary (`common._deterministic_tokenizer`). The conftest is not modified.
- **Size.** Weights are shared per hidden size (`weights-h64.safetensors`, `weights-h128.safetensors`), not copied per case.
  Vendorable (< 5 MB): `inventory.json`, `weights-h64`, `short`, `medium`, `padded`. Optional: `fastpath_off` + `weights-h128` (3.6 MB more). Kernel goldens are
  inherently large (about 5 MB at L=256, 130 MB total) and stay git-ignored.
- `meta.json` drops `_name_or_path` (a temp path) from the encoder config to stay byte-identical.
- `laya:006` adds `fixture-h<hidden>/{encoder/config.json,rl_agent_config.json}` beside each weights file (the 64-hidden one belongs to the vendorable core, the 128-hidden one to the optional `fastpath_off` set).
