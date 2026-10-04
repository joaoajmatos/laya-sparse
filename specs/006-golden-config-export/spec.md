# Feature Specification: Golden Config Export (laya:006)

**Extends laya:004** (golden fixtures), which implements **raya:002**. Cross-reference as `laya:006` / `laya:004` / `raya:002`. `laya:003` stays reserved for ContractNLI.

**Feature Branch**: `raya/golden-config-export` (local)
**Created**: 2026-10-04
**Status**: Implemented (see tasks.md)
**Depends on**: `laya:004`.

**Input**: Scout finding (raya side): the vendored `fixture-tiny` config says `vocab_size: 100`, while the golden weights (`weights-h64.safetensors`) have an embedding of 231 rows.
Cause: the `laya:004` generator exports the weights and a `meta.json` per case, but never writes the checkpoint's own config files; the hand-made `raya/tests/data/fixture-tiny/` config
(vocab 100) was never produced by the generator, so it can disagree with the weights.

## User Scenarios & Testing

The user is the Raya implementer writing the weights loader and config types, who needs the exact config the golden weights were produced with.

### User Story 1 - Config beside the weights (P1)

`python -m experiments golden export` also writes, for each fixture size, the encoder `config.json` and `rl_agent_config.json` exactly as the fixture checkpoint carries them
(`<out>/fixture-h64/encoder/config.json`, `<out>/fixture-h64/rl_agent_config.json`, and the same for `fixture-h128`), next to `weights-h64.safetensors` / `weights-h128.safetensors`.

**Acceptance**:
1. `fixture-h64/encoder/config.json` has `vocab_size` equal to the number of embedding rows in `weights-h64.safetensors` and to the `vocab_size` in the `laya:005` tokenizer goldens' `meta.json`; likewise `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `intermediate_size`.
2. `rl_agent_config.json` matches the `agent_config` recorded in each case `meta.json`.
3. Every case `meta.json` names its config directory (`config_dir`).
4. A repeated export is byte-identical, and every `meta.json` carries the laya-sparse commit SHA.

## Requirements

- **FR-001**: Extend `experiments/golden/forward.py:export_case`; no new CLI command.
- **FR-002**: The config files are byte-for-byte copies of the fixture checkpoint's own files (what `laya.Agent` loads), not a re-serialization.
- **FR-003**: Written once per fixture size (64, 128), next to the weights file that was produced from that checkpoint.
- **FR-004**: No file under `laya/` changes. The conftest fixture is unchanged. Deterministic: same seeds, fp32, eval, one thread.
- **FR-005**: Offline pytest in `tests/experiments/test_golden.py`.

## Success Criteria

- **SC-001**: Config vocab, weights and tokenizer goldens agree (test). **SC-002**: Two exports identical bytes. **SC-003**: Vendorable core set stays under 5 MB (the config files add a few KB).

## Clarifications (resolved by default, 2026-10-04)

- Q: Copy or re-serialize? A: Copy the checkpoint's files; re-serializing could drift from what Laya reads. Their bytes are deterministic (verified by the repeat-run test).
- Q: Layout? A: `fixture-h<hidden>/encoder/config.json` and `fixture-h<hidden>/rl_agent_config.json`, mirroring a checkpoint directory (the weights file stays at `weights-h<hidden>.safetensors`, unchanged, to keep the `laya:004` layout and every existing path stable).
- Q: Tokenizer? A: Unchanged; it stays in the `laya:005` output. The test ties the three together.
- Q: Does this change the golden tensors? A: No; tensors and weights are produced by the same code path, only meta.json gains `config_dir` (and the SHA changes with every commit).
