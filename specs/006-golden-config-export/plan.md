# Implementation Plan: Golden Config Export (laya:006, extends laya:004)

**Spec**: [spec.md](spec.md).

- `experiments/golden/forward.py:export_case`: in the existing "first time for this weights file" block, copy `<checkpoint>/encoder/config.json` and
  `<checkpoint>/rl_agent_config.json` from the fixture checkpoint (`fx.path`, the temp directory the case was built in) to `<out>/fixture-h<hidden>/...`.
  Add `config_dir="../fixture-h<hidden>"` to the case `meta.json`.
- Tests (`tests/experiments/test_golden.py`): files exist; `vocab_size` equals the embedding rows of the weights file and the `laya:005` tokenizer goldens' `vocab_size`;
  hidden size / layers / heads / intermediate size match the weights; `rl_agent_config.json` equals the `agent_config` in `meta.json`; every case names its `config_dir`.
  The existing repeat-export test already covers byte identity of every file under the output directory, the new ones included.
- Regeneration: from a clean worktree of the committed branch tip, twice, compared by sha256 (see tasks.md).
