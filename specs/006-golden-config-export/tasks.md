# Tasks: Golden Config Export (laya:006)

- [x] T001 Spec with clarifications (spec.md)
- [x] T002 `export_case` copies the encoder `config.json` and `rl_agent_config.json` per fixture size; `config_dir` in the case `meta.json`
- [x] T003 Tests: vocab and shape agreement with the weights and the tokenizer goldens, agent config equality, `config_dir`
- [x] T004 Regenerate from a clean worktree of the committed tip, twice, byte-identical, `git_dirty: false` in every `meta.json`
