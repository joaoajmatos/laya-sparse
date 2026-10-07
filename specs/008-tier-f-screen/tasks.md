# Tasks: Tier F Screen of A1 and B1 (laya:008)

Offline pieces first (T001-T015, no run of any kind). Held tasks (T016-T024) need laya:007 merged and/or the human's go for runs; none starts without the orchestrator.

- [x] T001 spec.md, plan.md, checklists/requirements.md
- [x] T002 `candidates.global_tokens(ids, special_ids, mode)`: header and markers modes, missing second [SEP] = all valid, padded rows; tests
- [x] T003 `kernels/block_global.py` (A1 CPU kernel) and `block_global_allowed` pattern; tests vs dense mask: padded tails, ragged batch, header smaller/larger than a block, no G, all G, differs from native and from Block3
- [x] T004 A1 variants `a1`, `a1_mask`, `a2`, `a2_mask` (global layers only; reversible; context hook); fixture test: layers replaced = audit global layers, CPU path equals mask-only within 1e-5, `a1` with all-G equals native
- [x] T005 B1 pattern S (`b1_keep`), manual pre-norm head layer, mask-only and compact paths, re-implemented `DecisionModel.forward` for B1; tests: layer function equals the native layer, forward equals native with S = everything, compact equals mask-only at marker/CLS outputs and logits, window clipping at ends, padded rows
- [x] T006 B1 variants `b1`, `b1_mask`, `b2`, `b2_mask`; registered in `variants.py` (`VARIANTS`, `unsupported_reason`: non-mask CPU-only, mask-only any device)
- [x] T007 `evalrun.eval_condition`: apply mask-only variants after loading a GPU agent (CPU fixture test with a stand-in device flag; no GPU used)
- [x] T008 `evalrun.select_items`: all `final` cases at 4,096 and 8,192, half-sample elsewhere; fixture-split test; `eval` still refuses `--split final`
- [x] T009 `evalplan.PLAN_V2` + `freeze_plan_v2` + CLI `freeze-plan --plan-version 2` (+ `--plan-md`): thresholds from constants, fingerprint, `final_scored_items: 0`, refusals, v1 untouched; tests
- [x] T010 `parity_draw.draw_fresh_sets` + CLI `parity-draw`: determinism, disjointness, outside the variant sample, 5 per workflow, error on shortfall; tests on a fixture split; run once offline on the real splits per plan s.4 (ids committed in parity-ids.json, no model)
- [x] T011 (written, NOT run) `slow` real-checkpoint checks written, not run: A1 and B1 CPU path vs mask-only on the real layers at 512 and 2,048 tokens
- [x] T012 AGENTS.md layout and commands lines; `pytest -m "not slow"` green; `git diff main -- laya/` empty
- [ ] T013 Checkpoint to the orchestrator: clarifications needing a human, cost estimate
- [ ] T014 PR (offline only, labeled "no runs"), Sextant review loop (author posts each round), no merge
- [ ] T015 Report PR URL, SHA, test output
- [x] T016 (laya:007 merged, e7d4487) compose `a1`/`b1` CPU variants with optimized native (fast path off + `local_exact`); tests
- [ ] T017 (held) F1 to F4 reporting: agreement with clustered bootstrap lower bound, F2, F3 (temperatures from calibration), F4 against optimized native, near-tie strata, failure slices, C-ref at matched p50 (summary.py, report2.py)
- [ ] T018 (held, GPU go) re-score native in the gate dtype on dev and calibration; A1, B1 on dev and calibration (fp32 <= 2K, fp16 4K/8K)
- [ ] T019 (held, v2.0 pre-registration by the orchestrator first) ids already drawn and committed (parity-ids.json); fresh parity runs per candidate (CPU and GPU), 4K cell included
- [ ] T020 (held, CPU go, idle machine) latency and peak memory: A1, B1, optimized native, raw native, original-length reference
- [ ] T021 (held) rescue configurations A2/B2 only if the plan rule triggers; report either way
- [ ] T022 (held) `phase2-report`-style gate report for S1
- [ ] T023 (held, after S1, orchestrator) freeze v2.1 with dev-selected configurations and calibration temperatures
- [ ] T024 (held) final-split scoring is S3 and not part of this spec
