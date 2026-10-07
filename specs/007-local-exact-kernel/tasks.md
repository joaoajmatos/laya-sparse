# Tasks: Exact Local-Attention Kernel (laya:007)

Offline work first (T001-T012, no gate run); gate runs (T013-T018) need a clean committed tree and, where over 1 CPU hour, the orchestrator's go-ahead.

- [x] T001 Spec with clarifications (spec.md); plan.md
- [ ] T002 `reference.py`: `band` pattern + `half_width` in `MaskSpec` (`allowed`, `n_blocks` unaffected); dense masked and reference cover it
- [ ] T003 `kernels/local_exact.py`: exact band kernel (block 64, 3-block span, exact mask, validity, zero-empty-row convention)
- [ ] T004 `test_kernels.py`: kernel vs reference/dense masked at lengths that do and do not divide 64, padded tails, batch > 1, ragged lengths, distance 64 in / 65 out at block edges, differs from Block3 `local.py` where the patterns differ
- [ ] T005 `variants.py`: `local_exact`, `local_exact_fastpath_off`; replace `forward` of local attention modules, half-width from the module, refuse a model with no local layers, reversible, CPU-only `unsupported_reason`, record lists replaced layers
- [ ] T006 `test_variants.py`: fixture model: exactly the audit's local layers replaced, global untouched, output equals native within 1e-5 incl. padded batch, restore works, GPU unsupported, unknown variant message lists new names
- [ ] T007 Key validity from the 4-D mask diagonal equals the 2-D attention mask (fixture test, padded tails; `None` mask case)
- [ ] T008 `tier_e.py` + CLI `tier-e`: `probs` (E1 + per-local-layer max diff), `compare` (E2 vs p2-dev native predictions), `report` (E1/E2/E3 verdicts, records `git_sha`, `git_dirty`, `laya_diff_empty`, device, dtype, `final_scored_items: 0`); `--dry-run` prints the batch estimate
- [ ] T009 `test_tier_e.py`: probs and layer diffs on the fixture (exact variant passes, a deliberately Block3 variant fails E1), compare counts flips, report verdict logic (E1 fail reported with observed maximum, nothing loosened), refuses dirty tree
- [ ] T010 `slow` test on the real checkpoint (cached): isolated layer equivalence 512 to 8,192 with padded tails; marked `slow`
- [ ] T011 Update `AGENTS.md` layout/commands lines for the new files and variants; `pytest -m "not slow"` green; `git diff main -- laya/` empty
- [ ] T012 Dry run: time one real item at 2,048 for native, `fastpath_off`, optimized native (fast, under 1 CPU h) and the mask-build share; refresh the plan's estimates; report to the orchestrator
- [ ] T013 (gate, clean tree) Real-checkpoint layer tests, 512-8K (about 0.3 h)
- [ ] T014 (gate) `tier-e probs` E1, split by length batches, 20 items per length
- [ ] T015 (gate, **go-ahead**) E2 predictions: `eval --variants local_exact_fastpath_off --device cpu` then `tier-e compare`
- [ ] T016 (gate) Latency `none,fastpath_off,local_exact_fastpath_off` at 512/1K/2K/4K/8K; E3 and the F4 reference table
- [ ] T017 `tier-e report`, `results.md` with E1/E2/E3, exact commands, test output, run ids, SHA
- [ ] T018 PR, review rounds with Sextant, report URL, SHA and test output (do not merge)
