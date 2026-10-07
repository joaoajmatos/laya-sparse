# Feature Specification: Exact Local-Attention Kernel (laya:007, "S0")

Cross-reference as `laya:007`. `laya:003` stays reserved for ContractNLI, `laya:009` for `compare-raya`. This is step S0 of the compression-gate plan v2 (Research Lead, 2026-10-07; adopted by the human) and the Tier E check of that plan. `laya:008` (S1, Tier F screen) depends on it: its F4 cost reference is the optimized native defined here.

**Feature Branch**: `007-local-exact-kernel`
**Created**: 2026-10-07
**Status**: Draft
**Depends on**: Phase 1/2 tooling on `main` (357ad53): `experiments/variants.py` (R14, R17), `experiments/kernels/`, `experiments/latency.py`, p2-dev native CPU predictions.

**Input**: Feasibility roadmap step 2. The English checkpoint has 28 encoder layers: 18 local (sliding window, |i-j| <= 64, `config.sliding_window = 64`) and 10 global. The audit verdict is that the local layers run as dense masked attention (L x L scores, mask applied): the window saves nothing. Replacing them with an exact banded kernel should cut the 8K CPU time (about 75 s to about 40 s, *an estimate*) with no retraining and no change in outputs.

**Superset trap.** `experiments/kernels/local.py` (the 85 ms kernel) is the *three-block span with block 128*: every key in the query's own and the two neighbouring 128-blocks, up to 255 positions each side. That is a superset of the native +/-64 window (raya `docs/kernel-notes.md`). Using it for the native local layers would change outputs and is not Tier E. S0 therefore needs a new, exact kernel; `local.py` stays untouched and is only a performance reference.

## User Scenarios & Testing

The user is the researcher (and the Research Lead who wrote the gate plan), who needs to know whether native Laya can be made cheaper on the i7-8700 with provably identical outputs, and needs the resulting "optimized native" as the honest baseline for every later cost comparison.

### User Story 1 - Exact banded kernel (P1)

A kernel computes attention for query i over keys j with |i-j| <= 64 and j inside the row's valid length, in work that grows linearly with length, and matches the dense masked reference.

**Acceptance**:
1. Matches the dense reference within 1e-5 on random tensors, at lengths that do and do not divide the block, with padded tails, batch > 1, rows of different length.
2. Keys at distance exactly 64 are attended and at 65 are not (tested at block edges).
3. A query with no allowed key returns zeros, never NaN (repo convention); padded query rows match native.
4. Not the Block3 pattern: a test asserts the kernel differs from `local.py` on inputs where the two patterns differ.

### User Story 2 - Variants on the real checkpoint (P1)

`local_exact` (exact kernel in the 18 local layers, head fast path as native) and `local_exact_fastpath_off` (the same plus the head fast path off: the **optimized native**) are selectable wherever `--variants` is accepted (`eval`, `latency`), applied outside `laya/`, recorded in the run manifest.

**Acceptance**:
1. On the tiny fixture and on the real checkpoint, applying the variant replaces exactly the layers the audit reports as local, and no others; the record lists them.
2. Layer-level equivalence: each replaced layer's output matches the native layer on the same hidden states within 1e-5 (fixture always; real checkpoint in `slow` tests), 512 to 8,192 tokens, padded tails included.
3. Variants are CPU-only (`unsupported` on GPU, as the other optimized variants).

### User Story 3 - Tier E verdict (P1)

A single report states E1, E2, E3 with numbers.

**Acceptance** (plan v2 Tier E; all must hold, none loosened):
- **E1**: probabilities within **1e-5 absolute** of native fp32 on CPU at 512, 1,024, 2,048, 4,096 and 8,192 tokens, 20 items per length (items named in the report); per local layer max abs difference reported. If the maximum is just above 1e-5 from summation-order noise, report the observed maximum and stop.
- **E2**: **100% identical predicted answers** on the CPU parity subset (20-case dev variant sample, `distractor@mid`, 512 / 2,048 / 8,192: 95 / 100 / 100 items) against the p2-dev native CPU predictions.
- **E3**: `local_exact_fastpath_off` p50 not slower than `fastpath_off` native at every length; p95 reported.

### User Story 4 - F4 reference latency (P2)

Latency of optimized native vs raw native (and `fastpath_off`) at 512 / 1K / 2K / 4K / 8K on the 12-item latency sample, i7-8700, fp32, same protocol as Phase 1 `latency`. This table is the F4 reference for `laya:008`.

## Requirements

- **FR-001**: New kernel in `experiments/kernels/local_exact.py`, with its semantics defined against `reference.py` through a new `band` pattern (half-width 64) in `MaskSpec`, so the reference and dense masked paths cover the same pattern.
- **FR-002**: Variants `local_exact` and `local_exact_fastpath_off` added to `experiments/variants.py` (`VARIANTS`, `unsupported_reason`, `apply_variant`, `load_for_variant`). The window half-width is read from the loaded encoder config, never hard-coded; a model with no local layers is refused with a clear error.
- **FR-003**: `laya/` unchanged (`laya_diff_empty: true` recorded). Nothing persists across variants in one process (reversible in tests).
- **FR-004**: Every number is labeled with device and dtype (CPU fp32 here). No GPU run. Runs record the manifest and go under `experiments/results/<run-id>/`.
- **FR-005**: Any gate run starts from a clean git tree (`git_dirty: false` in the manifest) with the commit SHA recorded. No final-split item is scored (`final_scored_items: 0`); only dev cases and goldens are used.
- **FR-006**: A dry-run estimate of CPU hours per batch is printed before any batch. No batch above 1 CPU hour starts without the orchestrator's go-ahead; CPU jobs never run in parallel.
- **FR-007**: Offline tests in `tests/experiments/` on the fixture; real-checkpoint checks marked `slow`.
- **FR-008**: A report (run summary plus `specs/007-local-exact-kernel/results.md`) with E1, E2, E3 verdicts, exact commands and test output.

## Success Criteria

- **SC-001**: E1, E2, E3 each resolve to pass, or to a reported failure with the observed numbers; no criterion is changed after seeing data.
- **SC-002**: The optimized-native latency table (5 lengths, p50 and p95) exists for `laya:008`.
- **SC-003**: `pytest -m "not slow"` passes and `git diff main -- laya/` is empty.

## Clarifications (resolved from the brief and plan, 2026-10-07)

- Q: Variant names? A: `local_exact` and `local_exact_fastpath_off` (= optimized native). `fastpath_off` stays as is; E3 compares against it.
- Q: Reuse `local.py`? A: No, it is Block3 (superset). The new kernel is separate; `local.py` is a cost reference only.
- Q: Which 20 items per length for E1? A: The 20 dev cases of the fixed variant sample, `distractor@mid`, first question of each case in sorted order, at each length; ids listed in the report. Dev only.
- Q: E1 lengths 1,024 and 4,096 are not in the parity subset: items? A: The same 20 cases rebuilt at those lengths by the existing family builder.
- Q: Native reference for E1? A: Computed in the same run, same device and dtype (fp32 CPU, fast path off, so the head path is identical). p2-dev native predictions (fast path on) are used for E2 only, as the brief says (R17: fast path off equals native within 5e-7).
- Q: Cost? A: About 3 CPU hours in total (estimate). Batches over 1 h (E2, about 1.3 h) need the human's go-ahead.
