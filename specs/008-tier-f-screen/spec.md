# Feature Specification: Tier F Screen of Candidates A1 and B1 (laya:008, "S1")

Cross-reference as `laya:008`. `laya:003` stays reserved for ContractNLI, `laya:007` is S0 (exact local kernel), `laya:009` is compare-raya, `laya:010` is proposed for `cpu-rescore`. Bare "008" is never used (raya:008 is the bench harness). Governing document: gate plan v2 (Research Lead, 2026-10-07), whose text the orchestrator commits as the v2.0 pre-registration before any S1 gate run.

**Feature Branch**: `008-tier-f-screen`
**Created**: 2026-10-07
**Status**: Draft (offline pieces only; no run of any kind is scheduled by this spec's first PR)
**Depends on**: `laya:007` merged for the F4 reference (optimized native) and for gate runs. The offline pieces here (candidate variants, mask-only references, plan v2 emitter, parity drawing) do not need it.

**Input**: Tier F of plan v2: two training-free candidates, one frozen configuration each, screened on dev:
- **A1**: the 10 global-attention layers of the encoder become block-local (block 128, three-block span, `experiments/kernels/local.py` semantics), with the question header's global tokens kept global.
- **B1**: the decision head's two self-attention layers attend only to the global tokens plus a +/-128 window around each. The encoder stays native.
- Rescue configurations A2 (block 256) and B2 (window +/-256) are declared and **offline-implemented as parameters only**; whether they run is decided by the plan v2 rule (dev point estimate of agreement in [90%, 95%) at 4K or 8K and not dropped by a point estimate below 90% at 2K, 4K or 8K; dropping wins).
- No int8, no training, no other candidate.

## Scope of this spec's first PR (offline) and what is held

Done offline in the first PR: candidate attention definitions with mask-only references and real CPU paths (US1, US2), plan v2 JSON emitter and the final-split length rule (US3), fresh parity subset drawing function (US4). **Held until the human schedules runs** (each its own task, listed in tasks.md): native re-scoring in the gate dtype, A1/B1 dev and calibration scoring on the GPU, fresh parity runs on CPU and GPU, latency, F1 to F4 reporting code (summary/report2 extensions), C-ref comparison.

## User Scenarios & Testing

The user is the researcher deciding whether Raya should support a compressed architecture; the Research Lead who owns the gate plan reads the verdicts.

### User Story 1 - Candidate A1 (P1)

A1 makes the 10 global layers attend, per query, to the keys of its own and the two neighbouring 128-blocks plus every global token, and lets every global-token query attend to all keys. Everything else (the 18 local layers, the head) is native.

**Information flow** (to be written in plan.md section "Information flow"): a document token sees at most 384 neighbouring tokens plus the global tokens in each global layer; tokens farther apart than that exchange information only through the global tokens (hub-and-spoke) across the 10 global layers; the global tokens see everything. The 18 local layers are unchanged (+/-64).
**Executed operations** (CPU path): per global layer, block-local attention over a 3-block key span plus the global-token keys (de-duplicated where they fall inside the span), and one dense row block for the global queries. Work grows linearly in length plus |G| x L; the dense L x L product is not formed.

**Acceptance**:
1. The CPU path output equals the mask-only reference (dense SDPA with the same pattern as a boolean mask) within 1e-5 on the fixture and on random tensors: padded tails, ragged batch, header larger and smaller than a block, no global token, all tokens global.
2. The pattern is not native and not the Block3-only pattern: tests assert it differs from both on suitable inputs.
3. Variants `a1` (CPU path, CPU-only) and `a1_mask` (mask-only; any device, diagnostic for GPU quality) change exactly the layers the audit reports as global; the record lists them and the configuration (block, global-token rule).

### User Story 2 - Candidate B1 (P1)

B1 restricts keys of both head attention layers to S = G plus all positions within 128 of any position in G. Queries are unchanged.

**Information flow**: head outputs are read only at the option-marker positions and at the CLS position (`h[:, 0]`). Because those positions are in G, their outputs depend only on keys in S, so computing the head only on the positions in S gives the same marker and CLS outputs as the masked dense head. Positions outside S would otherwise be attended to by nobody who is read, so they are never needed. **Executed operations** (CPU path): gather S, run both layers' attention and FFNs on that compact sequence only; the encoder, the type embedding, the scorer and the act head run as native.

**Acceptance**:
1. The compact path equals the mask-only reference (the head's dense layers with the key restriction as a mask) at marker and CLS outputs within 1e-5 on the fixture, with padded rows and a window that clips at the sequence ends.
2. Final logits and `act_logits` of the DecisionModel match between the two paths within 1e-5.
3. Variants `b1` and `b1_mask` as for A1; the layers replaced are the head layers.
4. Honest cost note recorded in the spec (not decided here): the head is 2 of 30 transformer layers, so F4's 20% gain at 8K is unlikely to be reached by B1 alone; measuring that is a gate-run item.

### User Story 3 - Plan v2 JSON and the final-split rule (P1)

`freeze-plan` can emit plan v2 as JSON: tiers, thresholds (F1 95%, F2 -2 points, F3 +0.02, F4 20%, bootstrap 5,000 resamples), comparisons, candidate configurations, rescue rule, stop conditions, parity rule (margin 0.01, 20 cases, +4,096), the final-split length rule, `final_scored_items: 0`, a fingerprint. The v1 file (`evaluation_plan.json`, fingerprint `58ea2c38f100d4f6`) is never overwritten.

**Acceptance**:
1. The emitter writes `evaluation_plan_v2.json` and `.md`, with `version: 2`, `version_reason: "compression gate tiers E/F/Q"`, and the thresholds above taken from one constants block (a test reads the constants from the JSON, not the markdown).
2. It refuses when any final item appears in the results (FR-030), and when a v2 file already exists unless asked for a new version with a reason.
3. The tier rule: at 4,096 and 8,192 tokens the **final** split runs all 200 cases; dev and calibration keep the recorded half-sample (60 cases at 4K and 8K on dev). Tested on a fixture split. No final item is scored.
4. The plan text itself (the markdown the orchestrator commits) is an input only through an optional `--plan-md` path whose sha256 is recorded in the JSON.

### User Story 4 - Fresh parity subsets, drawn by function (P1)

A function draws two disjoint 20-case sets (A, B) from dev cases that are not in the 20-case variant sample, from a fixed seed, deterministic and independent of any measured result. Items are `distractor@mid`, at 512 / 2,048 / 8,192 and 4,096 (approved).

**Acceptance**:
1. The same seed gives the same sets; A and B are disjoint, both disjoint from the variant sample, both inside dev, each balanced across the four workflows as far as 20 allows (5 per workflow, reported).
2. The function and a CLI subcommand exist and are tested on a fixture; **it is not run on the real splits in this PR and no case id is committed**. The orchestrator schedules the pre-registration commit (v2.0) that records seed and ids.

## Requirements

- **FR-001**: Candidates are applied outside `laya/` (`laya_diff_empty: true`), as variants in `experiments/variants.py` backed by `experiments/candidates.py` and `experiments/kernels/` (new `block_global.py`), reversible in tests.
- **FR-002**: The global-token rule is one function of the input ids and the tokenizer's special ids: [CLS], the question tokens and the option [MASK] markers (see Clarifications). It never reads gold labels or the state text.
- **FR-003**: Every mask-only variant has a CPU path twin; both are proven equivalent on the fixture (tolerance 1e-5). A mask-only variant is labeled a quality diagnostic; only a CPU path may support an F4 claim.
- **FR-004**: Rescue configurations differ from the primary only by one numeric parameter (block 256, window 256), declared in the plan JSON, registered as `a2`/`a2_mask`/`b2`/`b2_mask`; no run is scheduled by this spec.
- **FR-005**: Nothing scores a final-split item. `eval` still refuses `--split final`.
- **FR-006**: Offline tests in `tests/experiments/`; `pytest -m "not slow"` passes; real-checkpoint checks are `slow` and not run until runs are permitted.
- **FR-007**: Every run (later) records device and dtype per cell, from a clean worktree of the merged commit with `git_dirty: false` (laya:007 plan, item 10).

## Success Criteria (of the offline PR)

- **SC-001**: All equivalence tests pass (A1 and B1, mask-only vs CPU path, 1e-5).
- **SC-002**: `freeze-plan` v2 output validates against the constants and carries `final_scored_items: 0`; v1 untouched.
- **SC-003**: The tier rule test shows 200 final cases at 4K/8K and the half-sample on dev.
- **SC-004**: `git diff main -- laya/` is empty; no run was executed.

## Clarifications

Resolved from the brief and plan (2026-10-07):
- Q: Rescue rule and drop rule order? A: Dropping wins (plan v2 section 3 and stop condition 2).
- Q: 4K cell? A: dtype fp16 and a parity cell, approved by the human ('approve atlas stuff').
- Q: Where does the plan JSON come from? A: constants in `experiments/evalplan.py`; the markdown is referenced by hash only.

Answered by the Research Lead and approved by the human (standing approval 'approve atlas's answers', 2026-10-07):
1. **Global-token set = (b)**: [CLS], the question tokens (between [CLS] and the first [SEP]) and each option [MASK] marker. Option description tokens, both [SEP] tokens and the state are not global (plan v2.0 section 3, literal wording). Whole-header-global is an unfunded ablation, not part of this gate. (The default of an earlier draft, the whole header, was wrong and is replaced.)
2. **Parity seed 2026100801.** Draw: pool = dev cases in `splits.json` `half_sample.dev` minus `variant_sample` (40 cases, 10 per workflow); within each workflow order by `sha256("2026100801:" + case_id)`; first 5 to set A, next 5 to set B. Offline, deterministic, no model. Script and ids are committed (with their sha256) before any gate run; if the real splits cannot meet plan section 4 the failure is reported and the rule is not changed.
3. **B1 stays in the F4 comparison** (measured, not asserted). The B1 CPU path is built offline here because it is small and tested; running it is conditional on B1 not being dropped on F1 (plan section 2).
4. **Plan text**: the pre-registration markdown (`gate-plan-v2.0-prereg.md`, sha256 `2580c62a2d9e79387fd4e79d1212099c03852785d441cad6da99e97b808b5dc8`) is committed unchanged as `docs/gate-plan-v2.md`; the emitter records its sha256.
