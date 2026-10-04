# Feature Specification: Decision Benchmark and Practical Baselines (Research Phase 2)

**Feature Branch**: `002-decision-benchmark-baselines` (specification directory only; no branch created)

**Created**: 2026-09-29

**Status**: Draft

**Input**: User description: "Phase 2 of docs/research-plan.md, built on upstream Laya's published data. Build the evaluation data and run native Laya and the simple baselines, reporting quality, calibration, latency and memory at 512 to 8K tokens. No fixed latency budget: report the curves. Raya is a separate future project that may reuse what is learned here."

## Context

Phase 1 measured cost only, on synthetic inputs with no labels. It found that native Laya takes seconds per decision on the target CPU. Attention is 45% of time near 2K tokens and 74% at 8K. Even with free attention, 8K costs about 20 s on this CPU. Phase 1 could not say whether long context is *worth* that cost, because it never measured decision quality.

This phase answers that question for Laya, using the data upstream Laya publishes:

- **Upstream `typed-decisions` dataset** (Apache 2.0, synthetic, English). Four workflows: customer service, invoice processing, security incidents, and agent-trace observability. Each has 300 training cases and 100 test cases. Each case has a structured state, several typed questions (`choice`, `noul`, ordinal `score`), gold labels, and label confidence.
- **Upstream `laya-typed-decisions` checkpoint**. It was fine-tuned from Laya on that dataset's training split, and upstream reports 0.766 accuracy on the test split against 0.362 for the base English checkpoint. It has the same architecture as the Phase 1 checkpoint, with a configured input cap of 1,024 tokens. It is already pinned in this fork.

Upstream cases are short: a state is a structured record of a few hundred characters. They show what Laya inputs look like in practice, but they cannot test 4K to 8K quality on their own. Long inputs are therefore built from upstream cases under controlled rules, and every conclusion about long context is limited to that construction.

## Clarifications

### Session 2026-09-29

- Q: How should the 400 upstream test cases be divided into development, calibration, and final-test splits? → A: 30% dev / 20% calibration / 50% final (120 / 80 / 200 cases), stratified by workflow (30 / 20 / 50 per workflow)
- Q: How should "low-confidence" gold labels be defined? → A: Bottom quartile of upstream label confidence, computed once on the imported data, with the cutoff value recorded in the data manifest
- Q: Which per-item outcome counts as correct for accuracy? → A: Exact match with the gold label for all question types; ordinal level error is reported separately
- Q: How many questions per case does each evaluation item ask, given single-question latency? → A: One forward pass per question; every question of a case is scored and timed separately, and primary latency is per question
- Q: What retained-token budgets should the lexical retrieval baseline be tuned over? → A: 512, 1,024, and 2,048 retained tokens, tuned on the development split only

### Session 2026-09-30

- Q: The full grid is about 134 hours of CPU quality scoring; how should quality scoring be run? → A: Score quality on the GPU, keep CPU for latency, memory and a fixed parity subset; GPU-scored quality stands in for CPU quality only if the parity check passes (FR-024). Constitution principle VI concerns latency and cost measurements, which stay on CPU.

## User Scenarios & Testing *(mandatory)*

The user is the researcher deciding whether long context is worth its cost for Laya, and which Phase 3 directions (sparse attention, selection, compression, or none) the evidence supports. CPU is the primary deployment target. GPU is a secondary target and is reported alongside CPU, never in its place.

### User Story 1 - Upstream Data Intake and Length Profile (Priority: P1)

The researcher imports the pinned upstream dataset and gets a record of exactly what was imported. The record also gives the token-length distribution of real upstream inputs as Laya sees them, and how many cases each native checkpoint can solve at their original length. This is the first measured answer to "how long are Laya's inputs, and does the native model solve them?"

**Why this priority**: Every later result depends on this data. The length profile decides how much the long-context question matters in practice. The solvability check decides which checkpoint is a valid quality reference: a model that cannot solve the short case cannot show a context-processing failure.

**Independent Test**: Import the dataset and check that case counts, splits, workflows, and question types match the upstream source, and that token counts reconcile for every case. Run both checkpoints on the original-length cases of the development split and confirm the results come with accuracy, calibration, and per-workflow breakdowns.

**Acceptance Scenarios**:

1. **Given** the pinned upstream dataset, **When** it is imported, **Then** a data manifest records source, revision, license, subsets, splits, case counts, question counts by type, and a content fingerprint. A later import that does not match the fingerprint is refused.
2. **Given** the imported cases, **When** they are profiled, **Then** the report gives each case's total input tokens under Laya's tokenizer, including state, questions, options, and special tokens. It also gives the distribution per workflow (min, median, p90, p99, max) and the share of cases that fit within 512, 1,024, 2,048, 4,096, and 8,192 tokens.
3. **Given** the fine-tuned checkpoint and the base English checkpoint, **When** each runs on original-length development cases, **Then** accuracy, calibration error, and per-workflow and per-question-type results are reported for each. The checkpoint that serves as the quality reference is the one that solves the short cases.
4. **Given** cases whose gold labels have low recorded confidence or annotator agreement, **When** results are reported, **Then** those cases are also reported separately, so label noise is not mistaken for model error.

---

### User Story 2 - Controlled Long-Context Families and Frozen Splits (Priority: P2)

The researcher builds evaluation items at 512, 1,024, 2,048, 4,096, and 8,192 total tokens from upstream test cases. The target case keeps its gold labels, and the extra context is built so it cannot change the correct answer. The researcher also gets disjoint development, calibration, and final test splits that are frozen before any baseline is compared.

**Why this priority**: Without controlled long inputs, the question "does quality hold at 8K?" cannot be asked with upstream data. Without frozen, disjoint splits, any conclusion can be tuned into existence. This story depends on Story 1's intake.

**Independent Test**: Generate the families. Confirm every item's total length is exactly the named length, that each item's target case, evidence position, and construction rule are recorded, and that no case appears in two splits. Hand-audit a sample of items to confirm the added context does not change the correct answer.

**Acceptance Scenarios**:

1. **Given** an upstream test case, **When** its length variants are built, **Then** each variant contains the unchanged target case plus added context, reaches exactly the named total length, and records the evidence span of the target case in the final input.
2. **Given** the length families, **When** they are built, **Then** at least three families exist:
   - **neutral padding**: added text unrelated to the decision
   - **near-matching distractors**: other cases from the same workflow, clearly marked as not under review
   - **position**: the target case at the beginning, middle, and end of the context
3. **Given** added context taken from other upstream cases, **When** it is chosen, **Then** it comes only from upstream *training* cases, so no test-split case appears as a distractor inside another item.
4. **Given** the upstream test cases, **When** splits are made, **Then** cases are divided into development, calibration, and final test splits by case, and all length, position, and family variants of a case stay in the same split.
5. **Given** frozen splits, **When** this phase runs, **Then** no final-test item is scored by any model. The final-test split is created, fingerprinted, and left unopened for Phase 4.
6. **Given** a hand audit of at least 50 generated items spread across families and lengths, **When** an item's added context changes the correct answer or makes it ambiguous, **Then** the construction rule is fixed and the families are regenerated. The audit result is reported.

---

### User Story 3 - Native and Baseline Quality–Cost Curves on CPU and GPU (Priority: P3)

On the development split, the researcher runs native Laya at each matching length alongside the simple alternatives: truncation, windowing, cheap retrieval, oracle evidence selection, the decision-head fast-path fix, and CPU quantization. For each, at each length, the researcher gets quality, calibration, latency, and memory on the target CPU, plus latency on the GPU. This shows which simple option already gives most of the quality at a fraction of the cost.

**Why this priority**: This is the central measurement of the phase, and the research plan says it can falsify the need for a complex attention prototype. It depends on Stories 1 and 2.

**Independent Test**: Run every baseline on the development split at every supported length. Confirm each result row carries quality, calibration, latency, memory, and visible-evidence records, and that unsupported or failed conditions appear with reasons.

**Acceptance Scenarios**:

1. **Given** the development split at each length, **When** the baselines run, **Then** the following each report quality and cost, with what each actually saw:
   - native Laya at the matching length
   - native Laya truncated to 512 tokens and to its configured input cap
   - the existing windowed long-document path
   - cheap lexical retrieval followed by native Laya
   - oracle evidence selection followed by native Laya
2. **Given** a baseline that drops or selects context, **When** it runs, **Then** each item records how much of the target case's evidence was visible to the model, so a quality loss can be traced to lost evidence.
3. **Given** the decision-head fast-path fix and CPU quantization, **When** they are measured, **Then** they are reported as optimized-system variants, separate from the architectural baselines. Their quality and calibration change against native is stated, including when it is zero.
4. **Given** latency for any baseline, **When** it is collected, **Then** single-document, single-question CPU latency is the primary figure, timed from tokenization through postprocessing with no reused document representations, as in Phase 1. GPU latency is reported beside it and labeled as GPU.
5. **Given** retrieval budgets or window settings, **When** they are tuned, **Then** they are tuned on the development split only, and calibration temperatures and thresholds are fitted on the calibration split only.
6. **Given** a length a baseline cannot run (out of memory, time limit, positional limit), **When** results are assembled, **Then** the condition appears as failed or unsupported with its reason, and is never replaced by a truncated or other-device result.

---

### User Story 4 - Phase 2 Report and Frozen Evaluation Plan (Priority: P4)

The researcher receives one report with quality-against-cost curves for every baseline at every length on CPU and GPU, and a verdict on whether long context improves Laya's decisions on this data. The report also says which Phase 3 directions the evidence supports, and gives the frozen protocol and sample sizes for final evaluation.

**Why this priority**: It turns the measurements into the Phase 3 decision. It depends on all earlier stories.

**Independent Test**: Check the report against the raw results. Every curve point and every claim traces to a result file, every claim is labeled measured, estimated, or hypothesized, and the final-test protocol is complete before any final-test item is scored.

**Acceptance Scenarios**:

1. **Given** completed development results, **When** the report is written, **Then** it shows accuracy and calibration against CPU latency, and against GPU latency, for each baseline at each length, with paired uncertainty that treats questions on the same case as dependent.
2. **Given** the native curve across lengths, **When** the report is written, **Then** it states whether quality at 2K, 4K, and 8K is better than, equal to, or worse than truncation to 512 and to the configured cap, or whether the comparison is inconclusive at this sample size.
3. **Given** the simple baselines, **When** the report is written, **Then** it states whether any baseline reaches native-matching-length quality within the research plan's -2 percentage-point margin at lower cost, and names it.
4. **Given** the whole phase, **When** the report is written, **Then** it lists the Phase 3 directions the evidence supports, weakens, or leaves open, and states the limits of synthetic data and constructed long contexts.
5. **Given** pilot paired differences, **When** the evaluation plan is frozen, **Then** it states the final-test metrics, quality margin, comparisons, and required sample size. If the upstream test split is too small to resolve the margin, the report says so, rather than lowering the bar.

---

### Edge Cases

- **Distractors that change the answer.** Added context could change or blur the correct decision (for example, a distractor case the model confuses with the target). The hand audit catches it, the construction rule is fixed, and affected items are regenerated. Such items are never kept with their original labels.
- **Target case longer than the smallest length.** When a target case plus questions exceeds 512 tokens, that length is unsupported for the case and recorded as such, not truncated.
- **Many options.** Upstream notes that options share a fixed head budget. Cases whose options exceed that budget are recorded and reported separately.
- **Baselines that cannot see the evidence.** When truncation or retrieval drops the target case's evidence, the item stays in the results with evidence-visible set to false.
- **Low-confidence gold labels.** Upstream labels carry a confidence value. Cases in the bottom quartile of label confidence are reported separately and are never dropped silently.
- **Upstream changes.** If the upstream dataset or checkpoint changes after pinning, the pinned revision governs. A mismatch refuses the run.
- **Evaluation independence.** The fine-tuned checkpoint was trained on the upstream training split, and upstream has already reported results on the test split. The report states both facts: the test split is not a never-inspected confirmatory set in the strict sense.
- **GPU differences.** GPU and CPU outputs differ in low-order bits, so an answer near a tie can flip. Quality is scored on the GPU for the full grid and on CPU for a fixed parity subset; the parity result decides whether GPU-scored quality is reported as CPU-equivalent (FR-024). Latency and memory always come from the device they name.
- **Windowed probabilities.** The windowed path's probabilities are not calibrated for the whole document, and upstream documents this. Its calibration is reported but not presented as a native-comparable probability.

## Requirements *(mandatory)*

### Functional Requirements

**Data intake and profiling**

- **FR-001**: The tooling MUST import the upstream `typed-decisions` dataset at a pinned revision and record source, revision, license, subsets, splits, case counts, question counts by type, and a content fingerprint.
- **FR-002**: The tooling MUST refuse to proceed when imported data does not match the pinned fingerprint.
- **FR-003**: The tooling MUST NOT commit raw upstream records, per constitution principle V. Derived items and manifests MUST be regenerable from the pinned source and recorded seeds.
- **FR-004**: The tooling MUST report each case's total input length under Laya's tokenizer, including state, questions, options, and special tokens. It MUST also report per-workflow distributions and the share of cases within each of 512, 1,024, 2,048, 4,096, and 8,192 tokens.
- **FR-005**: The tooling MUST record each case's gold labels, label confidence or agreement, question types, and option counts, and MUST report low-confidence cases separately. Low-confidence means the bottom quartile of upstream label confidence, computed once on the imported data, with the cutoff value recorded in the data manifest.

**Checkpoints**

- **FR-006**: The quality reference MUST be the pinned fine-tuned `laya-typed-decisions` checkpoint, unless Story 1 shows it cannot solve original-length development cases. In that case the report MUST say so, and later quality comparisons MUST be labeled as lacking a valid reference.
- **FR-007**: The pinned base English checkpoint MUST be evaluated on original-length development cases for continuity with Phase 1, and MUST NOT be used as the long-context quality reference when it fails the short-case check.
- **FR-008**: Inputs above a checkpoint's configured input cap but within positional capacity MUST be run and labeled as beyond the configured cap, as in Phase 1.

**Length families and splits**

- **FR-009**: The tooling MUST build evaluation items at exactly 512, 1,024, 2,048, 4,096, and 8,192 total tokens, as defined in Phase 1, for at least the neutral-padding, near-matching-distractor, and position families.
- **FR-010**: Each item MUST keep its target case unchanged and record the target case's evidence span in the final input, its family, its position, its length, and the seed and rule used to build it.
- **FR-011**: Added context taken from upstream cases MUST come only from the upstream training split, and MUST be marked in the input as not under review.
- **FR-012**: The tooling MUST divide upstream test cases into development, calibration, and final test splits by case, keep every variant of a case in one split, and fingerprint each split. Splits MUST be 30% development, 20% calibration, and 50% final test (120 / 80 / 200 cases), stratified by workflow (30 / 20 / 50 cases per workflow).
- **FR-013**: Final-test items MUST NOT be scored by any model in this phase.
- **FR-014**: At least 50 generated items across families and lengths MUST be hand-audited for answer changes or ambiguity before baseline comparisons, and the audit result MUST be reported.
- **FR-015**: Option order MUST be counterbalanced where the question type allows, and construction MUST NOT create label leakage through padding, identifiers, or position.

**Baselines**

- **FR-016**: The tooling MUST evaluate on the development split, at each supported length:
  - native Laya at the matching length
  - native Laya truncated to 512 tokens and to its configured input cap
  - the existing windowed long-document path
  - lexical retrieval followed by native Laya, with retained-token budgets of 512, 1,024, and 2,048 tokens
  - oracle evidence selection followed by native Laya
- **FR-017**: The tooling MUST evaluate the decision-head fast-path fix and CPU quantization as optimized-system variants, reported separately from the architectural baselines.
- **FR-018**: Each result MUST record, per item, the tokens the model actually saw and whether the target case's evidence was fully, partly, or not visible.
- **FR-019**: Retrieval budgets and window settings MUST be tuned on the development split only. Calibration temperatures and thresholds MUST be fitted on the calibration split only.
- **FR-020**: Oracle evidence selection MUST be labeled a diagnostic control and MUST NOT be ranked as a deployable option.

**Quality, calibration, cost**

- **FR-021**: Each baseline condition MUST report accuracy per question type, where an item is correct only on exact match with the gold label for every type (including ordinal `score`), and mean absolute level error for ordinal questions. It MUST also report classification calibration error computed from the predicted answer's probability, per constitution technical constraints.
- **FR-022**: Quality comparisons MUST include paired uncertainty estimates that treat questions on the same case as dependent, and MUST label underpowered comparisons as inconclusive.
- **FR-023**: Each question of a case MUST be run as its own forward pass, scored and timed separately. Each baseline condition MUST report single-document, single-question latency (p50, p95) and peak memory on the target CPU as the primary cost, with the Phase 1 timing rules.
- **FR-024**: Each baseline condition MUST report latency on the GPU where it runs, labeled as GPU and never substituted for CPU latency or memory. Quality (accuracy and calibration) MAY be scored on the GPU, because scoring the full grid on CPU takes days (research.md R11); every quality result is then labeled with its scoring device. A CPU parity check on a fixed subset of items MUST compare CPU and GPU predictions and report the share of identical predicted answers, the accuracy difference, and the largest probability difference. GPU-scored quality stands in for CPU quality only when the check passes (at least 98% identical predicted answers and an accuracy difference of at most 0.5 points); otherwise the report says GPU quality is unverified. *Amended 2026-10-03 (human-approved: on 2026-10-03 the researcher instructed implementing the margin-aware criterion with a margin of 0.01; `parity-margin.md`):* the check passes when no CPU/GPU disagreement has a CPU top-two probability margin above 0.01 (disagreements at or below it are reported as tolerated near-tie flips, with their margins); the original 98% / 0.5-point result is still computed and reported next to it as `strict_passed`. The margin is a named constant and a CLI option, not a measured property; it was chosen from the same flips it tolerates.
- **FR-025**: Failures, out-of-memory events, time-limited runs, and unsupported lengths MUST appear in results with their reason, and MUST NOT be replaced by results from another length, device, or configuration.

**Reporting**

- **FR-026**: The report MUST show, per baseline and length, quality and calibration against CPU latency and against GPU latency.
- **FR-027**: The report MUST state, for 2K, 4K, and 8K, whether native quality is better than, equal to, or worse than truncation, or inconclusive.
- **FR-028**: The report MUST state whether any baseline reaches native matching-length accuracy within -2 percentage points at lower CPU cost.
- **FR-029**: The report MUST label every statement as measured, estimated, or hypothesized. It MUST state the limits of synthetic upstream data and constructed contexts, and MUST include commands to reproduce every result.
- **FR-030**: The report MUST freeze the final-test protocol (metrics, margin, comparisons, sample size) before any final-test scoring. When the upstream test split cannot resolve the margin, it MUST say so.
- **FR-031**: Generated items, results, model weights, and caches MUST be written to ignored locations and MUST NOT be committed.

### Key Entities

- **Data Manifest**: The pinned upstream source: revision, license, subsets, splits, counts, and fingerprint.
- **Upstream Case**: One upstream record: workflow, state, typed questions with options, gold labels, label confidence, and upstream split.
- **Length Profile**: Per-case token counts and per-workflow length distributions under Laya's tokenizer.
- **Evaluation Item**: A target case placed in constructed context at one named length. It records family, position, evidence span, construction seed and rule, and its split.
- **Split Manifest**: The frozen assignment of upstream test cases to development, calibration, and final test, with fingerprints.
- **Baseline Condition**: One method (native, truncation, windowing, retrieval, oracle, fast-path fix, quantization) at one length on one device.
- **Quality Result**: Per-item predictions, probabilities, correctness, and evidence visibility for one baseline condition, with aggregate accuracy, level error, calibration, and paired uncertainty.
- **Cost Result**: Latency and memory for one baseline condition on one device, with status (measured, unsupported, failed, partial).
- **Evaluation Plan**: The frozen final-test protocol: metrics, margin, comparisons, sample size, and the power statement.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A second researcher can reproduce the data manifest, length profile, splits, and every evaluation item from the pinned source and recorded seeds, and fingerprints match.
- **SC-002**: The report states the share of upstream cases within each named length, per workflow, from measured token counts.
- **SC-003**: Every evaluation item reaches exactly its named length, 100% of items record their evidence span, and no case appears in more than one split.
- **SC-004**: The hand audit covers at least 50 items, and the share of items whose correct answer the construction changed is reported and is 0% after any regeneration.
- **SC-005**: Every baseline has a result at every length on the development split, either measured or marked unsupported, failed, or partial with a reason. At 4,096 and 8,192 tokens the development result covers a fixed, recorded half-sample of development cases (see Assumptions).
- **SC-006**: Every measured baseline condition reports accuracy and calibration error (with the device that scored them), CPU p50 and p95 latency, peak memory, and GPU latency where it runs. The CPU parity check of FR-024 is reported.
- **SC-007**: The report gives a better, equal, worse, or inconclusive verdict for native quality against truncation at 2K, 4K, and 8K, with uncertainty.
- **SC-008**: Zero final-test items are scored in this phase, and the frozen evaluation plan exists before Phase 4 begins.
- **SC-009**: The researcher can decide from the report alone whether Phase 3 should pursue sparse attention, selection, compression, or none, with each recommendation tied to a measured quality–cost result.

## Assumptions

- The user is a single researcher. The target CPU is the Phase 1 machine (Intel i7-8700, 16 GB RAM), and the GPU is its RTX 4060. Both are recorded in the manifest.
- No fixed latency budget applies. The report presents curves, and a budget can be chosen from them later. Laya is mainly CPU-first, and GPU results inform the same decisions without replacing CPU results.
- Raya is a separate future project. This phase studies Laya only. Its findings may inform Raya, but no Raya design choice is made here.
- Upstream `typed-decisions` data is synthetic and short. Long-context conclusions hold only for the constructed families, and the report says so. A realistic long-document set is out of scope for this phase and is listed as a gap.
- Upstream public benchmarks (AG News, SST-5, Banking77, MASSIVE, XNLI, BoolQ) have short inputs, and whether any of them were in the base checkpoint's training data is not established. They are out of scope here.
- The decoder baseline in the research plan (Qwen3-0.6B) is out of scope: this fork has no decoder scoring path. It is listed as a deferred baseline.
- Sampling at long lengths: 512, 1,024, and 2,048 tokens run on every case of a split. Because native inference at 4,096 and 8,192 tokens takes tens of seconds per question on the target CPU, those lengths run on a fixed half-sample of each split, stratified by workflow and recorded in the split manifest; paired comparisons always use identical cases. The optimized-system variants (FR-017) run on a fixed 20-case development sample at 512, 2,048, and 8,192 tokens; on 2026-09-30 the user cut that run to a 10-case subset of the sample, spread evenly over the workflows (`eval --variant-cases 10`), to save CPU time, and the CPU parity subset keeps all 20. Sample sizes are reported with every result.
- Realistic tasks: the constitution (principle VII) asks length studies to include realistic tasks alongside controlled examples. This phase has controlled families only, on synthetic upstream data. That is a documented deviation, and its effect on interpretation is stated in the report.
- Training of any kind is out of scope. Compression, model-size, and distillation experiments are Phase 3 or later.
- The upstream test split has 400 cases (100 per workflow). Splits are 120 development, 80 calibration, and 200 final-test cases. A statement that the final test may be underpowered is expected, not a failure.
- The ~33 ms historical reference is upstream's T4 GPU measurement (32.8 ms for the multilingual checkpoint, 39.5 ms for English, one question). It is a GPU figure and is not compared against CPU results.
- Dependencies: Hub access to the pinned dataset and checkpoints, and the Phase 1 tooling and timing rules, which this phase reuses. The constitution (v1.1.0) governs.
