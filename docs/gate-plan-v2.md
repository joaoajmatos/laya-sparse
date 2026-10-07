# Compression gate: plan v2.0, pre-registration text (Research Lead, 2026-10-07)


> **Human decision, 2026-10-07 (quoted by the orchestrator):** 'adopt Atlas's CPU design and start laya:009 after 007. also, we wont do any test runs today'. Recorded effect: proposal A (margin-gated CPU rescoring, tau 0.01, about 6 CPU h on dev for 3 conditions, validation shared with S1) is ADOPTED as the definition of a CPU number; laya:009 compare-raya starts after laya:007 merges; NO runs of any kind today (this overrides every cost or run-batch item in these files for 2026-10-07). Not covered: funding of dev rescoring runs beyond this adoption, S2, S3, target CPU/precision, Phase 2 audit, ContractNLI timing, any raya:008 run on the i7-8700.
Status: **v2.0 pre-registration text, content final as of 2026-10-07 (clarifications of S1 questions Q1 to Q3 applied).** The orchestrator commits it unchanged before any S1 gate run; any later change is a new numbered version with its reason, never an edit in place. The v2.1 freeze (JSON twin, see section 9) follows after S1 and before any final-split item is scored. No open proposals remain in this text: every approval it relies on is quoted in section 0 or below. On 2026-10-07 the human said 'approve atlas stuff', which covers exactly: the 4K fp16 + 4K parity cell, and the S0 (~3 CPU h) / S1 (~3-4 GPU h, ~11-13 CPU h incl. the 4K cell) cost estimates. It does NOT cover S2 training, S3 final scoring, rescue configs A2/B2 (conditional on the section 3 rule, each run batch needs the orchestrator's go-ahead), target CPU/precision, the Phase 2 audit verdict or ContractNLI timing.

Supersedes nothing: plan v1 (`experiments/results/p2-dev/evaluation_plan.json`, fingerprint `58ea2c38f100d4f6`, laya-sparse run p2-dev)
stays as recorded. v2 is a new version with `version_reason: "compression gate tiers E/F/Q"`.

## 0. Decisions already taken by the human (copied, not re-decided)

Provenance: the first block below is the orchestrator's reading, applied as recommended, of the human's verbatim words of 2026-10-07: 'ok apply all recommended stuff from atlas, assign some work to lantern as well. and make sure ferrules PR is merged'. It lists my proposal's recommended decisions; **nothing in it is described as human-approved beyond that wording**. Other verbatim quotes held: 'approve atlas stuff' (the 4K cell and the S0/S1 cost estimates only) and 'adopt Atlas's CPU design and start laya:009 after 007. also, we wont do any test runs today'. Human decision, verbatim 'I approve' (2026-10-07), given after the orchestrator listed three open items; **orchestrator's reading, applied conservatively**: (1) E2 reference = option (a): reuse the dirty-tree p2-dev CPU fp32 native predictions as the E2 equivalence reference only, labelled dirty-tree; (2) S0 CPU cost approved up to about 4.2 h, with the fastpath_off latency staying in S0. It does NOT approve the raya constitution amendment (a separate item). Nothing else changed.

- Three tiers E / F / Q. F1 = top-label agreement with native, case-clustered lower bound >= 95%.
- Q margin 5 points is the *build* gate; the 2-point claim is also reported and called inconclusive unless it resolves.
- Parity: GPU fp32 up to 2K tokens, fp16 at 8K (4K fp16 added and approved separately, see section 1); margin-aware rule with margin 0.01, frozen BEFORE running on 20 fresh dev cases per candidate.
- Funded: S0 (exact kernel) and S1 (Tier F screen) with exactly two candidates: **A** global layers made local, **B** decision head restricted to marker windows.
  int8 stays OUT. S2 (training) is DEFERRED. Final split scored once, all 200 cases at 4K and 8K.
- Not decided here and not mine: target CPU and precision, the human Phase 2 audit verdict, ContractNLI (laya:003) timing.

## 1. Hardware and numbers policy

- **Quality** is scored on the RTX 4060 (`.venv-gpu`): fp32 at 512, 1,024 and 2,048 tokens; fp16 autocast at 4,096 and 8,192 **(4K = fp16 plus a 4K parity cell, +about 1.5 CPU h; approved: human, 'approve atlas stuff', 2026-10-07, relayed by the orchestrator)**. GPU quality is called CPU-equivalent only for the cells whose fresh parity check passes (section 4).
  Existing bf16 GPU predictions in p2-dev are NOT reused for any gate number. The p2-dev CPU fp32 native predictions are used only as the reference for the Tier E equivalence check (E2): they come from a dirty-tree run (`laya/` unchanged, `laya_diff_empty: true`, experiments code dirty), and the new `local_exact` predictions must come from a clean tree. Reuse for E2 only is approved (see the provenance note in section 0: 'I approve', 2026-10-07, orchestrator's reading); a clean-tree native re-run (about 2.7 h CPU) is not required.
- **Latency and peak memory** are measured only on the i7-8700 (fp32, 6 threads, same protocol as Phase 1 `latency`). The cloud Xeon and the GPU never supply a cost figure.
- Every number in the gate report carries: repo, commit SHA, run id, device, dtype. Code must run from a **clean tree** with `laya/` unchanged (`laya_diff_empty: true`); p2-dev ran from a dirty tree and this plan does not accept that.

## 2. Tiers, thresholds, comparisons

Device rule for F1/F2/F3: scored on the GPU in the gate dtype (section 1) for candidate and reference alike, and reported as CPU-corrected under the adopted Proposal A (near-tie items re-scored on the i7-8700 at fp32). On the GPU the reference is plain native in the same dtype: fastpath_off is a CPU setting and `local_exact` is a CPU variant, and both are prediction-identical to native (E1/E2), so plain GPU native stands in. F4 (cost) is CPU only, against optimized native.`n`nReference for every comparison: **optimized native** = `fastpath_off` plus, once S0 passes, the exact +/-64 local kernel. Both are prediction-identical to native (fastpath: p2-dev, 0 changed predictions in 150 items;
exact kernel: S0 criteria), so accuracy and agreement are computed against native fp32/fp16 on the same device. Latency F4 uses the optimized-native latency.

### Tier E (exact transform; here: the exact +/-64 local kernel, plus fastpath off)

Pass = all of:
- E1: per local layer and end to end, probabilities within 1e-5 absolute of native fp32 on CPU, at 512, 1,024, 2,048, 4,096 and 8,192 tokens.
- E2: 100% identical predicted answers on the CPU parity subset (the original 20 dev cases, `distractor@mid`, 512 / 2,048 / 8,192; 95 / 100 / 100 items).
- E3: i7-8700 p50 not slower than fastpath-off native at any length; the speed is reported, not gated, because a Tier E change needs no quality study.

### Tier F (training-free approximations; candidates A and B, one frozen configuration each)

Per candidate and per length in {512, 1,024, 2,048, 4,096, 8,192} (and original length for the short-context regression):
- **F1 fidelity:** top-label agreement of candidate vs optimized native, same device, same dtype, same items; case-clustered percentile bootstrap, 5,000 resamples; **lower end of the 95% interval >= 95%**.
- **F2 accuracy:** lower bound of (candidate - native) accuracy >= -2 points (the v1 margin), same bootstrap.
- **F3 calibration:** temperature-scaled classification ECE (temperatures fitted on the calibration split, one per condition) no more than 0.02 above native's. The 0.02 is a proposal with no power analysis; ECE is reported raw and scaled and F3 cannot rescue a failed F1/F2.
- **F4 cost:** i7-8700 p50 at least 20% below optimized-native p50 at the same length, and p95 not worse. The candidate must be a CPU implementation that actually skips work; a mask-only prototype is a quality diagnostic, not an F4 result.
- **F4 for candidate B1 is measured, not asserted.** The head is 2 of 30 layers. Estimate from Phase 1 run `full` and the variant run (head 37% of 75.2 s at 8K with the fast path on; fast path off saves about 17 s, so the head keeps about 10.6 s, of which about 9 s is attention and about 1.5 s linear work at the measured ~270 GFLOP/s): against optimized native (about 40 s at 8K, an estimate) B1 can gain at most about 20-26% at 8K and about 10% at 4K, so B1 alone will probably miss F4 at 4K. B1 stays in the F4 comparison and the number is reported either way. **Ordering rule (saves builder and CPU time):** a candidate's CPU latency path is built and measured only if the candidate is not dropped on dev F1 (stop condition 2); F1 to F3 are computed first on the GPU.
- **Component-only outcome.** If a candidate passes F1 to F3 but misses F4 only, the report says "quality-preserving component, not a standalone pass". A combination of A1 and B1 is a different candidate, is not pre-registered here, and needs a new human decision.
- **Candidate verdict.** A candidate "passes at a length" if F1, F2, F3 and F4 hold there. Raya support is justified for a candidate only if it passes at **4,096 and 8,192** and does not fail F1 or F2 at 512, 1,024, 2,048 or original length (all of F1 to F3 are evaluated and reported at every length 512 / 1K / 2K / 4K / 8K). If it fails only at lengths up to 2K, the verdict is "usable above N tokens", reported as such, not as a pass.

### Tier Q (retrained): defined, not funded (S2 deferred)

Q0 competent reference (lower bound > 0 vs majority at the gated length); Q1 short-context preservation vs the fine-tuned checkpoint (74.5%); Q2 long-context non-inferiority vs same-length native and vs a matched-adaptation native control, at 2K/4K/8K separately;
Q3 not dominated by the best cheaper baseline (today retrieve1024 49.5% at 8K, 4.6 s CPU; retrieve512 48.2%, 2.2 s); Q4 at least 2x faster than optimized native at 4K and 8K on the i7-8700. Margin M = 5 points as the build gate; the 2-point claim is reported alongside and labeled inconclusive unless its interval resolves.

### Comparisons list (frozen)

| id | comparison | lengths | split | scored |
|---|---|---|---|---|
| C-E | exact-kernel vs native (probabilities, predictions) | 512..8K | goldens + CPU parity subset | S0 |
| C-A | candidate A vs optimized native: F1, F2, F3 | 512, 1K, 2K, 4K, 8K, original | dev (selection), then final (confirmation) | S1, S3 |
| C-B | candidate B vs optimized native: F1, F2, F3 | same | same | S1, S3 |
| C-lat | A, B, optimized native, raw native: p50/p95/peak memory (F4) | same | 12-item latency sample | S0 (reference), S1 |
| C-par | fresh parity per candidate (GPU vs CPU) | 512, 2K, 8K (+4K, approved) | 20 fresh dev cases each | S1 |
| C-ref | candidate vs retrieve1024 / retrieve512 at matched p50 (reported only; gates Q3, not F) | 2K..8K | dev, final | S1, S3 |

No other comparison is a gate. Anything else is exploratory and labeled so.

## 3. Candidate definitions (one primary configuration each; fixed now)

- **A1 (primary):** the 10 global-attention layers of the English checkpoint use the block-local pattern (block 128, three-block span, the existing `experiments/kernels/local.py` semantics) with the following positions kept global (they attend to, and are attended by, every token): the [CLS] token, the question tokens (the `<type> question: <instructions>` text between [CLS] and the first [SEP], i.e. `head_ids` in `build_sequence`), and each option's [MASK] marker token. **Not global:** the option description tokens that follow each marker, the two [SEP] tokens, and all state tokens. (Layout: `[CLS] <type> question: ins [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]`, `laya/common.py` `build_sequence`.) Rationale: this is the literal reading of "question, option-marker and CLS positions" (research plan item 7: existing positions only, no extra tokens), the markers are the positions the head reads, and the option descriptions sit next to their marker so local layers already carry them to it; marking them global would add up to `head_max_len` (192) global tokens and test a different, larger candidate. Whole-header-global is an unfunded ablation, not part of this gate. The 18 native local layers stay as they are. The decision head stays native.
  Information flow, initialization and executed operations must be written by the builder in the spec (research plan item 7); CPU path uses the existing block-local kernel.
- **B1 (primary):** the decision head's two self-attention layers attend only to keys in {the [CLS] token, the question tokens, and the option [MASK] markers (the same set as A1)} plus a +/-128 token window around each such token (the windows take in the nearby option description tokens and state tokens). All other document tokens do not act as keys. Queries at the marker positions are unchanged; head FFNs follow the builder's declared plan. The encoder stays native.
- **Rescue configurations A2 (block 256) and B2 (window +/-256)** exist **only** if the primary configuration is not dropped under stop condition 2 and its dev agreement **point estimate** is in [90%, 95%) at 4K **or** at 8K (either length). Dropping wins over rescuing: a point estimate below 90% at any of 2K, 4K or 8K drops the candidate even if another length is in the rescue range. (F1 itself passes on the case-clustered lower bound of 95%; the rescue and drop triggers use point estimates.) They are declared now to prevent post-hoc invention; at most one rescue each, and the multiplicity cost is stated in the report (dev selection, final confirmation once). Never tune windows on the final split or on the fresh parity sets.

## 4. Parity rule for gate numbers (frozen before any run)

- Margin-aware rule from laya-sparse PR #2 (merge `c10f207`) with margin **0.01**, recorded here before the fresh runs. A CPU/GPU disagreement counts as failure only if the CPU top-two probability margin on that item exceeds 0.01.
- Fresh sets: for each candidate, 20 dev cases that are not in the original variant sample, disjoint between A and B, `distractor@mid`, at 512, 2,048, 4,096 (approved) and 8,192. **Seed 2026100801, fixed here before any case id is drawn.** Draw procedure (deterministic, version-independent): pool = dev cases that are in `splits.json` `half_sample.dev` and not in `variant_sample` (40 cases, 10 per workflow; dev split fingerprint `9fe123db65aa5de37b9431ccb394fbc165a2286805f907a804cfc76f6c9f5580`); within each workflow order the pool by `sha256("2026100801:" + case_id)`; the first 5 go to set A and the next 5 to set B (5 per workflow, 20 cases per set). The pool equals the 40 cases that are needed, so the seed only decides the A/B split; using the half-sample keeps every fresh case inside the dev grid at 4K and 8K, so their CPU labels are pre-paid rescoring under adopted Proposal A. The draw script is committed with the ids and their sha256 before any gate run, and ids are never re-drawn after results are seen.
- Also required: minimum 20 counted items per cell; strict-rule results are computed and reported next to the margin-aware ones; the verdict word stays "undetermined" under the min-sample guard.
- "Fresh" means not used to derive the 0.01 margin; the cases are still development cases and not independent of dev tuning of the retrieval/window baselines.
- A candidate's GPU numbers at a length become "CPU-equivalent" only if that length's parity cell passes. If a cell fails, that length's gate verdict is reported as GPU-labeled and unverified, and the gate does not pass at that length.

## 5. Analysis plan

1. Per condition and length: accuracy, agreement with native, ECE raw and scaled, ordinal MAE (secondary), clustered bootstrap intervals (5,000 resamples, seed fixed), tables like report2.
2. F1 resolution (estimate): at 4K/8K the dev set has 60 cases (300 items). With about 98% true agreement and a design effect of about 2 the lower bound is about 95.8%; with 97% it is about 94.3%. So **F1 on dev can pass only if true agreement is about 97.5% or better**; this is intended, and a borderline dev result is reported as inconclusive, not rounded up.
3. F2 resolution on the 200-case final split: with per-case SD 0.10 about 196 cases suffice for a 2-point margin at 80% power; with agreement of 97% the per-case SD is about 0.08-0.11 (estimate from the disagreement rate), so F2 is borderline-resolvable on 200 cases. This is the SD that governs F2 for a candidate that passed F1; the SD of 0.19-0.26 in the proposal (E7, int8, agreement 64-72%) applies only to low-agreement or retrained models, and it is the reason the Q-tier 2-point claim stays inconclusive on 200 cases. A final F2 comparison whose interval straddles -2 is "inconclusive", not a pass.
4. Multiplicity: two candidates x six lengths are reported without correction; the gate decision rule requires all of its lengths to pass (a conjunction), which is more conservative than any single test. Dev picks a configuration; the final split confirms it once.
5. Cost figures: p50 of 12 items per length after 2 warmups, drift canary from Phase 1, no run in parallel with other CPU jobs; background load recorded (the i7-8700 has shown load in earlier sweeps).
6. Failure slices (reported, not gating): per workflow, per variant (`neutral@mid`, `distractor@begin/mid/end`), per question type; and agreement stratified by CPU top-two margin (near ties vs not).

## 6. Stop conditions (declared before data)

1. **Training-free stop:** if neither A nor B reaches F1 at 4K and 8K on dev (primary plus allowed rescue), Tier F produces no compression candidate; Raya builds only Tier E and Raya-L. Report the negative result with the numbers.
2. **Candidate stop (takes precedence over the rescue rule in section 3):** a candidate is dropped on dev if its agreement point estimate is below 90% at any of 2K, 4K, 8K, or if F4 shows less than 20% gain on the i7-8700 at 8K with a real CPU implementation.
3. **Speed stop:** a passing-quality candidate with less than 2x gain over optimized native at 4K and 8K is noted as "not worth Raya support alone"; it can still ship in Laya.
4. **Parity stop:** a length whose fresh parity cell fails is unverified; two or more failing cells for a candidate remove its GPU numbers from the gate (CPU rescoring would be a new costed request).
5. **Undecidable:** if the optimized-native reference itself does not clear majority at a length (today 2K and above), the Q branch cannot be evaluated on this benchmark there; report it, do not lower a bar.
6. **Selection wins:** if retrieve1024/retrieve512 baselines beat or match a candidate at equal or lower p50 at 4K/8K, report "selection, not compression" for that candidate.
7. No stop condition is changed after any final-split item is scored.

## 7. What is written before any final item is scored

1. Two commits, owned by the orchestrator (the builder generates the JSON): **v2.0 pre-registration**, before any S1 gate run: this plan text, thresholds, configurations A1/B1, rescue and drop rules, margin 0.01, and the parity seed (2026100801) with its draw procedure (the case ids are a deterministic output of the committed draw script and are committed with their sha256, still before any gate run); and **v2.1 freeze**, after S1 and before any final item: the JSON twin (`evaluation_plan.json` v2) with fingerprint, `frozen_at`, `final_scored_items: 0`, the dev-selected configurations and the calibration temperatures.
2. Configuration of A1, B1 (and the rescue rule), code commit SHA with a clean tree, and the seed, draw script and case ids of the fresh parity sets.
3. Margin 0.01, thresholds (F1 95%, F2 -2, F3 +0.02, F4 20%), bootstrap seed and resamples.
4. Temperatures fitted on the calibration split only (scored for both candidates and native); written to the run before final scoring.
5. The analysis scripts committed. The final-split tier rule changed from half-sample to **all 200 cases at 4,096 and 8,192** (a code change to the length tier rule in `splits`/`select_items`), recorded in the plan.
6. The dev-selected configuration per candidate, with the dev results that selected it, linked.
7. The lock: the existing final-test lock (FR-030) stays; the plan v2 fingerprint is recorded in each final run manifest.

## 8. Sequence and cost (estimates scaled from measured p50s; builder time extra)

| Step | GPU (RTX 4060) | i7-8700 CPU | Notes |
|---|---|---|---|
| S0 exact kernel (laya:007; not raya:007) | about 0 | about 3 h estimate, approved up to about 4.2 h incl. fastpath_off latency ('I approve', 2026-10-07, orchestrator's reading) | probabilities 20 items x 5 lengths (about 0.7 h), parity-subset predictions about 1.3 h (native CPU predictions reused from p2-dev), optimized-native latency about 0.5 h, layer tests about 0.3 h |
| S1 Tier F screen (laya:008; not raya:008, the bench harness) | about 3-4 h (fp32 <=2K and fp16 >=4K; native re-scored in the same dtype; A1, B1, calibration split; rescue configs add about 1.5 h only if triggered) | about 11-13 h (fresh parity: about 4.2 h per candidate-set incl. native, about 8.5 h for two; 4K cell +1.5 h, approved; latency about 1 h per candidate incl. original-length reference) | the earlier "about 2 GPU h, 14 CPU h" assumed bf16 reuse; re-scoring native in fp32/fp16 is why GPU is higher |
| Freeze plan v2 | 0 | 0 | after S1, before S3 |
| S3 final (NOT funded) | about 7-8 h: 3 gate conditions (native, A1, B1; about 4.8 h at the measured bf16 p50s, about 6-7 h if fp32 at 2K and below is about 3x slower, an unmeasured assumption) plus 2 reported-only C-ref baselines (retrieve1024, retrieve512; about 0.7 h) = 5 conditions | about 3-4 h latency/parity, plus CPU rescoring under adopted Proposal A (about 18 h for 3 conditions at tau 0.01, about 33 h at 0.02; unfunded) | scored once |

## 9. Handoffs

- The JSON twin of this plan is produced by extending `experiments/evalplan.py` (`freeze-plan`); that extension is part of S1 (brief in `quartz-briefs.md`).
- The orchestrator commits the plan (v2.0 before S1 gate runs, v2.1 after S1 and before S3) and records the fingerprint; workers do not edit the orchestrator's roadmap note.`n- Spec numbers, always with the repo prefix: laya:007 = S0, laya:008 = S1, laya:009 = compare-raya (the laya-sparse half of raya:008; starts after laya:007 merges per the human 2026-10-07), laya:010 proposed for `cpu-rescore`; laya:003 reserved for ContractNLI. Bare "008" is never used.
















