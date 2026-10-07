# Research: Decision Benchmark and Practical Baselines

Decisions for the open technical questions in the plan. No `NEEDS CLARIFICATION` markers remain.
Items marked *spike* were checked on 2026-09-29 on the measuring machine and are facts about the
pinned artifacts. Items marked *to verify* are hypotheses the tooling itself must confirm, and the report must
label them as such.

## R1. Upstream data source and format

**Decision**: Read `LocalLLaMA/typed-decisions` (Apache 2.0) at the pinned dataset revision
`d51d993547ad8355b1c25157fbc1fea0649e8ffa`, using the `all` config parquet files
(`all/train-*.parquet`, `all/test-*.parquet`) through `pyarrow`. Cross-check that the per-workflow configs give the same
case ids and counts. The checkpoint's model card links this dataset by that name, and the dataset sha is what the
manifest pins (FR-001).

**Record shape** (*spike*): each row has `id`, `workflow`, `split`, `state` (JSON string), `questions` (JSON string, five
question definitions with `type`, `instructions`, `criteria`), `gold` (per question: `label`, `confidence`,
`probabilities`, and `score` or `noul`), `factors` (the generating factors), `label_agreement` (per question
`argmax_agree`, `argmax_majority`, `total_variation`) and `n_questions`. Test: 400 cases (100 per workflow),
2,000 decisions. Train: 1,200 cases.

**Fingerprint**: SHA-256 over the canonical JSON (sorted keys, no whitespace) of every row, sorted by `id`, for
each of train and test, then over the two. A mismatch with the manifest's recorded value refuses the run (FR-002).
The Hub `sha` is recorded next to it, and offline reruns read the local snapshot.

**Rationale**: `pyarrow` is the smallest way to read parquet. `datasets` and `pandas` are heavy and add nothing.

**Alternatives considered**: `datasets.load_dataset` (large dependency for four small files); downloading the JSON
from a mirror (not the pinned source).

**Use of fields**: `factors` and `label_agreement` are *never* model inputs. `factors` seeds the optional
diagnostic and audit views (which facts of the record matter), and `label_agreement` is a report stratum.

## R2. Splits

**Decision**: Seeded, stratified split of the 400 test cases by workflow: for each workflow, shuffle its 100 case ids
with the split seed and assign the first 30 to development, the next 20 to calibration and the last 50 to final
test (120 / 80 / 200 overall, clarified 2026-09-29). The split manifest stores ids, seed and a fingerprint per
split. All variants of a case (every family, position, length and question) inherit its split (FR-012).

**Rationale**: case-level assignment stops the same record from appearing as a target in two splits.
Stratifying by workflow keeps each split's workflow mix equal, so per-workflow results are comparable.

**Freeze rule**: the split manifest is written once. A second run with the same seed and source reproduces it. A
different result refuses. The final-test split's items are built (so the fingerprint exists) but no scoring
command accepts the `final` split in this phase (FR-013). The check is in code, not in convention (R11).

## R3. The evaluation item is one question row; exact length

**Decision**: An evaluation item is one *question row* of one case: (case, question, family, position, length). Each
question of a case is run as its own forward pass (clarified 2026-09-29, FR-023), so an item is the unit that is
scored, timed and given an evidence-visibility record.

**Length**: "N total tokens" means the finished model row, `[CLS] head [SEP] options [SEP] state [SEP]`, has exactly
N tokens, counted by `tokens.account` as in Phase 1. Question heads differ in length across the five questions
of a case (the *spike* saw row lengths of 124 to 597 tokens at original length), so the state is built per row:

1. Take the target record text exactly as the native path serializes it (`serialize_state(state)`), unchanged.
2. Choose context records (R5, R6) for the (case, family, position, length) and lay them out around the target
   with the framing text of R4. All questions of a case draw reference records from one ordered sequence and
   take the prefix that fits their row budget, so the sets are nested across questions and lengths.
3. Absorb the per-question head difference and the record-size rounding with a neutral **background block**
   (`[BACKGROUND NOTE]` plus filler words that are each one token), sized by tokenizing and correcting by the
   measured difference until the row hits N exactly (a few iterations at most). The background block sits at the
   far end from the target (after it for begin and middle positions, before it for the end position; two blocks
   around it for `neutral@mid`), is never inside the target record, and never lies between the target and its marker.
4. `tokens.account` verifies the row; a miss raises instead of recording a wrong length (FR-009, SC-003).

**Original-length rows**: rows longer than N at original length (max 597 tokens, *spike*) cannot exist at 512:
the item is recorded `unsupported` with reason `target_exceeds_length`, never truncated (spec edge case).
The largest question rows are counted per workflow in the length profile.

**Alternatives considered**: one state per case with unequal row lengths (fails "exactly N"); padding the
tensor (pad tokens are not context and change the cost); making all five question heads equal (would alter the
questions).

## R4. Framing, markers and the framing control

**Decision**: The state is a plain-text document of records, each introduced by a one-line marker:

```
[RECORD UNDER REVIEW]
<target record JSON, byte-identical to the native serialization>
[REFERENCE RECORD, NOT UNDER REVIEW]
<distractor record JSON>
```

Neutral padding paragraphs sit between records with the marker `[BACKGROUND NOTE]`. The question text is unchanged. Marker
strings are fixed constants in `families.py`, recorded in the item and in the manifest.

**Confound and control**: the fine-tuned checkpoint saw bare JSON states at training time. Wrapping changes the
input distribution independently of length. The **oracle** baseline (R8) is the target record with only its marker
line, and it is the *framing control*: native quality on `unwrapped original` versus `oracle` isolates the
framing effect, and every length comparison uses `oracle` (the short evidence-only counterpart, as the research
plan asks) as its short reference, not the unwrapped original.

## R5. Families and the variant grid

**Decision**: Four variants per (case, question, length), which cover the three required families (FR-009):

| Variant | Context | Target position |
|---|---|---|
| `neutral@mid` | neutral padding only | middle |
| `distractor@mid` | near-matching reference records | middle |
| `distractor@begin` | near-matching reference records | beginning |
| `distractor@end` | near-matching reference records | end |

"Position" is defined by the target's first token offset within the state: begin means the target starts within the
first 5% of state tokens, end means it ends within the last 5%, and mid centers it. The three position variants share
the same distractor set, only its order changes, so any quality difference is due to position. Only the
mid variant of neutral padding is built: neutral text is the easy case, and position is studied with distractors.
Options are counterbalanced where the question type allows (FR-015): `choice` option order is permuted
with a seed and the gold label is remapped; `noul` and ordinal `score` have a fixed order and are not permuted.

**Leakage checks (FR-015)**: distractor ids and identifiers that repeat the target's `id` or its value strings
are rejected; identifiers such as record ids, if present in upstream states, are re-drawn independently of the
label; neutral padding is generated from a vocabulary that contains no workflow terms (a filtered word list, unlike
`inputs.FILLER_WORDS`, which includes words such as "refund" and "urgent" and is *not* reused). A test asserts the
gold label distribution does not depend on the padding length or position.

## R6. Distractor selection

**Decision**: Reference records come from the upstream **training** split only (FR-011), same workflow as the target,
ranked by lexical similarity of the serialized state (BM25 over word tokens). They are drawn in seeded groups of 20,
best-ranked group first, without replacement, so "near-matching" is measured and not guessed; the sequence goes deeper
into the ranking only when a long item needs more than 20 records (an 8,192-token item holds about 25 invoice or
customer-service records). A record is skipped when it repeats the target's text or one of its identifier-like strings
(three or more digits, which also skips records that share a date). Each reference record is used at most once per item.

**Known limits, stated in the report**:

- The pinned checkpoint was fine-tuned on this training split, so it has seen the distractor records as training inputs. A model
  may treat them as familiar. This can only be observed, not removed. The `neutral@mid` variant, which has no
  training records, gives a comparison point.
- A near-matching distractor may legitimately blur the answer. R12 handles this by hand audit, and affected rules are
  fixed and regenerated (FR-014).

## R7. Evidence span and visibility

**Decision**: The evidence span of an item is the token range of the *target record text* in the final input (the
record after its marker, not the marker or neighbours), stored as `[start, end)` positions in the row's input ids, plus the same
range in state tokens. For each baseline, evidence visibility is the fraction of these tokens that reach the model:

- `full` when every evidence token is in the model's input, `partial` for some, `none` for none (FR-018).
- Truncation: from the retained token count. Windowing: `full` when at least one window contains the whole span,
  `partial` when the best window contains part of it. The deciding window's own coverage is recorded separately.
- Retrieval: from the token ids retained by the selector, mapped back to state positions.

The comparison of a baseline's quality with its visibility answers "was the loss due to lost evidence?" without
extra runs: quality is reported for `full`-visibility items and for the rest separately.

## R8. Baseline conditions

All baselines use the same target checkpoint, the same items and the same questions, and change one thing.

| Condition | What it does | Notes |
|---|---|---|
| `native` | `max_len` = the item's named length | Above the configured cap (1,024 for `laya-typed-decisions`, 512 for base English) the run is labeled `beyond_configured_max_len` (FR-008). Positional capacity is 8,192 tokens (*spike*), so 8,192 fits. |
| `trunc512`, `truncCap` | `max_len` = 512 and the configured cap | The default right cut of `build_sequence`. Target at the end is cut off, by design. |
| `window` | `agent.predict_long`, window size tuned on dev over {256, 512, default 760} (default is `max_len - head_max_len - 8`: 1,024 - 256 - 8 for this checkpoint, *spike*), stride = window // 2, `batch_size=1` above 2,048 tokens | Probabilities are the deciding window's, not calibrated for the whole document (upstream says so). Reported as such (Edge Cases). |
| `retrieve<B>` | BM25 chunk selection to a retained-token budget B in {512, 1,024, 2,048} | R9. |
| `oracle` | The target record with only its marker | Diagnostic control, never ranked as deployable (FR-020). Also the framing control (R4). |

The "matching length" in `native` uses the same model but with `max_len` raised to the item's length, exactly as
Phase 1 ran beyond-cap lengths. For the tuning rule (FR-019): window size and retrieval budget are chosen on the
dev split by accuracy, with ties broken by lower cost, and the choice is written to the baseline manifest before
any calibration-split run.

## R9. Lexical retrieval

**Decision**: Pure-Python BM25 in `baselines.py`. The state token ids are cut into fixed 64-token chunks with 50%
overlap (no use of record boundaries, so retrieval cannot exploit the framing markers). The query is the
question's instructions plus its option texts, and never the marker text. Highest-scoring chunks are kept until the
retained-token budget B is reached, then re-ordered by position and joined. The selection time is inside the
timed call (constitution VI, "selection/compression is included").

**Rationale**: BM25 is the standard cheap retriever, and it needs no model and no dependency. Chunk size and
overlap are fixed, not tuned: only B is tuned (spec clarification), so the search cannot be stretched.

**Alternatives considered**: record-boundary retrieval (uses information the deployed system would not
have); an embedding retriever (needs a second model and is a Phase 3-scale choice).

## R10. Metrics

- **Accuracy**: exact match of the predicted answer with the gold `label`, for every question type including ordinal
  `score` (clarified 2026-09-29). Reported per question type, per workflow, per baseline and length.
- **Ordinal level error**: mean absolute error between the predicted level and the gold level for `score` questions.
- **Calibration error**: classification ECE from the *predicted answer's probability* (constitution), ten
  equal-width bins. For `noul` the predicted answer is the more probable of true/false; for `score` it is the
  most probable level. Entropy is not used.
- **Temperature**: one temperature per (question type, condition) fitted on the calibration split by minimizing
  log loss, and applied to the dev-split probabilities of the same condition (FR-019). Raw ECE and
  temperature-scaled ECE are both reported. The window baseline's scaled ECE is labeled non-comparable.
- **Paired uncertainty**: a case-clustered bootstrap (resample cases with replacement, 5,000 resamples, seeded)
  of paired accuracy differences between two conditions on the same items. Questions of one case are dependent, so the
  case is the resampling unit (FR-022). A comparison whose 95% interval spans zero and the margin is labeled
  inconclusive.
- **Verdict rule (FR-027)**: "better" if the lower bound of the interval on native minus truncation is above 0,
  "worse" if the upper bound is below 0, "equal" if the whole interval lies within ±2 points, otherwise
  "inconclusive". The -2 point margin comes from the research plan (FR-028).
- **Low-confidence stratum** (R13) and **workflow** strata are reported with the same paired procedure.

## R11. Compute budget, sampling and resumability

*Estimated, to be corrected from the first measured runs.* Phase 1 measured about 1.8 s at 512 tokens, 7 s near
2K and 20 s or more at 8K per question row on this CPU, with the fast path off. **Revised 2026-09-29 by `phase2-all --dry-run`** (rates 2, 4, 9, 25 and 80 s per row at 512, 1,024, 2,048, 4,096 and 8,192 tokens): native alone is about 45 hours and all architectural baselines about 134 hours on the development split, before the calibration split, latency and variants. That is days, not the "one to two days" first written in plan.md; run it in stages (`--lengths`, `--conditions`, `--max-cases`) and correct these rates from the first measured runs. Native at 8,192 tokens over a full grid
(600 dev rows × 4 variants) would take about a day by itself. The plan therefore fixes:

- **Tier rule**: lengths 512, 1,024 and 2,048 run on every case of a split. Lengths 4,096 and 8,192 run on a fixed
  half-sample of each split's cases, stratified by workflow (dev: 60 cases; calibration: 40), chosen by the split seed
  and recorded in the split manifest. Paired comparisons always use identical cases across conditions. This changes
  no baseline's coverage: every baseline has a result at every length on the dev split (SC-005), at the
  sample size for that length.
- **Variants and calibration cost**: the optimized variants (R14) run on a fixed 20-case dev sample at 512, 2,048 and 8,192,
  enough to measure a change against native on the same items and to state it, including when it is zero.
- **Quality runs vs latency runs**: quality runs make one pass per item, resumable, one forward pass per
  question row with no batching of rows (FR-023), and `batch_size=1` for windowed calls above 2,048 tokens. Latency runs use the Phase 1 timer, a fixed latency sample of 12 dev
  items per condition (three per workflow, `distractor@mid`), warmup 1, repeats per `timing.default_repeats`
  capped for time, with a per-condition time cap as in Phase 1.
- **Resumability**: each item result is appended to `predictions.jsonl` as it finishes. A restart skips
  finished item ids. A condition that fails or times out is recorded with its reason and the run continues (FR-025).
- **Final-test lock**: the CLI's split argument accepts `dev` and `calibration` only. `final` raises a tool error
  naming FR-013, and the evaluation-plan command checks that no results file names a final-split item (SC-008).

## R12. Hand audit

**Decision**: `audit-sample` draws at least 60 items, stratified across family, length and workflow, from the
dev split (seeded, recorded), and writes an audit sheet (`audit_sheet.md` and `.json`) showing each item's target
record, the gold labels, the context layout with markers, and a rubric:

1. Do the added records or text change the correct answer for this question?
2. Do they make the answer ambiguous?
3. Is the target record's evidence intact and clearly marked?

The researcher records verdicts in `audit_result.json` through `audit-record`. An item that fails 1 or 2 triggers a
rule fix (for example a distractor filter) and full regeneration of the families, then a new audit sample. The
reported figure is the share of audited items whose answer the construction changed (SC-004). Automatic
pre-checks (label leakage, identifier reuse, exact-length, span integrity) run before the sheet is produced, so the
human review is spent on semantic changes only. The audit is a **manual gate** in the task list: baseline comparison
commands refuse to run while `audit_result.json` is absent or shows a failed item on the current families
fingerprint.

## R13. Low-confidence gold labels

**Decision** (clarified 2026-09-29): a question is low-confidence when its gold `confidence` falls in the bottom
quartile. The cutoff is computed once, on the **training** split's 6,000 question confidences, so it does not
depend on any dev, calibration or final case, and it is stored in the data manifest. The same cutoff is applied to
the test split's questions. Also reported as extra strata, without a separate threshold: `label_agreement.argmax_agree`
false, and the top quartile of `total_variation`.

## R14. Optimized-system variants and the quality reference

**Fast-path fix**: the Phase 1 `--mha-fastpath off` runtime setting, run as a labeled variant in its own run directory
(Phase 1 R17). Expected: outputs equal within float rounding, so a zero quality change, stated as such.

**Quantization** (*spike*): `torch.ao.quantization.quantize_dynamic` with `torch.nn.Linear` and `qint8` works on the
loaded checkpoint, with two limits:

- Quantizing the whole model fails in the decision head with the PyTorch `mha` fast path on: the `TransformerEncoderLayer`
  fast-path check reads `weight` from a quantized layer, which is a method (`AttributeError: 'function' object has
  no attribute 'device'`).
- Two variants run instead: **`int8_encoder`** (encoder Linear layers only, head and fast path native), and
  **`int8_all_nofast`** (all Linear layers, fast path off, which also skips the fast-path bug).

The spike measured about 0.72 s against 0.97 s for a 300-token row (about 25% faster) and moved the answer
probabilities by about 0.02 on one example. Quality, calibration and latency changes against native are the results
this phase reports. `quantize_dynamic` prints a deprecation notice (PyTorch is moving it to `torchao`); the version
used is recorded in the manifest, and the replacement API is out of scope. Quantization is CPU-only. On GPU it is
recorded `unsupported` with that reason.

**Quality reference rule (FR-006, FR-007)**: a checkpoint "solves" the short cases when its accuracy on the
original-length dev cases is above the per-question-type majority-class accuracy (computed on the training split),
with a case-clustered paired interval that excludes zero. The majority reference is the most frequent train label per (workflow, question). The upstream test accuracies (0.766
fine-tuned, 0.362 base, below the 0.461 majority prior) suggested the fine-tuned checkpoint passes and the base fails;
the run decided (R19).

## R15. Latency: CPU and GPU

**CPU (primary)**: one document, one question, tokenization through postprocessing, no reused representations, one
subprocess per condition, the Phase 1 timer, clean path only, same threads and drift-canary rules (FR-023).
Windowed and retrieval latency include window splitting and retrieval. Peak memory comes from the runner.

**GPU (labeled)**: the same latency sample on the RTX 4060 from `.venv-gpu`, native precision defaults, with
`gpu_`-prefixed result files. Baselines that run on GPU: native, truncation, window, retrieval, oracle.
Quantization is CPU-only; the fast-path switch is a CPU setting and is not repeated on GPU. GPU results are drawn
next to the CPU curves and never enter CPU rankings, the -2 point comparison, or CPU cost figures (FR-024, Phase 1 rule
7). Quality is scored on the GPU and checked against CPU on a fixed subset (R20); GPU *latency* stays a separate table.

## R16. Frozen evaluation plan and power

**Decision**: `freeze-plan` reads the dev pilot's paired case-level differences, derives the intra-case
correlation of the five questions, and computes the number of final-test cases needed to resolve a -2 point margin at
80% power with a 95% interval, by a case-clustered simulation seeded from the pilot. It writes the plan (metrics,
margin, comparisons, sample size, power statement, the decision rule of R10) with a fingerprint, before any
final-test scoring exists. If the 200-case final split cannot resolve the margin, the plan and report say so and
give the number of cases it would take, without lowering the bar (FR-030, User Story 4 scenario 5).

## R17. Testing offline

Unit tests use the Phase 1 tiny random ModernBERT fixture and a small synthetic dataset in the upstream schema
(a few cases per workflow, five questions each), written to a temp directory. They cover: fingerprint mismatch refusal,
split disjointness and stratification, exact length on every family, evidence-span integrity, train-only
distractors, truncation and windowing evidence visibility, BM25 selection, oracle content, metrics against hand-computed
values, the bootstrap on a known difference, the final-test lock, and quality-run resume. The `slow` tests need the real
checkpoints and network: they check the real dataset counts and fingerprint, that both checkpoints run on original-length
dev rows, and the quantization variants' equivalence bounds.

## R18. Items carried to the report

- The upstream test split is public and upstream reported results on it, so it is not a never-inspected confirmatory set.
- The fine-tuned checkpoint trained on the training split that supplies the distractors.
- Constructed long contexts are not naturally long documents. Conclusions apply to the four families only. Realistic long
  documents, multilingual data, and the decoder baseline (no scoring path in this fork) are gaps.
- Cross-session machine drift, as in Phase 1: the canary rules are reused, and cross-run differences carry the drift caveat.

## R19. Findings from the first real runs (2026-09-29)

- **Dataset shape.** The per-workflow parquet files store gold labels flattened (`<question>__label`, `__probabilities`,
  ...) instead of one JSON column, and train ids carry a `tr_` prefix. The import therefore cross-checks case ids and every
  shared column, and checks flattened labels against `all`'s `gold`, instead of comparing fingerprints of the two layouts.
  Counts match upstream: 1,200 train and 400 test cases, 2,000 test questions (600 choice, 600 noul, 800 score).
- **Length profile matches the spike.** Original test rows are 124 to 597 tokens (median 308, p90 382, p99 550); 98.25% of test
  cases fit within 512 tokens (customer service 93%, the rest 100%); no question overflows the head budget.
- **Solvability (development split, original length, CPU).** The fine-tuned checkpoint scores 0.745 over 600 questions against a
  per-question majority reference of 0.498 (paired lower bound +0.197): it solves the short cases and is the quality reference.
  The base English checkpoint scores 0.372 (paired lower bound -0.183): it does not, which agrees with upstream's 0.362.
- **Family construction.** All 12,600 development items pass the automatic pre-checks (exact length, target intact, span, markers,
  identifier and label leakage). At 512 tokens the distractor variants often hold no whole record and degrade to padding; the
  per-item `position` field records where the target actually sits. An 8,192-token item holds about 25 reference records.
- **int8 dynamic quantization is not benign on the real model.** On 6 original-length rows the encoder-only variant moved answer
  probabilities by up to 0.145, against about 0.02 in the one-example spike of R14. The report states the quality change; this is
  an early indication, not a result.
- **Compute.** See the revised estimate in R11: staged runs, or a scaled grid, are needed.

## R20. Quality scored on the GPU, checked against CPU (change of 2026-09-30)

**Decision**: Quality (accuracy and calibration) for the full grid is scored on the RTX 4060 from `.venv-gpu`
(`eval --device gpu`). CPU stays the device for latency, memory, the optimized variants and a fixed **parity subset**:
the 20-case dev variant sample, `distractor@mid`, at 512, 2,048 and 8,192 tokens, for `native`, `truncCap` and
`retrieve1024` (`eval --device cpu --sample variant`). Spec FR-024 was changed to allow this.

**Why**: the dry-run estimate for scoring the grid on CPU was about 134 hours (45 for native alone), which is not a
practical schedule. The same grid on the GPU is estimated at about 2.5 hours, and a one-case smoke run measured 12
conditions in 158 s on GPU against 1,091 s on CPU (mostly model loading).

**Parity rule**: for every condition scored on both devices, on the items both scored: the share of identical predicted
answers must be at least 98% and the accuracy difference at most 0.5 points. If any compared condition misses this,
the report says GPU-scored quality is unverified and does not call it CPU-equivalent. GPU and CPU differ in low-order
bits (Laya runs CUDA in autocast precision), so an answer near a tie can flip; the parity subset measures how often.

**First evidence (smoke run, not a result)**: on 12 conditions x 5 items (60 comparisons, one dev case at 2,048 and 8,192
tokens) CPU and GPU gave the same predicted answer every time; the largest probability difference was 0.0045.

**What does not change**: latency, peak memory and the int8 and fast-path variants are CPU results; quantization and the
fast-path switch are unsupported on GPU. Constitution VI is about latency and cost measurements, which stay on CPU;
the GPU never substitutes for them.

**Compute estimate (estimated, to be corrected)**: GPU quality about 2.5 hours for the architectural baselines on dev plus the
calibration split; CPU parity subset about 3.4 hours (native at 8,192 tokens is most of it); CPU latency and variants on top.


## R21. Results of the p2-dev run (2026-10-01)

All numbers are on the development split (120 cases; 60 at 4,096 and 8,192 tokens) unless stated, from `summary.json` and `report2.md` of run `p2-dev`.
No final-split item was scored.

**Parity: failed.** Nine comparisons (native, truncCap, retrieve1024 at 512, 2,048 and 8,192 tokens, 20 cases). The share of identical
predicted answers was 98.0% to 100% in every row, and the largest probability difference was 0.0078. Five rows missed the accuracy criterion (0.5
points): native at 512 (-1.1) and 8,192 (-2.0), retrieve1024 at 512 (-1.1) and 8,192 (-1.0), and truncCap at 512 (-1.1). With 95 to 100 items one
item is 1 point, so the 0.5-point criterion cannot be met by any single flipped item; the criterion is finer than the sample. The rule was written
before the run and was not changed. GPU-scored quality therefore stays labeled unverified, as R20 says. A future spec could choose a larger parity
sample or a criterion that fits it. **Update 2026-10-03:** a margin-aware criterion was then added next to the strict one (parity-margin.md); on these
same predictions it passes, every one of the 5 distinct disagreements being a tolerated near-tie flip (CPU margins 0.0002 to 0.0092). The strict
result above is unchanged and stays as recorded.

**Quality (GPU-scored, unverified against CPU).** Native accuracy: 60.1% at 512 tokens, 51.3% at 1,024, 49.0% at 2,048, 46.8% at 4,096, 44.5% at
8,192. The oracle control (the record under review with its question, framed like the items) scores 74.0% to 74.7% at every length, and the fine-tuned
checkpoint scores 74.5% on the original rows. Truncation to the checkpoint cap: 51.3% / 48.5% / 46.3% / 46.9% at 1,024 to 8,192. Window (size 256, tuned on dev):
56.2% / 52.8% / 52.0% / 50.1%. Retrieval (1,024-token budget, chosen on dev): 51.3% / 48.9% / 49.0% / 49.5%. No baseline recovers the oracle level.
Native beats truncation to 512 tokens by 3.0 points at 2,048 tokens (95% [+1.1, +5.0], 120 cases); every other native-versus-truncation comparison
at 2,048 to 8,192 tokens is inconclusive. The gap to the oracle suggests the loss comes from the surrounding text, not from the cost of attention, and
the 2026-09-30 smoke observation (one case) pointed the same way; this phase did not test the reason.

**Cost (CPU, 12-item sample, p50).** Native: 1.9 s at 512, 4.1 s at 1,024, 9.7 s at 2,048, 27.2 s at 4,096, 83.9 s at 8,192. Truncation to the cap about 4.1 s
at 1,024 and above; retrieval with a 1,024 budget 4.1 to 4.6 s; window 7.3 s at 1,024 to 69.2 s at 8,192. GPU latency (separate table): native 46 ms at 512 to 865 ms at 8,192.

**Variants (CPU, 10 cases, 50 items per length).** fastpath_off changes no prediction and is 3% faster at 512, 16% at 2,048 and 29% at 8,192. int8_encoder and
int8_all_nofast change 28% to 36% of the predicted answers and lose 10 to 16 points of accuracy against native; their latency gain is 14% to 32% at 512 and 2,048, and
8% to 37% at 8,192. Quantization is not acceptable without retraining.

**Calibration.** Two window cells (`noul` at 4,096 and 8,192 tokens) hit the temperature ceiling of 4.0, so their scaled calibration error is a bound, not a fit.

**Plan.** The frozen plan (version 1) says the 200-case final split cannot resolve a 2-point margin for 20 of 30 pilot comparisons at 80% power; the most demanding needs about
1,510 cases. Those comparisons will be inconclusive unless cases are added.

**Partial 512 cells.** 24 of the 2,400 items at 512 tokens are unsupported, because the target row alone needs more than 512 tokens (six customer-service cases,
four variants each; the length profile shows 93% of customer-service cases fit). This is the reason recorded per item.

**Caveats that apply to all of the above.** The data is synthetic and short; the distractor records come from the training split the checkpoint was trained on; the audit was by an
AI assistant, not a human; the variants run used 10 cases (spec, 2026-09-30); and the upstream test split is public.


## R22. Accuracy by variant (descriptive, from the existing p2-dev dev predictions, 2026-10-01)

Not a new run: the dev GPU-scored predictions split by item variant (600 items per cell at 512 to 2,048 tokens, 300 at 4,096 and 8,192). These are descriptive
percentages without intervals; with this many items a cell has a standard error of about 2 to 3 points (more once cases are clustered), so differences of a few points are not resolved.
GPU-scored, so unverified against CPU (R21).

Native accuracy (%), 512 / 1,024 / 2,048 / 4,096 / 8,192 tokens:
- `neutral@mid` (filler text, no other records): 57.7 / 54.5 / 47.7 / 46.3 / 41.7
- `distractor@begin`: 62.1 / 53.0 / 53.7 / 50.3 / 49.3
- `distractor@mid`: 61.8 / 50.3 / 47.2 / 46.3 / 43.7
- `distractor@end`: 58.8 / 47.5 / 47.7 / 44.3 / 43.3
- oracle (every variant): 74.1 / 74.0 / 74.0 / 74.7 / 74.7

Window (size 256): `neutral@mid` 61.4 / 57.8 / 58.8 / 57.7 / 58.0; the three distractor variants fall from 59.8 to 63.0 at 512 tokens to 46.0 to 49.0 at 8,192.
Retrieval (1,024 budget): `neutral@mid` 57.7 / 54.5 / 52.8 / 51.3 / 53.7; distractor variants 46.7 to 49.0 at 2,048 tokens and above.

What this does and does not show:
- The loss is not mainly a distractor effect. Native does worst on neutral filler at 8,192 tokens (41.7), no worse on near-matching distractors, which were suspected
  because the checkpoint was trained on the records that supply them. The earlier suspicion (R21, Limits) is not supported as the main driver.
- Added text of any kind costs accuracy even where nothing is truncated: at 512 tokens native is 57.7 to 62.1 against the oracle's 74.1. The oracle has the same framing, so the framing alone is not the cause.
- Window stays near 58 on filler at every length, so it removes the length effect for filler. It does not for distractors (46.0 to 52.0 at 2,048 tokens and above), and neither does
  retrieval. Both pick passages by similarity to the question, and a near-matching distractor is similar, which is a hypothesis this breakdown does not test.
- Not established: which part of the added text hurts (position, amount, or the markers), and whether training with such inputs would fix it. Position labels (`begin`, `mid`, `end`)
  are the construction's, recorded per item; the differences between them are within the resolution of this breakdown except possibly `distractor@begin`.

Implication for Phase 3 (a reading, not a result): if the checkpoint degrades under any surrounding text, then selection that strips the text is what helps on filler, and a training or
robustness question may matter more than attention cost. Nothing here was measured on realistic documents.
