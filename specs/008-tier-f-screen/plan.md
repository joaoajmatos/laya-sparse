# Implementation Plan: Tier F Screen of A1 and B1 (laya:008)

**Spec**: [spec.md](spec.md) · **Branch**: `008-tier-f-screen` (from main 56d9007; nothing from the unmerged `laya:007` branch) · **Date**: 2026-10-07
**Constitution check**: inference contract kept (variants applied from `experiments/`); every claim a measurement or labeled estimate; GPU numbers never stand in for CPU numbers (mask-only = quality diagnostic, CPU path = the only F4 evidence); dev and calibration only, final split untouched; reproducible from a clean worktree.

## What the code does today (read, not assumed)

- Input layout (`laya/common.py:build_sequence`): `[CLS] "<type> question: ..." [SEP] ([MASK] option tokens)* [SEP] state [SEP]`. The header (everything up to the [SEP] that closes the options) is capped by `head_max_len` (192 in the builder, 256 in the real config). The decision head reads `h[:, 0]` (CLS) and `h` at `marker_pos` only.
- Encoder: ModernBERT, 28 layers, 18 local (+/-64) and 10 global; the attention interface gets a dense boolean mask for each. Global layers have no window.
- Decision head: `nn.TransformerEncoder` of `head_layers` (2 for the real checkpoint) pre-norm layers (`norm_first=True`), dense attention over all L tokens with `src_key_padding_mask`; attention is `_DynamicMultiheadAttention` (SDPA).
- `kernels/local.py`: Block3 (block 128, 3-block span). `kernels/gas.py`: gather-attend-scatter over blocks. `MaskSpec` patterns: full, block_local, gather (and `band` once laya:007 merges; not needed here).

## Decisions

1. **Global-token set G** from the input ids only (`experiments/candidates.py:global_tokens(ids, special_ids, mode)`): default `header` = positions 0 through the second [SEP]; alternative `markers` = [CLS], question tokens up to the first [SEP], and each option [MASK]. If the second [SEP] is missing (header cut by `max_len`), G is every valid token, which makes the layer native: documented, tested. G never uses labels or state text. (Clarification 1 in the spec: default is the header.)
2. **Hook for context**: a forward pre-hook on the encoder stores G per forward (`input_ids` are known there); the replaced attention modules and the B1 forward read it. Context is removed on restore.
3. **A1 pattern** (`block_global_allowed`): query i may attend key j iff `valid_j` and (`|block(i)-block(j)| <= 1` or `i in G` or `j in G`).
   - **Mask-only reference** (`a1_mask`): dense SDPA with that pattern as a boolean mask (any device).
   - **CPU path** (`a1`, `kernels/block_global.py`): per query block, keys = the 3-block span plus the G keys that lie outside it (padded to the batch's largest |G|; the span part masks invalid keys); G queries attend all keys in one dense `[|G|, L]` product and overwrite their rows. Work is O(L x (3 x block + |G|) + |G| x L).
4. **B1 pattern**: S = G plus every position within `window` of a position in G (valid only). Both head layers' keys are restricted to S for every query.
   - **Mask-only reference** (`b1_mask`): the head's layers run densely (one manual pre-norm layer function that reproduces `nn.TransformerEncoderLayer` with `norm_first=True`, `layer.activation`, eval mode) with keys masked to S.
   - **CPU path** (`b1`): gather the rows of S into a compact `[B, max|S|, d]` (padded, key-padding masked), run the same layer function there, read the marker and CLS outputs by their position in S. Marker and CLS positions are in G, hence in S, so their outputs are identical to the reference; positions outside S are never read (stated, and proven by the equivalence test).
   - The manual layer function is tested against the native layer with S = everything (guards drift from `nn.TransformerEncoderLayer`).
   - `DecisionModel.forward` is re-implemented in `candidates.py` for B1 only (head loop swapped; embeddings, scorer, entropy features, act head copied); tested equal to the native `forward` with S = everything. `laya/` is not touched.
5. **Variants** (`variants.py`): `a1`, `a1_mask`, `b1`, `b1_mask`, rescue twins `a2*` (block 256) and `b2*` (window 256) as parameter variants of the same code. Non-mask variants are CPU-only. Mask-only variants are allowed on a GPU agent: `evalrun.eval_condition` applies them after `_load_gpu_agent` (small change, tested on the CPU fixture; the GPU itself is not touched offline).
6. **Composition with optimized native** (F4 reference): on CPU, `a1` and `b1` are meant to run on top of optimized native (fast path off + `laya:007`'s `local_exact`), so F4 isolates the candidate's saving. That composition needs `laya:007` merged; it is task T016 (held). Until then the candidates are applied alone, which is what the offline equivalence tests need.
7. **Plan v2 emitter** (`evalplan.py`): `PLAN_V2` constants; `freeze_plan_v2(run_path, plan_md=None, new_version=False, reason="")`; writes `evaluation_plan_v2.json` (+ `.md`), refuses on any final item in results and on an existing v2 file without `new_version` + reason; fingerprint = sha256 of the canonical JSON; records the sha256 of `--plan-md` when given. CLI: `freeze-plan --plan-version 2` (default stays v1 behaviour).
8. **Final-split rule** (`evalrun.select_items`): half-sample at 4,096 and 8,192 for dev and calibration; **all cases for `final`** (`final_full_lengths: [4096, 8192]` in the plan JSON). No final item is scored; the rule is tested on a fixture split.
9. **Fresh parity draw** (`experiments/parity_draw.py`): `draw_fresh_sets(splits, seed, n=20)` and CLI `parity-draw --seed S [--out FILE]`: pool = dev case ids not in `variant_sample`; per workflow shuffle with a seed-derived stream; take 5 per workflow for A then 5 for B (disjoint); fewer than 5 available in a workflow is an error, never a silent shortfall. Prints/returns ids; **never run on the real splits in this PR**.
10. **Held** (each a task; needs the human's go and, for CPU runs, an idle i7-8700): native gate-dtype re-scoring, A1/B1 dev and calibration scoring, parity runs, latency/memory, F1 to F4 reporting (summary, report2), C-ref, rescue triggers.

## Information flow (research-plan item 7)

A1, per global layer: document tokens exchange information with at most 3 x 128 neighbours and with G; two document tokens farther apart than that interact only via G (hub-and-spoke), once per global layer, 10 layers. The 18 local layers are unchanged. Executed operations: block-local attention over the span plus G keys, one dense row block for G queries, FFNs and norms as native on all L tokens (A1 saves attention only).
B1: the encoder is native, so every token's final encoder state is native. The head sees only S: marker and CLS outputs lose attention to document tokens farther than `window` from G (state tokens after position |header| + 128 are never keys). Executed operations on the CPU path: head attention and head FFNs over |S| tokens instead of L; all encoder work remains over L.

## Cost of this PR

Offline only: no run, no timing, no GPU. The held run estimates are those of plan v2 section 8 (GPU 3 to 4 h, CPU about 11 to 13 h incl. the approved 4K cell); a CPU F4 expectation for B1 is flagged in the spec and not asserted.

## Project structure

```
experiments/candidates.py            global_tokens, A1/B1 patterns, variant application, B1 forward
experiments/kernels/block_global.py  A1 CPU kernel
experiments/variants.py              + a1/a1_mask/a2/a2_mask/b1/b1_mask/b2/b2_mask
experiments/evalrun.py               final-split tier rule; mask-only variants on a GPU agent
experiments/evalplan.py              PLAN_V2, freeze_plan_v2
experiments/parity_draw.py           fresh parity sets
experiments/cli.py                   freeze-plan --plan-version, parity-draw
tests/experiments/                   test_candidates.py, test_block_global.py, test_parity_draw.py, test_evalplan.py, test_evalrun.py
specs/008-tier-f-screen/             spec, plan, tasks
```
