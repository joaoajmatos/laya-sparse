# Implementation Plan: Exact Local-Attention Kernel (laya:007)

**Spec**: [spec.md](spec.md) · **Branch**: `007-local-exact-kernel` · **Date**: 2026-10-07
**Constitution check**: inference contract kept (`laya/` unchanged, variants applied from `experiments/`); claims need measurements (every number labeled device + dtype, estimates labeled); never mix hardware (CPU only, no GPU); reproducible (clean tree, manifest, run ids); dev only (principle VII, no final-split item).

## What the code does today (read, not assumed)

- transformers 5.18 ModernBERT: each `ModernBertAttention` computes `Wqkv`, RoPE, then calls the configured attention interface (SDPA on CPU) with a **dense 4-D boolean mask** built by `create_bidirectional_sliding_window_mask` for local layers: `|q-kv| <= config.sliding_window (64)` and key inside the row's attention mask. Local modules hold `sliding_window = config.sliding_window + 1` (65); global layers hold `None`. So the native local layer is dense L x L masked SDPA (audit: `dense_masked`).
- `experiments/kernels/local.py` is Block3 (block 128): a superset, not used for S0.
- `variants.py` applies variants outside `laya/`; `latency` and `eval` already take `--variants` (eval restricts variants to the 20-case sample, `distractor@mid`, 512/2,048/8,192 = the E2 parity subset).

## Decisions

1. **Kernel** (`experiments/kernels/local_exact.py`): queries in blocks of 64; keys from blocks b-1, b, b+1 (every key within 64 lies there) via the `unfold`, blocks-in-batch layout of `local.py` (fused 4-D SDPA); then the **exact band mask** `|i-j| <= 64` and key validity are applied inside the 192-key span. Work is about 192 keys per query against 129 ideal. Correctness first; block size is a parameter, not tuned.
2. **Semantics anchor**: `MaskSpec` gains pattern `band` with `half_width` (default 64), so `reference_attention` and `dense_masked_attention` define it; the kernel is tested against both.
3. **Hook**: for each encoder layer whose attention module has `sliding_window is not None`, replace the module's `forward` with a function that repeats the stock projection and RoPE, calls the exact kernel and `Wo`. The half-width is read from `module.sliding_window - 1`. Not a registered global attention function, so global layers are untouched by construction. `reversible=True` restores the original forwards (tests).
4. **Key validity** without changing `laya/`: the model still passes the 4-D mask to the layer; the kernel reads valid keys from its diagonal (`mask[b,0,j,j]` is true exactly when key j is a real token). A fixture test proves this equals the 2-D attention mask, including padded tails. If the mask is `None`, all keys are valid.
5. **Known cost left in**: building the dense 4-D mask (bool, L x L) is still done by transformers. Its share of time is measured in the dry run (task T012). Avoiding it is an optimization beyond S0 and is reported, not done, unless it decides E3.
6. **Variants**: `local_exact` (fast path as native) and `local_exact_fastpath_off` (= optimized native, loaded with the fast path off, as `fastpath_off` does). CPU-only, `unsupported` on GPU.
7. **E1/E1-layers driver** (`experiments/tier_e.py`, CLI `tier-e probs`): per length, the 20 variant-sample cases, first question each, `distractor@mid`; runs `fastpath_off` native, records option probabilities and, with forward hooks, each local layer's output; applies `local_exact_fastpath_off` reversibly and repeats; writes max abs difference per item, per length, per local layer (end-to-end, cumulative) to `experiments/results/<run-id>/tier_e/`. Isolated per-layer equivalence (same input hidden states) lives in `slow` tests.
8. **E2**: existing `eval --variants local_exact_fastpath_off --device cpu` (parity subset by construction), compared by `tier-e compare` to `p2-dev/quality/native.none.cpu.*` predictions: count of changed answers must be 0 of 295 (95/100/100). Margins of any flip are reported.
9. **E3/F4 table**: existing `latency --variants none,fastpath_off,local_exact_fastpath_off` at 512/1,024/2,048/4,096/8,192 (12-item sample); `tier-e report` gathers p50/p95 and the E3 verdict. `local_exact` alone is not timed (not needed by E3 or F4).
10. **Cleanliness** (human decision 2026-10-07): nothing is hidden with `.git/info/exclude`. A gate run uses a **clean detached git worktree of the merged laya-sparse commit** (`git worktree add --detach <dir> <sha>`, as for the laya:004-006 fixtures), the main repo's venv python (`.venv\Scripts\python`, run with the worktree as working directory), and writes to the worktree's own new `experiments/results/<run-id>/` (git-ignored, so the tree stays clean), which is copied out afterwards. The worktree has no `experiments/data`: pass `--data-root` pointing at the main checkout's data (read only) and, for E2, `--reference-root` pointing at the main checkout's `experiments/results` (the p2-dev native predictions). `tier-e` refuses to start with `git_dirty` unless `--allow-dirty` (tests only) and records `git_sha`, `git_dirty`, `laya_diff_empty`.
11. **Risk for E3**: transformers still builds a dense L x L boolean mask per local layer (18 per forward), which costs memory traffic quadratic in length even though the kernel no longer reads it as scores. Its share of the 8K time is **not measured** (a tiny-fixture timing would not be a hardware claim); if it is large the E3 margin shrinks, and avoiding it is the first follow-up, not part of S0.

## Dry-run cost estimate (CPU, i7-8700, fp32; *estimate*, scaled from measured p50s in `p2-dev/latency`)

Measured p50 (s): fastpath_off native 1.80 (512), 8.14 (2K), 59.97 (8K); fast-path-on native 1.86, 4.11 (1K), 9.68, 27.19 (4K), 83.88. Optimized native assumed about 0.55-0.65x of fastpath_off at 8K (the brief's 75 s to 40 s), a hypothesis the run tests.

| Batch | Estimate | Over 1 h? |
|---|---|---|
| Layer tests (slow, real checkpoint, 512-8K) | 0.3 h | no |
| E1 probabilities: 20 items x 5 lengths x (native + variant, hooks) | about 0.9 h | borderline |
| E2 predictions: 95+100+100 items, optimized native | 1.4 to 2.0 h (8K cell about 1.1-1.7 h) | **yes, needs go-ahead** |
| Latency: 3 variants x 5 lengths x (2 warmups + 12) | about 1.0 h | borderline |
| Total | about 3.6 to 4.2 h | brief said about 3 h |

The estimate is above the brief's 3 h mostly because the latency batch covers three variants and E1 times a native pass per item. E1 and E2 may each be split by length so no single batch exceeds 1 h; every batch is announced before it starts.

## Project structure

```
experiments/kernels/local_exact.py    exact band kernel
experiments/kernels/reference.py      + band pattern in MaskSpec
experiments/variants.py               + local_exact, local_exact_fastpath_off
experiments/tier_e.py                 probs / compare / report drivers
experiments/cli.py                    tier-e command; variant choices follow VARIANTS
tests/experiments/test_kernels.py     band kernel vs reference
tests/experiments/test_variants.py    variant application, layers replaced, reversibility
tests/experiments/test_tier_e.py      driver on the fixture, report verdict logic
specs/007-local-exact-kernel/         spec, plan, tasks, results.md
```
