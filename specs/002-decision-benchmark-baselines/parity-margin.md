# Margin-aware CPU/GPU parity (laya:002, 2026-10-03)

Follows research.md R20 and R21. The strict parity rule (98% identical predictions, accuracy difference at most 0.5 points) failed on the
p2-dev subset because one flipped item is 1 point on 95 to 100 items. This note records the margin-aware criterion added next to it. The strict
numbers are still computed and reported (`strict_passed`).

## Criterion

A CPU/GPU label disagreement counts as a parity **failure** only if the CPU top-two probability margin on that item is **greater than 0.01**
(`experiments/summary.py:PARITY_NEAR_TIE_MARGIN`; override with `eval-summary --parity-margin`). Disagreements at a margin of 0.01 or less are
reported as **tolerated near-tie flips**, listed with their margins (`tolerated_flips`); the others are in `failing_flips`. Parity passes when no
disagreement is above the margin. Basis: the bf16 flips sat at CPU margins 0.0092 and 0.0010 (worst margin seen is 0.0092), bf16 versus CPU
probability differences were 0.0036 to 0.0078 across the 9 cells and fp16 versus CPU 0.0010. The margin is a researcher's choice and it is circular: it was set from the very flips it now judges (worst margin 0.0092, so any margin of 0.01 or more tolerates all of them), with no held-out data. The 9 bf16 cells below reuse the same 20 cases and 100 items, so they are not 9 independent confirmations: they contain 4 distinct flipped items in total. A pass on this run therefore says "every disagreement seen here is a near-tie at margin 0.01", not that 0.01 is a validated threshold.

## Result on the existing predictions (no new model run)

CPU: i7-8700, fp32. GPU: RTX 4060, labeled by dtype. 20-case dev variant sample, `distractor@mid`, 95 items at 512 tokens, 100 at 2,048 and 8,192.

- **GPU bf16 autocast (run p2-dev), 9 cells** (native, truncCap, retrieve1024 at 512, 2,048, 8,192): margin-aware **pass**; strict rule still **missed** in 5 cells.
  Disagreements: 8 in total over 5 cells, 4 distinct items, all tolerated, CPU margins 0.0002 to 0.0092.
- **GPU fp16 autocast (run p2-parity-fp16-8192), native 8,192:** 100% identical, accuracy 52.0 vs 52.0, largest probability difference 0.0010; passes both.
- **GPU fp32 (run p2-parity-fp32), native 512 and 2,048:** 100% identical, largest probability difference 0.0001; passes both. The 8,192-token cell of that run is
  not valid (two items, scored through Laya's CPU out-of-memory fallback) and is not evidence.

Caveat: this does not make GPU-scored quality CPU-equivalent beyond near-ties. It says every observed disagreement is a coin-flip-level one on this subset;
20 cases remain a small sample.
