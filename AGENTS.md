# AGENTS.md

Guidance for AI coding agents (Codex, Claude Code) and human contributors. The public overview is in [README.md](README.md).

## Project

Research fork of Laya: sparse attention for encoder-only decision models at 4K–8K tokens on CPU. Primary workload is one fresh document plus one question per decision, model weights resident. Read [`docs/research-plan.md`](docs/research-plan.md) and the [constitution](.specify/memory/constitution.md) before changing anything measured or claimed.

## Ground rules

- **Preserve the inference contract.** `laya.load(...).predict(...)` and typed outputs must not change. Only `native` attention is implemented; `laya/layers/attention.py` is still empty.
- **Claims need measurements.** Mark unmeasured statements as hypotheses. Report negative results.
- **Never mix hardware.** The ~33 ms figure is upstream's Tesla T4 number, not a CPU baseline. GPU results (`gpu-reference`) never stand in for CPU results. Fixture-model timings are not measurements.
- **Keep the runtime focused.** No router, server, integrations or export paths (all removed from upstream on purpose).
- **Reproducibility.** Runs record the pinned manifest and go under `experiments/results/<run-id>/` (git-ignored).

## Layout

```
laya/                  Core library trimmed from upstream (agent.py, common.py, layers/attention.py)
experiments/           Measurement tooling, run with `python -m experiments`
  cli.py               Commands: manifest, audit, sweep, profile, kernels, report, all, gpu-reference, tier-e
  kernels/             Dense, masked, block-local, gather-attend-scatter and exact +/-64 band (local_exact) and block-local-with-global-tokens (A1) attention kernels
  variants.py          Optimized-native variants: fastpath_off, local_exact, local_exact_fastpath_off (laya:007), int8
  candidates.py        Tier F candidates A1/B1 (laya:008): global tokens, mask-only references, CPU paths
  parity_draw.py       Fresh parity case sets (seeded, offline; ids in specs/008-tier-f-screen/parity-ids.json)
  tier_e.py            Tier E check of the compression gate (E1 probabilities, E2 predictions, E3 latency)
tests/experiments/     Offline tests on a tiny fixture model
specs/                 Spec Kit features (001-cpu-path-audit, 002-decision-benchmark-baselines, 007-local-exact-kernel, 008-tier-f-screen)
docs/                  research-plan, sparse-attention-report (draft), gate-plan-v2 (pre-registered plan text, byte-exact)
.specify/              Spec Kit templates, scripts and memory/constitution.md
```

## Commands

```bash
pip install -e ".[experiments]"          # install with test deps
pytest -m "not slow"                     # offline tests (slow = real checkpoint / network)
python -m experiments all --run-id full --revision reviewed   # Phase 1 measurements
```

On Windows, `experiments/setup_windows.ps1` builds the CPU environment and `experiments/setup_gpu.ps1` the GPU reference environment (`.venv`, `.venv-gpu`). Details: [`specs/001-cpu-path-audit/quickstart.md`](specs/001-cpu-path-audit/quickstart.md).

## Spec-driven workflow

This repo uses [GitHub Spec Kit](https://github.com/github/spec-kit). Skills are installed in `.agents/skills/` (Codex, invoked `$speckit-*`) and `.claude/skills/` (Claude Code, invoked `/speckit-*`).

Order for a feature: `specify` → `clarify` (if ambiguous) → `plan` → `tasks` → `implement`. `analyze` and `checklist` are optional review steps. Feature artifacts live in `specs/<NNN-name>/`.

## Conventions

- Python 3.10+, packaging via `pyproject.toml` (setuptools).
- Match surrounding style; keep comments sparse and explain why, not what.
- Add or update offline tests in `tests/experiments/` for tooling changes.
