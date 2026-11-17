# CBO-MPPI

This repository contains the public implementation and reproduction materials
for the CBO-MPPI manuscript.

**Paper:** [Consensus-Based Optimization of Path-Integral Free Energy for Model
Predictive Control](https://arxiv.org/abs/2611.18208) (arXiv:2611.18208).

Two reproduction workflows are supported:

1. **Frozen-data figure replay** regenerates all nine manuscript figures from
   compact, audited inputs.
2. **Full numerical reproduction** starts from an empty result directory,
   reruns the numerical experiments other than machine-dependent timing
   benchmarks, aggregates the results, and regenerates the figures.

Timing benchmarks are intentionally outside the numerical reproduction target:
they depend on the execution machine and are reported as measurements of the
original environment, not as hardware-independent scientific outputs.

The main workflows are also available through a small `Makefile`:

```bash
make setup
make verify          # tests and isolated replay of all nine figures
make figures         # write the nine figures under artifacts/figures/
make reproduce-plan  # inspect the full numerical campaign without running it
```

## Fast figure replay

```bash
uv sync --frozen
uv run python reproduce.py --source frozen
```

This generates the nine PDFs under `artifacts/figures/`. The plotting pipeline
uses only compact numerical arrays and tables under `data/frozen/`; existing
PDFs are not used as generator inputs. It does not run controller simulations.

## Full numerical reproduction

A CUDA-capable machine is strongly recommended. Install the pinned GPU stack,
inspect the plan, and then execute it in a new output directory:

```bash
uv sync --frozen --extra cuda --group dev
uv run python reproduce.py --source simulation --output artifacts/full-run
uv run python reproduce.py --source simulation --execute --gpu-index 0 \
  --output artifacts/full-run
```

The equivalent Makefile commands are `make setup-cuda`, `make reproduce-plan`,
and `make reproduce GPU_INDEX=0`. Add `REPRO_FLAGS=--resume` to resume a
compatible interrupted run.

The full workflow runs 5880 closed-loop trajectories, 6280 common-noise
free-energy evaluations, three landscape calculations, aggregation, all nine
figures, and predeclared numerical verification. Machine-dependent timing
benchmarks are not run. Interrupted executions can be resumed with `--resume`;
individual stages can be selected with, for example,
`--stages nominal,sensitivity,horizon`.

Before launching the full campaign, a one-seed GPU gate covering every task and
controller family can be run with:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m cbo_mppi.reproduction.smoke \
  --output-root artifacts/smoke
```

Verification thresholds were frozen before the independent recomputation in
`data/reference/verification_targets.json`.

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the release contract
and [docs/CAMPAIGNS.md](docs/CAMPAIGNS.md) for the campaign inventory.

## Citation

Please cite the [CBO-MPPI preprint](https://arxiv.org/abs/2611.18208) when using
this implementation. Machine-readable citation metadata for the paper and
software is provided in [`CITATION.cff`](CITATION.cff).

## License

The original source code in this repository is distributed under the
[MIT License](LICENSE), copyright 2026 Daisuke Inoue and contributors.
External Python and optional CUDA packages remain subject to their respective
licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
