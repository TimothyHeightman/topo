# Topology Obstructs Pure-State Foundation Neural Quantum States

Code for the numerical experiments and figures of
*Topology Obstructs Pure-State Foundation Neural Quantum States*.

Everything retrains from scratch on your machine: no checkpoints or evaluation
data ship with this repository.  All experiments are deterministic, seeded,
CPU-only evaluations of one-qubit families; where Monte Carlo sampling is used
(`run_sampled_static.py`) the sampler is seeded as well.

## Environment

Python 3.12 or newer with the pinned packages in `requirements.txt`:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Every experiment script calls `check_environment()` so please make sure you
have the right env to reproduce the same numerics as the paper.

## Reproduce everything with one command

```sh
.venv/bin/python experiments/run_all.py
```

runs the complete pipeline in dependency order — every training run, every
banked evaluation, and every figure — and prints per-stage timing.  Expect a
few hours on a laptop CPU for a cold start.  Stages whose banked artifacts
already exist are skipped, so an interrupted run resumes where it stopped.

## Layout

- `experiments/` — training/evaluation scripts (`run_*.py` and the two
  `one_qubit_*.py` scripts) bank artifacts — checkpoints, dense evaluations,
  refined worldline/node tables, manifests — into `experiments/results/`,
  which is created locally and ignored by git; `plot_*.py` scripts render
  figures **only** from those banked artifacts, so visual changes never
  retrain a model.
- `figures/` — the figures as used in the paper; rerunning the pipelines
  regenerates them in place.

## Reproducing individual figures

Data scripts first (their defaults reproduce the paper's protocol exactly),
then the plot scripts.  Note the ordering: run `one_qubit_fnqs_obstruction.py`,
`one_qubit_density_fnqs_lift.py`, and `run_figure2a_finite_time.py` before
`run_supp_susceptibility.py`, which reuses their checkpoints and banked data
(`run_all.py` handles this ordering for you).

| Figure | Bank the data | Render |
| --- | --- | --- |
| Fig. 1 (static obstruction) | `one_qubit_fnqs_obstruction.py` | banked alongside the data; `figures/figure1_static.tex` composes the panels |
| Density-lift comparison | `one_qubit_fnqs_obstruction.py`, `one_qubit_density_fnqs_lift.py` | `plot_density_lift_comparison.py` |
| Fig. 2 (dynamical obstruction) | `run_figure2a_finite_time.py`, `run_figure2b_periodic_torus.py` (auxiliary: `run_figure2c_gluing_phase.py`) | `plot_figure2_dynamics.py` |
| Density lift of both dynamical obstructions | `run_density_finite_time.py`, `run_density_periodic_torus.py` | `plot_figure3_sixpanel.py` (compact variants: `plot_figure3_dynamic_lift.py`, `plot_density_worldline.py`) |
| Fig. S1 (fidelity susceptibility) | statics and `run_figure2a_finite_time.py` above, then `run_supp_susceptibility.py` | `plot_supp_susceptibility.py` |
| Fig. S2 (neutral node pairs, width-8 model) | `run_figure3_fnqs_pair_creation.py`, `run_supp_pair_slice.py` | `plot_supp_pairs.py` (variant: `plot_endmatter_neutral_pair.py`; interactive export: `export_figure3_interactive.py`) |
| Throttling robustness (node census under data/step/capacity restriction) | `run_supp_throttling.py`, `run_supp_throttling_seeds.py` | `plot_supp_throttling.py` |
| Sampled-gradient controls (Metropolis-sampled static obstruction and density lift) | `run_sampled_static.py` | same script (figures rendered on completion) |

To re-render every figure without retraining anything, run

```sh
.venv/bin/python experiments/plot_all.py
```

(equivalently `make figures`) against your locally banked results.
