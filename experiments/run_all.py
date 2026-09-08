#!/usr/bin/env python3
"""Reproduce every result and figure of the paper with one command.

Runs the complete pipeline in dependency order: static obstruction and its
density lift, the dynamical obstructions and their density lifts, the
neutral-pair example, the susceptibility, throttling, and sampled-gradient
supplements, and every figure script.  Each stage is a subprocess of the same
interpreter; training stages skip themselves when their banked artifacts
already exist, so an interrupted run resumes where it stopped.

Expect a few hours on a laptop CPU for a cold start; progress and per-stage
timing are printed as it goes.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES = SCRIPT_DIR.parent / "figures"

STAGES: tuple[tuple[str, list[str]], ...] = (
    ("static obstruction (Fig. 1 data)", ["one_qubit_fnqs_obstruction.py"]),
    ("static density lift", ["one_qubit_density_fnqs_lift.py"]),
    ("density-lift comparison figure", ["plot_density_lift_comparison.py"]),
    ("finite-time quench (Fig. 2a,b data)", ["run_figure2a_finite_time.py"]),
    ("periodic torus (Fig. 2c data)", ["run_figure2b_periodic_torus.py"]),
    ("endpoint gluing phase (auxiliary)", ["run_figure2c_gluing_phase.py"]),
    ("Fig. 2 assembly", ["plot_figure2_dynamics.py"]),
    ("density lift, finite time", ["run_density_finite_time.py"]),
    ("density lift, periodic", ["run_density_periodic_torus.py"]),
    ("dynamical density-lift figure", ["plot_figure3_sixpanel.py"]),
    ("width-8 neutral pairs", ["run_figure3_fnqs_pair_creation.py"]),
    ("five-node slice", ["run_supp_pair_slice.py"]),
    ("fidelity susceptibility", ["run_supp_susceptibility.py"]),
    ("Fig. S1 assembly", ["plot_supp_susceptibility.py"]),
    ("Fig. S2 assembly", ["plot_supp_pairs.py"]),
    ("data/step throttling census", ["run_supp_throttling.py"]),
    ("throttling seed exploration", ["run_supp_throttling_seeds.py"]),
    ("throttling figure", ["plot_supp_throttling.py"]),
    ("sampled-gradient controls (Figs. S3, S4)", ["run_sampled_static.py"]),
)


def compose_figure1() -> None:
    """Copy the banked zoom panel and compose Fig. 1 if latexmk is available."""

    source = (
        SCRIPT_DIR
        / "results"
        / "one_qubit_fnqs"
        / "log_infidelity_sphere_zoom_t241_p481.pdf"
    )
    shutil.copy(source, FIGURES / "figure1_obstruction.pdf")
    if shutil.which("latexmk") is None:
        print("latexmk not found; skipping the figure1_static.tex composition")
        return
    subprocess.run(
        [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "figure1_static.tex",
        ],
        cwd=FIGURES,
        check=True,
    )


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    start = time.perf_counter()
    for index, (label, command) in enumerate(STAGES, start=1):
        stage_start = time.perf_counter()
        print(f"\n=== [{index}/{len(STAGES)}] {label}: {' '.join(command)} ===")
        subprocess.run(
            [sys.executable, *command], cwd=SCRIPT_DIR, check=True
        )
        print(f"=== done in {time.perf_counter() - stage_start:.1f} s ===")
    compose_figure1()
    print(
        f"\nComplete: every banked result and figure reproduced in "
        f"{(time.perf_counter() - start) / 60:.1f} min."
    )


if __name__ == "__main__":
    main()
