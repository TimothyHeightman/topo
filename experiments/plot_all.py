#!/usr/bin/env python3
"""Re-render every figure from banked artifacts, training nothing.

The plotting counterpart of ``run_all.py``: run it after the data scripts (or
after ``run_all.py``) to regenerate every figure of the paper, including the
compact variants, purely from ``experiments/results/``.  Fails with a clear
FileNotFoundError naming the missing bank if a required data stage has not
been run yet.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from run_all import compose_figure1

SCRIPT_DIR = Path(__file__).resolve().parent

STAGES: tuple[tuple[str, list[str]], ...] = (
    ("density-lift comparison", ["plot_density_lift_comparison.py"]),
    ("Fig. 2", ["plot_figure2_dynamics.py"]),
    ("dynamical density lift (six-panel)", ["plot_figure3_sixpanel.py"]),
    ("dynamical density lift (compact)", ["plot_figure3_dynamic_lift.py"]),
    ("finite-time density worldline", ["plot_density_worldline.py"]),
    ("neutral pairs (End Matter variant)", ["plot_endmatter_neutral_pair.py"]),
    ("Fig. S1", ["plot_supp_susceptibility.py"]),
    ("Fig. S2", ["plot_supp_pairs.py"]),
    ("throttling census", ["plot_supp_throttling.py"]),
)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    start = time.perf_counter()
    for index, (label, command) in enumerate(STAGES, start=1):
        print(f"\n=== [{index}/{len(STAGES)}] {label}: {' '.join(command)} ===")
        subprocess.run([sys.executable, *command], cwd=SCRIPT_DIR, check=True)
    compose_figure1()
    print(
        f"\nComplete: every figure re-rendered in "
        f"{time.perf_counter() - start:.1f} s."
    )


if __name__ == "__main__":
    main()
