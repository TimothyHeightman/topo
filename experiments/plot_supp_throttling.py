#!/usr/bin/env python3
"""Assemble supplement Fig. S3 from banked throttling censuses (no JAX).

Two panels: refined node count per time slice for (a) the expressive model
retrained on shrinking data sets at the full step budget, and (b) the full
data set trained for shrinking step budgets.  The black curve in both panels
is the same banked Fig. 2 checkpoint (N = 20384, 4500 steps).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS = SCRIPT_DIR / "results" / "supp_throttling"
DEFAULT_OUTPUT = PROJECT_ROOT / "figures" / "figure_throttling_census"

PAIR_COUNTS = SCRIPT_DIR / "results" / "figure3_fnqs_pair" / "nodal_worldlines.csv"
DATA_SERIES = (
    ("data_26", r"$N=26$", 0.74, (0, (1.2, 1.2))),
    ("data_234", r"$N=234$", 0.58, (0, (4.2, 1.8))),
    ("data_2106", r"$N=2106$", 0.40, (0, (7.0, 2.6))),
    ("data_20384", r"$N=20384$", None, "solid"),
)
STEP_SERIES = (
    ("steps_45", r"$45$ steps", 0.66, (0, (1.2, 1.2))),
    ("steps_450", r"$450$ steps", 0.42, (0, (4.2, 1.8))),
    ("data_20384", r"$4500$ steps", None, "solid"),
)


def style_axes(axis: mpl.axes.Axes) -> None:
    axis.tick_params(direction="out", length=2.5, width=0.65, pad=1.5)
    for spine in axis.spines.values():
        spine.set_linewidth(0.7)


def width8_counts() -> tuple[np.ndarray, np.ndarray]:
    """Per-slice node counts of the banked width-8 neutral-pair run."""

    rows = np.genfromtxt(PAIR_COUNTS, delimiter=",", names=True)
    taus = np.unique(rows["tau"])
    counts = np.asarray([np.sum(rows["tau"] == tau) for tau in taus])
    return taus, counts


def draw_panel(
    axis: mpl.axes.Axes, series: tuple[tuple[str, str, float | None, object], ...]
) -> None:
    magma = plt.get_cmap("magma")
    pair_tau, pair_counts = width8_counts()
    axis.step(
        pair_tau,
        pair_counts,
        where="mid",
        color="0.72",
        linewidth=1.0,
        label="width 8 (Fig. S2)",
        zorder=3,
    )
    maximum = int(np.max(pair_counts))
    for key, label, shade, dashes in series:
        data = np.genfromtxt(
            RESULTS / key / "node_counts.csv", delimiter=",", names=True
        )
        color = "black" if shade is None else magma(shade)
        width = 1.25 if shade is None else 1.05
        axis.step(
            data["tau"],
            data["count"],
            where="mid",
            color=color,
            linewidth=width,
            linestyle=dashes,
            label=label,
            zorder=4 if shade is None else 5 + shade,
        )
        maximum = max(maximum, int(np.max(data["count"])))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, maximum + 0.9)
    axis.set_xticks([0.0, 0.5, 1.0], ["0", "1/2", "1"])
    axis.set_yticks(np.arange(1, maximum + 1, 2))
    axis.set_xlabel(r"$t/T$", labelpad=1)
    axis.set_ylabel("nodes", labelpad=0)
    axis.legend(
        loc="upper right",
        fontsize=6.0,
        frameon=True,
        framealpha=0.9,
        edgecolor="0.7",
        handlelength=1.6,
        labelspacing=0.25,
        borderpad=0.35,
    )
    style_axes(axis)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7.3,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "axes.linewidth": 0.7,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure = plt.figure(figsize=(4.7, 1.85), facecolor="white")
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=(1.0, 0.16, 1.0),
        left=0.075,
        right=0.99,
        bottom=0.225,
        top=0.9,
        wspace=0.0,
    )
    axis_a = figure.add_subplot(grid[0, 0])
    axis_b = figure.add_subplot(grid[0, 2])
    draw_panel(axis_a, DATA_SERIES)
    draw_panel(axis_b, STEP_SERIES)

    for axis, label in ((axis_a, "(a)"), (axis_b, "(b)")):
        figure.text(
            axis.get_position().x0 - 0.045, 0.97, label, ha="left", va="top"
        )

    for suffix in (".pdf", ".png"):
        figure.savefig(args.output.with_suffix(suffix), dpi=args.dpi)
    plt.close(figure)
    print(f"saved {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
