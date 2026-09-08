#!/usr/bin/env python3
"""Assemble supplement Fig. S2 from banked pair-creation artifacts (no JAX)."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PAIR_RESULTS = SCRIPT_DIR / "results" / "figure3_fnqs_pair"
SLICE_RESULTS = SCRIPT_DIR / "results" / "supp_pair_slice"
DEFAULT_OUTPUT = PROJECT_ROOT / "figures" / "figureS2_neutral_pairs"
NEGATIVE_COLOR = "#2f7f5f"
POSITIVE_COLOR = "#1e90ff"


def style_axes(axis: mpl.axes.Axes) -> None:
    axis.tick_params(direction="out", length=2.5, width=0.65, pad=1.5)
    for spine in axis.spines.values():
        spine.set_linewidth(0.7)


def read_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


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
    # Color scale for panel (c): failure depth log10(F), inverted so the node
    # cores read as dark points on a light background, matching Fig. S1(b).
    magma = plt.get_cmap("magma")
    cmap = mpl.colormaps["magma"].copy()
    cmap.set_under(magma(0.0))
    norm = mpl.colors.Normalize(vmin=-5.0, vmax=0.0)

    figure = plt.figure(figsize=(7.05, 1.95), facecolor="white")
    grid = figure.add_gridspec(
        1,
        4,
        width_ratios=(0.95, 1.0, 1.5, 0.05),
        left=0.052,
        right=0.955,
        bottom=0.215,
        top=0.9,
        wspace=0.34,
    )
    axis_a = figure.add_subplot(grid[0, 0])
    axis_b = figure.add_subplot(grid[0, 1])
    axis_c = figure.add_subplot(grid[0, 2])
    color_axis = figure.add_subplot(grid[0, 3])

    # (a) node count and total charge against time.
    worldlines = read_rows(PAIR_RESULTS / "nodal_worldlines.csv")
    per_time: dict[float, list[float]] = defaultdict(list)
    for row in worldlines:
        per_time[row["tau"]].append(row["charge"])
    taus = np.asarray(sorted(per_time))
    counts = np.asarray([len(per_time[tau]) for tau in taus])
    total_charge = np.asarray([sum(per_time[tau]) for tau in taus])
    axis_a.step(taus, counts, where="mid", color="black", linewidth=1.1, zorder=4)
    axis_a.step(
        taus,
        total_charge,
        where="mid",
        color=NEGATIVE_COLOR,
        linewidth=1.1,
        linestyle=(0, (4, 2)),
        zorder=3,
    )
    axis_a.set_xlim(0.0, 1.0)
    axis_a.set_ylim(-1.9, 5.9)
    axis_a.set_xticks([0.0, 0.5, 1.0], ["0", "1/2", "1"])
    axis_a.set_yticks([-1, 1, 3, 5])
    axis_a.set_xlabel(r"$t/T$", labelpad=1)
    axis_a.set_ylabel(r"nodes, $\sum q$", labelpad=0)
    style_axes(axis_a)

    # (b) the first neutral birth: refined branch worldlines in the local frame.
    events = read_rows(PAIR_RESULTS / "event_worldlines.csv")
    branches: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
    for row in events:
        charge_sign = 1 if row["charge"] > 0 else -1
        branches[(int(row["node_index"]), charge_sign)].append(
            (row["u"], row["tau"])
        )
    birth_tau = min(row["tau"] for row in events)
    birth_u = float(
        np.mean([row["u"] for row in events if row["tau"] == birth_tau])
    )
    for (_, charge_sign), samples in sorted(branches.items()):
        samples.sort(key=lambda pair: pair[1])
        u_branch = np.asarray([pair[0] for pair in samples])
        tau_branch = np.asarray([pair[1] for pair in samples])
        color = POSITIVE_COLOR if charge_sign > 0 else NEGATIVE_COLOR
        axis_b.plot(u_branch, tau_branch, color=color, linewidth=1.7, zorder=5)
    axis_b.scatter(
        [birth_u],
        [birth_tau],
        s=16,
        facecolor="black",
        edgecolor="none",
        zorder=6,
    )
    axis_b.set_xlim(-0.42, 0.42)
    axis_b.set_ylim(0.3595, 0.3755)
    axis_b.set_xticks([-0.3, 0.0, 0.3])
    axis_b.set_yticks([0.36, 0.365, 0.37, 0.375])
    axis_b.set_xlabel(r"$u$ (rad)", labelpad=1)
    axis_b.set_ylabel(r"$t/T$", labelpad=-1)
    style_axes(axis_b)

    # (c) the five-node slice with charge-colored markers.
    slice_data = np.load(SLICE_RESULTS / "pair_slice.npz")
    log_fidelity = np.log10(
        np.clip(slice_data["fidelity"].astype(np.float64), 1.0e-12, 1.0)
    )
    theta_grid = slice_data["theta"]
    upper = theta_grid >= 0.5 * np.pi
    image = axis_c.imshow(
        log_fidelity[upper],
        origin="lower",
        extent=(0.0, 2.0, 0.5, 1.0),
        cmap=cmap,
        norm=norm,
        interpolation="bilinear",
        aspect="auto",
        rasterized=True,
    )
    for theta, phi, charge in zip(
        slice_data["node_theta"], slice_data["node_phi"], slice_data["node_charge"]
    ):
        color = POSITIVE_COLOR if charge > 0 else NEGATIVE_COLOR
        for size, ring_color in ((150, color), (114, "white"), (84, color)):
            axis_c.scatter(
                [phi / np.pi],
                [theta / np.pi],
                s=size,
                facecolor="none",
                edgecolor=ring_color,
                linewidth=1.15,
                zorder=5,
            )
    axis_c.set_xticks((0.0, 0.5, 1.0, 1.5, 2.0))
    axis_c.set_yticks((0.5, 0.75, 1.0), ("1/2", "3/4", "1"))
    axis_c.set_ylim(0.5, 1.0)
    axis_c.set_xlabel(r"$\phi/\pi$", labelpad=1)
    axis_c.set_ylabel(r"$\theta/\pi$", labelpad=0)
    style_axes(axis_c)

    colorbar = figure.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=color_axis,
        ticks=[-5, -4, -3, -2, -1, 0],
    )
    colorbar.ax.invert_yaxis()
    colorbar.ax.tick_params(length=2.5, width=0.65, pad=2)
    colorbar.outline.set_linewidth(0.7)
    colorbar.set_label(r"$\log_{10}F$", rotation=90, labelpad=2)

    for axis, label in ((axis_a, "(a)"), (axis_b, "(b)"), (axis_c, "(c)")):
        figure.text(
            axis.get_position().x0 - 0.028, 0.965, label, ha="left", va="top"
        )

    for suffix in (".pdf", ".png"):
        figure.savefig(args.output.with_suffix(suffix), dpi=args.dpi)
    plt.close(figure)
    print(f"saved {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
