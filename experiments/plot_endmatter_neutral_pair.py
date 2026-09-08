#!/usr/bin/env python3
"""Plot Fig. 3 from the banked trained-FNQS neutral-pair data."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "experiments" / "results" / "figure3_fnqs_pair"
OUTPUT = PROJECT_ROOT / "figures" / "figure3_neutral_pair_creation"
LOG_FLOOR = -3.0
NEGATIVE_CHARGE = "#3f7090"
POSITIVE_CHARGE = "#2f7f5f"


def load_rows(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def style_axis(axis: mpl.axes.Axes) -> None:
    axis.tick_params(direction="out", length=3.0, width=0.7, pad=2.0)
    for spine in axis.spines.values():
        spine.set_linewidth(0.75)


def charge_branch(
    rows: list[dict[str, float]], sign: int
) -> list[dict[str, float]]:
    branch = [row for row in rows if sign * row["charge"] > 0.5]
    return sorted(branch, key=lambda row: row["tau"])


def main() -> None:
    profile_path = RESULTS / "local_profiles.npz"
    worldline_path = RESULTS / "event_worldlines.csv"
    if not profile_path.exists() or not worldline_path.exists():
        raise FileNotFoundError(
            "Run experiments/run_figure3_fnqs_pair_creation.py before plotting"
        )

    with np.load(profile_path, allow_pickle=False) as bank:
        tau = bank["tau"]
        u = bank["u"]
        v = bank["v"]
        # The numerical bank stores the transverse maximum of 1-F.  Therefore
        # 1-max(1-F)=min(F), whose logarithm exposes the zero-fidelity lines.
        log_u = np.clip(
            np.log10(np.maximum(1.0 - 10.0 ** bank["log_infidelity_u"], 1.0e-3)),
            LOG_FLOOR,
            0.0,
        )
        log_v = np.clip(
            np.log10(np.maximum(1.0 - 10.0 ** bank["log_infidelity_v"], 1.0e-3)),
            LOG_FLOOR,
            0.0,
        )
    rows = load_rows(worldline_path)
    negative = charge_branch(rows, -1)
    positive = charge_branch(rows, 1)

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9.5,
            "mathtext.fontset": "cm",
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axes = plt.subplots(1, 2, figsize=(6.7, 2.7), sharey=True)
    figure.subplots_adjust(
        left=0.085, right=0.88, bottom=0.18, top=0.94, wspace=0.19
    )
    cmap = mpl.colormaps["magma"]
    norm = mpl.colors.Normalize(vmin=LOG_FLOOR, vmax=0.0)
    image_a = axes[0].imshow(
        log_u,
        origin="lower",
        extent=(u[0], u[-1], tau[0], tau[-1]),
        cmap=cmap,
        norm=norm,
        interpolation="bicubic",
        aspect="auto",
        rasterized=True,
    )
    axes[1].imshow(
        log_v,
        origin="lower",
        extent=(v[0], v[-1], tau[0], tau[-1]),
        cmap=cmap,
        norm=norm,
        interpolation="bicubic",
        aspect="auto",
        rasterized=True,
    )

    for branch, color in (
        (negative, NEGATIVE_CHARGE),
        (positive, POSITIVE_CHARGE),
    ):
        axes[0].plot(
            [row["u"] for row in branch],
            [row["tau"] for row in branch],
            color="black",
            linewidth=3.0,
            zorder=5,
        )
        axes[0].plot(
            [row["u"] for row in branch],
            [row["tau"] for row in branch],
            color=color,
            linewidth=1.55,
            zorder=6,
        )

    for branch, color, dash in (
        (negative, NEGATIVE_CHARGE, (0.0, (6.0, 5.0))),
        (positive, POSITIVE_CHARGE, (5.5, (6.0, 5.0))),
    ):
        axes[1].plot(
            [row["v"] for row in branch],
            [row["tau"] for row in branch],
            color="black",
            linewidth=3.0,
            zorder=5,
        )
        axes[1].plot(
            [row["v"] for row in branch],
            [row["tau"] for row in branch],
            color=color,
            linewidth=1.65,
            linestyle=dash,
            zorder=6,
        )

    axes[0].set_xlabel(r"$u$ (rad)")
    axes[1].set_xlabel(r"$v$ (rad)")
    axes[0].set_ylabel(r"$t/T$")
    axes[0].set_xticks([-0.4, -0.2, 0.0, 0.2, 0.4])
    axes[1].set_xticks([-0.4, -0.2, 0.0, 0.2, 0.4])
    axes[0].set_yticks([0.355, 0.360, 0.365, 0.370, 0.374])
    for axis in axes:
        style_axis(axis)
    axes[0].text(-0.16, 1.025, "(a)", transform=axes[0].transAxes, fontsize=10.5)
    axes[1].text(-0.16, 1.025, "(b)", transform=axes[1].transAxes, fontsize=10.5)

    colorbar_axis = figure.add_axes((0.905, 0.18, 0.022, 0.76))
    colorbar = figure.colorbar(
        image_a, cax=colorbar_axis, ticks=[-3, -2, -1, 0]
    )
    colorbar.set_label(r"$\log_{10}F$", rotation=90, labelpad=7)
    colorbar.outline.set_linewidth(0.75)
    colorbar.ax.tick_params(length=3.0, width=0.7, pad=2.0)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"saved {OUTPUT.with_suffix('.pdf')}")
    print(f"saved {OUTPUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
