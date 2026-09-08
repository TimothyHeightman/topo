#!/usr/bin/env python3
"""Assemble the multi-panel dynamical density-lift figure from banked data.

Four equal panels on one shared color scale: (a) the wavefunction-valued
finite-time worldline heatmap and (b) its matched density-matrix control, then
(c) the wavefunction-valued periodic node and (d) its matched density-matrix
control.  Vector panels reuse the banked Fig. 2 artifacts; density panels use
the banked run_density_finite_time.py / run_density_periodic_torus.py
artifacts.  No training or JAX evaluation happens here.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS = SCRIPT_DIR / "results" / "figure2_dynamics"
DEFAULT_OUTPUT = PROJECT_ROOT / "figures" / "figure3_dynamic_lift"
WORLDLINE_GREEN = "#2f7f5f"
GHOST_DASHES = (0.0, (2.4, 1.5))
THETA_WINDOW = 0.08
TORUS_ZOOM = 0.06


def read_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def require_inputs(paths: list[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        listing = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Banked data are missing:\n{listing}")


def shared_colormap() -> mpl.colors.Colormap:
    """The green-tail magma scale of the static density-lift comparison."""

    magma = plt.get_cmap("magma")
    green_tail = mpl.colors.LinearSegmentedColormap.from_list(
        "green_tail", ("#4BE27A", magma(0.0)), N=64
    )(np.linspace(0.0, 1.0, 64))
    magma_body = magma(np.linspace(0.0, 1.0, 192))
    colormap = mpl.colors.ListedColormap(
        np.vstack((green_tail, magma_body)), name="green_tail_magma"
    )
    colormap.set_under("#4BE27A")
    return colormap


def log_infidelity(values: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(values, 1.0e-12, 1.0))


def style_axes(axis: mpl.axes.Axes) -> None:
    axis.tick_params(direction="out", length=2.5, width=0.65, pad=1.5)
    for spine in axis.spines.values():
        spine.set_linewidth(0.7)


def vector_worldline() -> tuple[np.ndarray, np.ndarray, float]:
    rows = read_rows(RESULTS / "panel_a" / "nodal_worldline.csv")
    tau = np.asarray([row["tau"] for row in rows])
    theta_star = np.asarray([row["theta"] for row in rows])
    return tau, theta_star, float(np.median(theta_star))


def draw_worldline_panel(
    axis: mpl.axes.Axes,
    projection_path: Path,
    theta_center: float,
    cmap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
) -> None:
    projection = np.load(projection_path)
    theta = projection["theta"]
    infidelity = projection["max_phi_infidelity"]
    relative_theta = (theta - theta_center) / np.pi
    theta_mask = np.abs(relative_theta) <= THETA_WINDOW
    axis.imshow(
        log_infidelity(infidelity[:, theta_mask]),
        origin="lower",
        extent=(
            relative_theta[theta_mask][0],
            relative_theta[theta_mask][-1],
            0.0,
            1.0,
        ),
        cmap=cmap,
        norm=norm,
        interpolation="bicubic",
        aspect="auto",
        rasterized=True,
    )
    axis.set_xticks([-0.05, 0.0, 0.05], [r"$-.05$", "0", r"$.05$"])
    axis.set_yticks([0.0, 0.5, 1.0], ["0", "1/2", "1"])
    axis.set_xlabel(r"$(\theta-\theta_*)/\pi$", labelpad=1)
    style_axes(axis)


def periodic_centered_surface(
    k: np.ndarray,
    t: np.ndarray,
    values: np.ndarray,
    k_star: float,
    t_star: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Drop duplicate endpoints, shift the reference node to the cell center,
    # and retain a regular periodic grid suitable for imshow.
    k0 = k[:-1]
    t0 = t[:-1]
    data = values[:-1, :-1]
    shifted_k = (k0 - k_star + np.pi) % (2.0 * np.pi) - np.pi
    shifted_t = (t0 - t_star + np.pi) % (2.0 * np.pi) - np.pi
    order_k = np.argsort(shifted_k)
    order_t = np.argsort(shifted_t)
    return (
        shifted_k[order_k] / (2.0 * np.pi),
        shifted_t[order_t] / (2.0 * np.pi),
        data[np.ix_(order_k, order_t)],
    )


def draw_periodic_panel(
    axis: mpl.axes.Axes,
    surface_path: Path,
    k_star: float,
    t_star: float,
    cmap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
) -> None:
    surface = np.load(surface_path)
    shifted_k, shifted_t, centered = periodic_centered_surface(
        surface["k"], surface["t"], surface["infidelity"], k_star, t_star
    )
    k_mask = np.abs(shifted_k) <= TORUS_ZOOM
    t_mask = np.abs(shifted_t) <= TORUS_ZOOM
    local = centered[np.ix_(k_mask, t_mask)]
    local_k = shifted_k[k_mask]
    local_t = shifted_t[t_mask]
    axis.imshow(
        log_infidelity(local).T,
        origin="lower",
        extent=(local_k[0], local_k[-1], local_t[0], local_t[-1]),
        cmap=cmap,
        norm=norm,
        interpolation="bilinear",
        aspect="auto",
        rasterized=True,
    )
    axis.set_xticks([-0.05, 0.0, 0.05], [r"$-.05$", "0", r"$.05$"])
    axis.set_yticks([-0.05, 0.0, 0.05], [r"$-.05$", "0", r"$.05$"])
    axis.set_xlabel(r"$(k-k_*)/2\pi$", labelpad=1)
    style_axes(axis)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--dpi", type=int, default=600)
    return result


def main() -> None:
    args = parser().parse_args()
    required = [
        RESULTS / "panel_a" / "worldline_projection.npz",
        RESULTS / "panel_a" / "nodal_worldline.csv",
        RESULTS / "panel_a_density" / "worldline_projection.npz",
        RESULTS / "panel_a_density" / "summary.json",
        RESULTS / "panel_b" / "periodic_surface.npz",
        RESULTS / "panel_b" / "periodic_nodes.csv",
        RESULTS / "panel_b_density" / "periodic_surface.npz",
        RESULTS / "panel_b_density" / "summary.json",
    ]
    require_inputs(required)
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
    cmap = shared_colormap()
    norm = mpl.colors.Normalize(vmin=-8.0, vmax=0.0)

    worldline_tau, theta_star, theta_center = vector_worldline()
    node = read_rows(RESULTS / "panel_b" / "periodic_nodes.csv")[0]

    figure = plt.figure(figsize=(7.05, 2.02), facecolor="white")
    grid = figure.add_gridspec(
        1,
        6,
        width_ratios=(1.0, 1.0, 0.16, 1.0, 1.0, 0.05),
        left=0.052,
        right=0.958,
        bottom=0.215,
        top=0.905,
        wspace=0.20,
    )
    axis_a = figure.add_subplot(grid[0, 0])
    axis_b = figure.add_subplot(grid[0, 1])
    axis_c = figure.add_subplot(grid[0, 3])
    axis_d = figure.add_subplot(grid[0, 4])

    # (a) wavefunction-valued worldline with the refined charge -1 worldline.
    draw_worldline_panel(
        axis_a, RESULTS / "panel_a" / "worldline_projection.npz", theta_center, cmap, norm
    )
    projected = (theta_star - theta_center) / np.pi
    axis_a.plot(projected, worldline_tau, color="black", linewidth=2.0, zorder=5)
    axis_a.plot(
        projected, worldline_tau, color=WORLDLINE_GREEN, linewidth=1.0, zorder=6
    )
    axis_a.set_ylabel(r"$t/T$", labelpad=0)

    # (b) density-matrix control on the same window; dashed reference worldline.
    draw_worldline_panel(
        axis_b,
        RESULTS / "panel_a_density" / "worldline_projection.npz",
        theta_center,
        cmap,
        norm,
    )
    axis_b.plot(
        projected,
        worldline_tau,
        color="black",
        linewidth=1.8,
        linestyle=GHOST_DASHES,
        zorder=5,
    )
    axis_b.plot(
        projected,
        worldline_tau,
        color="white",
        linewidth=0.8,
        linestyle=GHOST_DASHES,
        zorder=6,
    )
    axis_b.set_yticklabels([])

    # (c) wavefunction-valued periodic node, circled.
    draw_periodic_panel(
        axis_c,
        RESULTS / "panel_b" / "periodic_surface.npz",
        node["k"],
        node["t"],
        cmap,
        norm,
    )
    axis_c.scatter(
        [0.0], [0.0], s=34, facecolor="none", edgecolor="black", linewidth=2.0, zorder=5
    )
    axis_c.scatter(
        [0.0], [0.0], s=27, facecolor="none", edgecolor="#fff6b0", linewidth=1.1, zorder=6
    )
    axis_c.set_ylabel(r"$(t-t_*)/T$", labelpad=-1)

    # (d) density-matrix control on the same window; dashed reference ring.
    draw_periodic_panel(
        axis_d,
        RESULTS / "panel_b_density" / "periodic_surface.npz",
        node["k"],
        node["t"],
        cmap,
        norm,
    )
    for color, width in (("black", 1.8), ("white", 0.8)):
        axis_d.add_patch(
            mpl.patches.Circle(
                (0.0, 0.0),
                radius=0.0075,
                facecolor="none",
                edgecolor=color,
                linewidth=width,
                linestyle=GHOST_DASHES,
                zorder=5 if color == "black" else 6,
            )
        )
    axis_d.set_yticklabels([])

    for axis, label in ((axis_a, "(a)"), (axis_b, "(b)"), (axis_c, "(c)"), (axis_d, "(d)")):
        figure.text(axis.get_position().x0, 0.985, label, ha="left", va="top")

    color_axis = figure.add_subplot(grid[0, 5])
    colorbar = figure.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=color_axis,
        ticks=[-8, -6, -4, -2, 0],
    )
    colorbar.ax.tick_params(length=2.5, width=0.65, pad=2)
    colorbar.outline.set_linewidth(0.7)
    colorbar.set_label(r"$\log_{10}(1-F)$", rotation=90, labelpad=2)

    pdf_path = args.output.with_suffix(".pdf")
    png_path = args.output.with_suffix(".png")
    figure.savefig(pdf_path, dpi=args.dpi)
    figure.savefig(png_path, dpi=args.dpi)
    plt.close(figure)
    summaries = {
        name: json.loads((RESULTS / name / "summary.json").read_text(encoding="utf-8"))
        for name in ("panel_a_density", "panel_b_density")
    }
    for name, summary in summaries.items():
        print(
            f"{name}: grid max {summary['grid_max_infidelity']:.3e}, "
            f"refined max {summary['refined_max_infidelity']:.3e}"
        )
    print(f"saved {pdf_path}")
    print(f"saved {png_path}")


if __name__ == "__main__":
    main()
