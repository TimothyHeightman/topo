#!/usr/bin/env python3
"""Assemble the six-panel dynamical density-lift figure from banked data.

Layout: (a, b) the wavefunction- and density-valued finite-time slice stacks
spanning the full figure height, then a 2 x 2 block of matched 2D comparisons:
(c, d) the azimuth-profiled worldline heatmaps and (e, f) the periodic-node
zooms.  One shared color scale throughout.  Vector panels reuse the banked
Fig. 2 artifacts; density panels use the banked density-control artifacts.
No training or JAX evaluation happens here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_density_worldline import GHOST_DASHES, bilinear_upsample
from plot_figure3_dynamic_lift import (
    RESULTS,
    WORLDLINE_GREEN,
    draw_periodic_panel,
    draw_worldline_panel,
    log_infidelity,
    read_rows,
    require_inputs,
    shared_colormap,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "figures" / "figure3_dynamic_lift_six"


def full_worldline() -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    rows = read_rows(RESULTS / "panel_a" / "nodal_worldline.csv")
    tau = np.asarray([row["tau"] for row in rows])
    theta_star = np.asarray([row["theta"] for row in rows])
    phi_star = np.asarray([row["phi_unwrapped"] for row in rows])
    return (
        tau,
        theta_star,
        phi_star,
        float(np.median(theta_star)),
        float(np.median(phi_star)),
    )


def draw_slice_stack(
    axis: mpl.axes.Axes,
    surface_path: Path,
    theta_center: float,
    phi_center: float,
    cmap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
) -> None:
    """Six upsampled, antialiased (theta, phi) slices stacked over time."""

    surface = np.load(surface_path)
    theta = surface["theta"]
    phi = surface["phi"]
    tau = surface["tau"]
    infidelity = surface["infidelity"]
    theta_half_width = 0.31
    phi_half_width = 0.36
    theta_mask = np.abs(theta - theta_center) <= theta_half_width
    phi_mask = np.abs(phi - phi_center) <= phi_half_width
    upsample = 3
    theta_local = bilinear_upsample(theta[theta_mask][:, None], upsample)[:, 0]
    phi_local = bilinear_upsample(phi[phi_mask][:, None], upsample)[:, 0]
    pp, tt = np.meshgrid(phi_local, theta_local)

    chosen = np.linspace(0, len(tau) - 1, 6, dtype=int)
    for index in chosen:
        values = bilinear_upsample(
            infidelity[index][np.ix_(theta_mask, phi_mask)].astype(np.float64),
            upsample,
        )
        logs = log_infidelity(values)
        rgba = cmap(norm(logs))
        strength = np.clip((logs + 8.0) / 8.0, 0.0, 1.0)
        rgba[..., 3] = 0.42 + 0.50 * strength**1.25
        axis.plot_surface(
            pp,
            tt,
            np.full_like(pp, tau[index]),
            facecolors=rgba,
            linewidth=0.0,
            antialiased=True,
            shade=False,
            rcount=theta_local.size,
            ccount=phi_local.size,
            rasterized=True,
        )

    axis.set_xlim(phi_center - phi_half_width, phi_center + phi_half_width)
    axis.set_ylim(theta_center - theta_half_width, theta_center + theta_half_width)
    axis.set_zlim(0.0, 1.0)
    axis.set_xticks([phi_center - 0.25, phi_center, phi_center + 0.25])
    axis.set_yticks([theta_center - 0.2, theta_center, theta_center + 0.2])
    axis.set_zticks([0.0, 0.5, 1.0])
    axis.set_xticklabels([r"$-.25$", "0", r"$.25$"])
    axis.set_yticklabels([r"$-.2$", "0", r"$.2$"])
    axis.set_zticklabels(["0", "1/2", "1"])
    axis.set_xlabel(r"$\phi-\phi_*$", labelpad=-3)
    axis.set_ylabel(r"$\theta-\theta_*$", labelpad=-3)
    axis.set_zlabel(r"$t/T$", labelpad=-6)
    axis.view_init(elev=23, azim=-53)
    axis.set_box_aspect((1.22, 1.0, 1.42))
    axis.grid(False)
    axis.xaxis.pane.set_alpha(0.0)
    axis.yaxis.pane.set_alpha(0.0)
    axis.zaxis.pane.set_alpha(0.0)
    axis.tick_params(length=0.0, pad=-3)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--dpi", type=int, default=600)
    return result


def main() -> None:
    args = parser().parse_args()
    required = [
        RESULTS / "panel_a" / "finite_surface.npz",
        RESULTS / "panel_a" / "worldline_projection.npz",
        RESULTS / "panel_a" / "nodal_worldline.csv",
        RESULTS / "panel_a_density" / "finite_surface.npz",
        RESULTS / "panel_a_density" / "worldline_projection.npz",
        RESULTS / "panel_b" / "periodic_surface.npz",
        RESULTS / "panel_b" / "periodic_nodes.csv",
        RESULTS / "panel_b_density" / "periodic_surface.npz",
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

    worldline_tau, theta_star, phi_star, theta_center, phi_center = full_worldline()
    node = read_rows(RESULTS / "panel_b" / "periodic_nodes.csv")[0]

    figure = plt.figure(figsize=(7.05, 2.80), facecolor="white")
    grid = figure.add_gridspec(
        2,
        6,
        width_ratios=(1.18, 1.18, 0.13, 1.02, 1.02, 0.055),
        height_ratios=(0.78, 1.0),
        left=0.012,
        right=0.952,
        bottom=0.145,
        top=0.925,
        wspace=0.24,
        hspace=0.50,
    )
    axis_a = figure.add_subplot(grid[:, 0], projection="3d")
    axis_b = figure.add_subplot(grid[:, 1], projection="3d")
    axis_c = figure.add_subplot(grid[0, 3])
    axis_d = figure.add_subplot(grid[0, 4])
    axis_e = figure.add_subplot(grid[1, 3])
    axis_f = figure.add_subplot(grid[1, 4])

    # (a) wavefunction-valued slice stack with the solid charge -1 worldline.
    draw_slice_stack(
        axis_a,
        RESULTS / "panel_a" / "finite_surface.npz",
        theta_center,
        phi_center,
        cmap,
        norm,
    )
    axis_a.plot(
        phi_star, theta_star, worldline_tau, color="black", linewidth=2.2, zorder=13
    )
    axis_a.plot(
        phi_star,
        theta_star,
        worldline_tau,
        color=WORLDLINE_GREEN,
        linewidth=1.1,
        zorder=14,
    )

    # (b) density-valued slice stack with the dashed reference worldline.
    draw_slice_stack(
        axis_b,
        RESULTS / "panel_a_density" / "finite_surface.npz",
        theta_center,
        phi_center,
        cmap,
        norm,
    )
    axis_b.plot(
        phi_star,
        theta_star,
        worldline_tau,
        color="black",
        linewidth=2.0,
        linestyle=GHOST_DASHES,
        zorder=13,
    )
    axis_b.plot(
        phi_star,
        theta_star,
        worldline_tau,
        color="white",
        linewidth=0.85,
        linestyle=GHOST_DASHES,
        zorder=14,
    )

    # (c, d) azimuth-profiled worldline heatmaps.
    projected = (theta_star - theta_center) / np.pi
    draw_worldline_panel(
        axis_c, RESULTS / "panel_a" / "worldline_projection.npz", theta_center, cmap, norm
    )
    axis_c.plot(projected, worldline_tau, color="black", linewidth=1.7, zorder=5)
    axis_c.plot(
        projected, worldline_tau, color=WORLDLINE_GREEN, linewidth=0.85, zorder=6
    )
    axis_c.set_ylabel(r"$t/T$", labelpad=0)
    draw_worldline_panel(
        axis_d,
        RESULTS / "panel_a_density" / "worldline_projection.npz",
        theta_center,
        cmap,
        norm,
    )
    axis_d.plot(
        projected,
        worldline_tau,
        color="black",
        linewidth=1.6,
        linestyle=GHOST_DASHES,
        zorder=5,
    )
    axis_d.plot(
        projected,
        worldline_tau,
        color="white",
        linewidth=0.75,
        linestyle=GHOST_DASHES,
        zorder=6,
    )
    axis_d.set_yticklabels([])

    # (e, f) periodic-node zooms.
    draw_periodic_panel(
        axis_e,
        RESULTS / "panel_b" / "periodic_surface.npz",
        node["k"],
        node["t"],
        cmap,
        norm,
    )
    axis_e.scatter(
        [0.0], [0.0], s=30, facecolor="none", edgecolor="black", linewidth=1.8, zorder=5
    )
    axis_e.scatter(
        [0.0], [0.0], s=24, facecolor="none", edgecolor="#fff6b0", linewidth=1.0, zorder=6
    )
    axis_e.set_ylabel(r"$(t-t_*)/T$", labelpad=-1)
    draw_periodic_panel(
        axis_f,
        RESULTS / "panel_b_density" / "periodic_surface.npz",
        node["k"],
        node["t"],
        cmap,
        norm,
    )
    for color, width in (("black", 1.6), ("white", 0.75)):
        axis_f.add_patch(
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
    axis_f.set_yticklabels([])

    figure.text(axis_a.get_position().x0 + 0.012, 0.985, "(a)", ha="left", va="top")
    figure.text(axis_b.get_position().x0 + 0.012, 0.985, "(b)", ha="left", va="top")
    for axis, label in (
        (axis_c, "(c)"),
        (axis_d, "(d)"),
        (axis_e, "(e)"),
        (axis_f, "(f)"),
    ):
        axis.text(
            -0.02, 1.06, label, transform=axis.transAxes, ha="left", va="bottom"
        )

    color_axis = figure.add_subplot(grid[:, 5])
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
    for name in ("panel_a_density", "panel_b_density"):
        summary = json.loads(
            (RESULTS / name / "summary.json").read_text(encoding="utf-8")
        )
        print(
            f"{name}: grid max {summary['grid_max_infidelity']:.3e}, "
            f"refined max {summary['refined_max_infidelity']:.3e}"
        )
    print(f"saved {pdf_path}")
    print(f"saved {png_path}")


if __name__ == "__main__":
    main()
