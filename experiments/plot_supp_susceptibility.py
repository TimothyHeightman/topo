#!/usr/bin/env python3
"""Assemble supplement Fig. S1 from banked susceptibility scans (no JAX).

Three panels: (a) the fidelity-metric map on the display sphere, viewed
from the same oblique angle as Fig. 1, with the dashed scan great circle and
a geodesic circle of radius delta_min around x_*; (b) chi_F along the
through-node scan; (c) peak chi_F against the scan's closest approach
delta_min.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS = SCRIPT_DIR / "results" / "supp_susceptibility"
DEFAULT_OUTPUT = PROJECT_ROOT / "figures" / "figureS1_susceptibility"
VECTOR_COLOR = "#8c2d5a"
DENSITY_COLOR = "#2f7f5f"
GHOST_DASHES = (0.0, (2.6, 1.6))
OFFSET_CIRCLE_RADIUS = 0.3


def style_axes(axis: mpl.axes.Axes) -> None:
    axis.tick_params(direction="out", length=2.5, width=0.65, pad=1.5)
    for spine in axis.spines.values():
        spine.set_linewidth(0.7)


def sphere_coordinates(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return np.stack(
        (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)),
        axis=-1,
    )


def visible_mask(points: np.ndarray, elev: float, azim: float) -> np.ndarray:
    view_direction = np.asarray(
        (
            math.cos(math.radians(elev)) * math.cos(math.radians(azim)),
            math.cos(math.radians(elev)) * math.sin(math.radians(azim)),
            math.sin(math.radians(elev)),
        )
    )
    return points @ view_direction > 0.05


def draw_masked_curve(
    axis: mpl.axes.Axes,
    points: np.ndarray,
    elev: float,
    azim: float,
    core_style: dict,
    casing_style: dict,
) -> None:
    masked = np.where(
        visible_mask(points, elev, azim)[:, None], 1.012 * points, np.nan
    )
    axis.plot(masked[:, 0], masked[:, 1], masked[:, 2], **casing_style)
    axis.plot(masked[:, 0], masked[:, 1], masked[:, 2], **core_style)


def panel_a_sphere(
    axis: mpl.axes.Axes,
    maps: np.lib.npyio.NpzFile,
    cmap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
) -> None:
    theta = maps["theta"]
    phi = maps["phi"]
    field = np.log10(np.clip(4.0 * maps["vector_lambda_max"], 1.0e-3, None))
    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    points = sphere_coordinates(tt, pp)
    surface = axis.plot_surface(
        points[..., 0],
        points[..., 1],
        points[..., 2],
        facecolors=cmap(norm(field)),
        rcount=theta.size,
        ccount=phi.size,
        linewidth=0.0,
        antialiased=False,
        shade=False,
    )
    surface.set_rasterized(True)

    # The same oblique view as Fig. 1: x_* in the upper-right hemisphere.
    x_star_theta = float(maps["x_star_theta"])
    x_star_phi = float(maps["x_star_phi"])
    x_star = sphere_coordinates(np.asarray(x_star_theta), np.asarray(x_star_phi))
    elev = 40.0
    azim = math.degrees(math.atan2(x_star[1], x_star[0])) - 60.0
    axis.view_init(elev=elev, azim=azim)

    scan_path = sphere_coordinates(maps["scan_path_theta"], maps["scan_path_phi"])
    draw_masked_curve(
        axis,
        scan_path,
        elev,
        azim,
        core_style={
            "color": "black",
            "linewidth": 1.0,
            "linestyle": GHOST_DASHES,
            "zorder": 11,
        },
        casing_style={
            "color": "white",
            "linewidth": 2.1,
            "linestyle": GHOST_DASHES,
            "zorder": 10,
        },
    )

    # Geodesic circle of radius delta_min around x_*, illustrating panel (d).
    reference = np.asarray((0.0, 0.0, 1.0))
    if abs(float(x_star @ reference)) > 0.85:
        reference = np.asarray((0.0, 1.0, 0.0))
    tangent_one = np.cross(reference, x_star)
    tangent_one /= np.linalg.norm(tangent_one)
    tangent_two = np.cross(x_star, tangent_one)
    angles = np.linspace(0.0, 2.0 * np.pi, 721)
    offset_circle = (
        math.cos(OFFSET_CIRCLE_RADIUS) * x_star[None, :]
        + math.sin(OFFSET_CIRCLE_RADIUS)
        * (
            np.cos(angles)[:, None] * tangent_one[None, :]
            + np.sin(angles)[:, None] * tangent_two[None, :]
        )
    )
    draw_masked_curve(
        axis,
        offset_circle,
        elev,
        azim,
        core_style={"color": "black", "linewidth": 0.9, "zorder": 11},
        casing_style={"color": "white", "linewidth": 1.9, "zorder": 10},
    )

    # Radius of the geodesic circle, drawn as an arrowed arc labeled delta_min.
    radius_angle = math.radians(205.0)
    radius_direction = (
        math.cos(radius_angle) * tangent_one + math.sin(radius_angle) * tangent_two
    )
    beta = np.linspace(0.0, OFFSET_CIRCLE_RADIUS - 0.045, 121)
    radius_arc = 1.012 * (
        np.cos(beta)[:, None] * x_star[None, :]
        + np.sin(beta)[:, None] * radius_direction[None, :]
    )
    axis.plot(
        radius_arc[:, 0],
        radius_arc[:, 1],
        radius_arc[:, 2],
        color="black",
        linewidth=0.9,
        zorder=12,
    )
    tip_base = radius_arc[-1]
    tip_direction = (
        -math.sin(beta[-1]) * x_star + math.cos(beta[-1]) * radius_direction
    )
    axis.quiver(
        tip_base[0],
        tip_base[1],
        tip_base[2],
        *(0.045 * tip_direction),
        color="black",
        linewidth=0.9,
        arrow_length_ratio=0.9,
        normalize=False,
        zorder=12,
    )
    label_point = 1.04 * (
        math.cos(1.30 * OFFSET_CIRCLE_RADIUS) * x_star
        + math.sin(1.30 * OFFSET_CIRCLE_RADIUS) * radius_direction
    )
    axis.text(
        label_point[0],
        label_point[1],
        label_point[2],
        r"$\delta_{\min}$",
        fontsize=6.6,
        ha="center",
        va="top",
        zorder=13,
    )

    axis.set_box_aspect((1.0, 1.0, 1.0), zoom=1.8)
    axis.set_axis_off()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))

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
    cmap = plt.get_cmap("magma_r")
    norm = mpl.colors.Normalize(vmin=0.0, vmax=3.5)
    maps = np.load(RESULTS / "metric_maps.npz")

    figure = plt.figure(figsize=(7.05, 1.95), facecolor="white")
    # Explicit gap columns (wspace=0) so each inter-panel gap is set on its
    # own: the colorbar hangs directly off (c) inside the (c)-(d) gap, which
    # holds the bar, its label, one line of air, and (d)'s axis labels.
    grid = figure.add_gridspec(
        1,
        5,
        width_ratios=(1.46, 0.66, 1.72, 0.50, 1.72),
        left=0.008,
        right=0.995,
        bottom=0.215,
        top=0.9,
        wspace=0.0,
    )
    axis_a = figure.add_subplot(grid[0, 0], projection="3d")
    axis_b = figure.add_subplot(grid[0, 2])
    axis_c = figure.add_subplot(grid[0, 4])
    a_position = axis_a.get_position()
    color_axis = figure.add_axes(
        (
            a_position.x1 + 0.004,
            a_position.y0 + 0.14 * a_position.height,
            0.0095,
            0.72 * a_position.height,
        )
    )

    # (a) metric map on the sphere with the scan geometry.
    panel_a_sphere(axis_a, maps, cmap, norm)

    # (b) susceptibility along the through-node path.
    scan = np.load(RESULTS / "scan_offset_0.000.npz")
    s = scan["s"]
    axis_b.semilogy(
        s, scan["chi_vector"], color=VECTOR_COLOR, linewidth=1.1, zorder=4
    )
    estimator = scan["chi_hat_vector_d1e-03"]
    keep = ~np.isnan(estimator)
    axis_b.semilogy(
        s[keep][::150],
        estimator[keep][::150],
        linestyle="none",
        marker="o",
        markersize=2.4,
        markerfacecolor="none",
        markeredgecolor="black",
        markeredgewidth=0.6,
        zorder=5,
    )
    axis_b.semilogy(
        s, scan["chi_density"], color=DENSITY_COLOR, linewidth=1.1, zorder=3
    )
    axis_b.axhline(0.25, color="black", linewidth=0.8, linestyle=(0, (4, 2)), zorder=2)
    axis_b.set_xlim(-0.6, 0.6)
    axis_b.set_ylim(0.07, 2.0e3)
    axis_b.set_xticks([-0.5, 0.0, 0.5])
    axis_b.set_xlabel(r"$s$ (rad)", labelpad=1)
    axis_b.set_ylabel(r"$\chi_F$", labelpad=0)
    style_axes(axis_b)

    colorbar = figure.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=color_axis,
        ticks=[0, 1, 2, 3],
    )
    colorbar.ax.tick_params(length=2.5, width=0.65, pad=2)
    colorbar.outline.set_linewidth(0.7)
    colorbar.set_label(r"$\log_{10}4\lambda_{\max}$", rotation=90, labelpad=2)

    # (c) peak susceptibility against the scan's closest approach.
    peaks = np.genfromtxt(RESULTS / "scan_peaks.csv", delimiter=",", names=True)
    axis_c.semilogy(
        peaks["offset"],
        peaks["peak_chi_vector"],
        linestyle="-",
        linewidth=0.9,
        marker="o",
        markersize=2.0,
        color=VECTOR_COLOR,
        zorder=4,
    )
    axis_c.semilogy(
        peaks["offset"],
        peaks["max_chi_density"],
        linestyle="-",
        linewidth=0.9,
        marker="s",
        markersize=1.8,
        color=DENSITY_COLOR,
        zorder=3,
    )
    axis_c.axhline(0.25, color="black", linewidth=0.8, linestyle=(0, (4, 2)), zorder=2)
    axis_c.axvline(
        summary["static_epsilon_0.5"],
        color="black",
        linewidth=0.8,
        linestyle=(0, (1.5, 1.5)),
        zorder=2,
    )
    axis_c.set_xlim(-0.015, 0.315)
    axis_c.set_ylim(0.07, 2.0e3)
    axis_c.set_xticks([0.0, 0.1, 0.2, 0.3])
    axis_c.set_xlabel(r"$\delta_{\min}$ (rad)", labelpad=1)
    axis_c.set_ylabel(r"$\max_s\chi_F$", labelpad=0)
    style_axes(axis_c)

    figure.text(0.012, 0.965, "(a)", ha="left", va="top")
    for axis, label in ((axis_b, "(b)"), (axis_c, "(c)")):
        figure.text(
            axis.get_position().x0 - 0.028, 0.965, label, ha="left", va="top"
        )

    for suffix in (".pdf", ".png"):
        figure.savefig(args.output.with_suffix(suffix), dpi=args.dpi)
    plt.close(figure)
    print(f"saved {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
