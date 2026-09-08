#!/usr/bin/env python3
"""Assemble the density-matrix worldline figure from banked data only.

Panels mirror Fig. 2(a) and 2(b) exactly: the same space-time window centered
on the vector model's refined charge -1 worldline, the same six time slices,
and the same azimuth-profiled (theta, t) heatmap.  The color scale is the
shared green-tail scale of the density-lift comparison figure, and the vector
model's worldline is reproduced as a dashed reference curve.  No training or
JAX evaluation happens here.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS = SCRIPT_DIR / "results" / "figure2_dynamics"
DEFAULT_OUTPUT = PROJECT_ROOT / "figures" / "figure_density_worldline"
GHOST_DASHES = (0.0, (2.6, 1.6))


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
        raise FileNotFoundError(
            "Banked density-worldline data are missing:\n"
            f"{listing}\nRun run_density_finite_time.py (and the vector "
            "run_figure2a_finite_time.py) first."
        )


def shared_colormap() -> mpl.colors.Colormap:
    """The green-tail magma scale used by the density-lift comparison."""

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


def bilinear_upsample(values: np.ndarray, factor: int) -> np.ndarray:
    """Densify a regular 2D field so surface quads fall below pixel scale."""

    rows = np.linspace(0.0, values.shape[0] - 1.0, factor * values.shape[0])
    columns = np.linspace(0.0, values.shape[1] - 1.0, factor * values.shape[1])
    along_rows = np.stack(
        [
            np.interp(rows, np.arange(values.shape[0]), values[:, index])
            for index in range(values.shape[1])
        ],
        axis=1,
    )
    return np.stack(
        [
            np.interp(columns, np.arange(values.shape[1]), along_rows[index])
            for index in range(along_rows.shape[0])
        ],
        axis=0,
    )


def style_axes(axis: mpl.axes.Axes) -> None:
    axis.tick_params(direction="out", length=2.5, width=0.65, pad=1.5)
    for spine in axis.spines.values():
        spine.set_linewidth(0.7)


def vector_worldline() -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    rows = read_rows(RESULTS / "panel_a" / "nodal_worldline.csv")
    tau = np.asarray([row["tau"] for row in rows])
    theta_star = np.asarray([row["theta"] for row in rows])
    phi_star = np.asarray([row["phi_unwrapped"] for row in rows])
    return tau, theta_star, phi_star, float(np.median(theta_star)), float(
        np.median(phi_star)
    )


def panel_a_slices(
    figure: mpl.figure.Figure,
    slot: mpl.gridspec.SubplotSpec,
    cmap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
) -> mpl.axes.Axes:
    surface = np.load(RESULTS / "panel_a_density" / "finite_surface.npz")
    worldline_tau, theta_star, phi_star, theta_center, phi_center = vector_worldline()
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

    axis = figure.add_subplot(slot, projection="3d")
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

    axis.plot(
        phi_star,
        theta_star,
        worldline_tau,
        color="black",
        linewidth=2.2,
        linestyle=GHOST_DASHES,
        zorder=13,
    )
    axis.plot(
        phi_star,
        theta_star,
        worldline_tau,
        color="white",
        linewidth=0.95,
        linestyle=GHOST_DASHES,
        zorder=14,
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
    axis.set_xlabel(r"$\phi-\phi_*$", labelpad=-1)
    axis.set_ylabel(r"$\theta-\theta_*$", labelpad=-1)
    axis.set_zlabel(r"$t/T$", labelpad=-5)
    axis.view_init(elev=23, azim=-53)
    axis.set_box_aspect((1.22, 1.0, 1.42))
    axis.grid(False)
    axis.xaxis.pane.set_alpha(0.0)
    axis.yaxis.pane.set_alpha(0.0)
    axis.zaxis.pane.set_alpha(0.0)
    axis.tick_params(length=0.0, pad=-2)
    return axis


def panel_b_profile(
    figure: mpl.figure.Figure,
    slot: mpl.gridspec.SubplotSpec,
    cmap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
) -> mpl.axes.Axes:
    projection = np.load(RESULTS / "panel_a_density" / "worldline_projection.npz")
    worldline_tau, theta_star, _, theta_center, _ = vector_worldline()
    theta = projection["theta"]
    infidelity = projection["max_phi_infidelity"]
    relative_theta = (theta - theta_center) / np.pi
    theta_mask = np.abs(relative_theta) <= 0.08

    axis = figure.add_subplot(slot)
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
    projected_worldline = (theta_star - theta_center) / np.pi
    axis.plot(
        projected_worldline,
        worldline_tau,
        color="black",
        linewidth=2.0,
        linestyle=GHOST_DASHES,
        zorder=6,
    )
    axis.plot(
        projected_worldline,
        worldline_tau,
        color="white",
        linewidth=0.85,
        linestyle=GHOST_DASHES,
        zorder=7,
    )
    axis.set_xticks([-0.05, 0.0, 0.05], [r"$-.05$", "0", r"$.05$"])
    axis.set_yticks([0.0, 0.5, 1.0], ["0", "1/2", "1"])
    axis.set_xlabel(r"$(\theta-\theta_*)/\pi$", labelpad=1)
    axis.set_ylabel(r"$t/T$", labelpad=0)
    style_axes(axis)
    return axis


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--dpi", type=int, default=600)
    return result


def main() -> None:
    args = parser().parse_args()
    required = [
        RESULTS / "panel_a_density" / "finite_surface.npz",
        RESULTS / "panel_a_density" / "worldline_projection.npz",
        RESULTS / "panel_a_density" / "summary.json",
        RESULTS / "panel_a" / "nodal_worldline.csv",
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

    figure = plt.figure(figsize=(5.05, 2.55), facecolor="white")
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=(1.42, 0.96, 0.045),
        left=0.025,
        right=0.95,
        bottom=0.17,
        top=0.89,
        wspace=0.27,
    )
    axis_a = panel_a_slices(figure, grid[0, 0], cmap, norm)
    axis_b = panel_b_profile(figure, grid[0, 1], cmap, norm)
    figure.text(axis_a.get_position().x0, 0.955, "(a)", ha="left", va="top")
    figure.text(axis_b.get_position().x0, 0.955, "(b)", ha="left", va="top")

    color_axis = figure.add_subplot(grid[0, 2])
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
    print(f"saved {pdf_path}")
    print(f"saved {png_path}")


if __name__ == "__main__":
    main()
