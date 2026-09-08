#!/usr/bin/env python3
"""Assemble Fig. 2 from banked panel data without running JAX or training.

The two panel run scripts own all numerical work.  This script
only reads their NPZ/CSV/JSON artifacts and may therefore be rerun freely while
the visual design is refined.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS = SCRIPT_DIR / "results" / "figure2_dynamics"
DEFAULT_OUTPUT = PROJECT_ROOT / "figures" / "figure2_dynamics"
WORLDLINE_GREEN = "#2f7f5f"
TIME_ARROW_INTERVALS = ((0.26, 0.34), (0.86, 0.94))


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
            "Banked Fig. 2 data are missing:\n"
            f"{listing}\nRun the corresponding run_figure2*.py script first."
        )


def log_infidelity(values: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(values, 1.0e-3, 1.0))


def style_axes(axis: mpl.axes.Axes) -> None:
    axis.tick_params(direction="out", length=2.5, width=0.65, pad=1.5)
    for spine in axis.spines.values():
        spine.set_linewidth(0.7)


def panel_a_slices(
    figure: mpl.figure.Figure,
    slot: mpl.gridspec.SubplotSpec,
    cmap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
) -> mpl.axes.Axes:
    surface = np.load(RESULTS / "panel_a" / "finite_surface.npz")
    worldline = read_rows(RESULTS / "panel_a" / "nodal_worldline.csv")
    theta = surface["theta"]
    phi = surface["phi"]
    tau = surface["tau"]
    infidelity = surface["infidelity"]
    theta_star = np.asarray([row["theta"] for row in worldline])
    phi_star = np.asarray([row["phi_unwrapped"] for row in worldline])

    theta_center = float(np.median(theta_star))
    phi_center = float(np.median(phi_star))
    theta_half_width = 0.31
    phi_half_width = 0.36
    theta_mask = np.abs(theta - theta_center) <= theta_half_width
    phi_mask = np.abs(phi - phi_center) <= phi_half_width
    theta_local = theta[theta_mask]
    phi_local = phi[phi_mask]
    pp, tt = np.meshgrid(phi_local, theta_local)

    axis = figure.add_subplot(slot, projection="3d")
    chosen = np.linspace(0, len(tau) - 1, 6, dtype=int)
    for index in chosen:
        values = infidelity[index][np.ix_(theta_mask, phi_mask)]
        rgba = cmap(norm(log_infidelity(values)))
        strength = np.clip((log_infidelity(values) + 3.0) / 3.0, 0.0, 1.0)
        rgba[..., 3] = 0.015 + 0.94 * strength**1.25
        axis.plot_surface(
            pp,
            tt,
            np.full_like(pp, tau[index]),
            facecolors=rgba,
            linewidth=0.0,
            antialiased=False,
            shade=False,
            rcount=theta_local.size,
            ccount=phi_local.size,
            rasterized=True,
        )

    axis.plot(phi_star, theta_star, tau, color="black", linewidth=2.4, zorder=12)
    axis.plot(phi_star, theta_star, tau, color=WORLDLINE_GREEN, linewidth=1.25, zorder=13)
    for start_tau, end_tau in TIME_ARROW_INTERVALS:
        start_phi = np.interp(start_tau, tau, phi_star)
        start_theta = np.interp(start_tau, tau, theta_star)
        end_phi = np.interp(end_tau, tau, phi_star)
        end_theta = np.interp(end_tau, tau, theta_star)
        direction = (
            end_phi - start_phi,
            end_theta - start_theta,
            end_tau - start_tau,
        )
        axis.quiver(
            start_phi,
            start_theta,
            start_tau,
            *direction,
            color="black",
            linewidth=2.8,
            arrow_length_ratio=0.42,
            normalize=False,
            zorder=14,
        )
        axis.quiver(
            start_phi,
            start_theta,
            start_tau,
            *direction,
            color=WORLDLINE_GREEN,
            linewidth=1.35,
            arrow_length_ratio=0.42,
            normalize=False,
            zorder=15,
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
    projection = np.load(RESULTS / "panel_a" / "worldline_projection.npz")
    worldline = read_rows(RESULTS / "panel_a" / "nodal_worldline.csv")
    theta = projection["theta"]
    tau = projection["tau"]
    infidelity = projection["max_phi_infidelity"]
    worldline_tau = np.asarray([row["tau"] for row in worldline])
    theta_star = np.asarray([row["theta"] for row in worldline])
    theta_center = float(np.median(theta_star))
    relative_theta = (theta - theta_center) / np.pi
    theta_mask = np.abs(relative_theta) <= 0.08

    axis = figure.add_subplot(slot)
    axis.imshow(
        log_infidelity(infidelity[:, theta_mask]),
        origin="lower",
        extent=(relative_theta[theta_mask][0], relative_theta[theta_mask][-1], 0.0, 1.0),
        cmap=cmap,
        norm=norm,
        interpolation="bicubic",
        aspect="auto",
        rasterized=True,
    )
    projected_worldline = (theta_star - theta_center) / np.pi
    axis.plot(projected_worldline, worldline_tau, color="black", linewidth=2.2, zorder=5)
    axis.plot(
        projected_worldline,
        worldline_tau,
        color=WORLDLINE_GREEN,
        linewidth=1.15,
        zorder=6,
    )
    for start_tau, end_tau in TIME_ARROW_INTERVALS:
        start_theta = np.interp(start_tau, worldline_tau, projected_worldline)
        end_theta = np.interp(end_tau, worldline_tau, projected_worldline)
        axis.annotate(
            "",
            xy=(end_theta, end_tau),
            xytext=(start_theta, start_tau),
            arrowprops={
                "arrowstyle": "-|>",
                "color": "black",
                "linewidth": 2.6,
                "mutation_scale": 9.5,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=7,
        )
        axis.annotate(
            "",
            xy=(end_theta, end_tau),
            xytext=(start_theta, start_tau),
            arrowprops={
                "arrowstyle": "-|>",
                "color": WORLDLINE_GREEN,
                "linewidth": 1.25,
                "mutation_scale": 8.0,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=8,
        )
    axis.set_xticks([-0.05, 0.0, 0.05], [r"$-.05$", "0", r"$.05$"])
    axis.set_yticks([0.0, 0.5, 1.0], ["0", "1/2", "1"])
    axis.set_xlabel(r"$(\theta-\theta_*)/\pi$", labelpad=1)
    axis.set_ylabel(r"$t/T$", labelpad=0)
    style_axes(axis)
    return axis


def periodic_centered_surface(
    k: np.ndarray,
    t: np.ndarray,
    values: np.ndarray,
    k_star: float,
    t_star: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Drop duplicate endpoints, shift the physical node to the cell center, and
    # retain a regular periodic grid suitable for imshow.
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


def panel_c_periodic(
    figure: mpl.figure.Figure,
    slot: mpl.gridspec.SubplotSpec,
    cmap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
) -> mpl.axes.Axes:
    surface = np.load(RESULTS / "panel_b" / "periodic_surface.npz")
    nodes = read_rows(RESULTS / "panel_b" / "periodic_nodes.csv")
    if not nodes:
        raise RuntimeError("Panel (b) has no refined periodic node")
    node = nodes[0]
    shifted_k, shifted_t, centered = periodic_centered_surface(
        surface["k"],
        surface["t"],
        surface["infidelity"],
        node["k"],
        node["t"],
    )

    zoom = 0.06
    k_mask = np.abs(shifted_k) <= zoom
    t_mask = np.abs(shifted_t) <= zoom
    local = centered[np.ix_(k_mask, t_mask)]
    local_k = shifted_k[k_mask]
    local_t = shifted_t[t_mask]

    axis = figure.add_subplot(slot)
    axis.imshow(
        log_infidelity(local).T,
        origin="lower",
        extent=(local_k[0], local_k[-1], local_t[0], local_t[-1]),
        cmap=cmap,
        norm=norm,
        interpolation="bilinear",
        aspect="equal",
        rasterized=True,
    )
    axis.scatter(
        [0.0],
        [0.0],
        s=38,
        facecolor="none",
        edgecolor="black",
        linewidth=2.2,
        zorder=5,
    )
    axis.scatter(
        [0.0],
        [0.0],
        s=30,
        facecolor="none",
        edgecolor="#fff6b0",
        linewidth=1.2,
        zorder=6,
    )
    axis.set_xticks([-0.05, 0.0, 0.05], [r"$-.05$", "0", r"$.05$"])
    axis.set_yticks([-0.05, 0.0, 0.05], [r"$-.05$", "0", r"$.05$"])
    axis.set_xlabel(r"$(k-k_*)/2\pi$", labelpad=1)
    axis.set_ylabel(r"$(t-t_*)/T$", labelpad=0)
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
        RESULTS / "panel_a" / "finite_surface.npz",
        RESULTS / "panel_a" / "worldline_projection.npz",
        RESULTS / "panel_a" / "nodal_worldline.csv",
        RESULTS / "panel_a" / "summary.json",
        RESULTS / "panel_b" / "periodic_surface.npz",
        RESULTS / "panel_b" / "periodic_nodes.csv",
        RESULTS / "panel_b" / "summary.json",
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
    cmap = mpl.colormaps["magma"].copy()
    cmap.set_under("black")
    norm = mpl.colors.Normalize(vmin=-3.0, vmax=0.0)

    figure = plt.figure(figsize=(7.10, 2.55), facecolor="white")
    grid = figure.add_gridspec(
        1,
        4,
        width_ratios=(1.42, 0.96, 0.96, 0.045),
        left=0.018,
        right=0.965,
        bottom=0.17,
        top=0.89,
        wspace=0.27,
    )
    axis_a = panel_a_slices(figure, grid[0, 0], cmap, norm)
    axis_b = panel_b_profile(figure, grid[0, 1], cmap, norm)
    axis_c = panel_c_periodic(figure, grid[0, 2], cmap, norm)
    figure.text(axis_a.get_position().x0, 0.955, "(a)", ha="left", va="top")
    figure.text(axis_b.get_position().x0, 0.955, "(b)", ha="left", va="top")
    figure.text(axis_c.get_position().x0, 0.955, "(c)", ha="left", va="top")

    color_axis = figure.add_subplot(grid[0, 3])
    colorbar = figure.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=color_axis,
        ticks=[-3, -2, -1, 0],
    )
    colorbar.ax.tick_params(length=2.5, width=0.65, pad=2)
    colorbar.outline.set_linewidth(0.7)
    colorbar.set_label(r"$\log_{10}(1-F)$", rotation=90, labelpad=4)

    pdf_path = args.output.with_suffix(".pdf")
    png_path = args.output.with_suffix(".png")
    figure.savefig(pdf_path, dpi=args.dpi)
    figure.savefig(png_path, dpi=args.dpi)
    plt.close(figure)
    print(f"saved {pdf_path}")
    print(f"saved {png_path}")


if __name__ == "__main__":
    main()
