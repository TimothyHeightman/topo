#!/usr/bin/env python3
"""Compare vector- and density-valued FNQS on the same one-qubit family."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VECTOR_SURFACE = (
    ROOT / "experiments/results/one_qubit_fnqs/surface_t241_p481.csv"
)
DEFAULT_DENSITY_SURFACE = (
    ROOT
    / "experiments/results/one_qubit_density_fnqs"
    / "density_surface_t241_p481.csv"
)
DEFAULT_OUTPUT = ROOT / "figures/figure_density_lift_comparison"


def load_surface(path: Path) -> dict[str, np.ndarray]:
    fields = ("theta_index", "phi_index", "theta", "phi", "infidelity")
    data = np.genfromtxt(path, delimiter=",", names=True, usecols=fields)
    n_theta = int(np.max(data["theta_index"])) + 1
    n_phi = int(np.max(data["phi_index"])) + 1
    return {
        name: np.asarray(data[name]).reshape((n_theta, n_phi))
        for name in fields
    }


def validate_matching_grids(
    vector_surface: dict[str, np.ndarray],
    density_surface: dict[str, np.ndarray],
) -> None:
    for coordinate in ("theta", "phi"):
        if vector_surface[coordinate].shape != density_surface[coordinate].shape:
            raise ValueError(f"Mismatched {coordinate} grid shapes.")
        if not np.allclose(
            vector_surface[coordinate], density_surface[coordinate], atol=1.0e-14
        ):
            raise ValueError(f"Mismatched {coordinate} coordinates.")


def draw_panel(
    axis: Any,
    surface: dict[str, np.ndarray],
    title: str,
    colormap: Any,
    normalization: Any,
) -> Any:
    theta = surface["theta"][:, 0] / np.pi
    phi = surface["phi"][0, :] / np.pi
    log_infidelity = np.log10(
        np.clip(surface["infidelity"], np.finfo(np.float64).tiny, None)
    )
    image = axis.imshow(
        log_infidelity,
        origin="lower",
        aspect="auto",
        extent=(float(phi[0]), float(phi[-1]), float(theta[0]), float(theta[-1])),
        cmap=colormap,
        norm=normalization,
        interpolation="bilinear",
        rasterized=True,
    )
    axis.set_title(title, pad=4)
    axis.set_xlabel(r"$\phi/\pi$")
    axis.set_ylabel(r"$\theta/\pi$")
    axis.set_xticks((0.0, 0.5, 1.0, 1.5, 2.0))
    axis.set_yticks((0.0, 0.5, 1.0))
    axis.tick_params(direction="in", length=3, pad=2)
    for spine in axis.spines.values():
        spine.set_linewidth(0.8)
    return image


def plot_comparison(
    vector_surface: dict[str, np.ndarray],
    density_surface: dict[str, np.ndarray],
    output_stem: Path,
) -> None:
    validate_matching_grids(vector_surface, density_surface)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    magma = plt.get_cmap("magma")
    green_tail = matplotlib.colors.LinearSegmentedColormap.from_list(
        "green_tail",
        ("#4BE27A", magma(0.0)),
        N=64,
    )(np.linspace(0.0, 1.0, 64))
    magma_body = magma(np.linspace(0.0, 1.0, 192))
    colormap = matplotlib.colors.ListedColormap(
        np.vstack((green_tail, magma_body)),
        name="green_tail_magma",
    )
    colormap.set_under("#4BE27A")
    normalization = matplotlib.colors.Normalize(vmin=-8.0, vmax=0.0)
    figure, axes = plt.subplots(1, 2, figsize=(6.6, 2.45))
    figure.subplots_adjust(left=0.08, right=0.89, bottom=0.19, top=0.87, wspace=0.28)
    image = draw_panel(
        axes[0],
        vector_surface,
        r"(a)",
        colormap,
        normalization,
    )
    draw_panel(
        axes[1],
        density_surface,
        r"(b)",
        colormap,
        normalization,
    )
    colorbar_axis = figure.add_axes((0.915, 0.19, 0.018, 0.68))
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label(r"$\log_{10}(1-F_0)$")
    colorbar.set_ticks((-8.0, -6.0, -4.0, -2.0, 0.0))
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in (".pdf", ".png"):
        figure.savefig(
            output_stem.with_suffix(extension),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.03,
        )
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-surface", type=Path, default=DEFAULT_VECTOR_SURFACE)
    parser.add_argument("--density-surface", type=Path, default=DEFAULT_DENSITY_SURFACE)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    plot_comparison(
        load_surface(args.vector_surface.resolve()),
        load_surface(args.density_surface.resolve()),
        args.output_stem.resolve(),
    )
    print(f"Wrote {args.output_stem.resolve().with_suffix('.pdf')}")
    print(f"Wrote {args.output_stem.resolve().with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
