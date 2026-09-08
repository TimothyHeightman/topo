#!/usr/bin/env python3
"""Train a density-matrix-valued FNQS on the obstructed one-qubit family.

This is the controlled companion to ``one_qubit_fnqs_obstruction.py``.  Both
experiments use the same Hamiltonian family,

    H(n) = -n . sigma,       n in S^2,

the same deterministic training grid and optimizer, and the same
3-48-48-3 MLP (2,691 trainable real parameters by default).  Only the output
representation changes.  Here the network returns a Bloch vector ``r`` and

    rho(n) = [I + r(n) . sigma] / 2,

which is positive and trace one by construction.  The exact solution is the
globally continuous projector-valued map ``rho_0(n) = P_0(n)``.  All one-qubit
matrix elements and observables are evaluated exactly; no Monte Carlo sampling
is used.

The script checkpoints weights, banks an independent dense evaluation as CSV,
refines the worst-fidelity point continuously, and writes diagnostic plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
from tqdm.auto import tqdm

from one_qubit_fnqs_obstruction import (
    ExperimentConfig,
    Params,
    check_environment,
    display_to_hamiltonian_matrix,
    equal_area_training_grid,
    initialize_mlp,
    installed_versions,
    make_optimizer,
    mlp_apply,
    plotting_grid,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "one_qubit_density_fnqs"
BLOCH_REGULARIZER = 1.0e-6


def parameter_count(params: Params) -> int:
    """Return the number of trainable real scalars."""

    return int(
        sum(layer["weight"].size + layer["bias"].size for layer in params)
    )


def architecture(params: Params) -> list[int]:
    """Return the input width followed by every layer output width."""

    return [
        int(params[0]["weight"].shape[0]),
        *(int(layer["weight"].shape[1]) for layer in params),
    ]


def density_matrix(params: Params, points: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return a positive trace-one qubit density matrix and its Bloch vector.

    Dividing the raw three-vector by ``sqrt(|v|^2 + epsilon^2)`` maps all MLP
    outputs smoothly into the closed Bloch ball.  The small fixed regularizer
    avoids a singularity at ``v=0`` while changing unit-radius outputs only at
    order ``epsilon^2``.
    """

    raw_bloch = mlp_apply(params, points)
    denominator = jnp.sqrt(
        jnp.sum(raw_bloch**2, axis=-1, keepdims=True) + BLOCH_REGULARIZER**2
    )
    bloch = raw_bloch / denominator
    r_x, r_y, r_z = (bloch[..., index] for index in range(3))
    rho_00 = 0.5 * (1.0 + r_z)
    rho_11 = 0.5 * (1.0 - r_z)
    rho_01 = 0.5 * (r_x - 1j * r_y)
    rho = jnp.stack(
        (
            jnp.stack((rho_00, rho_01), axis=-1),
            jnp.stack((jnp.conj(rho_01), rho_11), axis=-1),
        ),
        axis=-2,
    )
    return rho, bloch


def observables(params: Params, points: jax.Array) -> dict[str, jax.Array]:
    """Evaluate exact density-matrix observables for ``H(n)=-n.sigma``."""

    rho, bloch = density_matrix(params, points)
    alignment = jnp.sum(points * bloch, axis=-1)
    energy = -alignment
    energy_error = 1.0 - alignment
    fidelity = 0.5 * (1.0 + alignment)
    infidelity = 0.5 * (1.0 - alignment)
    bloch_norm = jnp.linalg.norm(bloch, axis=-1)
    purity = 0.5 * (1.0 + bloch_norm**2)
    minimum_eigenvalue = 0.5 * (1.0 - bloch_norm)
    return {
        "rho": rho,
        "bloch": bloch,
        "energy": energy,
        "energy_error": energy_error,
        "fidelity": fidelity,
        "infidelity": infidelity,
        "purity": purity,
        "minimum_eigenvalue": minimum_eigenvalue,
    }


def mean_energy(params: Params, training_points: jax.Array) -> jax.Array:
    return jnp.mean(observables(params, training_points)["energy"])


def train(
    config: ExperimentConfig,
) -> tuple[Params, list[dict[str, float]], float]:
    """Train with the same deterministic objective used for the vector model."""

    training_points = equal_area_training_grid(
        config.train_cos_theta, config.train_phi
    )
    params = initialize_mlp(
        jax.random.PRNGKey(config.seed),
        input_dim=3,
        hidden_width=config.hidden_width,
        hidden_layers=config.hidden_layers,
    )
    optimizer = make_optimizer(config)
    optimizer_state = optimizer.init(params)

    @jax.jit
    def training_step(
        current_params: Params, current_state: optax.OptState
    ) -> tuple[Params, optax.OptState, jax.Array]:
        loss, gradients = jax.value_and_grad(mean_energy)(
            current_params, training_points
        )
        updates, next_state = optimizer.update(
            gradients, current_state, current_params
        )
        return (
            optax.apply_updates(current_params, updates),
            next_state,
            loss,
        )

    @jax.jit
    def training_metrics(current_params: Params) -> tuple[jax.Array, ...]:
        values = observables(current_params, training_points)
        return (
            jnp.mean(values["energy"]),
            jnp.mean(values["energy_error"]),
            jnp.max(values["energy_error"]),
            jnp.min(values["fidelity"]),
        )

    history: list[dict[str, float]] = []
    start_time = time.perf_counter()
    with tqdm(
        total=config.steps,
        desc=f"density FNQS ({training_points.shape[0]:,} sphere points)",
        unit="step",
        dynamic_ncols=True,
        mininterval=0.2,
    ) as progress:
        for step in range(1, config.steps + 1):
            params, optimizer_state, _ = training_step(params, optimizer_state)
            if step == 1 or step % config.log_every == 0 or step == config.steps:
                metrics = jax.device_get(training_metrics(params))
                elapsed = time.perf_counter() - start_time
                record = {
                    "step": float(step),
                    "mean_energy": float(metrics[0]),
                    "mean_energy_error": float(metrics[1]),
                    "max_grid_energy_error": float(metrics[2]),
                    "min_grid_fidelity": float(metrics[3]),
                    "elapsed_seconds": elapsed,
                }
                history.append(record)
                progress.set_postfix(
                    E=f"{record['mean_energy']:+.9f}",
                    mean_dE=f"{record['mean_energy_error']:.2e}",
                    max_dE=f"{record['max_grid_energy_error']:.2e}",
                    min_F0=f"{record['min_grid_fidelity']:.8f}",
                    refresh=False,
                )
            progress.update(1)

    return params, history, time.perf_counter() - start_time


def save_checkpoint(
    params: Params,
    path: Path,
    config: ExperimentConfig,
    versions: dict[str, str],
    elapsed_seconds: float,
) -> None:
    arrays: dict[str, np.ndarray] = {}
    for index, layer in enumerate(params):
        arrays[f"layer_{index}_weight"] = np.asarray(layer["weight"])
        arrays[f"layer_{index}_bias"] = np.asarray(layer["bias"])
    np.savez_compressed(path, **arrays)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": path.name,
        "representation": "rho=(I+r.sigma)/2 with a smoothly normalized neural Bloch vector",
        "hamiltonian": "H(n) = -n . sigma",
        "architecture": architecture(params),
        "parameter_count": parameter_count(params),
        "bloch_regularizer": BLOCH_REGULARIZER,
        "config": asdict(config),
        "versions": versions,
        "python": platform.python_version(),
        "training_elapsed_seconds": elapsed_seconds,
        "layer_count": len(params),
    }
    path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_checkpoint(path: Path) -> Params:
    metadata_path = path.with_suffix(".json")
    if not path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Missing {path} or {metadata_path}.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    layers: list[dict[str, jax.Array]] = []
    with np.load(path, allow_pickle=False) as checkpoint:
        for index in range(int(metadata["layer_count"])):
            layers.append(
                {
                    "weight": jnp.asarray(checkpoint[f"layer_{index}_weight"]),
                    "bias": jnp.asarray(checkpoint[f"layer_{index}_bias"]),
                }
            )
    return tuple(layers)


def write_history(history: Sequence[dict[str, float]], path: Path) -> None:
    fieldnames = list(history[0].keys())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def refine_worst_case(
    params: Params,
    display_points: np.ndarray,
    grid_infidelity: np.ndarray,
    display_to_hamiltonian: np.ndarray,
    seed_count: int,
    steps: int,
    learning_rate: float,
) -> dict[str, float]:
    """Continuously maximize infidelity from the worst dense-grid seeds."""

    count = min(seed_count, grid_infidelity.size)
    indices = np.argpartition(grid_infidelity, -count)[-count:]
    vectors = jnp.asarray(display_points[indices], dtype=jnp.float64)
    rotation = jnp.asarray(display_to_hamiltonian, dtype=jnp.float64)
    optimizer = optax.adam(learning_rate)
    optimizer_state = optimizer.init(vectors)

    def normalize(candidate: jax.Array) -> jax.Array:
        return candidate / jnp.linalg.norm(candidate, axis=-1, keepdims=True)

    @jax.jit
    def step(
        candidate: jax.Array, state: optax.OptState
    ) -> tuple[jax.Array, optax.OptState]:
        def objective(raw: jax.Array) -> jax.Array:
            points = normalize(raw) @ rotation
            return -jnp.sum(observables(params, points)["infidelity"])

        gradients = jax.grad(objective)(candidate)
        updates, next_state = optimizer.update(gradients, state)
        return normalize(optax.apply_updates(candidate, updates)), next_state

    for _ in range(steps):
        vectors, optimizer_state = step(vectors, optimizer_state)

    display_vectors = np.asarray(jax.device_get(normalize(vectors)))
    values = jax.device_get(observables(params, normalize(vectors) @ rotation))
    infidelity = np.asarray(values["infidelity"])
    best = int(np.argmax(infidelity))
    n_star = display_vectors[best]
    rho = np.asarray(values["rho"])[best]
    return {
        "theta": math.acos(float(np.clip(n_star[2], -1.0, 1.0))),
        "phi": math.atan2(float(n_star[1]), float(n_star[0]))
        % (2.0 * math.pi),
        "n_x": float(n_star[0]),
        "n_y": float(n_star[1]),
        "n_z": float(n_star[2]),
        "energy": float(np.asarray(values["energy"])[best]),
        "energy_error": float(np.asarray(values["energy_error"])[best]),
        "fidelity": float(np.asarray(values["fidelity"])[best]),
        "infidelity": float(infidelity[best]),
        "purity": float(np.asarray(values["purity"])[best]),
        "minimum_eigenvalue": float(
            np.asarray(values["minimum_eigenvalue"])[best]
        ),
        "rho_00_real": float(np.real(rho[0, 0])),
        "rho_01_real": float(np.real(rho[0, 1])),
        "rho_01_imag": float(np.imag(rho[0, 1])),
        "rho_11_real": float(np.real(rho[1, 1])),
    }


SURFACE_FIELDS = [
    "theta_index",
    "phi_index",
    "theta",
    "phi",
    "n_x",
    "n_y",
    "n_z",
    "energy",
    "energy_error",
    "ground_fidelity",
    "infidelity",
    "purity",
    "minimum_eigenvalue",
    "bloch_x",
    "bloch_y",
    "bloch_z",
    "rho_00_real",
    "rho_01_real",
    "rho_01_imag",
    "rho_11_real",
]


def write_surface(
    path: Path,
    theta: np.ndarray,
    phi: np.ndarray,
    display_points: np.ndarray,
    values: dict[str, np.ndarray],
) -> None:
    n_theta, n_phi = theta.shape
    theta_index, phi_index = np.meshgrid(
        np.arange(n_theta), np.arange(n_phi), indexing="ij"
    )
    rho = values["rho"].reshape((n_theta, n_phi, 2, 2))
    bloch = values["bloch"].reshape((n_theta, n_phi, 3))
    rows = np.column_stack(
        (
            theta_index.ravel(),
            phi_index.ravel(),
            theta.ravel(),
            phi.ravel(),
            display_points[:, 0],
            display_points[:, 1],
            display_points[:, 2],
            values["energy"].ravel(),
            values["energy_error"].ravel(),
            values["fidelity"].ravel(),
            values["infidelity"].ravel(),
            values["purity"].ravel(),
            values["minimum_eigenvalue"].ravel(),
            bloch[..., 0].ravel(),
            bloch[..., 1].ravel(),
            bloch[..., 2].ravel(),
            np.real(rho[..., 0, 0]).ravel(),
            np.real(rho[..., 0, 1]).ravel(),
            np.imag(rho[..., 0, 1]).ravel(),
            np.real(rho[..., 1, 1]).ravel(),
        )
    )
    formats = ["%d", "%d"] + ["%.17e"] * (len(SURFACE_FIELDS) - 2)
    np.savetxt(
        path,
        rows,
        delimiter=",",
        header=",".join(SURFACE_FIELDS),
        comments="",
        fmt=formats,
    )


def read_surface(path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    n_theta = int(np.max(data["theta_index"])) + 1
    n_phi = int(np.max(data["phi_index"])) + 1
    return {
        name: np.asarray(data[name]).reshape((n_theta, n_phi))
        for name in data.dtype.names or ()
    }


def plot_surface(surface: dict[str, np.ndarray], output_stem: Path) -> None:
    """Plot full-domain log infidelity and log energy error."""

    floor = 1.0e-14
    log_infidelity = np.log10(np.clip(surface["infidelity"], floor, None))
    log_energy_error = np.log10(
        np.clip(np.abs(surface["energy_error"]), floor, None)
    )
    theta = surface["theta"][:, 0] / np.pi
    phi = surface["phi"][0, :] / np.pi
    extent = (float(phi[0]), float(phi[-1]), float(theta[0]), float(theta[-1]))
    vmin = min(float(np.min(log_infidelity)), float(np.min(log_energy_error)))
    vmax = max(float(np.max(log_infidelity)), float(np.max(log_energy_error)))

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(6.6, 2.55), constrained_layout=True)
    for axis, field, label, colormap in (
        (axes[0], log_infidelity, r"$\log_{10}(1-F_0)$", "magma"),
        (
            axes[1],
            log_energy_error,
            r"$\log_{10}|\mathrm{Tr}(\hat\rho\hat H)-E_0|$",
            "viridis",
        ),
    ):
        image = axis.imshow(
            field,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            interpolation="bilinear",
            rasterized=True,
        )
        axis.set_xlabel(r"$\phi/\pi$")
        axis.set_ylabel(r"$\theta/\pi$")
        axis.set_xticks((0.0, 0.5, 1.0, 1.5, 2.0))
        axis.set_yticks((0.0, 0.5, 1.0))
        colorbar = figure.colorbar(image, ax=axis, pad=0.025)
        colorbar.set_label(label)
    for suffix in (".pdf", ".png"):
        figure.savefig(
            output_stem.with_suffix(suffix),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.03,
        )
    plt.close(figure)


def evaluate_and_bank(
    params: Params,
    output_dir: Path,
    n_theta: int,
    n_phi: int,
    coordinate_rotation_name: str,
    refine_seeds: int,
    refine_steps: int,
    refine_learning_rate: float,
) -> tuple[Path, dict[str, Any]]:
    theta, phi, display_points = plotting_grid(n_theta, n_phi)
    rotation = display_to_hamiltonian_matrix(coordinate_rotation_name)
    hamiltonian_points = display_points @ jnp.asarray(rotation)
    print(f"Evaluating {display_points.shape[0]:,} regular theta-phi points.")
    values = jax.device_get(jax.jit(observables)(params, hamiltonian_points))
    numpy_values = {key: np.asarray(value) for key, value in values.items()}
    numpy_display_points = np.asarray(display_points)
    worst_case = refine_worst_case(
        params,
        numpy_display_points,
        numpy_values["infidelity"].reshape(-1),
        rotation,
        refine_seeds,
        refine_steps,
        refine_learning_rate,
    )

    tag = f"t{n_theta}_p{n_phi}"
    surface_path = output_dir / f"density_surface_{tag}.csv"
    worst_path = output_dir / f"density_worst_case_{tag}.csv"
    write_surface(
        surface_path,
        theta,
        phi,
        numpy_display_points,
        numpy_values,
    )
    with worst_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(worst_case))
        writer.writeheader()
        writer.writerow(worst_case)
    plot_surface(
        read_surface(surface_path),
        output_dir / f"density_log_errors_{tag}",
    )

    trace_error = float(
        np.max(np.abs(np.trace(numpy_values["rho"], axis1=-2, axis2=-1) - 1.0))
    )
    hermiticity_error = float(
        np.max(
            np.abs(
                numpy_values["rho"]
                - np.swapaxes(np.conj(numpy_values["rho"]), -1, -2)
            )
        )
    )
    identity_error = float(
        np.max(
            np.abs(
                numpy_values["energy_error"]
                - 2.0 * numpy_values["infidelity"]
            )
        )
    )
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "representation": "density matrix",
        "architecture": architecture(params),
        "parameter_count": parameter_count(params),
        "matched_vector_model_parameter_count": parameter_count(params),
        "parameter_count_difference": 0,
        "surface_csv": surface_path.name,
        "worst_case_csv": worst_path.name,
        "grid": {"theta": n_theta, "phi": n_phi},
        "coordinate_rotation": coordinate_rotation_name,
        "worst_case": worst_case,
        "grid_max_infidelity": float(np.max(numpy_values["infidelity"])),
        "grid_max_energy_error": float(np.max(numpy_values["energy_error"])),
        "grid_min_fidelity": float(np.min(numpy_values["fidelity"])),
        "grid_min_density_eigenvalue": float(
            np.min(numpy_values["minimum_eigenvalue"])
        ),
        "grid_max_trace_error": trace_error,
        "grid_max_hermiticity_error": hermiticity_error,
        "max_abs_energy_error_minus_2_infidelity": identity_error,
    }
    summary_path = output_dir / f"density_summary_{tag}.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "Refined worst case: "
        f"theta={worst_case['theta']:.8f}, phi={worst_case['phi']:.8f}, "
        f"1-F0={worst_case['infidelity']:.6e}, "
        f"dE={worst_case['energy_error']:.6e}"
    )
    print(f"Banked dense surface: {surface_path}")
    return surface_path, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("all", "train", "evaluate", "plot"), default="all"
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--coordinate-rotation", choices=("none", "generic"), default="generic"
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--hidden-width", type=int, default=48)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--train-cos-theta", type=int, default=96)
    parser.add_argument("--train-phi", type=int, default=192)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--final-learning-rate", type=float, default=3.0e-5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-theta", type=int, default=241)
    parser.add_argument("--eval-phi", type=int, default=481)
    parser.add_argument("--refine-seeds", type=int, default=16)
    parser.add_argument("--refine-steps", type=int, default=750)
    parser.add_argument("--refine-learning-rate", type=float, default=2.0e-2)
    parser.add_argument("--surface-csv", type=Path, default=None)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--skip-version-check", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    versions = installed_versions() if args.skip_version_check else check_environment()
    if args.smoke_test:
        args.hidden_width = 8
        args.hidden_layers = 1
        args.train_cos_theta = 8
        args.train_phi = 16
        args.steps = 5
        args.log_every = 1
        args.eval_theta = 13
        args.eval_phi = 25
        args.refine_seeds = 2
        args.refine_steps = 10

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else DEFAULT_OUTPUT_DIR
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (
        args.checkpoint.resolve()
        if args.checkpoint is not None
        else output_dir / "density_weights.npz"
    )
    config = ExperimentConfig(
        seed=args.seed,
        hidden_width=args.hidden_width,
        hidden_layers=args.hidden_layers,
        train_cos_theta=args.train_cos_theta,
        train_phi=args.train_phi,
        steps=args.steps,
        learning_rate=args.learning_rate,
        final_learning_rate=args.final_learning_rate,
        gradient_clip=args.gradient_clip,
        log_every=args.log_every,
    )

    if args.mode in ("all", "train"):
        if checkpoint_path.exists() and not args.force_train:
            print(f"Checkpoint exists; training skipped: {checkpoint_path}")
        else:
            params, history, elapsed = train(config)
            save_checkpoint(params, checkpoint_path, config, versions, elapsed)
            write_history(history, output_dir / "density_training_history.csv")
            print(f"Saved checkpoint: {checkpoint_path}")
    if args.mode == "train":
        return 0

    if args.mode in ("all", "evaluate"):
        params = load_checkpoint(checkpoint_path)
        evaluate_and_bank(
            params,
            output_dir,
            args.eval_theta,
            args.eval_phi,
            args.coordinate_rotation,
            args.refine_seeds,
            args.refine_steps,
            args.refine_learning_rate,
        )
        return 0

    surface_path = (
        args.surface_csv.resolve()
        if args.surface_csv is not None
        else output_dir / f"density_surface_t{args.eval_theta}_p{args.eval_phi}.csv"
    )
    if not surface_path.exists():
        raise FileNotFoundError(f"Plot mode requires {surface_path}.")
    tag = surface_path.stem.removeprefix("density_surface_")
    plot_surface(read_surface(surface_path), output_dir / f"density_log_errors_{tag}")
    print(f"Replotted banked surface: {surface_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
