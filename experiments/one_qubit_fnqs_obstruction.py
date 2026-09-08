#!/usr/bin/env python3
"""Train and visualize a continuous one-qubit foundation NQS on S^2.

The Hamiltonian family is

    H(n) = -n . sigma,      n in S^2,

with exact ground energy E0 = -1.  A single neural network receives the
Cartesian Hamiltonian parameters n = (nx, ny, nz) and returns the two complex
computational-basis amplitudes of a normalized qubit.  The output is
parameterized directly on S^3, so no Monte-Carlo wavefunction sampling or
normalization estimate appears anywhere in the experiment.

The default ``all`` mode

1. trains on a deterministic equal-area grid uniform in cos(theta) and phi;
2. stores the trained weights in a compressed NumPy checkpoint;
3. evaluates an independent dense grid uniform in theta and phi;
4. banks every evaluated point in CSV form; and
5. writes energy-error and infidelity surfaces, together with the ground-state
   fidelity painted directly on S^2.

Re-running ``evaluate`` at a higher grid density loads the checkpoint and does
not retrain.  Re-running ``plot`` reads the banked CSV and does not evaluate the
network.

Examples
--------
Full default run:

    python experiments/one_qubit_fnqs_obstruction.py

Evaluate the saved model on a denser plotting grid:

    python experiments/one_qubit_fnqs_obstruction.py \
        --mode evaluate --eval-theta 361 --eval-phi 721

Replot a previously banked surface:

    python experiments/one_qubit_fnqs_obstruction.py \
        --mode plot \
        --surface-csv experiments/results/one_qubit_fnqs/surface_t241_p481.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
from tqdm.auto import tqdm


PINNED_VERSIONS = {
    "jax": "0.11.0",
    "jaxlib": "0.11.0",
    "optax": "0.2.8",
    "numpy": "2.5.2",
    "matplotlib": "3.11.1",
    "tqdm": "4.67.1",
}

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "one_qubit_fnqs"


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration stored alongside every trained checkpoint."""

    seed: int = 7
    hidden_width: int = 48
    hidden_layers: int = 2
    train_cos_theta: int = 96
    train_phi: int = 192
    steps: int = 5_000
    learning_rate: float = 3.0e-3
    final_learning_rate: float = 3.0e-5
    gradient_clip: float = 1.0
    log_every: int = 100


Params = tuple[dict[str, jax.Array], ...]


def installed_versions() -> dict[str, str]:
    """Return the dependency versions recorded in result metadata."""

    versions: dict[str, str] = {}
    for package in PINNED_VERSIONS:
        versions[package] = importlib.metadata.version(package)
    return versions


def check_environment() -> dict[str, str]:
    """Fail early unless the public Hamilton-Zero environment is active."""

    versions = installed_versions()
    mismatches = {
        name: (PINNED_VERSIONS[name], actual)
        for name, actual in versions.items()
        if actual != PINNED_VERSIONS[name]
    }
    if sys.version_info < (3, 12):
        mismatches["python"] = (">=3.12", platform.python_version())
    if mismatches:
        details = "\n".join(
            f"  {name}: expected {expected}, found {actual}"
            for name, (expected, actual) in mismatches.items()
        )
        raise RuntimeError(
            "This experiment is pinned to the public Hamilton-Zero environment.\n"
            f"Version mismatch:\n{details}"
        )
    return versions


def equal_area_training_grid(n_cos_theta: int, n_phi: int) -> jax.Array:
    """Return deterministic equal-area points on S^2.

    Midpoints equally spaced in u = cos(theta), combined with equally spaced
    azimuths, carry equal quadrature weights for the uniform sphere measure.
    The poles are excluded from training because each pole is a single point,
    not an entire azimuthal ring.
    """

    u = -1.0 + (jnp.arange(n_cos_theta, dtype=jnp.float64) + 0.5) * (
        2.0 / n_cos_theta
    )
    phi = 2.0 * jnp.pi * jnp.arange(n_phi, dtype=jnp.float64) / n_phi
    uu, pp = jnp.meshgrid(u, phi, indexing="ij")
    radial = jnp.sqrt(jnp.maximum(0.0, 1.0 - uu**2))
    points = jnp.stack(
        (radial * jnp.cos(pp), radial * jnp.sin(pp), uu), axis=-1
    )
    return points.reshape((-1, 3))


def plotting_grid(
    n_theta: int, n_phi: int
) -> tuple[np.ndarray, np.ndarray, jax.Array]:
    """Return a regular theta-phi grid and its Cartesian S^2 points."""

    theta = np.linspace(0.0, np.pi, n_theta, endpoint=True, dtype=np.float64)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=True, dtype=np.float64)
    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    points = np.stack(
        (
            np.sin(tt) * np.cos(pp),
            np.sin(tt) * np.sin(pp),
            np.cos(tt),
        ),
        axis=-1,
    )
    return tt, pp, jnp.asarray(points.reshape((-1, 3)))


def display_to_hamiltonian_matrix(name: str) -> np.ndarray:
    """Map displayed row vectors to the coefficients entering H.

    ``generic`` is a fixed SO(3) coordinate change, chosen before evaluation:
    a rotation by 1.0 rad about y followed by 0.8 rad about z.  It moves the
    computational-basis pole away from theta = 0 and the phi seam while leaving
    the Hamiltonian family and every scalar observable unchanged.
    """

    if name == "none":
        return np.eye(3, dtype=np.float64)
    if name != "generic":
        raise ValueError(f"Unknown coordinate rotation: {name}")
    z_angle = 0.8
    y_angle = 1.0
    rotate_z = np.asarray(
        (
            (math.cos(z_angle), -math.sin(z_angle), 0.0),
            (math.sin(z_angle), math.cos(z_angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    rotate_y = np.asarray(
        (
            (math.cos(y_angle), 0.0, math.sin(y_angle)),
            (0.0, 1.0, 0.0),
            (-math.sin(y_angle), 0.0, math.cos(y_angle)),
        )
    )
    # For row vectors q in the displayed frame, m = q @ rotation gives the
    # coefficients m entering H(m) = -m.sigma.
    return rotate_z @ rotate_y


def initialize_mlp(
    key: jax.Array, input_dim: int, hidden_width: int, hidden_layers: int
) -> Params:
    """Initialize a tanh MLP with three hyperspherical output angles."""

    dimensions = [input_dim] + [hidden_width] * hidden_layers + [3]
    keys = jax.random.split(key, len(dimensions) - 1)
    layers: list[dict[str, jax.Array]] = []
    for index, (fan_in, fan_out, layer_key) in enumerate(
        zip(dimensions[:-1], dimensions[1:], keys, strict=True)
    ):
        scale = math.sqrt(2.0 / (fan_in + fan_out))
        weights = scale * jax.random.normal(
            layer_key, (fan_in, fan_out), dtype=jnp.float64
        )
        bias = jnp.zeros((fan_out,), dtype=jnp.float64)
        if index == len(dimensions) - 2:
            # Start away from a hyperspherical coordinate pole, where two of
            # the three angular derivatives would vanish.
            bias = jnp.asarray((1.0, 1.0, 1.0), dtype=jnp.float64)
        layers.append({"weight": weights, "bias": bias})
    return tuple(layers)


def mlp_apply(params: Params, points: jax.Array) -> jax.Array:
    """Evaluate the shared Hamiltonian-conditioned neural network."""

    activations = points
    for layer_index, layer in enumerate(params):
        activations = activations @ layer["weight"] + layer["bias"]
        if layer_index < len(params) - 1:
            activations = jnp.tanh(activations)
    return activations


def normalized_qubit(params: Params, points: jax.Array) -> jax.Array:
    """Map network angles to two complex amplitudes on S^3 exactly.

    For angles (a, b, c), the four real S^3 coordinates are

        (cos a, sin a cos b, sin a sin b cos c, sin a sin b sin c).

    They are interpreted as the real and imaginary parts of amplitudes for
    the complete one-qubit basis |0>, |1>.
    """

    angles = mlp_apply(params, points)
    a, b, c = (angles[..., index] for index in range(3))
    sin_a = jnp.sin(a)
    sin_b = jnp.sin(b)
    q0 = jnp.cos(a)
    q1 = sin_a * jnp.cos(b)
    q2 = sin_a * sin_b * jnp.cos(c)
    q3 = sin_a * sin_b * jnp.sin(c)
    return jnp.stack((q0 + 1j * q1, q2 + 1j * q3), axis=-1)


def observables(params: Params, points: jax.Array) -> dict[str, jax.Array]:
    """Evaluate exact one-qubit energy and ground-state fidelity.

    For H(n) = -n.sigma, E0 = -1 and P0 = (I + n.sigma)/2.  Consequently

        <H> - E0 = 2 (1 - F0)

    pointwise.  The calculation below enumerates both basis amplitudes; there
    is no stochastic estimator.
    """

    psi = normalized_qubit(params, points)
    psi0, psi1 = psi[..., 0], psi[..., 1]
    overlap = jnp.conj(psi0) * psi1
    bloch = jnp.stack(
        (
            2.0 * jnp.real(overlap),
            2.0 * jnp.imag(overlap),
            jnp.abs(psi0) ** 2 - jnp.abs(psi1) ** 2,
        ),
        axis=-1,
    )
    alignment = jnp.sum(points * bloch, axis=-1)
    energy = -alignment
    energy_error = 1.0 - alignment
    fidelity = 0.5 * (1.0 + alignment)
    infidelity = 0.5 * (1.0 - alignment)
    return {
        "psi": psi,
        "bloch": bloch,
        "energy": energy,
        "energy_error": energy_error,
        "fidelity": fidelity,
        "infidelity": infidelity,
    }


def mean_energy(params: Params, training_points: jax.Array) -> jax.Array:
    """The deterministic full-batch variational objective."""

    return jnp.mean(observables(params, training_points)["energy"])


def make_optimizer(config: ExperimentConfig) -> optax.GradientTransformation:
    schedule = optax.cosine_decay_schedule(
        init_value=config.learning_rate,
        decay_steps=config.steps,
        alpha=config.final_learning_rate / config.learning_rate,
    )
    return optax.chain(
        optax.clip_by_global_norm(config.gradient_clip),
        optax.adam(schedule),
    )


def train(
    config: ExperimentConfig,
) -> tuple[Params, list[dict[str, float]], float]:
    """Train one continuous FNQS using deterministic full-batch gradients."""

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
        next_params = optax.apply_updates(current_params, updates)
        return next_params, next_state, loss

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
    description = f"training ({training_points.shape[0]:,} sphere points)"
    with tqdm(
        total=config.steps,
        desc=description,
        unit="step",
        dynamic_ncols=True,
        mininterval=0.2,
    ) as progress:
        for step in range(1, config.steps + 1):
            params, optimizer_state, _ = training_step(
                params, optimizer_state
            )
            should_log = (
                step == 1
                or step % config.log_every == 0
                or step == config.steps
            )
            if should_log:
                mean_e, mean_error, max_error, min_fidelity = training_metrics(
                    params
                )
                mean_e, mean_error, max_error, min_fidelity = jax.device_get(
                    (mean_e, mean_error, max_error, min_fidelity)
                )
                elapsed = time.perf_counter() - start_time
                record = {
                    "step": float(step),
                    "mean_energy": float(mean_e),
                    "mean_energy_error": float(mean_error),
                    "max_grid_energy_error": float(max_error),
                    "min_grid_fidelity": float(min_fidelity),
                    "elapsed_seconds": elapsed,
                }
                history.append(record)
                progress.set_postfix(
                    E=f"{record['mean_energy']:+.7f}",
                    mean_dE=f"{record['mean_energy_error']:.2e}",
                    max_dE=f"{record['max_grid_energy_error']:.6f}",
                    min_F0=f"{record['min_grid_fidelity']:.2e}",
                    refresh=False,
                )
            progress.update(1)

    elapsed = time.perf_counter() - start_time
    return params, history, elapsed


def save_checkpoint(
    params: Params,
    path: Path,
    config: ExperimentConfig,
    versions: dict[str, str],
    elapsed_seconds: float,
) -> None:
    """Save weights without pickle and write human-readable metadata."""

    arrays: dict[str, np.ndarray] = {}
    for index, layer in enumerate(params):
        arrays[f"layer_{index}_weight"] = np.asarray(layer["weight"])
        arrays[f"layer_{index}_bias"] = np.asarray(layer["bias"])
    np.savez_compressed(path, **arrays)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": path.name,
        "parameterization": "three neural hyperspherical angles mapped exactly to S^3",
        "hamiltonian": "H(n) = -n . sigma",
        "ground_energy": -1.0,
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
    """Load the compressed, non-pickle checkpoint."""

    metadata_path = path.with_suffix(".json")
    if not path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint {path} or metadata {metadata_path}. Run --mode train first."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    layer_count = int(metadata["layer_count"])
    layers: list[dict[str, jax.Array]] = []
    with np.load(path, allow_pickle=False) as checkpoint:
        for index in range(layer_count):
            layers.append(
                {
                    "weight": jnp.asarray(checkpoint[f"layer_{index}_weight"]),
                    "bias": jnp.asarray(checkpoint[f"layer_{index}_bias"]),
                }
            )
    return tuple(layers)


def write_history_csv(history: Sequence[dict[str, float]], path: Path) -> None:
    fieldnames = [
        "step",
        "mean_energy",
        "mean_energy_error",
        "max_grid_energy_error",
        "min_grid_fidelity",
        "elapsed_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def refine_x_star(
    params: Params,
    grid_points: np.ndarray,
    grid_infidelity: np.ndarray,
    display_to_hamiltonian: np.ndarray,
    seed_count: int,
    steps: int,
    learning_rate: float,
) -> dict[str, Any]:
    """Continuously maximize infidelity from the worst dense-grid seeds."""

    count = min(seed_count, grid_infidelity.size)
    seed_indices = np.argpartition(grid_infidelity, -count)[-count:]
    vectors = jnp.asarray(grid_points[seed_indices], dtype=jnp.float64)
    coordinate_rotation = jnp.asarray(
        display_to_hamiltonian, dtype=jnp.float64
    )
    optimizer = optax.adam(learning_rate)
    optimizer_state = optimizer.init(vectors)

    def normalize(candidate_vectors: jax.Array) -> jax.Array:
        return candidate_vectors / jnp.linalg.norm(
            candidate_vectors, axis=-1, keepdims=True
        )

    @jax.jit
    def refinement_step(
        candidate_vectors: jax.Array, state: optax.OptState
    ) -> tuple[jax.Array, optax.OptState]:
        def objective(raw_vectors: jax.Array) -> jax.Array:
            sphere_vectors = normalize(raw_vectors)
            hamiltonian_vectors = sphere_vectors @ coordinate_rotation
            return -jnp.sum(
                observables(params, hamiltonian_vectors)["infidelity"]
            )

        gradients = jax.grad(objective)(candidate_vectors)
        updates, next_state = optimizer.update(gradients, state)
        next_vectors = optax.apply_updates(candidate_vectors, updates)
        return normalize(next_vectors), next_state

    for _ in range(steps):
        vectors, optimizer_state = refinement_step(vectors, optimizer_state)

    refined_vectors = np.asarray(jax.device_get(normalize(vectors)))
    hamiltonian_vectors = normalize(vectors) @ coordinate_rotation
    values = jax.device_get(observables(params, hamiltonian_vectors))
    refined_infidelity = np.asarray(values["infidelity"])
    best = int(np.argmax(refined_infidelity))
    n_star = refined_vectors[best]
    theta_star = math.acos(float(np.clip(n_star[2], -1.0, 1.0)))
    phi_star = math.atan2(float(n_star[1]), float(n_star[0])) % (2.0 * math.pi)
    return {
        "theta": theta_star,
        "phi": phi_star,
        "n_x": float(n_star[0]),
        "n_y": float(n_star[1]),
        "n_z": float(n_star[2]),
        "energy": float(np.asarray(values["energy"])[best]),
        "energy_error": float(np.asarray(values["energy_error"])[best]),
        "fidelity": float(np.asarray(values["fidelity"])[best]),
        "infidelity": float(refined_infidelity[best]),
        "psi_0_real": float(np.real(np.asarray(values["psi"])[best, 0])),
        "psi_0_imag": float(np.imag(np.asarray(values["psi"])[best, 0])),
        "psi_1_real": float(np.real(np.asarray(values["psi"])[best, 1])),
        "psi_1_imag": float(np.imag(np.asarray(values["psi"])[best, 1])),
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
    "psi_0_real",
    "psi_0_imag",
    "psi_1_real",
    "psi_1_imag",
]


def write_surface_csv(
    path: Path,
    theta: np.ndarray,
    phi: np.ndarray,
    points: np.ndarray,
    values: dict[str, np.ndarray],
) -> None:
    """Bank every dense-grid prediction in a long-form CSV."""

    n_theta, n_phi = theta.shape
    theta_index, phi_index = np.meshgrid(
        np.arange(n_theta), np.arange(n_phi), indexing="ij"
    )
    psi = values["psi"].reshape((n_theta, n_phi, 2))
    rows = np.column_stack(
        (
            theta_index.ravel(),
            phi_index.ravel(),
            theta.ravel(),
            phi.ravel(),
            points[:, 0],
            points[:, 1],
            points[:, 2],
            values["energy"].ravel(),
            values["energy_error"].ravel(),
            values["fidelity"].ravel(),
            values["infidelity"].ravel(),
            np.real(psi[..., 0]).ravel(),
            np.imag(psi[..., 0]).ravel(),
            np.real(psi[..., 1]).ravel(),
            np.imag(psi[..., 1]).ravel(),
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


def write_x_star_csv(path: Path, x_star: dict[str, Any]) -> None:
    fieldnames = list(x_star.keys())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(x_star)


def read_x_star_csv(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    return {key: float(value) for key, value in row.items()}


def load_surface_csv(path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    n_theta = int(np.max(data["theta_index"])) + 1
    n_phi = int(np.max(data["phi_index"])) + 1
    return {
        name: np.asarray(data[name]).reshape((n_theta, n_phi))
        for name in data.dtype.names or ()
    }


def pi_ticks(axis: Any, extent: float) -> None:
    if math.isclose(extent, math.pi):
        ticks = [0.0, 0.5 * math.pi, math.pi]
        labels = [r"$0$", r"$\pi/2$", r"$\pi$"]
    else:
        ticks = [0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi, 2.0 * math.pi]
        labels = [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
    axis.set_ticks(ticks, labels=labels)


def plot_single_surface(
    theta: np.ndarray,
    phi: np.ndarray,
    z_values: np.ndarray,
    x_star: dict[str, float],
    z_star_key: str,
    z_label: str,
    cmap: str,
    z_limit: tuple[float, float],
    output_stem: Path,
) -> None:
    """Write one restrained publication-style 3D surface as PDF and PNG."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    figure = plt.figure(figsize=(6.2, 4.7))
    figure.subplots_adjust(left=0.03, right=0.87, bottom=0.04, top=0.98)
    axis = figure.add_subplot(111, projection="3d")
    surface = axis.plot_surface(
        theta,
        phi,
        z_values,
        cmap=cmap,
        vmin=z_limit[0],
        vmax=z_limit[1],
        rcount=min(theta.shape[0], 181),
        ccount=min(theta.shape[1], 241),
        linewidth=0.0,
        antialiased=True,
        shade=True,
    )
    surface.set_rasterized(True)
    axis.scatter(
        [x_star["theta"]],
        [x_star["phi"]],
        [x_star[z_star_key]],
        marker="*",
        s=110,
        c="#d62728",
        edgecolors="black",
        linewidths=0.6,
        depthshade=False,
        label=r"$x_*$",
        zorder=10,
    )
    axis.set_xlabel(r"$\theta$", labelpad=7)
    axis.set_ylabel(r"$\phi$", labelpad=7)
    axis.set_zlabel(z_label, labelpad=7)
    axis.set_xlim(0.0, math.pi)
    axis.set_ylim(0.0, 2.0 * math.pi)
    axis.set_zlim(*z_limit)
    pi_ticks(axis.xaxis, math.pi)
    pi_ticks(axis.yaxis, 2.0 * math.pi)
    axis.view_init(elev=28.0, azim=-132.0)
    axis.set_box_aspect((1.0, 1.35, 0.72))
    axis.legend(loc="upper left", frameon=False, fontsize=9, markerscale=0.75)
    colorbar = figure.colorbar(surface, ax=axis, shrink=0.68, pad=0.08)
    colorbar.set_label(z_label)
    figure.savefig(
        output_stem.with_suffix(".pdf"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    figure.savefig(
        output_stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(figure)


def sphere_view(x_star: dict[str, float]) -> tuple[float, float]:
    """Return an oblique view with x_* in the upper-right hemisphere."""

    star_azimuth = math.degrees(
        math.atan2(x_star["n_y"], x_star["n_x"])
    )
    return 40.0, star_azimuth - 60.0


def fidelity_colormap() -> matplotlib.colors.LinearSegmentedColormap:
    """Resolve high fidelities while retaining a black zero-fidelity core."""

    return matplotlib.colors.LinearSegmentedColormap.from_list(
        "stratified_fidelity",
        (
            (0.00, "#000000"),
            (0.25, "#520000"),
            (0.50, "#d7191c"),
            (0.70, "#f46d43"),
            (0.90, "#ffe066"),
            (1.00, "#168344"),
        ),
    )


def draw_fidelity_sphere(
    axis: Any,
    surface: dict[str, np.ndarray],
    x_star: dict[str, float],
    normalization: Any,
    colormap: Any,
    zoom_window: dict[str, Any] | None = None,
    color_values: np.ndarray | None = None,
    mark_x_star: bool = True,
    zoom_outline_color: str = "black",
) -> np.ndarray:
    """Draw the colored parameter sphere and the computed obstruction point."""

    if color_values is None:
        color_values = surface["ground_fidelity"]
    facecolors = colormap(normalization(color_values))
    sphere = axis.plot_surface(
        surface["n_x"],
        surface["n_y"],
        surface["n_z"],
        facecolors=facecolors,
        rcount=surface["n_x"].shape[0],
        ccount=surface["n_x"].shape[1],
        linewidth=0.0,
        antialiased=False,
        shade=False,
    )
    sphere.set_rasterized(True)

    n_star = np.asarray(
        (x_star["n_x"], x_star["n_y"], x_star["n_z"]), dtype=np.float64
    )
    if mark_x_star:
        marker_position = 1.015 * n_star
        axis.scatter(
            [marker_position[0]],
            [marker_position[1]],
            [marker_position[2]],
            marker="o",
            s=75,
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
            depthshade=False,
            zorder=10,
        )

        label_reference = np.asarray((0.0, 0.0, 1.0))
        if abs(float(n_star @ label_reference)) > 0.9:
            label_reference = np.asarray((0.0, 1.0, 0.0))
        label_tangent = label_reference - (n_star @ label_reference) * n_star
        label_tangent /= np.linalg.norm(label_tangent)
        label_position = 1.02 * n_star + 0.14 * label_tangent
        axis.text(
            *label_position,
            r"$x_*$",
            color="black",
            fontsize=11,
            horizontalalignment="center",
            verticalalignment="bottom",
            zorder=11,
        )

    if zoom_window is not None:
        theta_values = zoom_window["theta"]
        phi_values = zoom_window["phi"]
        boundary_segments = (
            (np.full_like(phi_values, theta_values[0]), phi_values),
            (np.full_like(phi_values, theta_values[-1]), phi_values),
            (theta_values, np.full_like(theta_values, phi_values[0])),
            (theta_values, np.full_like(theta_values, phi_values[-1])),
        )
        for theta_edge, phi_edge in boundary_segments:
            boundary = np.stack(
                (
                    np.sin(theta_edge) * np.cos(phi_edge),
                    np.sin(theta_edge) * np.sin(phi_edge),
                    np.cos(theta_edge),
                ),
                axis=0,
            )
            axis.plot(
                1.008 * boundary[0],
                1.008 * boundary[1],
                1.008 * boundary[2],
                color=zoom_outline_color,
                linewidth=1.0,
                linestyle=(0, (3, 2)),
                zorder=9,
            )

    axis.set_xlabel(r"$n_x$", labelpad=5)
    axis.set_ylabel(r"$n_y$", labelpad=5)
    axis.set_zlabel(r"$n_z$", labelpad=5)
    axis.set_xlim(-1.05, 1.05)
    axis.set_ylim(-1.05, 1.05)
    axis.set_zlim(-1.05, 1.05)
    axis.set_xticks((-1.0, 0.0, 1.0))
    axis.set_yticks((-1.0, 0.0, 1.0))
    axis.set_zticks((-1.0, 0.0, 1.0))
    axis.set_box_aspect((1.0, 1.0, 1.0), zoom=1.7)
    axis.set_proj_type("ortho")
    view_elevation, view_azimuth = sphere_view(x_star)
    axis.view_init(elev=view_elevation, azim=view_azimuth)
    return n_star


def add_fidelity_colorbar(
    figure: Any,
    axis: Any,
    normalization: Any,
    colormap: Any,
    shrink: float = 0.72,
    colorbar_axis: Any | None = None,
    label: str = r"Ground-state fidelity $F_0$",
    ticks: Sequence[float] = (0.0, 0.5, 0.9, 1.0),
) -> None:
    color_scale = matplotlib.cm.ScalarMappable(
        norm=normalization, cmap=colormap
    )
    color_scale.set_array([])
    if colorbar_axis is None:
        colorbar = figure.colorbar(
            color_scale, ax=axis, shrink=shrink, pad=0.08
        )
    else:
        colorbar = figure.colorbar(color_scale, cax=colorbar_axis)
    colorbar.set_label(label)
    colorbar.set_ticks(ticks)


def save_figure_pair(figure: Any, output_stem: Path) -> None:
    for extension in (".pdf", ".png"):
        figure.savefig(
            output_stem.with_suffix(extension),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.03,
        )
    plt.close(figure)


def add_orientation_triad(
    figure: Any,
    x_star: dict[str, float],
    bounds: tuple[float, float, float, float],
) -> None:
    """Add a compact axis triad without surrounding the sphere in a box."""

    axis = figure.add_axes(bounds, projection="3d", zorder=30)
    for direction in np.eye(3):
        axis.quiver(
            0.0,
            0.0,
            0.0,
            direction[0],
            direction[1],
            direction[2],
            color="black",
            linewidth=1.0,
            arrow_length_ratio=0.16,
        )
    axis.text(1.12, 0.0, 0.0, r"$n_x$", fontsize=9)
    axis.text(0.0, 1.12, 0.0, r"$n_y$", fontsize=9)
    axis.text(0.0, 0.0, 1.12, r"$n_z$", fontsize=9)
    axis.set_xlim(0.0, 1.2)
    axis.set_ylim(0.0, 1.2)
    axis.set_zlim(0.0, 1.2)
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.set_proj_type("ortho")
    view_elevation, view_azimuth = sphere_view(x_star)
    axis.view_init(elev=view_elevation, azim=view_azimuth)
    axis.patch.set_alpha(0.0)
    axis.set_axis_off()


def plot_fidelity_sphere(
    surface: dict[str, np.ndarray],
    x_star: dict[str, float],
    output_stem: Path,
) -> None:
    """Paint the exact ground-state fidelity on the parameter sphere S^2."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    normalization = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
    colormap = fidelity_colormap()
    figure = plt.figure(figsize=(5.6, 5.0))
    figure.subplots_adjust(left=0.02, right=0.86, bottom=0.02, top=0.98)
    axis = figure.add_subplot(111, projection="3d")
    draw_fidelity_sphere(
        axis, surface, x_star, normalization, colormap
    )
    axis.set_axis_off()
    add_orientation_triad(figure, x_star, (0.075, 0.17, 0.15, 0.15))
    add_fidelity_colorbar(figure, axis, normalization, colormap)
    save_figure_pair(figure, output_stem)


def fidelity_zoom_radius(
    surface: dict[str, np.ndarray], n_star: np.ndarray
) -> float:
    """Choose a local cap that contains the low-fidelity core and context."""

    points = np.column_stack(
        tuple(surface[name].ravel() for name in ("n_x", "n_y", "n_z"))
    )
    distance = np.arccos(np.clip(points @ n_star, -1.0, 1.0))
    fidelity = surface["ground_fidelity"].ravel()
    core_distance = distance[fidelity < 0.75]
    theta_spacing = np.pi / max(surface["theta"].shape[0] - 1, 1)
    phi_spacing = 2.0 * np.pi / max(surface["phi"].shape[1] - 1, 1)
    minimum_radius = 6.0 * max(theta_spacing, phi_spacing)
    if core_distance.size:
        proposed_radius = 1.8 * float(np.max(core_distance))
    else:
        proposed_radius = minimum_radius
    return float(np.clip(proposed_radius, minimum_radius, 0.35))


def fidelity_zoom_window(
    surface: dict[str, np.ndarray],
    x_star: dict[str, float],
    zoom_radius: float,
    coordinate_aspect: float | None = None,
) -> dict[str, Any]:
    """Return the actual sampled theta-phi rectangle used by the inset."""

    theta_star = float(x_star["theta"])
    phi_star = float(x_star["phi"])
    theta_values = surface["theta"][:, 0]
    phi_values = surface["phi"][0, :]
    phi_half_width = (
        zoom_radius / max(math.sin(theta_star), 0.2)
        if coordinate_aspect is None
        else coordinate_aspect * zoom_radius
    )
    wrapped_phi = (phi_values - phi_star + np.pi) % (2.0 * np.pi) - np.pi
    theta_indices = np.flatnonzero(
        np.abs(theta_values - theta_star) <= 1.05 * zoom_radius
    )
    phi_indices = np.flatnonzero(
        np.abs(wrapped_phi) <= 1.05 * phi_half_width
    )
    phi_indices = phi_indices[np.argsort(wrapped_phi[phi_indices])]
    return {
        "theta_indices": theta_indices,
        "phi_indices": phi_indices,
        "theta": theta_values[theta_indices],
        "phi": phi_star + wrapped_phi[phi_indices],
    }


def draw_fidelity_zoom(
    axis: Any,
    surface: dict[str, np.ndarray],
    zoom_window: dict[str, Any],
    normalization: Any,
    colormap: Any,
    color_values: np.ndarray | None = None,
) -> Any:
    """Draw an absolute theta-phi window around x_*."""

    theta_indices = zoom_window["theta_indices"]
    phi_indices = zoom_window["phi_indices"]
    displayed_phi = zoom_window["phi"]
    displayed_theta = zoom_window["theta"]
    if color_values is None:
        color_values = surface["ground_fidelity"]
    zoom_values = color_values[np.ix_(theta_indices, phi_indices)]

    image = axis.imshow(
        zoom_values,
        origin="lower",
        aspect="auto",
        extent=(
            float(displayed_phi[0]),
            float(displayed_phi[-1]),
            float(displayed_theta[0]),
            float(displayed_theta[-1]),
        ),
        cmap=colormap,
        norm=normalization,
        interpolation="bilinear",
        rasterized=True,
    )
    axis.set_xlim(float(displayed_phi[0]), float(displayed_phi[-1]))
    axis.set_ylim(float(displayed_theta[0]), float(displayed_theta[-1]))
    axis.set_xlabel(r"$\phi$ (rad)", labelpad=2)
    axis.set_ylabel(r"$\theta$ (rad)", labelpad=2)
    axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    axis.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    axis.tick_params(direction="in", length=3, pad=2, labelsize=8)
    for spine in axis.spines.values():
        spine.set_linewidth(0.8)
    return image


def plot_fidelity_sphere_with_zoom(
    surface: dict[str, np.ndarray],
    x_star: dict[str, float],
    output_stem: Path,
) -> None:
    """Write a separate S^2 figure with a local fidelity inset."""

    normalization = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
    colormap = fidelity_colormap()
    n_star = np.asarray(
        (x_star["n_x"], x_star["n_y"], x_star["n_z"]), dtype=np.float64
    )
    zoom_radius = fidelity_zoom_radius(surface, n_star)
    zoom_window = fidelity_zoom_window(surface, x_star, zoom_radius)

    figure = plt.figure(figsize=(7.2, 5.2))
    sphere_axis = figure.add_axes((0.00, 0.02, 0.74, 0.96), projection="3d")
    draw_fidelity_sphere(
        sphere_axis,
        surface,
        x_star,
        normalization,
        colormap,
        zoom_window=zoom_window,
        mark_x_star=False,
    )
    sphere_axis.set_axis_off()
    add_orientation_triad(figure, x_star, (0.075, 0.16, 0.14, 0.14))
    zoom_axis = figure.add_axes(
        (0.635, 0.59, 0.25, 0.31), facecolor="white", zorder=20
    )
    draw_fidelity_zoom(
        zoom_axis,
        surface,
        zoom_window,
        normalization,
        colormap,
    )
    colorbar_axis = figure.add_axes((0.925, 0.16, 0.022, 0.68))
    add_fidelity_colorbar(
        figure,
        sphere_axis,
        normalization,
        colormap,
        colorbar_axis=colorbar_axis,
    )
    save_figure_pair(figure, output_stem)


def draw_log_energy_heatmap(
    axis: Any,
    surface: dict[str, np.ndarray],
    zoom_window: dict[str, Any],
) -> Any:
    """Draw the local logarithmic energy error on the infidelity window."""

    log_energy_error = np.log10(np.abs(surface["energy_error"]))
    normalization = matplotlib.colors.Normalize(
        vmin=-3.0,
        vmax=float(np.max(log_energy_error)),
    )
    return draw_fidelity_zoom(
        axis,
        surface,
        zoom_window,
        normalization,
        plt.get_cmap("viridis"),
        color_values=log_energy_error,
    )


def plot_log_infidelity_sphere_with_zoom(
    surface: dict[str, np.ndarray],
    x_star: dict[str, float],
    output_stem: Path,
) -> None:
    """Write the same panel with log10 infidelity as the color field."""

    infidelity_floor = 1.0e-9
    log_infidelity = np.log10(
        np.clip(surface["infidelity"], infidelity_floor, 1.0)
    )
    normalization = matplotlib.colors.Normalize(vmin=-3.0, vmax=0.0)
    colormap = plt.get_cmap("magma").copy()
    colormap.set_under("#000000")
    n_star = np.asarray(
        (x_star["n_x"], x_star["n_y"], x_star["n_z"]), dtype=np.float64
    )
    zoom_radius = min(2.8 * fidelity_zoom_radius(surface, n_star), 0.58)
    zoom_window = fidelity_zoom_window(
        surface,
        x_star,
        zoom_radius,
        coordinate_aspect=9.0 / 16.0,
    )

    # Match the shallow footprint of the original PRL schematic: the sphere,
    # expanded flat view, and colorbar share one compact horizontal strip.
    figure = plt.figure(figsize=(7.1, 2.55))
    sphere_axis = figure.add_axes((-0.080, 0.02, 0.52, 0.96), projection="3d")
    draw_fidelity_sphere(
        sphere_axis,
        surface,
        x_star,
        normalization,
        colormap,
        zoom_window=zoom_window,
        color_values=log_infidelity,
        mark_x_star=False,
        zoom_outline_color="white",
    )
    sphere_axis.set_axis_off()
    add_orientation_triad(figure, x_star, (-0.075, 0.055, 0.085, 0.26))
    zoom_axis = figure.add_axes(
        (0.43, 0.14, 0.158, 0.78), facecolor="white", zorder=20
    )
    draw_fidelity_zoom(
        zoom_axis,
        surface,
        zoom_window,
        normalization,
        colormap,
        color_values=log_infidelity,
    )
    colorbar_axis = figure.add_axes((0.595, 0.14, 0.018, 0.78))
    add_fidelity_colorbar(
        figure,
        sphere_axis,
        normalization,
        colormap,
        colorbar_axis=colorbar_axis,
        label=r"$\log_{10}(1-F_0)$",
        ticks=(-3.0, -2.0, -1.0, 0.0),
    )
    energy_axis = figure.add_axes(
        (0.76, 0.14, 0.158, 0.78), facecolor="white", zorder=20
    )
    energy_image = draw_log_energy_heatmap(
        energy_axis,
        surface,
        zoom_window,
    )
    energy_colorbar_axis = figure.add_axes((0.94, 0.14, 0.018, 0.78))
    energy_colorbar = figure.colorbar(energy_image, cax=energy_colorbar_axis)
    energy_colorbar.set_label(
        r"$\log_{10}|\langle\hat H\rangle-E_0|$"
    )
    energy_colorbar.set_ticks((-3.0, -2.0, -1.0, 0.0))
    save_figure_pair(figure, output_stem)


def plot_from_csv(surface_csv: Path, x_star_csv: Path, output_dir: Path) -> None:
    surface = load_surface_csv(surface_csv)
    x_star = read_x_star_csv(x_star_csv)
    suffix = surface_csv.stem.removeprefix("surface_")
    plot_single_surface(
        surface["theta"],
        surface["phi"],
        surface["energy_error"],
        x_star,
        z_star_key="energy_error",
        z_label=r"$\langle\hat H\rangle-E_0$",
        cmap="viridis",
        z_limit=(0.0, 2.0),
        output_stem=output_dir / f"energy_gap_surface_{suffix}",
    )
    plot_single_surface(
        surface["theta"],
        surface["phi"],
        surface["infidelity"],
        x_star,
        z_star_key="infidelity",
        z_label=r"$1-F_0$",
        cmap="magma",
        z_limit=(0.0, 1.0),
        output_stem=output_dir / f"infidelity_surface_{suffix}",
    )
    plot_fidelity_sphere(
        surface,
        x_star,
        output_stem=output_dir / f"ground_fidelity_sphere_{suffix}",
    )
    plot_fidelity_sphere_with_zoom(
        surface,
        x_star,
        output_stem=output_dir / f"ground_fidelity_sphere_zoom_{suffix}",
    )
    plot_log_infidelity_sphere_with_zoom(
        surface,
        x_star,
        output_stem=output_dir / f"log_infidelity_sphere_zoom_{suffix}",
    )


def evaluate_and_bank(
    params: Params,
    output_dir: Path,
    n_theta: int,
    n_phi: int,
    coordinate_rotation_name: str,
    refine_seeds: int,
    refine_steps: int,
    refine_learning_rate: float,
) -> tuple[Path, Path, dict[str, Any]]:
    """Evaluate a dense grid, refine x_*, write CSVs, and draw both surfaces."""

    theta, phi, points = plotting_grid(n_theta, n_phi)
    coordinate_rotation = display_to_hamiltonian_matrix(
        coordinate_rotation_name
    )
    hamiltonian_points = points @ jnp.asarray(coordinate_rotation)
    print(f"Evaluating {points.shape[0]:,} regular theta-phi grid points.")
    values = jax.device_get(jax.jit(observables)(params, hamiltonian_points))
    numpy_values = {key: np.asarray(value) for key, value in values.items()}
    numpy_points = np.asarray(points)

    x_star = refine_x_star(
        params,
        numpy_points,
        numpy_values["infidelity"].reshape(-1),
        display_to_hamiltonian=coordinate_rotation,
        seed_count=refine_seeds,
        steps=refine_steps,
        learning_rate=refine_learning_rate,
    )
    tag = f"t{n_theta}_p{n_phi}"
    surface_csv = output_dir / f"surface_{tag}.csv"
    x_star_csv = output_dir / f"x_star_{tag}.csv"
    write_surface_csv(
        surface_csv, theta, phi, numpy_points, numpy_values
    )
    write_x_star_csv(x_star_csv, x_star)
    plot_from_csv(surface_csv, x_star_csv, output_dir)

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
        "surface_csv": surface_csv.name,
        "x_star_csv": x_star_csv.name,
        "grid": {"theta": n_theta, "phi": n_phi},
        "coordinate_rotation": coordinate_rotation_name,
        "display_to_hamiltonian_row_matrix": coordinate_rotation.tolist(),
        "x_star": x_star,
        "max_abs_energy_error_minus_2_infidelity": identity_error,
    }
    (output_dir / f"summary_{tag}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "Refined x_*: "
        f"theta={x_star['theta']:.8f}, phi={x_star['phi']:.8f}, "
        f"1-F0={x_star['infidelity']:.12f}, "
        f"<H>-E0={x_star['energy_error']:.12f}"
    )
    print(f"Banked dense surface: {surface_csv}")
    return surface_csv, x_star_csv, x_star


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=("all", "train", "evaluate", "plot"),
        default="all",
        help="all=train if needed, then evaluate and plot (default)",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="checkpoint to load in evaluate mode (defaults inside output-dir)",
    )
    parser.add_argument(
        "--coordinate-rotation",
        choices=("none", "generic"),
        default="generic",
        help="fixed displayed coordinate frame used during evaluation",
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--hidden-width", type=int, default=48)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--train-cos-theta", type=int, default=96)
    parser.add_argument("--train-phi", type=int, default=192)
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--final-learning-rate", type=float, default=3.0e-5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-theta", type=int, default=241)
    parser.add_argument("--eval-phi", type=int, default=481)
    parser.add_argument("--refine-seeds", type=int, default=16)
    parser.add_argument("--refine-steps", type=int, default=750)
    parser.add_argument("--refine-learning-rate", type=float, default=2.0e-2)
    parser.add_argument(
        "--surface-csv",
        type=Path,
        default=None,
        help="banked surface to use in plot mode",
    )
    parser.add_argument(
        "--x-star-csv",
        type=Path,
        default=None,
        help="matching x_star CSV to use in plot mode",
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="overwrite an existing checkpoint",
    )
    parser.add_argument(
        "--skip-version-check",
        action="store_true",
        help="allow a non-Hamilton-Zero environment (not recommended)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="use a tiny configuration to validate the complete pipeline",
    )
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
    if args.checkpoint is not None and args.mode in ("all", "train"):
        raise ValueError("--checkpoint is only valid with --mode evaluate.")
    checkpoint_path = (
        args.checkpoint.resolve()
        if args.checkpoint is not None
        else output_dir / "weights.npz"
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
            print(
                f"Checkpoint already exists at {checkpoint_path}; training skipped. "
                "Use --force-train to replace it."
            )
        else:
            params, history, elapsed_seconds = train(config)
            save_checkpoint(
                params,
                checkpoint_path,
                config,
                versions,
                elapsed_seconds,
            )
            write_history_csv(history, output_dir / "training_history.csv")
            print(f"Saved checkpoint: {checkpoint_path}")

    if args.mode == "train":
        return 0

    if args.mode in ("all", "evaluate"):
        params = load_checkpoint(checkpoint_path)
        evaluate_and_bank(
            params,
            output_dir,
            n_theta=args.eval_theta,
            n_phi=args.eval_phi,
            coordinate_rotation_name=args.coordinate_rotation,
            refine_seeds=args.refine_seeds,
            refine_steps=args.refine_steps,
            refine_learning_rate=args.refine_learning_rate,
        )
        return 0

    surface_csv = (
        args.surface_csv.resolve()
        if args.surface_csv is not None
        else output_dir / f"surface_t{args.eval_theta}_p{args.eval_phi}.csv"
    )
    x_star_csv = (
        args.x_star_csv.resolve()
        if args.x_star_csv is not None
        else surface_csv.with_name(
            surface_csv.name.replace("surface_", "x_star_", 1)
        )
    )
    if not surface_csv.exists() or not x_star_csv.exists():
        raise FileNotFoundError(
            f"Plot mode requires {surface_csv} and {x_star_csv}."
        )
    plot_from_csv(surface_csv, x_star_csv, output_dir)
    print(f"Replotted banked surface: {surface_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
