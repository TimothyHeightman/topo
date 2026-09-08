"""Shared numerical primitives for the dynamical-obstruction figure.

The panel entrypoints only generate and bank data.  Plotting is deliberately
kept in ``plot_figure2_dynamics.py`` so visual changes never retrain a model or
repeat a dense evaluation.
"""

from __future__ import annotations

import csv
import importlib.metadata
import json
import logging
import math
import platform
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import optax


PINNED_VERSIONS = {
    "jax": "0.11.0",
    "jaxlib": "0.11.0",
    "optax": "0.2.8",
    "numpy": "2.5.2",
    "matplotlib": "3.11.1",
    "tqdm": "4.67.1",
}

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE2_RESULTS = SCRIPT_DIR / "results" / "figure2_dynamics"
PUBLIC_ENV_PYTHON = Path(sys.executable)

Params = tuple[dict[str, jax.Array], ...]


def check_environment() -> dict[str, str]:
    """Require the same pinned numerical environment used for Fig. 1."""

    versions = {
        package: importlib.metadata.version(package) for package in PINNED_VERSIONS
    }
    mismatches = {
        package: (PINNED_VERSIONS[package], actual)
        for package, actual in versions.items()
        if actual != PINNED_VERSIONS[package]
    }
    if sys.version_info < (3, 12):
        mismatches["python"] = (">=3.12", platform.python_version())
    if mismatches:
        rows = "\n".join(
            f"  {name}: expected {expected}, found {actual}"
            for name, (expected, actual) in mismatches.items()
        )
        raise RuntimeError(
            "These experiments are pinned to the environment in "
            "requirements.txt (Python >= 3.12).\n"
            f"Version mismatch:\n{rows}"
        )
    return versions


def configure_logger(output_dir: Path, name: str) -> logging.Logger:
    """Log each panel run to both the terminal and a durable UTF-8 file."""

    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        fmt="%(asctime)sZ %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
    )
    file_handler = logging.FileHandler(output_dir / "run.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def config_dict(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    return dict(config)


def run_manifest(panel: str, config: Any, versions: dict[str, str]) -> dict[str, Any]:
    return {
        "panel": panel,
        "created_utc": utc_now(),
        "config": config_dict(config),
        "python": platform.python_version(),
        "versions": versions,
        "jax_backend": jax.default_backend(),
    }


def initialize_mlp(
    key: jax.Array, input_dim: int, hidden_width: int, hidden_layers: int
) -> Params:
    """Initialize a tanh MLP whose three outputs parameterize S^3."""

    dimensions = [input_dim] + [hidden_width] * hidden_layers + [3]
    keys = jax.random.split(key, len(dimensions) - 1)
    layers: list[dict[str, jax.Array]] = []
    for layer_index, (fan_in, fan_out, layer_key) in enumerate(
        zip(dimensions[:-1], dimensions[1:], keys, strict=True)
    ):
        scale = math.sqrt(2.0 / (fan_in + fan_out))
        weights = scale * jax.random.normal(
            layer_key, (fan_in, fan_out), dtype=jnp.float64
        )
        bias = jnp.zeros((fan_out,), dtype=jnp.float64)
        if layer_index == len(dimensions) - 2:
            bias = jnp.asarray((1.0, 1.0, 1.0), dtype=jnp.float64)
        layers.append({"weight": weights, "bias": bias})
    return tuple(layers)


def mlp_apply(params: Params, inputs: jax.Array) -> jax.Array:
    activations = inputs
    for layer_index, layer in enumerate(params):
        activations = activations @ layer["weight"] + layer["bias"]
        if layer_index < len(params) - 1:
            activations = jnp.tanh(activations)
    return activations


def normalized_qubit(params: Params, inputs: jax.Array) -> jax.Array:
    """Map three network angles to a normalized two-component complex state."""

    angles = mlp_apply(params, inputs)
    a, b, c = (angles[..., index] for index in range(3))
    sin_a = jnp.sin(a)
    sin_b = jnp.sin(b)
    return jnp.stack(
        (
            jnp.cos(a) + 1j * sin_a * jnp.cos(b),
            sin_a * sin_b * (jnp.cos(c) + 1j * jnp.sin(c)),
        ),
        axis=-1,
    )


def qubit_bloch(states: jax.Array) -> jax.Array:
    first, second = states[..., 0], states[..., 1]
    overlap = jnp.conj(first) * second
    return jnp.stack(
        (
            2.0 * jnp.real(overlap),
            2.0 * jnp.imag(overlap),
            jnp.abs(first) ** 2 - jnp.abs(second) ** 2,
        ),
        axis=-1,
    )


def fidelity_from_bloch(predicted: jax.Array, target: jax.Array) -> jax.Array:
    return 0.5 * (1.0 + jnp.sum(predicted * target, axis=-1))


def save_checkpoint(
    params: Params,
    path: Path,
    panel: str,
    config: Any,
    versions: dict[str, str],
    elapsed_seconds: float,
) -> None:
    arrays: dict[str, np.ndarray] = {}
    for index, layer in enumerate(params):
        arrays[f"layer_{index}_weight"] = np.asarray(layer["weight"])
        arrays[f"layer_{index}_bias"] = np.asarray(layer["bias"])
    np.savez_compressed(path, **arrays)
    metadata = run_manifest(panel, config, versions)
    metadata.update(
        {
            "checkpoint": path.name,
            "layer_count": len(params),
            "training_elapsed_seconds": elapsed_seconds,
            "parameterization": "three neural angles mapped exactly to S^3",
        }
    )
    write_json(path.with_suffix(".json"), metadata)


def load_checkpoint(path: Path) -> Params:
    metadata_path = path.with_suffix(".json")
    if not path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Missing checkpoint or metadata for {path}")
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


def make_optimizer(
    steps: int,
    learning_rate: float,
    final_learning_rate: float,
    gradient_clip: float,
) -> optax.GradientTransformation:
    schedule = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=steps,
        alpha=final_learning_rate / learning_rate,
    )
    return optax.chain(
        optax.clip_by_global_norm(gradient_clip), optax.adam(schedule)
    )


def write_history(path: Path, rows: Sequence[dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def equal_area_sphere(n_cos_theta: int, n_phi: int) -> jax.Array:
    u = -1.0 + (jnp.arange(n_cos_theta, dtype=jnp.float64) + 0.5) * (
        2.0 / n_cos_theta
    )
    phi = 2.0 * jnp.pi * jnp.arange(n_phi, dtype=jnp.float64) / n_phi
    uu, pp = jnp.meshgrid(u, phi, indexing="ij")
    radial = jnp.sqrt(jnp.maximum(0.0, 1.0 - uu**2))
    return jnp.stack(
        (radial * jnp.cos(pp), radial * jnp.sin(pp), uu), axis=-1
    ).reshape((-1, 3))


def regular_sphere(
    n_theta: int, n_phi: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, np.pi, n_theta, dtype=np.float64)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, dtype=np.float64)
    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    points = np.stack(
        (
            np.sin(tt) * np.cos(pp),
            np.sin(tt) * np.sin(pp),
            np.cos(tt),
        ),
        axis=-1,
    )
    return theta, phi, points


def generic_rotation() -> jax.Array:
    """The fixed display-to-target SO(3) rotation used in Fig. 1."""

    z_angle, y_angle = 0.8, 1.0
    rz = jnp.asarray(
        (
            (math.cos(z_angle), -math.sin(z_angle), 0.0),
            (math.sin(z_angle), math.cos(z_angle), 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=jnp.float64,
    )
    ry = jnp.asarray(
        (
            (math.cos(y_angle), 0.0, math.sin(y_angle)),
            (0.0, 1.0, 0.0),
            (-math.sin(y_angle), 0.0, math.cos(y_angle)),
        ),
        dtype=jnp.float64,
    )
    return rz @ ry


def finite_target_bloch(
    display_points: jax.Array, tau: jax.Array, total_angle: float
) -> jax.Array:
    """Evolve the Hopf target by U(t)=exp[-i angle(t) sigma_z/2]."""

    initial = display_points @ generic_rotation()
    angle = total_angle * tau
    cosine, sine = jnp.cos(angle), jnp.sin(angle)
    x = cosine * initial[..., 0] - sine * initial[..., 1]
    y = sine * initial[..., 0] + cosine * initial[..., 1]
    return jnp.stack((x, y, initial[..., 2]), axis=-1)


def finite_features(display_points: jax.Array, tau: jax.Array) -> jax.Array:
    return jnp.concatenate((display_points, (2.0 * tau - 1.0)[..., None]), axis=-1)


def qwz_target_bloch(k: jax.Array, t: jax.Array, mass: float) -> jax.Array:
    """Degree-one two-level projector on the momentum-time torus."""

    vector = jnp.stack(
        (jnp.sin(k), jnp.sin(t), mass + jnp.cos(k) + jnp.cos(t)), axis=-1
    )
    return vector / jnp.linalg.norm(vector, axis=-1, keepdims=True)


def periodic_features(k: jax.Array, t: jax.Array) -> jax.Array:
    return jnp.stack((jnp.cos(k), jnp.sin(k), jnp.cos(t), jnp.sin(t)), axis=-1)


def spinor_from_bloch(
    bloch: np.ndarray, prefer_north: bool | None = None
) -> np.ndarray:
    """Choose a stable local spinor gauge for one or many Bloch vectors."""

    vectors = np.asarray(bloch, dtype=np.float64)
    if prefer_north is None:
        prefer_north = float(np.mean(vectors[..., 2])) > 0.0
    x, y, z = (vectors[..., index] for index in range(3))
    if prefer_north:
        denominator = np.sqrt(np.maximum(2.0 * (1.0 + z), 1.0e-15))
        first = np.sqrt(np.maximum((1.0 + z) / 2.0, 0.0))
        second = (x + 1j * y) / denominator
    else:
        denominator = np.sqrt(np.maximum(2.0 * (1.0 - z), 1.0e-15))
        first = (x - 1j * y) / denominator
        second = np.sqrt(np.maximum((1.0 - z) / 2.0, 0.0))
    return np.stack((first, second), axis=-1).astype(np.complex128)


def phase_winding(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.complex128)
    closed = np.concatenate((values, values[:1]))
    increments = np.angle(closed[1:] * np.conj(closed[:-1]))
    return float(np.sum(increments) / (2.0 * np.pi))


def fukui_chern(target_states: np.ndarray) -> tuple[float, np.ndarray]:
    """Gauge-invariant lattice Chern number on a periodic k-t grid."""

    states = np.asarray(target_states, dtype=np.complex128)
    link_k = np.sum(np.conj(states) * np.roll(states, -1, axis=0), axis=-1)
    link_t = np.sum(np.conj(states) * np.roll(states, -1, axis=1), axis=-1)
    link_k /= np.maximum(np.abs(link_k), 1.0e-15)
    link_t /= np.maximum(np.abs(link_t), 1.0e-15)
    plaquette = (
        link_k
        * np.roll(link_t, -1, axis=0)
        * np.conj(np.roll(link_k, -1, axis=1))
        * np.conj(link_t)
    )
    flux = np.angle(plaquette)
    return float(np.sum(flux) / (2.0 * np.pi)), flux


def torus_distance(first: np.ndarray, second: np.ndarray) -> float:
    delta = np.abs(np.asarray(first) - np.asarray(second))
    delta = np.minimum(delta, 2.0 * np.pi - delta)
    return float(np.linalg.norm(delta))


def deduplicate_torus(points: Iterable[np.ndarray], tolerance: float) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for point in points:
        wrapped = np.mod(np.asarray(point, dtype=np.float64), 2.0 * np.pi)
        if all(torus_distance(wrapped, prior) > tolerance for prior in unique):
            unique.append(wrapped)
    return unique
