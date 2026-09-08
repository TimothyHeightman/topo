#!/usr/bin/env python3
"""Train and bank the density-matrix control for the finite-time worldline.

This is the dynamical companion to ``one_qubit_density_fnqs_lift.py`` and the
matched control for ``run_figure2a_finite_time.py``.  It uses the identical
4-64-64-64-3 MLP, seed, deterministic S^2 x I training grid, optimizer, and
schedule as the wavefunction-valued finite-time model; only the output map
changes.  The three network outputs are read as a raw Bloch vector ``v`` and

    r = v / sqrt(|v|^2 + 1e-12),        rho = (I + r . sigma) / 2,

which is Hermitian, trace one, and positive by construction.  Against the pure
target projector the fidelity is exactly Tr(rho P) = (1 + r . b_target)/2.

The script performs no plotting.  It banks a checkpoint, training history, the
dense space-time infidelity surface, the profiled (theta, t) heatmap, a
continuously refined per-slice worst-case locus, and a summary.  The separate
``plot_density_worldline.py`` consumes those artifacts.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm.auto import tqdm

from figure2_common import (
    FIGURE2_RESULTS,
    PUBLIC_ENV_PYTHON,
    Params,
    check_environment,
    configure_logger,
    equal_area_sphere,
    fidelity_from_bloch,
    finite_features,
    finite_target_bloch,
    initialize_mlp,
    load_checkpoint,
    make_optimizer,
    mlp_apply,
    regular_sphere,
    run_manifest,
    save_checkpoint,
    utc_now,
    write_history,
    write_json,
    write_rows,
)


BLOCH_REGULARIZER = 1.0e-6


@dataclass(frozen=True)
class Config:
    """Identical numerical protocol to the vector model's panel (a)."""

    seed: int = 37
    hidden_width: int = 64
    hidden_layers: int = 3
    train_cos_theta: int = 28
    train_phi: int = 56
    train_time: int = 13
    steps: int = 4_500
    learning_rate: float = 3.0e-3
    final_learning_rate: float = 3.0e-5
    gradient_clip: float = 1.0
    log_every: int = 100
    total_rotation: float = 2.35
    eval_theta: int = 361
    eval_phi: int = 721
    eval_time: int = 41
    projection_time: int = 161
    refine_seeds: int = 8
    refine_steps: int = 700
    refine_learning_rate: float = 2.0e-2


def parameter_count(params: Params) -> int:
    return int(sum(layer["weight"].size + layer["bias"].size for layer in params))


def density_bloch(params: Params, points: jax.Array, tau: jax.Array) -> jax.Array:
    """Smoothly normalized neural Bloch vector in the closed unit ball."""

    raw = mlp_apply(params, finite_features(points, tau))
    denominator = jnp.sqrt(
        jnp.sum(raw**2, axis=-1, keepdims=True) + BLOCH_REGULARIZER**2
    )
    return raw / denominator


def model_values(
    params: Params, points: jax.Array, tau: jax.Array, total_rotation: float
) -> tuple[jax.Array, jax.Array, jax.Array]:
    bloch = density_bloch(params, points, tau)
    target_bloch = finite_target_bloch(points, tau, total_rotation)
    fidelity = fidelity_from_bloch(bloch, target_bloch)
    return bloch, fidelity, 1.0 - fidelity


def training_set(config: Config) -> tuple[jax.Array, jax.Array]:
    points = equal_area_sphere(config.train_cos_theta, config.train_phi)
    tau = jnp.linspace(0.0, 1.0, config.train_time, dtype=jnp.float64)
    tiled_points = jnp.broadcast_to(
        points[None, :, :], (config.train_time, points.shape[0], 3)
    ).reshape((-1, 3))
    tiled_tau = jnp.broadcast_to(
        tau[:, None], (config.train_time, points.shape[0])
    ).reshape((-1,))
    return tiled_points, tiled_tau


def train(config: Config, logger: object) -> tuple[Params, list[dict[str, float]], float]:
    points, tau = training_set(config)
    params = initialize_mlp(
        jax.random.PRNGKey(config.seed), 4, config.hidden_width, config.hidden_layers
    )
    optimizer = make_optimizer(
        config.steps,
        config.learning_rate,
        config.final_learning_rate,
        config.gradient_clip,
    )
    state = optimizer.init(params)

    def loss_function(current: Params) -> jax.Array:
        return jnp.mean(model_values(current, points, tau, config.total_rotation)[2])

    @jax.jit
    def step(current: Params, current_state: optax.OptState):
        loss, gradients = jax.value_and_grad(loss_function)(current)
        updates, next_state = optimizer.update(gradients, current_state, current)
        return optax.apply_updates(current, updates), next_state, loss

    @jax.jit
    def metrics(current: Params):
        bloch, fidelity, infidelity = model_values(
            current, points, tau, config.total_rotation
        )
        return (
            jnp.mean(infidelity),
            jnp.max(infidelity),
            jnp.min(fidelity),
            jnp.min(jnp.linalg.norm(bloch, axis=-1)),
        )

    history: list[dict[str, float]] = []
    start = time.perf_counter()
    logger.info(
        "Training density control on %d deterministic S2 x I points", points.shape[0]
    )
    with tqdm(total=config.steps, desc="density control training", unit="step") as progress:
        for step_index in range(1, config.steps + 1):
            params, state, _ = step(params, state)
            if (
                step_index == 1
                or step_index % config.log_every == 0
                or step_index == config.steps
            ):
                mean_inf, max_inf, min_fid, min_norm = map(
                    float, jax.device_get(metrics(params))
                )
                row = {
                    "step": float(step_index),
                    "mean_infidelity": mean_inf,
                    "max_grid_infidelity": max_inf,
                    "min_grid_fidelity": min_fid,
                    "min_grid_bloch_norm": min_norm,
                    "elapsed_seconds": time.perf_counter() - start,
                }
                history.append(row)
                progress.set_postfix(
                    mean=f"{mean_inf:.2e}", max=f"{max_inf:.2e}", refresh=False
                )
            progress.update(1)
    elapsed = time.perf_counter() - start
    logger.info("Training completed in %.2f s", elapsed)
    return params, history, elapsed


def evaluate_dense(
    params: Params, config: Config, output_dir: Path, logger: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    theta, phi, sphere = regular_sphere(config.eval_theta, config.eval_phi)
    flat_points = jnp.asarray(sphere.reshape((-1, 3)), dtype=jnp.float64)
    tau = np.linspace(0.0, 1.0, config.eval_time, dtype=np.float64)
    infidelity = np.empty(
        (config.eval_time, config.eval_theta, config.eval_phi), dtype=np.float32
    )
    bloch_norm_min = math.inf
    bloch_norm_max = -math.inf

    @jax.jit
    def one_slice(time_value: jax.Array) -> tuple[jax.Array, jax.Array]:
        times = jnp.full((flat_points.shape[0],), time_value, dtype=jnp.float64)
        bloch, _, slice_infidelity = model_values(
            params, flat_points, times, config.total_rotation
        )
        return slice_infidelity, jnp.linalg.norm(bloch, axis=-1)

    logger.info(
        "Evaluating dense density bank: %d x %d x %d",
        config.eval_time,
        config.eval_theta,
        config.eval_phi,
    )
    for index, time_value in enumerate(tau):
        slice_infidelity, norms = jax.device_get(one_slice(jnp.asarray(time_value)))
        infidelity[index] = np.asarray(slice_infidelity).reshape(
            (config.eval_theta, config.eval_phi)
        )
        bloch_norm_min = min(bloch_norm_min, float(np.min(norms)))
        bloch_norm_max = max(bloch_norm_max, float(np.max(norms)))

    surface_path = output_dir / "finite_surface.npz"
    np.savez_compressed(
        surface_path,
        theta=theta,
        phi=phi,
        tau=tau,
        infidelity=infidelity,
    )
    positivity = {
        "grid_min_bloch_norm": bloch_norm_min,
        "grid_max_bloch_norm": bloch_norm_max,
        "grid_min_purity": 0.5 * (1.0 + bloch_norm_min**2),
        "grid_min_density_eigenvalue": 0.5 * (1.0 - bloch_norm_max),
    }
    logger.info("Saved dense density surface to %s", surface_path)
    logger.info(
        "Bloch norm range [%.12f, %.12f]; min density eigenvalue %.3e",
        bloch_norm_min,
        bloch_norm_max,
        positivity["grid_min_density_eigenvalue"],
    )
    return theta, phi, tau, sphere, infidelity, positivity


def evaluate_worldline_projection(
    params: Params,
    config: Config,
    theta: np.ndarray,
    phi: np.ndarray,
    sphere: np.ndarray,
    output_dir: Path,
    logger: object,
) -> Path:
    """Bank the same smooth (theta, t) azimuth-profiled heatmap as the vector run."""

    flat_points = jnp.asarray(sphere.reshape((-1, 3)), dtype=jnp.float64)
    tau = np.linspace(0.0, 1.0, config.projection_time, dtype=np.float64)
    maximum = np.empty((config.projection_time, config.eval_theta), dtype=np.float32)
    maximizing_phi = np.empty_like(maximum)

    @jax.jit
    def one_slice(time_value: jax.Array) -> jax.Array:
        times = jnp.full((flat_points.shape[0],), time_value, dtype=jnp.float64)
        return model_values(params, flat_points, times, config.total_rotation)[2]

    logger.info(
        "Evaluating profiled density heatmap: %d times x %d theta x %d phi",
        config.projection_time,
        config.eval_theta,
        config.eval_phi,
    )
    for index, time_value in enumerate(tau):
        values = np.asarray(
            jax.device_get(one_slice(jnp.asarray(time_value)))
        ).reshape((config.eval_theta, config.eval_phi))
        maximizing_index = np.argmax(values, axis=1)
        maximum[index] = values[np.arange(config.eval_theta), maximizing_index]
        maximizing_phi[index] = phi[maximizing_index]

    path = output_dir / "worldline_projection.npz"
    np.savez_compressed(
        path,
        theta=theta,
        tau=tau,
        max_phi_infidelity=maximum,
        maximizing_phi=maximizing_phi,
    )
    logger.info("Saved profiled density heatmap to %s", path)
    return path


def refine_worst_locus(
    params: Params,
    config: Config,
    tau: np.ndarray,
    sphere: np.ndarray,
    infidelity: np.ndarray,
    output_dir: Path,
    logger: object,
) -> list[dict[str, float]]:
    """Continuously maximize infidelity per time slice from the worst grid seeds.

    This mirrors the vector run's worldline refinement.  For the density model
    there is no forced node, so the refined curve is a worst-case locus rather
    than a nodal worldline; no topological charge is defined on it.
    """

    flat_points = sphere.reshape((-1, 3))
    seeds: list[np.ndarray] = []
    for time_index in range(config.eval_time):
        values = infidelity[time_index].reshape(-1)
        seed_indices = np.argpartition(values, -config.refine_seeds)[
            -config.refine_seeds :
        ]
        seeds.append(flat_points[seed_indices])
    vectors = jnp.asarray(np.stack(seeds, axis=0), dtype=jnp.float64)
    times = jnp.asarray(tau[:, None], dtype=jnp.float64)
    times = jnp.broadcast_to(times, vectors.shape[:-1])
    optimizer = optax.adam(config.refine_learning_rate)
    state = optimizer.init(vectors)

    def normalize(raw: jax.Array) -> jax.Array:
        return raw / jnp.linalg.norm(raw, axis=-1, keepdims=True)

    @jax.jit
    def step(current: jax.Array, current_state: optax.OptState):
        def objective(raw: jax.Array) -> jax.Array:
            _, _, candidate_infidelity = model_values(
                params, normalize(raw), times, config.total_rotation
            )
            return -jnp.sum(candidate_infidelity)

        gradients = jax.grad(objective)(current)
        updates, next_state = optimizer.update(gradients, current_state)
        return normalize(optax.apply_updates(current, updates)), next_state

    logger.info("Refining %d simultaneous per-slice worst cases", config.eval_time)
    for _ in range(config.refine_steps):
        vectors, state = step(vectors, state)
    vectors_np = np.asarray(jax.device_get(normalize(vectors)))
    bloch, fidelity, refined_infidelity = jax.device_get(
        model_values(params, normalize(vectors), times, config.total_rotation)
    )
    bloch_np = np.asarray(bloch)
    fidelity_np = np.asarray(fidelity)
    infidelity_np = np.asarray(refined_infidelity)

    rows: list[dict[str, float]] = []
    for time_index, time_value in enumerate(tau):
        best = int(np.argmax(infidelity_np[time_index]))
        point = vectors_np[time_index, best]
        norm = float(np.linalg.norm(bloch_np[time_index, best]))
        rows.append(
            {
                "tau": float(time_value),
                "theta": math.acos(float(np.clip(point[2], -1.0, 1.0))),
                "phi": math.atan2(float(point[1]), float(point[0]))
                % (2.0 * math.pi),
                "n_x": float(point[0]),
                "n_y": float(point[1]),
                "n_z": float(point[2]),
                "fidelity": float(fidelity_np[time_index, best]),
                "infidelity": float(infidelity_np[time_index, best]),
                "bloch_norm": norm,
                "purity": 0.5 * (1.0 + norm**2),
                "minimum_eigenvalue": 0.5 * (1.0 - norm),
            }
        )
    write_rows(output_dir / "worst_locus.csv", rows)
    logger.info(
        "Refined worst-case locus: max infidelity %.6e, min fidelity %.12f",
        max(row["infidelity"] for row in rows),
        min(row["fidelity"] for row in rows),
    )
    return rows


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("all", "train", "evaluate"), default="all")
    result.add_argument(
        "--output-dir", type=Path, default=FIGURE2_RESULTS / "panel_a_density"
    )
    result.add_argument("--force", action="store_true")
    result.add_argument("--steps", type=int, default=Config.steps)
    return result


def main() -> None:
    args = parser().parse_args()
    config = Config(steps=args.steps)
    versions = check_environment()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(output_dir, "figure2.panel_a_density")
    logger.info("Python executable: %s", PUBLIC_ENV_PYTHON)
    manifest = run_manifest("a_density", config, versions)
    manifest["representation"] = (
        "rho=(I+r.sigma)/2 with r=v/sqrt(|v|^2+1e-12) from the matched MLP"
    )
    write_json(output_dir / "manifest.json", manifest)
    checkpoint = output_dir / "checkpoint.npz"

    if args.mode in ("all", "train"):
        if checkpoint.exists() and not args.force:
            logger.info("Checkpoint exists; skipping training: %s", checkpoint)
            params = load_checkpoint(checkpoint)
        else:
            params, history, elapsed = train(config, logger)
            save_checkpoint(params, checkpoint, "a_density", config, versions, elapsed)
            write_history(output_dir / "training_history.csv", history)
    else:
        params = load_checkpoint(checkpoint)

    if args.mode in ("all", "evaluate"):
        surface_path = output_dir / "finite_surface.npz"
        locus_path = output_dir / "worst_locus.csv"
        projection_path = output_dir / "worldline_projection.npz"
        if (
            surface_path.exists()
            and locus_path.exists()
            and projection_path.exists()
            and not args.force
        ):
            logger.info("Banked evaluation exists; skipping dense evaluation")
            return
        theta, phi, tau, sphere, infidelity, positivity = evaluate_dense(
            params, config, output_dir, logger
        )
        rows = refine_worst_locus(
            params, config, tau, sphere, infidelity, output_dir, logger
        )
        projection_path = evaluate_worldline_projection(
            params, config, theta, phi, sphere, output_dir, logger
        )
        write_json(
            output_dir / "summary.json",
            {
                "created_utc": utc_now(),
                "representation": "density matrix",
                "parameter_count": parameter_count(params),
                "surface": surface_path.name,
                "worst_locus": locus_path.name,
                "projection": projection_path.name,
                "grid": {
                    "theta": len(theta),
                    "phi": len(phi),
                    "time": len(tau),
                    "projection_time": config.projection_time,
                },
                "grid_max_infidelity": float(np.max(infidelity)),
                "refined_max_infidelity": max(row["infidelity"] for row in rows),
                "refined_min_fidelity": min(row["fidelity"] for row in rows),
                "refined_min_bloch_norm": min(row["bloch_norm"] for row in rows),
                **positivity,
            },
        )
        logger.info("Density control artifacts complete")


if __name__ == "__main__":
    main()
