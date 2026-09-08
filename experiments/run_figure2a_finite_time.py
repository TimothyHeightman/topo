#!/usr/bin/env python3
"""Train and bank panel (a): a fidelity-node worldline on S^2 x I.

This script performs no plotting.  It writes a checkpoint, training history,
dense space-time surface, refined nodal worldline, summary, and run log.  The
separate ``plot_figure2_dynamics.py`` consumes those artifacts.
"""

from __future__ import annotations

import argparse
import csv
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
    normalized_qubit,
    phase_winding,
    qubit_bloch,
    regular_sphere,
    run_manifest,
    save_checkpoint,
    spinor_from_bloch,
    utc_now,
    write_history,
    write_json,
    write_rows,
)


@dataclass(frozen=True)
class Config:
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
    winding_samples: int = 256
    winding_radius: float = 0.035


def model_values(
    params: Params, points: jax.Array, tau: jax.Array, total_rotation: float
) -> tuple[jax.Array, jax.Array, jax.Array]:
    states = normalized_qubit(params, finite_features(points, tau))
    predicted_bloch = qubit_bloch(states)
    target_bloch = finite_target_bloch(points, tau, total_rotation)
    fidelity = fidelity_from_bloch(predicted_bloch, target_bloch)
    return states, fidelity, 1.0 - fidelity


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
        _, fidelity, infidelity = model_values(
            current, points, tau, config.total_rotation
        )
        return jnp.mean(infidelity), jnp.max(infidelity), jnp.min(fidelity)

    history: list[dict[str, float]] = []
    start = time.perf_counter()
    logger.info(
        "Training panel (a) on %d deterministic S2 x I points", points.shape[0]
    )
    with tqdm(total=config.steps, desc="panel (a) training", unit="step") as progress:
        for step_index in range(1, config.steps + 1):
            params, state, _ = step(params, state)
            if (
                step_index == 1
                or step_index % config.log_every == 0
                or step_index == config.steps
            ):
                mean_inf, max_inf, min_fid = map(
                    float, jax.device_get(metrics(params))
                )
                row = {
                    "step": float(step_index),
                    "mean_infidelity": mean_inf,
                    "max_grid_infidelity": max_inf,
                    "min_grid_fidelity": min_fid,
                    "elapsed_seconds": time.perf_counter() - start,
                }
                history.append(row)
                progress.set_postfix(
                    mean=f"{mean_inf:.2e}", max=f"{max_inf:.6f}", refresh=False
                )
            progress.update(1)
    elapsed = time.perf_counter() - start
    logger.info("Training completed in %.2f s", elapsed)
    return params, history, elapsed


def evaluate_dense(
    params: Params, config: Config, output_dir: Path, logger: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta, phi, sphere = regular_sphere(config.eval_theta, config.eval_phi)
    flat_points = jnp.asarray(sphere.reshape((-1, 3)), dtype=jnp.float64)
    tau = np.linspace(0.0, 1.0, config.eval_time, dtype=np.float64)
    infidelity = np.empty(
        (config.eval_time, config.eval_theta, config.eval_phi), dtype=np.float32
    )

    @jax.jit
    def one_slice(time_value: jax.Array) -> jax.Array:
        times = jnp.full((flat_points.shape[0],), time_value, dtype=jnp.float64)
        return model_values(params, flat_points, times, config.total_rotation)[2]

    logger.info(
        "Evaluating dense panel (a) bank: %d x %d x %d",
        config.eval_time,
        config.eval_theta,
        config.eval_phi,
    )
    for index, time_value in enumerate(tau):
        infidelity[index] = np.asarray(
            jax.device_get(one_slice(jnp.asarray(time_value)))
        ).reshape((config.eval_theta, config.eval_phi))

    surface_path = output_dir / "finite_surface.npz"
    np.savez_compressed(
        surface_path,
        theta=theta,
        phi=phi,
        tau=tau,
        n_x=sphere[..., 0].astype(np.float32),
        n_y=sphere[..., 1].astype(np.float32),
        n_z=sphere[..., 2].astype(np.float32),
        infidelity=infidelity,
    )
    logger.info("Saved dense panel (a) surface to %s", surface_path)
    return theta, phi, tau, sphere, infidelity


def evaluate_worldline_projection(
    params: Params,
    config: Config,
    theta: np.ndarray,
    phi: np.ndarray,
    sphere: np.ndarray,
    output_dir: Path,
    logger: object,
) -> Path:
    """Bank a smooth time-theta heatmap profiled over the azimuth phi."""

    flat_points = jnp.asarray(sphere.reshape((-1, 3)), dtype=jnp.float64)
    tau = np.linspace(0.0, 1.0, config.projection_time, dtype=np.float64)
    maximum = np.empty((config.projection_time, config.eval_theta), dtype=np.float32)
    maximizing_phi = np.empty_like(maximum)

    @jax.jit
    def one_slice(time_value: jax.Array) -> jax.Array:
        times = jnp.full((flat_points.shape[0],), time_value, dtype=jnp.float64)
        return model_values(params, flat_points, times, config.total_rotation)[2]

    logger.info(
        "Evaluating profiled worldline heatmap: %d times x %d theta x %d phi",
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
    logger.info("Saved profiled worldline heatmap to %s", path)
    return path


def refine_worldline(
    params: Params,
    config: Config,
    tau: np.ndarray,
    sphere: np.ndarray,
    infidelity: np.ndarray,
    output_dir: Path,
    logger: object,
) -> list[dict[str, float]]:
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

    logger.info("Refining %d simultaneous time-slice nodes", config.eval_time)
    for _ in range(config.refine_steps):
        vectors, state = step(vectors, state)
    vectors_np = np.asarray(jax.device_get(normalize(vectors)))
    _, fidelity, refined_infidelity = jax.device_get(
        model_values(params, normalize(vectors), times, config.total_rotation)
    )
    fidelity_np = np.asarray(fidelity)
    infidelity_np = np.asarray(refined_infidelity)

    rows: list[dict[str, float]] = []
    selected_vectors: list[np.ndarray] = []
    for time_index, time_value in enumerate(tau):
        best = int(np.argmax(infidelity_np[time_index]))
        point = vectors_np[time_index, best]
        selected_vectors.append(point)
        theta_star = math.acos(float(np.clip(point[2], -1.0, 1.0)))
        phi_star = math.atan2(float(point[1]), float(point[0])) % (2.0 * math.pi)
        winding = local_winding(params, config, point, float(time_value))
        rows.append(
            {
                "tau": float(time_value),
                "theta": theta_star,
                "phi": phi_star,
                "n_x": float(point[0]),
                "n_y": float(point[1]),
                "n_z": float(point[2]),
                "fidelity": float(fidelity_np[time_index, best]),
                "infidelity": float(infidelity_np[time_index, best]),
                "charge": winding,
            }
        )
    unwrapped = np.unwrap(np.asarray([row["phi"] for row in rows]))
    for row, value in zip(rows, unwrapped, strict=True):
        row["phi_unwrapped"] = float(value)
    write_rows(output_dir / "nodal_worldline.csv", rows)
    logger.info(
        "Refined worldline: min infidelity %.12f, charge range [%.3f, %.3f]",
        min(row["infidelity"] for row in rows),
        min(row["charge"] for row in rows),
        max(row["charge"] for row in rows),
    )
    return rows


def local_winding(
    params: Params, config: Config, center: np.ndarray, time_value: float
) -> float:
    reference = np.asarray((0.0, 0.0, 1.0))
    if abs(float(center @ reference)) > 0.85:
        reference = np.asarray((0.0, 1.0, 0.0))
    tangent_one = np.cross(reference, center)
    tangent_one /= np.linalg.norm(tangent_one)
    tangent_two = np.cross(center, tangent_one)
    angles = np.linspace(
        0.0, 2.0 * np.pi, config.winding_samples, endpoint=False
    )
    circle = (
        math.cos(config.winding_radius) * center[None, :]
        + math.sin(config.winding_radius)
        * (
            np.cos(angles)[:, None] * tangent_one[None, :]
            + np.sin(angles)[:, None] * tangent_two[None, :]
        )
    )
    times = np.full((circle.shape[0],), time_value, dtype=np.float64)
    model_states = np.asarray(
        jax.device_get(
            normalized_qubit(
                params,
                finite_features(jnp.asarray(circle), jnp.asarray(times)),
            )
        )
    )
    target_bloch = np.asarray(
        jax.device_get(
            finite_target_bloch(
                jnp.asarray(circle), jnp.asarray(times), config.total_rotation
            )
        )
    )
    center_target = np.asarray(
        jax.device_get(
            finite_target_bloch(
                jnp.asarray(center[None, :]),
                jnp.asarray((time_value,)),
                config.total_rotation,
            )
        )
    )[0]
    target_states = spinor_from_bloch(
        target_bloch, prefer_north=bool(center_target[2] > 0.0)
    )
    overlap = np.sum(np.conj(target_states) * model_states, axis=-1)
    return phase_winding(overlap)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("all", "train", "evaluate"), default="all")
    result.add_argument(
        "--output-dir", type=Path, default=FIGURE2_RESULTS / "panel_a"
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
    logger = configure_logger(output_dir, "figure2.panel_a")
    logger.info("Python executable: %s", PUBLIC_ENV_PYTHON)
    write_json(output_dir / "manifest.json", run_manifest("a", config, versions))
    checkpoint = output_dir / "checkpoint.npz"

    if args.mode in ("all", "train"):
        if checkpoint.exists() and not args.force:
            logger.info("Checkpoint exists; skipping training: %s", checkpoint)
            params = load_checkpoint(checkpoint)
        else:
            params, history, elapsed = train(config, logger)
            save_checkpoint(
                params, checkpoint, "a", config, versions, elapsed
            )
            write_history(output_dir / "training_history.csv", history)
    else:
        params = load_checkpoint(checkpoint)

    if args.mode in ("all", "evaluate"):
        surface_path = output_dir / "finite_surface.npz"
        worldline_path = output_dir / "nodal_worldline.csv"
        projection_path = output_dir / "worldline_projection.npz"
        if (
            surface_path.exists()
            and worldline_path.exists()
            and projection_path.exists()
            and not args.force
        ):
            logger.info("Banked evaluation exists; skipping dense evaluation")
            return
        theta, phi, tau, sphere, infidelity = evaluate_dense(
            params, config, output_dir, logger
        )
        rows = refine_worldline(
            params, config, tau, sphere, infidelity, output_dir, logger
        )
        projection_path = evaluate_worldline_projection(
            params, config, theta, phi, sphere, output_dir, logger
        )
        write_json(
            output_dir / "summary.json",
            {
                "created_utc": utc_now(),
                "surface": surface_path.name,
                "worldline": worldline_path.name,
                "projection": projection_path.name,
                "grid": {
                    "theta": len(theta),
                    "phi": len(phi),
                    "time": len(tau),
                    "projection_time": config.projection_time,
                },
                "minimum_refined_infidelity": min(
                    row["infidelity"] for row in rows
                ),
                "maximum_refined_fidelity": max(row["fidelity"] for row in rows),
                "median_charge": float(np.median([row["charge"] for row in rows])),
            },
        )
        logger.info("Panel (a) artifacts complete")


if __name__ == "__main__":
    main()
