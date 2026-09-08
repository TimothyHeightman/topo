#!/usr/bin/env python3
"""Train and bank the low-capacity FNQS neutral-pair example in Fig. 3.

This script performs no plotting.  It trains the same normalized one-qubit
FNQS used for the finite-time example, but with one width-8 hidden layer, then
banks its checkpoint, training history, refined nodal worldlines, local
space-time fidelity profiles, summary, and run log.  The separate
``plot_endmatter_neutral_pair.py`` script consumes only these saved artifacts.
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
    PUBLIC_ENV_PYTHON,
    Params,
    check_environment,
    configure_logger,
    finite_features,
    finite_target_bloch,
    initialize_mlp,
    make_optimizer,
    normalized_qubit,
    phase_winding,
    run_manifest,
    save_checkpoint,
    spinor_from_bloch,
    utc_now,
    write_history,
    write_json,
    write_rows,
)
from run_figure2a_finite_time import Config as Figure2Config
from run_figure2a_finite_time import model_values, training_set


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS = SCRIPT_DIR / "results" / "figure3_fnqs_pair"


@dataclass(frozen=True)
class Config:
    seed: int = 2
    hidden_width: int = 8
    hidden_layers: int = 1
    train_cos_theta: int = 28
    train_phi: int = 56
    train_time: int = 13
    steps: int = 1_500
    learning_rate: float = 3.0e-3
    final_learning_rate: float = 3.0e-5
    gradient_clip: float = 1.0
    log_every: int = 50
    total_rotation: float = 2.35
    root_time: int = 241
    root_seeds: int = 384
    root_steps: int = 650
    root_learning_rate: float = 2.5e-2
    root_fidelity_tolerance: float = 2.0e-8
    root_merge_angle: float = 1.8e-2
    winding_samples: int = 320
    winding_radius: float = 2.5e-2
    profile_time: int = 401
    profile_coordinate: int = 481
    profile_transverse: int = 121
    local_root_axis: int = 15
    local_root_steps: int = 550
    local_root_learning_rate: float = 2.0e-2
    local_root_merge_angle: float = 6.0e-3
    log_floor: float = -3.0


def figure2_config(config: Config) -> Figure2Config:
    """Return the common finite-time target and training-grid configuration."""

    return Figure2Config(
        seed=config.seed,
        hidden_width=config.hidden_width,
        hidden_layers=config.hidden_layers,
        train_cos_theta=config.train_cos_theta,
        train_phi=config.train_phi,
        train_time=config.train_time,
        steps=config.steps,
        learning_rate=config.learning_rate,
        final_learning_rate=config.final_learning_rate,
        gradient_clip=config.gradient_clip,
        log_every=config.log_every,
        total_rotation=config.total_rotation,
    )


def train(
    config: Config, logger: object
) -> tuple[Params, list[dict[str, float]], float, dict[str, float]]:
    common = figure2_config(config)
    points, tau = training_set(common)
    params = initialize_mlp(
        jax.random.PRNGKey(config.seed),
        4,
        config.hidden_width,
        config.hidden_layers,
    )
    optimizer = make_optimizer(
        config.steps,
        config.learning_rate,
        config.final_learning_rate,
        config.gradient_clip,
    )
    state = optimizer.init(params)

    def loss_function(current: Params) -> jax.Array:
        return jnp.mean(
            model_values(current, points, tau, config.total_rotation)[2]
        )

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
        return (
            jnp.mean(infidelity),
            jnp.max(infidelity),
            jnp.min(fidelity),
            jnp.quantile(fidelity, 0.01),
        )

    history: list[dict[str, float]] = []
    start = time.perf_counter()
    logger.info(
        "Training width-%d FNQS on %d deterministic S2 x I points",
        config.hidden_width,
        points.shape[0],
    )
    with tqdm(total=config.steps, desc="Fig. 3 training", unit="step") as progress:
        for step_index in range(1, config.steps + 1):
            params, state, _ = step(params, state)
            if (
                step_index == 1
                or step_index % config.log_every == 0
                or step_index == config.steps
            ):
                mean_inf, max_inf, min_fid, q01_fid = map(
                    float, jax.device_get(metrics(params))
                )
                history.append(
                    {
                        "step": float(step_index),
                        "mean_infidelity": mean_inf,
                        "max_grid_infidelity": max_inf,
                        "min_grid_fidelity": min_fid,
                        "one_percent_fidelity": q01_fid,
                        "elapsed_seconds": time.perf_counter() - start,
                    }
                )
                progress.set_postfix(
                    mean=f"{mean_inf:.3e}", min_f=f"{min_fid:.3e}", refresh=False
                )
            progress.update(1)
    elapsed = time.perf_counter() - start
    final_values = map(float, jax.device_get(metrics(params)))
    final = dict(
        zip(
            (
                "mean_infidelity",
                "max_grid_infidelity",
                "min_grid_fidelity",
                "one_percent_fidelity",
            ),
            final_values,
            strict=True,
        )
    )
    logger.info("Training completed in %.2f s", elapsed)
    return params, history, elapsed, final


def fibonacci_sphere(size: int) -> np.ndarray:
    index = np.arange(size, dtype=np.float64)
    z = 1.0 - 2.0 * (index + 0.5) / size
    phi = np.pi * (3.0 - np.sqrt(5.0)) * index
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack((radius * np.cos(phi), radius * np.sin(phi), z))


def normalize(vectors: jax.Array) -> jax.Array:
    return vectors / jnp.linalg.norm(vectors, axis=-1, keepdims=True)


def node_charge(
    params: Params,
    config: Config,
    center: np.ndarray,
    time_value: float,
    radius: float,
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
        math.cos(radius) * center[None, :]
        + math.sin(radius)
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


def refine_nodes(
    params: Params, config: Config, output_dir: Path, logger: object
) -> tuple[list[dict[str, float]], np.ndarray]:
    tau = np.linspace(0.0, 1.0, config.root_time, dtype=np.float64)
    seeds = fibonacci_sphere(config.root_seeds)
    vectors = jnp.asarray(
        np.broadcast_to(seeds, (config.root_time, config.root_seeds, 3)).copy()
    )
    times = jnp.asarray(
        np.broadcast_to(tau[:, None], (config.root_time, config.root_seeds))
    )
    optimizer = optax.adam(config.root_learning_rate)
    state = optimizer.init(vectors)

    @jax.jit
    def step(current: jax.Array, current_state: optax.OptState):
        def objective(raw: jax.Array) -> jax.Array:
            fidelity = model_values(
                params, normalize(raw), times, config.total_rotation
            )[1]
            return jnp.sum(fidelity)

        gradients = jax.grad(objective)(current)
        updates, next_state = optimizer.update(gradients, current_state)
        return normalize(optax.apply_updates(current, updates)), next_state

    logger.info(
        "Refining nodes from %d sphere seeds at %d times",
        config.root_seeds,
        config.root_time,
    )
    for _ in tqdm(range(config.root_steps), desc="Fig. 3 node refinement"):
        vectors, state = step(vectors, state)
    vectors_np = np.asarray(jax.device_get(normalize(vectors)))
    fidelity_np = np.asarray(
        jax.device_get(
            model_values(params, normalize(vectors), times, config.total_rotation)[1]
        )
    )

    roots_by_time: list[list[tuple[np.ndarray, float]]] = []
    for time_index in range(config.root_time):
        roots: list[tuple[np.ndarray, float]] = []
        for seed_index in np.argsort(fidelity_np[time_index]):
            fidelity = float(fidelity_np[time_index, seed_index])
            if fidelity > config.root_fidelity_tolerance:
                break
            point = vectors_np[time_index, seed_index]
            if all(
                math.acos(float(np.clip(point @ other, -1.0, 1.0)))
                > config.root_merge_angle
                for other, _ in roots
            ):
                roots.append((point, fidelity))
        roots_by_time.append(roots)

    rows: list[dict[str, float]] = []
    for time_index, roots in enumerate(roots_by_time):
        for node_index, (point, fidelity) in enumerate(roots):
            separations = [
                math.acos(float(np.clip(point @ other, -1.0, 1.0)))
                for other, _ in roots
                if not np.allclose(point, other)
            ]
            radius = config.winding_radius
            if separations:
                radius = min(radius, 0.28 * min(separations))
            charge = node_charge(params, config, point, float(tau[time_index]), radius)
            rows.append(
                {
                    "time_index": float(time_index),
                    "node_index": float(node_index),
                    "tau": float(tau[time_index]),
                    "n_x": float(point[0]),
                    "n_y": float(point[1]),
                    "n_z": float(point[2]),
                    "theta": math.acos(float(np.clip(point[2], -1.0, 1.0))),
                    "phi": math.atan2(float(point[1]), float(point[0]))
                    % (2.0 * math.pi),
                    "fidelity": fidelity,
                    "charge": charge,
                    "winding_radius": radius,
                }
            )

    counts = np.asarray([len(roots) for roots in roots_by_time], dtype=int)
    if int(np.max(counts)) < 3:
        raise RuntimeError("The trained checkpoint did not resolve the expected pair")
    multi_node_times = np.flatnonzero(counts > 1)
    for time_index in multi_node_times:
        time_rows = [
            row for row in rows if int(row["time_index"]) == int(time_index)
        ]
        rounded = sorted(int(round(row["charge"])) for row in time_rows)
        if sum(rounded) != -1 or rounded.count(1) != rounded.count(-1) - 1:
            raise RuntimeError(
                f"Unexpected charges at tau={tau[time_index]:.6f}: {rounded}"
            )
    write_rows(output_dir / "nodal_worldlines.csv", rows)
    logger.info(
        "Resolved node counts %s with net charge -1 on every slice",
        sorted(set(int(value) for value in counts)),
    )
    return rows, counts


def first_three_node_run(counts: np.ndarray) -> np.ndarray:
    """Select the first clean 1-to-3 transition before any later pair event."""

    indices = np.flatnonzero(counts == 3)
    if len(indices) == 0:
        raise RuntimeError("No three-node interval was resolved")
    splits = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    for run in splits:
        if run[0] > 0 and counts[run[0] - 1] == 1:
            return run
    return splits[0]


def logarithmic_map(
    center: np.ndarray,
    tangent_one: np.ndarray,
    tangent_two: np.ndarray,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cosine = np.clip(points @ center, -1.0, 1.0)
    angle = np.arccos(cosine)
    sine = np.sin(angle)
    factor = np.ones_like(angle)
    mask = sine > 1.0e-12
    factor[mask] = angle[mask] / sine[mask]
    tangent = factor[:, None] * (points - cosine[:, None] * center[None, :])
    return tangent @ tangent_one, tangent @ tangent_two


def exponential_map(
    center: np.ndarray,
    tangent_one: np.ndarray,
    tangent_two: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    tangent = (
        first[..., None] * tangent_one[None, None, :]
        + second[..., None] * tangent_two[None, None, :]
    )
    radius = np.sqrt(first * first + second * second)
    sinc = np.ones_like(radius)
    mask = radius > 1.0e-12
    sinc[mask] = np.sin(radius[mask]) / radius[mask]
    return (
        np.cos(radius)[..., None] * center[None, None, :]
        + sinc[..., None] * tangent
    )


def refine_local_pair(
    params: Params,
    config: Config,
    center: np.ndarray,
    tangent_one: np.ndarray,
    tangent_two: np.ndarray,
    coordinate_radius: float,
    tau: np.ndarray,
    logger: object,
) -> list[dict[str, float]]:
    """Resolve only the newly born pair inside the selected tangent chart."""

    seed_coordinate = np.linspace(
        -coordinate_radius, coordinate_radius, config.local_root_axis
    )
    seed_u, seed_v = np.meshgrid(seed_coordinate, seed_coordinate, indexing="ij")
    seeds = exponential_map(
        center, tangent_one, tangent_two, seed_u, seed_v
    ).reshape((-1, 3))
    vectors = jnp.asarray(
        np.broadcast_to(seeds, (len(tau), len(seeds), 3)).copy()
    )
    times = jnp.asarray(np.broadcast_to(tau[:, None], (len(tau), len(seeds))))
    optimizer = optax.adam(config.local_root_learning_rate)
    state = optimizer.init(vectors)

    @jax.jit
    def step(current: jax.Array, current_state: optax.OptState):
        def objective(raw: jax.Array) -> jax.Array:
            return jnp.sum(
                model_values(
                    params, normalize(raw), times, config.total_rotation
                )[1]
            )

        gradients = jax.grad(objective)(current)
        updates, next_state = optimizer.update(gradients, current_state)
        return normalize(optax.apply_updates(current, updates)), next_state

    logger.info(
        "Refining the local pair from %d chart seeds at %d times",
        len(seeds),
        len(tau),
    )
    for _ in tqdm(range(config.local_root_steps), desc="Fig. 3 local nodes"):
        vectors, state = step(vectors, state)
    vectors_np = np.asarray(jax.device_get(normalize(vectors)))
    fidelity_np = np.asarray(
        jax.device_get(
            model_values(params, normalize(vectors), times, config.total_rotation)[1]
        )
    )

    rows: list[dict[str, float]] = []
    for time_index, time_value in enumerate(tau):
        roots: list[tuple[np.ndarray, float, float, float]] = []
        for seed_index in np.argsort(fidelity_np[time_index]):
            fidelity = float(fidelity_np[time_index, seed_index])
            if fidelity > config.root_fidelity_tolerance:
                break
            point = vectors_np[time_index, seed_index]
            u_value, v_value = logarithmic_map(
                center, tangent_one, tangent_two, point[None, :]
            )
            u_scalar = float(u_value[0])
            v_scalar = float(v_value[0])
            if max(abs(u_scalar), abs(v_scalar)) > 0.96 * coordinate_radius:
                continue
            if all(
                math.acos(float(np.clip(point @ other[0], -1.0, 1.0)))
                > config.local_root_merge_angle
                for other in roots
            ):
                roots.append((point, fidelity, u_scalar, v_scalar))

        for node_index, (point, fidelity, u_value, v_value) in enumerate(roots):
            separations = [
                math.acos(float(np.clip(point @ other[0], -1.0, 1.0)))
                for other in roots
                if not np.allclose(point, other[0])
            ]
            radius = config.winding_radius
            if separations:
                radius = min(radius, 0.28 * min(separations))
            charge = node_charge(
                params, config, point, float(time_value), radius
            )
            rows.append(
                {
                    "time_index": float(time_index),
                    "node_index": float(node_index),
                    "tau": float(time_value),
                    "u": u_value,
                    "v": v_value,
                    "n_x": float(point[0]),
                    "n_y": float(point[1]),
                    "n_z": float(point[2]),
                    "fidelity": fidelity,
                    "charge": charge,
                    "winding_radius": radius,
                }
            )

    counts = np.zeros(len(tau), dtype=int)
    for row in rows:
        counts[int(row["time_index"])] += 1
    if int(np.max(counts)) != 2:
        raise RuntimeError(
            f"Expected one local pair, resolved local counts {sorted(set(counts))}"
        )
    for time_index in np.flatnonzero(counts == 2):
        charges = sorted(
            int(round(row["charge"]))
            for row in rows
            if int(row["time_index"]) == int(time_index)
        )
        if charges != [-1, 1]:
            raise RuntimeError(
                f"Local pair at tau={tau[time_index]:.8f} has charges {charges}"
            )
    logger.info(
        "Local pair first resolved at tau=%.8f with charges (-1,+1)",
        tau[np.flatnonzero(counts == 2)[0]],
    )
    return rows


def evaluate_profiles(
    params: Params,
    config: Config,
    rows: list[dict[str, float]],
    counts: np.ndarray,
    output_dir: Path,
    logger: object,
) -> dict[str, float]:
    root_tau = np.linspace(0.0, 1.0, config.root_time, dtype=np.float64)
    event_indices = first_three_node_run(counts)
    orientation_index = int(event_indices[min(5, len(event_indices) - 1)])
    orientation_rows = [
        row for row in rows if int(row["time_index"]) == orientation_index
    ]
    positive = next(row for row in orientation_rows if row["charge"] > 0.5)
    negatives = [row for row in orientation_rows if row["charge"] < -0.5]
    positive_point = np.asarray(
        (positive["n_x"], positive["n_y"], positive["n_z"])
    )
    negative = min(
        negatives,
        key=lambda row: math.acos(
            float(
                np.clip(
                    positive_point
                    @ np.asarray((row["n_x"], row["n_y"], row["n_z"])),
                    -1.0,
                    1.0,
                )
            )
        ),
    )
    negative_point = np.asarray(
        (negative["n_x"], negative["n_y"], negative["n_z"])
    )
    center = positive_point + negative_point
    center /= np.linalg.norm(center)
    direction = positive_point - (positive_point @ center) * center
    tangent_one = direction / np.linalg.norm(direction)
    tangent_two = np.cross(center, tangent_one)
    tangent_two /= np.linalg.norm(tangent_two)

    next_non_three = next(
        (
            index
            for index in range(int(event_indices[-1]) + 1, len(counts))
            if counts[index] != 3
        ),
        int(event_indices[-1]) + 1,
    )
    time_step = float(root_tau[1] - root_tau[0])
    tau_min = max(0.0, float(root_tau[event_indices[0]]) - 3.0 * time_step)
    tau_max = min(
        1.0,
        float(root_tau[next_non_three]) - 0.20 * time_step,
    )
    pair_points = np.asarray(
        [
            (row["n_x"], row["n_y"], row["n_z"])
            for row in orientation_rows
            if abs(row["charge"]) > 0.5
            and math.acos(
                float(
                    np.clip(
                        np.asarray((row["n_x"], row["n_y"], row["n_z"]))
                        @ center,
                        -1.0,
                        1.0,
                    )
                )
            )
            < 0.8
        ]
    )
    pair_u, pair_v = logarithmic_map(
        center, tangent_one, tangent_two, pair_points
    )
    coordinate_radius = max(
        0.42,
        1.55
        * max(
            float(np.max(np.abs(pair_u))),
            float(np.max(np.abs(pair_v))),
        ),
    )
    tau = np.linspace(tau_min, tau_max, config.profile_time, dtype=np.float64)
    event_rows = refine_local_pair(
        params,
        config,
        center,
        tangent_one,
        tangent_two,
        coordinate_radius,
        tau,
        logger,
    )
    write_rows(output_dir / "event_worldlines.csv", event_rows)
    coordinate = np.linspace(
        -coordinate_radius, coordinate_radius, config.profile_coordinate
    )
    transverse = np.linspace(
        -coordinate_radius, coordinate_radius, config.profile_transverse
    )
    first_grid, second_grid = np.meshgrid(coordinate, transverse, indexing="ij")

    @jax.jit
    def evaluate(points_flat: jax.Array, time_value: jax.Array) -> jax.Array:
        times = jnp.full((points_flat.shape[0],), time_value, dtype=jnp.float64)
        return model_values(
            params, points_flat, times, config.total_rotation
        )[2]

    profile_u = np.empty((config.profile_time, config.profile_coordinate))
    profile_v = np.empty_like(profile_u)
    points_u = exponential_map(
        center, tangent_one, tangent_two, first_grid, second_grid
    ).reshape((-1, 3))
    points_v = exponential_map(
        center, tangent_one, tangent_two, second_grid, first_grid
    ).reshape((-1, 3))
    points_u_jax = jnp.asarray(points_u)
    points_v_jax = jnp.asarray(points_v)
    logger.info(
        "Evaluating local profiles: %d times x %d coordinates x %d transverse",
        config.profile_time,
        config.profile_coordinate,
        config.profile_transverse,
    )
    for time_index, time_value in enumerate(
        tqdm(tau, desc="Fig. 3 local profiles")
    ):
        infidelity_u = np.asarray(
            jax.device_get(evaluate(points_u_jax, jnp.asarray(time_value)))
        ).reshape((config.profile_coordinate, config.profile_transverse))
        infidelity_v = np.asarray(
            jax.device_get(evaluate(points_v_jax, jnp.asarray(time_value)))
        ).reshape((config.profile_coordinate, config.profile_transverse))
        profile_u[time_index] = np.max(infidelity_u, axis=1)
        profile_v[time_index] = np.max(infidelity_v, axis=1)

    path = output_dir / "local_profiles.npz"
    np.savez_compressed(
        path,
        tau=tau,
        u=coordinate,
        v=coordinate,
        log_infidelity_u=np.clip(
            np.log10(np.maximum(profile_u, 10.0**config.log_floor)),
            config.log_floor,
            0.0,
        ).astype(np.float32),
        log_infidelity_v=np.clip(
            np.log10(np.maximum(profile_v, 10.0**config.log_floor)),
            config.log_floor,
            0.0,
        ).astype(np.float32),
        center=center,
        tangent_u=tangent_one,
        tangent_v=tangent_two,
    )
    logger.info("Saved trained-FNQS profiles to %s", path)
    return {
        "event_tau_first_resolved": float(root_tau[event_indices[0]]),
        "event_tau_last_three_node_slice": float(root_tau[event_indices[-1]]),
        "profile_tau_min": float(tau_min),
        "profile_tau_max": float(tau_max),
        "coordinate_radius": float(coordinate_radius),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-dir", type=Path, default=RESULTS)
    result.add_argument("--force", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    config = Config()
    versions = check_environment()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(output_dir, "figure3.fnqs_pair")
    logger.info("Python executable: %s", PUBLIC_ENV_PYTHON)
    write_json(output_dir / "manifest.json", run_manifest("figure3", config, versions))
    checkpoint = output_dir / "checkpoint.npz"
    required = (
        checkpoint,
        output_dir / "training_history.csv",
        output_dir / "nodal_worldlines.csv",
        output_dir / "event_worldlines.csv",
        output_dir / "local_profiles.npz",
        output_dir / "summary.json",
    )
    if all(path.exists() for path in required) and not args.force:
        logger.info("All banked artifacts exist; skipping numerical run")
        return

    params, history, elapsed, final = train(config, logger)
    save_checkpoint(params, checkpoint, "figure3", config, versions, elapsed)
    write_history(output_dir / "training_history.csv", history)
    rows, counts = refine_nodes(params, config, output_dir, logger)
    profile_summary = evaluate_profiles(
        params, config, rows, counts, output_dir, logger
    )
    three_node_indices = np.flatnonzero(counts == 3)
    write_json(
        output_dir / "summary.json",
        {
            "created_utc": utc_now(),
            "architecture": [4, config.hidden_width, 3],
            "training": final,
            "node_count_min": int(np.min(counts)),
            "node_count_max": int(np.max(counts)),
            "node_counts_observed": sorted(set(int(value) for value in counts)),
            "three_node_time_slices": int(len(three_node_indices)),
            "charges_on_three_node_slices": [-1, -1, 1],
            **profile_summary,
        },
    )
    logger.info("Figure 3 trained-FNQS artifacts complete")


if __name__ == "__main__":
    main()
