#!/usr/bin/env python3
"""Train and bank panel (b): periodic vector learning on a Chern-one torus.

The target is a smooth rank-one projector over (k,t) in T^2.  The neural
output is exactly periodic because it receives only sine/cosine features.
This script performs no plotting and saves every artifact needed to redraw the
panel without retraining or reevaluating the network.
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
    deduplicate_torus,
    fidelity_from_bloch,
    fukui_chern,
    initialize_mlp,
    load_checkpoint,
    make_optimizer,
    normalized_qubit,
    periodic_features,
    phase_winding,
    qubit_bloch,
    qwz_target_bloch,
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
    seed: int = 73
    hidden_width: int = 64
    hidden_layers: int = 3
    train_k: int = 112
    train_t: int = 112
    steps: int = 5_500
    learning_rate: float = 3.0e-3
    final_learning_rate: float = 2.0e-5
    gradient_clip: float = 1.0
    log_every: int = 100
    mass: float = -1.0
    eval_k: int = 641
    eval_t: int = 641
    refine_seeds: int = 64
    refine_steps: int = 1_000
    refine_learning_rate: float = 1.5e-2
    dedup_tolerance: float = 0.03
    winding_samples: int = 384
    winding_radius: float = 0.025
    chern_grid: int = 151


def model_values(
    params: Params, k: jax.Array, t: jax.Array, mass: float
) -> tuple[jax.Array, jax.Array, jax.Array]:
    states = normalized_qubit(params, periodic_features(k, t))
    predicted = qubit_bloch(states)
    target = qwz_target_bloch(k, t, mass)
    fidelity = fidelity_from_bloch(predicted, target)
    return states, fidelity, 1.0 - fidelity


def training_grid(config: Config) -> tuple[jax.Array, jax.Array]:
    k = 2.0 * jnp.pi * jnp.arange(config.train_k, dtype=jnp.float64) / config.train_k
    t = 2.0 * jnp.pi * jnp.arange(config.train_t, dtype=jnp.float64) / config.train_t
    kk, tt = jnp.meshgrid(k, t, indexing="ij")
    return kk.reshape(-1), tt.reshape(-1)


def train(config: Config, logger: object) -> tuple[Params, list[dict[str, float]], float]:
    k, t = training_grid(config)
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
        return jnp.mean(model_values(current, k, t, config.mass)[2])

    @jax.jit
    def step(current: Params, current_state: optax.OptState):
        loss, gradients = jax.value_and_grad(loss_function)(current)
        updates, next_state = optimizer.update(gradients, current_state, current)
        return optax.apply_updates(current, updates), next_state, loss

    @jax.jit
    def metrics(current: Params):
        _, fidelity, infidelity = model_values(current, k, t, config.mass)
        return jnp.mean(infidelity), jnp.max(infidelity), jnp.min(fidelity)

    history: list[dict[str, float]] = []
    start = time.perf_counter()
    logger.info("Training panel (b) on %d periodic torus points", k.shape[0])
    with tqdm(total=config.steps, desc="panel (b) training", unit="step") as progress:
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
                history.append(
                    {
                        "step": float(step_index),
                        "mean_infidelity": mean_inf,
                        "max_grid_infidelity": max_inf,
                        "min_grid_fidelity": min_fid,
                        "elapsed_seconds": time.perf_counter() - start,
                    }
                )
                progress.set_postfix(
                    mean=f"{mean_inf:.2e}", max=f"{max_inf:.6f}", refresh=False
                )
            progress.update(1)
    elapsed = time.perf_counter() - start
    logger.info("Training completed in %.2f s", elapsed)
    return params, history, elapsed


def evaluate_dense(
    params: Params, config: Config, output_dir: Path, logger: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = np.linspace(0.0, 2.0 * np.pi, config.eval_k, dtype=np.float64)
    t = np.linspace(0.0, 2.0 * np.pi, config.eval_t, dtype=np.float64)
    kk, tt = np.meshgrid(k, t, indexing="ij")
    logger.info("Evaluating %d x %d periodic torus grid", config.eval_k, config.eval_t)
    _, fidelity, infidelity = jax.device_get(
        jax.jit(model_values)(
            params, jnp.asarray(kk.reshape(-1)), jnp.asarray(tt.reshape(-1)), config.mass
        )
    )
    fidelity = np.asarray(fidelity).reshape(kk.shape)
    infidelity = np.asarray(infidelity).reshape(kk.shape)
    target = np.asarray(
        jax.device_get(qwz_target_bloch(jnp.asarray(kk), jnp.asarray(tt), config.mass))
    )
    surface_path = output_dir / "periodic_surface.npz"
    np.savez_compressed(
        surface_path,
        k=k,
        t=t,
        fidelity=fidelity.astype(np.float32),
        infidelity=infidelity.astype(np.float32),
        target_bloch=target.astype(np.float32),
    )
    logger.info("Saved dense panel (b) surface to %s", surface_path)
    return k, t, infidelity


def refine_nodes(
    params: Params,
    config: Config,
    k: np.ndarray,
    t: np.ndarray,
    infidelity: np.ndarray,
    output_dir: Path,
    logger: object,
) -> list[dict[str, float]]:
    flat = infidelity.reshape(-1)
    seed_indices = np.argpartition(flat, -config.refine_seeds)[
        -config.refine_seeds :
    ]
    k_indices, t_indices = np.unravel_index(seed_indices, infidelity.shape)
    angles = jnp.asarray(
        np.column_stack((k[k_indices], t[t_indices])), dtype=jnp.float64
    )
    optimizer = optax.adam(config.refine_learning_rate)
    state = optimizer.init(angles)

    @jax.jit
    def step(current: jax.Array, current_state: optax.OptState):
        def objective(raw: jax.Array) -> jax.Array:
            _, _, values = model_values(
                params, raw[:, 0], raw[:, 1], config.mass
            )
            return -jnp.sum(values)

        gradients = jax.grad(objective)(current)
        updates, next_state = optimizer.update(gradients, current_state)
        return optax.apply_updates(current, updates), next_state

    logger.info("Refining %d torus-node seeds", config.refine_seeds)
    for _ in range(config.refine_steps):
        angles, state = step(angles, state)
    candidates = np.mod(np.asarray(jax.device_get(angles)), 2.0 * np.pi)
    unique = deduplicate_torus(candidates, config.dedup_tolerance)
    rows: list[dict[str, float]] = []
    for point in unique:
        _, fidelity, candidate_infidelity = jax.device_get(
            model_values(
                params,
                jnp.asarray((point[0],)),
                jnp.asarray((point[1],)),
                config.mass,
            )
        )
        infidelity_value = float(np.asarray(candidate_infidelity)[0])
        if infidelity_value < 0.95:
            continue
        rows.append(
            {
                "k": float(point[0]),
                "t": float(point[1]),
                "fidelity": float(np.asarray(fidelity)[0]),
                "infidelity": infidelity_value,
                "charge": local_winding(params, config, point),
            }
        )
    rows.sort(key=lambda row: row["infidelity"], reverse=True)
    write_rows(output_dir / "periodic_nodes.csv", rows)
    logger.info(
        "Found %d distinct nodes above 0.95 infidelity; total charge %.3f",
        len(rows),
        sum(row["charge"] for row in rows),
    )
    return rows


def local_winding(params: Params, config: Config, center: np.ndarray) -> float:
    angle = np.linspace(
        0.0, 2.0 * np.pi, config.winding_samples, endpoint=False
    )
    k = center[0] + config.winding_radius * np.cos(angle)
    t = center[1] + config.winding_radius * np.sin(angle)
    model_states = np.asarray(
        jax.device_get(
            normalized_qubit(params, periodic_features(jnp.asarray(k), jnp.asarray(t)))
        )
    )
    target_bloch = np.asarray(
        jax.device_get(qwz_target_bloch(jnp.asarray(k), jnp.asarray(t), config.mass))
    )
    center_bloch = np.asarray(
        jax.device_get(
            qwz_target_bloch(
                jnp.asarray((center[0],)), jnp.asarray((center[1],)), config.mass
            )
        )
    )[0]
    target_states = spinor_from_bloch(
        target_bloch, prefer_north=bool(center_bloch[2] > 0.0)
    )
    overlap = np.sum(np.conj(target_states) * model_states, axis=-1)
    return phase_winding(overlap)


def compute_chern(config: Config, output_dir: Path) -> tuple[float, np.ndarray]:
    values = 2.0 * np.pi * np.arange(config.chern_grid) / config.chern_grid
    kk, tt = np.meshgrid(values, values, indexing="ij")
    bloch = np.asarray(
        jax.device_get(qwz_target_bloch(jnp.asarray(kk), jnp.asarray(tt), config.mass))
    )
    states = spinor_from_bloch(bloch)
    chern, flux = fukui_chern(states)
    np.savez_compressed(
        output_dir / "target_chern.npz",
        k=values,
        t=values,
        plaquette_flux=flux.astype(np.float32),
    )
    return chern, flux


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("all", "train", "evaluate"), default="all")
    result.add_argument(
        "--output-dir", type=Path, default=FIGURE2_RESULTS / "panel_b"
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
    logger = configure_logger(output_dir, "figure2.panel_b")
    logger.info("Python executable: %s", PUBLIC_ENV_PYTHON)
    write_json(output_dir / "manifest.json", run_manifest("b", config, versions))
    checkpoint = output_dir / "checkpoint.npz"

    if args.mode in ("all", "train"):
        if checkpoint.exists() and not args.force:
            logger.info("Checkpoint exists; skipping training: %s", checkpoint)
            params = load_checkpoint(checkpoint)
        else:
            params, history, elapsed = train(config, logger)
            save_checkpoint(params, checkpoint, "b", config, versions, elapsed)
            write_history(output_dir / "training_history.csv", history)
    else:
        params = load_checkpoint(checkpoint)

    if args.mode in ("all", "evaluate"):
        surface_path = output_dir / "periodic_surface.npz"
        node_path = output_dir / "periodic_nodes.csv"
        if surface_path.exists() and node_path.exists() and not args.force:
            logger.info("Banked evaluation exists; skipping dense evaluation")
            return
        k, t, infidelity = evaluate_dense(params, config, output_dir, logger)
        rows = refine_nodes(params, config, k, t, infidelity, output_dir, logger)
        chern, flux = compute_chern(config, output_dir)
        write_json(
            output_dir / "summary.json",
            {
                "created_utc": utc_now(),
                "surface": surface_path.name,
                "nodes": node_path.name,
                "grid": {"k": len(k), "t": len(t)},
                "lattice_chern": chern,
                "maximum_plaquette_flux": float(np.max(np.abs(flux))),
                "node_count": len(rows),
                "total_node_charge": float(sum(row["charge"] for row in rows)),
                "maximum_refined_infidelity": max(
                    (row["infidelity"] for row in rows), default=float("nan")
                ),
            },
        )
        logger.info("Panel (b) artifacts complete; lattice Chern %.12f", chern)


if __name__ == "__main__":
    main()
