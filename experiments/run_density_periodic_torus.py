#!/usr/bin/env python3
"""Train and bank the density-matrix control for the periodic Chern-one torus.

This is the periodic companion to ``run_density_finite_time.py`` and the
matched control for ``run_figure2b_periodic_torus.py``.  It uses the identical
4-64-64-64-3 MLP, seed, deterministic 112 x 112 torus training grid, optimizer,
and schedule as the wavefunction-valued periodic model; only the output map
changes.  The three network outputs are read as a raw Bloch vector ``v`` and

    r = v / sqrt(|v|^2 + 1e-12),        rho = (I + r . sigma) / 2,

which is Hermitian, trace one, and positive by construction.  Against the pure
target projector of the C = +1 family the fidelity is exactly
Tr(rho P) = (1 + r . b_target)/2.

The script performs no plotting.  It banks a checkpoint, training history, the
dense periodic infidelity surface, a continuously refined worst case over the
torus, and a summary.
"""

from __future__ import annotations

import math
import time
import argparse
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
    fidelity_from_bloch,
    initialize_mlp,
    load_checkpoint,
    make_optimizer,
    mlp_apply,
    periodic_features,
    qwz_target_bloch,
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
    """Identical numerical protocol to the vector model's panel (b)."""

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


def parameter_count(params: Params) -> int:
    return int(sum(layer["weight"].size + layer["bias"].size for layer in params))


def density_bloch(params: Params, k: jax.Array, t: jax.Array) -> jax.Array:
    """Smoothly normalized neural Bloch vector in the closed unit ball."""

    raw = mlp_apply(params, periodic_features(k, t))
    denominator = jnp.sqrt(
        jnp.sum(raw**2, axis=-1, keepdims=True) + BLOCH_REGULARIZER**2
    )
    return raw / denominator


def model_values(
    params: Params, k: jax.Array, t: jax.Array, mass: float
) -> tuple[jax.Array, jax.Array, jax.Array]:
    bloch = density_bloch(params, k, t)
    target = qwz_target_bloch(k, t, mass)
    fidelity = fidelity_from_bloch(bloch, target)
    return bloch, fidelity, 1.0 - fidelity


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
        bloch, fidelity, infidelity = model_values(current, k, t, config.mass)
        return (
            jnp.mean(infidelity),
            jnp.max(infidelity),
            jnp.min(fidelity),
            jnp.min(jnp.linalg.norm(bloch, axis=-1)),
        )

    history: list[dict[str, float]] = []
    start = time.perf_counter()
    logger.info("Training density control on %d periodic torus points", k.shape[0])
    with tqdm(
        total=config.steps, desc="periodic density training", unit="step"
    ) as progress:
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
                history.append(
                    {
                        "step": float(step_index),
                        "mean_infidelity": mean_inf,
                        "max_grid_infidelity": max_inf,
                        "min_grid_fidelity": min_fid,
                        "min_grid_bloch_norm": min_norm,
                        "elapsed_seconds": time.perf_counter() - start,
                    }
                )
                progress.set_postfix(
                    mean=f"{mean_inf:.2e}", max=f"{max_inf:.2e}", refresh=False
                )
            progress.update(1)
    elapsed = time.perf_counter() - start
    logger.info("Training completed in %.2f s", elapsed)
    return params, history, elapsed


def evaluate_dense(
    params: Params, config: Config, output_dir: Path, logger: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    k = np.linspace(0.0, 2.0 * np.pi, config.eval_k, dtype=np.float64)
    t = np.linspace(0.0, 2.0 * np.pi, config.eval_t, dtype=np.float64)
    kk, tt = np.meshgrid(k, t, indexing="ij")
    logger.info(
        "Evaluating %d x %d periodic density grid", config.eval_k, config.eval_t
    )
    bloch, fidelity, infidelity = jax.device_get(
        jax.jit(model_values)(
            params, jnp.asarray(kk.reshape(-1)), jnp.asarray(tt.reshape(-1)), config.mass
        )
    )
    fidelity = np.asarray(fidelity).reshape(kk.shape)
    infidelity = np.asarray(infidelity).reshape(kk.shape)
    norms = np.linalg.norm(np.asarray(bloch), axis=-1)
    surface_path = output_dir / "periodic_surface.npz"
    np.savez_compressed(
        surface_path,
        k=k,
        t=t,
        fidelity=fidelity.astype(np.float32),
        infidelity=infidelity.astype(np.float32),
    )
    positivity = {
        "grid_min_bloch_norm": float(np.min(norms)),
        "grid_max_bloch_norm": float(np.max(norms)),
        "grid_min_purity": 0.5 * (1.0 + float(np.min(norms)) ** 2),
        "grid_min_density_eigenvalue": 0.5 * (1.0 - float(np.max(norms))),
    }
    logger.info("Saved dense periodic density surface to %s", surface_path)
    logger.info(
        "Bloch norm range [%.12f, %.12f]; min density eigenvalue %.3e",
        positivity["grid_min_bloch_norm"],
        positivity["grid_max_bloch_norm"],
        positivity["grid_min_density_eigenvalue"],
    )
    return k, t, infidelity, positivity


def refine_worst_case(
    params: Params,
    config: Config,
    k: np.ndarray,
    t: np.ndarray,
    infidelity: np.ndarray,
    output_dir: Path,
    logger: object,
) -> dict[str, float]:
    """Continuously maximize infidelity over the torus from the worst seeds.

    Mirrors the vector run's node refinement.  For the density model there is
    no forced node, so the result is a worst-case point rather than a nodal
    set; no topological charge is defined at it.
    """

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
            _, _, values = model_values(params, raw[:, 0], raw[:, 1], config.mass)
            return -jnp.sum(values)

        gradients = jax.grad(objective)(current)
        updates, next_state = optimizer.update(gradients, current_state)
        return optax.apply_updates(current, updates), next_state

    logger.info("Refining %d torus worst-case seeds", config.refine_seeds)
    for _ in range(config.refine_steps):
        angles, state = step(angles, state)
    candidates = np.mod(np.asarray(jax.device_get(angles)), 2.0 * np.pi)
    bloch, fidelity, refined = jax.device_get(
        model_values(
            params,
            jnp.asarray(candidates[:, 0]),
            jnp.asarray(candidates[:, 1]),
            config.mass,
        )
    )
    refined = np.asarray(refined)
    best = int(np.argmax(refined))
    norm = float(np.linalg.norm(np.asarray(bloch)[best]))
    worst = {
        "k": float(candidates[best, 0]),
        "t": float(candidates[best, 1]),
        "fidelity": float(np.asarray(fidelity)[best]),
        "infidelity": float(refined[best]),
        "bloch_norm": norm,
        "purity": 0.5 * (1.0 + norm**2),
        "minimum_eigenvalue": 0.5 * (1.0 - norm),
    }
    write_rows(output_dir / "worst_case.csv", [worst])
    logger.info(
        "Refined torus worst case: 1-F=%.6e at (k, t)=(%.6f, %.6f)",
        worst["infidelity"],
        worst["k"],
        worst["t"],
    )
    return worst


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("all", "train", "evaluate"), default="all")
    result.add_argument(
        "--output-dir", type=Path, default=FIGURE2_RESULTS / "panel_b_density"
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
    logger = configure_logger(output_dir, "figure2.panel_b_density")
    logger.info("Python executable: %s", PUBLIC_ENV_PYTHON)
    manifest = run_manifest("b_density", config, versions)
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
            save_checkpoint(params, checkpoint, "b_density", config, versions, elapsed)
            write_history(output_dir / "training_history.csv", history)
    else:
        params = load_checkpoint(checkpoint)

    if args.mode in ("all", "evaluate"):
        surface_path = output_dir / "periodic_surface.npz"
        worst_path = output_dir / "worst_case.csv"
        if surface_path.exists() and worst_path.exists() and not args.force:
            logger.info("Banked evaluation exists; skipping dense evaluation")
            return
        k, t, infidelity, positivity = evaluate_dense(
            params, config, output_dir, logger
        )
        worst = refine_worst_case(
            params, config, k, t, infidelity, output_dir, logger
        )
        write_json(
            output_dir / "summary.json",
            {
                "created_utc": utc_now(),
                "representation": "density matrix",
                "parameter_count": parameter_count(params),
                "surface": surface_path.name,
                "worst_case": worst_path.name,
                "grid": {"k": len(k), "t": len(t)},
                "grid_max_infidelity": float(np.max(infidelity)),
                "refined_max_infidelity": worst["infidelity"],
                "refined_min_fidelity": worst["fidelity"],
                **positivity,
            },
        )
        logger.info("Periodic density control artifacts complete")


if __name__ == "__main__":
    main()
