#!/usr/bin/env python3
"""Train the static one-qubit models with Metropolis-sampled gradients.

This is the sampling-noise control for the static obstruction and its density
lift.  The identical 3-48-48-3 networks, seed, deterministic training grid,
optimizer, and schedule of ``one_qubit_fnqs_obstruction.py`` and
``one_qubit_density_fnqs_lift.py`` are retrained with the exact mean-energy
objective replaced by a Monte Carlo estimator: at every training point, basis
configurations are Metropolis-sampled from the model's own Born distribution
(16 chains, 16 burn-in and 64 recorded sweeps per step), and gradients follow
the standard baseline-subtracted local-energy surrogate.  Dense evaluation,
worst-case refinement, and plotting then reuse the exact pipelines of the two
parent scripts unchanged, so the banked artifacts are directly comparable.

Outputs: ``results/one_qubit_fnqs_sampled`` and
``results/one_qubit_density_fnqs_sampled`` plus
``figures/figureS3_sampled_obstruction`` (sampled wavefunction-valued
obstruction) and ``figures/figureS4_sampled_density_lift`` (sampled
vector-versus-density comparison).
"""

from __future__ import annotations

import argparse
import csv
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from tqdm.auto import tqdm

import one_qubit_density_fnqs_lift as density_static
import one_qubit_fnqs_obstruction as vector_static
import plot_density_lift_comparison
from one_qubit_fnqs_obstruction import (
    ExperimentConfig,
    Params,
    check_environment,
    equal_area_training_grid,
    initialize_mlp,
    make_optimizer,
    normalized_qubit,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
VECTOR_DIR = SCRIPT_DIR / "results" / "one_qubit_fnqs_sampled"
DENSITY_DIR = SCRIPT_DIR / "results" / "one_qubit_density_fnqs_sampled"
FIGURES = PROJECT_ROOT / "figures"


@dataclass(frozen=True)
class SamplerConfig:
    chains: int = 16
    burn_in: int = 16
    sweeps: int = 64
    sample_seed: int = 0


def hamiltonian_elements(points: jax.Array) -> tuple[jax.Array, ...]:
    n_x, n_y, n_z = (points[..., index] for index in range(3))
    return -n_z, n_z, -(n_x - 1j * n_y), -(n_x + 1j * n_y)


def safe_divide(numerator: jax.Array, denominator: jax.Array) -> jax.Array:
    safe = jnp.where(jnp.abs(denominator) > 1.0e-12, denominator, 1.0e-12)
    return numerator / safe


def metropolis(
    key: jax.Array, log_probability: jax.Array, sampler: SamplerConfig
) -> jax.Array:
    """Sample basis configurations per point from a two-outcome distribution."""

    log_probability = jax.lax.stop_gradient(log_probability)
    log_ratio = log_probability[..., 1] - log_probability[..., 0]
    key_init, key_sweeps = jax.random.split(key)
    state = jax.random.bernoulli(
        key_init, 0.5, (sampler.chains,) + log_probability.shape[:-1]
    )

    def sweep(current: jax.Array, sweep_key: jax.Array) -> tuple[jax.Array, jax.Array]:
        uniform = jax.random.uniform(sweep_key, current.shape)
        accept = jnp.log(uniform) < jnp.where(current, -log_ratio, log_ratio)
        proposal = jnp.where(accept, ~current, current)
        return proposal, proposal

    keys = jax.random.split(key_sweeps, sampler.burn_in + sampler.sweeps)
    _, states = jax.lax.scan(sweep, state, keys)
    return states[sampler.burn_in :].reshape((-1,) + state.shape[1:])


def vector_sampled_energy(
    params: Params, points: jax.Array, key: jax.Array, sampler: SamplerConfig
) -> tuple[jax.Array, jax.Array]:
    """Baseline-subtracted local-energy surrogate for the wavefunction model."""

    psi = normalized_qubit(params, points)
    h00, h11, h01, h10 = hamiltonian_elements(points)
    local = jnp.stack(
        (
            h00 + h01 * safe_divide(psi[..., 1], psi[..., 0]),
            h11 + h10 * safe_divide(psi[..., 0], psi[..., 1]),
        ),
        axis=-1,
    )
    log_psi = jnp.log(jnp.maximum(jnp.abs(psi), 1.0e-150)) + 1j * jnp.angle(psi)
    configurations = metropolis(key, 2.0 * jnp.real(log_psi), sampler)
    local_s = jnp.where(configurations, local[..., 1], local[..., 0])
    log_psi_s = jnp.where(configurations, log_psi[..., 1], log_psi[..., 0])
    baseline = jnp.mean(local_s, axis=0, keepdims=True)
    surrogate = 2.0 * jnp.real(
        jnp.mean(jax.lax.stop_gradient(jnp.conj(local_s - baseline)) * log_psi_s)
    )
    return surrogate, jnp.mean(jnp.real(local_s))


def density_sampled_energy(
    params: Params, points: jax.Array, key: jax.Array, sampler: SamplerConfig
) -> tuple[jax.Array, jax.Array]:
    """The analogous estimator sampling from the density-matrix diagonal."""

    rho, _ = density_static.density_matrix(params, points)
    h00, h11, h01, h10 = hamiltonian_elements(points)
    rho_00 = jnp.real(rho[..., 0, 0])
    rho_11 = jnp.real(rho[..., 1, 1])
    local = jnp.stack(
        (
            jnp.real(h00 + h10 * safe_divide(rho[..., 0, 1], rho_00)),
            jnp.real(h11 + h01 * safe_divide(rho[..., 1, 0], rho_11)),
        ),
        axis=-1,
    )
    log_p = jnp.log(jnp.maximum(jnp.stack((rho_00, rho_11), axis=-1), 1.0e-300))
    configurations = metropolis(key, log_p, sampler)
    local_s = jnp.where(configurations, local[..., 1], local[..., 0])
    log_p_s = jnp.where(configurations, log_p[..., 1], log_p[..., 0])
    baseline = jnp.mean(local_s, axis=0, keepdims=True)
    surrogate = jnp.mean(
        jax.lax.stop_gradient(local_s - baseline) * log_p_s + local_s
    )
    return surrogate, jnp.mean(local_s)


ESTIMATORS = {"vector": vector_sampled_energy, "density": density_sampled_energy}
EXACT_OBSERVABLES = {
    "vector": vector_static.observables,
    "density": density_static.observables,
}


def train(
    kind: str, config: ExperimentConfig, sampler: SamplerConfig
) -> tuple[Params, list[dict[str, float]], float]:
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
    estimator = ESTIMATORS[kind]
    exact = EXACT_OBSERVABLES[kind]
    root_key = jax.random.PRNGKey(sampler.sample_seed)

    @jax.jit
    def training_step(
        current_params: Params, current_state: optax.OptState, step: jax.Array
    ) -> tuple[Params, optax.OptState, jax.Array]:
        key = jax.random.fold_in(root_key, step)

        def loss(candidate: Params) -> tuple[jax.Array, jax.Array]:
            return estimator(candidate, training_points, key, sampler)

        (_, sampled_energy), gradients = jax.value_and_grad(loss, has_aux=True)(
            current_params
        )
        updates, next_state = optimizer.update(
            gradients, current_state, current_params
        )
        return optax.apply_updates(current_params, updates), next_state, sampled_energy

    @jax.jit
    def training_metrics(current_params: Params) -> tuple[jax.Array, ...]:
        values = exact(current_params, training_points)
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
        desc=(
            f"sampled {kind} ({training_points.shape[0]:,} points x "
            f"{sampler.chains * sampler.sweeps} samples)"
        ),
        unit="step",
        dynamic_ncols=True,
        mininterval=0.2,
    ) as progress:
        for step in range(1, config.steps + 1):
            params, optimizer_state, sampled_energy = training_step(
                params, optimizer_state, jnp.asarray(step)
            )
            if step == 1 or step % config.log_every == 0 or step == config.steps:
                metrics = jax.device_get(training_metrics(params))
                record = {
                    "step": float(step),
                    "sampled_energy": float(sampled_energy),
                    "mean_energy": float(metrics[0]),
                    "mean_energy_error": float(metrics[1]),
                    "max_grid_energy_error": float(metrics[2]),
                    "min_grid_fidelity": float(metrics[3]),
                    "elapsed_seconds": time.perf_counter() - start_time,
                }
                history.append(record)
                progress.set_postfix(
                    E_mc=f"{record['sampled_energy']:+.6f}",
                    E=f"{record['mean_energy']:+.9f}",
                    min_F0=f"{record['min_grid_fidelity']:.8f}",
                    refresh=False,
                )
            progress.update(1)
    return params, history, time.perf_counter() - start_time


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--force-train", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    versions = check_environment()
    config = ExperimentConfig(seed=args.seed, steps=args.steps)
    sampler = SamplerConfig(sample_seed=args.sample_seed)

    for kind, output_dir, module, checkpoint_name in (
        ("vector", VECTOR_DIR, vector_static, "weights.npz"),
        ("density", DENSITY_DIR, density_static, "density_weights.npz"),
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / checkpoint_name
        if checkpoint_path.exists() and not args.force_train:
            print(f"Checkpoint exists; training skipped: {checkpoint_path}")
            continue
        params, history, elapsed = train(kind, config, sampler)
        module.save_checkpoint(params, checkpoint_path, config, versions, elapsed)
        with (output_dir / "training_history.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        print(f"Saved checkpoint: {checkpoint_path}")

    # Dense evaluation, refinement, and plots via the exact parent pipelines.
    vector_static.main(
        [
            "--mode",
            "evaluate",
            "--output-dir",
            str(VECTOR_DIR),
            "--coordinate-rotation",
            "generic",
        ]
    )
    density_static.main(
        [
            "--mode",
            "evaluate",
            "--output-dir",
            str(DENSITY_DIR),
            "--coordinate-rotation",
            "generic",
        ]
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        shutil.copy(
            VECTOR_DIR / f"log_infidelity_sphere_zoom_t241_p481{suffix}",
            FIGURES / f"figureS3_sampled_obstruction{suffix}",
        )
    plot_density_lift_comparison.plot_comparison(
        plot_density_lift_comparison.load_surface(
            VECTOR_DIR / "surface_t241_p481.csv"
        ),
        plot_density_lift_comparison.load_surface(
            DENSITY_DIR / "density_surface_t241_p481.csv"
        ),
        FIGURES / "figureS4_sampled_density_lift",
    )
    print("Sampled static artifacts and figures complete")


if __name__ == "__main__":
    main()
