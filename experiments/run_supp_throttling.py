#!/usr/bin/env python3
"""Bank node censuses of data- and step-throttled expressive models.

The expressive finite-time architecture of Fig. 2(a,b) (4-64-64-64-3, seed 37)
is retrained under two independent restrictions and its refined space-time
fidelity nodes are counted per time slice, exactly as for the width-throttled
model of the neutral-pair figure:

* data throttling: the full 4500-step protocol on equal-area training grids of
  28 x 56, 9 x 18, 3 x 6, and 1 x 2 sphere points at 13 times
  (N = 20384, 2106, 234, 26 space-time points);
* step throttling: the full 20384-point grid trained for 45 and 450 steps
  (4500 steps is the banked Fig. 2 checkpoint, shared by both sweeps).

For every run the script banks the refined nodal worldlines, the per-slice
node count and total charge, and a summary.  No plotting happens here.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm.auto import tqdm

from figure2_common import (
    Params,
    check_environment,
    configure_logger,
    load_checkpoint,
    save_checkpoint,
    utc_now,
    write_json,
    write_rows,
)
from run_figure2a_finite_time import Config as Figure2Config
from run_figure2a_finite_time import model_values, train
from run_figure3_fnqs_pair_creation import (
    Config as PairConfig,
    fibonacci_sphere,
    node_charge,
    normalize,
)


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT = SCRIPT_DIR / "results" / "supp_throttling"
REFERENCE_CHECKPOINT = (
    SCRIPT_DIR / "results" / "figure2_dynamics" / "panel_a" / "checkpoint.npz"
)
CENSUS = PairConfig()

DATA_GRIDS = ((28, 56), (9, 18), (3, 6), (1, 2))
STEP_BUDGETS = (450, 45)
FULL_STEPS = 4500


def run_config(grid: tuple[int, int], steps: int) -> Figure2Config:
    return Figure2Config(
        seed=37,
        hidden_width=64,
        hidden_layers=3,
        train_cos_theta=grid[0],
        train_phi=grid[1],
        train_time=13,
        steps=steps,
    )


def census(
    params: Params, output_dir: Path, logger: object
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Refine and count fidelity nodes per time slice with winding charges.

    Identical protocol to the neutral-pair run, but every consistency check is
    recorded instead of asserted, since throttled models may carry any number
    of neutral pairs.
    """

    tau = np.linspace(0.0, 1.0, CENSUS.root_time, dtype=np.float64)
    seeds = fibonacci_sphere(CENSUS.root_seeds)
    vectors = jnp.asarray(
        np.broadcast_to(seeds, (CENSUS.root_time, CENSUS.root_seeds, 3)).copy()
    )
    times = jnp.asarray(
        np.broadcast_to(tau[:, None], (CENSUS.root_time, CENSUS.root_seeds))
    )
    def descend(current: jax.Array, learning_rate: float, steps: int) -> jax.Array:
        optimizer = optax.adam(learning_rate)
        state = optimizer.init(current)

        @jax.jit
        def step(vectors_in: jax.Array, state_in: optax.OptState):
            def objective(raw: jax.Array) -> jax.Array:
                fidelity = model_values(
                    params, normalize(raw), times, CENSUS.total_rotation
                )[1]
                return jnp.sum(fidelity)

            gradients = jax.grad(objective)(vectors_in)
            updates, next_state = optimizer.update(gradients, state_in)
            return normalize(optax.apply_updates(vectors_in, updates)), next_state

        for _ in range(steps):
            current, state = step(current, state)
        return current

    # Global descent as in the neutral-pair run, then a low-learning-rate
    # polish: the expressive model's fidelity basin is far narrower than the
    # width-8 model's, and the coarse stage alone stalls above the tolerance.
    vectors = descend(vectors, CENSUS.root_learning_rate, CENSUS.root_steps)
    vectors = descend(vectors, 1.0e-3, 500)
    vectors_np = np.asarray(jax.device_get(normalize(vectors)))
    fidelity_np = np.asarray(
        jax.device_get(
            model_values(params, normalize(vectors), times, CENSUS.total_rotation)[1]
        )
    )

    rows: list[dict[str, float]] = []
    count_rows: list[dict[str, float]] = []
    violation_count = 0
    for time_index in range(CENSUS.root_time):
        roots: list[tuple[np.ndarray, float]] = []
        for seed_index in np.argsort(fidelity_np[time_index]):
            fidelity = float(fidelity_np[time_index, seed_index])
            if fidelity > CENSUS.root_fidelity_tolerance:
                break
            point = vectors_np[time_index, seed_index]
            if all(
                math.acos(float(np.clip(point @ other, -1.0, 1.0)))
                > CENSUS.root_merge_angle
                for other, _ in roots
            ):
                roots.append((point, fidelity))
        total_charge = 0.0
        for node_index, (point, fidelity) in enumerate(roots):
            separations = [
                math.acos(float(np.clip(point @ other, -1.0, 1.0)))
                for other, _ in roots
                if not np.allclose(point, other)
            ]
            radius = CENSUS.winding_radius
            if separations:
                radius = min(radius, 0.28 * min(separations))
            charge = node_charge(
                params, CENSUS, point, float(tau[time_index]), radius
            )
            total_charge += charge
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
        if abs(total_charge + 1.0) > 0.25:
            violation_count += 1
        count_rows.append(
            {
                "tau": float(tau[time_index]),
                "count": float(len(roots)),
                "total_charge": total_charge,
            }
        )

    write_rows(output_dir / "nodal_worldlines.csv", rows)
    write_rows(output_dir / "node_counts.csv", count_rows)
    counts = [int(row["count"]) for row in count_rows]
    logger.info(
        "Census: counts %s, max %d, charge violations %d/%d slices",
        sorted(set(counts)),
        max(counts),
        violation_count,
        CENSUS.root_time,
    )
    if violation_count:
        logger.warning("Total charge deviated from -1 on %d slices", violation_count)
    return rows, count_rows


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    versions = check_environment()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(OUTPUT, "supp.throttling")

    jobs: list[tuple[str, Figure2Config, bool]] = []
    for grid in DATA_GRIDS:
        n_points = grid[0] * grid[1] * 13
        is_reference = grid == (28, 56)
        jobs.append((f"data_{n_points}", run_config(grid, FULL_STEPS), is_reference))
    for steps in STEP_BUDGETS:
        jobs.append((f"steps_{steps}", run_config((28, 56), steps), False))

    manifest: dict[str, dict[str, object]] = {}
    for key, config, is_reference in jobs:
        job_dir = OUTPUT / key
        job_dir.mkdir(parents=True, exist_ok=True)
        counts_path = job_dir / "node_counts.csv"
        if counts_path.exists():
            logger.info("[%s] banked census exists; skipping", key)
            continue
        n_points = config.train_cos_theta * config.train_phi * config.train_time
        logger.info(
            "[%s] N=%d training points, %d steps", key, n_points, config.steps
        )
        start = time.perf_counter()
        if is_reference:
            params = load_checkpoint(REFERENCE_CHECKPOINT)
            history: list[dict[str, float]] = []
            logger.info("[%s] using the banked Fig. 2 checkpoint", key)
        else:
            params, history, _ = train(config, logger)
            save_checkpoint(
                params,
                job_dir / "checkpoint.npz",
                key,
                config,
                versions,
                time.perf_counter() - start,
            )
        _, count_rows = census(params, job_dir, logger)
        counts = [int(row["count"]) for row in count_rows]
        summary = {
            "created_utc": utc_now(),
            "training_points": n_points,
            "steps": config.steps,
            "reference_checkpoint": is_reference,
            "count_min": min(counts),
            "count_max": max(counts),
            "count_mean": float(np.mean(counts)),
            "multi_node_slices": int(sum(count > 1 for count in counts)),
            "final_mean_infidelity": history[-1]["mean_infidelity"]
            if history
            else None,
        }
        write_json(job_dir / "summary.json", summary)
        manifest[key] = summary
        jax.clear_caches()
        logger.info(
            "[%s] complete in %.1f s: counts %d..%d",
            key,
            time.perf_counter() - start,
            summary["count_min"],
            summary["count_max"],
        )
    write_json(OUTPUT / "manifest.json", {"created_utc": utc_now(), "jobs": manifest})
    logger.info("Throttling bank complete")


if __name__ == "__main__":
    main()
