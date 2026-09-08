#!/usr/bin/env python3
"""Seed exploration for the throttling node census.

Repeats the most throttled data and step configurations of
``run_supp_throttling.py`` over several seeds, plus the width-8 capacity
throttle at extra seeds as a control, to establish whether neutral pairs from
data or step throttling are seed-dependent.  Banks the same artifacts per run
under results/supp_throttling/explore_<key>_seed<seed>/.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import numpy as np

from figure2_common import check_environment, configure_logger, utc_now, write_json
from run_figure2a_finite_time import Config as Figure2Config
from run_figure2a_finite_time import train
from run_supp_throttling import OUTPUT, census


EXPLORE_SEEDS = (2, 5, 11, 23)
CONFIGS = {
    "data_26": {"grid": (1, 2), "steps": 4500, "width": 64, "layers": 3},
    "data_234": {"grid": (3, 6), "steps": 4500, "width": 64, "layers": 3},
    "steps_45": {"grid": (28, 56), "steps": 45, "width": 64, "layers": 3},
    "width_8": {"grid": (28, 56), "steps": 1500, "width": 8, "layers": 1},
}


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    check_environment()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(OUTPUT / "explore", "supp.throttling.explore")

    manifest: dict[str, dict[str, object]] = {}
    for key, spec in CONFIGS.items():
        for seed in EXPLORE_SEEDS:
            if key == "width_8" and seed == 2:
                continue  # the banked neutral-pair run is exactly this seed
            run_key = f"explore_{key}_seed{seed}"
            job_dir = OUTPUT / run_key
            job_dir.mkdir(parents=True, exist_ok=True)
            if (job_dir / "node_counts.csv").exists():
                logger.info("[%s] banked; skipping", run_key)
                continue
            config = Figure2Config(
                seed=seed,
                hidden_width=spec["width"],
                hidden_layers=spec["layers"],
                train_cos_theta=spec["grid"][0],
                train_phi=spec["grid"][1],
                train_time=13,
                steps=spec["steps"],
            )
            start = time.perf_counter()
            params, history, _ = train(config, logger)
            _, count_rows = census(params, job_dir, logger)
            counts = [int(row["count"]) for row in count_rows]
            summary = {
                "created_utc": utc_now(),
                "config": key,
                "seed": seed,
                "training_points": config.train_cos_theta
                * config.train_phi
                * config.train_time,
                "steps": config.steps,
                "hidden_width": config.hidden_width,
                "count_min": min(counts),
                "count_max": max(counts),
                "count_mean": float(np.mean(counts)),
                "multi_node_slices": int(sum(count > 1 for count in counts)),
                "final_mean_infidelity": history[-1]["mean_infidelity"],
            }
            write_json(job_dir / "summary.json", summary)
            manifest[run_key] = summary
            jax.clear_caches()
            logger.info(
                "[%s] complete in %.1f s: counts %d..%d",
                run_key,
                time.perf_counter() - start,
                summary["count_min"],
                summary["count_max"],
            )
    write_json(
        OUTPUT / "explore_manifest.json",
        {"created_utc": utc_now(), "jobs": manifest},
    )
    logger.info("Seed exploration complete")


if __name__ == "__main__":
    main()
