#!/usr/bin/env python3
"""Bank panel (c): the endpoint gluing phase of the periodic target.

For each momentum k, the target ray is parallel transported once around the
time circle.  Its endpoint phase g(k) is saved together with an unwrapped phase
and winding.  No model training or plotting occurs in this script.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from figure2_common import (
    FIGURE2_RESULTS,
    check_environment,
    configure_logger,
    qwz_target_bloch,
    run_manifest,
    spinor_from_bloch,
    utc_now,
    write_json,
    write_rows,
)


@dataclass(frozen=True)
class Config:
    mass: float = -1.0
    k_points: int = 401
    transport_steps: int = 2_001


def adaptive_spinors(bloch: np.ndarray) -> np.ndarray:
    north = spinor_from_bloch(bloch, prefer_north=True)
    south = spinor_from_bloch(bloch, prefer_north=False)
    use_north = bloch[..., 2] >= 0.0
    return np.where(use_north[..., None], north, south)


def parallel_transport_phase(bloch_path: np.ndarray) -> complex:
    raw = adaptive_spinors(bloch_path)
    initial = raw[0] / np.linalg.norm(raw[0])
    previous = initial
    for candidate in raw[1:]:
        candidate = candidate / np.linalg.norm(candidate)
        overlap = np.vdot(previous, candidate)
        if abs(overlap) < 1.0e-12:
            raise RuntimeError("Transport grid is too coarse: adjacent rays are orthogonal")
        previous = candidate * np.exp(-1j * np.angle(overlap))
    endpoint_overlap = np.vdot(initial, previous)
    return complex(endpoint_overlap / abs(endpoint_overlap))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--output-dir", type=Path, default=FIGURE2_RESULTS / "panel_c"
    )
    result.add_argument("--force", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    config = Config()
    versions = check_environment()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(output_dir, "figure2.panel_c")
    output_csv = output_dir / "gluing_phase.csv"
    if output_csv.exists() and not args.force:
        logger.info("Banked gluing phase exists; skipping transport")
        return

    write_json(output_dir / "manifest.json", run_manifest("c", config, versions))
    k_values = np.linspace(0.0, 2.0 * np.pi, config.k_points, dtype=np.float64)
    t_values = np.linspace(
        0.0, 2.0 * np.pi, config.transport_steps, dtype=np.float64
    )
    phases = np.empty(config.k_points, dtype=np.complex128)
    logger.info(
        "Parallel transporting %d momentum rays across %d time samples",
        config.k_points,
        config.transport_steps,
    )
    for index, k_value in enumerate(k_values):
        k_path = jnp.full((config.transport_steps,), k_value, dtype=jnp.float64)
        bloch = np.asarray(
            jax.device_get(
                qwz_target_bloch(k_path, jnp.asarray(t_values), config.mass)
            )
        )
        phases[index] = parallel_transport_phase(bloch)

    wrapped = np.angle(phases)
    unwrapped = np.unwrap(wrapped)
    winding = float((unwrapped[-1] - unwrapped[0]) / (2.0 * np.pi))
    rounded_winding = int(np.rint(winding))
    rows = [
        {
            "k": float(k_value),
            "g_real": float(np.real(phase)),
            "g_imag": float(np.imag(phase)),
            "phase_wrapped": float(wrapped[index]),
            "phase_unwrapped": float(unwrapped[index]),
        }
        for index, (k_value, phase) in enumerate(zip(k_values, phases, strict=True))
    ]
    write_rows(output_csv, rows)
    write_json(
        output_dir / "summary.json",
        {
            "created_utc": utc_now(),
            "gluing_phase": output_csv.name,
            "winding_raw": winding,
            "winding_integer": rounded_winding,
            "maximum_modulus_error": float(np.max(np.abs(np.abs(phases) - 1.0))),
        },
    )
    logger.info(
        "Panel (c) artifacts complete; gluing winding %.12f (rounded %d)",
        winding,
        rounded_winding,
    )


if __name__ == "__main__":
    main()
