#!/usr/bin/env python3
"""Bank fidelity-susceptibility scans of banked models for the Supplement.

Evaluation only -- no training happens here.  Three banked checkpoints are
scanned: the static wavefunction-valued FNQS (seed 23), its matched
density-matrix control, and the finite-time wavefunction model of Fig. 2.
For pure (or numerically pure) qubit states the Fubini-Study fidelity
susceptibility along a unit-speed parameter path is

    chi(s) = <d psi|d psi> - |<psi|d psi>|^2 = |d b/ds|^2 / 4,

with b the Bloch vector, and the exact ground-state family gives chi = 1/4
identically.  The practitioner estimator chi_hat = 2[1-F(s, s+dl)]/dl^2 is
also banked for several dl.  The script additionally banks the dense
largest-eigenvalue map of the pulled-back fidelity metric over the display
sphere, and healing radii measured from the banked dense surfaces.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from figure2_common import (
    check_environment,
    configure_logger,
    finite_features,
    finite_target_bloch,
    qubit_bloch,
    utc_now,
    write_json,
    write_rows,
)
from figure2_common import load_checkpoint as load_dynamic_checkpoint
from figure2_common import normalized_qubit as dynamic_normalized_qubit
from one_qubit_fnqs_obstruction import (
    display_to_hamiltonian_matrix,
    load_checkpoint as load_static_checkpoint,
    normalized_qubit as static_normalized_qubit,
)
from one_qubit_density_fnqs_lift import (
    density_matrix,
    load_checkpoint as load_density_checkpoint,
)
from run_figure2a_finite_time import Config as DynamicConfig


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS = SCRIPT_DIR / "results"
OUTPUT = RESULTS / "supp_susceptibility"
STATIC_WEIGHTS = RESULTS / "one_qubit_fnqs" / "weights.npz"
STATIC_X_STAR = RESULTS / "one_qubit_fnqs" / "x_star_t241_p481.csv"
STATIC_SURFACE = RESULTS / "one_qubit_fnqs" / "surface_t241_p481.csv"
DENSITY_WEIGHTS = RESULTS / "one_qubit_density_fnqs" / "density_weights.npz"
DYNAMIC_CHECKPOINT = RESULTS / "figure2_dynamics" / "panel_a" / "checkpoint.npz"
DYNAMIC_WORLDLINE = RESULTS / "figure2_dynamics" / "panel_a" / "nodal_worldline.csv"
DYNAMIC_SURFACE = RESULTS / "figure2_dynamics" / "panel_a" / "finite_surface.npz"

SCAN_HALF_LENGTH = 0.6
SCAN_POINTS = 4801
OFFSETS = tuple(np.linspace(0.0, 0.3, 50))
SCAN_BANK_OFFSETS = (0.0,)
ESTIMATOR_DELTAS = (1.0e-3, 1.0e-2)
MAP_THETA = 241
MAP_PHI = 481
STATIC_TRAIN_SPACING = math.sqrt(4.0 * math.pi / (96 * 192))
DYNAMIC_TRAIN_SPACING = math.sqrt(4.0 * math.pi / (28 * 56))


def orthonormal_frame(center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray((0.0, 0.0, 1.0))
    if abs(float(center @ reference)) > 0.85:
        reference = np.asarray((0.0, 1.0, 0.0))
    tangent = np.cross(reference, center)
    tangent /= np.linalg.norm(tangent)
    binormal = np.cross(center, tangent)
    return tangent, binormal


def path_points(
    center: np.ndarray, offset: float, s: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Unit-speed great circle with closest approach `offset` to `center`."""

    tangent, binormal = orthonormal_frame(center)
    start = math.cos(offset) * center + math.sin(offset) * binormal
    return (
        np.cos(s)[:, None] * start[None, :] + np.sin(s)[:, None] * tangent[None, :],
        start,
    )


def susceptibility_from_bloch(bloch_of_scalar, s_values: np.ndarray) -> np.ndarray:
    """chi(s) = |db/ds|^2 / 4 by forward-mode differentiation."""

    derivative = jax.vmap(jax.jacfwd(bloch_of_scalar))(jnp.asarray(s_values))
    return 0.25 * np.asarray(jnp.sum(derivative**2, axis=-1))


def estimator(bloch: np.ndarray, s_step: float, delta: float) -> np.ndarray:
    """chi_hat(s) = 2[1 - F(s, s+delta)]/delta^2 with the amplitude overlap.

    F is the overlap amplitude |<psi(s)|psi(s+delta)>| of Gu's convention, so
    chi_hat converges to the Fubini-Study susceptibility as delta -> 0.
    """

    shift = int(round(delta / s_step))
    overlap_squared = 0.5 * (1.0 + np.sum(bloch[:-shift] * bloch[shift:], axis=-1))
    amplitude = np.sqrt(np.clip(overlap_squared, 0.0, None))
    values = 2.0 * (1.0 - amplitude) / (shift * s_step) ** 2
    return np.pad(values, (0, shift), constant_values=np.nan)


def read_x_star() -> np.ndarray:
    data = np.genfromtxt(STATIC_X_STAR, delimiter=",", names=True)
    return np.asarray((float(data["n_x"]), float(data["n_y"]), float(data["n_z"])))


def sphere_distance(points: np.ndarray, center: np.ndarray) -> np.ndarray:
    return np.arccos(np.clip(points @ center, -1.0, 1.0))


def healing_radius(
    points: np.ndarray, infidelity: np.ndarray, center: np.ndarray, level: float
) -> float:
    """Radius of the {1-F >= level} disk around the node."""

    inside = infidelity >= level
    if not np.any(inside):
        return float("nan")
    distances = sphere_distance(points[inside.reshape(-1)], center)
    local = distances[distances < 0.5]
    return float(np.max(local)) if local.size else float("nan")


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    check_environment()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(OUTPUT, "supp.susceptibility")

    static_params = load_static_checkpoint(STATIC_WEIGHTS)
    density_params = load_density_checkpoint(DENSITY_WEIGHTS)
    dynamic_params = load_dynamic_checkpoint(DYNAMIC_CHECKPOINT)
    rotation = np.asarray(display_to_hamiltonian_matrix("generic"))
    x_star_display = read_x_star()
    x_star = x_star_display @ rotation
    logger.info("Static node (Hamiltonian frame): %s", x_star)

    def static_bloch(points: jax.Array) -> jax.Array:
        return qubit_bloch(static_normalized_qubit(static_params, points))

    def density_bloch(points: jax.Array) -> jax.Array:
        return density_matrix(density_params, points)[1]

    s_values = np.linspace(-SCAN_HALF_LENGTH, SCAN_HALF_LENGTH, SCAN_POINTS)
    s_step = float(s_values[1] - s_values[0])
    scan_summary: list[dict[str, float]] = []
    for offset in OFFSETS:
        points, start = path_points(x_star, offset, s_values)
        tangent, _ = orthonormal_frame(x_star)
        start_j = jnp.asarray(start)
        tangent_j = jnp.asarray(tangent)

        def path(s: jax.Array) -> jax.Array:
            return jnp.cos(s) * start_j + jnp.sin(s) * tangent_j

        chi_vector = susceptibility_from_bloch(
            lambda s: static_bloch(path(s)), s_values
        )
        chi_density = susceptibility_from_bloch(
            lambda s: density_bloch(path(s)), s_values
        )
        bloch_vector = np.asarray(static_bloch(jnp.asarray(points)))
        bloch_density = np.asarray(density_bloch(jnp.asarray(points)))
        target_fidelity = 0.5 * (1.0 + np.sum(bloch_vector * points, axis=-1))
        density_fidelity = 0.5 * (1.0 + np.sum(bloch_density * points, axis=-1))
        record: dict[str, np.ndarray] = {
            "s": s_values,
            "chi_vector": chi_vector,
            "chi_density": chi_density,
            "vector_infidelity": 1.0 - target_fidelity,
            "density_infidelity": 1.0 - density_fidelity,
        }
        for delta in ESTIMATOR_DELTAS:
            record[f"chi_hat_vector_d{delta:.0e}"] = estimator(
                bloch_vector, s_step, delta
            )
            record[f"chi_hat_density_d{delta:.0e}"] = estimator(
                bloch_density, s_step, delta
            )
        if any(abs(offset - banked) < 1.0e-12 for banked in SCAN_BANK_OFFSETS):
            np.savez_compressed(OUTPUT / f"scan_offset_{offset:.3f}.npz", **record)
        peak_index = int(np.argmax(chi_vector))
        scan_summary.append(
            {
                "offset": offset,
                "peak_chi_vector": float(chi_vector[peak_index]),
                "peak_location": float(s_values[peak_index]),
                "peak_chi_hat_fine": float(
                    np.nanmax(record[f"chi_hat_vector_d{ESTIMATOR_DELTAS[0]:.0e}"])
                ),
                "peak_chi_hat_coarse": float(
                    np.nanmax(record[f"chi_hat_vector_d{ESTIMATOR_DELTAS[1]:.0e}"])
                ),
                "max_chi_density": float(np.max(chi_density)),
                "max_vector_infidelity": float(np.max(1.0 - target_fidelity)),
                "max_density_infidelity": float(np.max(1.0 - density_fidelity)),
            }
        )
    logger.info(
        "Swept %d offsets: through-node peak chi %.3e, max density chi %.6f",
        len(OFFSETS),
        scan_summary[0]["peak_chi_vector"],
        max(row["max_chi_density"] for row in scan_summary),
    )
    write_rows(OUTPUT / "scan_peaks.csv", scan_summary)

    # Dense largest-eigenvalue map of the pulled-back fidelity metric.
    theta = np.linspace(0.0, np.pi, MAP_THETA)
    phi = np.linspace(0.0, 2.0 * np.pi, MAP_PHI)
    rotation_j = jnp.asarray(rotation)

    def display_point(angles: jax.Array) -> jax.Array:
        t, p = angles[0], angles[1]
        return jnp.stack(
            (jnp.sin(t) * jnp.cos(p), jnp.sin(t) * jnp.sin(p), jnp.cos(t))
        )

    def lambda_max_of(bloch_function):
        def one(angles: jax.Array) -> jax.Array:
            jacobian = jax.jacfwd(
                lambda a: bloch_function(display_point(a) @ rotation_j)
            )(angles)
            gram = 0.25 * jacobian.T @ jacobian
            half_trace = 0.5 * (gram[0, 0] + gram[1, 1])
            radius = jnp.sqrt(
                0.25 * (gram[0, 0] - gram[1, 1]) ** 2 + gram[0, 1] ** 2
            )
            return half_trace + radius

        return one

    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    angle_grid = jnp.asarray(
        np.column_stack((tt.reshape(-1), pp.reshape(-1)))
    )
    logger.info("Evaluating %d x %d fidelity-metric maps", MAP_THETA, MAP_PHI)
    vector_map = np.asarray(
        jax.lax.map(lambda_max_of(static_bloch), angle_grid, batch_size=4096)
    ).reshape((MAP_THETA, MAP_PHI))
    density_map = np.asarray(
        jax.lax.map(lambda_max_of(density_bloch), angle_grid, batch_size=4096)
    ).reshape((MAP_THETA, MAP_PHI))
    full_circle = np.linspace(-np.pi, np.pi, 2001)
    scan_path_hamiltonian, _ = path_points(x_star, 0.0, full_circle)
    scan_path_display = scan_path_hamiltonian @ rotation.T
    np.savez_compressed(
        OUTPUT / "metric_maps.npz",
        theta=theta,
        phi=phi,
        vector_lambda_max=vector_map.astype(np.float32),
        density_lambda_max=density_map.astype(np.float32),
        x_star_theta=math.acos(float(np.clip(x_star_display[2], -1.0, 1.0))),
        x_star_phi=math.atan2(float(x_star_display[1]), float(x_star_display[0]))
        % (2.0 * math.pi),
        scan_path_theta=np.arccos(np.clip(scan_path_display[:, 2], -1.0, 1.0)),
        scan_path_phi=np.mod(
            np.arctan2(scan_path_display[:, 1], scan_path_display[:, 0]),
            2.0 * np.pi,
        ),
    )

    # Healing radii from the banked dense surfaces.
    surface = np.genfromtxt(
        STATIC_SURFACE,
        delimiter=",",
        names=True,
        usecols=("n_x", "n_y", "n_z", "infidelity"),
    )
    static_points = np.column_stack(
        (surface["n_x"], surface["n_y"], surface["n_z"])
    )
    static_epsilon = {
        f"static_epsilon_{level:g}": healing_radius(
            static_points, np.asarray(surface["infidelity"]), x_star_display, level
        )
        for level in (0.5, 0.1)
    }

    dynamic_surface = np.load(DYNAMIC_SURFACE)
    worldline = np.genfromtxt(DYNAMIC_WORLDLINE, delimiter=",", names=True)
    time_index = int(np.argmin(np.abs(dynamic_surface["tau"] - 0.5)))
    row = int(np.argmin(np.abs(worldline["tau"] - 0.5)))
    dynamic_center = np.asarray(
        (worldline["n_x"][row], worldline["n_y"][row], worldline["n_z"][row])
    )
    d_theta, d_phi = dynamic_surface["theta"], dynamic_surface["phi"]
    d_tt, d_pp = np.meshgrid(d_theta, d_phi, indexing="ij")
    dynamic_points = np.stack(
        (
            np.sin(d_tt) * np.cos(d_pp),
            np.sin(d_tt) * np.sin(d_pp),
            np.cos(d_tt),
        ),
        axis=-1,
    ).reshape((-1, 3))
    dynamic_epsilon = {
        f"dynamic_epsilon_{level:g}": healing_radius(
            dynamic_points,
            dynamic_surface["infidelity"][time_index].reshape(-1),
            dynamic_center,
            level,
        )
        for level in (0.5, 0.1)
    }

    # Susceptibility scan across the finite-time worldline at t/T = 1/2.
    dyn_tangent, _ = orthonormal_frame(dynamic_center)
    center_j = jnp.asarray(dynamic_center)
    dyn_tangent_j = jnp.asarray(dyn_tangent)

    def dynamic_bloch_of_s(s: jax.Array) -> jax.Array:
        point = jnp.cos(s) * center_j + jnp.sin(s) * dyn_tangent_j
        features = finite_features(point[None, :], jnp.asarray((0.5,)))
        return qubit_bloch(dynamic_normalized_qubit(dynamic_params, features))[0]

    chi_dynamic = susceptibility_from_bloch(dynamic_bloch_of_s, s_values)
    dynamic_points_path, _ = path_points(dynamic_center, 0.0, s_values)
    dynamic_target = np.asarray(
        finite_target_bloch(
            jnp.asarray(dynamic_points_path),
            jnp.full((SCAN_POINTS,), 0.5),
            DynamicConfig.total_rotation,
        )
    )
    dynamic_bloch_path = np.asarray(
        qubit_bloch(
            dynamic_normalized_qubit(
                dynamic_params,
                finite_features(
                    jnp.asarray(dynamic_points_path), jnp.full((SCAN_POINTS,), 0.5)
                ),
            )
        )
    )
    dynamic_infidelity = 0.5 * (
        1.0 - np.sum(dynamic_bloch_path * dynamic_target, axis=-1)
    )
    np.savez_compressed(
        OUTPUT / "scan_dynamic_worldline.npz",
        s=s_values,
        chi_vector=chi_dynamic,
        vector_infidelity=dynamic_infidelity,
    )

    summary = {
        "created_utc": utc_now(),
        "scan_half_length": SCAN_HALF_LENGTH,
        "scan_points": SCAN_POINTS,
        "offsets": list(OFFSETS),
        "estimator_deltas": list(ESTIMATOR_DELTAS),
        "exact_chi": 0.25,
        "peak_chi_vector_through_node": scan_summary[0]["peak_chi_vector"],
        "peak_over_exact_ratio": scan_summary[0]["peak_chi_vector"] / 0.25,
        "max_chi_density_all_scans": max(
            row["max_chi_density"] for row in scan_summary
        ),
        "max_density_infidelity_all_scans": max(
            row["max_density_infidelity"] for row in scan_summary
        ),
        "map_grid": {"theta": MAP_THETA, "phi": MAP_PHI},
        "map_max_vector_lambda": float(np.max(vector_map)),
        "map_max_density_lambda": float(np.max(density_map)),
        "map_max_density_excess": float(np.max(np.abs(4.0 * density_map - 1.0))),
        "static_train_spacing": STATIC_TRAIN_SPACING,
        "dynamic_train_spacing": DYNAMIC_TRAIN_SPACING,
        **static_epsilon,
        **dynamic_epsilon,
        "peak_chi_dynamic_worldline": float(np.max(chi_dynamic)),
        "training_grid_max_infidelity_dynamic": float(
            np.genfromtxt(
                RESULTS / "figure2_dynamics" / "panel_a" / "training_history.csv",
                delimiter=",",
                names=True,
            )["max_grid_infidelity"][-1]
        ),
    }
    write_json(OUTPUT / "summary.json", summary)
    logger.info(
        "Peak chi through node %.3e (exact 0.25); static epsilon(0.5) %.4f "
        "vs training spacing %.4f",
        summary["peak_chi_vector_through_node"],
        summary["static_epsilon_0.5"],
        STATIC_TRAIN_SPACING,
    )
    logger.info("Supplement susceptibility bank complete")


if __name__ == "__main__":
    main()
