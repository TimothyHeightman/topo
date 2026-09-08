#!/usr/bin/env python3
"""Bank a dense sphere slice of the throttled pair-creation model.

Evaluation only.  Loads the banked width-8 checkpoint of the neutral-pair
run, evaluates its log-infidelity on a dense (theta, phi) grid at one
five-node time slice, and banks the slice together with the refined node
positions and charges at that time (taken from the banked worldline CSV).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from figure2_common import (
    check_environment,
    configure_logger,
    load_checkpoint,
    utc_now,
    write_json,
)
from run_figure2a_finite_time import model_values
from run_figure3_fnqs_pair_creation import Config as PairConfig


SCRIPT_DIR = Path(__file__).resolve().parent
PAIR_RESULTS = SCRIPT_DIR / "results" / "figure3_fnqs_pair"
OUTPUT = SCRIPT_DIR / "results" / "supp_pair_slice"
EVAL_THETA = 361
EVAL_PHI = 721


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    check_environment()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(OUTPUT, "supp.pair_slice")
    params = load_checkpoint(PAIR_RESULTS / "checkpoint.npz")
    worldlines = np.genfromtxt(
        PAIR_RESULTS / "nodal_worldlines.csv", delimiter=",", names=True
    )

    taus = np.unique(worldlines["tau"])
    counts = {tau: int(np.sum(worldlines["tau"] == tau)) for tau in taus}
    five_node_taus = sorted(tau for tau, count in counts.items() if count == 5)
    slice_tau = five_node_taus[len(five_node_taus) // 2]
    at_slice = worldlines[worldlines["tau"] == slice_tau]
    logger.info(
        "Slice tau %.6f: %d nodes, charges %s",
        slice_tau,
        len(at_slice),
        [float(charge) for charge in at_slice["charge"]],
    )

    theta = np.linspace(0.0, np.pi, EVAL_THETA)
    phi = np.linspace(0.0, 2.0 * np.pi, EVAL_PHI)
    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    points = np.stack(
        (np.sin(tt) * np.cos(pp), np.sin(tt) * np.sin(pp), np.cos(tt)), axis=-1
    ).reshape((-1, 3))
    _, fidelity, infidelity = model_values(
        params,
        jnp.asarray(points),
        jnp.full((points.shape[0],), slice_tau),
        PairConfig.total_rotation,
    )
    fidelity = np.asarray(fidelity).reshape((EVAL_THETA, EVAL_PHI))
    infidelity = np.asarray(infidelity).reshape((EVAL_THETA, EVAL_PHI))
    np.savez_compressed(
        OUTPUT / "pair_slice.npz",
        theta=theta,
        phi=phi,
        tau=slice_tau,
        infidelity=infidelity.astype(np.float32),
        fidelity=fidelity.astype(np.float32),
        node_theta=at_slice["theta"],
        node_phi=at_slice["phi"],
        node_charge=at_slice["charge"],
    )
    write_json(
        OUTPUT / "summary.json",
        {
            "created_utc": utc_now(),
            "slice_tau": float(slice_tau),
            "node_count": int(len(at_slice)),
            "node_charges": [float(charge) for charge in at_slice["charge"]],
            "total_charge": float(np.sum(at_slice["charge"])),
            "grid": {"theta": EVAL_THETA, "phi": EVAL_PHI},
            "grid_max_infidelity": float(np.max(infidelity)),
        },
    )
    logger.info("Banked five-node slice at tau %.6f", slice_tau)


if __name__ == "__main__":
    main()
