#!/usr/bin/env python3
"""Calibrate ``soft_area_mu`` for a given Phase 1 config (A2).

The soft continuous equal-area penalty is

    P = (mu/2) * sum_k ((v.u_k - Abar)/Abar)^2,
    dP/du_ik = (mu/Abar^2) * v_i * (v.u_k - Abar),

so at a *relative* area deviation rho the restoring force on the heaviest
vertex is ``mu * v_max * rho / Abar`` per unit mu. The calibration rule is:

    choose mu so that this force reaches PARITY with the base energy's
    ||g||_inf at rho = 5% -- the discrete-area gate threshold
    (AREA_IMBALANCE_REL_THRESHOLD) -- evaluated at the level-0 seeded initial
    condition of the config in question.

Parity at the gate, not below it, because the penalty is now the ONLY thing
resisting a cell's mass collapsing; below the gate we want it gentle (it is
linear in rho, so 1% deviation costs a fifth of the force), above it, dominant.

mu is NOT N-invariant -- it scales roughly with vertices-per-cell (mu_parity
~ ||g||_inf * Abar / v_max, and Abar/v_max ~ V/N). The RULE is what transfers
between configs, so re-run this per config rather than copying the number.
Stiffness check: the penalty adds ``mu * ||v||^2 / Abar^2`` to the Hessian's
top eigenvalue; this script reports it against the config's own accepted step
sizes so you can confirm the penalty is not what limits the step.

Usage:
    python testing/calibrate_soft_area_mu.py --config parameters/<cfg>.yaml
    python testing/calibrate_soft_area_mu.py --config <cfg> --rho 0.05
"""

import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.surfaces.torus import TorusMeshProvider  # noqa: E402
from src.optimization.pgd_optimizer import ProjectedGradientOptimizer  # noqa: E402
from src.optimization.initialization import (  # noqa: E402
    create_seeded_initial_condition,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--rho",
        type=float,
        default=0.05,
        help="relative area deviation at which to demand force parity "
        "(default 0.05 = the discrete-area gate threshold)",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    rel = cfg.get("relaxation", cfg)
    surf = cfg["surface"]["torus"]
    N = int(rel["n_partitions"])
    lam = float(rel["lambda_penalty"])
    seed = int(rel["seed"])

    prov = TorusMeshProvider(
        n_theta=int(surf["n_theta"]),
        n_phi=int(surf["n_phi"]),
        R=float(surf["R"]),
        r=float(surf["r"]),
    )
    mesh = prov.build()
    mesh.compute_matrices()
    V = len(mesh.v)
    A = float(np.sum(mesh.v))
    Abar = A / N
    eps = float(np.sqrt(mesh.get_mesh_statistics()["mean_triangle_area"]))

    x0 = create_seeded_initial_condition(mesh, N, mesh.v, seed=seed)
    opt = ProjectedGradientOptimizer(
        K=mesh.K,
        M=mesh.M,
        v=mesh.v,
        n_partitions=N,
        epsilon=eps,
        total_area=A,
        lambda_penalty=lam,
    )
    g = opt.compute_gradient(x0)
    g_inf = float(np.abs(g).max())
    v_max = float(mesh.v.max())

    per_unit_mu = v_max * args.rho / Abar
    mu = g_inf / per_unit_mu

    v_sq = float(np.sum(mesh.v**2))
    L_pen = mu * v_sq / Abar**2

    print(f"config              : {args.config}")
    print(f"level 0             : V={V}, N={N}, verts/cell={V / N:.1f}")
    print(f"epsilon             : {eps:.4e}   lambda={lam}   seed={seed}")
    print(f"Abar                : {Abar:.6e}   v_max={v_max:.4e}")
    print(f"base ||g||_inf @ x0 : {g_inf:.6e}")
    print(f"rho (parity point)  : {args.rho:.1%}")
    print()
    print(f"  soft_area_mu: {mu:.1f}      <-- put this in the config")
    print()
    print(f"penalty Hessian top eigenvalue ~ {L_pen:.1f}")
    print(f"  => penalty alone permits steps up to ~{2.0 / L_pen:.3e}")
    print(
        "  compare with the accepted STEP column of this config's level-0 "
        "summary trace; if the accepted steps are already smaller, the "
        "penalty is not the step-limiting term."
    )


if __name__ == "__main__":
    sys.exit(main())
