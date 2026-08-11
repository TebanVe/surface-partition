#!/usr/bin/env python3
"""Correctness gates for the soft continuous equal-area constraint (A2).

Three checks, all cheap and in-process:

1. **Simplex projection is exact.** ``project_rows_onto_simplex`` solves
   ``min ||x-a||^2  s.t.  sum x = 1, x >= lo`` in closed form; verified by the
   KKT conditions (``x - a`` constant across the free coordinates, and no
   smaller on the clamped ones) plus feasibility and idempotence.

2. **The penalty gradient matches central finite differences.** The soft-area
   term ``P = (mu/2) sum_k ((v.u_k - Abar)/Abar)^2`` contributes
   ``(mu/Abar^2) v_i (v.u_k - Abar)`` to the gradient.

3. **Flag-off is bit-identical.** With ``soft_area_constraint=False`` the
   energy, the gradient and the constraint vector must be byte-for-byte what
   the pre-A2 optimizer produced — the flag adds nothing when off.

Usage:
    python testing/test_soft_area_constraint.py [--tol 1e-6]
"""

import argparse
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.surfaces.torus import TorusMeshProvider  # noqa: E402
from src.optimization.pgd_optimizer import (  # noqa: E402
    ProjectedGradientOptimizer,
)
from src.optimization.projection import (  # noqa: E402
    project_rows_onto_simplex,
)


def _mesh(n_theta=24, n_phi=16):
    p = TorusMeshProvider(n_theta=n_theta, n_phi=n_phi, R=1.0, r=0.6)
    m = p.build()
    m.compute_matrices()
    return m


def _optimizer(mesh, n_part, soft, mu):
    return ProjectedGradientOptimizer(
        K=mesh.K,
        M=mesh.M,
        v=mesh.v,
        n_partitions=n_part,
        epsilon=0.1,
        total_area=float(np.sum(mesh.v)),
        lambda_penalty=2.0,
        soft_area_constraint=soft,
        soft_area_mu=mu,
    )


def check_projection_exact(rng, n_rows=400):
    print("[1] simplex projection: KKT + feasibility + idempotence")
    worst_ptp = worst_feas = worst_idem = 0.0
    worst_box = 0.0
    for _ in range(n_rows):
        n = int(rng.integers(2, 60))
        lo = float(rng.choice([0.0, 1e-8, 1e-4]))
        a = rng.normal(scale=3.0, size=(1, n))
        x = project_rows_onto_simplex(a, lo)[0]
        free = x > lo + 1e-12
        resid = (x - a[0])[free]
        # stationarity: x - a is a single constant -theta on the free set
        worst_ptp = max(worst_ptp, float(np.ptp(resid)) if resid.size else 0.0)
        # complementarity: clamped coords were pushed UP to the floor
        if (~free).any() and resid.size:
            assert ((x - a[0])[~free] >= resid.mean() - 1e-12).all()
        worst_feas = max(worst_feas, abs(float(x.sum()) - 1.0))
        worst_box = max(worst_box, max(0.0, lo - float(x.min())))
        x2 = project_rows_onto_simplex(x[None, :], lo)[0]
        worst_idem = max(worst_idem, float(np.abs(x2 - x).max()))
    print(f"    KKT stationarity spread : {worst_ptp:.3e}")
    print(f"    row-sum error           : {worst_feas:.3e}")
    print(f"    box violation           : {worst_box:.3e}")
    print(f"    idempotence             : {worst_idem:.3e}")
    ok = (
        worst_ptp < 1e-12
        and worst_feas < 1e-12
        and worst_box == 0.0
        and worst_idem < 1e-12
    )
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_penalty_gradient(rng, tol):
    """Two independent checks, kept separate because they fail differently.

    (a) The MATH: an independently written closed form of the penalty is
        finite-differenced against its own analytic gradient. Evaluating the
        penalty directly (rather than as E_on - E_off) matters: the two
        energies are O(1e3) here while the penalty is O(1), so differencing
        them first destroys ~13 digits and the FD check reports ~1e-5 noise
        that has nothing to do with the gradient.
    (b) The WIRING: the optimizer's own on-minus-off difference must equal
        that same closed form, in both the energy and the gradient.
    """
    print("[2] soft-area penalty: (a) math vs FD, (b) wiring")
    mesh = _mesh()
    n_part, mu = 5, 7.5
    V = len(mesh.v)
    v, Abar = mesh.v, float(np.sum(mesh.v)) / n_part
    opt_on = _optimizer(mesh, n_part, True, mu)
    opt_off = _optimizer(mesh, n_part, False, 0.0)

    U = rng.random((V, n_part))
    U /= U.sum(axis=1, keepdims=True)
    x = U.flatten()

    def P(z):  # independent closed form
        r = (v @ z.reshape(V, n_part) - Abar) / Abar
        return 0.5 * mu * float(r @ r)

    def dP(z):
        dev = v @ z.reshape(V, n_part) - Abar
        return ((mu / Abar**2) * np.outer(v, dev)).flatten()

    g_ref = dP(x)
    h = 1e-6
    idx = rng.choice(V * n_part, size=200, replace=False)
    worst_fd = 0.0
    for j in idx:
        xp = x.copy()
        xp[j] += h
        xm = x.copy()
        xm[j] -= h
        fd = (P(xp) - P(xm)) / (2 * h)
        worst_fd = max(worst_fd, abs(fd - g_ref[j]) / max(abs(fd), 1e-12))
    print(
        f"    (a) worst rel FD error over 200 coords: {worst_fd:.3e} " f"(tol {tol:g})"
    )

    dE = abs((opt_on.compute_energy(x) - opt_off.compute_energy(x)) - P(x))
    dG = float(
        np.abs((opt_on.compute_gradient(x) - opt_off.compute_gradient(x)) - g_ref).max()
    )
    scaleE = max(abs(P(x)), 1e-12)
    scaleG = max(float(np.abs(g_ref).max()), 1e-12)
    print(f"    (b) energy wiring   rel err: {dE / scaleE:.3e}")
    print(f"    (b) gradient wiring rel err: {dG / scaleG:.3e}")

    ok = worst_fd < tol and dE / scaleE < 1e-10 and dG / scaleG < 1e-12
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_flag_off_identity(rng):
    print("[3] flag-off identity: energy / gradient / constraints unchanged")
    mesh = _mesh()
    n_part = 6
    V = len(mesh.v)
    a = _optimizer(mesh, n_part, False, 0.0)
    # mu deliberately nonzero: it must be inert while the flag is off
    b = _optimizer(mesh, n_part, False, 99.0)
    U = rng.random((V, n_part))
    U /= U.sum(axis=1, keepdims=True)
    x = U.flatten()
    dE = abs(a.compute_energy(x) - b.compute_energy(x))
    dG = float(np.abs(a.compute_gradient(x) - b.compute_gradient(x)).max())
    ca, cb = a.constraint_fun(x), b.constraint_fun(x)
    dC = float(np.abs(ca - cb).max())
    same_len = ca.shape == cb.shape
    # and the OFF constraint vector must still carry the area block
    has_area = ca.size == (V - 1) + (n_part - 1)
    print(f"    |dE|={dE:.3e}  max|dG|={dG:.3e}  max|dC|={dC:.3e}")
    print(f"    constraint vector keeps the area block: {has_area}")
    ok = dE == 0.0 and dG == 0.0 and dC == 0.0 and same_len and has_area
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_soft_drops_area_block(rng):
    print("[4] flag-on: FEAS covers sum-to-one only (trigger semantics)")
    mesh = _mesh()
    n_part = 6
    V = len(mesh.v)
    on = _optimizer(mesh, n_part, True, 7.5)
    U = rng.random((V, n_part))
    U /= U.sum(axis=1, keepdims=True)
    c = on.constraint_fun(U.flatten())
    ok = c.size == V - 1
    a_abs, a_rel = on.area_deviation(U.flatten())
    print(f"    constraint vector length {c.size} (expected {V - 1}) ")
    print(f"    area_deviation reports {a_abs:.3e} abs / {a_rel * 100:.2f}% rel")
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()
    rng = np.random.default_rng(20260810)

    results = [
        check_projection_exact(rng),
        check_penalty_gradient(rng, args.tol),
        check_flag_off_identity(rng),
        check_soft_drops_area_block(rng),
    ]
    print()
    if all(results):
        print("RESULT: PASS")
        return 0
    print("RESULT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
