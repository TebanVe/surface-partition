#!/usr/bin/env python3
"""Validate orthogonal_projection_newton against the TRUE Euclidean projection.

Unlike the incumbent iterative method (which is not the projection --- see
docs/experiments/03-dual-projection-verification/), the newton path computes the
exact projection onto {row-sum=1, area=d, box}. This test verifies that against
an INDEPENDENT reference (Dykstra) plus a solver-free KKT certificate, and
checks the analytical dual quantities by finite differences. It is self-contained
(synthetic inputs, torus lumped mass built directly --- no --solution needed).

Checks (self-contained, deterministic):
  1. Equivalence to the true projection (Dykstra), interior + binding + crisp,
     several V/N, realistic torus v. Gate: max|newton - true| <= 1e-8.
  2. Feasibility (row/area/nonneg) <= 1e-10; idempotency (feasible input) ~ eps.
  3. KKT stationarity via the solver's OWN duals (alpha,beta) <= 1e-9.
  4. Finite-difference of grad q = -R and of the Jacobian J (eq:jacobian);
     J symmetric PSD with structural kernel span{1} (J.1 = 0).
  5. Characterize the difference from the iterative method (informational) and
     assert newton is at least as close to Y (lower 1/2||U-Y||^2).

Usage:
    python testing/test_newton_projection_equivalence.py
    python testing/test_newton_projection_equivalence.py --seed 7

Exit code: 0 on PASS, 1 on FAIL.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.optimization.projection import (  # noqa: E402
    orthogonal_projection_newton,
    orthogonal_projection_iterative,
    _project_simplex_rows,
)
from src.surfaces.torus import TorusMeshProvider  # noqa: E402


# --- independent reference: Dykstra onto {row=1} n {area=d} n {box} ----------
def _dykstra(Y, v, d, n_iter=500000, tol=1e-15):
    def P_rows(U):
        return U - (U.sum(1, keepdims=True) - 1.0) / U.shape[1]

    def P_area(U):
        return U - np.outer(v, (v @ U - d) / float(v @ v))

    projs = [P_rows, P_area, lambda U: np.clip(U, 0.0, 1.0)]
    x = Y.copy()
    incr = [np.zeros_like(Y) for _ in projs]
    for _ in range(n_iter):
        xo = x.copy()
        for i, P in enumerate(projs):
            t = x + incr[i]
            y = P(t)
            incr[i] = t - y
            x = y
        if np.max(np.abs(x - xo)) < tol:
            break
    return x


def _feas(U, v, d):
    return (float(np.max(np.abs(U.sum(1) - 1.0))),
            float(np.max(np.abs(v @ U - d))),
            max(0.0, -float(U.min())))


def _torus_v(nt, npx):
    return np.asarray(TorusMeshProvider(nt, npx, 1.0, 0.6).build().v, float)


def _make_Y(V, n, regime, rng):
    if regime == "interior":
        return 1.0 / n + 0.05 * rng.standard_normal((V, n))
    if regime == "binding":
        Y = 0.2 * rng.standard_normal((V, n))
        Y[np.arange(V), rng.integers(0, n, V)] += 1.0
        return Y
    if regime == "crisp":
        Y = 0.02 * rng.standard_normal((V, n))
        Y[np.arange(V), rng.integers(0, n, V)] += 1.0
        return np.clip(Y, 1e-8, 1 - 1e-8)
    raise ValueError(regime)


# --- q, R and J reproduced for finite-difference checks (same inner solve) ---
def _qR(beta, Y, v, d):
    U, _ = _project_simplex_rows(Y + np.outer(v, beta))
    R = v @ U - d
    q = 0.5 * float(np.sum((U - Y) ** 2)) - float(beta @ R)
    return q, R


def _jacobian(beta, Y, v, d):
    U, _ = _project_simplex_rows(Y + np.outer(v, beta))
    mask = (U > 1e-12).astype(float)
    m = mask.sum(1)
    v2 = v ** 2
    return np.diag(v2 @ mask) - mask.T @ ((v2 / m)[:, None] * mask)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    ok = True

    v = _torus_v(16, 8)
    V = len(v)
    print(f"Newton projection equivalence test  (torus V={V}, sum(v)={v.sum():.4f})")

    # --- 1/2. equivalence vs true projection + feasibility ------------------
    print("\n[1/5] equivalence vs the true (Dykstra) projection + feasibility")
    worst_gap = 0.0
    for n, reg in [(4, "interior"), (4, "binding"), (10, "binding"), (10, "crisp"),
                   (25, "crisp")]:
        d = (v.sum() / n) * np.ones(n)
        Y = _make_Y(V, n, reg, rng)
        U = orthogonal_projection_newton(Y, np.ones(n), d, v, tol=1e-10)
        Ut = _dykstra(Y, v, d)
        gap = float(np.max(np.abs(U - Ut)))
        row, area, nn = _feas(U, v, d)
        c_ok = gap <= 1e-8 and area <= 1e-10 and row <= 1e-12 and nn <= 1e-12
        ok &= c_ok
        worst_gap = max(worst_gap, gap)
        print(f"   {reg:9s} n={n:2d}: gap={gap:.2e} feas(row,area,nn)="
              f"({row:.1e},{area:.1e},{nn:.1e}) [{'PASS' if c_ok else 'FAIL'}]")
    print(f"   worst gap = {worst_gap:.2e} (gate 1e-8)")

    # --- 3. idempotency ------------------------------------------------------
    print("\n[2/5] idempotency (project an already-feasible point)")
    d = (v.sum() / 8) * np.ones(8)
    Uf = _dykstra(_make_Y(V, 8, "crisp", rng), v, d)
    move = float(np.max(np.abs(orthogonal_projection_newton(Uf, np.ones(8), d, v, tol=1e-10) - Uf)))
    idem_ok = move <= 1e-10
    ok &= idem_ok
    print(f"   max|P(U)-U| = {move:.2e} (gate 1e-10) [{'PASS' if idem_ok else 'FAIL'}]")

    # --- 3b. KKT via the solver's own duals ---------------------------------
    print("\n[3/5] KKT stationarity via the solver's own duals")
    kkt_worst = 0.0
    for n, reg in [(4, "binding"), (10, "crisp")]:
        d = (v.sum() / n) * np.ones(n)
        Y = _make_Y(V, n, reg, rng)
        U, beta = orthogonal_projection_newton(Y, np.ones(n), d, v, tol=1e-10, return_beta=True)
        _, tau = _project_simplex_rows(Y + np.outer(v, beta))
        alpha = -tau
        inactive = (U > 1e-8) & (U < 1 - 1e-8)
        stat = float(np.max(np.abs((U - Y - alpha[:, None] - np.outer(v, beta))[inactive]))) \
            if inactive.any() else 0.0
        kkt_worst = max(kkt_worst, stat)
        s_ok = stat <= 1e-9
        ok &= s_ok
        print(f"   {reg:8s} n={n:2d}: stationarity residual = {stat:.2e} "
              f"[{'PASS' if s_ok else 'FAIL'}]")

    # --- 4. FD checks: grad q = -R, and J -----------------------------------
    print("\n[4/5] finite-difference checks of grad q = -R and J = dR/dbeta")
    n = 5
    d = (v.sum() / n) * np.ones(n)
    Y = _make_Y(V, n, "binding", rng)
    beta = 0.1 * rng.standard_normal(n)
    _, R = _qR(beta, Y, v, d)
    h = 1e-6
    gfd = np.array([(_qR(beta + h * e, Y, v, d)[0] - _qR(beta - h * e, Y, v, d)[0]) / (2 * h)
                    for e in np.eye(n)])
    g_err = float(np.max(np.abs(gfd - (-R))))
    J = _jacobian(beta, Y, v, d)
    Jfd = np.array([(_qR(beta + h * e, Y, v, d)[1] - _qR(beta - h * e, Y, v, d)[1]) / (2 * h)
                    for e in np.eye(n)]).T
    j_err = float(np.max(np.abs(J - Jfd)) / np.max(np.abs(J)))
    eig = float(np.min(np.linalg.eigvalsh(J)))
    j1 = float(np.max(np.abs(J @ np.ones(n))))
    fd_ok = g_err <= 1e-6 and j_err <= 1e-6 and eig >= -1e-10 and j1 <= 1e-9
    ok &= fd_ok
    print(f"   |FD grad q -(-R)|={g_err:.1e}  |J-FD|/|J|={j_err:.1e}  "
          f"min eig(J)={eig:.1e}  |J.1|={j1:.1e} [{'PASS' if fd_ok else 'FAIL'}]")

    # --- 5. difference from the iterative method (informational + soft gate) -
    print("\n[5/5] difference from the iterative method (newton must be >= as close to Y)")
    obj_ok = True
    for n, reg in [(4, "binding"), (10, "crisp")]:
        d = (v.sum() / n) * np.ones(n)
        Y = _make_Y(V, n, reg, rng)
        U_nt = orthogonal_projection_newton(Y, np.ones(n), d, v, tol=1e-10)
        try:
            U_it = orthogonal_projection_iterative(Y.copy(), np.ones(n), d, v,
                                                   max_iter=2000, tol=1e-8)
        except RuntimeError:
            U_it = orthogonal_projection_iterative(Y.copy(), np.ones(n), d, v,
                                                   max_iter=2000, tol=1e-6)
        diff = float(np.max(np.abs(U_nt - U_it)))
        o_nt = 0.5 * float(np.sum((U_nt - Y) ** 2))
        o_it = 0.5 * float(np.sum((U_it - Y) ** 2))
        better = o_nt <= o_it + 1e-9 * (1 + abs(o_it))
        obj_ok &= better
        print(f"   {reg:8s} n={n:2d}: max|newton-iterative|={diff:.2e}  "
              f"obj newton={o_nt:.4f} <= iterative={o_it:.4f} [{'PASS' if better else 'FAIL'}]")
    ok &= obj_ok

    print("\n" + ("RESULT: PASS" if ok else "RESULT: FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
