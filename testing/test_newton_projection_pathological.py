#!/usr/bin/env python3
"""Adversarial / pathological-input tests for orthogonal_projection_newton.

Exercises exactly the regimes where a naive dual solver fails (docs/math/08
Remark on the zero-J-row degeneracy; docs/plans/... section 8.6): an empty cell,
fully one-hot (crisp) rows, an isolated crisp cell (non-empty active column but a
zero J-row --- the case a plain "empty column" guard misses), deficient input
rows (row-sum < 1), and breakpoint ties. In every case the solver must NOT stall
or explode: the output must be finite, feasible, and equal to the true (Dykstra)
projection.

Self-contained (synthetic inputs, torus lumped mass built directly).

Usage:
    python testing/test_newton_projection_pathological.py
    python testing/test_newton_projection_pathological.py --seed 3

Exit code: 0 on PASS, 1 on FAIL.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.optimization.projection import orthogonal_projection_newton  # noqa: E402
from src.surfaces.torus import TorusMeshProvider  # noqa: E402


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


def _check(name, Y, v, d):
    """Run newton on a pathological Y; return (pass, message)."""
    n = Y.shape[1]
    try:
        U, beta = orthogonal_projection_newton(Y, np.ones(n), d, v, tol=1e-10,
                                               return_beta=True)
    except Exception as e:  # noqa: BLE001
        return False, f"raised {type(e).__name__}: {e}"
    if not np.all(np.isfinite(U)) or not np.all(np.isfinite(beta)):
        return False, f"non-finite output (max|beta|={np.max(np.abs(beta)):.2e})"
    Ut = _dykstra(Y, v, d)
    gap = float(np.max(np.abs(U - Ut)))
    row = float(np.max(np.abs(U.sum(1) - 1.0)))
    area = float(np.max(np.abs(v @ U - d)))
    nn = max(0.0, -float(U.min()))
    good = gap <= 1e-8 and area <= 1e-8 and row <= 1e-10 and nn <= 1e-12
    return good, (f"gap={gap:.2e} feas(row,area,nn)=({row:.1e},{area:.1e},{nn:.1e}) "
                  f"|beta|max={np.max(np.abs(beta)):.1e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    v = np.asarray(TorusMeshProvider(16, 8, 1.0, 0.6).build().v, float)
    V = len(v)
    print(f"Newton projection pathological-input test  (torus V={V})")
    ok = True

    # 1. Empty cell: cell 0 absent from the candidate; the area constraint must
    #    still force mass into it (empty active column -> zero J-row).
    n = 6
    d = (v.sum() / n) * np.ones(n)
    Y = 0.2 * rng.standard_normal((V, n))
    Y[np.arange(V), rng.integers(1, n, V)] += 1.0
    Y[:, 0] = -5.0
    p, msg = _check("empty cell", Y, v, d)
    ok &= p
    print(f"  [1] empty cell            : {msg} [{'PASS' if p else 'FAIL'}]")

    # 2. Fully one-hot rows: every vertex strictly one cell (J = 0 everywhere).
    n = 5
    d = (v.sum() / n) * np.ones(n)
    Y = np.zeros((V, n))
    Y[np.arange(V), rng.integers(0, n, V)] = 1.0
    p, msg = _check("fully one-hot", Y, v, d)
    ok &= p
    print(f"  [2] fully one-hot rows    : {msg} [{'PASS' if p else 'FAIL'}]")

    # 3. Isolated crisp cell: cell n-1 supported ONLY by strict one-hot (m_i=1)
    #    vertices -> non-empty active column but a zero J-row (the case the
    #    "empty column" guard would miss; must use the J_kk~0 safeguard).
    n = 4
    d = (v.sum() / n) * np.ones(n)
    idx_iso = np.arange(0, V, max(1, V // 5))
    rest = np.setdiff1d(np.arange(V), idx_iso)
    Y = -10.0 * np.ones((V, n))
    Y[idx_iso, n - 1] = 10.0
    Y[np.ix_(rest, np.arange(n - 1))] = 0.2 * rng.standard_normal((len(rest), n - 1))
    Y[rest, rng.integers(0, n - 1, len(rest))] += 1.0
    p, msg = _check("isolated crisp", Y, v, d)
    ok &= p
    print(f"  [3] isolated crisp cell   : {msg} [{'PASS' if p else 'FAIL'}]")

    # 4. Deficient input rows: rows sum to << 1 (the projection must still
    #    return unit-sum rows).
    n = 6
    d = (v.sum() / n) * np.ones(n)
    Y = 0.05 * rng.random((V, n))
    p, msg = _check("deficient rows", Y, v, d)
    ok &= p
    print(f"  [4] deficient rows (sum<1): {msg} [{'PASS' if p else 'FAIL'}]")

    # 5. Breakpoint ties: identical columns create tied simplex breakpoints.
    n = 6
    d = (v.sum() / n) * np.ones(n)
    Y = 0.2 * rng.standard_normal((V, n))
    Y[:, 1] = Y[:, 0]
    Y[:, 3] = Y[:, 2]
    p, msg = _check("breakpoint ties", Y, v, d)
    ok &= p
    print(f"  [5] breakpoint ties       : {msg} [{'PASS' if p else 'FAIL'}]")

    print("\n" + ("RESULT: PASS" if ok else "RESULT: FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
