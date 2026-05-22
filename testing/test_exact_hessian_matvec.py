#!/usr/bin/env python3
"""Hessian-vector-product check for large meshes.

For problems where ``n_active_vp`` is large, building the dense (n,n) FD Hessian
is slow and memory-hungry.  This test instead verifies, for several random unit
vectors ``v``:

    H_ana @ v  ≈  (∇L(x + ε·v) − ∇L(x − ε·v)) / (2ε)

This is the large-mesh counterpart of test_exact_hessian_vs_fd.py — same
Lagrangian, same tolerance semantics, O(n) memory instead of O(n²).

Usage:
    python testing/test_exact_hessian_matvec.py --solution <path.h5>
    python testing/test_exact_hessian_matvec.py --solution <path.h5> --n-vectors 8

Exit code: 0 on PASS (or SKIP), 1 on FAIL.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from _hessian_test_utils import build_optimizer  # noqa: E402
from src.optimization.perimeter_optimizer import IPOPTProblemAdapter  # noqa: E402


def lagrangian_grad(optimizer, x, lagrange, obj_factor):
    g = obj_factor * optimizer.objective_gradient(x)
    J = optimizer.constraint_area_jacobian(x)
    return g + lagrange @ J


def _hessian_matvec_fd_at(optimizer, x, lagrange, obj_factor, v, eps):
    """Central-FD directional second derivative H·v at one step size."""
    return (lagrangian_grad(optimizer, x + eps * v, lagrange, obj_factor)
            - lagrangian_grad(optimizer, x - eps * v, lagrange, obj_factor)) / (2.0 * eps)


def hessian_matvec_fd(optimizer, x, lagrange, obj_factor, v, eps=1e-5):
    """Richardson-extrapolated central-FD directional 2nd derivative — O(eps^4)."""
    hv_full = _hessian_matvec_fd_at(optimizer, x, lagrange, obj_factor, v, eps)
    hv_half = _hessian_matvec_fd_at(optimizer, x, lagrange, obj_factor, v, eps / 2.0)
    return (4.0 * hv_half - hv_full) / 3.0


def hessian_matvec_sparse(pa, vals, v):
    """H·v built directly from the lower-triangle sparse Hessian values."""
    out = np.zeros_like(v)
    np.add.at(out, pa.hess_row, vals * v[pa.hess_col])
    strict = pa.hess_row > pa.hess_col
    np.add.at(out, pa.hess_col[strict], vals[strict] * v[pa.hess_row[strict]])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--solution', required=True)
    ap.add_argument('--rtol', type=float, default=1e-6)
    ap.add_argument('--atol', type=float, default=1e-7)
    ap.add_argument('--obj-factor', type=float, default=1.0)
    ap.add_argument('--lagrange-mode', choices=['zero', 'ones', 'random'],
                    default='random')
    ap.add_argument('--n-vectors', type=int, default=5,
                    help='Number of random probe vectors (default 5).')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fd-eps', type=float, default=1e-5)
    args = ap.parse_args()

    opt = build_optimizer(args.solution)
    pa = opt._arrays
    n = pa.n_active_vp
    m = pa.n_cells - 1

    if pa.hess_row is None:
        print("SKIP: Hessian sparsity not compiled (pa.hess_row is None).")
        return 0

    rng = np.random.default_rng(args.seed)
    x = pa.vp_lambda.copy()
    if args.lagrange_mode == 'zero':
        lam = np.zeros(m)
    elif args.lagrange_mode == 'ones':
        lam = np.ones(m)
    else:
        lam = rng.standard_normal(m)

    adapter = IPOPTProblemAdapter(opt, exact_hessian=True)
    hess_vals = adapter._hessian_impl(x, lam, args.obj_factor)   # (hess_nnz,)

    print("Analytical Hessian-vector product vs. central FD")
    print(f"  solution      = {args.solution}")
    print(f"  lagrange-mode = {args.lagrange_mode}   obj-factor = {args.obj_factor}")
    print(f"  n_active_vp   = {n}")
    print(f"  n_triple_pts  = {pa.n_triple_points}")
    print(f"  probe vectors = {args.n_vectors}")

    worst = 0.0
    ok = True
    for k in range(args.n_vectors):
        v = rng.standard_normal(n)
        v /= np.linalg.norm(v)
        hv_ana = hessian_matvec_sparse(pa, hess_vals, v)
        hv_fd = hessian_matvec_fd(opt, x, lam, args.obj_factor, v, eps=args.fd_eps)
        abs_err = float(np.max(np.abs(hv_ana - hv_fd)))
        ref = max(float(np.max(np.abs(hv_fd))), 1e-30)
        passed = abs_err < max(args.atol, args.rtol * ref)
        worst = max(worst, abs_err)
        ok = ok and passed
        print(f"  v[{k}]  max|Δ|={abs_err:.3e}  rel={abs_err / ref:.3e}  "
              f"{'ok' if passed else 'FAIL'}")

    print(f"  worst max|Δ| over all probes = {worst:.3e}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
