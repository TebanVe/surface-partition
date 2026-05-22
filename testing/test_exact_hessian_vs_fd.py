#!/usr/bin/env python3
"""Verify the analytical Lagrangian Hessian against central finite differences.

Compares the Hessian assembled by ``IPOPTProblemAdapter._hessian_impl`` against
a finite-difference reference: central differences of the Lagrangian gradient
``∇L = obj_factor·∇f + λᵀ·∇c``.  This is the single test that can unmask a
wrong sign or chain-rule factor anywhere in the analytical Hessian path.

The FD reference is a Richardson extrapolation of central differences at
steps eps and eps/2 — O(eps^4) accurate (~1e-9), versus the ~1e-4 floor of a
plain central difference on stiff (short-segment) perimeter entries.  With the
fully analytical Steiner path the analytical Hessian agrees with it to ~1e-9;
defaults are --atol 1e-7 --rtol 1e-6.

Diagnostic: on FAIL, re-run with --lagrange-mode zero to isolate the objective
(perimeter) Hessian, then --obj-factor 0 --lagrange-mode ones to isolate the
constraint (area) Hessian.

Usage:
    python testing/test_exact_hessian_vs_fd.py --solution <path.h5>
    python testing/test_exact_hessian_vs_fd.py --solution <path.h5> \\
        --lagrange-mode zero --atol 1e-6 --rtol 1e-5

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
    """∇L = obj_factor·∇f + λᵀ·J, with J the area-constraint Jacobian."""
    g = obj_factor * optimizer.objective_gradient(x)
    J = optimizer.constraint_area_jacobian(x)          # (n_cells-1, n_active)
    return g + lagrange @ J


def hessian_from_adapter(optimizer, x, lagrange, obj_factor):
    """Call the adapter's Hessian and pack into a dense symmetric matrix."""
    adapter = IPOPTProblemAdapter(optimizer, exact_hessian=True)
    vals = adapter._hessian_impl(x, lagrange, obj_factor)   # (hess_nnz,)
    pa = optimizer._arrays
    n = len(x)
    H = np.zeros((n, n))
    H[pa.hess_row, pa.hess_col] = vals
    # Mirror the strict lower triangle into the upper triangle.
    strict_lower = pa.hess_row > pa.hess_col
    H[pa.hess_col[strict_lower], pa.hess_row[strict_lower]] = vals[strict_lower]
    return H


def _hessian_fd_at(optimizer, x, lagrange, obj_factor, eps):
    """Central-FD Hessian of the Lagrangian at one step size, symmetrized."""
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        H[i, :] = (lagrangian_grad(optimizer, xp, lagrange, obj_factor)
                   - lagrangian_grad(optimizer, xm, lagrange, obj_factor)) / (2.0 * eps)
    return 0.5 * (H + H.T)


def hessian_fd(optimizer, x, lagrange, obj_factor, eps=1e-5):
    """Richardson-extrapolated central-FD Hessian — O(eps^4) accurate.

    A plain central difference is O(eps^2) and floors at ~1e-4 on stiff
    short-segment perimeter entries.  Combining steps eps and eps/2 cancels
    the O(eps^2) term, giving a ~1e-9 reference.
    """
    h_full = _hessian_fd_at(optimizer, x, lagrange, obj_factor, eps)
    h_half = _hessian_fd_at(optimizer, x, lagrange, obj_factor, eps / 2.0)
    return (4.0 * h_half - h_full) / 3.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--solution', required=True)
    ap.add_argument('--rtol', type=float, default=1e-6,
                    help='Relative tolerance (default 1e-6).')
    ap.add_argument('--atol', type=float, default=1e-7,
                    help='Absolute floor for the comparison (default 1e-7).')
    ap.add_argument('--obj-factor', type=float, default=1.0)
    ap.add_argument('--lagrange-mode', choices=['zero', 'ones', 'random'],
                    default='random')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fd-eps', type=float, default=1e-5)
    ap.add_argument('--max-n', type=int, default=2500,
                    help='Skip if n_active_vp > this — the dense FD Hessian is '
                         'O(n^2) memory / O(n) gradient calls. Above this, use '
                         'test_exact_hessian_matvec.py instead.')
    args = ap.parse_args()

    opt = build_optimizer(args.solution)
    pa = opt._arrays
    n = pa.n_active_vp
    m = pa.n_cells - 1

    if pa.hess_row is None:
        print("SKIP: Hessian sparsity not compiled (pa.hess_row is None).")
        return 0
    if n > args.max_n:
        print(f"SKIP: n_active_vp={n} > --max-n={args.max_n}. "
              f"Run testing/test_exact_hessian_matvec.py instead.")
        return 0

    rng = np.random.default_rng(args.seed)
    x = pa.vp_lambda.copy()
    if args.lagrange_mode == 'zero':
        lam = np.zeros(m)
    elif args.lagrange_mode == 'ones':
        lam = np.ones(m)
    else:
        lam = rng.standard_normal(m)

    H_ana = hessian_from_adapter(opt, x, lam, args.obj_factor)
    H_fd = hessian_fd(opt, x, lam, args.obj_factor, eps=args.fd_eps)

    abs_err = float(np.max(np.abs(H_ana - H_fd)))
    ref = max(float(np.max(np.abs(H_fd))), 1e-30)
    rel_err = abs_err / ref

    print("Analytical Lagrangian Hessian vs. central FD")
    print(f"  solution      = {args.solution}")
    print(f"  lagrange-mode = {args.lagrange_mode}   obj-factor = {args.obj_factor}")
    print(f"  n_active_vp   = {n}")
    print(f"  n_triple_pts  = {pa.n_triple_points}")
    print(f"  hess_nnz      = {len(pa.hess_row)}")
    print(f"  ||H_ana||_inf = {np.max(np.abs(H_ana)):.3e}")
    print(f"  ||H_fd||_inf  = {np.max(np.abs(H_fd)):.3e}")
    print(f"  max |Δ|       = {abs_err:.3e}")
    print(f"  rel err       = {rel_err:.3e}")

    i, j = np.unravel_index(int(np.argmax(np.abs(H_ana - H_fd))), H_ana.shape)
    print(f"  worst entry   = ({i},{j})  ana={H_ana[i, j]:.3e}  fd={H_fd[i, j]:.3e}")

    ok = abs_err < max(args.atol, args.rtol * ref)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
