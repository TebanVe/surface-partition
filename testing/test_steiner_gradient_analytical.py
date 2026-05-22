#!/usr/bin/env python3
"""Verify the analytical Steiner first derivatives against central FD.

The analytical Steiner perimeter gradient and area Jacobian (closed-form;
derivation in docs/math/03-analytical-steiner-derivatives) replaced the
finite-difference versions.  This test checks both against a *central*
finite-difference reference computed directly from the exact analytical
forward values (compute_steiner_perimeter, compute_steiner_areas).

Central FD of an exact forward value is O(eps^2) accurate (~1e-9 here), so it
is a far better ground truth than the forward-difference *_fd_reference
functions (which are only O(eps) ~ 1e-4).  Agreement to ~1e-6 confirms the
closed-form derivatives are correct.

It also runs the analytical code with the module self-check enabled, which
asserts the translation-invariance identity sum_k dS/dp_k = I.

Usage:
    python testing/test_steiner_gradient_analytical.py --solution <path.h5>

Exit code: 0 on PASS (or SKIP), 1 on FAIL.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from _hessian_test_utils import build_optimizer  # noqa: E402
from src.partition import vectorized_steiner as vs  # noqa: E402


def central_fd_perimeter_gradient(pa, eps):
    """Central-FD reference for d(Steiner perimeter)/d(lambda)."""
    grad = np.zeros(pa.n_active_vp, dtype=np.float64)
    original = pa.vp_lambda.copy()
    for vp in pa.tp_affected_vps:
        pa.vp_lambda[vp] = original[vp] + eps
        f_plus = vs.compute_steiner_perimeter(pa, vs.compute_steiner_points(pa))
        pa.vp_lambda[vp] = original[vp] - eps
        f_minus = vs.compute_steiner_perimeter(pa, vs.compute_steiner_points(pa))
        pa.vp_lambda[vp] = original[vp]
        grad[vp] = (f_plus - f_minus) / (2.0 * eps)
    return grad


def central_fd_area_jacobian(pa, eps):
    """Central-FD reference for d(Steiner areas)/d(lambda), dense."""
    n_c = pa.n_cells - 1
    jac = np.zeros((n_c, pa.n_active_vp), dtype=np.float64)
    original = pa.vp_lambda.copy()
    for vp in pa.tp_affected_vps:
        pa.vp_lambda[vp] = original[vp] + eps
        a_plus = vs.compute_steiner_areas(pa, vs.compute_steiner_points(pa))
        pa.vp_lambda[vp] = original[vp] - eps
        a_minus = vs.compute_steiner_areas(pa, vs.compute_steiner_points(pa))
        pa.vp_lambda[vp] = original[vp]
        jac[:, vp] = (a_plus[:n_c] - a_minus[:n_c]) / (2.0 * eps)
    return jac


def _report(label, ana, ref, tol):
    abs_err = float(np.max(np.abs(ana - ref)))
    ref_mag = max(float(np.max(np.abs(ref))), 1e-30)
    rel_err = abs_err / ref_mag
    ok = abs_err < max(tol, tol * ref_mag)
    print(f"  {label}")
    print(f"    ||analytical||_inf = {np.max(np.abs(ana)):.6e}")
    print(f"    ||central-FD||_inf = {ref_mag:.6e}")
    print(f"    max |Δ|            = {abs_err:.3e}")
    print(f"    max rel err        = {rel_err:.3e}   [{'ok' if ok else 'FAIL'}]")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--solution', required=True)
    ap.add_argument('--tol', type=float, default=1e-6,
                    help='Absolute/relative tolerance (default 1e-6).')
    ap.add_argument('--fd-eps', type=float, default=1e-6,
                    help='Central-difference step (default 1e-6).')
    args = ap.parse_args()

    # Enable the analytical self-checks (translation-invariance assertion).
    vs._STEINER_SELF_CHECK = True

    opt = build_optimizer(args.solution)
    pa = opt._arrays

    if pa.n_triple_points == 0:
        print("SKIP: solution has no triple points — nothing to validate.")
        return 0

    print("Analytical Steiner first derivatives vs. central FD")
    print(f"  solution        = {args.solution}")
    print(f"  n_active_vp     = {pa.n_active_vp}")
    print(f"  n_triple_points = {pa.n_triple_points}")
    print(f"  fd-eps          = {args.fd_eps}")
    print()

    # --- Perimeter gradient ---
    grad_ana = vs.compute_steiner_perimeter_gradient_analytical(pa)
    grad_ref = central_fd_perimeter_gradient(pa, args.fd_eps)
    ok_grad = _report("Steiner perimeter gradient", grad_ana, grad_ref, args.tol)

    # --- Area Jacobian (dense) ---
    jac_ana = vs.compute_steiner_area_jacobian_analytical(pa)
    jac_ref = central_fd_area_jacobian(pa, args.fd_eps)
    ok_jac = _report("Steiner area Jacobian (dense)", jac_ana, jac_ref, args.tol)

    print()
    ok = ok_grad and ok_jac
    print("  (translation-invariance identity sum_k dS/dp_k = I asserted "
          "via _STEINER_SELF_CHECK)")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
