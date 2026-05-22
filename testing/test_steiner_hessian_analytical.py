#!/usr/bin/env python3
"""Verify the analytical Steiner second derivatives.

The analytical Steiner Hessians (closed-form; derivation in
docs/math/03-analytical-steiner-derivatives) are built on the analytical
second derivative of the Steiner point, d2S/dlambda^2.

Three checks:
  1. d2S/dlambda^2 vs a central finite difference of the analytical
     dS/dlambda — the foundational check (the plan's highest-leverage gate).
  2. Steiner perimeter Hessian: analytical vs the central-FD reference.
  3. Steiner area Hessian (random multipliers): analytical vs the FD reference.

The *_fd_reference Hessians central-difference the *analytical* first
derivatives, so they are ~1e-9-accurate references (not the noisy nested-FD
of Phase 1).  Agreement to ~1e-5 confirms the closed-form Hessians.

The module self-check (_STEINER_SELF_CHECK) is enabled, asserting the
second-order translation identity sum_k d2S/dp_k dp_l = 0 and mixed-partial
symmetry of d2S/dp2.

Usage:
    python testing/test_steiner_hessian_analytical.py --solution <path.h5>

Exit code: 0 on PASS (or SKIP), 1 on FAIL.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from _hessian_test_utils import build_optimizer  # noqa: E402
from src.partition import vectorized_steiner as vs  # noqa: E402


def central_fd_d2S_dlambda(pa, eps):
    """Central-FD reference for d2S/dlambda^2, shape (n_tp, 3, 3, dim).

    Perturbs the lambda of each triple-point slot and central-differences the
    analytical dS/dlambda.  eps=1e-5 is the validated smooth-regime step
    (larger steps can cross a near-120-degree degenerate-branch flip).
    """
    def dS_dl_of():
        g = vs._compute_tp_geometry(pa)
        return vs._compute_dS_dlambda(g, vs._compute_dS_dp(g))

    n_tp = pa.n_triple_points
    dim = pa.vertices.shape[1]
    ref = np.zeros((n_tp, 3, 3, dim), dtype=np.float64)
    original = pa.vp_lambda.copy()
    for t in range(n_tp):
        for j in range(3):
            vp = pa.tp_vp_indices[t, j]
            pa.vp_lambda[vp] = original[vp] + eps
            dsl_p = dS_dl_of()
            pa.vp_lambda[vp] = original[vp] - eps
            dsl_m = dS_dl_of()
            pa.vp_lambda[vp] = original[vp]
            ref[t, :, j, :] = (dsl_p[t] - dsl_m[t]) / (2.0 * eps)
    return ref


def _report(label, ana, ref, tol):
    abs_err = float(np.max(np.abs(ana - ref)))
    ref_mag = max(float(np.max(np.abs(ref))), 1e-30)
    rel_err = abs_err / ref_mag
    ok = abs_err < max(tol, tol * ref_mag)
    print(f"  {label}")
    print(f"    ||analytical||_inf = {np.max(np.abs(ana)):.6e}")
    print(f"    ||reference||_inf  = {ref_mag:.6e}")
    print(f"    max |Δ|            = {abs_err:.3e}")
    print(f"    max rel err        = {rel_err:.3e}   [{'ok' if ok else 'FAIL'}]")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--solution', required=True)
    ap.add_argument('--tol', type=float, default=1e-5,
                    help='Absolute/relative tolerance (default 1e-5).')
    ap.add_argument('--fd-eps', type=float, default=1e-5,
                    help='Central-difference step for the d2S check (default 1e-5).')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    vs._STEINER_SELF_CHECK = True

    opt = build_optimizer(args.solution)
    pa = opt._arrays

    if pa.n_triple_points == 0:
        print("SKIP: solution has no triple points — nothing to validate.")
        return 0

    print("Analytical Steiner second derivatives")
    print(f"  solution        = {args.solution}")
    print(f"  n_active_vp     = {pa.n_active_vp}")
    print(f"  n_triple_points = {pa.n_triple_points}")
    print()

    # --- 1. d2S/dlambda^2 vs central FD of dS/dlambda ---
    geom = vs._compute_tp_geometry(pa)
    dS_dp = vs._compute_dS_dp(geom)
    d2S_ana = vs._compute_d2S_dlambda(geom, vs._compute_d2S_dp2(geom, dS_dp))
    d2S_ref = central_fd_d2S_dlambda(pa, args.fd_eps)
    ok_d2s = _report("d2S/dlambda^2 (vs central FD of dS/dlambda)",
                     d2S_ana, d2S_ref, args.tol)

    # --- 2. Steiner perimeter Hessian ---
    hp_ana = vs.compute_steiner_perimeter_hessian_analytical(pa)
    hp_ref = vs.compute_steiner_perimeter_hessian_fd_reference(pa)
    ok_perim = _report("Steiner perimeter Hessian (vs FD reference)",
                       hp_ana, hp_ref, args.tol)

    # --- 3. Steiner area Hessian (random multipliers) ---
    rng = np.random.default_rng(args.seed)
    mu = rng.standard_normal(pa.n_cells - 1)
    ha_ana = vs.compute_steiner_area_hessian_analytical(pa, mu)
    ha_ref = vs.compute_steiner_area_hessian_fd_reference(pa, mu)
    ok_area = _report("Steiner area Hessian, random multipliers (vs FD reference)",
                      ha_ana, ha_ref, args.tol)

    print()
    ok = ok_d2s and ok_perim and ok_area
    print("  (2nd-order translation identity + d2S mixed-partial symmetry "
          "asserted via _STEINER_SELF_CHECK)")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
