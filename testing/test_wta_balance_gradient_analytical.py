#!/usr/bin/env python3
"""Verify the analytical WTA balance gradient against central finite differences.

The WTA balance term P_bal = (gamma/2) sum_k r_k^2 with r_k = (T_k - Abar)/Abar
and soft territory T_k = sum_i v_i u_ik^p / S_i (derivation in
docs/math/07-phase1-wta-balance) is added to the Phase 1 energy behind
`wta_balance_enabled`. This test is the Stage 3 correctness gate of
docs/plans/PHASE1_TERRITORY_AWARE_IMPLEMENTATION_PLAN.md: nothing downstream
may run until it passes.

Checks (all deterministic, fixed seeds):
  1. Isolated-term FD: P_bal is computed as E_on - E_off from two optimizers
     sharing (K, M, v); its analytical gradient (g_on - g_off) is compared to
     Richardson-extrapolated central differences along several random
     directions on a feasible, strictly-interior density (V=200, N=8,
     projected then blended with the uniform row to stay interior).
     Relative error must be < 1e-6. Exercised at p=2 (default) and p=3.
  2. Full-energy FD: the assembled gradient (Dirichlet + well + crispness
     penalty + balance) is checked the same way.
  3. Property checks: grad_bal == 0 exactly at U == 1/N; grad_bal ~ 0 at a
     balanced configuration (uniform v, equal one-hot territories, blended);
     S_i respects the simplex lower bound N^(1-p) (stability guard).

Usage:
    python testing/test_wta_balance_gradient_analytical.py [--tol 1e-6]

Exit code: 0 on PASS, 1 on FAIL.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.optimization.pgd_optimizer import ProjectedGradientOptimizer  # noqa: E402
from src.optimization.projection import orthogonal_projection_iterative  # noqa: E402
from src.surfaces.torus import TorusMeshProvider  # noqa: E402

GAMMA = 7.0  # the calibrated default (docs/math/07-phase1-wta-balance sec. 7)


def build_pair(mesh_K, mesh_M, v, n_parts, epsilon, lam, gamma, power):
    """Two optimizers sharing (K, M, v): balance term on / off."""
    common = dict(K=mesh_K, M=mesh_M, v=v, n_partitions=n_parts,
                  epsilon=epsilon, lambda_penalty=lam)
    opt_off = ProjectedGradientOptimizer(**common)
    opt_on = ProjectedGradientOptimizer(
        wta_balance_enabled=True, wta_balance_gamma=gamma,
        wta_balance_power=power, **common)
    return opt_on, opt_off


def interior_feasible_density(V, n_parts, v, seed):
    """Deterministic feasible density, strictly interior (entries >= 0.1/n)."""
    rng = np.random.default_rng(seed)
    A = rng.random((V, n_parts))
    c = np.ones(n_parts)
    d = (v.sum() / n_parts) * np.ones(n_parts)
    A = orthogonal_projection_iterative(A, c, d, v, max_iter=500, tol=1e-10)
    # Blend with the uniform row: preserves both equality constraints exactly
    # (the uniform density has row sums 1 and column masses Abar) and bounds
    # every entry away from 0 so U +/- delta*E stays in the smooth region.
    return 0.9 * A + 0.1 / n_parts


def richardson_fd(f, x, e, delta):
    """Richardson-extrapolated central difference of f at x along e."""
    def central(dl):
        return (f(x + dl * e) - f(x - dl * e)) / (2.0 * dl)
    d1 = central(delta)
    d2 = central(delta / 2.0)
    return (4.0 * d2 - d1) / 3.0


def check_directions(label, f, grad, x, rng, n_dirs, delta, tol):
    """Compare <grad, e> with Richardson central FD along random directions."""
    ok = True
    worst = 0.0
    for j in range(n_dirs):
        e = rng.standard_normal(x.shape)
        e /= np.linalg.norm(e)
        ana = float(np.dot(grad.ravel(), e.ravel()))
        ref = richardson_fd(f, x, e, delta)
        scale = max(abs(ref), abs(ana), 1e-14)
        rel = abs(ana - ref) / scale
        worst = max(worst, rel)
        ok = ok and (rel < tol)
    print(f"  {label}")
    print(f"    directions = {n_dirs}, delta = {delta:.0e} (Richardson)")
    print(f"    worst rel err = {worst:.3e}   "
          f"[{'ok' if ok else 'FAIL'}] (gate < {tol:.0e})")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tol', type=float, default=1e-6,
                    help='Relative-error gate (default 1e-6).')
    ap.add_argument('--delta', type=float, default=1e-4,
                    help='Base central-difference step (default 1e-4).')
    ap.add_argument('--n-dirs', type=int, default=6,
                    help='Random directions per check (default 6).')
    args = ap.parse_args()

    # Small deterministic torus: 20 x 10 grid -> V = 200 vertices.
    prov = TorusMeshProvider(n_theta=20, n_phi=10, R=2.0, r=1.0)
    prov.set_resolution(20, 10)
    mesh = prov.build()
    mesh.compute_matrices()
    V = len(mesh.v)
    n_parts = 8
    epsilon = float(np.sqrt(mesh.get_mesh_statistics()['mean_triangle_area']))
    lam = 1.0

    print("Analytical WTA balance gradient vs. Richardson central FD")
    print(f"  mesh: torus 20x10, V = {V}, N = {n_parts}, "
          f"epsilon = {epsilon:.4e}, lambda = {lam}, gamma = {GAMMA}")
    print()

    U = interior_feasible_density(V, n_parts, mesh.v, seed=20260716)
    x = U.flatten()
    rng = np.random.default_rng(42)
    ok = True

    for power in (2.0, 3.0):
        opt_on, opt_off = build_pair(
            mesh.K, mesh.M, mesh.v, n_parts, epsilon, lam, GAMMA, power)

        # S_i stability on the feasible set: simplex bound N^(1-p) <= S <= 1.
        S, _, r = opt_on._wta_soft_territory(U)
        s_lo, s_hi = float(S.min()), float(S.max())
        bound = n_parts ** (1.0 - power)
        s_ok = (s_lo >= bound - 1e-12) and (s_hi <= 1.0 + 1e-12)
        ok = ok and s_ok
        print(f"  [p={power:g}] S_i in [{s_lo:.4e}, {s_hi:.4e}], "
              f"simplex lower bound N^(1-p) = {bound:.4e} "
              f"[{'ok' if s_ok else 'FAIL'}]")
        print(f"  [p={power:g}] territory deviations r: "
              f"min {r.min():+.3f}, max {r.max():+.3f} "
              f"(imbalanced test point, as intended)")

        def p_bal(z):
            return opt_on.compute_energy(z) - opt_off.compute_energy(z)

        grad_bal = opt_on.compute_gradient(x) - opt_off.compute_gradient(x)
        ok = check_directions(
            f"[p={power:g}] isolated P_bal gradient", p_bal, grad_bal,
            x, rng, args.n_dirs, args.delta, args.tol) and ok

        ok = check_directions(
            f"[p={power:g}] full assembled energy gradient",
            opt_on.compute_energy, opt_on.compute_gradient(x),
            x, rng, args.n_dirs, args.delta, args.tol) and ok
        print()

    # --- Property checks (p = 2, the default) --------------------------
    print("  Property checks (p = 2)")
    opt_on, opt_off = build_pair(
        mesh.K, mesh.M, mesh.v, n_parts, epsilon, lam, GAMMA, 2.0)

    # (a) Exactly zero at the symmetric state U == 1/N.
    x_sym = np.full(V * n_parts, 1.0 / n_parts)
    g_sym = opt_on.compute_gradient(x_sym) - opt_off.compute_gradient(x_sym)
    e_sym = opt_on.compute_energy(x_sym) - opt_off.compute_energy(x_sym)
    sym_ok = (float(np.max(np.abs(g_sym))) == 0.0) and (e_sym == 0.0)
    ok = ok and sym_ok
    print(f"    U = 1/N: P_bal = {e_sym:.1e}, max|grad_bal| = "
          f"{np.max(np.abs(g_sym)):.1e}  "
          f"[{'ok (exactly zero)' if sym_ok else 'FAIL'}]")

    # (b) ~zero at a balanced configuration: uniform v, equal one-hot
    # territories, blended toward uniform (rows are permutations of one
    # profile -> T_k = Abar exactly).
    v_uni = np.full(V, mesh.v.sum() / V)
    opt_on_u, opt_off_u = build_pair(
        mesh.K, mesh.M, v_uni, n_parts, epsilon, lam, GAMMA, 2.0)
    labels = np.repeat(np.arange(n_parts), V // n_parts)
    U_bal = np.full((V, n_parts), 0.1 / n_parts)
    U_bal[np.arange(V), labels] += 0.9
    x_bal = U_bal.flatten()
    g_bal = opt_on_u.compute_gradient(x_bal) - opt_off_u.compute_gradient(x_bal)
    e_bal = opt_on_u.compute_energy(x_bal) - opt_off_u.compute_energy(x_bal)
    gscale = float(np.max(np.abs(opt_on_u.compute_gradient(x_bal))))
    bal_ok = (float(np.max(np.abs(g_bal))) < 1e-12 * max(gscale, 1.0)
              and abs(e_bal) < 1e-20)
    ok = ok and bal_ok
    print(f"    balanced U: P_bal = {e_bal:.1e}, max|grad_bal| = "
          f"{np.max(np.abs(g_bal)):.1e} (full-grad scale {gscale:.1e})  "
          f"[{'ok' if bal_ok else 'FAIL'}]")

    # (c) Flag off => bit-identical to the base optimizer on the same input.
    g_off_a = opt_off.compute_gradient(x)
    base = ProjectedGradientOptimizer(
        K=mesh.K, M=mesh.M, v=mesh.v, n_partitions=n_parts,
        epsilon=epsilon, lambda_penalty=lam)
    g_off_b = base.compute_gradient(x)
    off_ok = (np.array_equal(g_off_a, g_off_b)
              and opt_off.compute_energy(x) == base.compute_energy(x))
    ok = ok and off_ok
    print(f"    flag off: energy+gradient bit-identical to base "
          f"[{'ok' if off_ok else 'FAIL'}]")

    print()
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
