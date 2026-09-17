#!/usr/bin/env python3
"""Regenerate EVERY computed number in docs/math/10-mbo-auction-dynamics/main.tex.

    python docs/math/10-mbo-auction-dynamics/check_numerics.py            # ~5 min
    python docs/math/10-mbo-auction-dynamics/check_numerics.py --quick    # ~1 min; skips the
                                                        # 100x96 dense eigendecompositions

Writes the COMMITTED ``numerics.yaml`` beside this script. ``--quick`` writes
``numerics_quick.yaml`` instead (gitignored) so a partial run can never replace
the committed record. The document quotes
numbers from that file and nowhere else; a number that this script does not
produce is either derived by hand in the text or cited from a report/paper with
its source named. This is the math-folder counterpart of the experiments folder's
``extract_data.py``/``make_figures.py`` chain (``docs/experiments/README.md``):

    check_numerics.py  ->  numerics.yaml (committed)  ->  main.tex

Everything is deterministic: the one random element (the jittered disc centres of
Sec. 5.6) uses ``numpy.random.default_rng(SEED)`` with the draw order fixed below.
Re-running on the same machine must reproduce numerics.yaml to the printed digits;
floating-point differences across BLAS builds are possible in the last digit of
the eigenvalues and are noted where they would matter.

Sections, keyed as in numerics.yaml:

  constants   Sec. 4.3 / 5.2-5.5 / 8.3 / Remark 5.2 -- the identities the derivation
              rests on, checked numerically: resolvent kernel = Exp(tau)-mixture of
              heat kernels; interface constants 1/2 and 1/sqrt(pi); per-step
              displacement tau/2; Bessel integrals pi/2; second moments 2 tau;
              centre value 1 - x K1(x); the exact disc solution's ODE residual and
              Wronskian continuity.
  positivity  Sec. 5.3(iii) -- min entry of exp(-tL) at t = 0.1 h^2 (40x36) and of
              A_tau = (M + tau K)^-1 M at the production tau on 40x36 / 60x52 /
              100x96.
  pairing     Sec. 7.2 -- eigenvalue range of sym(D A_tau), ||skew||_F/||D A_tau||_F,
              and the skew term at an ACTUAL labelling pair (geodesic balanced
              init -> one balanced step) with the count of vertices whose
              unconstrained argmax it moves.
  disc_check  Sec. 5.6 -- mesh statistics of the 348x328 deliverable mesh; exact
              continuum resolvent and Gaussian disc losses; the vertex-centred
              one-step area change; the jittered-centre mean/std; ratios;
              displacement in edge lengths; lumped-area deviation of the ambient
              disc; Gaussian curvature and the geodesic-curvature correction.
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import sys
import time
from datetime import date

import numpy as np
import scipy
import scipy.sparse as sp
import yaml
from scipy import integrate
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.sparse.linalg import factorized
from scipy.special import gamma, i0, i1, k0, k1

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "numerics.yaml")
OUT_QUICK = os.path.join(HERE, "numerics_quick.yaml")

logging.disable(logging.CRITICAL)

from src.optimization.initialization import farthest_point_sampling  # noqa: E402
from src.partition.mbo_auction import (  # noqa: E402
    MBOConfig,
    assignment_quality_bar,
    balanced_assign,
    diffuse_indicators,
    geodesic_balanced_init,
    make_diffusion_solver,
    tau_diagnostics,
)
from src.surfaces.torus import TorusMeshProvider  # noqa: E402

SEED = 0            # jittered disc centres, Sec. 5.6
INIT_SEED = 84172851  # the project's production seed, for the labelling pair
N_JITTER = 24
TORUS = dict(R=1.0, r=0.6)
DELIVERABLE_MESH = (348, 328)
DISC_ROWS = [(0.20, 0.004), (0.25, 0.004), (0.30, 0.006), (0.20, 0.002)]
PAIRING_MESHES = [(40, 36, 10), (60, 52, 100)]
BASE_MESH_CASES = [(100, 96, 100), (100, 96, 400)]


def f(x, nd=6):
    return float(np.round(float(x), nd))


def build(nt, nph):
    return TorusMeshProvider(n_theta=nt, n_phi=nph, R=TORUS["R"], r=TORUS["r"]).build()


def edge_stats(mesh):
    F = mesh.faces
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    L = np.linalg.norm(mesh.vertices[e[:, 0]] - mesh.vertices[e[:, 1]], axis=1)
    return float(L.mean()), float(L.max())


# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------


def exact_resolvent_loss_factor(x):
    """Exact 1/2-threshold area loss of a disc under the resolvent kernel, / (pi tau).

    y(u) = 1 - x K1(x) I0(u) inside, x I1(x) K0(u) outside (eq. disc-exact).
    """
    inside = lambda u: 1.0 - x * k1(x) * i0(u) - 0.5
    if inside(x) < 0:
        u = brentq(inside, 1e-9, x)
    else:
        u = brentq(lambda u: x * i1(x) * k0(u) - 0.5, x, 50 * x)
    return x * x - u * u


def exact_gaussian_loss_factor(x):
    """Exact 1/2-threshold area loss of a disc under the heat kernel at time tau, / (2 pi tau).

    Weber's integral: y(u) = int_0^x (s/2) exp(-(s^2+u^2)/4) I0(s u / 2) ds.
    """
    y = lambda u: integrate.quad(
        lambda s: (s / 2.0) * np.exp(-(s * s + u * u) / 4.0) * i0(s * u / 2.0), 0, x
    )[0]
    u = brentq(lambda u: y(u) - 0.5, 1e-6, x)
    return (x * x - u * u) / 2.0


def constants():
    tau, r = 0.7, 0.9
    out = {}
    # mixture representation vs closed-form K0 kernel (eq. resolvent-kernel)
    mix = integrate.quad(
        lambda t: (1 / tau) * np.exp(-t / tau) * np.exp(-r * r / (4 * t)) / (4 * np.pi * t),
        0, np.inf,
    )[0]
    out["mixture_vs_K0_kernel"] = {
        "tau": tau, "r": r, "mixture_integral": f(mix, 12),
        "closed_form_K0": f(k0(r / np.sqrt(tau)) / (2 * np.pi * tau), 12),
    }
    # interface constants (Prop. constant): (1/sqrt tau) E_t[sqrt(t/pi)] and marginal form
    c_mix = integrate.quad(lambda t: (1 / tau) * np.exp(-t / tau) * np.sqrt(t / np.pi), 0, np.inf)[0] / np.sqrt(tau)
    c_marg = integrate.quad(lambda s: s * np.exp(-s / np.sqrt(tau)) / (2 * np.sqrt(tau)), 0, np.inf)[0] / np.sqrt(tau)
    out["interface_constant"] = {
        "resolvent_via_mixture": f(c_mix, 10), "resolvent_via_marginal": f(c_marg, 10),
        "gaussian_1_over_sqrt_pi": f(1 / np.sqrt(np.pi), 10),
    }
    # displacement (Prop. displacement): E[t^1/2]/E[t^-1/2] = tau/2
    num = integrate.quad(lambda t: (1 / tau) * np.exp(-t / tau) * np.sqrt(t), 0, np.inf)[0]
    den = integrate.quad(lambda t: (1 / tau) * np.exp(-t / tau) / np.sqrt(t), 0, np.inf)[0]
    Kres = lambda y: k0(abs(y) / np.sqrt(tau)) / (2 * np.pi * tau)
    m2 = 2 * integrate.quad(lambda y: y * y * Kres(y), 0, np.inf)[0]
    m0 = 2 * integrate.quad(Kres, 0, np.inf)[0]
    out["displacement_over_kappa"] = {
        "via_mixture": f(num / den, 10), "via_2d_kernel_ratio_half": f(m2 / m0 / 2, 10),
        "tau_over_2": f(tau / 2, 10), "marginal_k0_check": f(m0, 10),
        "marginal_k0_expected": f(1 / (2 * np.sqrt(tau)), 10),
    }
    # Bessel integrals (Remark second moments)
    out["bessel_integrals"] = {
        "int_K0": f(integrate.quad(k0, 0, np.inf)[0], 10),
        "int_s2_K0": f(integrate.quad(lambda s: s * s * k0(s), 0, np.inf)[0], 10),
        "pi_over_2": f(np.pi / 2, 10),
        "gamma_3_2_over_gamma_1_2": f(gamma(1.5) / gamma(0.5), 10),
    }
    # second moments per coordinate: both 2 tau
    out["second_moment_per_coordinate"] = {
        "resolvent": f(2 * integrate.quad(lambda s: s * s * np.exp(-s / np.sqrt(tau)) / (2 * np.sqrt(tau)), 0, np.inf)[0], 10),
        "two_tau": f(2 * tau, 10),
    }
    # centre value 1 - x K1(x) (eq. centre) vs Gaussian 1 - exp(-x^2/4)
    R, tt = 1.3, 0.5
    mixc = integrate.quad(lambda t: (1 / tt) * np.exp(-t / tt) * (1 - np.exp(-R * R / (4 * t))), 0, np.inf)[0]
    out["centre_value"] = {
        "at_x": {str(x): {"resolvent": f(1 - x * k1(x), 4), "gaussian": f(1 - np.exp(-x * x / 4), 4)} for x in (1, 2, 4)},
        "mixture_check": {"R": R, "tau": tt, "mixture": f(mixc, 10), "closed_form": f(1 - (R / np.sqrt(tt)) * k1(R / np.sqrt(tt)), 10)},
    }
    # exact disc solution: ODE residual and C^1 continuity at u = x (eq. disc-exact)
    x = 3.87
    yin = lambda u: 1 - x * k1(x) * i0(u)
    yout = lambda u: x * i1(x) * k0(u)
    h = 1e-3
    def resid(yf, u, rhs):
        d1 = (yf(u + h) - yf(u - h)) / (2 * h)
        d2 = (yf(u + h) - 2 * yf(u) + yf(u - h)) / (h * h)
        return yf(u) - d2 - d1 / u - rhs
    out["disc_exact_solution"] = {
        "x": x,
        "ode_residual_inside_u=2": f(resid(yin, 2.0, 1.0), 8),
        "ode_residual_outside_u=6": f(resid(yout, 6.0, 0.0), 8),
        "jump_at_x": f(yin(x) - yout(x), 12),
        "derivative_jump_at_x": f((yin(x + h) - yin(x - h)) / (2 * h) - (yout(x + h) - yout(x - h)) / (2 * h), 6),
        "wronskian_I0K1_plus_I1K0_minus_1_over_x": f(i0(x) * k1(x) + i1(x) * k0(x) - 1 / x, 12),
    }
    return out


# --------------------------------------------------------------------------
# positivity and pairing
# --------------------------------------------------------------------------


def dense_operators(mesh, N):
    M = mesh.M.toarray()
    K = mesh.K.toarray()
    D = np.diag(mesh.v)
    tau = tau_diagnostics(mesh, N, MBOConfig())["tau"]
    A = np.linalg.solve(M + tau * K, M)
    return M, K, D, tau, A


def pairing_case(nt, nph, N, do_eigs=True, do_labelling=True):
    mesh = build(nt, nph)
    M, K, D, tau, A = dense_operators(mesh, N)
    DA = D @ A
    S = (DA + DA.T) / 2
    Z = (DA - DA.T) / 2
    rec = {
        "mesh": f"{nt}x{nph}", "V": int(len(mesh.v)), "N": N, "tau": f(tau, 6),
        "min_entry_A_tau": f(A.min(), 12),
        "skew_over_DA_frobenius": f(np.linalg.norm(Z) / np.linalg.norm(DA), 8),
        "MA_tau_asymmetry_frobenius": f(np.linalg.norm(M @ A - (M @ A).T) / np.linalg.norm(M @ A), 12),
    }
    if do_eigs:
        ev = np.linalg.eigvalsh(S)
        rec["sym_DA_min_eig"] = f(ev.min(), 12)
        rec["sym_DA_max_eig"] = f(ev.max(), 8)
        evm = np.linalg.eigvalsh(M @ A)
        rec["MA_tau_min_eig"] = f(evm.min(), 12)
    if do_labelling:
        labels, _ = geodesic_balanced_init(mesh, N, INIT_SEED)
        chi = np.zeros((len(labels), N)); chi[np.arange(len(labels)), labels] = 1.0
        y = A @ chi
        bar = assignment_quality_bar(mesh.v, N)
        target = float(mesh.v.sum()) / N
        _, new_labels, _ = balanced_assign(y, mesh.v, target, bar, early_stop=False)
        chi2 = np.zeros_like(chi); chi2[np.arange(len(labels)), new_labels] = 1.0
        s_term = float(np.sum(chi2 * (S @ chi)))
        z_term = float(np.sum(chi2 * (Z @ chi)))
        moved = int(np.count_nonzero(np.argmax(S @ chi, axis=1) != np.argmax((S + Z) @ chi, axis=1)))
        rec["labelling_pair"] = {
            "init_seed": INIT_SEED,
            "skew_term_over_sym_term": f(z_term / s_term, 10),
            "argmax_moved_vertices": moved,
            "argmax_moved_fraction": f(moved / len(labels), 8),
            "labels_changed_by_step": int(np.count_nonzero(new_labels != labels)),
        }
    return mesh, M, K, rec


def positivity_and_pairing(quick):
    out = {"pairing_cases": [], "expm_check": None}
    for (nt, nph, N) in PAIRING_MESHES:
        mesh, M, K, rec = pairing_case(nt, nph, N)
        out["pairing_cases"].append(rec)
        if (nt, nph) == (40, 36):
            h_mean, _ = edge_stats(mesh)
            t = 0.1 * h_mean ** 2
            L = np.linalg.solve(M, K)
            E = expm(-t * L)
            out["expm_check"] = {
                "mesh": "40x36", "t": f(t, 8), "t_over_h_mean2": 0.1,
                "min_entry_expm": f(E.min(), 8), "row_sum_dev": f(np.abs(E.sum(axis=1) - 1).max(), 12),
                "K_max_positive_offdiag": f((K - np.diag(np.diag(K))).max(), 8),
            }
    if not quick:
        for (nt, nph, N) in BASE_MESH_CASES:
            _, _, _, rec = pairing_case(nt, nph, N, do_eigs=True, do_labelling=False)
            out["pairing_cases"].append(rec)
    else:
        out["base_mesh_skipped"] = True
    return out


# --------------------------------------------------------------------------
# Sec. 5.6 disc check on the deliverable mesh
# --------------------------------------------------------------------------


def disc_check():
    nt, nph = DELIVERABLE_MESH
    mesh = build(nt, nph)
    Vx, M, K, v = mesh.vertices, mesh.M, mesh.K, mesh.v
    h_mean, h_max = edge_stats(mesh)
    R, r = TORUS["R"], TORUS["r"]
    centre0 = np.array([R + r, 0.0, 0.0])
    rng = np.random.default_rng(SEED)
    dth, dph = 2 * np.pi / nt, 2 * np.pi / nph
    Kgauss = np.cos(0.0) / (r * (R + r * np.cos(0.0)))  # Gaussian curvature at the outer equator
    out = {
        "mesh": f"{nt}x{nph}", "V": int(len(v)), "h_mean": f(h_mean, 6), "h_max": f(h_max, 6),
        "anisotropy": f(h_max / h_mean, 4), "total_area": f(v.sum(), 6),
        "centre": [float(c) for c in centre0], "jitter": {
            "seed": SEED, "n_samples_per_row": N_JITTER,
            "parametrisation": "theta ~ U(0, 2pi/n_theta), phi ~ U(0, 2pi/n_phi) about the centre, "
                               "one shared rng consumed row by row in the order listed",
        },
        "gaussian_curvature_outer_equator": f(Kgauss, 6),
        "rows": [],
    }
    for Rd, tau in DISC_ROWS:
        solve = factorized((M + tau * K).tocsc())
        x = Rd / np.sqrt(tau)
        ex_factor = exact_resolvent_loss_factor(x)
        ga_factor = exact_gaussian_loss_factor(x)
        ex_loss, ga_loss = np.pi * tau * ex_factor, 2 * np.pi * tau * ga_factor
        chi = (np.linalg.norm(Vx - centre0, axis=1) <= Rd).astype(float)
        A0 = float(v @ chi)
        y = solve(M @ chi)
        dA0 = float(v @ (y >= 0.5) - A0)
        ratios, ratios_g = [], []
        for _ in range(N_JITTER):
            th = rng.uniform(0, dth); ph = rng.uniform(0, dph)
            c = np.array([(R + r * np.cos(ph)) * np.cos(th), (R + r * np.cos(ph)) * np.sin(th), r * np.sin(ph)])
            chij = (np.linalg.norm(Vx - c, axis=1) <= Rd).astype(float)
            yj = solve(M @ chij)
            dA = float(v @ (yj >= 0.5) - v @ chij)
            ratios.append(-dA / ex_loss); ratios_g.append(-dA / ga_loss)
        ratios, ratios_g = np.array(ratios), np.array(ratios_g)
        delta_meas = Rd - np.sqrt(Rd * Rd + dA0 / np.pi)   # displacement implied by the measured loss
        out["rows"].append({
            "R_d": Rd, "tau": tau, "sqrt_tau_over_h_mean": f(np.sqrt(tau) / h_mean, 3), "x": f(x, 3),
            "leading_order_loss_pi_tau": f(np.pi * tau, 6),
            "exact_resolvent_factor": f(ex_factor, 4), "exact_resolvent_loss": f(ex_loss, 6),
            "exact_gaussian_factor": f(ga_factor, 4), "exact_gaussian_loss_2pi_tau_times": f(ga_loss, 6),
            "lumped_area_ambient_disc": f(A0, 6), "pi_Rd2": f(np.pi * Rd * Rd, 6),
            "lumped_area_dev_pct": f(100 * (A0 / (np.pi * Rd * Rd) - 1), 3),
            "vertex_centred": {
                "dA": f(dA0, 6), "ratio_to_leading_order": f(-dA0 / (np.pi * tau), 4),
                "ratio_to_exact": f(-dA0 / ex_loss, 4), "ratio_to_gaussian_exact": f(-dA0 / ga_loss, 4),
                "displacement_over_h_mean": f(delta_meas / h_mean, 3),
                "y_min": f(y.min(), 6), "y_max": f(y.max(), 6),
            },
            "jittered": {
                "ratio_to_exact_mean": f(ratios.mean(), 4), "ratio_to_exact_std": f(ratios.std(), 4),
                "ratio_to_exact_min": f(ratios.min(), 4), "ratio_to_exact_max": f(ratios.max(), 4),
                "ratio_to_gaussian_exact_mean": f(ratios_g.mean(), 4),
            },
            "geodesic_curvature_correction_pct": f(100 * Kgauss * Rd * Rd / 3, 3),
        })
    return out


# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the 100x96 dense eigendecompositions")
    args = ap.parse_args()
    t0 = time.perf_counter()
    doc = {
        "provenance": {
            "generated": str(date.today()),
            "script": "docs/math/10-mbo-auction-dynamics/check_numerics.py",
            "quick": bool(args.quick),
            "seed_jitter": SEED, "seed_init": INIT_SEED,
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(),
            "torus": TORUS,
        },
        "constants": constants(),
    }
    print("constants done", flush=True)
    doc["positivity_and_pairing"] = positivity_and_pairing(args.quick)
    print("positivity/pairing done", flush=True)
    doc["disc_check"] = disc_check()
    print("disc check done", flush=True)
    doc["provenance"]["wall_seconds"] = f(time.perf_counter() - t0, 1)
    out = OUT_QUICK if args.quick else OUT
    with open(out, "w") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, default_flow_style=False, width=100)
    print(f"wrote {out} in {doc['provenance']['wall_seconds']} s")
    # console summary of the numbers the document quotes
    for row in doc["disc_check"]["rows"]:
        vc, jt = row["vertex_centred"], row["jittered"]
        print(f"  R_d={row['R_d']} tau={row['tau']} x={row['x']}: exact {row['exact_resolvent_loss']:.5f} "
              f"vertex-centred {vc['ratio_to_exact']:.3f} jittered {jt['ratio_to_exact_mean']:.3f}+-{jt['ratio_to_exact_std']:.3f} "
              f"vs gaussian {jt['ratio_to_gaussian_exact_mean']:.3f}")
    for rec in doc["positivity_and_pairing"]["pairing_cases"]:
        print(f"  {rec['mesh']} N={rec['N']}: tau={rec['tau']} min eig sym(DA)={rec.get('sym_DA_min_eig')} "
              f"max={rec.get('sym_DA_max_eig')} skew/DA={rec['skew_over_DA_frobenius']} minA={rec['min_entry_A_tau']}"
              + (f" skew/sym at pair={rec['labelling_pair']['skew_term_over_sym_term']} moved={rec['labelling_pair']['argmax_moved_vertices']}" if 'labelling_pair' in rec else ""))
    e = doc["positivity_and_pairing"]["expm_check"]
    print(f"  expm(-tL) 40x36 t={e['t']}: min entry {e['min_entry_expm']}")


if __name__ == "__main__":
    main()
