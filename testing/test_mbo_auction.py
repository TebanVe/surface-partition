#!/usr/bin/env python3
"""Correctness gates and negative controls for approach B (auction-dynamics MBO).

    python testing/test_mbo_auction.py                      # G1-G8
    python testing/test_mbo_auction.py --negative-controls  # NC1, NC4, NC5, NC3-CAL, NC2
    python testing/test_mbo_auction.py --all

**Every gate here has a failing counterpart, and that is the point.** Report 07
catalogues five occasions on which this project read a measurement artefact as a
result about the method, and names only two practices that caught any of them:
committing predicted numbers before running, and asking *"does anything fail?"*
rather than only *"does the shipped thing pass?"*. A suite in which everything
passes is indistinguishable from a suite that cannot fail. So G1 mutates its own
input to prove the check bites; G4 refuses to pass on a frozen tail; NC1/NC4 assert
that something MUST break; NC3-CAL is required to find a threshold that separates a
frozen state from a converged one, and reports failure if none exists; and NC5 must
distinguish c=8 from c=4 or the plan records that no instrument here can see
over-merging at all.

Runtime: gates ~20 min, negative controls ~70 min.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import eigsh, factorized

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.partition.arm_harness import (  # noqa: E402
    assignment_quality_bar,
    labels_to_one_hot,
    run_gates,
    vertex_granularity,
)
from src.partition.balanced_readout import (  # noqa: E402
    BalancedReadoutConfig,
    solve_dual_offsets,
)
from src.partition.find_contours import detect_area_imbalance  # noqa: E402
from src.partition.mbo_auction import (  # noqa: E402
    MBOConfig,
    balanced_assign,
    compactness,
    core_loss,
    descent_slack,
    diffuse_indicators,
    edge_graph,
    geodesic_balanced_init,
    lyapunov_energy,
    make_diffusion_solver,
    mbo_level,
    sparse_one_hot,
    tau_continuation_probe,
    tau_diagnostics,
)
from src.optimization.initialization import farthest_point_sampling  # noqa: E402
from src.surfaces.torus import TorusMeshProvider  # noqa: E402


def build(nt, nphi):
    return TorusMeshProvider(n_theta=nt, n_phi=nphi, R=1.0, r=0.6).build()


def banner(name, text=""):
    print("\n" + "=" * 76)
    print(f"{name}{'  --  ' + text if text else ''}")
    print("=" * 76)


def verdict(ok, label, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
    return bool(ok)


# ==========================================================================
# G1 -- diffusion identity, with a mutation test so the check is known to bite
# ==========================================================================


def gate1():
    banner("G1", "diffusion identity: mass conservation, partition of unity, y >= 0")
    ok = True
    for nt, nphi, N in [(60, 56, 20), (100, 96, 100)]:
        mesh, cfg = build(nt, nphi), MBOConfig()
        tau = tau_diagnostics(mesh, N, cfg)["tau"]
        solve = make_diffusion_solver(mesh, tau)
        # A deliberately scrambled labelling: the identities below are properties
        # of the operator, so they must hold for ANY indicator, not just a nice one.
        labels = np.random.default_rng(0).integers(0, N, len(mesh.vertices))
        y, _ = diffuse_indicators(solve, mesh, labels, N)

        chi = sparse_one_hot(labels, N)
        mass_y = mesh.v @ y
        mass_chi = np.asarray((mesh.v @ chi.toarray()))
        mass_err = float(np.abs(mass_y - mass_chi).max() / np.abs(mass_chi).max())
        pou = float(np.abs(y.sum(axis=1) - 1.0).max())
        ymin = float(y.min())

        print(
            f"  V={len(mesh.vertices):>7d} N={N:>4d}: mass rel err={mass_err:.2e}  "
            f"|sum_k y_k - 1|={pou:.2e}  min(y)={ymin:.3e}"
        )
        ok &= verdict(mass_err < 1e-12, f"mass conserved (V={len(mesh.vertices)})")
        ok &= verdict(pou < 1e-12, f"partition of unity (V={len(mesh.vertices)})")
        # y >= 0 is MEASURED, not assumed: the discrete maximum principle need not
        # hold on a non-Delaunay mesh, and a negative excursion is worth knowing
        # about even though nothing downstream requires positivity.
        print(f"       y >= 0 measured: {ymin >= 0.0} (informational, not a gate)")

    # Mutation test: the check must FAIL on a perturbed field, or it proves nothing.
    y_bad = y.copy()
    y_bad[0, 0] += 1.0
    mass_bad = float(np.abs((mesh.v @ y_bad) - mass_chi).max() / np.abs(mass_chi).max())
    ok &= verdict(
        mass_bad > 1e-12,
        "mutation test: perturbed y is REJECTED",
        f"(rel err {mass_bad:.2e})",
    )
    return ok


# ==========================================================================
# G2 -- tau diagnostics against an independent recomputation
# ==========================================================================


def gate2():
    banner("G2", "tau diagnostics reproduce an independent recomputation")
    ok = True
    for nt, nphi, N in [(100, 96, 100), (224, 212, 300), (348, 328, 100)]:
        mesh, cfg = build(nt, nphi), MBOConfig()
        d = tau_diagnostics(mesh, N, cfg)

        f = mesh.faces
        e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        L = np.linalg.norm(mesh.vertices[e[:, 0]] - mesh.vertices[e[:, 1]], axis=1)
        hm, hx = L.mean(), L.max()
        A = float(mesh.M.sum())
        rc = np.sqrt(A / N / np.pi)
        tau_ref = min((cfg.tau_c * hm) ** 2, (cfg.rho * rc) ** 2)
        st = np.sqrt(tau_ref)

        checks = {
            "tau": (d["tau"], tau_ref),
            "sqrt_tau/h_max": (d["sqrt_tau_over_h_max"], st / hx),
            "sqrt_tau/R_cell": (d["sqrt_tau_over_r_cell"], st / rc),
            "c_eff": (d["c_eff"], st / hm),
        }
        bad = {
            k: v
            for k, v in checks.items()
            if abs(v[0] - v[1]) > 1e-12 * max(1, abs(v[1]))
        }
        print(
            f"  V={len(mesh.vertices):>7d} N={N:>4d}: sqrt(t)/h_max={d['sqrt_tau_over_h_max']:.3f} "
            f"sqrt(t)/R_cell={d['sqrt_tau_over_r_cell']:.3f} c_eff={d['c_eff']:.3f} "
            f"cap={'ON' if d['cap_active'] else 'off'}"
        )
        ok &= verdict(
            not bad,
            f"diagnostics exact (V={len(mesh.vertices)}, N={N})",
            str(bad) if bad else "",
        )
    return ok


# ==========================================================================
# G3 -- the hard-required initialization is balanced
# ==========================================================================


def gate3():
    banner("G3", "geodesic-balanced init reaches the assignment quality bar")
    ok = True
    for nt, nphi, N in [(100, 96, 100), (224, 212, 300)]:
        mesh = build(nt, nphi)
        labels, info = geodesic_balanced_init(mesh, N, seed=84172851)
        bar = info["quality_bar"]
        print(
            f"  V={len(mesh.vertices):>7d} N={N:>4d}: worst={info['worst_rel_dev']*100:.4f}%  "
            f"bar={bar*100:.4f}%  gran={vertex_granularity(mesh.v, N)*100:.4f}%  "
            f"{info['wall_seconds']:.1f}s"
        )
        ok &= verdict(
            info["worst_rel_dev"] <= bar,
            f"init balanced (V={len(mesh.vertices)}, N={N})",
        )
        ok &= verdict(len(np.unique(labels)) == N, "every cell owns territory")
    return ok


# ==========================================================================
# G4 -- the descent inequality, counting ACTIVE steps only
# ==========================================================================


def gate4():
    banner("G4", "descent inequality <chi',Dy> - <chi,Dy> >= sum_k psi_k (T_k - T'_k)")
    print("  The monotonicity THEOREM does not transfer (non-symmetric D*A_tau, and")
    print(
        "  an inexact subgradient dual where Jacobs-Kim-Leger have an exact auction)."
    )
    print("  This inequality is what survives; raw E_tau is a trend, never a gate.\n")
    ok = True
    # N=300 at V=9,600 is 32 verts/cell: it freezes after ~12 steps, so the gate
    # would be certified on a 28-step flat tail. Measured, and it is why the
    # non-vacuity guard exists. N=300 is run at V=47,488 (158 verts/cell) instead,
    # where the dynamics stay live well past 40 steps.
    configs = [
        (100, 96, 100, False, "full-budget"),
        (224, 212, 300, True, "early-stop"),
    ]
    for nt, nphi, N, early, tag in configs:
        mesh = build(nt, nphi)
        cfg = MBOConfig(
            seed=84172851,
            max_iters=40,
            churn_tol=0.0,
            early_stop_in_loop=early,
            finalize_full_budget=False,
        )
        labels, _ = geodesic_balanced_init(mesh, N, cfg.seed)
        labels, rep = mbo_level(mesh, labels, N, cfg, level=0, run_probe=False)

        active = [s for s in rep.steps if s.churn > 0]
        worst_viol = max((s.violation_rel for s in active), default=float("nan"))
        worst_slack = max((s.slack_rel for s in active), default=0.0)
        e = [s.E_lumped for s in rep.steps]
        incs = [i for i in range(1, len(e)) if e[i] > e[i - 1]]
        worst_inc = max([(e[i] - e[i - 1]) / abs(e[i - 1]) for i in incs], default=0.0)

        # The excursion ceiling is tied to the run's OWN measured slack, floored at
        # the pre-registered 2e-3. P10(c) asks whether an excursion exceeds what
        # DUAL SLACK can produce, and a fixed constant was only ever a proxy for
        # that: the pre-registered 2e-3 was set as 1.5x a 1.45e-3 slack measured
        # over four configurations, and the very first new configuration run here
        # reached 1.868e-3 -- so a fixed bar was already within 1.07x of the
        # mechanism's own output, the same defect that sank the 1e-9 before it.
        # Amended BEFORE any scored run, and disclosed rather than silently raised.
        ceiling = max(2e-3, 1.5 * worst_slack)
        binds = ceiling > 2e-3
        print(
            f"  N={N} {tag}: {len(rep.steps)} steps, {len(active)} ACTIVE, "
            f"worst violation={worst_viol:.3e}, max slack={worst_slack:.3e}"
        )
        print(
            f"        E_lumped increases: {len(incs)}/{max(1,len(active))} active, "
            f"worst +{worst_inc:.3e} (ceiling {ceiling:.3e}"
            f"{' = 1.5x this run slack' if binds else ' = pre-registered floor'})"
        )
        # Non-vacuity: three of four probe runs froze early, which would have let a
        # broken gate pass on a flat tail. Report 07's failure shape, applied here.
        ok &= verdict(
            len(active) >= 20,
            f"non-vacuous: >=20 active steps (N={N})",
            f"got {len(active)}",
        )
        ok &= verdict(
            worst_viol <= 1e-12,
            f"inequality holds on all active steps (N={N})",
            f"worst {worst_viol:.3e}",
        )
        ok &= verdict(
            worst_inc <= ceiling,
            f"E_tau excursions within the slack ceiling (N={N})",
            f"worst +{worst_inc:.3e} vs {ceiling:.3e}",
        )
        ok &= verdict(
            len(incs) < 0.25 * max(1, len(active)),
            f"excursions on <25% of active steps (N={N})",
            f"{len(incs)}/{len(active)}",
        )
    return ok


# ==========================================================================
# G5 -- harness round trip
# ==========================================================================


def gate5():
    banner("G5", "one-hot -> run_gates matches detect_area_imbalance directly")
    mesh, N = build(100, 96), 100
    labels, _ = geodesic_balanced_init(mesh, N, seed=84172851)
    dens = labels_to_one_hot(labels, N)
    rep = run_gates(dens, mesh, N)
    direct = detect_area_imbalance(dens, mesh.v, N)
    ok = verdict(
        rep.n_imbalanced == int(direct["n_imbalanced"])
        and abs(rep.worst_rel_dev - float(direct["worst_rel_dev"])) < 1e-15,
        "harness and direct call agree",
        f"({rep.n_imbalanced} cells, worst {rep.worst_rel_dev*100:.4f}%)",
    )
    ok &= verdict(
        np.array_equal(np.argmax(dens, axis=1), labels),
        "one-hot round trip preserves labels",
    )
    print(
        f"  dormant: {rep.n_dead} dead / {rep.n_weak} weak, min peak {rep.min_peak_density:.3f} "
        f"(vacuous for arms by construction)"
    )
    print(f"  connectivity: {rep.n_fragmented} fragmented")
    return ok


# ==========================================================================
# G6 -- early-stop disclosure
# ==========================================================================


def gate6():
    banner("G6", "in-loop worst-devs are tagged capped; the final assignment is not")
    mesh, N = build(100, 96), 100
    cfg = MBOConfig(
        seed=84172851, max_iters=25, early_stop_in_loop=True, finalize_full_budget=True
    )
    labels, _ = geodesic_balanced_init(mesh, N, cfg.seed)
    labels, rep = mbo_level(mesh, labels, N, cfg, level=0, run_probe=False)
    bar = assignment_quality_bar(mesh.v, N)
    inloop = [s.worst_rel_dev for s in rep.steps]
    print(
        f"  bar {bar*100:.4f}%   in-loop worst devs {min(inloop)*100:.4f}-{max(inloop)*100:.4f}% "
        f"(all tagged capped={all(s.early_stop_capped for s in rep.steps)})"
    )
    print(
        f"  final full-budget worst dev: {rep.final_worst_rel_dev*100:.4f}%  "
        f"capped={rep.final_early_stop_capped}"
    )
    ok = verdict(
        all(s.early_stop_capped for s in rep.steps),
        "every in-loop figure carries the capped tag",
    )
    ok &= verdict(
        not rep.final_early_stop_capped,
        "the reported final figure is NOT early-stop capped",
    )
    ok &= verdict(rep.final_worst_rel_dev <= bar, "final assignment reaches the bar")
    return ok


# ==========================================================================
# G7 -- tau-scale identity (invisible to G1, G2 and NC4 alike)
# ==========================================================================


def gate7():
    banner("G7", "eigenmode identity: (M + tau K)^-1 M phi = phi / (1 + tau lambda)")
    print("  A tau-SCALE error passes G1, G2 and NC4: mass conservation holds for ANY")
    print("  tau because 1^T K = 0, and G2 recomputes the same formula from the same")
    print("  inputs. Only an identity that depends on tau's value can catch it.\n")
    mesh, N = build(60, 56), 20
    cfg = MBOConfig()
    tau = tau_diagnostics(mesh, N, cfg)["tau"]
    solve = make_diffusion_solver(mesh, tau)
    lam, phi = eigsh(mesh.K.tocsc(), k=6, M=mesh.M.tocsc(), sigma=-1.0, which="LM")
    ok = True
    used = 0
    for j in range(len(lam)):
        if lam[j] < 1e-8:
            continue  # the constant mode is tau-blind by construction
        p = phi[:, j]
        y = solve(mesh.M @ p)
        err = float(np.abs(y - p / (1.0 + tau * lam[j])).max() / np.abs(p).max())
        print(f"  lambda={lam[j]:.6f}: rel err={err:.3e}")
        ok &= verdict(err < 1e-10, f"mode lambda={lam[j]:.4f} decays exactly")
        used += 1
        if used == 3:
            break
    ok &= verdict(used == 3, "three non-constant modes tested", f"got {used}")

    # Mutation: a 2x tau error must be caught by this identity.
    solve_bad = make_diffusion_solver(mesh, 2.0 * tau)
    j = int(np.argmax(lam))
    p = phi[:, j]
    err_bad = float(
        np.abs(solve_bad(mesh.M @ p) - p / (1.0 + tau * lam[j])).max() / np.abs(p).max()
    )
    ok &= verdict(
        err_bad > 1e-3,
        "mutation test: a 2x tau error is REJECTED",
        f"(rel err {err_bad:.3e})",
    )
    return ok


# ==========================================================================
# G8 -- psi / labels consistency in caller units
# ==========================================================================


def gate8():
    banner("G8", "solver's labels == argmax(scores + returned psi), in CALLER units")
    print("  G4's inequality is evaluated with the returned psi against the returned")
    print("  labels, so the normalize_scores rescale round trip is load-bearing.\n")
    mesh, N = build(100, 96), 100
    cfg = MBOConfig()
    tau = tau_diagnostics(mesh, N, cfg)["tau"]
    labels, _ = geodesic_balanced_init(mesh, N, seed=84172851)
    solve = make_diffusion_solver(mesh, tau)
    y, _ = diffuse_indicators(solve, mesh, labels, N)
    target = float(mesh.v.sum()) / N
    bar = assignment_quality_bar(mesh.v, N)
    ok = True
    for early in (True, False):
        psi, lab, _ = balanced_assign(y, mesh.v, target, bar, early_stop=early)
        recomputed = np.argmax(y + psi, axis=1)
        mism = int(np.count_nonzero(recomputed != lab))
        print(
            f"  early_stop={early}: mismatches {mism} / {len(lab)} "
            f"({mism/len(lab):.2e})"
        )
        ok &= verdict(
            mism / len(lab) <= 1e-6,
            f"labels reproduce from returned psi (early_stop={early})",
        )
    return ok


# ==========================================================================
# NC1 -- cold, unbalanced init must blow up the first assignment
# ==========================================================================


def nc1():
    banner("NC1", "cold unbalanced init: the first diffused assignment must be far off")
    print("  SCOPE: this verifies only the CONDITIONAL under which gate 2's B PASS was")
    print("  recorded. It is NOT evidence that the init matters for final perimeter --")
    print(
        "  cold-started B is on record recovering unaided (45.35% -> 1.72% by it 19)."
    )
    print("  The attribution question is SF-a, the init-only Phase 2 control.\n")
    mesh, N = build(224, 212), 300
    cfg = MBOConfig()
    tau = tau_diagnostics(mesh, N, cfg)["tau"]
    v, target = mesh.v, float(mesh.v.sum()) / N
    bar = assignment_quality_bar(v, N)
    solve = make_diffusion_solver(mesh, tau)

    sites = farthest_point_sampling(mesh.vertices, N, seed=84172851)
    d = dijkstra(edge_graph(mesh), indices=sites, directed=False).T
    cold = np.argmin(d, axis=1)  # raw Voronoi, NOT balanced
    areas = np.bincount(cold, weights=v, minlength=N)
    cold_dev = float(np.abs(areas - target).max() / target)

    y, _ = diffuse_indicators(solve, mesh, cold, N)
    scores_dev = float(
        np.abs(np.bincount(np.argmax(y, axis=1), weights=v, minlength=N) - target).max()
        / target
    )
    _, _, solved = balanced_assign(y, v, target, bar, early_stop=False)

    warm, info = geodesic_balanced_init(mesh, N, seed=84172851)
    y_w, _ = diffuse_indicators(solve, mesh, warm, N)
    _, _, solved_w = balanced_assign(y_w, v, target, bar, early_stop=False)

    print(f"  cold Voronoi labels      : worst area dev {cold_dev*100:7.2f}%")
    print(f"  cold, one diffusion, raw argmax(y): {scores_dev*100:7.2f}%")
    print(f"  cold, after balancing    : {solved*100:7.4f}%")
    print(f"  WARM (balanced init)     : {solved_w*100:7.4f}%   bar {bar*100:.4f}%")
    ok = verdict(
        scores_dev > 5 * bar,
        "cold start's diffused assignment is far above the bar",
        f"({scores_dev*100:.2f}% vs bar {bar*100:.3f}%)",
    )
    ok &= verdict(
        scores_dev > 5 * solved_w,
        "cold is materially worse than warm",
        f"({scores_dev/max(solved_w,1e-12):.1f}x)",
    )
    return ok


# ==========================================================================
# NC4 -- the wrong diffusion operator must break mass conservation
# ==========================================================================


def nc4():
    banner("NC4", "wrong operator (M + tau K) y = chi must FAIL mass conservation")
    mesh, N = build(100, 96), 100
    cfg = MBOConfig()
    tau = tau_diagnostics(mesh, N, cfg)["tau"]
    labels = np.random.default_rng(0).integers(0, N, len(mesh.vertices))
    chi = sparse_one_hot(labels, N)
    mass_chi = np.asarray(mesh.v @ chi.toarray())

    good = make_diffusion_solver(mesh, tau)
    rhs_good = (mesh.M @ chi).tocsc()
    y_good = np.column_stack([good(rhs_good[:, k].toarray().ravel()) for k in range(N)])
    err_good = float(
        np.abs((mesh.v @ y_good) - mass_chi).max() / np.abs(mass_chi).max()
    )

    solve = factorized((mesh.M + tau * mesh.K).tocsc())
    chi_d = chi.toarray()
    y_bad = np.column_stack([solve(chi_d[:, k]) for k in range(N)])
    err_bad = float(np.abs((mesh.v @ y_bad) - mass_chi).max() / np.abs(mass_chi).max())

    print(f"  correct  (M + tau K) y = M chi : mass rel err {err_good:.3e}")
    print(f"  WRONG    (M + tau K) y =   chi : mass rel err {err_bad:.3e}")
    ok = verdict(err_bad > 1e-3, "the wrong operator is REJECTED", f"({err_bad:.3e})")
    ok &= verdict(err_good < 1e-12, "the correct operator passes", f"({err_good:.3e})")
    return ok


# ==========================================================================
# NC3-CAL -- pin the pinning detector's threshold, or report it unusable
# ==========================================================================


def nc3_cal():
    banner("NC3-CAL", "calibrate the pinning detector on a frozen / converged pair")
    print("  A bare 'labels moved AND E dropped' test fires on essentially everything:")
    print("  a wider blur re-tiebreaks boundary vertices at any labelling. The frozen")
    print("  and healthy states must be separated by a THRESHOLD, pinned here, before")
    print("  any scored run -- and if no threshold separates them, NC3 is not a usable")
    print("  detector and this reports that instead of inventing one.\n")
    mesh, N = build(100, 96), 100
    states = {}
    for c, tag in [(2.0, "FROZEN (c=2)"), (4.0, "CONVERGED (c=4)")]:
        cfg = MBOConfig(seed=84172851, tau_c=c, max_iters=200)
        labels, _ = geodesic_balanced_init(mesh, N, cfg.seed)
        labels, rep = mbo_level(mesh, labels, N, cfg, level=0, run_probe=False)
        e = [s.E_lumped for s in rep.steps]
        tail = [abs(e[i] - e[i - 1]) for i in range(max(1, len(e) - 5), len(e))]
        tail_med = float(np.median(tail)) if tail else 0.0
        cfg_probe = MBOConfig(seed=84172851, tau_c=c)
        tau = tau_diagnostics(mesh, N, cfg_probe)["tau"]
        probe = tau_continuation_probe(mesh, labels, N, tau, cfg_probe, tail_med)
        states[tag] = (rep, probe, tail_med)
        print(
            f"  {tag}: {len(rep.steps)} steps, {rep.active_steps} active, "
            f"tail median |dE| = {tail_med:.3e}"
        )
        for p in probe["probes"]:
            print(
                f"      factor {p['factor']:.0f}x: moved {p['moved']:>5d} "
                f"({p['moved_frac']:.2e})  dE {p['dE']:+.4e}  "
                f"dE/tail {p['dE_over_tail']:+.2f}"
            )

    fr = states["FROZEN (c=2)"][1]
    cv = states["CONVERGED (c=4)"][1]
    fr_best = max(p["dE_over_tail"] for p in fr["probes"])
    cv_best = max(p["dE_over_tail"] for p in cv["probes"])
    fr_moved = max(p["moved_frac"] for p in fr["probes"])
    cv_moved = max(p["moved_frac"] for p in cv["probes"])

    print(f"\n  frozen    : best dE/tail {fr_best:+.3f}, moved_frac {fr_moved:.3e}")
    print(f"  converged : best dE/tail {cv_best:+.3f}, moved_frac {cv_moved:.3e}")

    separated = fr_best > 2.0 * cv_best and fr_best > 0
    if separated:
        K = float(np.sqrt(fr_best * max(cv_best, 1e-6)))
        f_min = float(np.sqrt(fr_moved * max(cv_moved, 1e-12)))
        print(f"\n  PINNED CONSTANTS: nc3_K = {K:.4f}   nc3_f_min = {f_min:.3e}")
        print(
            f"  margin: frozen/converged = {fr_best/max(cv_best,1e-9):.2f}x  (need >= 2x)"
        )
    else:
        print("\n  NO SEPARATING THRESHOLD FOUND on dE/tail.")
        print("  Per the pre-registration this is reported as-is: NC3 is not a usable")
        print("  pinning detector at these settings, and P8/P8' must say so.")
    return verdict(
        separated,
        "detector separates frozen from converged with >=2x margin",
        f"(frozen {fr_best:.2f} vs converged {cv_best:.2f})",
    )


# ==========================================================================
# NC2 -- c=2 at the finest N=100 level must be detected as pinned
# ==========================================================================


def nc2(nc3_K=None, nc3_f_min=None):
    banner("NC2", "c=2 freeze probe at the finest N=100 level")
    print("  Doubly motivated: it tests OUR non-freeze criterion in a regime the")
    print("  FFT/Cartesian literature never had to enter. Its DETECTOR is NC3, so it")
    print("  is sequenced after NC3-CAL and cannot fail until that lands.\n")
    if nc3_K is None:
        print("  SKIPPED: NC3-CAL did not produce constants.")
        return None
    mesh, N = build(348, 328), 100
    # max_anneals=0 is essential: mbo_level RESPONDS to a pinned verdict by
    # annealing tau and re-probing, so with the default 3 the level would fix
    # itself and the final probe would report pinned=False -- the test would
    # silently measure the remedy instead of the detector.
    cfg = MBOConfig(
        seed=84172851,
        tau_c=2.0,
        max_iters=200,
        max_anneals=0,
        nc3_K=nc3_K,
        nc3_f_min=nc3_f_min,
    )
    d = tau_diagnostics(mesh, N, cfg)
    print(
        f"  V={len(mesh.vertices)} sqrt(t)/h_max={d['sqrt_tau_over_h_max']:.3f} "
        f"sqrt(t)/R_cell={d['sqrt_tau_over_r_cell']:.3f} c_lo(h_mean)={d['c_lo_hmean']:.2f}"
    )
    labels, _ = geodesic_balanced_init(mesh, N, cfg.seed)
    t0 = time.perf_counter()
    labels, rep = mbo_level(mesh, labels, N, cfg, level=0, run_probe=True)
    print(
        f"  {len(rep.steps)} steps, {rep.active_steps} active, "
        f"{rep.n_anneals} anneals, {time.perf_counter()-t0:.0f}s"
    )
    for p in rep.probe.get("probes", []):
        print(
            f"      factor {p['factor']:.0f}x: moved_frac {p['moved_frac']:.2e}  "
            f"dE/tail {p['dE_over_tail']:+.2f}"
        )
    return verdict(
        bool(rep.probe.get("pinned")), "c=2 at the finest level is FLAGGED PINNED"
    )


# ==========================================================================
# NC5 -- the over-merge discriminator: c-sweep at ONE fixed level
# ==========================================================================


def nc5():
    banner("NC5", "over-merge discriminator: c in {2,4,8} at N=300, V=47,488")
    print(
        "  Assignment quality CANNOT see over-merging -- the balanced assignment hits"
    )
    print("  its area target by construction. The instruments are fragmentation, cell")
    print("  compactness, and core loss. If c=8 is indistinguishable from c=4 on all")
    print("  three, the pre-registration says to report that no instrument here")
    print("  distinguishes over-merging, not that the risk is absent.\n")
    mesh, N = build(224, 212), 300
    rows = []
    for c in (2.0, 4.0, 8.0):
        # rho is lifted so the over-merge CAP does not clamp c=8 back to the
        # window -- the whole point of this probe is to enter that regime on
        # purpose and see whether any instrument notices.
        cfg = MBOConfig(seed=84172851, tau_c=c, rho=99.0, max_iters=60)
        d = tau_diagnostics(mesh, N, cfg)
        labels, _ = geodesic_balanced_init(mesh, N, cfg.seed)
        labels, rep = mbo_level(mesh, labels, N, cfg, level=0, run_probe=False)
        gates = run_gates(labels_to_one_hot(labels, N), mesh, N)
        rows.append(
            {
                "c": c,
                "st_rc": d["sqrt_tau_over_r_cell"],
                "frag": gates.n_fragmented,
                "q_mean": rep.geometry["q_mean"],
                "q_worst": rep.geometry["q_worst"],
                "core_loss": rep.geometry["core_loss"],
                "lbl": rep.label_boundary_length,
            }
        )
        print(
            f"  c={c:.0f}  sqrt(t)/R_cell={d['sqrt_tau_over_r_cell']:.3f}  "
            f"fragmented={gates.n_fragmented:>3d}  Q_mean={rep.geometry['q_mean']:.4f}  "
            f"Q_worst={rep.geometry['q_worst']:.3f}  core_loss={rep.geometry['core_loss']:>3d}  "
            f"boundary={rep.label_boundary_length:.2f}"
        )

    c4, c8 = rows[1], rows[2]
    sep = {
        "fragmentation": c8["frag"] > c4["frag"],
        "compactness Q_mean": c8["q_mean"] > c4["q_mean"] * 1.02,
        "core loss": c8["core_loss"] > c4["core_loss"],
    }
    print("\n  c=8 vs c=4, per instrument:")
    for k, val in sep.items():
        print(f"     {k:22s}: {'SEPARATES' if val else 'no difference'}")
    ok = verdict(
        any(sep.values()),
        "at least one instrument separates c=8 from c=4",
        "" if any(sep.values()) else "-> report: no instrument here sees over-merge",
    )
    mono = rows[0]["q_mean"] <= rows[1]["q_mean"] <= rows[2]["q_mean"]
    print(
        f"  Q_mean monotone in c: {mono} "
        f"({rows[0]['q_mean']:.4f} -> {rows[1]['q_mean']:.4f} -> {rows[2]['q_mean']:.4f})"
    )
    return ok


def nc5b():
    """POST-HOC, added 2026-08-19 after NC5, and labelled as post-hoc.

    NC5 ran at V=47,488 / N=300 -- **158 verts per cell** -- and found no
    over-merge signature anywhere up to sqrt(tau)/R_cell = 1.34. But the rho = 1.0
    cap that the pre-registration pins binds on exactly ONE level in current use,
    N=300 level 0, which is **32 verts per cell**: a different regime entirely.
    So NC5 leaves the cap untested precisely where it acts.

    This is not fishing after a failed test -- the verdict of NC5 stands as
    recorded either way. It closes a scope limit that NC5's own design created,
    and it runs BEFORE any scored run so it cannot be tuned against a result.
    """
    banner("NC5b", "over-merge at the coarse level where the rho cap ACTUALLY binds")
    print("  N=300 at V=9,600 is 32 verts/cell -- the level, and the only level, where")
    print("  tau = min((4h)^2, R_cell^2) clamps. NC5 tested 158 verts/cell and could")
    print("  not have seen a failure here.\n")
    mesh, N = build(100, 96), 300
    rows = []
    for c in (2.0, 4.0, 8.0):
        cfg = MBOConfig(seed=84172851, tau_c=c, rho=99.0, max_iters=60)
        d = tau_diagnostics(mesh, N, cfg)
        labels, _ = geodesic_balanced_init(mesh, N, cfg.seed)
        labels, rep = mbo_level(mesh, labels, N, cfg, level=0, run_probe=False)
        gates = run_gates(labels_to_one_hot(labels, N), mesh, N)
        rows.append((c, d, gates, rep))
        print(
            f"  c={c:.0f}  sqrt(t)/R_cell={d['sqrt_tau_over_r_cell']:.3f}  "
            f"fragmented={gates.n_fragmented:>3d}  Q_mean={rep.geometry['q_mean']:.4f}  "
            f"Q_worst={rep.geometry['q_worst']:.3f}  core_loss={rep.geometry['core_loss']:>3d}  "
            f"boundary={rep.label_boundary_length:.2f}"
        )
    c4, c8 = rows[1], rows[2]
    sep = (
        c8[2].n_fragmented > c4[2].n_fragmented
        or c8[3].geometry["q_mean"] > c4[3].geometry["q_mean"] * 1.02
        or c8[3].geometry["core_loss"] > c4[3].geometry["core_loss"]
    )
    print(
        f"\n  c=8 (sqrt(t)/R_cell={c8[1]['sqrt_tau_over_r_cell']:.2f}) vs c=4: "
        f"{'SEPARATES' if sep else 'no difference'}"
    )
    print("  Reported either way; a null here means the cap is unverified, NOT that")
    print("  over-merging is absent -- no instrument in this plan can see it.")
    return sep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--negative-controls", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument(
        "--only", type=str, default=None, help="comma-separated subset, e.g. G4,NC5,NC2"
    )
    # NC2's detector IS NC3, so it can only run once NC3-CAL has pinned these.
    # Passing them on the command line keeps the calibration and its use in
    # separate, individually auditable steps rather than one opaque run.
    ap.add_argument("--nc2-k", type=float, default=None)
    ap.add_argument("--nc2-f-min", type=float, default=None)
    args = ap.parse_args()

    gates = {
        "G1": gate1,
        "G2": gate2,
        "G3": gate3,
        "G4": gate4,
        "G5": gate5,
        "G6": gate6,
        "G7": gate7,
        "G8": gate8,
    }
    ncs = {"NC1": nc1, "NC4": nc4, "NC5": nc5}

    run_gates_ = not args.negative_controls or args.all
    results = {}
    t0 = time.perf_counter()

    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        for name, fn in {**gates, **ncs}.items():
            if name in wanted:
                results[name] = fn()
        if "NC3-CAL" in wanted:
            results["NC3-CAL"] = nc3_cal()
        if "NC5b" in wanted:
            results["NC5b"] = nc5b()
        if "NC2" in wanted:
            results["NC2"] = nc2(args.nc2_k, args.nc2_f_min)
    else:
        if run_gates_:
            for name, fn in gates.items():
                results[name] = fn()
        if args.negative_controls or args.all:
            for name, fn in ncs.items():
                results[name] = fn()
            results["NC3-CAL"] = nc3_cal()

    banner("SUMMARY", f"{time.perf_counter()-t0:.0f}s")
    for k, v in results.items():
        print(f"  {k:10s} {'PASS' if v else ('SKIP' if v is None else 'FAIL')}")
    failed = [k for k, v in results.items() if v is False]
    print(f"\n{'ALL PASS' if not failed else 'FAILED: ' + ', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
