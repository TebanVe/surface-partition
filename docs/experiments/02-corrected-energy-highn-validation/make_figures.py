#!/usr/bin/env python3
"""Regenerate the figures for docs/experiments/02-corrected-energy-highn-validation.

Measured study: validating the corrected Phase 1 double-well discretization
(interface_vec = u*(1-u); see docs/reference/phase1_energy_discretization_bug.md and
docs/math/06-phase1-energy-discretization/). Shows that (1) correcting the energy
alone does not cure the high-N "runt" of report 01, but corrected energy + a moderate
lambda_penalty=5.1 does (worst-cell 22.5% -> 0.8%); (2) Phase 2 then DECREASES
perimeter (214.34 -> 185.25, -13.6%) and is feasible (equal-area violation ~1e-10)
where the buggy energy made it rise and crash on infeasibility; (3) on the corrected
(steeper) well random init traps in the symmetric state while seeded init stays clean.

Run from the repo root:
    python docs/experiments/02-corrected-energy-highn-validation/make_figures.py

Produces (vector PDF, embedded by main.tex):
    fig_runt_resolution.pdf, fig_phase2_convergence.pdf, fig_init_trap.pdf

Provenance (runs under results/, seed 84172851):
    buggy   N=100 lam2.1 5-level 348x328   run_20260629_141012   (report 01 anchor)
    corr    N=100 lam2.1 3-level 224x212   run_20260707_192154
    corr    N=100 lam5.1 5-level 348x328   run_20260709_081548   (production)
    corr    N=30  random  lam2.1           run_20260707_002828
    corr    N=30  seeded  lam2.1           run_20260707_080824
The winner-take-all area = lumped P1 mass assigned to each vertex's argmax cell; the
worst-cell deviations reproduce the metadata area_imbalance blocks (22.5/21.8/0.8 %
for N=100; 43.2/0.7 % for N=30 random/seeded). The Phase-2 perimeter trajectory is
read from the production campaign's iteration_*.h5 (final_perimeter attr), prepended
with the Phase-1 extracted perimeter (P1_EXTRACTED below); the per-iteration
equal-area violation is max|optimization_info/constraint_violations|.
"""
import os
import sys
import glob
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())

HERE = os.path.dirname(os.path.abspath(__file__))
RES = "results"

# Okabe-Ito (colorblind-safe by construction)
BLUE = "#0072B2"; ORANGE = "#E69F00"; VERM = "#D55E00"; GREEN = "#009E73"; GREY = "#999999"
plt.rcParams.update({
    "font.size": 11, "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.edgecolor": "#666666", "axes.linewidth": 0.8,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})

# The Phase-1 extracted perimeter of the production run (iteration-0 start), from its
# refinement.log ("Perimeter at start of iteration: 214.3413402222") and its
# solution/metadata.yaml (initial_perimeter: 214.3413402221608).
P1_EXTRACTED = 214.3413402222
# Report 01's buggy N=100-coarse Phase 2 (Table 1): perimeter rose 217 -> 229.
BUGGY_P2 = (217.0, 229.0)

RUNS = {
    "buggy_l21":  "run_20260629_141012_surftorus_npart100_v1nt100-348_incr62_v2np96-328_incr58_lam2.1_seed84172851",
    "corr_l21":   "run_20260707_192154_surftorus_npart100_v1nt100-224_incr62_v2np96-212_incr58_lam2.1_seed84172851",
    "corr_l51":   "run_20260709_081548_surftorus_npart100_v1nt100-348_incr62_v2np96-328_incr58_lam5.1_seed84172851",
    "n30_random": "run_20260707_002828_surftorus_npart30_v1nt60-184_incr62_v2np46-162_incr58_lam2.1_seed84172851",
    "n30_seeded": "run_20260707_080824_surftorus_npart30_v1nt60-184_incr62_v2np46-162_incr58_lam2.1_seed84172851",
}


def lumped(V, F):
    """Lumped P1 mass per vertex (1/3 sum of incident triangle areas)."""
    tri = V[F]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    v = np.zeros(len(V))
    for j in range(3):
        np.add.at(v, F[:, j], area / 3.0)
    return v


def final_dev(key):
    """Sorted per-cell (territory - target)/target for a run's final solution."""
    d = os.path.join(RES, RUNS[key])
    sol = glob.glob(f"{d}/solution/surface_part*_*.h5")[0]
    with h5py.File(sol, "r") as f:
        V = f["vertices"][:]; F = f["faces"][:].astype(np.int64); x = f["x_opt"][:]
        N = int(f.attrs["n_partitions"])
    u = x.reshape(V.shape[0], N); v = lumped(V, F); tgt = v.sum() / N
    win = np.argmax(u, 1); a = np.zeros(N); np.add.at(a, win, v)
    dev = (a - tgt) / tgt
    return np.sort(dev) * 100.0, np.abs(dev).max() * 100.0, int((np.abs(dev) > 0.05).sum())


def phase2_trajectory():
    """Per-iteration (perimeter, max equal-area violation) for the production campaign."""
    its = glob.glob(os.path.join(RES, RUNS["corr_l51"], "refinement", "*", "iteration_*.h5"))
    rows = []
    for p in its:
        with h5py.File(p, "r") as f:
            it = int(f.attrs["iteration_number"])
            per = float(f.attrs["final_perimeter"])
            cv = f["optimization_info"]["constraint_violations"][:]
            rows.append((it, per, float(np.abs(cv).max())))
    rows.sort()
    its_n = np.array([0] + [r[0] for r in rows])
    per = np.array([P1_EXTRACTED] + [r[1] for r in rows])
    viol = np.array([r[2] for r in rows])           # per iteration (>=1)
    viol_it = np.array([r[0] for r in rows])
    return its_n, per, viol_it, viol


def main():
    print("reading solutions...", flush=True)
    dev_buggy, w_buggy, n_buggy = final_dev("buggy_l21")
    dev_c21, w_c21, n_c21 = final_dev("corr_l21")
    dev_c51, w_c51, n_c51 = final_dev("corr_l51")
    dev_rand, w_rand, n_rand = final_dev("n30_random")
    dev_seed, w_seed, n_seed = final_dev("n30_seeded")
    its_n, per, viol_it, viol = phase2_trajectory()
    red_pct = (per[0] - per[-1]) / per[0] * 100.0

    # ---------------------------------------------------------------
    # FIGURE 1 -- runt resolution: worst-cell bars + final distribution
    # ---------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.3))
    labels = ["buggy well\n$\\lambda$=2.1", "corrected\n$\\lambda$=2.1", "corrected\n$\\lambda$=5.1"]
    vals = [w_buggy, w_c21, w_c51]; cols = [VERM, ORANGE, GREEN]
    bars = ax1.bar(range(3), vals, color=cols, width=0.62)
    for i, (b, val) in enumerate(zip(bars, vals)):
        ax1.annotate(f"{val:.1f}%", (b.get_x() + b.get_width() / 2, val),
                     textcoords="offset points", xytext=(0, 3), ha="center",
                     fontsize=10, color=cols[i], fontweight="bold")
    ax1.axhline(5, color=GREY, ls="--", lw=1)
    ax1.annotate("5% imbalance gate", (2.35, 5), textcoords="offset points",
                 xytext=(0, 3), color=GREY, fontsize=8, ha="right")
    ax1.set_xticks(range(3)); ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("worst-cell area error  (% of target)")
    ax1.set_title("Energy fix alone isn't enough;\nfix + moderate $\\lambda$ cures the runt", fontsize=11)
    ax1.set_ylim(0, 26)

    ax2.plot(np.arange(len(dev_buggy)), dev_buggy, "-o", color=VERM, lw=1.6, ms=4,
             label=f"buggy $\\lambda$=2.1 (worst {w_buggy:.0f}%)")
    ax2.plot(np.arange(len(dev_c51)), dev_c51, "-o", color=GREEN, lw=1.6, ms=4,
             label=f"corrected $\\lambda$=5.1 (worst {w_c51:.1f}%)")
    ax2.axhline(0, color="#444444", lw=1)
    ax2.axhline(-5, color=GREY, ls="--", lw=1); ax2.axhline(5, color=GREY, ls="--", lw=1)
    ax2.annotate(f"{dev_buggy[0]:.0f}%\nrunt", (0, dev_buggy[0]), textcoords="offset points",
                 xytext=(10, 2), color=VERM, fontsize=9)
    ax2.set_xlabel("cell rank (sorted by area deviation)")
    ax2.set_ylabel("area deviation (% of target)")
    ax2.set_title("Final N=100 partition:\nproduction run is uniformly equal", fontsize=11)
    ax2.legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("The corrected energy plus a moderate $\\lambda$ resolves the high-N runt (N=100 torus)",
                 fontsize=12, y=1.02)
    fig.savefig(os.path.join(HERE, "fig_runt_resolution.pdf")); plt.close(fig)

    # ---------------------------------------------------------------
    # FIGURE 2 -- Phase 2 now decreases perimeter and is feasible
    # ---------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.3))
    ax1.plot(its_n, per, "-o", color=GREEN, lw=2, ms=5, label="corrected $\\lambda$=5.1 (production)")
    ax1.plot([0, 1], list(BUGGY_P2), "--s", color=VERM, lw=1.8, ms=6,
             label="buggy $\\lambda$=2.1 (report 01)")
    ax1.annotate(f"extracted\n{per[0]:.1f}", (0, per[0]), textcoords="offset points",
                 xytext=(8, -2), fontsize=9, color=GREEN)
    ax1.annotate(f"{per[-1]:.1f}\n(${-red_pct:.1f}\\%$)", (its_n[-1], per[-1]),
                 textcoords="offset points", xytext=(-2, 10), ha="right", fontsize=9,
                 color=GREEN, fontweight="bold")
    ax1.annotate("rises → infeasibility", (1, BUGGY_P2[1]), textcoords="offset points",
                 xytext=(6, 0), fontsize=9, color=VERM, va="center")
    ax1.set_xlabel("Phase 2 iteration"); ax1.set_ylabel("total perimeter")
    ax1.set_title("Phase 2 now DECREASES perimeter\n($-13.6\\%$), not rises", fontsize=11)
    ax1.legend(frameon=False, fontsize=9, loc="center right")

    ax2.semilogy(viol_it, np.maximum(viol, 1e-13), "-o", color=GREEN, lw=2, ms=5)
    ax2.axhspan(0.05, 0.16, color=VERM, alpha=0.15)
    ax2.annotate("buggy N=100: iter-0 violation\n0.05–0.16 (never resolves)",
                 (viol_it.mean(), 0.09), fontsize=8.5, color=VERM, ha="center", va="center")
    ax2.annotate("corrected $\\lambda$=5.1", (viol_it[-1], viol[-1]),
                 textcoords="offset points", xytext=(-4, 10), ha="right", fontsize=9, color=GREEN)
    ax2.set_xlabel("Phase 2 iteration")
    ax2.set_ylabel("max equal-area constraint violation")
    ax2.set_title("Phase 2 is FEASIBLE\n(violation $\\leq 3\\times10^{-6}$)", fontsize=11)
    ax2.set_ylim(1e-12, 1e0)
    fig.suptitle("Phase 2 on the corrected N=100 partition: perimeter falls and the equal-area constraint holds",
                 fontsize=12, y=1.02)
    fig.savefig(os.path.join(HERE, "fig_phase2_convergence.pdf")); plt.close(fig)

    # ---------------------------------------------------------------
    # FIGURE 3 -- the corrected-energy random-init trap (N=30)
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    ax.plot(np.arange(len(dev_rand)), dev_rand, "-o", color=VERM, lw=1.8, ms=5,
            label=f"random init (worst {w_rand:.0f}%, {n_rand} cells $>$5%)")
    ax.plot(np.arange(len(dev_seed)), dev_seed, "-o", color=GREEN, lw=1.8, ms=5,
            label=f"seeded init (worst {w_seed:.1f}%, {n_seed} cells $>$5%)")
    ax.axhline(0, color="#444444", lw=1)
    ax.axhline(-5, color=GREY, ls="--", lw=1); ax.axhline(5, color=GREY, ls="--", lw=1)
    ax.annotate("random init: frozen at the\nsymmetric state (trapped)",
                (1.0, 26), fontsize=9, color=VERM, ha="left", va="center")
    ax.set_xlabel("cell rank (sorted by area deviation)")
    ax.set_ylabel("area deviation (% of target)")
    ax.set_title("Corrected (steeper) energy: random init traps, seeded init escapes (N=30, $\\lambda$=2.1)",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.savefig(os.path.join(HERE, "fig_init_trap.pdf")); plt.close(fig)

    # ---------------------------------------------------------------
    print("RUNT (worst-cell % / #cells>5%):")
    print(f"  buggy  N=100 lam2.1 : {w_buggy:5.1f}%  ({n_buggy})")
    print(f"  corr   N=100 lam2.1 : {w_c21:5.1f}%  ({n_c21})")
    print(f"  corr   N=100 lam5.1 : {w_c51:5.1f}%  ({n_c51})   <- production")
    print(f"PHASE 2 perimeter: {per[0]:.2f} -> {per[-1]:.4f}  ({-red_pct:.1f}%)  over {len(per)-1} iters")
    print(f"PHASE 2 max equal-area violation: {viol.max():.2e} (all iters)")
    print(f"N=30 init: random worst {w_rand:.1f}% ({n_rand} imbalanced) | seeded worst {w_seed:.1f}% ({n_seed})")
    print("figures:", sorted(os.path.basename(p) for p in glob.glob(os.path.join(HERE, "*.pdf"))))
    print("DONE")


if __name__ == "__main__":
    main()
