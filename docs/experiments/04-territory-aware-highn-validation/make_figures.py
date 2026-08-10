#!/usr/bin/env python3
"""Figures for report 04 — territory-aware Phase 1 high-N validation.

Produces two vector figures:
  fig_balance_by_level.pdf  worst |area deviation| at each level's start/end vs the
                            control, showing the mechanism drives the bad-seed runt
                            from -34% (control) to 0 imbalanced cells (worst ~2%).
  fig_trajectory.pdf        worst |dev| vs PGD iteration within levels 0 and 1 --
                            level 0 flatlines at ~10% (floor-limited, above the 5%
                            gate); level 1 (finer mesh) descends steeply and crosses.

The trajectory arrays below are the MEASUREMENT extracted from the source run's
per-level trace snapshots (run_20260717_102306, config
torus_200part_coarse_seeded_lam9_territory_test.yaml) by rebuilding each level's
torus mesh (TorusMeshProvider), taking the winner-take-all argmax, assigning lumped
mass v_i, and comparing per-cell area to the equal-area target sum(v)/N -- the same
computation as detect_area_imbalance. They are embedded (not re-read from the ~4 GB
of traces) so the figures regenerate without the raw run present; see the provenance
block in main.tex for the run identifiers.

Run from the repo root:
    python docs/experiments/04-territory-aware-highn-validation/make_figures.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
GATE = 5.0  # detect_area_imbalance relative threshold, percent

# --- extracted measurement: max|dev|% (and n_imbalanced) per iteration -------
# level 0: V=9600 (48 verts/cell), ran to the 30k cap
L0_it = [0, 500, 3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 27000, 29999]
L0_md = [54.00, 41.39, 23.04, 50.07, 38.20, 38.64, 32.42, 22.86, 14.23, 11.21, 11.00, 10.06]
L0_ni = [149, 126, 72, 53, 35, 29, 23, 21, 16, 14, 5, 4]
# level 1: V=24948 (124 verts/cell), ran to the 30k cap
L1_it = [0, 500, 1000, 1500, 2000, 3000, 4000, 6000, 9000, 12000, 18000, 24000, 29999]
L1_md = [17.94, 11.52, 10.50, 10.27, 9.66, 7.77, 7.77, 5.65, 4.23, 4.23, 2.95, 1.66, 2.11]
L1_ni = [51, 8, 5, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0]

# per-level worst |dev|% at start / end (level 2 died at iter 7500), + control
LEVELS = ["Level 0\n(V=9.6k)", "Level 1\n(V=24.9k)", "Level 2\n(V=47.5k)"]
worst_start = [54.0, 17.9, 4.6]
worst_end = [10.1, 2.1, 2.4]   # level 2 = last snapshot (iter 7500)
CONTROL = 34.2                  # run_20260712_224424, same seed/lambda, original energy


def fig_balance_by_level():
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    x = np.arange(len(LEVELS))
    w = 0.38
    ax.bar(x - w / 2, worst_start, w, label="level start (init / post-interp)",
           color="#c44e52", alpha=0.85)
    ax.bar(x + w / 2, worst_end, w, label="level end", color="#4c72b0")
    ax.axhline(GATE, ls="--", lw=1.3, color="black")
    ax.text(len(LEVELS) - 0.5, GATE + 0.8, "5% gate", ha="right", fontsize=9)
    ax.axhline(CONTROL, ls=":", lw=1.4, color="#8172b3")
    ax.text(0.0, CONTROL + 0.8, f"control (orig. energy): {CONTROL:.0f}%",
            fontsize=9, color="#5b4a8a")
    for xi, (s, e) in enumerate(zip(worst_start, worst_end)):
        ax.text(xi + w / 2, e + 0.6, f"{e:.1f}%", ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(LEVELS)
    ax.set_ylabel("worst-cell area deviation  |dev|  (%)")
    ax.set_title("Discrete-area balance per level  (N=200, bad seed 84172851, "
                 "$\\lambda$=9)", fontsize=10.5)
    ax.set_ylim(0, 58)
    ax.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_balance_by_level.pdf"))
    plt.close(fig)


def fig_trajectory():
    fig, ax1 = plt.subplots(figsize=(6.6, 3.9))
    ax1.plot(L0_it, L0_md, "-o", ms=3.5, color="#c44e52",
             label="Level 0  (48 verts/cell) — floor-limited")
    ax1.plot(L1_it, L1_md, "-s", ms=3.5, color="#4c72b0",
             label="Level 1  (124 verts/cell) — crosses gate")
    ax1.axhline(GATE, ls="--", lw=1.3, color="black")
    ax1.text(29800, GATE + 0.7, "5% gate", ha="right", fontsize=9)
    ax1.annotate("level 0 flatlines ~10%\n(0.23%/1000it — noise)",
                 xy=(29999, 10.06), xytext=(20000, 30),
                 fontsize=8.5, color="#8a2f34",
                 arrowprops=dict(arrowstyle="->", color="#8a2f34", lw=1))
    ax1.annotate("level 1 steep:\n2.2%/1000it,\n0 imbalanced by ~9k",
                 xy=(9000, 4.23), xytext=(11000, 22),
                 fontsize=8.5, color="#2f3f6a",
                 arrowprops=dict(arrowstyle="->", color="#2f3f6a", lw=1))
    ax1.set_xlabel("PGD iteration within level")
    ax1.set_ylabel("worst-cell area deviation  |dev|  (%)")
    ax1.set_title("Refinement is necessary: the coarse level cannot cross the gate",
                  fontsize=10.5)
    ax1.set_xlim(-800, 30800)
    ax1.set_ylim(0, 58)
    ax1.legend(fontsize=8.5, loc="upper center")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_trajectory.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_balance_by_level()
    fig_trajectory()
    print("wrote fig_balance_by_level.pdf, fig_trajectory.pdf to", HERE)
