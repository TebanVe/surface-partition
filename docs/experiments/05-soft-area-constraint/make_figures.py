#!/usr/bin/env python3
"""Figures for report 05 — the soft continuous equal-area constraint (A2).

Produces two vector figures:
  fig_two_defects.pdf   the two independent failure mechanisms, side by side.
                        Left: at level 0 the fragmented-cell count rises 0->4
                        exactly as the imbalanced-cell count falls 79->0 -- the
                        flow buys area balance with disconnected territory.
                        Right: the Armijo step size collapses from 2.5e-1 to
                        9.095e-13 between iterations 2,500 and 2,750 and the
                        energy freezes to the digit, while the control's step
                        decays gracefully over 30,000 iterations and never
                        collapses.
  fig_churn.pdf         winner-take-all label churn at level 0 from the
                        identical seeded x0: the control decays monotonically,
                        the soft-area arm does not, and once its line search
                        dies the churn is not small but exactly zero.

The arrays below are the MEASUREMENT. Fragmentation/imbalance come from
rebuilding the level-0 torus mesh (TorusMeshProvider), taking the winner-take-all
argmax of each saved trace iterate and running detect_disconnected_cells /
detect_area_imbalance on it; churn is the count of vertices whose argmax differs
between consecutive saved iterates; step and energy are columns 9 and 4 of the
PGD summary trace. They are embedded rather than re-read so the figures
regenerate without the raw runs present -- see the provenance block in main.tex
for the run identifiers and how to recompute them.

Run from the repo root:
    python docs/experiments/05-soft-area-constraint/make_figures.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
V0 = 9600  # level-0 vertices, N=100

# --- measurement: A2 level 0, triggers disabled, run to 10,000 iterations ----
# (scratch run from parameters/torus_100part_coarse_seeded_softarea.yaml with
#  refinement_levels=1, max_iter=10000, enable_refinement_triggers=false)
FRAG_IT = [250, 500, 750, 1000, 1500, 2000, 2500, 2750, 3000, 4000, 6000, 9999]
FRAG_N = [0, 1, 1, 1, 2, 2, 3, 4, 4, 4, 4, 4]
IMBAL_N = [79, 63, 52, 37, 22, 16, 1, 0, 0, 0, 0, 0]

# --- measurement: Armijo step size, level 0 ---------------------------------
A2_STEP_IT = list(range(0, 5000, 250))
A2_STEP = [
    5.0000e-01, 1.5625e-02, 6.2500e-02, 6.2500e-02, 6.2500e-02, 6.2500e-02,
    6.2500e-02, 1.2500e-01, 1.2500e-01, 1.2500e-01, 2.5000e-01, 9.0949e-13,
    9.0949e-13, 9.0949e-13, 9.0949e-13, 9.0949e-13, 9.0949e-13, 9.0949e-13,
    9.0949e-13, 9.0949e-13,
]
CTRL_STEP_IT = list(range(0, 12000, 1000))
CTRL_STEP = [
    1.0000e00, 1.2500e-01, 1.2500e-01, 1.2500e-01, 1.5625e-02, 7.8125e-03,
    3.9062e-03, 1.9531e-03, 1.9531e-03, 9.7656e-04, 9.7656e-04, 9.7656e-04,
]

# --- measurement: label churn at level 0, both at trace stride 500 ----------
# vertices whose argmax changed since the previous sample, as a % of V.
CTRL_CHURN_IT = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000,
                 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000]
CTRL_CHURN = [12.4167, 3.9375, 2.9479, 2.1146, 1.4583, 1.1042, 0.5104, 0.2292,
              0.1146, 0.0729, 0.0417, 0.0104, 0.0104, 0.0000, 0.0000, 0.0104,
              0.0208, 0.0000, 0.0104, 0.0208]
A2_CHURN_IT = [500, 1000, 1500, 2000, 2500]
A2_CHURN = [2.7708, 5.1354, 3.7500, 2.9167, 3.6667]
# the probe (triggers off, stride 250) continues past the trigger point:
A2_PROBE_IT = [2750, 3000, 3500, 4000, 5000, 6000, 8000, 9999]
A2_PROBE_CHURN = [1.0833, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

C_CTRL = "#1f77b4"
C_A2 = "#d62728"
C_AUX = "#7f7f7f"


def fig_two_defects():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.5))

    ax1.plot(FRAG_IT, FRAG_N, "o-", color=C_A2, lw=1.8, ms=4,
             label="fragmented cells")
    ax1.set_xlabel("PGD iteration (level 0)")
    ax1.set_ylabel("fragmented cells", color=C_A2)
    ax1.tick_params(axis="y", labelcolor=C_A2)
    ax1.set_ylim(-0.3, 5)
    ax1b = ax1.twinx()
    ax1b.plot(FRAG_IT, IMBAL_N, "s--", color=C_CTRL, lw=1.6, ms=4,
              label="imbalanced cells")
    ax1b.set_ylabel("imbalanced cells (>5% off target)", color=C_CTRL)
    ax1b.tick_params(axis="y", labelcolor=C_CTRL)
    ax1.set_title("(a) balance is bought with disconnection", fontsize=10)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")

    ax2.semilogy(CTRL_STEP_IT, CTRL_STEP, "o-", color=C_CTRL, lw=1.8, ms=4,
                 label="exact projection (control)")
    ax2.semilogy(A2_STEP_IT, A2_STEP, "o-", color=C_A2, lw=1.8, ms=4,
                 label="soft area (A2)")
    ax2.axhline(1e-12, color=C_AUX, ls=":", lw=1.2)
    ax2.annotate("line-search abort floor", xy=(3000, 1.4e-12), fontsize=7.5,
                 color=C_AUX)
    ax2.annotate("energy frozen from here\n(119.5651878371, 7,000 iters)",
                 xy=(2750, 9.1e-13), xytext=(1500, 2e-9), fontsize=7.5,
                 color=C_A2,
                 arrowprops=dict(arrowstyle="->", color=C_A2, lw=0.9))
    ax2.set_xlabel("PGD iteration (level 0)")
    ax2.set_ylabel("accepted Armijo step")
    ax2.set_title("(b) the line search collapses", fontsize=10)
    ax2.legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    out = os.path.join(HERE, "fig_two_defects.pdf")
    fig.savefig(out)
    print("wrote", out)


def fig_churn():
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.semilogy(CTRL_CHURN_IT, [max(c, 5e-3) for c in CTRL_CHURN], "o-",
                color=C_CTRL, lw=1.8, ms=4,
                label="exact projection (control)")
    ax.semilogy(A2_CHURN_IT, A2_CHURN, "o-", color=C_A2, lw=1.8, ms=4,
                label="soft area (A2), as run")
    ax.semilogy(A2_PROBE_IT[:1], A2_PROBE_CHURN[:1], "o--", color=C_A2,
                lw=1.4, ms=4, alpha=0.7)
    ax.plot([A2_CHURN_IT[-1], A2_PROBE_IT[0]], [A2_CHURN[-1],
            A2_PROBE_CHURN[0]], "--", color=C_A2, lw=1.4, alpha=0.7)
    ax.axvline(2575, color=C_A2, ls=":", lw=1.2)
    ax.annotate("A2 refinement trigger fires\n(structure still moving)",
                xy=(2575, 8.0), fontsize=7.5, color=C_A2, ha="right")
    ax.annotate("churn becomes exactly 0\n(dead line search, not convergence)",
                xy=(3000, 6e-3), xytext=(3600, 0.06), fontsize=7.5, color=C_A2,
                arrowprops=dict(arrowstyle="->", color=C_A2, lw=0.9))
    ax.set_xlabel("PGD iteration (level 0)")
    ax.set_ylabel("vertices relabeled since previous sample (% of V)")
    ax.set_title("label churn from the identical seeded $x_0$ (stride 500)",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    out = os.path.join(HERE, "fig_churn.pdf")
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    fig_two_defects()
    fig_churn()
