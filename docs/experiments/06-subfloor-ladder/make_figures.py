#!/usr/bin/env python3
"""Figures for report 06 -- the sub-floor coarsest level at N=100.

Produces two vector figures:
  fig_floor_onset.pdf     when the line search reaches the backtracking floor,
                          as a function of vertices per cell. Every level of the
                          validated control floors and then runs exactly
                          refine_patience=30 more iterations before the trigger
                          fires -- flooring IS normal termination. What is
                          diagnostic is WHEN: a level 0 at 31 v/cell floors at
                          iteration 31, while the control's level 0 at 96
                          v/cell never floors across 30,000 iterations.
  fig_runt_persistence.pdf the damage a starved coarsest level leaves behind.
                          The worst-cell discrete-area deviation opens at 46.80%
                          on the 31 v/cell level and is still 15.92% on the
                          finest mesh -- 37x more vertices, never crossing the
                          5% gate -- against a control that ends at 0.78%.
                          Fragmented cells are 0 at every level in BOTH runs.

The arrays below are the MEASUREMENT. Iteration counts and the first floored
iteration come from column 9 (STEP) of the PGD summary traces under
`traces/*_summary.out`, the floor being the value 9.0949470177292824e-13
(= step0 * rho^40 with step0=1.0, rho=0.5). Gate values come from
`testing/watch_level_gates.py`, which copies each per-level checkpoint before
the pipeline overwrites it and runs detect_dormant_cells /
detect_area_imbalance / detect_disconnected_cells on it; the final row is
`testing/check_fragmentation.py` on the final solution. They are embedded rather
than re-read so the figures regenerate without the raw runs present -- see the
provenance block in main.tex for the run identifiers.

Run from the repo root:
    python docs/experiments/06-subfloor-ladder/make_figures.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

C_CTRL = "#1f77b4"
C_SUB = "#d62728"
C_AUX = "#7f7f7f"

# --- measurement: (verts/cell, iterations run, first floored iteration) ------
# None = the level never reached the floor.
# CONTROL run_20260709_081548 -- 100x96 base, the validated N=100 deliverable.
CTRL_LADDER = [
    (96, 30000, None),   # level 0 -- ran to the max_iter cap, never floored
    (249, 1372, 1343),
    (475, 986, 957),
    (772, 1197, 1168),
    (1141, 1279, 1250),
]
# SUB-FLOOR run_20260814_002405 (levels 0-3) + run_20260814_091130 (level 4).
# 60x52 base = 31 v/cell, matching N=300's level-0 condition of 32 v/cell.
SUB_LADDER = [
    (31, 60, 31),        # level 0 -- floors at iteration 31, ends at 60
    (160, 3000, None),   # level 1 -- stopped by the structure trigger @2,999
    (388, 1141, 1112),
    (715, 982, 953),
    (1141, 1100, 1071),
]

# --- measurement: gate evolution down the sub-floor ladder -------------------
SUB_VPC = [31, 160, 388, 715, 1141]
SUB_WORST = [46.80, 18.78, 18.13, 16.59, 15.92]   # worst cell, % off target
SUB_IMBAL = [74, 1, 1, 1, 1]                       # cells over the 5% gate
SUB_FRAG = [0, 0, 0, 0, 0]                         # disconnected cells
SUB_MINPEAK = [0.7367, 0.9978, 0.9980, 0.9981, 0.9982]

CTRL_FINAL_WORST = 0.78      # control, finest level
CTRL_FINAL_IMBAL = 0
CTRL_FINAL_FRAG = 0
GATE = 5.0                   # AREA_IMBALANCE_REL_THRESHOLD, %


def fig_floor_onset():
    fig, ax = plt.subplots(figsize=(6.6, 3.9))

    for ladder, color, label, marker in (
        (CTRL_LADDER, C_CTRL, "control (100$\\times$96 base, 96 v/cell)", "o"),
        (SUB_LADDER, C_SUB, "sub-floor (60$\\times$52 base, 31 v/cell)", "s"),
    ):
        floored = [(v, f) for v, _, f in ladder if f is not None]
        ax.loglog([v for v, _ in floored], [f for _, f in floored],
                  marker + "-", color=color, lw=1.8, ms=5, label=label)
        # levels that never floored: plot at the iteration count, open marker
        never = [(v, n) for v, n, f in ladder if f is None]
        if never:
            ax.loglog([v for v, _ in never], [n for _, n in never], marker,
                      color=color, ms=8, mfc="none", mew=1.6)

    ax.annotate("31 v/cell:\nfloors at iteration 31",
                xy=(31, 31), xytext=(36, 170), fontsize=7.5, color=C_SUB,
                arrowprops=dict(arrowstyle="->", color=C_SUB, lw=0.9))
    ax.annotate("96 v/cell: never floors\nin 30,000 iterations",
                xy=(96, 30000), xytext=(150, 12000), fontsize=7.5, color=C_CTRL,
                arrowprops=dict(arrowstyle="->", color=C_CTRL, lw=0.9))
    ax.annotate("stopped by the\nstructure trigger",
                xy=(160, 3000), xytext=(215, 4200), fontsize=7.5, color=C_AUX,
                arrowprops=dict(arrowstyle="->", color=C_AUX, lw=0.9))
    ax.annotate("normal convergence band\n(floor, then exactly 30 more)",
                xy=(700, 1000), xytext=(300, 300), fontsize=7.5, color=C_AUX)

    ax.set_xlabel("vertices per cell at that level")
    ax.set_ylabel("iteration at which the step reaches the floor")
    ax.set_title("when the line search floors (open marker = never floored)",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, which="both", alpha=0.2, lw=0.5)

    fig.tight_layout()
    out = os.path.join(HERE, "fig_floor_onset.pdf")
    fig.savefig(out)
    print("wrote", out)


def fig_runt_persistence():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.5))

    ax1.semilogx(SUB_VPC, SUB_WORST, "s-", color=C_SUB, lw=1.8, ms=5,
                 label="sub-floor ladder")
    ax1.semilogx([SUB_VPC[-1]], [CTRL_FINAL_WORST], "o", color=C_CTRL, ms=7,
                 label="control, finest level")
    ax1.axhline(GATE, color=C_AUX, ls=":", lw=1.3)
    ax1.annotate("5% equal-area gate", xy=(40, 6.2), fontsize=7.5, color=C_AUX)
    ax1.annotate("born at 31 v/cell", xy=(31, 46.80), xytext=(45, 40),
                 fontsize=7.5, color=C_SUB,
                 arrowprops=dict(arrowstyle="->", color=C_SUB, lw=0.9))
    ax1.annotate("37$\\times$ more vertices,\nstill 15.92%",
                 xy=(1141, 15.92), xytext=(300, 24), fontsize=7.5, color=C_SUB,
                 arrowprops=dict(arrowstyle="->", color=C_SUB, lw=0.9))
    ax1.set_xlabel("vertices per cell at that level")
    ax1.set_ylabel("worst cell, % off equal-area target")
    ax1.set_title("(a) the runt is manufactured, then inherited", fontsize=10)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_ylim(-2, 52)

    ax2.semilogx(SUB_VPC, SUB_IMBAL, "s-", color=C_SUB, lw=1.8, ms=5,
                 label="imbalanced cells (>5%)")
    ax2.semilogx(SUB_VPC, SUB_FRAG, "^--", color="#2ca02c", lw=1.8, ms=6,
                 label="fragmented cells")
    ax2.axhline(0, color=C_AUX, ls="-", lw=0.6)
    ax2.annotate("0 at every level --\nthe pre-registered\nprediction is refuted",
                 xy=(388, 0), xytext=(120, 22), fontsize=7.5, color="#2ca02c",
                 arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=0.9))
    ax2.annotate("74 $\\to$ 1: refinement clears\nall but one", xy=(31, 74),
                 xytext=(90, 60), fontsize=7.5, color=C_SUB,
                 arrowprops=dict(arrowstyle="->", color=C_SUB, lw=0.9))
    ax2.set_xlabel("vertices per cell at that level")
    ax2.set_ylabel("cells failing the gate (of 100)")
    ax2.set_title("(b) imbalance, not disconnection", fontsize=10)
    ax2.legend(fontsize=8, loc="center right")
    ax2.set_ylim(-3, 82)

    fig.tight_layout()
    out = os.path.join(HERE, "fig_runt_persistence.pdf")
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    fig_floor_onset()
    fig_runt_persistence()
