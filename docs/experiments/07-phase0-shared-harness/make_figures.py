#!/usr/bin/env python3
"""Figures for report 07 -- Phase 0: qualifying the measurement instrument.

Run from the repo root:
    python docs/experiments/07-phase0-shared-harness/make_figures.py

Every number below is a MEASUREMENT recorded in this repository, not a
re-derivation: the gate-2 tables come from the committed runs listed in the
report's provenance block, and the discrimination table from the review-4
solver sweep. They are transcribed here so the figures are reproducible without
re-running 2 h 39 m of gates. To re-measure instead of re-plot:

    python testing/test_balanced_assignment_solver.py            # gates 1,3,4,2
    python testing/test_phase0d_shakedown.py --scratch <dir>     # 0d, ~34 min
    python testing/test_arm_harness.py                           # 0b acceptance
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})

# --- Measured 2026-08-18, final 18-fixture gate 2 (task b4ynpmhfr) ------------
FINAL = {
    "V=47,488\nN=300":   dict(bar=2.022, C=[1.472, 1.392, 1.495], B=[1.405, 1.390, 1.406]),
    "V=114,144\nN=100":  dict(bar=0.280, C=[0.146, 0.142, 0.144], B=[0.161, 0.151, 0.138]),
    "V=114,144\nN=300":  dict(bar=0.841, C=[0.624, 0.573, 0.609], B=[0.537, 0.573, 0.607]),
}
# Incumbent A's own dual, normalized, at the SAME 2000-iteration budget.
A_PARITY = {"V=47,488\nN=300": 1.3517, "V=114,144\nN=300": 0.5991}


def fig_final_vs_bar():
    """Where the arms sit relative to their bar and to the incumbent."""
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    xs, labels = [], []
    for i, (name, d) in enumerate(FINAL.items()):
        base = i * 3.0
        ax.hlines(d["bar"], base - 0.55, base + 1.55, color="crimson",
                  lw=1.6, zorder=3)
        ax.text(base + 1.6, d["bar"], f"bar {d['bar']:.3f}%", va="center",
                fontsize=7.5, color="crimson")
        if name in A_PARITY:
            ax.hlines(A_PARITY[name], base - 0.55, base + 1.55, color="grey",
                      lw=1.4, ls="--", zorder=3)
            ax.text(base + 1.6, A_PARITY[name], f"incumbent {A_PARITY[name]:.3f}%",
                    va="center", fontsize=7.5, color="grey")
        for j, (arm, col) in enumerate((("C", "#2c7fb8"), ("B", "#31a354"))):
            ax.scatter([base + j] * 3, d[arm], s=26, color=col, zorder=4,
                       label=arm if i == 0 else None)
        xs += [base, base + 1]
        labels += ["C", "B"]
        ax.text(base + 0.5, -0.28, name, ha="center", fontsize=7.5,
                transform=ax.get_xaxis_transform())
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("worst |area deviation| (%)")
    ax.set_yscale("log")
    ax.set_title("Gate 2, final protocol: 3 seeds per arm, 18/18 pass",
                 fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(os.path.join(OUT, "fig_final_vs_bar.pdf"), bbox_inches="tight")
    plt.close(fig)


# --- Review-4 discrimination sweep, C scores, production pin ------------------
# Progressively degraded solvers vs the shipped config. bar = 0.841% (re-anchored).
SWEEP = [("null (psi=0)", 16.588), ("un-normalized 400", 16.588),
         ("budget 20", 7.974), ("budget 50", 3.968), ("budget 100", 2.163),
         ("budget 200", 1.332), ("un-normalized 2000", 1.221),
         ("budget 400", 0.919), ("budget 1000", 0.726),
         ("shipped (2000)", 0.624), ("budget 4000", 0.619)]


def fig_discrimination():
    """The gate's failure boundary -- how far the shipped config sits from it."""
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    names = [n for n, _ in SWEEP][::-1]
    vals = np.array([v for _, v in SWEEP][::-1])
    bar = 0.841
    cols = ["#31a354" if v <= bar else "#d95f02" for v in vals]
    ax.barh(range(len(vals)), vals, color=cols, height=0.66)
    ax.axvline(bar, color="crimson", lw=1.6)
    ax.text(bar * 1.06, len(vals) - 0.4, f"bar {bar}%", color="crimson",
            fontsize=8, va="top")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xscale("log"); ax.set_xlabel("worst |area deviation| (%), C scores")
    ax.set_title("Gate 2 discriminates: the pre-0a solver is indistinguishable "
                 "from doing nothing", fontsize=9)
    for i, v in enumerate(vals):
        ax.text(v * 1.07, i, f"{v:.3f}", va="center", fontsize=7)
    fig.savefig(os.path.join(OUT, "fig_discrimination.pdf"), bbox_inches="tight")
    plt.close(fig)


# --- The early-stop masking episode ------------------------------------------
MASK = {"C prod": (0.888, 0.624), "B prod": (0.877, 0.537),
        "C @47k": (1.946, 1.472), "C @N=100": (0.213, 0.146)}


def fig_masking():
    """Why a 12/12 PASS was not trustworthy: every number hugged its bar."""
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    names = list(MASK)
    masked = [MASK[k][0] for k in names]
    true = [MASK[k][1] for k in names]
    y = np.arange(len(names))
    ax.barh(y - 0.19, masked, height=0.36, color="#d95f02",
            label="reported with early stop inside the gate")
    ax.barh(y + 0.19, true, height=0.36, color="#31a354",
            label="true achieved (full budget)")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("worst |area deviation| (%)")
    ax.set_title("Early stopping inside the measurement path masked quality",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.savefig(os.path.join(OUT, "fig_masking.pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_final_vs_bar()
    fig_discrimination()
    fig_masking()
    print("wrote fig_final_vs_bar.pdf, fig_discrimination.pdf, fig_masking.pdf")
