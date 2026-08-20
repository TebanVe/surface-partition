#!/usr/bin/env python3
"""Figures for report 08 -- approach B, auction-dynamics MBO.

    python docs/experiments/08-mbo-auction-dynamics/make_figures.py

Reads ONLY ``data.yaml`` in this directory, which is committed. ``results/`` is
gitignored, so a figure script reading it directly could not be re-run by anyone
who does not already hold the run output. The chain is:

    scored run -> results/<run>/arm_report.yaml -> extract_data.py -> data.yaml
               -> make_figures.py -> fig_*.pdf -> main.tex

To re-derive data.yaml from the runs (requires results/ on disk):
    python docs/experiments/08-mbo-auction-dynamics/extract_data.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
D = yaml.safe_load(open(os.path.join(HERE, "data.yaml")))
R, A, M = D["runs"], D["anchors"], D["measurements"]

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
    }
)
BLUE, ORANGE, GREY, RED, GREEN = "#3b6ea5", "#d1873b", "#8a8a8a", "#b5433a", "#4a8a5c"


def save(fig, name):
    p = os.path.join(HERE, name)
    # metadata CreationDate=None makes the PDF byte-reproducible: without it
    # matplotlib stamps the current time and two runs of this script produce
    # different bytes from identical data, which defeats auditing by checksum.
    fig.savefig(p, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)
    print("wrote", os.path.basename(p))


# ---------------------------------------------------------------------------
def fig_perimeter():
    """The headline. Note the two anchors are DIFFERENT pipelines -- never pooled."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

    for ax, N, keys, title in [
        (
            axes[0],
            100,
            ["n100_s84172851", "n100_s61803399"],
            "$N=100$, $V=114{,}144$\nanchor: raw PGD",
        ),
        (
            axes[1],
            300,
            ["n300_s61803399"],
            "$N=300$, $V=47{,}488$\nanchor: PGD $+$ balanced readout (A$+$E)",
        ),
    ]:
        anch = A[N]["perimeter"]
        ax.axhspan(anch, anch * 1.01, color=GREEN, alpha=0.10)
        ax.axhspan(anch * 1.01, anch * 1.05, color=ORANGE, alpha=0.10)
        ax.axhline(anch, color=GREY, lw=1.4)
        ax.text(
            0.02,
            anch,
            "  anchor (a BOUND)",
            va="bottom",
            ha="left",
            color=GREY,
            fontsize=7.5,
            transform=ax.get_yaxis_transform(),
        )
        ax.axhline(anch * 1.01, color=GREEN, lw=0.8, ls=":")
        ax.axhline(anch * 1.05, color=ORANGE, lw=0.8, ls=":")

        xs, ys, lab = [], [], []
        for i, k in enumerate(keys):
            xs.append(i)
            ys.append(R[k]["phase2"]["best_perimeter"])
            lab.append(f"B\nseed {R[k]['seed']}")
        if N == 100:
            xs.append(len(xs))
            ys.append(R["init_n100"]["phase2"]["best_perimeter"])
            lab.append("init only\n(C step 0)")
        cols = [BLUE] * len(keys) + ([RED] if N == 100 else [])
        ax.bar(xs, ys, color=cols, width=0.5, zorder=3)
        for x, y in zip(xs, ys):
            rel = (y - anch) / anch * 100
            ax.annotate(
                f"{y:.4f}\n{rel:+.3f}%",
                (x, y),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=7.5,
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(lab, fontsize=7.5)
        lo = min(ys + [anch])
        hi = max(ys + [anch * 1.05])
        ax.set_ylim(lo - (hi - lo) * 0.08, hi + (hi - lo) * 0.16)
        ax.set_title(title, fontsize=8.5)
        ax.set_ylabel("Phase 2 best perimeter")
    axes[0].text(
        0.5,
        0.02,
        "green band: $\\leq +1\\%$ success   orange: $+1..+5\\%$ partial",
        transform=axes[0].transAxes,
        ha="center",
        fontsize=7,
        color=GREY,
    )
    fig.suptitle(
        "Phase 2 perimeter against each anchor's own pipeline "
        "(the two panels are NOT comparable)",
        fontsize=9,
    )
    save(fig, "fig_perimeter.pdf")


# ---------------------------------------------------------------------------
def fig_ladder():
    """Per-level behaviour: balance, churn decay, cost."""
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6))
    runs = [
        ("n100_s84172851", "N=100 s84172851", BLUE),
        ("n100_s61803399", "N=100 s61803399", "#6b9bd1"),
        ("n300_s61803399", "N=300 s61803399", ORANGE),
    ]

    ax = axes[0]
    for k, lbl, c in runs:
        lv = R[k]["levels"]
        ax.plot(
            [l["V"] for l in lv],
            [l["worst_rel_dev"] * 100 for l in lv],
            "o-",
            color=c,
            label=lbl,
            ms=3.5,
            lw=1.2,
        )
    ax.plot(
        [R["n300_s61803399"]["levels"][i]["V"] for i in range(3)],
        [
            M["per_level_connectivity"]["n300_fragmented_by_level"][i]
            and R["n300_s61803399"]["levels"][i]["worst_rel_dev"] * 100
            or np.nan
            for i in range(3)
        ],
        "x",
        color=RED,
        ms=9,
        mew=2,
        label="level with a fragmented cell",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("vertices $V$")
    ax.set_ylabel("worst cell area deviation (\\%)")
    ax.set_title("balance improves down the ladder", fontsize=8.5)
    ax.legend(fontsize=6.2, loc="upper right")

    ax = axes[1]
    for k, lbl, c in runs:
        for i, l in enumerate(R[k]["levels"]):
            ax.plot(
                np.arange(len(l["churn_frac"])),
                np.array(l["churn_frac"]) * 100,
                color=c,
                lw=0.9,
                alpha=0.35 + 0.14 * i,
            )
    ax.set_yscale("log")
    ax.set_xlabel("MBO step within level")
    ax.set_ylabel("label churn (\\% of $V$)")
    ax.set_title("churn decays; levels stop on their own", fontsize=8.5)

    ax = axes[2]
    for k, lbl, c in runs:
        lv = R[k]["levels"]
        ax.plot(
            [l["V"] for l in lv],
            [l["wall_s"] for l in lv],
            "o-",
            color=c,
            ms=3.5,
            lw=1.2,
            label=lbl,
        )
    ax.set_xscale("log")
    ax.set_xlabel("vertices $V$")
    ax.set_ylabel("level wall time (s)")
    ax.set_title("cost per level", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig_ladder.pdf")


# ---------------------------------------------------------------------------
def fig_tau_window():
    """The two-sided tau window, and that a fixed c does not hold it."""
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for k, lbl, c, mk in [
        ("n100_s84172851", "$N=100$ ladder", BLUE, "o"),
        ("n300_s61803399", "$N=300$ ladder", ORANGE, "s"),
    ]:
        lv = R[k]["levels"]
        vpc = [l["verts_per_cell"] for l in lv]
        ax.plot(
            vpc,
            [l["sqrt_tau_over_r_cell"] for l in lv],
            mk + "-",
            color=c,
            lw=1.3,
            ms=4,
            label=f"{lbl}: $\\sqrt{{\\tau}}/R_{{cell}}$",
        )
        ax.plot(
            vpc,
            [l["sqrt_tau_over_h_max"] for l in lv],
            mk + "--",
            color=c,
            lw=1.0,
            ms=3,
            alpha=0.65,
            label=f"{lbl}: $\\sqrt{{\\tau}}/h_{{max}}$",
        )
    ax.axhline(1.0, color=RED, lw=1.0, ls=":")
    ax.text(
        35,
        1.05,
        "over-merge ceiling $\\sqrt{\\tau}=R_{cell}$ ($\\rho=1$)",
        color=RED,
        fontsize=7,
    )
    ax.set_xscale("log")
    ax.set_xlabel("vertices per cell")
    ax.set_ylabel("ratio")
    ax.set_title(
        "A fixed $c=4$ holds $\\sqrt{\\tau}/h_{max}$ at 2.20 while "
        "$\\sqrt{\\tau}/R_{cell}$ moves 6$\\times$",
        fontsize=8.5,
    )
    ax.legend(fontsize=6.6, ncol=2)
    save(fig, "fig_tau_window.pdf")


# ---------------------------------------------------------------------------
def fig_overmerge_null():
    """The headline NULL: a three-part instrument that detected nothing."""
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6))
    sets = [
        ("NC5_V47488_N300", "$V=47{,}488$ (158 v/cell)", BLUE, "o"),
        ("NC5b_V9600_N300", "$V=9{,}600$ (32 v/cell)", ORANGE, "s"),
    ]
    for ax, field, name, lo_hi in [
        (axes[0], "q_mean", "mean isoperimetric ratio $\\bar{Q}$", None),
        (axes[1], "q_worst", "worst $Q_k$", None),
        (axes[2], "fragmented", "fragmented cells / core loss", (-0.5, 2.0)),
    ]:
        for key, lbl, c, mk in sets:
            rows = M["negative_controls"][key]
            x = [r["sqrt_tau_over_r_cell"] for r in rows]
            y = [r[field] for r in rows]
            ax.plot(x, y, mk + "-", color=c, ms=5, lw=1.3, label=lbl)
            if field == "fragmented":
                ax.plot(
                    x,
                    [r["core_loss"] for r in rows],
                    mk + ":",
                    color=c,
                    ms=4,
                    lw=1.0,
                    alpha=0.7,
                )
            for xi, yi, r in zip(x, y, rows):
                ax.annotate(
                    f"c={r['c']}",
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=6.5,
                    color=c,
                )
        ax.axvline(1.0, color=RED, lw=0.9, ls=":")
        ax.set_xlabel("$\\sqrt{\\tau}/R_{cell}$")
        ax.set_ylabel(name, fontsize=8)
        if lo_hi:
            ax.set_ylim(*lo_hi)
    axes[2].legend(fontsize=6.4, loc="upper left")
    fig.suptitle(
        "Over-merge instrument: every measure is FLAT or IMPROVING across "
        "a 9$\\times$ span of $\\sqrt{\\tau}/R_{cell}$",
        fontsize=9,
    )
    fig.tight_layout()
    save(fig, "fig_overmerge_null.pdf")


# ---------------------------------------------------------------------------
def fig_predictions():
    """Pre-registered predictions against outcomes."""
    rows = M["predictions"]["rows"]
    colour = {"HELD": GREEN, "HIT": GREEN, "MISS": RED, "HELD (point missed)": ORANGE}
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ys = np.arange(len(rows))[::-1]
    for y, r in zip(ys, rows):
        c = colour.get(r["verdict"], GREY)
        ax.barh(y, 1, color=c, alpha=0.16, height=0.82)
        ax.text(0.012, y, r["id"], va="center", fontsize=8, fontweight="bold")
        ax.text(0.075, y, r["predicted"], va="center", fontsize=7)
        ax.text(0.545, y, r["actual"], va="center", fontsize=7)
        ax.text(
            0.985,
            y,
            r["verdict"],
            va="center",
            ha="right",
            fontsize=7.5,
            color=c,
            fontweight="bold",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.axis("off")
    ax.text(
        0.075,
        len(rows) - 0.15,
        "PREDICTED (committed 9ffdefb)",
        fontsize=7.5,
        fontweight="bold",
        color=GREY,
    )
    ax.text(
        0.545, len(rows) - 0.15, "MEASURED", fontsize=7.5, fontweight="bold", color=GREY
    )
    save(fig, "fig_predictions.pdf")


if __name__ == "__main__":
    fig_perimeter()
    fig_ladder()
    fig_tau_window()
    fig_overmerge_null()
    fig_predictions()
    print("all figures written to", HERE)
