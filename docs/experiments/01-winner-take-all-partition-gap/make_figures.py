#!/usr/bin/env python3
"""Regenerate the figures for docs/experiments/01-winner-take-all-partition-gap.

Measured study: the winner-take-all discrete cell-area gap at high N. Tracks each
cell's winner-take-all territory across the 5 refinement levels of three Phase 1
runs and shows (1) the level-by-level trajectory, (2) the final per-cell
distribution, (3) the mass-vs-territory split of the runt cell.

Run from the repo root:
    python docs/experiments/01-winner-take-all-partition-gap/make_figures.py

Produces (vector PDF, embedded by main.tex):
    fig_area_trajectory.pdf, fig_final_distribution.pdf, fig_mass_vs_territory.pdf

Provenance (the three anchor runs under results/, seed 84172851, lambda 2.1):
    N=50 (works)      run_20260625_113015_surftorus_npart50_...
    N=100 coarse mesh run_20260629_141012_surftorus_npart100_...
    N=100 finer mesh  run_20260701_143238_surftorus_npart100_...
The winner-take-all area = lumped P1 mass assigned to each vertex's argmax cell;
the worst-cell absolute deviation reproduces the Phase 2 iteration-0 equal-area
constraint violation (0.0036 / 0.053 / 0.160), which anchors the reconstruction.
"""
import os
import sys
import re
import glob
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())
from src.surfaces.torus import TorusMeshProvider  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# Okabe-Ito (colorblind-safe by construction)
BLUE = "#0072B2"; ORANGE = "#E69F00"; VERM = "#D55E00"; GREEN = "#009E73"; GREY = "#999999"
plt.rcParams.update({
    "font.size": 11, "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.edgecolor": "#666666", "axes.linewidth": 0.8,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})

RUNS = {
    "N50":  dict(path="results/run_20260625_113015_surftorus_npart50_v1nt100-348_incr62_v2np96-328_incr58_lam2.1_seed84172851",
                 N=50,  base=(100, 96),  incr=(62, 58), color=BLUE,   label="N=50 (works)"),
    "C100": dict(path="results/run_20260629_141012_surftorus_npart100_v1nt100-348_incr62_v2np96-328_incr58_lam2.1_seed84172851",
                 N=100, base=(100, 96),  incr=(62, 58), color=ORANGE, label="N=100 coarse mesh"),
    "F100": dict(path="results/run_20260701_143238_surftorus_npart100_v1nt142-494_incr88_v2np136-464_incr82_lam2.1_seed84172851",
                 N=100, base=(142, 136), incr=(88, 82), color=VERM,   label="N=100 finer mesh"),
}


def lumped(V, F):
    """Lumped P1 mass per vertex (1/3 sum of incident triangle areas)."""
    tri = V[F]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    v = np.zeros(len(V))
    for j in range(3):
        np.add.at(v, F[:, j], area / 3.0)
    return v


def level_end_stats(r):
    N = r["N"]; nt0, np0 = r["base"]; dnt, dnp = r["incr"]; out = []
    for L in range(5):
        nt, nph = nt0 + L * dnt, np0 + L * dnp
        prov = TorusMeshProvider(nt, nph, 1.0, 0.6, n_theta_increment=dnt, n_phi_increment=dnp)
        m = prov.build(); V = np.asarray(m.vertices); F = np.asarray(m.faces).astype(np.int64)
        v = lumped(V, F); nv = len(V); tgt = v.sum() / N
        cand = glob.glob(f"{r['path']}/traces/pgd_part{N}_v1nt{nt}_v2np{nph}_level{L}_internal_data.hdf5")
        with h5py.File(cand[0], "r") as f:
            it = max(int(re.findall(r"\d+", k)[0]) for k in f.keys())
            u = f[f"iter_{it}"]["x"][:].reshape(nv, N)
        win = np.argmax(u, 1); a = np.zeros(N); np.add.at(a, win, v)
        out.append(dict(level=L, std=a.std() / tgt * 100, worst=np.abs(a - tgt).max() / tgt * 100))
    return out


def final_solution(r):
    N = r["N"]; sol = glob.glob(f"{r['path']}/solution/surface_part{N}_*.h5")[0]
    with h5py.File(sol, "r") as f:
        V = f["vertices"][:]; F = f["faces"][:].astype(np.int64); x = f["x_opt"][:]
    u = x.reshape(V.shape[0], N); v = lumped(V, F); tgt = v.sum() / N
    win = np.argmax(u, 1); a = np.zeros(N); np.add.at(a, win, v)
    return u, v, tgt, a


def main():
    print("computing trajectory...", flush=True)
    stats = {k: level_end_stats(r) for k, r in RUNS.items()}
    finals = {k: final_solution(r) for k, r in RUNS.items()}

    # FIGURE 1 -- spread vs worst-cell across levels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    for k, r in RUNS.items():
        xs = [s["level"] for s in stats[k]]
        ax1.plot(xs, [s["std"] for s in stats[k]], "-o", color=r["color"], lw=2, ms=6)
        ax2.plot(xs, [s["worst"] for s in stats[k]], "-o", color=r["color"], lw=2, ms=6, label=r["label"])
        ax1.annotate(f"{stats[k][-1]['std']:.1f}", (xs[-1], stats[k][-1]["std"]),
                     textcoords="offset points", xytext=(6, 0), color=r["color"], fontsize=9, va="center")
        ax2.annotate(f"{stats[k][-1]['worst']:.0f}%", (xs[-1], stats[k][-1]["worst"]),
                     textcoords="offset points", xytext=(6, 0), color=r["color"], fontsize=9, va="center")
    ax1.set_title("Overall area spread\n(the bulk equalizes everywhere)", fontsize=11)
    ax1.set_ylabel("std of cell areas  (% of target)"); ax1.set_xlabel("refinement level"); ax1.set_xticks(range(5))
    ax2.set_title("Worst single cell\n(N=100 sacrifices one)", fontsize=11)
    ax2.set_ylabel("largest cell-area error  (% of target)"); ax2.set_xlabel("refinement level"); ax2.set_xticks(range(5))
    ax2.legend(frameon=False, fontsize=9, loc="center left")
    ax2.axhline(5, color=GREY, ls="--", lw=1)
    ax2.annotate("5% gate", (0, 5), textcoords="offset points", xytext=(2, 3), color=GREY, fontsize=8)
    fig.suptitle("Relaxation equalizes the bulk but manufactures a runt at N=100 (level 2 onward)", fontsize=12, y=1.02)
    fig.savefig(os.path.join(HERE, "fig_area_trajectory.pdf")); plt.close(fig)

    # FIGURE 2 -- final sorted per-cell deviation
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for k, r in RUNS.items():
        _, v, tgt, a = finals[k]; d = np.sort((a - tgt) / tgt * 100)
        ax.plot(np.arange(len(d)), d, "-o", color=r["color"], lw=1.6, ms=4, label=r["label"])
        ax.annotate(f"{d[0]:.0f}%", (0, d[0]), textcoords="offset points", xytext=(4, -2), color=r["color"], fontsize=9)
    ax.axhline(0, color="#444444", lw=1); ax.axhline(-5, color=GREY, ls="--", lw=1); ax.axhline(5, color=GREY, ls="--", lw=1)
    ax.set_xlabel("cell rank (sorted by area deviation)"); ax.set_ylabel("area deviation (% of target)")
    ax.set_title("Final partition: one catastrophic outlier at N=100, the rest tightly equal", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.savefig(os.path.join(HERE, "fig_final_distribution.pdf")); plt.close(fig)

    # FIGURE 3 -- mass vs territory for the runt (cell 92, finer run)
    u, v, tgt, a = finals["F100"]
    mass92 = (u[:, 92] * v).sum(); terr92 = a[92]
    healthy = int(np.argsort(np.abs(a - tgt))[50]); massH = (u[:, healthy] * v).sum(); terrH = a[healthy]
    fig, ax = plt.subplots(figsize=(7, 4.2)); xpos = np.arange(2); w = 0.36
    ax.bar(xpos - w / 2, [mass92, massH], w, color=GREY, label="continuous mass  $\\int u\\,dA$")
    ax.bar(xpos + w / 2, [terr92, terrH], w, color=[VERM, GREEN], label="winner-take-all territory")
    ax.axhline(tgt, color="#444444", ls="--", lw=1.2)
    ax.annotate("equal-area target", (1.18, tgt), textcoords="offset points", xytext=(0, 4), fontsize=9, color="#444444", ha="right")
    ax.annotate(f"only {terr92/mass92*100:.0f}% of its\nmass wins ground", (w / 2, terr92),
                textcoords="offset points", xytext=(0, 8), fontsize=9, color=VERM, ha="center")
    ax.set_xticks(xpos); ax.set_xticklabels([f"runt cell 92", f"healthy cell {healthy}"]); ax.set_ylabel("area")
    ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, 0.27)
    ax.set_title("Same continuous mass, very different territory (N=100 finer run)", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="center")
    fig.savefig(os.path.join(HERE, "fig_mass_vs_territory.pdf")); plt.close(fig)

    print("TRAJECTORY (end-of-level):")
    for k in RUNS:
        print(f"  {k}: std% =", [round(s["std"], 1) for s in stats[k]], " worst% =", [round(s["worst"], 1) for s in stats[k]])
    print(f"RUNT 92: mass={mass92:.4f} territory={terr92:.4f} ({terr92/mass92*100:.1f}% of mass) target={tgt:.4f}")
    print("figures:", sorted(glob.glob(os.path.join(HERE, "*.pdf"))))
    print("DONE")


if __name__ == "__main__":
    main()
