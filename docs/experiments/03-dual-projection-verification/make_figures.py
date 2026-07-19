#!/usr/bin/env python3
"""Regenerate the figures for docs/experiments/03-dual-projection-verification.

Measured study: does the incumbent Phase-1 constraint projection
`orthogonal_projection_iterative` (src/optimization/projection.py) compute the TRUE
Euclidean projection onto {row-sum=1, area=d, 0<=U<=1}? It does not. We measure, from
scratch, the gap between the incumbent's fixed point and the exact projection, using an
INDEPENDENT reference (Dykstra over {row=1} n {area=d} n {box}, which the Boyle-Dykstra
theorem guarantees converges to the exact Euclidean projection), plus a KKT stationarity
residual that needs no external solver.

This is the evidence base for the plan
`docs/plans/PHASE1_DUAL_NEWTON_PROJECTION_PLAN.md` (which replaces the incumbent with the
exact projection via the concave dual).

Run from the repo root:
    python docs/experiments/03-dual-projection-verification/make_figures.py

Produces (vector PDF, embedded by main.tex):
    fig_projection_gap.pdf, fig_kkt_certificate.pdf

Provenance: NO results/ run is involved -- inputs are synthetic and generated here with a
fixed seed (SEED below). The lumped mass v is built directly from the project's
TorusMeshProvider (T(R=1.0, r=0.6), n_theta=16, n_phi=8 -> V=128). The code under test is
`orthogonal_projection_iterative` as committed. Reference truth = Dykstra (stopping
delta <= 1e-14); cross-checked by the stationarity residual (~1e-15 for the true
projection, 0.1-0.3 for the incumbent). Deterministic: same seed -> same numbers.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())
from src.optimization.projection import orthogonal_projection_iterative  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260718

# Okabe-Ito (colorblind-safe), matching sibling reports
BLUE = "#0072B2"; ORANGE = "#E69F00"; VERM = "#D55E00"; GREEN = "#009E73"; GREY = "#999999"
plt.rcParams.update({
    "font.size": 11, "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.edgecolor": "#666666", "axes.linewidth": 0.8,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})


# ----------------------------------------------------------------- references
def P_rows(U):
    return U - (U.sum(axis=1, keepdims=True) - 1.0) / U.shape[1]


def P_area(U, v, d):
    return U - np.outer(v, (v @ U - d) / float(v @ v))


def dykstra(Y, v, d, sets, n_iter=500000, tol=1e-14):
    """Exact Euclidean projection onto the intersection of `sets` (Boyle-Dykstra)."""
    x = Y.copy()
    incr = [np.zeros_like(Y) for _ in sets]
    for _ in range(n_iter):
        x_old = x.copy()
        for i, P in enumerate(sets):
            t = x + incr[i]
            y = P(t)
            incr[i] = t - y
            x = y
        if np.max(np.abs(x - x_old)) < tol:
            break
    return x


def true_projection(Y, v, d):
    return dykstra(Y, v, d, [lambda U: P_rows(U),
                             lambda U: P_area(U, v, d),
                             lambda U: np.clip(U, 0.0, 1.0)])


def affine_projection(Y, v, d):
    """Euclidean projection onto the affine set {row=1, area=d} only (no box)."""
    return dykstra(Y, v, d, [lambda U: P_rows(U), lambda U: P_area(U, v, d)])


def safe_iterative(Y, n, d, v):
    """Incumbent at its production tolerance; tolerate its RuntimeError floor."""
    for t in (1e-8, 1e-7, 1e-6, 1e-5):
        try:
            return orthogonal_projection_iterative(Y.copy(), np.ones(n), d, v,
                                                   max_iter=5000, tol=t)
        except RuntimeError:
            continue
    return Y.copy()


def stationarity_residual(U, Y, v):
    """max |(U-Y) - (alpha_i + v_i beta_k)| over box-INACTIVE entries, with (alpha,beta)
    fit by least squares. ~0 iff U-Y lies in the projection's dual span (i.e. U is the
    exact projection). The clean discriminator; independent of the sign-condition weakness
    of a full LS certificate (see the report's Implications section)."""
    N, n = U.shape
    r = U - Y
    inactive = (U > 1e-8) & (U < 1 - 1e-8)
    ridx, cidx = np.where(inactive)
    m = len(ridx)
    if m == 0:
        return 0.0
    A = np.zeros((m, N + n))
    A[np.arange(m), ridx] = 1.0
    A[np.arange(m), N + cidx] = v[ridx]
    sol, *_ = np.linalg.lstsq(A, r[inactive], rcond=None)
    return float(np.max(np.abs(A @ sol - r[inactive])))


def feas(U, v, d):
    return (float(np.max(np.abs(U.sum(1) - 1.0))),
            float(np.max(np.abs(v @ U - d))),
            max(0.0, -float(U.min())))


# ----------------------------------------------------------------- inputs
def torus_v(nt=16, npx=8):
    from src.surfaces.torus import TorusMeshProvider
    return np.asarray(TorusMeshProvider(nt, npx, 1.0, 0.6).build().v, float)


def make_Y(N, n, regime, rng):
    if regime == "interior":
        return 1.0 / n + 0.03 * rng.standard_normal((N, n))
    if regime == "binding":
        Y = 0.2 * rng.standard_normal((N, n))
        Y[np.arange(N), rng.integers(0, n, N)] += 1.0
        return Y
    if regime == "crisp":  # realistic PGD operating point: near one-hot, pre-clipped
        Y = 0.02 * rng.standard_normal((N, n))
        Y[np.arange(N), rng.integers(0, n, N)] += 1.0
        return np.clip(Y, 1e-8, 1 - 1e-8)
    raise ValueError(regime)


# ----------------------------------------------------------------- study
def run_case(v, n, regime, rng, label):
    V = len(v)
    d = (float(v.sum()) / n) * np.ones(n)
    Y = make_Y(V, n, regime, rng)
    U_it = safe_iterative(Y, n, d, v)
    U_tr = true_projection(Y, v, d)
    gap = float(np.max(np.abs(U_it - U_tr)))
    obj_it = 0.5 * float(np.sum((U_it - Y) ** 2))
    obj_tr = 0.5 * float(np.sum((U_tr - Y) ** 2))
    obj_excess = 100.0 * (obj_it - obj_tr) / obj_tr
    frac0 = float(np.mean(U_tr < 1e-9))
    return dict(label=label, regime=regime, n=n, gap=gap, obj_excess=obj_excess,
                frac0=frac0, kkt_it=stationarity_residual(U_it, Y, v),
                kkt_tr=stationarity_residual(U_tr, Y, v),
                feas_it=feas(U_it, v, d), feas_tr=feas(U_tr, v, d))


def main():
    rng = np.random.default_rng(SEED)
    v = torus_v()
    V = len(v)
    print(f"torus T(1.0,0.6) 16x8 -> V={V}, sum(v)={v.sum():.4f}")

    cases = [
        run_case(v, 4, "interior", rng, "interior\nN=4"),
        run_case(v, 4, "binding", rng, "binding\nN=4"),
        run_case(v, 10, "binding", rng, "binding\nN=10"),
        run_case(v, 10, "crisp", rng, "crisp op-pt\nN=10"),
    ]

    # --- mechanism: interior gap == pre-loop multiplicative row renormalization
    d4 = (float(v.sum()) / 4) * np.ones(4)
    Yi = make_Y(V, 4, "interior", np.random.default_rng(SEED + 1))
    U_it_i = safe_iterative(Yi, 4, d4, v)
    rn = Yi / Yi.sum(axis=1, keepdims=True)            # multiplicative row-normalize
    mech_rownorm = float(np.max(np.abs(U_it_i - affine_projection(rn, v, d4))))
    mech_raw = float(np.max(np.abs(U_it_i - affine_projection(Yi, v, d4))))

    # --- idempotency: exact-feasible re-projection vs PGD-style clip-then-project
    Uf = true_projection(make_Y(V, 8, "crisp", np.random.default_rng(SEED + 2)), v,
                         (float(v.sum()) / 8) * np.ones(8))
    idem_bare = float(np.max(np.abs(safe_iterative(Uf, 8, (float(v.sum()) / 8) * np.ones(8), v) - Uf)))
    Uc = np.clip(Uf, 1e-8, 1 - 1e-8)
    idem_clip = float(np.max(np.abs(
        safe_iterative(Uc, 8, (float(v.sum()) / 8) * np.ones(8), v) - Uf)))

    # ================================================================= FIGURE 1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.3))
    labels = [c["label"] for c in cases]
    gaps = [c["gap"] for c in cases]
    excess = [c["obj_excess"] for c in cases]
    cols = [GREEN, ORANGE, VERM, "#7a0177"]
    b1 = ax1.bar(range(4), gaps, color=cols, width=0.62)
    for b, g in zip(b1, gaps):
        ax1.annotate(f"{g:.3f}", (b.get_x() + b.get_width() / 2, g),
                     textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)
    ax1.set_xticks(range(4)); ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel(r"max$|$incumbent $-$ true projection$|$")
    ax1.set_title("The incumbent fixed point is far from\nthe true projection (grows with crispness)",
                  fontsize=11)
    ax1.set_ylim(0, max(gaps) * 1.25)

    b2 = ax2.bar(range(4), excess, color=cols, width=0.62)
    for b, e in zip(b2, excess):
        ax2.annotate(f"+{e:.0f}%", (b.get_x() + b.get_width() / 2, e),
                     textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)
    ax2.set_xticks(range(4)); ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel(r"objective excess  $\frac{1}{2}\|U-Y\|^2$  (% over true)")
    ax2.set_title("Both are feasible, but the incumbent is\nstrictly farther from Y (suboptimal)",
                  fontsize=11)
    ax2.set_ylim(0, max(excess) * 1.25)
    fig.suptitle(r"The incumbent projection is not the Euclidean projection (torus $T(1.0,0.6)$, $V=128$)",
                 fontsize=12, y=1.02)
    fig.savefig(os.path.join(HERE, "fig_projection_gap.pdf")); plt.close(fig)

    # ================================================================= FIGURE 2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.3))
    x = np.arange(4)
    w = 0.38
    kt = [max(c["kkt_tr"], 1e-16) for c in cases]
    ki = [max(c["kkt_it"], 1e-16) for c in cases]
    ax1.bar(x - w / 2, kt, w, color=GREEN, label="true projection (Dykstra)")
    ax1.bar(x + w / 2, ki, w, color=VERM, label="incumbent iterative")
    ax1.set_yscale("log")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("KKT stationarity residual  (log)")
    ax1.set_title("Dual-span test: the true projection satisfies it\nto ~1e-15; the incumbent does not",
                  fontsize=11)
    ax1.axhline(1e-9, color=GREY, ls="--", lw=1)
    ax1.annotate("guard 1e-9", (3.4, 1e-9), textcoords="offset points", xytext=(0, 3),
                 color=GREY, fontsize=8, ha="right")
    ax1.legend(frameon=False, fontsize=9, loc="center left")
    ax1.set_ylim(1e-16, 1e1)

    mech_labels = ["vs affine proj.\nof raw Y", "vs affine proj. of\nrow-normalized Y"]
    mvals = [max(mech_raw, 1e-16), max(mech_rownorm, 1e-16)]
    b = ax2.bar(range(2), mvals, color=[GREY, BLUE], width=0.5)
    for bb, mv in zip(b, mvals):
        ax2.annotate(f"{mv:.1e}", (bb.get_x() + bb.get_width() / 2, mv),
                     textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)
    ax2.set_yscale("log")
    ax2.set_xticks(range(2)); ax2.set_xticklabels(mech_labels, fontsize=9)
    ax2.set_ylabel(r"max$|$incumbent(interior) $- \cdot|$  (log)")
    ax2.set_title("Mechanism: the interior gap IS the pre-loop\nmultiplicative row renormalization",
                  fontsize=11)
    ax2.set_ylim(1e-14, 1e0)
    fig.suptitle("Why the incumbent is not the projection: it solves the affine projection of the "
                 "row-renormalized input", fontsize=11.5, y=1.02)
    fig.savefig(os.path.join(HERE, "fig_kkt_certificate.pdf")); plt.close(fig)

    # ================================================================= ANCHORS
    print("\n=== ANCHORS (reproduce these) ===")
    for c in cases:
        print(f"  {c['regime']:9s} N={c['n']:2d}: gap={c['gap']:.4f}  obj_excess=+{c['obj_excess']:.1f}%  "
              f"frac0={c['frac0']:.3f}  KKT(true/incumbent)={c['kkt_tr']:.1e}/{c['kkt_it']:.2f}")
    print(f"  mechanism: |incumbent_interior - P_affine(rownorm Y)| = {mech_rownorm:.2e}  "
          f"(vs raw Y: {mech_raw:.2e})")
    print(f"  idempotency: bare={idem_bare:.2e}  clip-then-project={idem_clip:.2e}")
    print(f"  incumbent feasibility (crisp op-pt): row/area/nonneg = "
          f"{cases[3]['feas_it'][0]:.1e}/{cases[3]['feas_it'][1]:.1e}/{cases[3]['feas_it'][2]:.1e}")
    print(f"  true    feasibility (crisp op-pt): row/area/nonneg = "
          f"{cases[3]['feas_tr'][0]:.1e}/{cases[3]['feas_tr'][1]:.1e}/{cases[3]['feas_tr'][2]:.1e}")
    print("figures:", sorted(os.path.basename(p) for p in [
        os.path.join(HERE, "fig_projection_gap.pdf"),
        os.path.join(HERE, "fig_kkt_certificate.pdf")]))
    print("DONE")


if __name__ == "__main__":
    main()
