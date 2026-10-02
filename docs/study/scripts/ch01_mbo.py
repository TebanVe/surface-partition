"""Figures and numbers for Chapter 1 of the study companion (the MBO primer).

Every number quoted in docs/study/chapters/ch01_mbo.tex comes from the YAML this
script writes. Run from the repo root (~1 min):

    python docs/study/scripts/ch01_mbo.py

Writes docs/study/data/ch01.yaml and docs/study/figures/ch01_*.pdf. Deterministic:
the only randomness (the Voronoi seeds of experiment E) uses a fixed seed.

Two kinds of computation:
  * Gridless (A, D): a disc diffused by the plane Gaussian has a closed-form
    profile -- the probability that a 2-D normal vector lands in the disc, i.e.
    a noncentral chi-squared CDF. These test the continuum analysis exactly.
  * Grid (B, C, E): the flat periodic unit square (a flat torus), FFT heat step
    ``u_hat = exp(-tau |xi|^2) chi_hat``. These show the algorithm as run.
"""

import platform
import sys
from pathlib import Path

import matplotlib
import numpy as np
import scipy
import yaml
from scipy import integrate, optimize, stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
DATA = ROOT / "data"
SEED = 20261002

plt.rcParams.update(
    {"font.size": 9, "axes.titlesize": 9, "figure.dpi": 150, "savefig.bbox": "tight"}
)


# --------------------------------------------------------------------------
# continuum: the disc under the plane heat kernel, in closed form
# --------------------------------------------------------------------------


def disc_profile(r, R, tau):
    """(G_tau * chi_disc)(x) at |x| = r for the heat equation u_t = Laplace u.

    G_tau is the N(0, 2 tau I) density, so u = P(|x + X| <= R) with
    X ~ N(0, 2 tau I): a noncentral chi-squared CDF with 2 degrees of freedom.
    """
    s2 = 2.0 * tau
    return stats.ncx2.cdf(R**2 / s2, df=2, nc=np.asarray(r) ** 2 / s2)


def half_level_radius(R, tau):
    """Radius at which the diffused disc indicator equals 1/2."""
    return optimize.brentq(lambda r: disc_profile(r, R, tau) - 0.5, 1e-9, R)


def flat_profile(x, tau):
    """Diffused indicator of the half-plane {x < 0}: Phi(-x / sqrt(2 tau))."""
    return stats.norm.cdf(-np.asarray(x) / np.sqrt(2.0 * tau))


def disc_heat_content(R, tau):
    """Two-phase heat content (1/sqrt tau) * integral_disc (1 - G_tau * chi)."""
    val, _ = integrate.quad(
        lambda r: 2.0 * np.pi * r * (1.0 - disc_profile(r, R, tau)),
        0.0,
        R,
        limit=200,
        epsabs=1e-13,
        epsrel=1e-11,
    )
    return val / np.sqrt(tau)


# --------------------------------------------------------------------------
# grid: the flat periodic unit square
# --------------------------------------------------------------------------


class Grid:
    def __init__(self, n):
        self.n = n
        self.h = 1.0 / n
        c = (np.arange(n) + 0.5) / n
        self.X, self.Y = np.meshgrid(c, c, indexing="ij")
        xi = 2.0 * np.pi * np.fft.fftfreq(n, d=1.0 / n)
        self.xi2 = xi[:, None] ** 2 + xi[None, :] ** 2

    def heat(self, f, tau):
        """One exact heat step on the periodic square: multiply by exp(-tau xi^2)."""
        return np.real(np.fft.ifft2(np.exp(-tau * self.xi2) * np.fft.fft2(f)))

    def area(self, chi):
        return float(chi.sum()) * self.h**2


def mbo_two_phase(grid, chi, tau, n_steps):
    """Two-phase MBO: diffuse, keep {u >= 1/2}. Returns the area after each step."""
    areas = [grid.area(chi)]
    for _ in range(n_steps):
        chi = (grid.heat(chi.astype(float), tau) >= 0.5).astype(np.uint8)
        areas.append(grid.area(chi))
        if areas[-1] == 0.0:
            break
    return chi, np.array(areas)


def mbo_multiphase(grid, labels, n_cells, tau, n_steps, snap_steps):
    """Multiphase MBO (argmax of diffused indicators) with the heat-content energy.

    E_tau = (1/sqrt tau) * sum_k int chi_k (1 - G_tau * chi_k): every interface
    counted from both sides.
    """
    energies, alive, snaps = [], [], {}
    for step in range(n_steps + 1):
        chi = np.stack([(labels == k) for k in range(n_cells)]).astype(float)
        u = np.stack([grid.heat(chi[k], tau) for k in range(n_cells)])
        inner = float((chi * u).sum()) * grid.h**2
        energies.append((1.0 - inner) / np.sqrt(tau))
        alive.append(int(np.unique(labels).size))
        if step in snap_steps:
            snaps[step] = labels.copy()
        if step < n_steps:
            labels = np.argmax(u, axis=0)
    return np.array(energies), np.array(alive), snaps


def periodic_voronoi(grid, seeds):
    dx = np.abs(grid.X[None] - seeds[:, 0, None, None])
    dy = np.abs(grid.Y[None] - seeds[:, 1, None, None])
    dx, dy = np.minimum(dx, 1 - dx), np.minimum(dy, 1 - dy)
    return np.argmin(dx**2 + dy**2, axis=0)


# --------------------------------------------------------------------------
# experiments
# --------------------------------------------------------------------------


def exp_a_displacement(out):
    """One step from a disc: the 1/2 level moves inward by kappa * tau."""
    radii = [0.1, 0.2, 0.3]
    ratios = np.geomspace(0.02, 0.6, 25)  # sqrt(tau) / R
    rows = {}
    for R in radii:
        taus = (ratios * R) ** 2
        d = np.array([R - half_level_radius(R, t) for t in taus])
        rows[R] = (ratios, d / (taus / R))
    table = []
    R = 0.2
    for q in [0.05, 0.1, 0.2, 0.4]:
        tau = (q * R) ** 2
        d = R - half_level_radius(R, tau)
        table.append(
            {
                "sqrt_tau_over_R": q,
                "tau": float(tau),
                "kappa_tau": float(tau / R),
                "displacement": float(d),
                "ratio": float(d / (tau / R)),
            }
        )
    out["A_displacement"] = {
        "description": "R - r_half for a disc of radius R after one Gaussian step; "
        "ratio = displacement / (kappa tau), kappa = 1/R",
        "R": R,
        "table": table,
    }

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    tau = 0.02**2
    x = np.linspace(-0.08, 0.08, 400)
    ax0.plot(x, flat_profile(x, tau), color="0.5", label="flat interface")
    ax0.plot(x, disc_profile(0.1 + x, 0.1, tau), color="C0", label=r"disc, $R=0.1$")
    ax0.axhline(0.5, color="k", lw=0.6, ls=":")
    ax0.axvline(0.0, color="k", lw=0.6, ls=":")
    xh = half_level_radius(0.1, tau) - 0.1
    ax0.plot([xh], [0.5], "o", color="C0", ms=4)
    ax0.annotate(
        r"$\approx -\kappa\tau$",
        (xh, 0.5),
        xytext=(-0.075, 0.25),
        arrowprops={"arrowstyle": "->", "lw": 0.6},
    )
    ax0.set_xlabel("signed distance from the original boundary")
    ax0.set_ylabel(r"$u = G_\tau * \chi$")
    ax0.set_title(r"Diffused indicator, $\sqrt{\tau} = 0.02$")
    ax0.legend(frameon=False, loc="upper right")
    for i, R in enumerate(radii):
        q, ratio = rows[R]
        ax1.plot(q, ratio, "o-", ms=2.5, color=f"C{i}", label=rf"$R={R}$")
    ax1.axhline(1.0, color="k", lw=0.6, ls=":")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$\sqrt{\tau}/R$")
    ax1.set_ylabel(r"displacement $/\ \kappa\tau$")
    ax1.set_title("One-step displacement, exact")
    ax1.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "ch01_displacement.pdf")
    plt.close(fig)


def exp_b_shrinking_disc(out):
    """Repeated MBO on a pixel disc: R^2 = R0^2 - 2t, and the tau window."""
    n, R0 = 512, 0.25
    g = Grid(n)
    chi0 = ((g.X - 0.5) ** 2 + (g.Y - 0.5) ** 2 <= R0**2).astype(np.uint8)
    t_end = R0**2 / 2
    cases = {"pinned": 2.0e-5, "in_window": 1.0e-3, "too_large": 1.0e-2}
    rec = {
        "n": n,
        "h": g.h,
        "R0": R0,
        "extinction_time_theory": t_end,
        "cases": {},
    }
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    tt = np.linspace(0, t_end, 100)
    ax.plot(tt, R0**2 - 2 * tt, color="k", lw=0.8, label=r"$R_0^2 - 2t$")
    for i, (name, tau) in enumerate(cases.items()):
        steps = int(np.ceil(1.2 * t_end / tau))
        _, areas = mbo_two_phase(g, chi0, tau, steps)
        t = tau * np.arange(len(areas))
        r2 = areas / np.pi
        ext = float(t[np.argmax(areas == 0)]) if (areas == 0).any() else None
        half = int(round(0.5 * t_end / tau))
        rec["cases"][name] = {
            "tau": tau,
            "kappa0_tau_over_h": float(tau / R0 / g.h),
            "sqrt_tau_over_R0": float(np.sqrt(tau) / R0),
            "steps_run": int(len(areas) - 1),
            "extinction_time": ext,
            "R2_at_half_extinction": float(r2[half]) if half < len(r2) else None,
            "R2_theory_at_half_extinction": float(R0**2 - 2 * tau * half),
        }
        ax.plot(t, r2, "o", ms=1.8 if len(t) < 100 else 0.6, color=f"C{i}",
                label=rf"$\tau = {tau:g}$")
    ax.set_xlabel("time $t$ = steps $\\times\\ \\tau$")
    ax.set_ylabel(r"$R^2 = $ area$/\pi$")
    ax.set_title("Shrinking disc under MBO")
    ax.set_ylim(-0.003, 0.068)
    ax.legend(frameon=False, fontsize=7.5, markerscale=3)
    fig.tight_layout()
    fig.savefig(FIG / "ch01_shrinking_disc.pdf")
    plt.close(fig)
    out["B_shrinking_disc"] = rec


def exp_c_dumbbell(out):
    """A non-convex set becomes convex, then shrinks; its area falls at rate 2 pi.

    In the plane, curve shortening never pinches an embedded curve (Grayson), and
    dA/dt = -(integral of kappa ds) = -2 pi for any simple closed curve.
    """
    n, tau = 512, 5.0e-4
    g = Grid(n)
    blob = lambda cx, r: (g.X - cx) ** 2 + (g.Y - 0.5) ** 2 <= r**2  # noqa: E731
    bar = (np.abs(g.X - 0.5) <= 0.2) & (np.abs(g.Y - 0.5) <= 0.035)
    sq = (np.abs(g.X - 0.72) <= 0.12) & (np.abs(g.Y - 0.5) <= 0.12)
    chi = (blob(0.27, 0.15) | bar | sq).astype(np.uint8)
    snaps_t = [0.0, 0.002, 0.008, 0.016]
    fig, axes = plt.subplots(1, len(snaps_t), figsize=(7.2, 1.9))
    t, step, done = 0.0, 0, []
    for ax, ts in zip(axes, snaps_t):
        while t < ts - 1e-12:
            chi = (g.heat(chi.astype(float), tau) >= 0.5).astype(np.uint8)
            t, step = t + tau, step + 1
        ax.imshow(chi.T, origin="lower", cmap="Greys", extent=(0, 1, 0, 1),
                  vmin=0, vmax=1.6)
        ax.set_xlim(0.05, 0.95)
        ax.set_ylim(0.25, 0.75)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"$t = {ts:g}$ ({step} steps)")
        a = g.area(chi)
        rate = (done[0]["area"] - a) / ts if done else None
        done.append({"t": ts, "steps": step, "area": a, "mean_area_loss_rate": rate})
    fig.tight_layout()
    fig.savefig(FIG / "ch01_dumbbell.pdf")
    plt.close(fig)
    out["C_dumbbell"] = {"n": n, "tau": tau, "theory_rate_2pi": 2 * np.pi,
                          "snapshots": done}


def exp_d_heat_content(out):
    """Two-phase heat content of a disc tends to Per / sqrt(pi)."""
    R = 0.2
    per_c = 2 * np.pi * R / np.sqrt(np.pi)
    table = []
    for q in [0.4, 0.2, 0.1, 0.05, 0.025]:
        tau = (q * R) ** 2
        E = disc_heat_content(R, tau)
        table.append(
            {"sqrt_tau_over_R": q, "E_tau": float(E), "ratio_to_limit": float(E / per_c)}
        )
    out["D_heat_content"] = {
        "description": "(1/sqrt tau) int_disc (1 - G_tau*chi) against 2 pi R / sqrt(pi)",
        "R": R,
        "limit_Per_over_sqrt_pi": float(per_c),
        "table": table,
    }


def exp_e_multiphase(out):
    """Multiphase MBO on the flat torus: argmax step, monotone energy, cells vanish."""
    n, n_cells, tau, n_steps = 256, 16, 1.0e-3, 400
    g = Grid(n)
    rng = np.random.default_rng(SEED)
    seeds = rng.random((n_cells, 2))
    labels = periodic_voronoi(g, seeds)
    snap_steps = [0, 10, 100, 400]
    E, alive, snaps = mbo_multiphase(g, labels, n_cells, tau, n_steps, snap_steps)
    dE = np.diff(E)
    out["E_multiphase"] = {
        "n": n,
        "n_cells": n_cells,
        "tau": tau,
        "seed": SEED,
        "steps": n_steps,
        "E_initial": float(E[0]),
        "E_final": float(E[-1]),
        "max_energy_increase": float(dE.max()),
        "alive_at": {int(s): int(alive[s]) for s in snap_steps},
        "cell_radius_initial": float(np.sqrt(1 / n_cells / np.pi)),
    }

    cmap = plt.get_cmap("tab20")
    fig = plt.figure(figsize=(7.2, 2.3))
    for j, s in enumerate(snap_steps):
        ax = fig.add_axes([0.005 + 0.165 * j, 0.08, 0.155, 0.8])
        ax.imshow(snaps[s].T % 20, origin="lower", cmap=cmap, vmin=0, vmax=19,
                  extent=(0, 1, 0, 1), interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"step {s}: {alive[s]} cells", fontsize=8)
    ax = fig.add_axes([0.70, 0.2, 0.29, 0.68])
    ax.plot(np.arange(n_steps + 1), E, color="C0", lw=1)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("step")
    ax.set_ylabel(r"$E_\tau$")
    ax.set_title("Heat-content energy", fontsize=8)
    fig.savefig(FIG / "ch01_multiphase.pdf")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    out = {
        "provenance": {
            "script": "docs/study/scripts/ch01_mbo.py",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "seed": SEED,
            "heat_equation": "u_t = Laplace u; kernel N(0, 2 tau I)",
        }
    }
    for exp in (exp_a_displacement, exp_b_shrinking_disc, exp_c_dumbbell,
                exp_d_heat_content, exp_e_multiphase):
        print(f"[ch01] {exp.__name__}", file=sys.stderr)
        exp(out)
    with open(DATA / "ch01.yaml", "w") as f:
        yaml.safe_dump(out, f, sort_keys=False, width=88)
    print(yaml.safe_dump(out, sort_keys=False, width=88))


if __name__ == "__main__":
    main()
