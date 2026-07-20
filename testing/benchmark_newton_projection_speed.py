#!/usr/bin/env python3
"""Per-level speed + peak-memory benchmark: newton vs iterative projection.

Closes DoD item 7 of docs/plans/PHASE1_DUAL_NEWTON_PROJECTION_PLAN.md (sec. 8.4):
measure the exact-dual projection against the incumbent iterative method at each
refinement level, INCLUDING a fine level (V ~ 1.7e5) where the 86-91% wall-time
motivation lives. A coarse-only measurement understates (or inverts) the win --
at V ~ 5k the newton path is only ~2x warm and can be slower cold.

For each resolution it times three things over an identical, deterministic
sequence of PGD-like candidates Y_k (small random-walk drift, so consecutive
projections are close and the warm-started dual actually helps):

  * iterative : orthogonal_projection_iterative(clip(Y), ...) tol=1e-8, max_iter=300
                (the optimizer's real call, WITH the [1e-8, 1-1e-8] pre-clip)
  * newton-cold: orthogonal_projection_newton(Y, ...) tol=1e-10, beta0=None each call
  * newton-warm: same, threading beta across steps (the PGD steady state)

and reports mean per-call ms, speedup ratios, and process peak RSS (ru_maxrss).
Run a single fine resolution in its own process for a clean peak-memory reading:

    python testing/benchmark_newton_projection_speed.py --resolutions 452,414 --n 10

Self-contained (synthetic inputs, torus lumped mass built directly). The default
ladder mirrors the torus_10part refinement levels (V ~ 5.3k -> 187k, N=10).
"""
import argparse
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.optimization.projection import (  # noqa: E402
    orthogonal_projection_newton,
    orthogonal_projection_iterative,
)
from src.surfaces.torus import TorusMeshProvider  # noqa: E402


def _peak_rss_mb() -> float:
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024 * 1024) if sys.platform == "darwin" else ru / 1024


class _Counter:
    """Minimal _prof shim: captures projection_inner_iters_total across calls.

    For the iterative method this is the alternating-projection iteration count
    (mPII in the timing profile, ~33 in real N=100 runs); for newton it is the
    L-BFGS outer count plus accepted polish steps.
    """

    def __init__(self):
        self.c = {}

    def record(self, cb, elapsed):
        pass

    def add_counter(self, name, n=1):
        self.c[name] = self.c.get(name, 0) + n

    def inner(self):
        return self.c.get("projection_inner_iters_total", 0)


def _torus_v(nt, npx):
    return np.asarray(TorusMeshProvider(nt, npx, 1.0, 0.6).build().v, float)


def _crisp_Y0(V, n, rng):
    """A near-one-hot density -- the crisp operating point late in Phase 1."""
    Y = 0.02 * rng.standard_normal((V, n))
    Y[np.arange(V), rng.integers(0, n, V)] += 1.0
    return np.clip(Y, 1e-8, 1 - 1e-8)


def _candidate_trajectory(Y0, n_steps, drift, rng):
    """A fixed PGD-like random walk; all methods project the identical list."""
    Y, out = Y0.copy(), []
    for _ in range(n_steps):
        Y = Y + drift * rng.standard_normal(Y.shape)
        out.append(Y.copy())
    return out


def _time_method(fn, Y_list, warmup, prof):
    """Mean per-call ms and mean inner iters over Y_list[warmup:]; last result."""
    per_call, inner = [], []
    last = None
    for k, Y in enumerate(Y_list):
        before = prof.inner()
        t0 = time.perf_counter()
        last = fn(Y)
        dt = time.perf_counter() - t0
        if k >= warmup:
            per_call.append(dt)
            inner.append(prof.inner() - before)
    return float(np.mean(per_call)) * 1e3, float(np.mean(inner)), last


def _area_err(U, v, d):
    return float(np.max(np.abs(v @ U - d)))


def bench_resolution(nt, npx, n, n_steps, drift, seed, iter_tol, newton_tol):
    rng = np.random.default_rng(seed)
    v = _torus_v(nt, npx)
    V = len(v)
    d = (v.sum() / n) * np.ones(n)
    ones = np.ones(n)
    Y0 = _crisp_Y0(V, n, rng)
    Y_list = _candidate_trajectory(Y0, n_steps, drift, rng)
    warmup = max(1, n_steps // 4)

    p_iter, p_cold, p_warm = _Counter(), _Counter(), _Counter()

    def f_iter(Y):
        return orthogonal_projection_iterative(
            np.clip(Y, 1e-8, 1 - 1e-8).copy(), ones, d, v, max_iter=300, tol=iter_tol,
            _prof=p_iter)

    def f_cold(Y):
        return orthogonal_projection_newton(Y, ones, d, v, tol=newton_tol, beta0=None,
                                            _prof=p_cold)

    # warm: thread beta across the trajectory (closure state)
    warm_state = {"beta": None}

    def f_warm(Y):
        U, beta = orthogonal_projection_newton(
            Y, ones, d, v, tol=newton_tol, beta0=warm_state["beta"], return_beta=True,
            _prof=p_warm)
        warm_state["beta"] = beta
        return U

    it_ms, it_inner, U_it = _time_method(f_iter, Y_list, warmup, p_iter)
    cold_ms, cold_inner, U_cold = _time_method(f_cold, Y_list, warmup, p_cold)
    warm_ms, warm_inner, U_warm = _time_method(f_warm, Y_list, warmup, p_warm)

    return {
        "nt": nt, "np": npx, "V": V, "N": n,
        "it_ms": it_ms, "cold_ms": cold_ms, "warm_ms": warm_ms,
        "it_inner": it_inner, "warm_inner": warm_inner,
        "speedup_warm": it_ms / warm_ms if warm_ms else float("nan"),
        "speedup_cold": it_ms / cold_ms if cold_ms else float("nan"),
        "area_it": _area_err(U_it, v, d),
        "area_warm": _area_err(U_warm, v, d),
        "rss_mb": _peak_rss_mb(),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resolutions", type=str,
                    default="80,66 204,182 328,298 452,414",
                    help="space-separated nt,np pairs (torus). Default: torus_10part ladder.")
    ap.add_argument("--n", type=int, default=10, help="number of regions N")
    ap.add_argument("--steps", type=int, default=15, help="PGD-like candidates timed per level")
    ap.add_argument("--drift", type=float, default=5e-3, help="per-step random-walk magnitude")
    ap.add_argument("--iter-tol", type=float, default=1e-8,
                    help="iterative projection tol (optimizer default 1e-8)")
    ap.add_argument("--newton-tol", type=float, default=1e-10,
                    help="newton projection tol. 1e-10 = current optimizer setting "
                         "(triggers O(V*N^2) polish); 1e-8 = tolerance-matched to iterative.")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    pairs = []
    for tok in args.resolutions.split():
        a, b = tok.split(",")
        pairs.append((int(a), int(b)))

    print(f"Newton vs iterative projection speed  (N={args.n}, steps={args.steps}, "
          f"drift={args.drift}, seed={args.seed}, iter_tol={args.iter_tol:.0e}, "
          f"newton_tol={args.newton_tol:.0e})")
    print("mPII_it = iterative inner iters (calibration target ~33 from real N=100 runs)")
    print(f"{'nt x np':>10} {'V':>8} | {'iter ms':>9} {'nt-cold':>9} {'nt-warm':>9} | "
          f"{'x warm':>7} {'x cold':>7} | {'mPII_it':>7} {'nt_out':>6} | "
          f"{'area_it':>8} {'area_wm':>8} | {'peakRSS':>8}")
    print("-" * 124)
    for nt, npx in pairs:
        r = bench_resolution(nt, npx, args.n, args.steps, args.drift, args.seed,
                             args.iter_tol, args.newton_tol)
        print(f"{nt:>4}x{npx:<5} {r['V']:>8} | {r['it_ms']:>9.2f} {r['cold_ms']:>9.2f} "
              f"{r['warm_ms']:>9.2f} | {r['speedup_warm']:>6.2f}x {r['speedup_cold']:>6.2f}x | "
              f"{r['it_inner']:>7.1f} {r['warm_inner']:>6.1f} | "
              f"{r['area_it']:>8.1e} {r['area_warm']:>8.1e} | {r['rss_mb']:>6.0f}MB")

    print("\nRatios > 1 mean newton is faster. The fine-level (rightmost) warm ratio is the "
          "headline number; peak RSS is the memory footprint at that size.")
    print("Check mPII_it ~ 33: if far below, drift is too small (candidates too near-feasible, "
          "unfairly favouring iterative) -- raise --drift and re-run.")


if __name__ == "__main__":
    main()
