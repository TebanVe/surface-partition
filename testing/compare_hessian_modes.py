#!/usr/bin/env python3
"""Compare IPOPT L-BFGS vs. exact-Hessian on the same problem.

Two kinds of output on the same run:

  1. End-to-end totals per mode — final perimeter, constraint violation,
     IPOPT iteration count, wall-clock time, success flag.
  2. Per-component breakdown of the exact-Hessian path — wall time in
     compute_perimeter_hessian_sparse, compute_area_hessian_sparse, the two
     analytical Steiner Hessians, the Python-side sum, IPOPT's PDSystemSolver
     (linear algebra), and an "other" bucket.

Reading the breakdown:
  * Steiner rows dominate       -> unexpected post-Phase-3; the analytical
    Steiner Hessian is O(n_tp) — profile with cProfile.
  * perimeter/area_hess dominate-> unexpected; profile with cProfile.
  * IPOPT PDSystemSolver dominates -> a linear-algebra problem (try MA57, or
    Tier 3 of docs/reference/SCALABILITY_ANALYSIS.md).

This script is informational — it asserts nothing about which mode wins.

Implementation notes:
  * The four Hessian kernels are reached via module-qualified attribute access
    inside perimeter_optimizer.py, so monkey-patching the module attributes
    works.
  * IPOPT writes its timing block via C-level stdout (fd 1), which Python's
    contextlib.redirect_stdout does NOT capture — so this script redirects the
    file descriptor directly.

Usage:
    python testing/compare_hessian_modes.py --solution <path.h5>
    python testing/compare_hessian_modes.py --solution <path.h5> --no-profile
"""

import argparse
import contextlib
import os
import re
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from _hessian_test_utils import build_optimizer  # noqa: E402
from src.profiling import ProfilingState  # noqa: E402
from src.partition import vectorized_perimeter as _vperim  # noqa: E402
from src.partition import vectorized_area as _varea  # noqa: E402
from src.partition import vectorized_steiner as _vstein  # noqa: E402


# ---------------------------------------------------------------------------
# Per-component profiler — monkey-patches the four Hessian kernels.
# ---------------------------------------------------------------------------

_PROFILE = defaultdict(lambda: {'calls': 0, 'time': 0.0})


def _timed(name, fn):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        _PROFILE[name]['calls'] += 1
        _PROFILE[name]['time'] += time.perf_counter() - t0
        return out
    wrapper.__wrapped__ = fn
    return wrapper


@contextlib.contextmanager
def profile_hessian_components():
    """Time the four Hessian kernels for the duration of the block.

    Use around a single IPOPT run — wrapping two would double-count.
    """
    originals = {
        ('_vperim', 'compute_perimeter_hessian_sparse'):
            _vperim.compute_perimeter_hessian_sparse,
        ('_varea', 'compute_area_hessian_sparse'):
            _varea.compute_area_hessian_sparse,
        ('_vstein', 'compute_steiner_perimeter_hessian'):
            _vstein.compute_steiner_perimeter_hessian,
        ('_vstein', 'compute_steiner_area_hessian'):
            _vstein.compute_steiner_area_hessian,
    }
    _PROFILE.clear()
    _vperim.compute_perimeter_hessian_sparse = _timed(
        'perimeter_hess', _vperim.compute_perimeter_hessian_sparse)
    _varea.compute_area_hessian_sparse = _timed(
        'area_hess', _varea.compute_area_hessian_sparse)
    _vstein.compute_steiner_perimeter_hessian = _timed(
        'steiner_perim_hess', _vstein.compute_steiner_perimeter_hessian)
    _vstein.compute_steiner_area_hessian = _timed(
        'steiner_area_hess', _vstein.compute_steiner_area_hessian)
    try:
        yield _PROFILE
    finally:
        modules = {'_vperim': _vperim, '_varea': _varea, '_vstein': _vstein}
        for (mod_name, attr), orig in originals.items():
            setattr(modules[mod_name], attr, orig)


# ---------------------------------------------------------------------------
# C-level stdout capture (IPOPT prints via printf, not Python sys.stdout).
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def capture_fd_stdout():
    """Capture file-descriptor-1 output into ``captured['text']``."""
    captured = {'text': ''}
    saved_fd = os.dup(1)
    tmp = tempfile.TemporaryFile(mode='w+b')
    sys.stdout.flush()
    os.dup2(tmp.fileno(), 1)
    try:
        yield captured
    finally:
        sys.stdout.flush()
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
        tmp.flush()
        tmp.seek(0)
        captured['text'] = tmp.read().decode('utf-8', errors='replace')
        tmp.close()


# IPOPT 3.14 timing block labels (dot-padded to a fixed column):
#   OverallAlgorithm....................:  0.198 (sys: 0.001 wall: 0.199)
#   PDSystemSolverTotal.................:  0.027 (sys: 0.001 wall: 0.028)
_PD_RE = re.compile(r'PDSystemSolverTotal\.*:\s*\S+\s*\(sys:\s*\S+\s*wall:\s*(\S+)\)')
_OA_RE = re.compile(r'OverallAlgorithm\.*:\s*\S+\s*\(sys:\s*\S+\s*wall:\s*(\S+)\)')


def _extract_ipopt_timing(log_text):
    pd = _PD_RE.search(log_text)
    oa = _OA_RE.search(log_text)

    def _f(match):
        if match is None:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    return {'pd_solver_time': _f(pd), 'overall_ipopt_time': _f(oa)}


# ---------------------------------------------------------------------------
# One-mode runner.
# ---------------------------------------------------------------------------

def run_once(solution_path, exact_hessian, max_iter, tol, do_profile):
    opt = build_optimizer(solution_path)
    pa = opt._arrays
    profile = ProfilingState(
        n_cells=pa.n_cells, n_active_vps=pa.n_active_vp,
        n_triple_points=pa.n_triple_points)

    component_profile = None
    t0 = time.perf_counter()
    if do_profile:
        with profile_hessian_components() as comp, capture_fd_stdout() as cap:
            result = opt.optimize(
                max_iter=max_iter, tol=tol, method='ipopt',
                exact_hessian=exact_hessian, profile=profile,
                extra_ipopt_options={'print_timing_statistics': 'yes'})
        component_profile = {k: dict(v) for k, v in comp.items()}
    else:
        with capture_fd_stdout() as cap:
            result = opt.optimize(
                max_iter=max_iter, tol=tol, method='ipopt',
                exact_hessian=exact_hessian, profile=profile,
                extra_ipopt_options={'print_timing_statistics': 'yes'})
    elapsed = time.perf_counter() - t0

    ipopt_timing = _extract_ipopt_timing(cap['text'])
    final_viol = float(np.max(np.abs(opt.constraint_area_equality(result.x))))

    return {
        'final_perimeter': float(result.fun),
        'final_viol': final_viol,
        'iters': int(profile.ipopt_iter_count),
        'success': bool(result.success),
        'time': elapsed,
        'ipopt_timing': ipopt_timing,
        'profile': component_profile,
    }


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------

def _fmt_row(label, seconds, calls, total_wall):
    pct = 100.0 * seconds / total_wall if total_wall > 0 else 0.0
    per_call = 1000.0 * seconds / calls if calls > 0 else 0.0
    return (f"  {label:32s} {seconds:8.3f} s  {pct:5.1f}%  "
            f"{calls:6d} calls  {per_call:8.3f} ms/call")


def print_component_breakdown(result_exact):
    total = result_exact['time']
    prof = result_exact['profile']
    ipopt_t = result_exact['ipopt_timing']

    print(f"\n  --- Per-component breakdown (total run = {total:.3f} s) ---")
    hess_keys = ('perimeter_hess', 'area_hess',
                 'steiner_perim_hess', 'steiner_area_hess')
    for name in hess_keys:
        info = prof.get(name, {'calls': 0, 'time': 0.0})
        print(_fmt_row(name, info['time'], info['calls'], total))

    py_hess = sum(prof.get(k, {'time': 0.0})['time'] for k in hess_keys)
    print(_fmt_row("Sum Python Hessian kernels", py_hess, 0, total))

    pd = ipopt_t['pd_solver_time']
    if pd is not None:
        pct = 100.0 * pd / total if total > 0 else 0.0
        print(f"  {'IPOPT PDSystemSolver (lin. alg.)':32s} {pd:8.3f} s  "
              f"{pct:5.1f}%   (print_timing_statistics)")
        other = max(0.0, total - py_hess - pd)
        pct = 100.0 * other / total if total > 0 else 0.0
        print(f"  {'other (f, g, c, jac, overhead)':32s} {other:8.3f} s  {pct:5.1f}%")
    else:
        print("  IPOPT PDSystemSolver: not parsed from stdout "
              "(print_timing_statistics block not found).")


def _print_totals(label, r):
    print(f"=== {label} ===")
    for k in ('final_perimeter', 'final_viol', 'iters', 'time', 'success'):
        print(f"  {k:17s} {r[k]}")
    if r['ipopt_timing']['pd_solver_time'] is not None:
        print(f"  {'PDSystemSolver':17s} {r['ipopt_timing']['pd_solver_time']:.3f} s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--solution', required=True)
    ap.add_argument('--max-iter', type=int, default=200)
    ap.add_argument('--tol', type=float, default=1e-7)
    ap.add_argument('--no-profile', action='store_true',
                    help='Skip the per-component breakdown (faster).')
    args = ap.parse_args()

    r_lbfgs = run_once(args.solution, exact_hessian=False,
                       max_iter=args.max_iter, tol=args.tol, do_profile=False)
    _print_totals("L-BFGS", r_lbfgs)

    print()
    r_exact = run_once(args.solution, exact_hessian=True,
                       max_iter=args.max_iter, tol=args.tol,
                       do_profile=not args.no_profile)
    _print_totals("Exact Hessian", r_exact)

    if r_exact['profile'] is not None:
        print_component_breakdown(r_exact)

    print("\n=== Summary ===")
    dp = r_exact['final_perimeter'] - r_lbfgs['final_perimeter']
    di = r_exact['iters'] - r_lbfgs['iters']
    dt = r_exact['time'] - r_lbfgs['time']
    print(f"  Δ perimeter (exact − lbfgs) = {dp:+.6e}")
    print(f"  Δ iters     (exact − lbfgs) = {di:+d}")
    print(f"  Δ wall time (exact − lbfgs) = {dt:+.2f} s")
    print(f"  per-iter  exact  = {r_exact['time'] / max(1, r_exact['iters']):.3f} s")
    print(f"  per-iter  lbfgs  = {r_lbfgs['time'] / max(1, r_lbfgs['iters']):.3f} s")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
