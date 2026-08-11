#!/usr/bin/env python3
"""Replay the structure-based refinement trigger against completed runs.

The structure trigger (``struct_trigger_enabled``) is a stuck detector: it fires
when the winner-take-all label field ``argmax_k u_ik`` has been frozen for
``struct_window`` iterations at a flip rate below ``struct_rate_tol`` per
iteration per vertex. Turning it on changes when every level stops, so before
trusting it we replay it OFFLINE against runs whose behaviour we already know.

Two questions, and the second matters more:

1. Does it reclaim the pathology? On the N=100 deliverable it should fire on
   level 0 far short of the 30,000-iteration cap.
2. **Does it leave alone the runs that already behave?** On the N=300 five-level
   run every level triggered early on its own (3,838 and 2,247 against the same
   cap). The structure rule must not preempt those -- a trigger that reclaims
   27% at N=100 but destabilises N=300 is a bad trade.

Resolution caveat: this reads ``x`` from the ``traces/*_internal_data.hdf5``
files, which are written every ``h5_save_stride`` iterations (500 in these
runs). So the replay evaluates the rule on a 500-iteration grid and reports the
firing iteration to that granularity. That is ample for question 2, which turns
on whether a full window ever completes before the level ends.

Usage:
    python testing/validate_struct_trigger.py --run <run_dir> --n-partitions 100
    python testing/validate_struct_trigger.py --run <dir> --n-partitions 300 \\
        --window 2000 --rate-tol 1e-6
"""

import argparse
import glob
import os
import re
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def level_flip_series(path, n_partitions):
    """(iterations, flips_between_samples, V) from one level's trace."""
    with h5py.File(path, "r") as h:
        its = sorted(int(k.split("_")[1]) for k in h.keys() if k.startswith("iter_"))
        if not its or "x" not in h[f"iter_{its[0]}"]:
            return None, None, None
        flips, prev, V = [], None, None
        for i in its:
            x = h[f"iter_{i}"]["x"][:]
            V = x.size // n_partitions
            lab = np.argmax(x.reshape(V, n_partitions), axis=1).astype(np.int32)
            if prev is not None:
                flips.append(int(np.count_nonzero(lab != prev)))
            prev = lab
            del x
    return its, flips, V


def replay(its, flips, V, window, rate_tol):
    """First iteration at which the windowed rule would fire.

    Returns ``(fire_iteration_or_None, total_at_fire_or_None, budget_or_None)``.
    ``budget`` is None only when the level is too short to evaluate the rule at
    all (fewer than two samples), which is itself a valid "cannot fire" answer.

    The budget uses the EFFECTIVE window ``n_intervals * stride`` rather than
    the nominal ``window``, mirroring the live implementation: both round the
    window to a whole number of samples and price the budget on what they
    actually measured.
    """
    if len(its) < 2 or not flips:
        return None, None, None
    stride = its[1] - its[0]
    n_intervals = max(1, int(round(window / stride)))
    eff_window = n_intervals * stride
    budget = rate_tol * V * eff_window
    for j in range(len(flips)):
        if (j + 1) < n_intervals:
            continue  # not enough history for a full window yet
        total = sum(flips[j + 1 - n_intervals : j + 1])
        if total <= budget:
            return its[j + 1], total, budget
    return None, None, budget


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run directory")
    ap.add_argument("--n-partitions", type=int, required=True)
    ap.add_argument("--window", type=int, default=2000)
    ap.add_argument("--rate-tol", type=float, default=1e-6)
    ap.add_argument(
        "--verbose", action="store_true", help="print the per-sample flip series"
    )
    args = ap.parse_args()

    print(f"run        : {os.path.basename(args.run.rstrip('/'))}")
    print(f"rule       : <= rate_tol*V*window flips over {args.window} iterations")
    print(f"             (rate_tol={args.rate_tol:g} flips/iter/vertex)")
    print()

    traces = sorted(glob.glob(f"{args.run}/traces/*internal_data.hdf5"))
    if not traces:
        print("no traces found")
        return 1
    verdicts = []
    for t in traces:
        lvl = re.search(r"level(\d+)", t).group(1)
        its, flips, V = level_flip_series(t, args.n_partitions)
        if its is None:
            print(f"level {lvl}: no x data in trace")
            continue
        last_iter = its[-1]
        fire, total, budget = replay(its, flips, V, args.window, args.rate_tol)
        if args.verbose:
            for j, f in enumerate(flips):
                print(f"      iter {its[j+1]:6d}: {f:7d} flips / {V}")
        if budget is None:
            print(
                f"level {lvl} (V={V:>7}, trace ends {last_iter:>6}): "
                f"TOO SHORT to evaluate ({len(its)} sample(s)) -- cannot fire"
            )
            verdicts.append((lvl, None, last_iter, "too short"))
        elif fire is None:
            print(
                f"level {lvl} (V={V:>7}, trace ends {last_iter:>6}): "
                f"WOULD NOT FIRE  (budget {budget:.1f} per window)"
            )
            verdicts.append((lvl, None, last_iter, "never quiet enough"))
        else:
            print(
                f"level {lvl} (V={V:>7}, trace ends {last_iter:>6}): "
                f"would fire at ~{fire}  ({total} flips vs budget {budget:.1f})"
            )
            verdicts.append((lvl, fire, last_iter, None))

    print()
    print("interpretation:")
    for lvl, fire, last, why in verdicts:
        if fire is None:
            print(
                f"  level {lvl}: rule stays silent ({why}) -> the existing "
                f"energy/gnorm trigger governs, run UNCHANGED"
            )
        else:
            saved = last - fire
            pct = 100.0 * saved / last if last else 0.0
            print(
                f"  level {lvl}: fires at ~{fire} instead of {last}+ "
                f"-> ~{saved} iterations reclaimed ({pct:.0f}% of the level)"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
