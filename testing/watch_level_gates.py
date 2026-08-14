#!/usr/bin/env python3
"""Run the three validity gates on every level checkpoint, as the run produces them.

`run_relaxation` writes `solution/checkpoint_level{L}.h5` after each completed
level but keeps only the newest, and deletes all of them once the final solution
is written. So by the time a run finishes, the per-level history is gone — and
that history is exactly what you need to answer *where* a defect first appears,
rather than only whether it is present at the end.

This polls a run directory, copies each new checkpoint aside before it can be
replaced, runs `detect_dormant_cells` / `detect_area_imbalance` /
`detect_disconnected_cells` on it, and appends a one-line-per-level summary. It
reports dormant/weak separately from fragmented, because at low
vertices-per-cell "the interface cannot be resolved at all" and "a cell has been
torn into pieces" are different failures that must not be conflated.

Run it alongside the relaxation, from the repo root:

    python testing/watch_level_gates.py --run results/run_<...> &

or point it at a finished run's kept checkpoints / final solution to grade
whatever is still on disk.

Usage:
    python testing/watch_level_gates.py --run <run_dir> [--interval 120]
    python testing/watch_level_gates.py --run <run_dir> --once
"""

import argparse
import glob
import os
import shutil
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mesh.tri_mesh import TriMesh  # noqa: E402
from src.partition.find_contours import (  # noqa: E402
    detect_dormant_cells,
    detect_area_imbalance,
    detect_disconnected_cells,
)


def grade(path):
    """Return a dict of the three gates for one solution/checkpoint file."""
    with h5py.File(path, "r") as f:
        V = f["vertices"][:]
        F = f["faces"][:].astype(np.int64)
        x = f["x_opt"][:]
        n = int(f.attrs["n_partitions"])
        completed = int(f.attrs.get("completed_levels", -1))
    u = x.reshape(V.shape[0], n)
    mesh = TriMesh(V, F)
    mesh.compute_matrices()
    v = np.asarray(mesh.v).ravel()
    dorm = detect_dormant_cells(u)
    imb = detect_area_imbalance(u, v, n)
    disc = detect_disconnected_cells(u, F, v)
    return {
        "completed_levels": completed,
        "V": V.shape[0],
        "verts_per_cell": V.shape[0] / n,
        "dead": len(dorm.get("dead_cells", [])),
        "weak": len(dorm.get("weak_cells", [])),
        "min_peak": float(min(dorm["max_density_per_cell"])),
        "imbalanced": int(imb["n_imbalanced"]),
        "worst_dev": float(imb["worst_rel_dev"]),
        "fragmented": int(disc["n_fragmented"]),
        "worst_stray": float(disc.get("worst_stray_rel", 0.0)),
        "frag_cells": sorted(disc.get("fragmented", [])),
    }


def line(tag, g):
    # dormant/weak deliberately printed apart from fragmented: at low
    # verts/cell they are a different failure, not a milder one.
    return (
        f"{tag:<22s} V={g['V']:>7d} ({g['verts_per_cell']:>6.0f} v/cell)  "
        f"DORMANT dead={g['dead']:<3d} weak={g['weak']:<3d} minpeak={g['min_peak']:.4f}  |  "
        f"IMBALANCE n={g['imbalanced']:<3d} worst={g['worst_dev']*100:>6.2f}%  |  "
        f"CONNECTIVITY frag={g['fragmented']:<3d} stray={g['worst_stray']*100:>6.2f}% "
        f"{g['frag_cells'] if g['frag_cells'] else ''}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument(
        "--once", action="store_true", help="grade what is on disk and exit"
    )
    args = ap.parse_args()

    keep = os.path.join(args.run, "level_gates")
    os.makedirs(keep, exist_ok=True)
    report = os.path.join(keep, "level_gates.txt")
    seen = set(os.path.basename(p) for p in glob.glob(os.path.join(keep, "*.h5")))

    def emit(text):
        print(text, flush=True)
        with open(report, "a") as fh:
            fh.write(text + "\n")

    while True:
        # per-level checkpoints (the pipeline keeps only the newest)
        for p in sorted(
            glob.glob(os.path.join(args.run, "solution", "checkpoint_level*.h5"))
        ):
            name = os.path.basename(p)
            if name in seen:
                continue
            dst = os.path.join(keep, name)
            try:
                shutil.copy2(p, dst)
            except (OSError, IOError):
                continue  # mid-write; pick it up next pass
            seen.add(name)
            try:
                emit(line(name.replace(".h5", ""), grade(dst)))
            except Exception as e:  # a partially written copy
                emit(f"{name}: could not grade yet ({e})")
                seen.discard(name)

        # the final solution, once it exists
        finals = [
            f
            for f in glob.glob(os.path.join(args.run, "solution", "*.h5"))
            if "checkpoint" not in os.path.basename(f)
        ]
        if finals and "FINAL" not in seen:
            seen.add("FINAL")
            emit(line("FINAL SOLUTION", grade(finals[0])))
            emit("run complete")
            return 0

        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
