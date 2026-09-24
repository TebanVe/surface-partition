#!/usr/bin/env python3
"""Verify docs/reference/deliverables.yaml against the exported HDF5 files.

The YAML is the *record*: it lists every partition this project has exported,
including ones produced in a sibling git worktree. ``results/`` is gitignored,
so which files are present depends on the checkout -- a row whose file is absent
here is reported and skipped, never treated as an error and never deleted.

    python scripts/check_deliverables.py            # verify
    python scripts/check_deliverables.py --emit     # regenerate rows from disk

Exit status is non-zero only when a present file DISAGREES with its row.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import h5py
import yaml

REPO = Path(__file__).resolve().parent.parent
RECORD = REPO / "docs" / "reference" / "deliverables.yaml"

# Fields read straight off the exported file. `worst_cell_pct` is deliberately
# absent: it is Phase 1 discrete imbalance, which lives in the run's
# arm_report.yaml / solution/metadata.yaml and is not carried by the export.
FIELDS = {
    "n": ("attr", "n_cells", int),
    "iteration": ("attr", "source_iteration", int),
    "perimeter": ("attr", "final_perimeter", float),
    "finalised": ("attr", "finalised", bool),
    "seed": ("attr", "seed", int),
    "source_run_id": ("attr", "source_run_id", str),
    "mesh_vertices": ("shape", "mesh/vertices", int),
}

PERIMETER_TOL = 5e-5  # the record quotes 4 decimals


def read_file(path: Path) -> dict:
    out = {}
    with h5py.File(path, "r") as f:
        for key, (kind, name, cast) in FIELDS.items():
            if kind == "attr":
                out[key] = cast(f.attrs[name]) if name in f.attrs else None
            else:
                out[key] = cast(f[name].shape[0])
    return out


def emit() -> int:
    rows = []
    # Recursive: a readout-derived export nests one level deeper, under
    # <run>/readout/<campaign>/partition/, not <run>/partition/.
    pattern = str(REPO / "results" / "**" / "partition" / "*.h5")
    for p in sorted(glob.glob(pattern, recursive=True)):
        rel = Path(p).relative_to(REPO).as_posix()
        d = read_file(Path(p))
        d["file"] = rel
        d["perimeter"] = round(d["perimeter"], 4)
        rows.append(d)
    rows.sort(key=lambda r: (r["mesh_vertices"], r["n"]))
    print(yaml.safe_dump({"deliverables": rows}, sort_keys=False, width=100))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emit", action="store_true",
                    help="print YAML rows generated from the files on disk")
    args = ap.parse_args()
    if args.emit:
        return emit()

    record = yaml.safe_load(RECORD.read_text())
    rows = record["deliverables"]

    checked = absent = 0
    problems: list[str] = []

    for row in rows:
        path = REPO / row["file"]
        if not path.exists():
            absent += 1
            print(f"  absent here   N={row['n']:>4}  {row['file']}")
            continue
        checked += 1
        actual = read_file(path)
        diffs = []
        for key, value in actual.items():
            if key not in row:
                continue
            expected = row[key]
            if key == "perimeter":
                if abs(value - float(expected)) > PERIMETER_TOL:
                    diffs.append(f"{key}: record {expected} != file {value:.4f}")
            elif value != expected:
                diffs.append(f"{key}: record {expected!r} != file {value!r}")
        if diffs:
            problems.append(f"{row['file']}\n      " + "\n      ".join(diffs))
            print(f"  MISMATCH      N={row['n']:>4}  {row['file']}")
        else:
            print(f"  ok            N={row['n']:>4}  {row['file']}")

    print(f"\n{checked} verified, {absent} absent from this checkout, "
          f"{len(problems)} mismatched")
    if absent:
        print("Absent rows are expected: results/ is gitignored and each git "
              "worktree holds its own runs.")
    if problems:
        print("\nMISMATCHES:")
        for p in problems:
            print(f"  - {p}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
