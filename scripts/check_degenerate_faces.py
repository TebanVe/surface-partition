#!/usr/bin/env python3
"""Count degenerate sub-faces and coincident vertices in the exported partitions.

Regenerates every number quoted in `docs/downstream/issue-4-collapsed-faces-reply.md`
(surface-partition issue #4). Two criteria are reported, because they differ by
orders of magnitude and the distinction is the substance of that issue:

* **exact-zero** -- ``area == 0.0``, what the issue counted;
* **relative mask** -- ``area <= 1e-10 * median(positive areas)``, what the
  consumer's ``FaceLabeledMesh.degenerate_face_mask`` actually excludes.

    python scripts/check_degenerate_faces.py            # table
    python scripts/check_degenerate_faces.py --emit      # write the YAML record

Deterministic: no seeds, no sampling. Rows come from
`docs/reference/deliverables.yaml`, so a file absent from this checkout is
skipped and reported (``results/`` is gitignored and per-worktree).
"""
from __future__ import annotations

import argparse
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
RECORD = REPO / "docs" / "reference" / "deliverables.yaml"
OUT = REPO / "docs" / "downstream" / "degenerate_faces.yaml"

#: The consumer's threshold, from
#: link-list-torus/src/lnk_list_torus/mesh/face_labeled_mesh.py
DEGENERATE_AREA_REL = 1e-10

#: Decimals used when detecting coincident vertices -- the issue's own snippet.
COINCIDENT_DECIMALS = 12


def measure(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        verts = f["/partition/sub_vertices"][:]
        faces = f["/partition/sub_faces"][:]
    p = verts[faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1
    )
    positive = areas[areas > 0.0]
    median = float(np.median(positive)) if positive.size else 0.0
    unique = np.unique(np.round(verts, COINCIDENT_DECIMALS), axis=0)
    return {
        "sub_faces": int(len(faces)),
        "sub_vertices": int(len(verts)),
        "exact_zero_faces": int((areas == 0.0).sum()),
        "relative_mask_faces": int((areas <= DEGENERATE_AREA_REL * median).sum()),
        "coincident_vertices": int(len(verts) - len(unique)),
        "median_positive_area": median,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--emit", action="store_true",
                    help=f"write the YAML record to {OUT.relative_to(REPO)}")
    args = ap.parse_args()

    rows = yaml.safe_load(RECORD.read_text())["deliverables"]
    out: list[dict] = []
    absent = 0

    header = (f"{'export':<44}{'fin':>5}{'F_sub':>9}{'zero':>6}"
              f"{'<=relxmed':>11}{'coincid':>9}")
    print(header)
    print("-" * len(header))
    for row in sorted(rows, key=lambda r: (r["group"], r["n"], r["mesh_vertices"])):
        path = REPO / row["file"]
        if not path.exists():
            absent += 1
            continue
        m = measure(path)
        label = f"{row['group']} N={row['n']} V={row['mesh_vertices']}"
        print(f"{label:<44}{str(row['finalised'])[0]:>5}{m['sub_faces']:>9}"
              f"{m['exact_zero_faces']:>6}{m['relative_mask_faces']:>11}"
              f"{m['coincident_vertices']:>9}")
        out.append({"file": row["file"], "group": row["group"], "n": row["n"],
                    "mesh_vertices": row["mesh_vertices"],
                    "finalised": row["finalised"], **m})

    fin = [r for r in out if r["finalised"]]
    with_zero = [r for r in fin if r["exact_zero_faces"] > 0]
    print(f"\n{len(out)} measured, {absent} absent from this checkout")
    print(f"finalised exports: {len(with_zero)} of {len(fin)} contain at least one "
          f"exactly-zero face")
    print("the relative mask always catches more than the exact-zero count: "
          f"{all(r['relative_mask_faces'] >= r['exact_zero_faces'] for r in out)}")

    if args.emit:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(yaml.safe_dump({
            "provenance": {
                "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "script": "scripts/check_degenerate_faces.py",
                "python": platform.python_version(),
                "numpy": np.__version__,
                "h5py": h5py.__version__,
                "platform": f"{platform.system()} {platform.release()} "
                            f"{platform.machine()}",
                "degenerate_area_rel": DEGENERATE_AREA_REL,
                "coincident_decimals": COINCIDENT_DECIMALS,
                "note": "Deterministic; no seeds. Absent files are skipped, "
                        "not failed: results/ is gitignored and per-worktree.",
            },
            "measurements": out,
        }, sort_keys=False, width=120))
        print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
