#!/usr/bin/env python3
"""
Smoke test for the triple-point sub-face subdivision in the consolidated
partition export (Rep-3 builder).

Each triple-point triangle must be subdivided into exactly **six** sub-faces
in ``/partition/sub_faces``: three corner sub-triangles
``(V_X, vp_to_next, vp_to_prev)`` plus three central sub-triangles
``(vp_to_prev, vp_to_next, S)``. Before the fix only the three central
sub-faces were emitted, leaving a visible hole at every triple point and
breaking the area invariant.

This test exports a Phase-2 checkpoint to a temporary HDF5 and asserts:

1. For every triple-point triangle: exactly six sub-faces are present in
   the closed star of its corners (and the Steiner vertex), they tile the
   parent area to ``1e-9`` relative tolerance, and their cell labels are a
   permutation of ``cell_triple[k]`` with each cell appearing twice.
2. Global count: ``F_sub == F_singletons + 3 * n_2 + 6 * n_tp`` where
   ``F_singletons`` is the count of single-label original faces, ``n_2`` is
   the count of two-label faces, and ``n_tp`` is the triple-point count.
3. ``/.attrs["schema_version"] == "1.1"``.

Usage:
    python testing/test_export_triple_point_subdivision.py \\
        --solution <path>/iteration_*.h5 \\
        --config parameters/torus_10part.yaml
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.export import export_partition
from src.partition.steiner_handler import SteinerHandler
from src.pipeline.io import find_base_solution_path, load_partition_from_refined_file


def _triangle_area(p0, p1, p2):
    return 0.5 * float(np.linalg.norm(np.cross(p1 - p0, p2 - p0)))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify triple-point sub-face subdivision in a fresh export"
    )
    parser.add_argument("--solution", required=True, help="Phase-2 checkpoint HDF5")
    parser.add_argument("--config", required=True, help="Experiment YAML")
    parser.add_argument(
        "--area-rtol", type=float, default=1e-9,
        help="Relative tolerance for parent-vs-child area equality (default: 1e-9)"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("EXPORT TRIPLE-POINT SUB-FACE SUBDIVISION SMOKE TEST")
    print("=" * 80)

    checkpoint_path = os.path.abspath(args.solution)
    with open(args.config, "r") as f:
        config = yaml.safe_load(f) or {}

    with h5py.File(checkpoint_path, "r") as f:
        source_iteration = int(f.attrs["iteration_number"])
        final_perimeter = float(f.attrs["final_perimeter"])
        pending_migration = bool(f.attrs.get("pending_migration", False))

    base_solution_path = find_base_solution_path(checkpoint_path, verbose=False)
    with h5py.File(base_solution_path, "r") as f:
        seed = int(f.attrs["seed"])
        n_theta_final = int(f.attrs["var1"])
        n_phi_final = int(f.attrs["var2"])

    mesh, partition = load_partition_from_refined_file(checkpoint_path, verbose=False)
    steiner_handler = SteinerHandler(mesh, partition)

    source_run_id = Path(checkpoint_path).parent.parent.parent.name

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_partition.h5")
        export_partition(
            partition=partition,
            mesh=mesh,
            steiner_handler=steiner_handler,
            config=config,
            output_path=output_path,
            source_run_id=source_run_id,
            source_iteration=source_iteration,
            seed=seed,
            final_perimeter=final_perimeter,
            pending_migration=pending_migration,
            n_theta_final=n_theta_final,
            n_phi_final=n_phi_final,
            strict=False,
        )

        with h5py.File(output_path, "r") as f:
            schema_version = str(f.attrs["schema_version"])
            sv = f["partition/sub_vertices"][:]
            sf = f["partition/sub_faces"][:]
            fl = f["partition/face_labels"][:]
            mv = f["mesh/vertices"][:]
            mf = f["mesh/faces"][:]
            tp_tris = f["snapshot/triple_points/triangle_index"][:]
            cell_triple = f["snapshot/triple_points/cell_triple"][:]
            steiner_xyz = f["snapshot/triple_points/steiner_xyz"][:]
            vp_edges = f["snapshot/vp_edges"][:]
            vertex_labels = f["snapshot/vertex_labels"][:]

    V = mv.shape[0]
    n_vp = vp_edges.shape[0]
    n_tp = len(tp_tris)

    print(f"\nschema_version : {schema_version}")
    print(f"V              : {V}")
    print(f"F (original)   : {mf.shape[0]}")
    print(f"F_sub          : {sf.shape[0]}")
    print(f"n_vp           : {n_vp}")
    print(f"n_tp           : {n_tp}")

    if schema_version != "1.1":
        print(f"\nRESULT: FAIL  (expected schema_version='1.1', got {schema_version!r})")
        return 1

    edge_to_vp_subidx = {
        frozenset((int(vp_edges[i, 0]), int(vp_edges[i, 1]))): V + i
        for i in range(n_vp)
    }
    tp_set = set(int(t) for t in tp_tris)

    # ---- Per-triple-point invariants ----
    worst_rel = 0.0
    tp_failures = []
    for k in range(n_tp):
        t = int(tp_tris[k])
        a, b, c = (int(x) for x in mf[t])
        parent_area = _triangle_area(mv[a], mv[b], mv[c])

        steiner_idx = None
        for i in range(V + n_vp, sv.shape[0]):
            if np.allclose(sv[i], steiner_xyz[k], atol=0, rtol=0):
                steiner_idx = i
                break
        if steiner_idx is None:
            tp_failures.append(f"TP{k} (tri={t}): could not locate Steiner sub-vertex")
            continue

        vp_ab = edge_to_vp_subidx.get(frozenset((a, b)))
        vp_bc = edge_to_vp_subidx.get(frozenset((b, c)))
        vp_ca = edge_to_vp_subidx.get(frozenset((c, a)))
        if None in (vp_ab, vp_bc, vp_ca):
            tp_failures.append(
                f"TP{k} (tri={t}): missing a VP on one of the three edges"
            )
            continue

        allowed = {a, b, c, vp_ab, vp_bc, vp_ca, steiner_idx}
        mask = np.array([all(int(x) in allowed for x in row) for row in sf])
        idxs = np.where(mask)[0]
        n_sub = len(idxs)

        child_area = 0.0
        labels_here = []
        for i in idxs:
            child_area += _triangle_area(sv[sf[i, 0]], sv[sf[i, 1]], sv[sf[i, 2]])
            labels_here.append(int(fl[i]))
        rel = abs(parent_area - child_area) / parent_area
        worst_rel = max(worst_rel, rel)

        expected_labels = sorted([int(x) for x in cell_triple[k]] * 2)
        label_ok = sorted(labels_here) == expected_labels

        if n_sub != 6:
            tp_failures.append(f"TP{k} (tri={t}): {n_sub} sub-faces, expected 6")
        elif rel > args.area_rtol:
            tp_failures.append(
                f"TP{k} (tri={t}): area rel error {rel:.3e} > {args.area_rtol}"
            )
        elif not label_ok:
            tp_failures.append(
                f"TP{k} (tri={t}): labels {sorted(labels_here)} != {expected_labels}"
            )

    print(f"worst per-TP area rel error: {worst_rel:.3e}")

    # ---- Global count invariant ----
    n_singleton = 0
    n_2 = 0
    for face in mf:
        a, b, c = (int(x) for x in face)
        labset = {int(vertex_labels[a]), int(vertex_labels[b]), int(vertex_labels[c])}
        if len(labset) == 1:
            n_singleton += 1
        elif len(labset) == 2:
            n_2 += 1
    expected_F_sub = n_singleton + 3 * n_2 + 6 * n_tp

    print(f"n_singleton    : {n_singleton}")
    print(f"n_2 (two-label): {n_2}")
    print(f"expected F_sub : {n_singleton} + 3*{n_2} + 6*{n_tp} = {expected_F_sub}")
    print(f"actual F_sub   : {sf.shape[0]}")

    count_ok = expected_F_sub == sf.shape[0]

    if tp_failures or not count_ok:
        print()
        for msg in tp_failures:
            print(f"  FAIL: {msg}")
        if not count_ok:
            print(f"  FAIL: F_sub count mismatch ({sf.shape[0]} != {expected_F_sub})")
        print("\nRESULT: FAIL")
        return 1

    print("\nRESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
