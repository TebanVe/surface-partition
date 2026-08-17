#!/usr/bin/env python3
"""Phase 0b acceptance: the evaluation harness itself is correct.

Cheap structural checks only -- no Phase 2, no runs. The expensive end-to-end
check is Phase 0d, which pushes the N=100 control's own labels through the whole
harness and must reproduce its recorded perimeter 185.2546144718457.

What is checked here:
  1. one-hot round trip: labels -> densities -> argmax recovers the labels;
  2. the written solution file is readable by the SAME loader the gates and
     Phase 2 use, and its argmax is unchanged;
  3. hardening a continuous field to one-hot does not move the gates -- the
     claim that an arm's hard labels and PGD's continuous field are scored on
     equal terms. Verified on a real N=300 solution;
  4. the vertex-granularity floor matches the value the shipped readout records;
  5. censoring detection: best-is-last is flagged, best-in-middle is not.

Usage:  python testing/test_arm_harness.py
"""
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mesh.tri_mesh import TriMesh  # noqa: E402
from src.partition.arm_harness import (  # noqa: E402
    Phase2Report,
    labels_to_one_hot,
    run_gates,
    vertex_granularity,
    write_arm_solution,
)
from src.surfaces.torus import TorusMeshProvider  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def test_roundtrip(tmp):
    print("\n1/5  one-hot round trip and file round trip")
    mesh = TorusMeshProvider(n_theta=40, n_phi=34, R=1.0, r=0.6).build()
    V, N = len(mesh.vertices), 6
    rng = np.random.default_rng(0)
    labels = rng.integers(0, N, size=V)

    dens = labels_to_one_hot(labels, N)
    ok = check("argmax(one_hot) == labels", bool(np.array_equal(np.argmax(dens, axis=1), labels)))
    ok &= check("rows sum to 1", bool(np.allclose(dens.sum(axis=1), 1.0)))

    p = str(tmp / "arm.h5")
    write_arm_solution(p, labels, mesh.vertices, mesh.faces, N)
    with h5py.File(p, "r") as f:
        x = f["x_opt"][:]
        n = int(f.attrs["n_partitions"])
        vs = f["vertices"][:]
    back = np.argmax(x.reshape(len(vs), n), axis=1)
    ok &= check("labels survive the file round trip", bool(np.array_equal(back, labels)))
    return ok


def test_hardening_is_neutral():
    """The apples-to-apples claim, on real data."""
    print("\n2/5  hardening a continuous field does not move the gates")
    hits = sorted(ROOT.glob("results/run_20260714_224821*/solution/surface_*.h5"))
    if not hits:
        print("  SKIP: no N=300 solution on disk")
        return None
    with h5py.File(hits[0], "r") as f:
        x, verts, faces = f["x_opt"][:], f["vertices"][:], f["faces"][:]
        N = int(f.attrs["n_partitions"])
    mesh = TriMesh(verts, faces)
    cont = x.reshape(len(verts), N)
    hard = labels_to_one_hot(np.argmax(cont, axis=1), N)

    g_cont = run_gates(cont, mesh, N)
    g_hard = run_gates(hard, mesh, N)
    ok = check(
        "area gate identical",
        g_cont.n_imbalanced == g_hard.n_imbalanced
        and abs(g_cont.worst_rel_dev - g_hard.worst_rel_dev) < 1e-12,
        f"{g_cont.n_imbalanced}/{g_cont.worst_rel_dev*100:.2f}% vs "
        f"{g_hard.n_imbalanced}/{g_hard.worst_rel_dev*100:.2f}%",
    )
    ok &= check(
        "connectivity gate identical",
        g_cont.n_fragmented == g_hard.n_fragmented
        and g_cont.fragmented == g_hard.fragmented,
        f"{g_cont.fragmented} vs {g_hard.fragmented}",
    )
    ok &= check(
        "dormant gate is vacuous once hardened (peak density 1.0)",
        abs(g_hard.min_peak_density - 1.0) < 1e-12,
        f"min peak {g_hard.min_peak_density:.4f}",
    )
    return ok


def test_granularity():
    print("\n3/5  vertex-granularity floor matches the shipped readout's record")
    hits = sorted(ROOT.glob(
        "results/run_20260806_123326*/readout/*/metadata.yaml"))
    if not hits:
        print("  SKIP: no readout metadata on disk")
        return None
    import yaml
    recorded = None
    for h in hits:
        meta = yaml.safe_load(open(h))
        recorded = _find_key(meta, "vertex_granularity_rel")
        if recorded is not None:
            break
    if recorded is None:
        print("  SKIP: vertex_granularity_rel not recorded")
        return None
    sol = sorted(ROOT.glob(
        "results/run_20260806_123326*/readout/*/solution_balanced.h5"))[0]
    with h5py.File(sol, "r") as f:
        verts, faces = f["vertices"][:], f["faces"][:]
        N = int(f.attrs["n_partitions"])
    got = vertex_granularity(TriMesh(verts, faces).v, N)
    return check(
        "granularity reproduces", abs(got - recorded) < 1e-9,
        f"recorded {recorded:.6f} vs computed {got:.6f}",
    )


def _find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = _find_key(v, key)
            if r is not None:
                return r
    return None


def test_censoring():
    print("\n4/5  censoring detection")
    mid = Phase2Report("d", 4, None, 1.0, 2, censored=False,
                       trajectory=[(1, 3.0), (2, 1.0), (3, 2.0), (4, 2.5)])
    last = Phase2Report("d", 4, None, 1.0, 4, censored=True,
                        trajectory=[(1, 3.0), (2, 2.0), (3, 1.5), (4, 1.0)])
    ok = check("best-in-middle is not censored", mid.to_dict()["censored"] is False)
    ok &= check("best-is-last is censored", last.to_dict()["censored"] is True)
    return ok


def test_area_lane():
    print("\n5/5  area gate flags solver failure rather than blaming the arm")
    mesh = TorusMeshProvider(n_theta=40, n_phi=34, R=1.0, r=0.6).build()
    V, N = len(mesh.vertices), 6
    # A deliberately unbalanced assignment: one cell takes a huge share.
    labels = np.zeros(V, dtype=int)
    labels[: V // 2] = 0
    labels[V // 2:] = np.arange(V - V // 2) % (N - 1) + 1
    g = run_gates(labels_to_one_hot(labels, N), mesh, N)
    d = g.to_dict()["area_imbalance"]
    return check(
        "suspect_solver_failure set on a grossly unbalanced assignment",
        d["suspect_solver_failure"] is True,
        f"worst {g.worst_rel_dev*100:.1f}% vs threshold "
        f"{d['harness_fault_threshold']*100:.3f}%",
    )


def main():
    import tempfile
    print("=" * 70)
    print("PHASE 0b ACCEPTANCE -- evaluation harness")
    print("=" * 70)
    with tempfile.TemporaryDirectory() as td:
        res = [
            test_roundtrip(Path(td)),
            test_hardening_is_neutral(),
            test_granularity(),
            test_censoring(),
            test_area_lane(),
        ]
    hard = [r for r in res if r is not None]
    print("\n" + "=" * 70)
    print("RESULT:", "PASS" if all(hard) else "FAIL")
    return 0 if all(hard) else 1


if __name__ == "__main__":
    sys.exit(main())
