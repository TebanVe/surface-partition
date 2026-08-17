#!/usr/bin/env python3
"""Phase 0d -- prove the evaluation harness reproduces a known answer.

Takes the validated N=100 deliverable's OWN winner-take-all labels, pushes them
through the whole harness as if they were an arm's output -- one-hot densities in
the Phase 1 schema, then Phase 2 under the control's own campaign
``refinement.yaml`` -- and requires the recorded perimeter back:

    185.2546144718457   (run_20260709_081548, campaign
                         ipopt_btol0.001_lbfgs30_hess_bestiter_partial, iter 020)

If the harness cannot reproduce a known answer, no arm number it later produces
means anything. This is the one test that qualifies the instrument itself.

Tolerance: bitwise on this machine and environment, otherwise <= 0.01%. Phase 2
runs IPOPT with topology migrations and its bit-determinism has never been
demonstrated in this repo, so demanding 16-digit equality would fail for reasons
that carry no information.

Writes everything under a scratch root so results/ is never touched.

Usage:
    python testing/test_phase0d_shakedown.py --scratch /path/to/scratch
"""
import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.partition.arm_harness import (  # noqa: E402
    run_phase2,
    scan_campaign,
    write_arm_solution,
)

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = 185.2546144718457
REL_TOL = 1e-4  # 0.01%


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    args = ap.parse_args()

    sols = sorted(ROOT.glob(
        "results/run_20260709_081548*/solution/surface_*.h5"))
    camps = sorted(ROOT.glob(
        "results/run_20260709_081548*/refinement/*/refinement.yaml"))
    if not sols or not camps:
        print("FAIL: control run or campaign config not on disk")
        return 1
    solution, campaign_cfg = str(sols[0]), str(camps[0])
    print(f"control solution : {solution}")
    print(f"pinned campaign  : {campaign_cfg}")

    with h5py.File(solution, "r") as f:
        x, verts, faces = f["x_opt"][:], f["vertices"][:], f["faces"][:]
        N = int(f.attrs["n_partitions"])
    labels = np.argmax(x.reshape(len(verts), N), axis=1)
    print(f"mesh             : V={len(verts)}, N={N}")
    print(f"labels           : {len(np.unique(labels))} distinct cells")

    run_root = Path(args.scratch) / "phase0d_run"
    (run_root / "solution").mkdir(parents=True, exist_ok=True)
    arm_sol = str(run_root / "solution" / "control_labels_onehot.h5")
    write_arm_solution(
        arm_sol, labels, verts, faces, N,
        extra_attrs={"provenance": "phase0d: control labels re-hardened"},
    )
    print(f"one-hot solution : {arm_sol}")
    print("\nrunning Phase 2 through the harness (this takes a while) ...\n")

    report, _ = run_phase2(arm_sol, campaign_cfg, str(ROOT))

    rel = abs(report.best_perimeter - REFERENCE) / REFERENCE
    print("=" * 70)
    print(f"reference   : {REFERENCE:.13f}")
    print(f"harness     : {report.best_perimeter:.13f}  (iter "
          f"{report.best_iteration} of {report.n_iterations})")
    print(f"relative diff: {rel*100:.6f}%   tolerance 0.01%")
    print(f"bitwise      : {report.best_perimeter == REFERENCE}")
    print(f"censored     : {report.censored}")
    print(f"wall         : {report.wall_seconds:.0f}s")
    ok = rel <= REL_TOL
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
