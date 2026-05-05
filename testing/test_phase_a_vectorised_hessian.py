#!/usr/bin/env python3
"""Phase A regression test — vectorised np.add.at Hessian vs. legacy Python loop.

Compares the output of ``compute_perimeter_hessian_sparse`` and
``compute_area_hessian_sparse`` on a compiled :class:`PartitionArrays` between

  * the new vectorised path (uses pre-computed ``seg_hess_off_*`` /
    ``btri{1,2}_hess_off_*`` offset arrays via ``np.add.at``), and

  * the legacy per-row Python loop (the ``else:`` fallback branches
    deliberately preserved in both files).

The fallback is forced by monkey-patching the offset fields on the
``PartitionArrays`` instance to ``None``.  Both paths share the same
arithmetic above the accumulation, so any difference would be a bug
in the new offset arrays — which is exactly what this test guards
against.

Specification: docs/PROMPT_PHASE_A_VECTORISED_HESSIAN.md ("Stronger
validation — write a CLI test").

Usage
-----
    python testing/test_phase_a_vectorised_hessian.py \\
        --solution <path.h5>

Optional:
    --atol <float>   Tolerance, default 1e-12 (matches plan §3.6).

Exit code 0 on PASS, 1 on FAIL.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.mesh.tri_mesh import TriMesh  # noqa: E402
from src.optimization.perimeter_optimizer import PerimeterOptimizer  # noqa: E402
from src.partition import vectorized_area, vectorized_perimeter  # noqa: E402
from src.partition.steiner_handler import SteinerHandler  # noqa: E402
from src.pipeline.io import (  # noqa: E402
    load_partition_from_base_file,
    load_partition_from_refined_file,
)
from src.pipeline.pipeline_orchestrator import detect_file_type  # noqa: E402


# Names of the Phase A offset fields on PartitionArrays.  When any of
# these is None, the Hessian builders take the legacy Python-loop path.
PHASE_A_FIELDS = (
    "seg_hess_off_aa",
    "seg_hess_off_bb",
    "seg_hess_off_ab",
    "btri1_hess_off_aa",
    "btri1_hess_off_bb",
    "btri1_hess_off_ab",
    "btri1_cell_active",
    "btri2_hess_off_aa",
    "btri2_hess_off_bb",
    "btri2_hess_off_ab",
    "btri2_cell_active",
)


def build_optimizer(solution_path: str) -> PerimeterOptimizer:
    """Load a base or refined HDF5 file and return a compiled optimizer."""
    file_type = detect_file_type(solution_path)
    if file_type == "base":
        mesh, partition = load_partition_from_base_file(
            solution_path, verbose=False)
    else:
        mesh, partition = load_partition_from_refined_file(
            solution_path, verbose=False)

    if not isinstance(mesh, TriMesh):
        # Loader returned something else — defensive check, not expected.
        raise TypeError(f"Expected TriMesh, got {type(mesh)!r}")

    total_area = float(mesh.v.sum())
    target_area = total_area / partition.n_cells

    steiner = SteinerHandler(mesh, partition)
    optimizer = PerimeterOptimizer(
        partition,
        mesh,
        target_area,
        steiner_handler=steiner,
        use_vectorized=True,
    )
    optimizer.compile()
    return optimizer


class _NullOffsets:
    """Context manager that nulls the Phase A offset fields on a
    PartitionArrays instance and restores them on exit, forcing the
    legacy fallback path inside the Hessian builders."""

    def __init__(self, pa) -> None:
        self._pa = pa
        self._saved: dict = {}

    def __enter__(self):
        for name in PHASE_A_FIELDS:
            self._saved[name] = getattr(self._pa, name)
            setattr(self._pa, name, None)
        return self._pa

    def __exit__(self, *exc):
        for name, value in self._saved.items():
            setattr(self._pa, name, value)
        return False


def _compare(label: str, h_new: np.ndarray, h_old: np.ndarray,
             atol: float) -> tuple[bool, float]:
    diff = float(np.max(np.abs(h_new - h_old)))
    ok = bool(np.allclose(h_new, h_old, atol=atol, rtol=0.0))
    status = "PASS" if ok else "FAIL"
    print(
        f"  [{status}] {label:32s} "
        f"||H_new||_∞={np.max(np.abs(h_new)):.3e}  "
        f"max|Δ|={diff:.3e}"
    )
    return ok, diff


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase A vectorised-Hessian regression test "
                    "(new np.add.at path vs. legacy Python-loop fallback).",
    )
    ap.add_argument("--solution", required=True,
                    help="Path to a Phase 1 base solution or Phase 2 "
                         "refined-contours HDF5 file.")
    ap.add_argument("--atol", type=float, default=1e-12,
                    help="Absolute tolerance for np.allclose (default 1e-12).")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for the random multiplier vector.")
    args = ap.parse_args()

    print(f"Phase A vectorised-Hessian regression test")
    print(f"  solution = {args.solution}")
    print(f"  atol     = {args.atol}")
    print()

    optimizer = build_optimizer(args.solution)
    pa = optimizer._arrays

    # Sanity: the new offsets must actually be populated.
    missing = [name for name in PHASE_A_FIELDS if getattr(pa, name) is None]
    if missing:
        print("FAIL: PartitionArrays is missing Phase A offset fields: "
              f"{missing}")
        return 1

    n = pa.n_active_vp
    n_c = pa.n_cells - 1
    print(f"  n_active_vp = {n}")
    print(f"  n_cells     = {pa.n_cells}  (constrained: {n_c})")
    print(f"  hess_nnz    = {len(pa.hess_row)}")
    print(f"  segments    = {len(pa.seg_vp1)}")
    print(f"  btri rows   = {len(pa.btri_cell)}")
    print()

    ok_total = True
    worst = 0.0

    print("[1/3] compute_perimeter_hessian_sparse")
    h_perim_new = vectorized_perimeter.compute_perimeter_hessian_sparse(pa)
    with _NullOffsets(pa):
        h_perim_old = vectorized_perimeter.compute_perimeter_hessian_sparse(pa)
    ok, d = _compare("perimeter Hessian", h_perim_new, h_perim_old, args.atol)
    ok_total &= ok
    worst = max(worst, d)

    multipliers_zero = np.zeros(n_c, dtype=np.float64)
    print("[2/3] compute_area_hessian_sparse  (multipliers = 0)")
    h_area0_new = vectorized_area.compute_area_hessian_sparse(
        pa, multipliers_zero)
    with _NullOffsets(pa):
        h_area0_old = vectorized_area.compute_area_hessian_sparse(
            pa, multipliers_zero)
    ok, d = _compare("area Hessian (mu=0)", h_area0_new, h_area0_old,
                     args.atol)
    ok_total &= ok
    worst = max(worst, d)

    rng = np.random.default_rng(args.seed)
    multipliers_rand = rng.standard_normal(n_c)
    print(f"[3/3] compute_area_hessian_sparse  (multipliers = N(0,1), "
          f"seed={args.seed})")
    h_area_new = vectorized_area.compute_area_hessian_sparse(
        pa, multipliers_rand)
    with _NullOffsets(pa):
        h_area_old = vectorized_area.compute_area_hessian_sparse(
            pa, multipliers_rand)
    ok, d = _compare("area Hessian (random mu)", h_area_new, h_area_old,
                     args.atol)
    ok_total &= ok
    worst = max(worst, d)

    print()
    print(f"  worst max|Δ| over all cases = {worst:.3e}")
    if ok_total and worst < args.atol:
        print("RESULT: PASS")
        return 0
    print("RESULT: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
