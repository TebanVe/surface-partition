#!/usr/bin/env python3
"""Verify the sparse area Jacobian equals the dense reference.

Checks that ``compute_area_jacobian_sparse + compute_steiner_area_jacobian_sparse``
returns exactly the same values, at the ``(jac_row, jac_col)`` sparsity
positions, as the dense path ``compute_area_jacobian + compute_steiner_area_jacobian``.

Both paths use identical Steiner finite-difference code, so the only
difference is the scatter pattern — agreement should be at machine precision.

Usage:
    python testing/test_sparse_jacobian_equivalence.py --solution <path.h5>
    python testing/test_sparse_jacobian_equivalence.py --solution <path.h5> --atol 1e-12

Exit code: 0 on PASS, 1 on FAIL.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from _hessian_test_utils import build_optimizer  # noqa: E402
from src.partition import vectorized_area, vectorized_steiner  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--solution', required=True,
                    help='Base solution or refined-contours HDF5 file.')
    ap.add_argument('--atol', type=float, default=1e-10,
                    help='Absolute tolerance for the comparison (default 1e-10).')
    args = ap.parse_args()

    opt = build_optimizer(args.solution)
    pa = opt._arrays

    # --- Dense reference: analytical area Jacobian + FD Steiner Jacobian ---
    jac_dense_area = vectorized_area.compute_area_jacobian(pa)
    jac_dense_steiner = vectorized_steiner.compute_steiner_area_jacobian(pa)
    jac_dense = jac_dense_area + jac_dense_steiner          # (n_cells-1, n_active_vp)

    # --- Sparse path: same quantities, scattered into (nnz,) ---
    jac_sparse_area = vectorized_area.compute_area_jacobian_sparse(pa)
    jac_sparse_steiner = vectorized_steiner.compute_steiner_area_jacobian_sparse(pa)
    jac_sparse = jac_sparse_area + jac_sparse_steiner       # (nnz,)

    # Extract the dense non-zeros in jac_row/jac_col order.
    dense_vals = jac_dense[pa.jac_row, pa.jac_col]

    abs_err = float(np.max(np.abs(dense_vals - jac_sparse))) if len(jac_sparse) else 0.0
    ref = max(float(np.max(np.abs(dense_vals))) if len(dense_vals) else 0.0, 1e-30)
    rel_err = abs_err / ref

    print("Sparse vs. dense area-Jacobian equivalence")
    print(f"  solution         = {args.solution}")
    print(f"  n_active_vp      = {pa.n_active_vp}")
    print(f"  n_cells          = {pa.n_cells}  (constrained: {pa.n_cells - 1})")
    print(f"  n_triple_points  = {pa.n_triple_points}")
    print(f"  nnz              = {len(pa.jac_row)}")
    print(f"  dense shape      = {jac_dense.shape}")
    print(f"  max |Δ|          = {abs_err:.3e}")
    print(f"  max rel err      = {rel_err:.3e}")

    ok = abs_err < args.atol
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
