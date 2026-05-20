---
name: Exact-Hessian Validation & Sparse-Accumulation Performance
overview: >
  Follow-up plan to docs/SPARSE_JACOBIAN_AND_EXACT_HESSIAN_PLAN.md.
  The functional implementation of the sparse Jacobian (Phase 2) and the
  exact Lagrangian Hessian (Phase 4) is already in the codebase.  What
  remains is (i) a validation harness that proves the analytical
  derivatives agree with finite differences and with the dense reference
  path, (ii) a small refactor that replaces the three Python-level
  accumulation loops in the Hessian builders with vectorised
  ``np.add.at`` calls using pre-computed offset arrays, and (iii)
  documentation housekeeping on the parent plan.
todos:
  - id: precomp-seg-offsets
    content: Pre-compute seg_hess_off_{aa,bb,ab} in compile_arrays() and add to PartitionArrays
    status: completed
  - id: precomp-btri-offsets
    content: Pre-compute btri1/btri2 hess offset arrays in compile_arrays() and add to PartitionArrays
    status: completed
  - id: vectorize-perim-hess
    content: Replace Python for-loop in compute_perimeter_hessian_sparse() with np.add.at
    status: completed
  - id: vectorize-area-hess
    content: Replace Python for-loops in compute_area_hessian_sparse() with np.add.at (both 1-inside and 2-inside)
    status: completed
  - id: test-sparse-jac
    content: Write testing/test_sparse_jacobian_equivalence.py — sparse Jacobian == dense Jacobian at (jac_row, jac_col)
    status: pending
  - id: test-hess-vs-fd
    content: Write testing/test_exact_hessian_vs_fd.py — analytical Lagrangian Hessian agrees with central-FD to ~1e-4
    status: pending
  - id: test-hess-matvec
    content: Optional large-scale matvec variant of the Hessian FD test (avoids O(n^2) memory)
    status: pending
  - id: compare-lbfgs-vs-exact
    content: Write testing/compare_hessian_modes.py — run IPOPT twice on same problem, record perimeter/iters/time
    status: pending
  - id: compare-hessian-profiling
    content: Extend compare_hessian_modes.py with per-component profiling (perimeter-H / area-H / Steiner-H / IPOPT linear solve)
    status: pending
  - id: doc-housekeeping
    content: Update docs/SPARSE_JACOBIAN_AND_EXACT_HESSIAN_PLAN.md — flip completed todos, fix src/core paths
    status: pending
isProject: false
---

# Exact-Hessian Validation & Sparse-Accumulation Performance Plan

## Table of Contents

1. [Context for the implementing agent](#1-context)
2. [Current state of the code](#2-current-state)
3. [Phase A — Vectorised Hessian accumulation](#3-phase-a)
4. [Phase B — Validation harness](#4-phase-b)
5. [Phase C — Documentation housekeeping](#5-phase-c)
6. [Acceptance criteria](#6-acceptance)
7. [Reference: key file paths](#7-paths)

---

## 1. Context for the Implementing Agent <a id="1-context"></a>

This plan is a direct continuation of
`docs/SPARSE_JACOBIAN_AND_EXACT_HESSIAN_PLAN.md` (referred to below as "the
parent plan"). Read §1, §2, and §4 of the parent plan before starting; the
mathematical derivations there are not repeated in this document.

The parent plan's sparse-Jacobian and exact-Hessian code is already merged
and is exercised by `--method ipopt --exact-hessian` on
`scripts/refine_perimeter.py`. Two things were deferred:

1. The "Performance note" in §4.2 of the parent plan — pre-computed
   offset arrays so the three inner accumulation loops become a handful
   of `np.add.at` calls. Those loops are the only Python-level work left
   in the Hessian path, and they dominate at large segment counts.
2. §6 of the parent plan — a validation harness that checks the
   analytical derivatives against finite differences and against the
   dense reference path. None of it was written.

This plan delivers both, plus a short doc cleanup on the parent plan.

**Prerequisites** the agent is expected to know:

- The "λ convention" from `CLAUDE.md` / parent-plan §1.5.
- How `PartitionArrays` is built by
  `PartitionContour.compile_arrays(steiner_handler)` (defined in
  `src/partition/contour_partition.py` around line 814).
- How the IPOPT adapter in `src/optimization/perimeter_optimizer.py`
  uses the sparse Hessian callback.

**Scope guard.** Do **not** change any derivative formulas, the sparsity
pattern, the L-BFGS / exact-Hessian toggle, or any IPOPT option. This plan
is purely (a) internal refactor for speed and (b) validation. If a
validation test in Phase B fails, stop and report rather than editing the
derivative code — a failure indicates a real bug that this plan does not
authorise fixing.

### 1.1 Scale target and applicability

The codebase's long-term ambition is partitions of **thousands of
cells**. This plan sits at **Tier 2** in `docs/SCALABILITY_ANALYSIS.md`
and is one of several necessary steps toward that ambition; it is not
by itself sufficient.

| Problem size | What Phase A delivers | Still blocking? |
|---|---|---|
| N ≲ 20 | per-iter Hessian build drops from a few tens of ms to sub-ms | nothing — already fast |
| N ≈ 20–100 | per-iter Hessian build becomes negligible compared to IPOPT internals | FD Steiner Hessian (grows O(N)), iteration count |
| N ≈ 100–500 | Python-side Hessian cost is firmly out of the profile | **FD Steiner** (addressed by `docs/ANALYTICAL_STEINER_DERIVATIVES_PLAN.md`), dense (N−1)×(N−1) Schur complement block in IPOPT's KKT system |
| N ≈ 500–1000+ | Phase A by itself does not reach this regime | **Tier 3** from `SCALABILITY_ANALYSIS.md` §3 — Lloyd CVT initialiser, multigrid, augmented-Lagrangian decomposition, or curve-shortening flow |

What Phase A does guarantee at every scale:

- Eliminates the Python for-loop in the Hessian accumulation path (the
  one remaining Python-level cost in `h()` calls on current code).
- Makes the per-component profile from §4.4 readable: once this plan
  lands, the "perimeter_hess" / "area_hess" buckets reflect the
  arithmetic cost of computing the Hessian entries, not the cost of
  Python dict lookups.

What Phase B / C add:

- A validation harness so that future Hessian changes (including the
  analytical Steiner work) are regression-tested against FD.
- The per-component profiler in §4.4 is the primary evidence for
  deciding the next step: if Steiner dominates, proceed to
  `docs/ANALYTICAL_STEINER_DERIVATIVES_PLAN.md`; if the IPOPT linear
  solver dominates, the next dollar is in Tier 3 or in a linear-solver
  swap (MA57/MA97 over MUMPS).

Do not oversell this plan as a scalability unlock. It is the
single-issue refactor that removes the most obvious Python-level
bottleneck and builds the measurement infrastructure to direct all
subsequent scaling work.

---

## 2. Current State of the Code <a id="2-current-state"></a>

Everything in this section is already implemented and correct. You do not
need to modify it.

### 2.1 `PartitionArrays` (`src/partition/partition_arrays.py`)

Already contains, at the bottom of the dataclass:

```python
nnz_lookup:      Optional[np.ndarray] = None   # (n_cells-1, n_active_vp) int32
hess_row:        Optional[np.ndarray] = None   # (hess_nnz,) int32, row >= col
hess_col:        Optional[np.ndarray] = None   # (hess_nnz,) int32
hess_offset_map: Optional[Dict[Tuple[int, int], int]] = field(default=None, repr=False)
```

### 2.2 `compile_arrays()` (`src/partition/contour_partition.py`)

Between roughly line 1037 and line 1079 it builds `nnz_lookup`, the
Hessian sparsity pattern `(hess_row, hess_col)`, and the offset dict
`hess_offset_map`. All quantities needed by the Hessian-building
functions (`seg_vp1`, `seg_vp2`, `btri_vp1`, `btri_vp2`,
`btri_n_inside`, `btri_cell`, `tp_vp_indices`) are in local scope at
that point.

### 2.3 Hessian-building functions (currently using Python loops)

- `compute_perimeter_hessian_sparse()` at
  `src/partition/vectorized_perimeter.py:68-117`. Computes
  `H_aa`, `H_bb`, `H_ab` vectorised over segments (arrays of shape
  `(n_segments,)`), but then accumulates them into `values` via a Python
  `for s in range(len(vp1))` loop with three dict lookups per segment
  (lines 105-115).
- `compute_area_hessian_sparse()` at
  `src/partition/vectorized_area.py:370-545`. Same story: arithmetic is
  vectorised, but lines 440-451 (1-inside) and 532-543 (2-inside) are
  per-row Python loops that call `hess_offset_map[...]` three times each
  and also multiply by `multipliers[c]` one row at a time.
- `compute_steiner_perimeter_hessian_fd` and
  `compute_steiner_area_hessian_fd` in `vectorized_steiner.py` iterate
  over `pa.tp_affected_vps` (typically << 100 entries). These are not
  performance-critical and are **out of scope** for Phase A — do not
  touch them.

### 2.4 Adapter (`src/optimization/perimeter_optimizer.py`)

`IPOPTProblemAdapter` attaches `hessian` / `hessianstructure` only when
`exact_hessian=True` (lines 66-68). The `hessian()` method sets
`pa.vp_lambda[:] = x` and calls the four Hessian builders (two analytical,
two finite-difference for Steiner). Do not change this file in Phase A.
In Phase B you only *read* from it.

---

## 3. Phase A — Vectorised Hessian Accumulation <a id="3-phase-a"></a>

Goal: remove the per-segment and per-boundary-triangle Python loops from
the two analytical Hessian builders. Result must be bit-for-bit (or at
least float-tolerance, ~1e-12) identical to the current implementation on
every mesh.

### 3.1 Add three perimeter-segment offset arrays to `PartitionArrays`

Edit `src/partition/partition_arrays.py`. Append these fields **after**
`hess_offset_map` (dataclass default-after-non-default rule):

```python
# --- Phase A: pre-computed Hessian offsets for fast accumulation ---
# Offsets into the flat (hess_nnz,) values array, one offset per
# perimeter segment.  Populated by compile_arrays().
seg_hess_off_aa: Optional[np.ndarray] = None   # int32 (n_segments,)
seg_hess_off_bb: Optional[np.ndarray] = None   # int32 (n_segments,)
seg_hess_off_ab: Optional[np.ndarray] = None   # int32 (n_segments,) — lower-triangle (max,min)

# Offsets for boundary triangles, split by n_inside.  Order matches
# how compute_area_hessian_sparse iterates (mask m1 then mask m2).
btri1_hess_off_aa: Optional[np.ndarray] = None   # int32 (n_btri_1,)
btri1_hess_off_bb: Optional[np.ndarray] = None   # int32 (n_btri_1,)
btri1_hess_off_ab: Optional[np.ndarray] = None   # int32 (n_btri_1,)
btri1_cell_active: Optional[np.ndarray] = None   # bool  (n_btri_1,) — cell < n_cells-1

btri2_hess_off_aa: Optional[np.ndarray] = None   # int32 (n_btri_2,)
btri2_hess_off_bb: Optional[np.ndarray] = None   # int32 (n_btri_2,)
btri2_hess_off_ab: Optional[np.ndarray] = None   # int32 (n_btri_2,)
btri2_cell_active: Optional[np.ndarray] = None   # bool  (n_btri_2,)
```

Rationale for the `_ab` array: for each segment/triangle we need the
lower-triangle offset `(max(a,b), min(a,b))`. We pre-compute it once so
runtime code never has to call `max`/`min` again.

Rationale for `_cell_active`: the area Hessian only contributes for cells
with index `< n_cells - 1` (the last cell is determined by conservation
and is not a constraint). The current code does this with `if c >= n_c:
continue` inside the Python loop. The vectorised version needs the same
filter as a boolean mask.

### 3.2 Populate them in `compile_arrays()`

In `src/partition/contour_partition.py`, **after** the block that builds
`hess_offset_map` (around line 1077) and **before** the
`return PartitionArrays(...)` call (around line 1081), insert:

```python
# 10. Pre-computed Hessian offsets for fast accumulation (Phase A of
#     docs/EXACT_HESSIAN_VALIDATION_AND_PERF_PLAN.md)

def _hess_off(i: int, j: int) -> int:
    """Lower-triangle offset for the Hessian entry at row=max(i,j), col=min(i,j)."""
    return hess_offset_map[(max(i, j), min(i, j))]

# Perimeter segments: each segment s has endpoints (seg_vp1[s], seg_vp2[s])
if len(seg_vp1) > 0:
    seg_hess_off_aa = np.fromiter(
        (hess_offset_map[(int(a), int(a))] for a in seg_vp1),
        dtype=np.int32, count=len(seg_vp1))
    seg_hess_off_bb = np.fromiter(
        (hess_offset_map[(int(b), int(b))] for b in seg_vp2),
        dtype=np.int32, count=len(seg_vp2))
    seg_hess_off_ab = np.fromiter(
        (_hess_off(int(a), int(b)) for a, b in zip(seg_vp1, seg_vp2)),
        dtype=np.int32, count=len(seg_vp1))
else:
    seg_hess_off_aa = np.empty(0, dtype=np.int32)
    seg_hess_off_bb = np.empty(0, dtype=np.int32)
    seg_hess_off_ab = np.empty(0, dtype=np.int32)

# Boundary triangles, split by n_inside — preserve the row order that
# compute_area_hessian_sparse will see (pa.btri_n_inside == 1 first, then == 2)
m1_idx = np.flatnonzero(btri_n_inside == 1)
m2_idx = np.flatnonzero(btri_n_inside == 2)

def _btri_offsets(mask_indices):
    if len(mask_indices) == 0:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty, empty, np.empty(0, dtype=bool)
    a = btri_vp1[mask_indices]
    b = btri_vp2[mask_indices]
    off_aa = np.fromiter((hess_offset_map[(int(x), int(x))] for x in a),
                         dtype=np.int32, count=len(a))
    off_bb = np.fromiter((hess_offset_map[(int(x), int(x))] for x in b),
                         dtype=np.int32, count=len(b))
    off_ab = np.fromiter((_hess_off(int(x), int(y)) for x, y in zip(a, b)),
                         dtype=np.int32, count=len(a))
    cell_active = btri_cell[mask_indices] < (self.n_cells - 1)
    return off_aa, off_bb, off_ab, cell_active

(btri1_hess_off_aa, btri1_hess_off_bb,
 btri1_hess_off_ab, btri1_cell_active) = _btri_offsets(m1_idx)
(btri2_hess_off_aa, btri2_hess_off_bb,
 btri2_hess_off_ab, btri2_cell_active) = _btri_offsets(m2_idx)

self.logger.info(
    f"  Hessian offset arrays: {len(seg_hess_off_aa)} seg, "
    f"{len(btri1_hess_off_aa)} btri-1, {len(btri2_hess_off_aa)} btri-2")
```

Then add the nine new fields to the `return PartitionArrays(...)` call at
the end of the function — one line each, keyword form.

**Sanity check during development:** every returned offset must satisfy
`0 <= offset < len(hess_row)`. Add a single assertion
`assert (seg_hess_off_aa >= 0).all() and (seg_hess_off_aa < len(hess_row)).all()`
during development, then remove it before committing.

### 3.3 Vectorise `compute_perimeter_hessian_sparse()`

Edit `src/partition/vectorized_perimeter.py`. Replace the Python for-loop
at lines 102-115 with:

```python
    # Phase A: vectorised accumulation using pre-computed offset arrays.
    # Fall back to the per-segment loop only if the offsets are missing
    # (backwards compatibility with an old PartitionArrays snapshot).
    if pa.seg_hess_off_aa is not None:
        np.add.at(values, pa.seg_hess_off_aa, H_aa)
        np.add.at(values, pa.seg_hess_off_bb, H_bb)
        np.add.at(values, pa.seg_hess_off_ab, H_ab)
    else:
        for s in range(len(pa.seg_vp1)):
            a, b = int(pa.seg_vp1[s]), int(pa.seg_vp2[s])
            values[pa.hess_offset_map[(a, a)]] += H_aa[s]
            values[pa.hess_offset_map[(b, b)]] += H_bb[s]
            hi, lo = max(a, b), min(a, b)
            values[pa.hess_offset_map[(hi, lo)]] += H_ab[s]
```

Keep every line above the accumulation unchanged (the arithmetic is
already vectorised). The `np.add.at` calls handle the case where two
segments share the same Hessian entry correctly — do **not** replace
them with `values[...] += ...` because that would lose duplicate
contributions.

### 3.4 Vectorise `compute_area_hessian_sparse()`

Edit `src/partition/vectorized_area.py`. In both the `m1` block and the
`m2` block, replace the per-row Python loop (currently lines 440-451 and
532-543) with the same pattern:

```python
        # Phase A: vectorised accumulation
        if pa.btri1_hess_off_aa is not None:      # or btri2_... in the m2 block
            active = pa.btri1_cell_active          # (n_btri_1,) bool
            if np.any(active):
                cells_active = pa.btri_cell[m1][active]   # or pa.btri_cell[m2][active]
                mu = multipliers[cells_active]            # (n_active_rows,)
                np.add.at(values, pa.btri1_hess_off_aa[active], mu * H_11[active])
                np.add.at(values, pa.btri1_hess_off_bb[active], mu * H_22[active])
                np.add.at(values, pa.btri1_hess_off_ab[active], mu * H_12[active])
        else:
            # Fallback: original per-row loop (kept for backwards compat)
            cells = pa.btri_cell[m1]
            vp1_arr = pa.btri_vp1[m1]
            vp2_arr = pa.btri_vp2[m1]
            for k in range(len(cells)):
                c = int(cells[k])
                if c >= n_c:
                    continue
                mu = multipliers[c]
                a = int(vp1_arr[k]); b = int(vp2_arr[k])
                values[pa.hess_offset_map[(a, a)]] += mu * H_11[k]
                values[pa.hess_offset_map[(b, b)]] += mu * H_22[k]
                hi, lo = max(a, b), min(a, b)
                values[pa.hess_offset_map[(hi, lo)]] += mu * H_12[k]
```

Apply the same transformation in the `m2` block, substituting
`btri2_*` for `btri1_*` and `m2` for `m1`.

**Important subtlety: row ordering.** The pre-computed `btri1_*` arrays
were built from `np.flatnonzero(btri_n_inside == 1)`, which is the same
ordering you get from `pa.btri_cell[m1]` at runtime (boolean mask over the
same underlying array). Do not sort or permute either side. If you
suspect an ordering mismatch, verify it locally with
`np.testing.assert_array_equal(pa.btri_cell[pa.btri_n_inside == 1],
pa.btri_cell[np.flatnonzero(pa.btri_n_inside == 1)])` before touching the
code.

### 3.5 Do NOT touch the Steiner FD Hessians

`compute_steiner_perimeter_hessian_fd` and
`compute_steiner_area_hessian_fd` in `vectorized_steiner.py` iterate over
`pa.tp_affected_vps`, which is at most `3 × n_triple_points` — typically
well under 100. The work inside their loops is dominated by the FD
gradient/Jacobian recomputation, not the offset lookup. Vectorising them
would add complexity for a negligible speedup. Leave them as-is.

### 3.6 Phase A acceptance

Before moving to Phase B:

1. On any existing Phase-2 checkpoint, run the IPOPT exact-Hessian path
   once with and once without your changes, and verify the Hessian
   values returned by the adapter are numerically identical
   (`np.allclose(H_new, H_old, atol=1e-12)`). The cleanest way to do this
   is to stash the old file, run a single IPOPT iteration, stash the
   returned Hessian values (e.g. via a breakpoint or a temporary
   `np.save` in `_hessian_impl`), restore the new file, repeat.
2. Confirm `scripts/refine_perimeter.py --method ipopt --exact-hessian`
   completes end-to-end with no errors on
   `parameters/torus_10part.yaml` (or whichever small problem you have a
   solution for).

---

## 4. Phase B — Validation Harness <a id="4-phase-b"></a>

Add three CLI scripts under `testing/`. Do **not** add pytest
fixtures — this project has no pytest suite (see `CLAUDE.md`, "Gotchas
and Known Issues"). Each script should be runnable as
`python testing/<name>.py --solution <path.h5> [--config <yaml>]`
and should print a clear PASS / FAIL line plus numeric diagnostics.

All three tests need a compiled `PartitionArrays`. The shared helper
below should live at the top of each script (or be factored into a
single `testing/_hessian_test_utils.py` if you prefer):

```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.pipeline.io import (
    load_partition_from_base_file, load_partition_from_refined_file,
    detect_run_layout,
)
from src.pipeline.pipeline_orchestrator import detect_file_type
from src.partition.steiner_handler import SteinerHandler
from src.optimization.perimeter_optimizer import PerimeterOptimizer


def build_optimizer(solution_path: str):
    """Load a solution/checkpoint file and return a compiled PerimeterOptimizer."""
    file_type = detect_file_type(solution_path)
    if file_type == 'base':
        mesh, partition = load_partition_from_base_file(solution_path, verbose=False)
    else:
        mesh, partition = load_partition_from_refined_file(solution_path, verbose=False)

    total_area = mesh.v.sum()
    target_area = float(total_area) / partition.n_cells

    steiner = SteinerHandler(mesh, partition)
    optimizer = PerimeterOptimizer(
        partition, mesh, target_area, steiner_handler=steiner,
        use_vectorized=True,
    )
    optimizer.compile()       # builds optimizer._arrays
    return optimizer
```

### 4.1 `testing/test_sparse_jacobian_equivalence.py`

**Goal.** Prove `compute_area_jacobian_sparse + compute_steiner_area_jacobian_sparse`
returns exactly the same values at positions `(jac_row, jac_col)` as the
dense reference path.

**Implementation:**

```python
#!/usr/bin/env python3
"""Verify that the sparse area Jacobian equals the dense reference."""
import argparse
import numpy as np

from _hessian_test_utils import build_optimizer   # or inline the helper
from src.partition import vectorized_area, vectorized_steiner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--solution', required=True)
    ap.add_argument('--atol', type=float, default=1e-10)
    args = ap.parse_args()

    opt = build_optimizer(args.solution)
    pa = opt._arrays

    # Dense reference (analytical area + FD Steiner)
    jac_dense_area   = vectorized_area.compute_area_jacobian(pa)
    jac_dense_stein  = vectorized_steiner.compute_steiner_area_jacobian(pa)
    jac_dense = jac_dense_area + jac_dense_stein   # (n_c-1, n_active)

    # Sparse path
    jac_sparse_area  = vectorized_area.compute_area_jacobian_sparse(pa)
    jac_sparse_stein = vectorized_steiner.compute_steiner_area_jacobian_sparse(pa)
    jac_sparse = jac_sparse_area + jac_sparse_stein    # (nnz,)

    # Extract non-zeros from dense in jac_row/jac_col order
    dense_vals = jac_dense[pa.jac_row, pa.jac_col]

    abs_err = np.max(np.abs(dense_vals - jac_sparse))
    rel_err = abs_err / max(np.max(np.abs(dense_vals)), 1e-30)

    print(f"nnz              = {len(pa.jac_row)}")
    print(f"dense shape      = {jac_dense.shape}")
    print(f"max |Δ|          = {abs_err:.3e}")
    print(f"max rel err      = {rel_err:.3e}")

    ok = abs_err < args.atol
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
```

**Pass criterion:** `abs_err < 1e-10`. The Steiner contributions on both
sides use the same finite-difference code, so the only difference between
the paths is the scatter pattern, which should be exact.

### 4.2 `testing/test_exact_hessian_vs_fd.py`

**Goal.** Prove the full Lagrangian Hessian returned by
`IPOPTProblemAdapter._hessian_impl` agrees with a finite-difference
reference computed by central differences on the Lagrangian gradient.

**Why this matters.** This is the single test that could unmask a wrong
sign or a wrong chain-rule factor in any of the analytical Hessian
derivations (§2.3, §2.4 of the parent plan). If it passes, we know the
analytical and FD Steiner pieces combine into something consistent with
the gradient, which is itself analytical and already tested by IPOPT's
derivative checker.

**Implementation.** Two modes: a full `(n, n)` build for small problems,
and a Hessian-vector-product ("matvec") mode for large ones.

```python
#!/usr/bin/env python3
"""Verify the analytical Lagrangian Hessian against central FD."""
import argparse
import numpy as np

from _hessian_test_utils import build_optimizer
from src.optimization.perimeter_optimizer import IPOPTProblemAdapter


def lagrangian_grad(optimizer, x, lagrange, obj_factor):
    g  = obj_factor * optimizer.objective_gradient(x)
    J  = optimizer.constraint_area_jacobian(x)     # (n_c-1, n_active)
    return g + lagrange @ J


def hessian_from_adapter(optimizer, x, lagrange, obj_factor):
    """Call the adapter's Hessian and pack into a dense symmetric matrix."""
    adapter = IPOPTProblemAdapter(optimizer, exact_hessian=True)
    vals = adapter._hessian_impl(x, lagrange, obj_factor)   # (hess_nnz,)
    n    = len(x)
    H    = np.zeros((n, n))
    pa   = optimizer._arrays
    H[pa.hess_row, pa.hess_col] = vals
    # Mirror the lower triangle to upper
    strict_lower = pa.hess_row > pa.hess_col
    H[pa.hess_col[strict_lower], pa.hess_row[strict_lower]] = vals[strict_lower]
    return H


def hessian_fd(optimizer, x, lagrange, obj_factor, eps=1e-5):
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        H[i, :] = (lagrangian_grad(optimizer, xp, lagrange, obj_factor)
                   - lagrangian_grad(optimizer, xm, lagrange, obj_factor)) / (2 * eps)
    return 0.5 * (H + H.T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--solution', required=True)
    ap.add_argument('--rtol', type=float, default=1e-3,
                    help='Relative tolerance (loose — Steiner FD contributes noise).')
    ap.add_argument('--atol', type=float, default=1e-4,
                    help='Absolute floor for the comparison.')
    ap.add_argument('--obj-factor', type=float, default=1.0)
    ap.add_argument('--lagrange-mode', choices=['zero', 'ones', 'random'],
                    default='random')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-n', type=int, default=400,
                    help='Skip test if n_active_vp > this (use matvec mode instead).')
    args = ap.parse_args()

    opt = build_optimizer(args.solution)
    pa  = opt._arrays
    n   = pa.n_active_vp
    m   = pa.n_cells - 1

    if n > args.max_n:
        print(f"SKIP: n_active_vp={n} > --max-n={args.max_n}. "
              f"Run testing/test_exact_hessian_matvec.py instead.")
        return 0

    rng = np.random.default_rng(args.seed)
    x   = pa.vp_lambda.copy()
    if args.lagrange_mode == 'zero':
        lam = np.zeros(m)
    elif args.lagrange_mode == 'ones':
        lam = np.ones(m)
    else:
        lam = rng.standard_normal(m)

    H_ana = hessian_from_adapter(opt, x, lam, args.obj_factor)
    H_fd  = hessian_fd(opt, x, lam, args.obj_factor)

    abs_err = np.max(np.abs(H_ana - H_fd))
    ref     = max(np.max(np.abs(H_fd)), 1e-30)
    rel_err = abs_err / ref

    # Diagnostic: per-source contribution
    print(f"n_active_vp  = {n}")
    print(f"hess_nnz     = {len(pa.hess_row)}")
    print(f"||H_ana||_∞  = {np.max(np.abs(H_ana)):.3e}")
    print(f"||H_fd||_∞   = {np.max(np.abs(H_fd)):.3e}")
    print(f"max |Δ|      = {abs_err:.3e}")
    print(f"rel err      = {rel_err:.3e}")

    # Also report where the biggest disagreement lives
    i, j = np.unravel_index(np.argmax(np.abs(H_ana - H_fd)), H_ana.shape)
    print(f"worst entry  = ({i},{j}) ana={H_ana[i,j]:.3e} fd={H_fd[i,j]:.3e}")

    ok = abs_err < max(args.atol, args.rtol * ref)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
```

**Tolerance guidance.** Because Steiner perimeter + Steiner area Hessians
use finite differences internally (central-FD on the gradient for
perimeter, forward-FD on the Jacobian for area), expect ~1e-4 absolute
agreement, not machine precision. That is what the parent plan predicted
in §6.2. Suggested tolerances:

- `--atol 1e-4 --rtol 1e-3` for meshes with triple points.
- `--atol 1e-6 --rtol 1e-5` when `--lagrange-mode zero` and the problem
  has no triple points (pure analytical Hessian, no FD noise).

**Diagnostic suggestion.** If the test fails, re-run with
`--lagrange-mode zero` to isolate the objective Hessian (perimeter only),
then with `--obj-factor 0 --lagrange-mode ones` to isolate the
constraint Hessian (area only). Whichever branch fails identifies the
module with the bug (perimeter, area, or Steiner FD). Again: if this
test fails, **stop** and report — this plan does not authorise editing
the derivative code.

### 4.3 `testing/test_exact_hessian_matvec.py` (optional, for large meshes)

For problems where `n_active_vp > ~400` the dense `(n, n)` FD build is
slow and memory-hungry. Provide a Hessian-vector-product check instead:
for a random unit vector `v`, verify

    H_ana @ v  ≈  (L_grad(x + ε v) − L_grad(x − ε v)) / (2 ε)

Loop over ~5 random `v`'s with `seed` CLI option. Accept if every
matvec agrees to within the same tolerance as §4.2.

```python
def hessian_matvec(optimizer, x, lagrange, obj_factor, v, eps=1e-5):
    xp = x + eps * v
    xm = x - eps * v
    return (lagrangian_grad(optimizer, xp, lagrange, obj_factor)
            - lagrangian_grad(optimizer, xm, lagrange, obj_factor)) / (2 * eps)


def hessian_matvec_from_adapter(optimizer, x, lagrange, obj_factor, v):
    H = hessian_from_adapter(optimizer, x, lagrange, obj_factor)   # dense
    return H @ v
```

This is still O(n²) in memory because `hessian_from_adapter` builds a
dense `H`. For truly large problems (`n > 10000`) build the matvec
directly from the sparse values instead:

```python
def hessian_matvec_sparse(pa, vals, v):
    out = np.zeros_like(v)
    # Lower triangle contribution
    np.add.at(out, pa.hess_row, vals * v[pa.hess_col])
    # Upper triangle contribution (from symmetry, excluding the diagonal)
    strict = pa.hess_row > pa.hess_col
    np.add.at(out, pa.hess_col[strict], vals[strict] * v[pa.hess_row[strict]])
    return out
```

This is a nice utility anyway — consider adding it to
`src/partition/vectorized_perimeter.py` or similar so other code can
reuse it. But keep the scope of this plan to just the test.

### 4.4 `testing/compare_hessian_modes.py`

**Goal.** Quantitatively decide whether `--exact-hessian` is worth using
by default, and — just as importantly — answer **where the per-iteration
time actually goes** when the exact-Hessian path is slower. The script
must produce two kinds of output on the same run:

1. **End-to-end totals** per mode: final perimeter, constraint violation,
   iteration count, wall-clock time, success flag.
2. **Per-component breakdown of one Hessian evaluation** on the
   exact-Hessian path: time spent in `compute_perimeter_hessian_sparse`,
   `compute_area_hessian_sparse`, the two Steiner FD helpers, and (via
   IPOPT's built-in timing statistics) the linear solver.

The second output is what tells you whether to invest in
Phase A (vectorised accumulation), whether analytical Steiner
(`docs/ANALYTICAL_STEINER_DERIVATIVES_PLAN.md`) would move the needle,
or whether you are pegged on IPOPT's linear solver (a swap-the-
solver problem, not a Python problem).

**Implementation sketch.** The easiest route is to *not* shell out; call
`PerimeterOptimizer.optimize()` directly from one Python process so both
runs see the same initial state. The per-component profiler is a small
monkey-patch that intercepts the four Hessian-building functions and
accumulates their wall-clock into a counter dict.

```python
#!/usr/bin/env python3
"""Compare IPOPT L-BFGS vs. exact Hessian on the same problem.

Also emits a per-component profile of one Hessian evaluation on the
exact-Hessian path, so the user can see at a glance which piece is the
bottleneck (Python accumulation, Steiner FD, or IPOPT's linear solver).
"""
import argparse
import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np

from _hessian_test_utils import build_optimizer

# The four functions that dominate the exact-Hessian callback.  All four
# live in src/partition/.  We wrap them with a timing decorator below.
from src.partition import vectorized_perimeter as _vperim
from src.partition import vectorized_area      as _varea
from src.partition import vectorized_steiner   as _vstein


# ---------------------------------------------------------------------
# Per-component profiler
# ---------------------------------------------------------------------

_PROFILE = defaultdict(lambda: {'calls': 0, 'time': 0.0})


def _timed(name, fn):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        _PROFILE[name]['calls'] += 1
        _PROFILE[name]['time']  += time.perf_counter() - t0
        return out
    wrapper.__wrapped__ = fn
    return wrapper


@contextmanager
def profile_hessian_components():
    """Monkey-patch the four Hessian kernels to record cumulative time.

    Only active inside the `with` block.  Use around a single IPOPT
    run — not around two, or the accumulators would double-count.
    """
    originals = {
        ('_vperim', 'compute_perimeter_hessian_sparse'):
            _vperim.compute_perimeter_hessian_sparse,
        ('_varea',  'compute_area_hessian_sparse'):
            _varea.compute_area_hessian_sparse,
        ('_vstein', 'compute_steiner_perimeter_hessian_fd'):
            _vstein.compute_steiner_perimeter_hessian_fd,
        ('_vstein', 'compute_steiner_area_hessian_fd'):
            _vstein.compute_steiner_area_hessian_fd,
    }
    _PROFILE.clear()
    try:
        _vperim.compute_perimeter_hessian_sparse     = _timed(
            'perimeter_hess', _vperim.compute_perimeter_hessian_sparse)
        _varea.compute_area_hessian_sparse           = _timed(
            'area_hess',      _varea.compute_area_hessian_sparse)
        _vstein.compute_steiner_perimeter_hessian_fd = _timed(
            'steiner_perim_hess_fd',
            _vstein.compute_steiner_perimeter_hessian_fd)
        _vstein.compute_steiner_area_hessian_fd      = _timed(
            'steiner_area_hess_fd',
            _vstein.compute_steiner_area_hessian_fd)
        yield _PROFILE
    finally:
        for (mod_name, attr), orig in originals.items():
            mod = {'_vperim': _vperim, '_varea': _varea,
                   '_vstein': _vstein}[mod_name]
            setattr(mod, attr, orig)


# ---------------------------------------------------------------------
# IPOPT built-in timing statistics
# ---------------------------------------------------------------------
#
# `print_timing_statistics yes` makes IPOPT write a block like:
#
#   OverallAlgorithm....................:     12.345 (sys:  0.021 wall:  12.380)
#   PDSystemSolver.....................:      7.234 (sys:  0.012 wall:   7.250)
#   ...
#
# to stdout after the run.  We redirect stdout to a buffer, run IPOPT,
# then parse the "PDSystemSolver" line for its wall time.  This is the
# only reliable way to separate "Python callback time" from "IPOPT
# linear-solve time".

import io, contextlib, re

_PD_RE = re.compile(
    r'PDSystemSolver\.*:\s*\S+\s*\(sys:\s*\S+\s*wall:\s*(\S+)\)')
_OA_RE = re.compile(
    r'OverallAlgorithm\.*:\s*\S+\s*\(sys:\s*\S+\s*wall:\s*(\S+)\)')


def _extract_ipopt_timing(log_text):
    pd = _PD_RE.search(log_text)
    oa = _OA_RE.search(log_text)
    return {
        'pd_solver_time':      float(pd.group(1)) if pd else None,
        'overall_ipopt_time':  float(oa.group(1)) if oa else None,
    }


# ---------------------------------------------------------------------
# One-mode runner
# ---------------------------------------------------------------------

def run_once(solution_path, exact_hessian, max_iter, tol, do_profile):
    opt = build_optimizer(solution_path)

    # IPOPT options to get the PDSystemSolver wall time in stdout.
    opt._ipopt_extra_options = {'print_timing_statistics': 'yes'}

    stdout_capture = io.StringIO()
    profile = None
    t0 = time.perf_counter()
    if do_profile:
        with profile_hessian_components() as p, \
             contextlib.redirect_stdout(stdout_capture):
            result = opt.optimize(
                max_iter=max_iter, tol=tol, method='ipopt',
                exact_hessian=exact_hessian,
            )
        profile = {k: dict(v) for k, v in p.items()}
    else:
        with contextlib.redirect_stdout(stdout_capture):
            result = opt.optimize(
                max_iter=max_iter, tol=tol, method='ipopt',
                exact_hessian=exact_hessian,
            )
    elapsed = time.perf_counter() - t0
    ipopt_timing = _extract_ipopt_timing(stdout_capture.getvalue())
    final_viol = float(np.max(np.abs(opt.constraint_area_equality(result.x))))

    return {
        'final_perimeter': float(result.fun),
        'final_viol':      final_viol,
        'iters':           int(getattr(result, 'nit', -1)),
        'status':          int(getattr(result, 'status', -1)),
        'time':            elapsed,
        'success':         bool(result.success),
        'ipopt_timing':    ipopt_timing,
        'profile':         profile,
    }


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def _fmt_row(label, seconds, calls, total_wall):
    pct = 100.0 * seconds / total_wall if total_wall > 0 else 0.0
    per_call = 1000.0 * seconds / calls if calls > 0 else 0.0
    return f"  {label:32s} {seconds:8.3f} s  "\
           f"{pct:5.1f}%  {calls:6d} calls  "\
           f"{per_call:7.2f} ms/call"


def print_component_breakdown(result_exact):
    total = result_exact['time']
    prof = result_exact['profile']
    ipopt_t = result_exact['ipopt_timing']

    print(f"\n  --- Per-component breakdown (total run = {total:.3f} s) ---")
    # Python-side hot-spots
    for name in ('perimeter_hess', 'area_hess',
                 'steiner_perim_hess_fd', 'steiner_area_hess_fd'):
        info = prof.get(name, {'calls': 0, 'time': 0.0})
        print(_fmt_row(name, info['time'], info['calls'], total))

    # Sum of the Python Hessian kernels
    py_hess = sum(prof[k]['time'] for k in prof
                  if k in ('perimeter_hess', 'area_hess',
                           'steiner_perim_hess_fd',
                           'steiner_area_hess_fd'))
    print(_fmt_row("  Σ Python Hessian kernels", py_hess, 0, total))

    # IPOPT linear solver (from stdout parsing)
    if ipopt_t['pd_solver_time'] is not None:
        pd = ipopt_t['pd_solver_time']
        pct = 100.0 * pd / total
        print(f"  {'IPOPT PDSystemSolver (lin. alg.)':32s} "
              f"{pd:8.3f} s  {pct:5.1f}%   (from print_timing_statistics)")

    # Everything else: evaluations of f, g, c, jac, Ipopt overhead, Python glue
    if ipopt_t['pd_solver_time'] is not None:
        other = max(0.0, total - py_hess - ipopt_t['pd_solver_time'])
        pct   = 100.0 * other / total
        print(f"  {'other (f, g, c, jac, overhead)':32s} "
              f"{other:8.3f} s  {pct:5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--solution', required=True)
    ap.add_argument('--max-iter', type=int, default=200)
    ap.add_argument('--tol',      type=float, default=1e-7)
    ap.add_argument('--no-profile', action='store_true',
                    help='Skip the per-component profile (faster).')
    args = ap.parse_args()

    print("=== L-BFGS ===")
    r_lbfgs = run_once(args.solution, exact_hessian=False,
                       max_iter=args.max_iter, tol=args.tol,
                       do_profile=False)
    for k in ('final_perimeter', 'final_viol', 'iters', 'time', 'success'):
        print(f"  {k:17s} {r_lbfgs[k]}")
    if r_lbfgs['ipopt_timing']['pd_solver_time'] is not None:
        print(f"  IPOPT PDSystemSolver "
              f"{r_lbfgs['ipopt_timing']['pd_solver_time']:.3f} s")

    print("\n=== Exact Hessian ===")
    r_exact = run_once(args.solution, exact_hessian=True,
                       max_iter=args.max_iter, tol=args.tol,
                       do_profile=not args.no_profile)
    for k in ('final_perimeter', 'final_viol', 'iters', 'time', 'success'):
        print(f"  {k:17s} {r_exact[k]}")
    if r_exact['ipopt_timing']['pd_solver_time'] is not None:
        print(f"  IPOPT PDSystemSolver "
              f"{r_exact['ipopt_timing']['pd_solver_time']:.3f} s")

    if r_exact['profile'] is not None:
        print_component_breakdown(r_exact)

    print("\n=== Summary ===")
    dp = r_exact['final_perimeter'] - r_lbfgs['final_perimeter']
    dt = r_exact['time']            - r_lbfgs['time']
    di = r_exact['iters']           - r_lbfgs['iters']
    print(f"  Δ perimeter (exact − lbfgs) = {dp:+.6e}")
    print(f"  Δ iters     (exact − lbfgs) = {di:+d}")
    print(f"  Δ wall time (exact − lbfgs) = {dt:+.2f} s")
    print(f"  per-iter  exact  = {r_exact['time']/max(1, r_exact['iters']):.3f} s")
    print(f"  per-iter  lbfgs  = {r_lbfgs['time']/max(1, r_lbfgs['iters']):.3f} s")


if __name__ == '__main__':
    raise SystemExit(main())
```

**Plumbing note (monkey-patch robustness).** The profiler patches
module-level symbols in `vectorized_perimeter`, `vectorized_area`, and
`vectorized_steiner`. That works **only** if the call sites inside
`PerimeterOptimizer` reach those functions via attribute access on the
module (`vectorized_perimeter.compute_perimeter_hessian_sparse(...)`)
rather than via `from ... import compute_perimeter_hessian_sparse` into
the optimizer's own namespace. Before writing this script, grep
`src/optimization/perimeter_optimizer.py` for the four function names
and confirm that is how they are called. If they are bound directly,
either (a) switch to module-qualified access in the optimizer (one-line
change), or (b) patch the names in `perimeter_optimizer` too. Either is
fine — document which you picked at the top of the script.

**Plumbing note (IPOPT options hook).** The sketch assumes the existing
`PerimeterOptimizer` already exposes a way to pass extra IPOPT options
through to `cyipopt.Problem.add_option`. If it does not, the cleanest
addition is a kwarg on `optimize()`: `extra_ipopt_options: dict[str, Any]
= None`. Patching the `IPOPTProblemAdapter` to accept this dict and call
`problem.add_option(k, v)` for each entry is ~5 lines. Do this as a
side-edit; it is generally useful (e.g. also for `linear_solver`,
`mu_strategy`, `hessian_approximation` overrides).

**Expected output on a reference checkpoint (illustrative — actual
numbers will differ).**

```
=== L-BFGS ===
  final_perimeter   12.3456789
  final_viol        1.2e-08
  iters             47
  time              8.42
  success           True
  IPOPT PDSystemSolver 2.145 s

=== Exact Hessian ===
  final_perimeter   12.3456701
  final_viol        9.4e-09
  iters             19
  time              6.18
  success           True
  IPOPT PDSystemSolver 1.802 s

  --- Per-component breakdown (total run = 6.180 s) ---
  perimeter_hess                      1.830 s   29.6%      19 calls    96.32 ms/call
  area_hess                           1.570 s   25.4%      19 calls    82.63 ms/call
  steiner_perim_hess_fd               0.090 s    1.5%      19 calls     4.74 ms/call
  steiner_area_hess_fd                0.085 s    1.4%      19 calls     4.47 ms/call
    Σ Python Hessian kernels          3.575 s   57.8%           0 calls
  IPOPT PDSystemSolver (lin. alg.)    1.802 s   29.2%   (from print_timing_statistics)
  other (f, g, c, jac, overhead)      0.803 s   13.0%
```

The scenario above would tell you: Phase A (vectorised accumulation of
perimeter and area Hessian) is the obvious win — it targets 55 % of
total time. Steiner FD is ~3 % — analytical Steiner is a correctness
/robustness improvement, not a wall-clock one. IPOPT's linear solver is
the second-largest bucket at 29 % — swapping MUMPS for MA57 is worth
trying afterwards.

**What "success" looks like.** The parent plan predicted (§6.3) that the
exact Hessian should give:

- Equal or lower final perimeter,
- Fewer IPOPT iterations,
- Fewer restoration phases (user will see this in the IPOPT log during
  the run — the script does not need to parse it).

Report both the end-to-end comparison and the component breakdown; do
**not** change any defaults based on a single problem. If exact-Hessian
is convincingly better on the reference problem(s), raise that as a
separate follow-up — it is outside the scope of this plan.

**Optional — cProfile dump for deeper digging.** If the four
wrapped functions do not account for most of the Python time (i.e. the
"other" bucket above is >40 %), add `--cprofile path.prof` to the CLI
and wrap the exact-Hessian run in `cProfile.Profile()`. Dump with
`pstats.Stats(prof).strip_dirs().sort_stats('cumulative').print_stats(25)`.
This takes ten lines and gives you the full top-of-stack distribution
when the coarse breakdown is inconclusive. Keep it behind a flag — it
adds ~30 % overhead to the profiled run.

### 4.5 Optional: hook tests into `testing/README_testing.md`

Add a short section listing the three new CLIs and their purpose, in the
same style as the existing entries. One paragraph each.

---

## 5. Phase C — Documentation Housekeeping <a id="5-phase-c"></a>

Open `docs/SPARSE_JACOBIAN_AND_EXACT_HESSIAN_PLAN.md` and make these
edits:

1. **Frontmatter todos.** Flip `status: pending` → `status: completed`
   for every id *except* `validate`. Leave `validate` as `pending` until
   Phase B in the present plan lands.
2. **§1.3 file paths.** Replace:
   - `src/core/partition_arrays.py` → `src/partition/partition_arrays.py`
   - `src/core/contour_partition.py` → `src/partition/contour_partition.py`
   - `src/core/vectorized_area.py` → `src/partition/vectorized_area.py`
   - `src/core/vectorized_steiner.py` → `src/partition/vectorized_steiner.py`
   - `src/core/vectorized_perimeter.py` → `src/partition/vectorized_perimeter.py`
   - `src/core/perimeter_optimizer.py` → `src/optimization/perimeter_optimizer.py`
   - `testing/refine_perimeter_iterative.py` → `scripts/refine_perimeter.py`
3. **§4.3 "2D case would go here".** Add a note that the 2D branch of
   `compute_area_hessian_sparse` was in fact implemented (it lives at
   `src/partition/vectorized_area.py:421-434, 500-526`); the cancellation
   is correct because `sign**2 == 1`.
4. **§4.2 "Performance note".** Add a one-line pointer:
   `See docs/EXACT_HESSIAN_VALIDATION_AND_PERF_PLAN.md — Phase A.`

Do not delete any content of the parent plan; it remains a useful
reference for the derivations.

---

## 6. Acceptance Criteria <a id="6-acceptance"></a>

The implementing agent should consider this plan complete when **all** of
the following hold. Do not ship partial deliveries.

### Phase A (performance)

- [ ] All nine new fields on `PartitionArrays` exist, have
      `Optional` type annotations, `= None` defaults, and are placed
      after `hess_offset_map`.
- [ ] `compile_arrays()` populates every new field and includes each
      one in the `return PartitionArrays(...)` call.
- [ ] `compute_perimeter_hessian_sparse()` uses `np.add.at` on the
      pre-computed offsets; a fallback path is preserved for backwards
      compatibility with an uncompiled `PartitionArrays`.
- [ ] Both blocks (`m1`, `m2`) of `compute_area_hessian_sparse()` use
      `np.add.at` with the pre-computed offsets and with the
      `btri*_cell_active` mask applied.
- [ ] On at least one reference checkpoint, the Hessian values returned
      by `_hessian_impl` before and after the refactor agree to within
      `1e-12` absolute.

### Phase B (validation)

- [ ] `testing/test_sparse_jacobian_equivalence.py` exits 0 on the
      reference problem with `--atol 1e-10`.
- [ ] `testing/test_exact_hessian_vs_fd.py` exits 0 on the reference
      problem with default tolerances (`--atol 1e-4 --rtol 1e-3`) in
      each of the three `--lagrange-mode` settings.
- [ ] (Optional) `testing/test_exact_hessian_matvec.py` exits 0 for a
      larger mesh where `test_exact_hessian_vs_fd.py` would print SKIP.
- [ ] `testing/compare_hessian_modes.py` runs end-to-end and prints a
      sensible summary block. No assertion on which method wins — this
      script is informational.
- [ ] `testing/compare_hessian_modes.py`, when invoked **without**
      `--no-profile`, additionally prints the "Per-component breakdown"
      block on the exact-Hessian run. The breakdown must list
      `perimeter_hess`, `area_hess`, `steiner_perim_hess_fd`,
      `steiner_area_hess_fd`, the Python-side sum, IPOPT's
      `PDSystemSolver` wall time (parsed from `print_timing_statistics
      yes`), and an "other" bucket. All seven rows have non-negative
      times; the row percentages add to ~100 % within rounding.
- [ ] The per-component breakdown is documented in
      `testing/README_testing.md` alongside the script's usage, with a
      one-paragraph reading guide (what "perimeter-H dominates" vs
      "linear solver dominates" vs "Steiner FD dominates" implies for
      further optimisation).
- [ ] Each script starts with a usage comment at the top and exits with
      code 0 on PASS / 1 on FAIL.

### Phase C (docs)

- [ ] `docs/SPARSE_JACOBIAN_AND_EXACT_HESSIAN_PLAN.md` frontmatter
      reflects the implementation status (all completed except
      `validate` → `pending` until Phase B lands, then `completed`).
- [ ] All old `src/core/*` paths in §1.3 of that file point to the
      current layout.
- [ ] A pointer from the parent plan's §4.2 Performance note to this
      plan is added.

### Global

- [ ] No change to any derivative formula.
- [ ] No change to any IPOPT option or adapter wiring.
- [ ] `pip install -e ".[ipopt]"` followed by
      `python scripts/refine_perimeter.py --method ipopt --exact-hessian
      --solution <ref> --config <ref yaml>` still runs to completion.

---

## 7. Reference: Key File Paths <a id="7-paths"></a>

Absolute paths inside the repository (you may need to `sys.path.insert`
the repo root when writing test scripts; see the helper at the top of
§4):

```
src/partition/partition_arrays.py            # Phase A edits: add fields
src/partition/contour_partition.py           # Phase A edits: compile_arrays(), ~L1077
src/partition/vectorized_perimeter.py        # Phase A edits: compute_perimeter_hessian_sparse
src/partition/vectorized_area.py             # Phase A edits: compute_area_hessian_sparse
src/partition/vectorized_steiner.py          # Do NOT edit (Phase A scope guard)
src/optimization/perimeter_optimizer.py      # Read-only for this plan
scripts/refine_perimeter.py                  # Entry point for acceptance runs
src/pipeline/io.py                           # load_partition_from_{base,refined}_file
testing/test_sparse_jacobian_equivalence.py  # Phase B NEW
testing/test_exact_hessian_vs_fd.py          # Phase B NEW
testing/test_exact_hessian_matvec.py         # Phase B NEW (optional)
testing/compare_hessian_modes.py             # Phase B NEW
testing/README_testing.md                    # Phase B: add entries for the above
docs/SPARSE_JACOBIAN_AND_EXACT_HESSIAN_PLAN.md  # Phase C edits
```

**Reference runs.** The agent should test against whichever small Phase-1
solution is available on the machine. A reasonable default is any
solution under `results/` produced from `parameters/torus_10part.yaml`.
If none exists, generate one with:

```bash
python scripts/find_surface_partition.py --config parameters/torus_10part.yaml
# Then pick the newest solution .h5 under results/
```

for the initial reference, followed by a short
`scripts/refine_perimeter.py --solution ... --max-iterations 1` run if a
post-refinement iteration file is needed (Phase 2 checkpoints exercise
the triple-point Steiner path, which the base solution does not).

---

**End of plan.**  Estimated effort: ~1 day for Phase A, ~1–1.5 days for
Phase B, ~30 minutes for Phase C.  Phase A must land before Phase B
because the tests should validate the vectorised code path, not the
legacy loop-based one.
