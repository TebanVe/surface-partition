---
name: Sparse Jacobian & Exact Hessian for IPOPT
overview: >
  Implement Phase 2 (direct sparse area Jacobian) and Phase 4 (exact Lagrangian
  Hessian) from the IPOPT integration roadmap.  Phase 2 eliminates the dense
  matrix allocation in every IPOPT callback.  Phase 4 replaces the L-BFGS
  approximation with an analytical Hessian, giving IPOPT full curvature
  information for smooth, high-quality partition boundaries at all mesh
  resolutions.
todos:
  - id: sparse-area-jac
    content: Implement compute_area_jacobian_sparse() in vectorized_area.py
    status: pending
  - id: sparse-steiner-jac
    content: Implement compute_steiner_area_jacobian_sparse() in vectorized_steiner.py
    status: pending
  - id: sparse-lookup
    content: Add nnz_lookup array to PartitionArrays for (row,col)→offset mapping
    status: pending
  - id: adapter-sparse-jac
    content: Replace dense extraction in IPOPTProblemAdapter.jacobian() with direct sparse call
    status: pending
  - id: hessian-structure
    content: Build Hessian sparsity pattern in compile_arrays() and add to PartitionArrays
    status: pending
  - id: perimeter-hessian
    content: Implement compute_perimeter_hessian_sparse() in vectorized_perimeter.py
    status: pending
  - id: area-hessian
    content: Implement compute_area_hessian_sparse() in vectorized_area.py
    status: pending
  - id: steiner-hessian
    content: Implement steiner Hessian contribution (finite-difference or analytical)
    status: pending
  - id: adapter-hessian
    content: Add hessian() and hessianstructure() to IPOPTProblemAdapter
    status: pending
  - id: cli-flag
    content: Add --exact-hessian CLI flag and wire through optimize()
    status: pending
  - id: validate
    content: Run IPOPT with exact Hessian vs L-BFGS and compare quality/convergence
    status: pending
isProject: false
---

# Sparse Jacobian & Exact Hessian Implementation Plan

## Table of Contents

1. [Context for the implementing agent](#1-context)
2. [Mathematical foundations](#2-math)
3. [Phase 2: Sparse Jacobian](#3-phase2)
4. [Phase 4: Exact Hessian of the Lagrangian](#4-phase4)
5. [Integration into the adapter and optimizer](#5-integration)
6. [Validation](#6-validation)
7. [Correctness checklist](#7-checklist)

---

## 1. Context for the Implementing Agent <a id="1-context"></a>

### 1.1 Current state

Phase 1 of the IPOPT integration is complete and working. The solver uses
`hessian_approximation='limited-memory'` (L-BFGS), and the constraint Jacobian
is computed as a **dense** `(n_cells-1, n_active_vp)` matrix, then the non-zero
entries are extracted by indexing with `pa.jac_row`/`pa.jac_col`. This is correct
but wastes memory at scale. A runtime warning fires when the dense matrix exceeds
50 MB. The L-BFGS Hessian also lacks curvature information needed for
high-quality boundaries at coarser mesh resolutions.

### 1.2 Goal

- **Phase 2**: Replace the dense Jacobian allocation with a direct sparse
  computation that writes only the non-zero entries.
- **Phase 4**: Provide IPOPT with an exact (analytical) Hessian of the
  Lagrangian, eliminating the L-BFGS approximation entirely.

### 1.3 Files to modify

| File | Changes |
|------|---------|
| `src/core/partition_arrays.py` | Add `nnz_lookup`, Hessian sparsity arrays |
| `src/core/contour_partition.py` | Build `nnz_lookup` and Hessian sparsity in `compile_arrays()` |
| `src/core/vectorized_area.py` | Add `compute_area_jacobian_sparse()`, `compute_area_hessian_sparse()` |
| `src/core/vectorized_steiner.py` | Add `compute_steiner_area_jacobian_sparse()`, Steiner Hessian |
| `src/core/vectorized_perimeter.py` | Add `compute_perimeter_hessian_sparse()` |
| `src/core/perimeter_optimizer.py` | Update `IPOPTProblemAdapter`, add `hessian()`/`hessianstructure()`, add `exact_hessian` parameter to `optimize()` |
| `testing/refine_perimeter_iterative.py` | Add `--exact-hessian` CLI flag |

### 1.4 Key data structures

Read `src/core/partition_arrays.py` for the full `PartitionArrays` dataclass.
The fields most relevant to this plan are:

- **VP geometry**: `vp_edge_v1`, `vp_edge_v2`, `vp_lambda` — each VP sits on
  a mesh edge at position `p = λ·v1 + (1-λ)·v2`. The optimization variable is
  the scalar `λ ∈ [0, 1]`.
- **Perimeter segments**: `seg_vp1`, `seg_vp2`, `seg_cell_a`, `seg_cell_b` —
  each segment connects two VPs and separates two cells.
- **Boundary triangles**: `btri_cell`, `btri_vp1`, `btri_vp2`, `btri_n_inside`,
  `btri_v_in` — mesh triangles cut by a cell boundary, with 1 or 2 mesh
  vertices inside the cell.
- **Triple points**: `tp_vp_indices` (n_tp, 3), `tp_contrib_*` arrays —
  Steiner point data for junctions where 3 cells meet.
- **Sparse Jacobian**: `jac_row`, `jac_col` — already computed in Phase 1.

### 1.5 How VP positions depend on λ

Every VP `i` sits on mesh edge `(v1_i, v2_i)`:

    p_i(λ_i) = λ_i · V[v1_i] + (1 − λ_i) · V[v2_i]

The derivative of the 3D position w.r.t. the scalar parameter is the constant:

    d_i = dp_i/dλ_i = V[v1_i] − V[v2_i]     (the edge direction vector)

This is the single most important identity for all derivative computations.
Since `d_i` is constant (independent of λ), all **second derivatives of p_i
w.r.t. λ_i are zero**: `d²p_i/dλ_i² = 0`. This simplifies the Hessian
calculations significantly — many second derivative terms vanish.

---

## 2. Mathematical Foundations <a id="2-math"></a>

### 2.1 The optimization problem

Minimize total perimeter f(λ) subject to area equality constraints:

    min_λ  f(λ) = P_regular(λ) + P_steiner(λ)
    s.t.   c_k(λ) = A_k(λ) − A_target = 0,   k = 0, ..., n_cells−2

where `λ ∈ ℝⁿ` is the vector of VP parameters (n = n_active_vp).

### 2.2 The Lagrangian

IPOPT's Hessian callback requires the Hessian of the Lagrangian:

    L(λ, σ, μ) = σ · f(λ) + Σ_k μ_k · c_k(λ)

where σ is the objective scaling factor and μ_k are the constraint multipliers
(both provided by IPOPT at each iteration).

The Hessian is:

    H_ij = σ · ∂²f/∂λ_i∂λ_j + Σ_k μ_k · ∂²c_k/∂λ_i∂λ_j

IPOPT expects the **lower triangle** of H (it is symmetric).

### 2.3 Perimeter: function, gradient, Hessian

#### Regular perimeter

Each segment connects VP `a` and VP `b`:

    L_s = w_s · ‖p_a − p_b‖

where `w_s ∈ {1, 2}` (2 for proper inter-cell segments, 1 for fallback).

Total regular perimeter:

    P = Σ_s L_s = Σ_s w_s · ‖p_a(λ_a) − p_b(λ_b)‖

**Gradient** (already implemented in `vectorized_perimeter.py`):

    ∂P/∂λ_a = Σ_{s: vp1=a} w_s · (Δ_s · d_a) / ‖Δ_s‖
    ∂P/∂λ_b = Σ_{s: vp2=b} w_s · (−Δ_s · d_b) / ‖Δ_s‖

where `Δ_s = p_a − p_b` and `d_i = V[v1_i] − V[v2_i]`.

**Hessian derivation**:

For a single segment with endpoints `p_a(λ_a)` and `p_b(λ_b)`:

    L_s = w · ‖Δ‖,    Δ = p_a − p_b,    ‖Δ‖ = √(Δ·Δ)

Let `r = ‖Δ‖`. The first derivatives are:

    ∂L/∂λ_a = w · (Δ · d_a) / r
    ∂L/∂λ_b = w · (−Δ · d_b) / r

For the second derivatives, we need `∂²L/∂λ_i∂λ_j`. Using the quotient rule
on `(Δ · d_a) / r`:

The key identity: for `g(Δ) = ‖Δ‖`, we have

    ∂²‖Δ‖/∂Δ_m∂Δ_n = (δ_mn − Δ_m Δ_n / r²) / r = (I − Δ̂Δ̂ᵀ) / r

where `Δ̂ = Δ/r` is the unit direction vector. This is the standard Hessian of
the Euclidean norm.

Since `∂Δ/∂λ_a = d_a` and `∂Δ/∂λ_b = −d_b`, by chain rule:

    ∂²L_s/∂λ_a² = w · dₐᵀ · (I − Δ̂Δ̂ᵀ)/r · dₐ
                 = w · (‖dₐ‖² − (dₐ·Δ̂)²) / r

    ∂²L_s/∂λ_b² = w · d_bᵀ · (I − Δ̂Δ̂ᵀ)/r · d_b
                 = w · (‖d_b‖² − (d_b·Δ̂)²) / r

    ∂²L_s/∂λ_a∂λ_b = w · dₐᵀ · (I − Δ̂Δ̂ᵀ)/r · (−d_b)
                    = −w · (dₐ·d_b − (dₐ·Δ̂)(d_b·Δ̂)) / r

**In scalar notation** (the form you should implement):

For segment s with endpoints VP a, VP b:

```
r = ‖p_a − p_b‖
Δ̂ = (p_a − p_b) / r         # unit direction, shape (dim,)
d_a = V[v1_a] − V[v2_a]      # edge direction of VP a
d_b = V[v1_b] − V[v2_b]      # edge direction of VP b

H_aa = w · (‖d_a‖² − (d_a · Δ̂)²) / r      # diagonal entry for VP a
H_bb = w · (‖d_b‖² − (d_b · Δ̂)²) / r      # diagonal entry for VP b
H_ab = −w · (d_a · d_b − (d_a · Δ̂)(d_b · Δ̂)) / r   # off-diagonal
```

These are scalar values to be accumulated into the Hessian array. Each segment
contributes at most 3 Hessian entries: (a,a), (b,b), (a,b). Due to symmetry,
only the lower triangle is stored, so (a,b) is stored at `(max(a,b), min(a,b))`.

**Geometric intuition**: `(I − Δ̂Δ̂ᵀ)/r` is the projection perpendicular to
the segment, divided by its length. When the edge direction `d` is parallel to
the segment, the Hessian entry is zero (moving along the segment direction
doesn't change its length to first order — that's already captured by the
gradient). When `d` is perpendicular, the Hessian entry is `w·‖d‖²/r`
(maximum curvature).

#### Steiner perimeter

The Steiner perimeter contribution involves the Fermat-Torricelli point, which
is itself a function of the 3 VP positions at each triple point. The
closed-form formula for the Steiner point is complex (involving arccos,
barycentric weights, etc.) — its second derivatives are analytically tractable
but extremely verbose.

**Recommended approach: finite differences for the Steiner Hessian contribution.**

The Steiner perimeter gradient is already computed via finite differences in
`compute_steiner_perimeter_gradient()`. For the Hessian, use second-order
central finite differences:

```
∂²P_steiner/∂λ_i∂λ_j ≈ (g_j(λ_i+ε) − g_j(λ_i−ε)) / (2ε)
```

where `g_j(λ) = ∂P_steiner/∂λ_j` evaluated at the perturbed λ.

Only VPs in `pa.tp_affected_vps` have non-zero Steiner derivatives, so the
finite-difference loop is over at most `3 × n_triple_points` VPs (typically a
very small fraction of all VPs). The cost is `O(n_tp_vps²)` which is negligible.

### 2.4 Area constraints: function, Jacobian, Hessian

#### Boundary triangle areas (1-inside case)

For a mesh triangle with 1 interior vertex `p_in` and 2 cut points
`pc1 = p(λ_1)`, `pc2 = p(λ_2)`:

    A = ½ ‖(pc1 − p_in) × (pc2 − p_in)‖

Let:
- `u = pc1 − p_in`, `v = pc2 − p_in`
- `d₁ = V[v1₁] − V[v2₁]` (edge direction for VP 1)
- `d₂ = V[v1₂] − V[v2₂]` (edge direction for VP 2)
- `n̂ = (u × v) / ‖u × v‖` (unit normal of the triangle)

**First derivatives** (already implemented):

    ∂A/∂λ₁ = ½ n̂ · (d₁ × v)
    ∂A/∂λ₂ = ½ n̂ · (u × d₂)

**Second derivatives**:

Since `d²p/dλ² = 0`, the second derivatives come from differentiating the unit
normal `n̂` and the cross products.

The area is `A = ½‖C‖` where `C = u × v`. By the chain rule:

    ∂A/∂λ₁ = ½ Ĉ · (∂C/∂λ₁)

where `Ĉ = C/‖C‖ = n̂`.

Now `∂C/∂λ₁ = d₁ × v` (since `∂u/∂λ₁ = d₁` and `∂v/∂λ₁ = 0`).
And `∂C/∂λ₂ = u × d₂` (since `∂u/∂λ₂ = 0` and `∂v/∂λ₂ = d₂`).

For the second derivative, use the product rule on `Ĉ · (d₁ × v)`:

    ∂²A/∂λ₁² = ½ [(d₁ × v)ᵀ · ∂Ĉ/∂λ₁ + Ĉ · ∂(d₁ × v)/∂λ₁]

The second term is zero because `d₁ × v` does not depend on λ₁ (both d₁ and
v = pc2 − p_in are independent of λ₁).

For `∂Ĉ/∂λ₁`: since `Ĉ = C/‖C‖`:

    ∂Ĉ/∂λ₁ = (I/‖C‖ − C·Cᵀ/‖C‖³) · ∂C/∂λ₁
             = (I − n̂·n̂ᵀ) · (d₁ × v) / ‖C‖

So:

    ∂²A/∂λ₁² = ½ (d₁ × v)ᵀ · (I − n̂·n̂ᵀ) · (d₁ × v) / ‖C‖
              = [‖d₁ × v‖² − (n̂ · (d₁ × v))²] / (2‖C‖)

Similarly:

    ∂²A/∂λ₂² = [‖u × d₂‖² − (n̂ · (u × d₂))²] / (2‖C‖)

For the cross term:

    ∂²A/∂λ₁∂λ₂ = ½ [(d₁ × v)ᵀ · ∂Ĉ/∂λ₂ + Ĉ · ∂(d₁ × v)/∂λ₂]

The second term: `∂(d₁ × v)/∂λ₂ = d₁ × d₂` (since `∂v/∂λ₂ = d₂`).

The first term uses `∂Ĉ/∂λ₂ = (I − n̂·n̂ᵀ) · (u × d₂) / ‖C‖`:

    ∂²A/∂λ₁∂λ₂ = ½ [(d₁ × v)ᵀ · (I − n̂n̂ᵀ) · (u × d₂) / ‖C‖ + n̂ · (d₁ × d₂)]

**In scalar notation for implementation (3D)**:

```
C = u × v                   # cross product, shape (dim,)
normC = ‖C‖                 # = 2A
n̂ = C / normC

g1 = d₁ × v                 # ∂C/∂λ₁
g2 = u × d₂                 # ∂C/∂λ₂

# Diagonal entries
H_11 = (‖g1‖² − (n̂ · g1)²) / (2 · normC)
H_22 = (‖g2‖² − (n̂ · g2)²) / (2 · normC)

# Off-diagonal
H_12 = ((g1 · g2 − (n̂ · g1)(n̂ · g2)) / normC + n̂ · (d₁ × d₂)) / 2
```

**IMPORTANT sign handling**: The area computation uses `|signed_area|`. The
sign discontinuity at zero area is non-differentiable, but in practice
triangles never have zero area during optimization. The sign from the first
derivative computation must be consistently applied.

For the **2D case**: the cross product becomes a scalar and the formulas
simplify. However, this codebase operates on 3D surface meshes (torus), so
**implement the 3D case first**. The 2D case can be added later if needed.

#### Boundary triangle areas (2-inside case)

The quad area is split into two sub-triangles:

    A = A₁(p_in1, pc1, pc2) + A₂(p_in1, pc2, p_in2)

VP 1 appears only in A₁ (through pc1). VP 2 appears in both A₁ and A₂
(through pc2). The Hessian contributions add:

```
H_11 = ∂²A₁/∂λ₁²                   (only from sub-triangle 1)
H_22 = ∂²A₁/∂λ₂² + ∂²A₂/∂λ₂²      (from both sub-triangles)
H_12 = ∂²A₁/∂λ₁∂λ₂                 (only from sub-triangle 1)
```

Each sub-triangle Hessian uses the same formulas as the 1-inside case with
appropriate substitutions of vertices.

#### Steiner area Hessian

Same recommendation as Steiner perimeter: **use finite differences**. The
Steiner area Jacobian is already computed via finite differences in
`compute_steiner_area_jacobian()`. For the Hessian, differentiate the Jacobian
numerically:

```python
def compute_steiner_area_hessian_fd(pa, eps=1e-6):
    """∂²(steiner_areas)/∂λ_i∂λ_j for affected VPs only."""
    base_jac = compute_steiner_area_jacobian(pa)  # (n_cells-1, n_vp)
    original = pa.vp_lambda.copy()
    # For each affected VP i, perturb and compute Jacobian
    # H[:, i, j] = (jac_perturbed[:, j] - base_jac[:, j]) / eps
```

Only `tp_affected_vps` need perturbation. The result is a set of Hessian
entries for each constraint, but only involving the (typically few) triple-point
VPs.

### 2.5 Hessian sparsity structure

The Hessian `H` is `n × n` (n = n_active_vp) and symmetric. IPOPT expects only
the **lower triangle** entries.

**Non-zero pattern**:

1. **Perimeter Hessian**: Non-zero at `(a, b)` for every segment connecting
   VP a and VP b, and diagonal entries `(a, a)`, `(b, b)` for each segment.
   Source: `pa.seg_vp1`, `pa.seg_vp2`.

2. **Area Hessian (boundary triangles)**: Non-zero at `(vp1, vp2)` for every
   boundary triangle, plus diagonals `(vp1, vp1)`, `(vp2, vp2)`.
   Source: `pa.btri_vp1`, `pa.btri_vp2`.

3. **Steiner Hessian**: Non-zero for all pairs of VPs within each triple
   point. Since each triple point has 3 VPs, this adds up to 6 entries per
   triple point (3 diagonal + 3 off-diagonal pairs).
   Source: `pa.tp_vp_indices`.

The union of these gives the Hessian sparsity pattern. Since the perimeter
segments connect the same VPs as the boundary triangles (they are the same
boundary edges), the Hessian sparsity pattern is roughly a **superset** of the
Jacobian column-pair structure, augmented with diagonal entries and
triple-point cross-entries.

---

## 3. Phase 2: Sparse Jacobian <a id="3-phase2"></a>

### 3.1 Add nnz_lookup to PartitionArrays

The key data structure for direct sparse evaluation is a lookup array that maps
each `(row, col)` pair in the Jacobian to its offset in the flat values array.

In `src/core/partition_arrays.py`, add:

```python
# Sparse Jacobian value-offset lookup
# For boundary triangle k with cell c and VPs (vp1, vp2):
#   nnz_lookup[(c, vp1)] → offset into the flat values array
# Built by compile_arrays() after jac_row/jac_col are known.
nnz_lookup: np.ndarray   # int32 (n_cells-1, n_active_vp) — offset or -1
```

**Alternative (recommended for memory at scale)**: Instead of a dense lookup
matrix, store a dictionary or use `np.searchsorted` on lexicographically sorted
`(jac_row, jac_col)` pairs:

```python
# In compile_arrays(), after building jac_row/jac_col:
# Sort by (row, col) for binary search — already done by np.unique
jac_nnz_map: np.ndarray   # int32 (n_cells-1, n_active_vp) or sparse lookup
```

For current scale (< 50 cells, < 10k VPs), a dense `(n_cells-1, n_active_vp)`
int32 lookup is fine (~40 KB for 4×10000). At thousands of cells, use a
hash-based approach.

**Implementation in `compile_arrays()`**:

After the existing `jac_row`/`jac_col` computation:

```python
nnz = len(jac_row)
nnz_lookup = -np.ones((self.n_cells - 1, n_active), dtype=np.int32)
for idx in range(nnz):
    nnz_lookup[jac_row[idx], jac_col[idx]] = idx
```

Add `nnz_lookup` to the `PartitionArrays` dataclass and the `return` call.

### 3.2 Implement compute_area_jacobian_sparse()

In `src/core/vectorized_area.py`, add a new function:

```python
def compute_area_jacobian_sparse(pa: PartitionArrays) -> np.ndarray:
    """Compute area Jacobian non-zero values in jac_row/jac_col order.

    Direct sparse computation — no dense matrix allocation.

    Returns:
        (nnz,) float64 — values at positions (pa.jac_row, pa.jac_col).
    """
    nnz = len(pa.jac_row)
    values = np.zeros(nnz, dtype=np.float64)
    pos = _compute_vp_positions(pa)
    dim = pa.vertices.shape[1]

    # --- 1-inside triangles ---
    m1 = pa.btri_n_inside == 1
    if np.any(m1):
        p_in = pa.vertices[pa.btri_v_in[m1, 0]]
        pc1 = pos[pa.btri_vp1[m1]]
        pc2 = pos[pa.btri_vp2[m1]]

        u = pc1 - p_in
        v = pc2 - p_in

        d1 = (pa.vertices[pa.vp_edge_v1[pa.btri_vp1[m1]]]
              - pa.vertices[pa.vp_edge_v2[pa.btri_vp1[m1]]])
        d2 = (pa.vertices[pa.vp_edge_v1[pa.btri_vp2[m1]]]
              - pa.vertices[pa.vp_edge_v2[pa.btri_vp2[m1]]])

        if dim == 3:
            cross_uv = np.cross(u, v)
            norm_uv = np.maximum(np.linalg.norm(cross_uv, axis=1, keepdims=True), 1e-30)
            n_hat = cross_uv / norm_uv
            dA_dl1 = 0.5 * np.sum(n_hat * np.cross(d1, v), axis=1)
            dA_dl2 = 0.5 * np.sum(n_hat * np.cross(u, d2), axis=1)
        else:
            signed_area = 0.5 * (u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
            sign = np.sign(signed_area)
            sign[sign == 0] = 1.0
            dA_dl1 = sign * 0.5 * (d1[:, 0] * v[:, 1] - d1[:, 1] * v[:, 0])
            dA_dl2 = sign * 0.5 * (u[:, 0] * d2[:, 1] - u[:, 1] * d2[:, 0])

        cells = pa.btri_cell[m1]
        vp1 = pa.btri_vp1[m1]
        vp2 = pa.btri_vp2[m1]
        mask = cells < (pa.n_cells - 1)  # only constrained cells

        offsets1 = pa.nnz_lookup[cells[mask], vp1[mask]]
        offsets2 = pa.nnz_lookup[cells[mask], vp2[mask]]
        np.add.at(values, offsets1, dA_dl1[mask])
        np.add.at(values, offsets2, dA_dl2[mask])

    # --- 2-inside triangles (similar pattern, using both sub-triangles) ---
    m2 = pa.btri_n_inside == 2
    if np.any(m2):
        # ... (mirror the analytical computation from compute_area_jacobian_analytical,
        #       but write into `values` via nnz_lookup instead of the dense matrix)
        pass  # Full implementation follows same pattern as 1-inside case

    return values
```

The critical change from the dense version: instead of `jacobian[c, vp_idx] += val`,
write `np.add.at(values, pa.nnz_lookup[c, vp_idx], val)`.

### 3.3 Implement compute_steiner_area_jacobian_sparse()

In `src/core/vectorized_steiner.py`, add:

```python
def compute_steiner_area_jacobian_sparse(pa: PartitionArrays,
                                         eps: float = 1e-7) -> np.ndarray:
    """Steiner area Jacobian — sparse values in jac_row/jac_col order.

    Uses the same finite-difference approach as compute_steiner_area_jacobian()
    but writes into a sparse values array instead of a dense matrix.

    Returns:
        (nnz,) float64
    """
    nnz = len(pa.jac_row)
    values = np.zeros(nnz, dtype=np.float64)
    if pa.n_triple_points == 0:
        return values

    n_constraints = pa.n_cells - 1
    base_steiner = compute_steiner_points(pa)
    base_areas = compute_steiner_areas(pa, base_steiner)
    original_lambda = pa.vp_lambda.copy()

    for vp_idx in pa.tp_affected_vps:
        pa.vp_lambda[vp_idx] = original_lambda[vp_idx] + eps
        pert_steiner = compute_steiner_points(pa)
        pert_areas = compute_steiner_areas(pa, pert_steiner)
        dA = (pert_areas[:n_constraints] - base_areas[:n_constraints]) / eps

        for c in range(n_constraints):
            offset = pa.nnz_lookup[c, vp_idx]
            if offset >= 0:
                values[offset] += dA[c]

        pa.vp_lambda[vp_idx] = original_lambda[vp_idx]

    return values
```

### 3.4 Update IPOPTProblemAdapter.jacobian()

Replace the current dense-then-extract approach:

```python
def jacobian(self, x: np.ndarray) -> np.ndarray:
    """Return non-zero Jacobian values in jacobianstructure() order.

    Phase 2: direct sparse computation — no dense matrix allocated.
    """
    self._opt._arrays.vp_lambda[:] = x
    pa = self._opt._arrays

    area_vals = vectorized_area.compute_area_jacobian_sparse(pa)
    steiner_vals = vectorized_steiner.compute_steiner_area_jacobian_sparse(pa)

    return area_vals + steiner_vals
```

**IMPORTANT**: The adapter must set `vp_lambda` before calling the sparse
functions, just as the optimizer's `constraint_area_jacobian()` does. Check
that the sparse functions do NOT call `_compute_vp_positions` with stale
lambdas. The safest approach is to set `pa.vp_lambda[:] = x` at the start.

### 3.5 Remove the 50 MB warning

Once Phase 2 is implemented, the dense matrix is no longer allocated, so the
warning in `jacobian()` should be removed.

---

## 4. Phase 4: Exact Hessian of the Lagrangian <a id="4-phase4"></a>

### 4.1 Build Hessian sparsity pattern

In `compile_arrays()` in `src/core/contour_partition.py`, build the Hessian
sparsity pattern analogous to the Jacobian pattern.

The Hessian is symmetric, so IPOPT requires only the **lower triangle** (i.e.,
entries where `row >= col`).

```python
# --- Hessian sparsity pattern ---
hess_pairs = set()  # (row, col) with row >= col

# Perimeter segments: each segment (a, b) contributes (a,a), (b,b), (max,min)
for s in range(n_seg):
    a = seg_vp1[s]
    b = seg_vp2[s]
    hess_pairs.add((a, a))
    hess_pairs.add((b, b))
    hess_pairs.add((max(a, b), min(a, b)))

# Boundary triangles: each (vp1, vp2) pair
for k in range(n_btri):
    a = btri_vp1[k]
    b = btri_vp2[k]
    hess_pairs.add((a, a))
    hess_pairs.add((b, b))
    hess_pairs.add((max(a, b), min(a, b)))

# Triple points: all 3×3 pairs within each triple point
for tp_i in range(n_tp):
    vps = tp_vp_indices[tp_i]  # 3 VPs
    for i in range(3):
        for j in range(i + 1):
            hess_pairs.add((max(vps[i], vps[j]), min(vps[i], vps[j])))

hess_pairs_arr = np.array(sorted(hess_pairs), dtype=np.int32)
hess_row = hess_pairs_arr[:, 0]
hess_col = hess_pairs_arr[:, 1]
```

Add to `PartitionArrays`:

```python
# Hessian sparsity (lower triangle only, for IPOPT)
hess_row: np.ndarray     # int32 (hess_nnz,) — row indices (row >= col)
hess_col: np.ndarray     # int32 (hess_nnz,) — col indices
hess_nnz_lookup: np.ndarray  # maps (vp_i, vp_j) → offset (with i >= j)
```

For `hess_nnz_lookup`, build a mapping similar to `nnz_lookup` but for the
VP×VP space. Since the Hessian is sparse, a dictionary `{(i,j): offset}` is
acceptable and cleaner than a dense n×n matrix:

```python
hess_offset_map = {}
for idx, (r, c) in enumerate(zip(hess_row, hess_col)):
    hess_offset_map[(r, c)] = idx
```

Store this as a Python dict in the dataclass (it won't be serialized to HDF5
but is only used during the optimization loop). Alternatively, store it as two
parallel arrays for `np.searchsorted`-based lookup.

### 4.2 Implement compute_perimeter_hessian_sparse()

In `src/core/vectorized_perimeter.py`, add:

```python
def compute_perimeter_hessian_sparse(pa: PartitionArrays) -> np.ndarray:
    """Compute ∂²P/∂λ_i∂λ_j for the regular perimeter.

    Returns:
        (hess_nnz,) float64 — values at positions (pa.hess_row, pa.hess_col).
    """
    hess_nnz = len(pa.hess_row)
    values = np.zeros(hess_nnz, dtype=np.float64)
    pos = _compute_vp_positions(pa)

    p1 = pos[pa.seg_vp1]
    p2 = pos[pa.seg_vp2]
    delta = p1 - p2
    r = np.linalg.norm(delta, axis=1)
    r = np.maximum(r, 1e-12)
    delta_hat = delta / r[:, None]    # unit direction

    d_a = (pa.vertices[pa.vp_edge_v1[pa.seg_vp1]]
           - pa.vertices[pa.vp_edge_v2[pa.seg_vp1]])
    d_b = (pa.vertices[pa.vp_edge_v1[pa.seg_vp2]]
           - pa.vertices[pa.vp_edge_v2[pa.seg_vp2]])

    weights = np.where(pa.seg_cell_a != pa.seg_cell_b, 2.0, 1.0)

    # Dot products needed for the Hessian formula
    da_dot_da = np.sum(d_a * d_a, axis=1)
    db_dot_db = np.sum(d_b * d_b, axis=1)
    da_dot_db = np.sum(d_a * d_b, axis=1)
    da_dot_dh = np.sum(d_a * delta_hat, axis=1)
    db_dot_dh = np.sum(d_b * delta_hat, axis=1)

    # Hessian entries per segment
    H_aa = weights * (da_dot_da - da_dot_dh**2) / r
    H_bb = weights * (db_dot_db - db_dot_dh**2) / r
    H_ab = -weights * (da_dot_db - da_dot_dh * db_dot_dh) / r

    # Accumulate into values array
    vp1 = pa.seg_vp1
    vp2 = pa.seg_vp2

    for s in range(len(vp1)):
        a, b = int(vp1[s]), int(vp2[s])

        # Diagonal entries: (a,a) and (b,b)
        off_aa = pa.hess_offset_map[(a, a)]
        values[off_aa] += H_aa[s]
        off_bb = pa.hess_offset_map[(b, b)]
        values[off_bb] += H_bb[s]

        # Off-diagonal: store in lower triangle (max, min)
        hi, lo = max(a, b), min(a, b)
        off_ab = pa.hess_offset_map[(hi, lo)]
        values[off_ab] += H_ab[s]

    return values
```

**Performance note**: The Python loop over segments is acceptable for the
initial implementation but can be vectorized later if it becomes a bottleneck.
To vectorize, pre-compute the offset arrays during `compile_arrays()`:

```python
# Pre-computed in compile_arrays():
seg_hess_off_aa = np.array([hess_offset_map[(a, a)] for a, b in zip(seg_vp1, seg_vp2)])
seg_hess_off_bb = np.array([hess_offset_map[(b, b)] for a, b in zip(seg_vp1, seg_vp2)])
seg_hess_off_ab = np.array([hess_offset_map[(max(a,b), min(a,b))] for a, b in zip(seg_vp1, seg_vp2)])
```

Then accumulate with `np.add.at(values, seg_hess_off_aa, H_aa)`.

### 4.3 Implement compute_area_hessian_sparse()

In `src/core/vectorized_area.py`, add:

```python
def compute_area_hessian_sparse(pa: PartitionArrays,
                                multipliers: np.ndarray) -> np.ndarray:
    """Compute Σ_k μ_k · ∂²A_k/∂λ_i∂λ_j for the area constraints.

    The multiplier-weighted sum is computed directly to avoid building
    per-constraint Hessians separately.

    Args:
        pa: PartitionArrays snapshot
        multipliers: (n_cells-1,) float64 — Lagrange multipliers from IPOPT

    Returns:
        (hess_nnz,) float64 — values at positions (pa.hess_row, pa.hess_col).
    """
    hess_nnz = len(pa.hess_row)
    values = np.zeros(hess_nnz, dtype=np.float64)
    pos = _compute_vp_positions(pa)
    dim = pa.vertices.shape[1]

    n_c = pa.n_cells - 1

    # --- 1-inside triangles ---
    m1 = pa.btri_n_inside == 1
    if np.any(m1):
        p_in = pa.vertices[pa.btri_v_in[m1, 0]]
        pc1 = pos[pa.btri_vp1[m1]]
        pc2 = pos[pa.btri_vp2[m1]]

        u = pc1 - p_in
        v = pc2 - p_in

        d1 = (pa.vertices[pa.vp_edge_v1[pa.btri_vp1[m1]]]
              - pa.vertices[pa.vp_edge_v2[pa.btri_vp1[m1]]])
        d2 = (pa.vertices[pa.vp_edge_v1[pa.btri_vp2[m1]]]
              - pa.vertices[pa.vp_edge_v2[pa.btri_vp2[m1]]])

        if dim == 3:
            C = np.cross(u, v)
            normC = np.maximum(np.linalg.norm(C, axis=1), 1e-30)
            n_hat = C / normC[:, None]

            g1 = np.cross(d1, v)            # ∂C/∂λ₁
            g2 = np.cross(u, d2)            # ∂C/∂λ₂
            d1xd2 = np.cross(d1, d2)        # d₁ × d₂

            # Dot products
            n_dot_g1 = np.sum(n_hat * g1, axis=1)
            n_dot_g2 = np.sum(n_hat * g2, axis=1)
            g1_dot_g1 = np.sum(g1 * g1, axis=1)
            g2_dot_g2 = np.sum(g2 * g2, axis=1)
            g1_dot_g2 = np.sum(g1 * g2, axis=1)
            n_dot_d1xd2 = np.sum(n_hat * d1xd2, axis=1)

            H_11 = (g1_dot_g1 - n_dot_g1**2) / (2.0 * normC)
            H_22 = (g2_dot_g2 - n_dot_g2**2) / (2.0 * normC)
            H_12 = ((g1_dot_g2 - n_dot_g1 * n_dot_g2) / normC
                     + n_dot_d1xd2) / 2.0

        # 2D case would go here (omitted for brevity — implement if needed)

        cells = pa.btri_cell[m1]
        vp1_arr = pa.btri_vp1[m1]
        vp2_arr = pa.btri_vp2[m1]

        for k in range(len(cells)):
            c = int(cells[k])
            if c >= n_c:
                continue
            mu = multipliers[c]
            a = int(vp1_arr[k])
            b = int(vp2_arr[k])

            values[pa.hess_offset_map[(a, a)]] += mu * H_11[k]
            values[pa.hess_offset_map[(b, b)]] += mu * H_22[k]
            hi, lo = max(a, b), min(a, b)
            values[pa.hess_offset_map[(hi, lo)]] += mu * H_12[k]

    # --- 2-inside triangles ---
    m2 = pa.btri_n_inside == 2
    if np.any(m2):
        # Similar structure: compute Hessian for both sub-triangles
        # and accumulate with multiplier weighting.
        # Sub-triangle 1: (p_in1, pc1, pc2) — both λ₁ and λ₂ appear
        # Sub-triangle 2: (p_in1, pc2, p_in2) — only λ₂ appears
        pass  # implement following the same pattern

    return values
```

**CRITICAL**: The `multipliers` array comes from IPOPT's `hessian()` callback
signature. IPOPT passes `(x, lagrange, obj_factor)` — the `lagrange` argument
is the vector of constraint multipliers μ_k.

### 4.4 Implement Steiner Hessian (finite differences)

In `src/core/vectorized_steiner.py`, add:

```python
def compute_steiner_perimeter_hessian_fd(pa: PartitionArrays,
                                         eps: float = 1e-5) -> np.ndarray:
    """∂²P_steiner/∂λ_i∂λ_j via central finite differences on the gradient.

    Only tp_affected_vps have non-zero entries.

    Returns:
        (hess_nnz,) float64
    """
    hess_nnz = len(pa.hess_row)
    values = np.zeros(hess_nnz, dtype=np.float64)
    if pa.n_triple_points == 0:
        return values

    original = pa.vp_lambda.copy()
    affected = pa.tp_affected_vps

    base_grad = compute_steiner_perimeter_gradient(pa, eps=eps)

    for vp_i in affected:
        pa.vp_lambda[vp_i] = original[vp_i] + eps
        grad_plus = compute_steiner_perimeter_gradient(pa, eps=eps)
        pa.vp_lambda[vp_i] = original[vp_i] - eps
        grad_minus = compute_steiner_perimeter_gradient(pa, eps=eps)
        pa.vp_lambda[vp_i] = original[vp_i]

        hess_col_i = (grad_plus - grad_minus) / (2.0 * eps)

        # Only accumulate for affected VPs (others are zero)
        for vp_j in affected:
            if vp_j > vp_i:
                continue  # lower triangle only
            key = (max(vp_i, vp_j), min(vp_i, vp_j))
            if key in pa.hess_offset_map:
                values[pa.hess_offset_map[key]] += hess_col_i[vp_j]

    return values


def compute_steiner_area_hessian_fd(pa: PartitionArrays,
                                    multipliers: np.ndarray,
                                    eps: float = 1e-5) -> np.ndarray:
    """Σ_k μ_k · ∂²(steiner_area_k)/∂λ_i∂λ_j via finite differences.

    Returns:
        (hess_nnz,) float64
    """
    hess_nnz = len(pa.hess_row)
    values = np.zeros(hess_nnz, dtype=np.float64)
    if pa.n_triple_points == 0:
        return values

    n_c = pa.n_cells - 1
    original = pa.vp_lambda.copy()
    affected = pa.tp_affected_vps

    base_jac = compute_steiner_area_jacobian(pa)  # (n_c, n_vp)

    for vp_i in affected:
        pa.vp_lambda[vp_i] = original[vp_i] + eps
        jac_plus = compute_steiner_area_jacobian(pa)
        pa.vp_lambda[vp_i] = original[vp_i]

        # ∂(jac[:, j])/∂λ_i ≈ (jac_plus[:, j] - base_jac[:, j]) / eps
        djac = (jac_plus - base_jac) / eps  # (n_c, n_vp)

        # Weight by multipliers and accumulate
        for vp_j in affected:
            if vp_j > vp_i:
                continue
            h_val = 0.0
            for c in range(n_c):
                h_val += multipliers[c] * djac[c, vp_j]
            key = (max(vp_i, vp_j), min(vp_i, vp_j))
            if key in pa.hess_offset_map:
                values[pa.hess_offset_map[key]] += h_val

    return values
```

### 4.5 Add hessian() and hessianstructure() to IPOPTProblemAdapter

```python
def hessianstructure(self) -> tuple:
    """Return Hessian sparsity pattern (lower triangle)."""
    return (self._pa.hess_row, self._pa.hess_col)

def hessian(self, x: np.ndarray, lagrange: np.ndarray,
            obj_factor: float) -> np.ndarray:
    """Compute the Hessian of the Lagrangian.

    H = obj_factor * ∇²f(x) + Σ_k lagrange[k] * ∇²c_k(x)

    Args:
        x: current VP parameters
        lagrange: constraint multipliers (n_cells-1,)
        obj_factor: scaling for the objective (σ)

    Returns:
        (hess_nnz,) float64 — lower triangle values.
    """
    self._opt._arrays.vp_lambda[:] = x
    pa = self._opt._arrays

    # Objective Hessian (perimeter)
    perim_hess = vectorized_perimeter.compute_perimeter_hessian_sparse(pa)
    steiner_perim_hess = vectorized_steiner.compute_steiner_perimeter_hessian_fd(pa)

    # Constraint Hessian (area) — multiplier-weighted
    area_hess = vectorized_area.compute_area_hessian_sparse(pa, lagrange)
    steiner_area_hess = vectorized_steiner.compute_steiner_area_hessian_fd(
        pa, lagrange)

    return (obj_factor * (perim_hess + steiner_perim_hess)
            + area_hess + steiner_area_hess)
```

### 4.6 Update optimize() to support exact Hessian

Add an `exact_hessian: bool = False` parameter to `optimize()`.

When `exact_hessian=True` and `method='ipopt'`:

1. Do NOT set `problem.add_option('hessian_approximation', 'limited-memory')`.
   IPOPT's default is to expect an exact Hessian, so simply omitting the
   `hessian_approximation` option is sufficient.
2. The adapter must have `hessian()` and `hessianstructure()` methods — IPOPT
   will call them automatically.

When `exact_hessian=False` (the default), keep the current L-BFGS behavior.

**Implementation approach**: The `IPOPTProblemAdapter` always defines
`hessian()` and `hessianstructure()` when `exact_hessian=True`. When False,
it does NOT define them (or defines them as `None`), and IPOPT falls back to
L-BFGS.

The cleanest way is to pass `exact_hessian` to the adapter's `__init__` and
conditionally define the methods:

```python
class IPOPTProblemAdapter:
    def __init__(self, optimizer, track_best=False, best_feas_tol=1e-6,
                 exact_hessian=False):
        self._opt = optimizer
        self._pa = optimizer._arrays
        self._exact_hessian = exact_hessian
        # ... existing init ...

        if not exact_hessian:
            # Remove hessian methods so IPOPT falls back to L-BFGS
            self.hessian = None
            self.hessianstructure = None
```

**IMPORTANT**: In the `optimize()` method, only set
`problem.add_option('hessian_approximation', 'limited-memory')` when
`exact_hessian=False`:

```python
if not exact_hessian:
    problem.add_option('hessian_approximation', 'limited-memory')
    problem.add_option('limited_memory_max_history', lbfgs_memory)
```

---

## 5. Integration into the Adapter and Optimizer <a id="5-integration"></a>

### 5.1 Updated IPOPTProblemAdapter (complete)

After both phases, the adapter will have these methods:

| Method | Phase 1 (current) | Phase 2 | Phase 4 |
|--------|-------------------|---------|---------|
| `objective()` | Delegates to optimizer | Unchanged | Unchanged |
| `gradient()` | Delegates to optimizer | Unchanged | Unchanged |
| `constraints()` | Delegates to optimizer | Unchanged | Unchanged |
| `jacobianstructure()` | Returns `(jac_row, jac_col)` | Unchanged | Unchanged |
| `jacobian()` | Dense → extract | **Direct sparse** | Direct sparse |
| `hessianstructure()` | Not defined | Not defined | **Returns `(hess_row, hess_col)`** |
| `hessian()` | Not defined | Not defined | **Analytical + FD** |
| `intermediate()` | Logging + best-iterate | Unchanged | Unchanged |

### 5.2 Updated PartitionArrays

New fields added:

```python
# Phase 2 additions:
nnz_lookup: np.ndarray        # int32 (n_cells-1, n_active_vp) — Jacobian offset lookup

# Phase 4 additions:
hess_row: np.ndarray          # int32 (hess_nnz,) — Hessian row indices (lower tri)
hess_col: np.ndarray          # int32 (hess_nnz,) — Hessian col indices (lower tri)
hess_offset_map: dict         # {(int, int): int} — (row, col) → flat offset
```

### 5.3 Updated optimize() signature

```python
def optimize(self, max_iter=1000, tol=1e-7, method='SLSQP',
             lbfgs_memory=6, best_iterate=False,
             exact_hessian=False) -> OptimizeResult:
```

### 5.4 CLI changes

In `testing/refine_perimeter_iterative.py`:

```python
parser.add_argument('--exact-hessian', action='store_true',
                    help='Provide IPOPT with an analytical Hessian of the '
                         'Lagrangian instead of the L-BFGS approximation. '
                         'Gives exact curvature for smoother boundaries. '
                         'Requires more computation per iteration. '
                         'Ignored for SLSQP / trust-constr.')
```

Thread it through:

```python
result = optimizer.optimize(
    max_iter=args.max_opt_iter,
    tol=args.tolerance,
    method=args.method,
    lbfgs_memory=args.lbfgs_memory,
    best_iterate=args.best_iterate,
    exact_hessian=args.exact_hessian,
)
```

---

## 6. Validation <a id="6-validation"></a>

### 6.1 Sparse Jacobian validation (Phase 2)

Run the dense and sparse Jacobian computations on the same problem and compare:

```python
# In a test script or notebook:
pa.vp_lambda[:] = x
jac_dense = vectorized_area.compute_area_jacobian(pa)
jac_sparse_vals = vectorized_area.compute_area_jacobian_sparse(pa)
steiner_dense = vectorized_steiner.compute_steiner_area_jacobian(pa)
steiner_sparse_vals = vectorized_steiner.compute_steiner_area_jacobian_sparse(pa)

# Reconstruct dense from sparse
jac_reconstructed = np.zeros_like(jac_dense)
jac_reconstructed[pa.jac_row, pa.jac_col] = jac_sparse_vals + steiner_sparse_vals

total_dense = jac_dense + steiner_dense
assert np.allclose(total_dense[pa.jac_row, pa.jac_col], jac_reconstructed[pa.jac_row, pa.jac_col], atol=1e-10)
```

### 6.2 Hessian validation (Phase 4)

Validate the analytical Hessian against finite differences:

```python
def hessian_fd(optimizer, x, lagrange, obj_factor, eps=1e-5):
    """Full Lagrangian Hessian via finite differences on the gradient."""
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        x_plus = x.copy(); x_plus[i] += eps
        x_minus = x.copy(); x_minus[i] -= eps

        # Lagrangian gradient = obj_factor * ∇f + Σ μ_k * ∇c_k
        grad_plus = (obj_factor * optimizer.objective_gradient(x_plus)
                     + lagrange @ optimizer.constraint_area_jacobian(x_plus))
        grad_minus = (obj_factor * optimizer.objective_gradient(x_minus)
                      + lagrange @ optimizer.constraint_area_jacobian(x_minus))
        H[i, :] = (grad_plus - grad_minus) / (2 * eps)

    return 0.5 * (H + H.T)  # symmetrize
```

Compare the analytical Hessian values against this reference. Agreement within
`1e-4` relative tolerance is expected (the Steiner FD contribution introduces
some noise).

### 6.3 End-to-end validation

1. Run IPOPT with L-BFGS on the 5-cell reference problem. Record final perimeter.
2. Run IPOPT with exact Hessian on the same problem. Record final perimeter.
3. The exact Hessian run should achieve:
   - Equal or lower perimeter
   - Smoother boundaries (visual inspection)
   - Fewer total iterations to converge
   - Fewer restoration phases

### 6.4 Performance check

The exact Hessian adds computation per iteration. Verify the trade-off:

- If Hessian computation takes < 0.5s per iteration and convergence is 2–3×
  faster (fewer iterations), the exact Hessian is a net win.
- If Hessian computation is expensive (> 1s), consider the sparse
  vectorization optimizations described in Section 4.2.

---

## 7. Correctness Checklist <a id="7-checklist"></a>

### Phase 2 (Sparse Jacobian)

- [ ] `compute_area_jacobian_sparse()` returns `(nnz,)` array matching
      `jac_row`/`jac_col` order exactly
- [ ] Values match `compute_area_jacobian_analytical()` dense output when
      reconstructed
- [ ] `compute_steiner_area_jacobian_sparse()` output matches dense version
- [ ] `nnz_lookup` correctly maps all `(row, col)` pairs — no `-1` for
      valid entries
- [ ] `vp_lambda` is set before calling sparse functions in `jacobian()`
- [ ] `np.add.at` is used (not `+=`) for accumulation to handle duplicate
      VP appearances in multiple triangles
- [ ] No dense `(n_cells-1, n_active_vp)` matrix is allocated anywhere in
      the sparse path
- [ ] 50 MB warning in `jacobian()` is removed or made conditional on Phase 1

### Phase 4 (Exact Hessian)

- [ ] IPOPT expects **lower triangle** — all `(row, col)` pairs satisfy
      `row >= col`
- [ ] `hessian()` signature matches cyipopt: `(x, lagrange, obj_factor)`
- [ ] `obj_factor` multiplies the objective Hessian; `lagrange[k]` multiplies
      constraint k's Hessian
- [ ] Hessian is returned as `(hess_nnz,)` flat array, NOT a matrix
- [ ] `hessian_approximation='limited-memory'` is NOT set when
      `exact_hessian=True`
- [ ] Perimeter Hessian uses `r = max(r, 1e-12)` to avoid division by zero
- [ ] Area Hessian uses `normC = max(normC, 1e-30)` for degenerate triangles
- [ ] Steiner Hessian FD uses central differences for accuracy
- [ ] Steiner FD epsilon is large enough to avoid numerical noise (1e-5 or
      1e-6, not 1e-7)
- [ ] `hessianstructure()` and `hessian()` are only defined on the adapter
      when `exact_hessian=True`
- [ ] The adapter does NOT define `hessian()`/`hessianstructure()` when
      `exact_hessian=False` — IPOPT must fall back to L-BFGS
- [ ] `--exact-hessian` flag is added to CLI and threaded to `optimize()`
- [ ] Analytical Hessian validated against FD Hessian to within 1e-4 relative
      tolerance on the reference problem before deployment

### General

- [ ] All new functions have docstrings with return shape specification
- [ ] No new files created — all additions are in existing modules
- [ ] `PartitionArrays` changes are reflected in both the dataclass definition
      AND the `return PartitionArrays(...)` call in `compile_arrays()`
- [ ] Existing SLSQP / trust-constr paths are not affected by any changes
- [ ] The dense Jacobian path (Phase 1) still works when `nnz_lookup` is None
      (backward compatibility)
