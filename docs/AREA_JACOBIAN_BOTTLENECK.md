# Area Jacobian Bottleneck: Diagnosis and Proposed Fix

## Problem Statement

The vectorized evaluation implementation (Steps 1–11 of the optimization plan)
was expected to produce a **massive** speedup in the SLSQP perimeter optimization
loop. In practice, the improvement is only **~8–15%** (24 s/iter vs 28 s/iter on the
46,080-vertex torus, 5 partitions, 1,977 variable points).

| Metric | Old (non-vectorized) | New (vectorized) |
|---|---|---|
| Steady-state per iteration (iter 50–100) | ~26.2 s | ~24.3 s |
| Later iterations (300–400) | ~27.7 s | ~25.6 s |
| Late iterations (450–470) | — | ~31.3 s (degrading) |
| Total (534 iters) | 14,977 s | projected ~13,500 s |

The vectorized code also shows **progressive slowdown** in later iterations
(24 s → 31 s), likely from memory allocation pressure.

Source logs:

- Old: `logs/ring_partition_20260319_225821.log` (534 iterations, 14,977 s)
- New: `logs/ring_partition_20260323_174137.log` (470+ iterations, still running)


## Root Cause: The Area Jacobian Finite-Difference Loop

### What the original plan assumed

The vectorized evaluation plan (`docs/VECTORIZED_EVALUATION_OPTIMIZATION.md`)
identified the bottleneck as **Python interpreter overhead** in per-element loops
over dataclass objects, and estimated that BFGS Steiner solves dominated the
cost. The plan vectorized perimeter, area, and Steiner computations, and
replaced BFGS with an analytical Fermat–Torricelli formula.

### What actually dominates

Profiling the two runs reveals that **BFGS Steiner was only ~4 s/iter** of the
~28 s/iter total. The remaining **~24 s/iter is the area constraint Jacobian**,
and this cost is almost identical in both the old and new code.

| Component | Old | New | Speedup |
|---|---|---|---|
| SLSQP Fortran internals | ~0.5 s | ~0.5 s | 1× |
| Perimeter (objective) | ~0.5 s | < 0.01 s | 50×+ |
| Perimeter gradient | ~0.5 s | < 0.01 s | 50×+ |
| Area constraints | ~1 s | < 0.01 s | 100×+ |
| **Area Jacobian (FD loop)** | **~21 s** | **~20 s** | **~1×** |
| BFGS Steiner (all calls) | ~4 s | 0 s | ∞ |
| Callback overhead | ~1.5 s | ~0 s | ∞ |
| **Total** | **~28 s** | **~24 s** | **1.17×** |

The area Jacobian is untouched by vectorization because it still uses a
**Python for-loop** over all variable points.


## The Redundancy Problem

### How the new (vectorized) area Jacobian works

```
vectorized_area.compute_area_jacobian(pa):

    base_areas = compute_cell_areas(pa)          ← 1 full evaluation (3,930 btri)

    for vp_idx in area_affected_vps:             ← ~1,977 Python iterations
        perturb pa.vp_lambda[vp_idx]
        perturbed = compute_cell_areas(pa)       ← 1 full evaluation (3,930 btri) EACH
        restore pa.vp_lambda[vp_idx]
        jacobian[:, vp_idx] = (perturbed - base) / eps
```

Each of the ~1,977 calls to `compute_cell_areas` recomputes **all** 3,930
boundary triangle areas, even though perturbing a single λ only affects the
**2–4 boundary triangles** that use that VP. This means:

- **Triangle area evaluations per Jacobian call**: 1,977 × 3,930 = **7,769,610**
- **Triangle area evaluations actually needed**: 1,977 × ~4 = **~7,908**
- **Redundancy factor**: **~982×**

Additionally, each `compute_cell_areas` call invokes `_compute_vp_positions(pa)`,
which allocates a new `(1977, 3)` array and recomputes **all** 1,977 VP positions
from scratch. Over 1,977 FD iterations, this produces **~1 GB of temporary array
allocations** per Jacobian call, explaining the progressive slowdown from GC/
allocator pressure.


### How the old (per-triangle) area Jacobian worked

The original `AreaCalculator` computed the Jacobian with **zero redundancy**:

```
AreaCalculator.compute_area_jacobian(lambda_vec):

    for cell_idx in range(n_cells - 1):          ← 4 cells
        for tri_idx in boundary_triangles[cell]:  ← ~786 triangles per cell
            area, gradient = _triangle_contribution(tri_idx, cell_idx, ...)
            jacobian[cell_idx] += gradient
```

Inside `_triangle_contribution`, the gradient is computed via finite differences
**on only the 2 VPs that appear in that specific triangle**:

```
_partial_area_two_inside(...):

    area = quadrilateral_area(p_in1, p_cut1, p_cut2, p_in2)

    # Perturb ONLY vp1 (1 scalar cross product)
    p_cut1_perturbed = ...
    gradient[vp_idx1] = (area_perturbed1 - area) / eps

    # Perturb ONLY vp2 (1 scalar cross product)
    p_cut2_perturbed = ...
    gradient[vp_idx2] = (area_perturbed2 - area) / eps
```

- **Triangle area evaluations per Jacobian call**: 4 cells × 786 triangles × 2 FD perturbations = **6,288**
- **Redundancy factor**: **1× (zero redundant work)**

The old approach was efficient in terms of FLOPs but slow in wall time due to
Python-loop overhead: ~3,144 Python function calls with dict lookups, attribute
access, tuple creation, and allocation of `np.zeros(1977)` gradient arrays that
are mostly empty.


### Summary of the paradox

| Approach | Triangle evals | Per-eval cost | Total cost |
|---|---|---|---|
| Old (per-triangle FD) | 6,288 | ~3 ms (Python overhead) | ~20 s |
| New (per-VP vectorized FD) | 7,769,610 | ~0.003 ms (vectorized) | ~20 s |

The new code does **1,000× more arithmetic** but each operation is **1,000×
faster** due to vectorization. The two effects cancel out, producing
identical wall time.


## Proposed Solutions

### Solution A: Sparse Finite Differences (vectorized old approach)

Adapt the old per-triangle FD technique to work on the flat `PartitionArrays`,
eliminating Python-loop overhead while preserving zero-redundancy FD.

**Concept**: Instead of looping over VPs and recomputing all triangles, loop over
boundary triangles and compute the 2 VP gradients per triangle in batch.

**Required new arrays in `PartitionArrays`** (computed once in `compile_arrays`):

```python
# For each boundary triangle row, store the edge direction vectors
# dp_cut/dlambda = vertices[edge[0]] - vertices[edge[1]]
btri_d1: np.ndarray    # float64 (n_btri, dim) — edge direction for VP1
btri_d2: np.ndarray    # float64 (n_btri, dim) — edge direction for VP2
```

These are constant (they depend on mesh geometry, not on λ) and can be
pre-computed once.

**Implementation** (`vectorized_area.py`):

```python
def compute_area_jacobian_sparse_fd(pa, eps=1e-7):
    """Area Jacobian via per-triangle sparse FD on flat arrays.

    For each boundary triangle, perturb only the 2 affected VPs and
    compute the area change.  Scatter-add into the Jacobian.

    Cost: O(n_btri) — one pass over boundary triangles, no VP loop.
    """
    n_c = pa.n_cells - 1
    pos = _compute_vp_positions(pa)
    jacobian = np.zeros((n_c, pa.n_active_vp))

    # --- 1-inside triangles ---
    m1 = pa.btri_n_inside == 1
    if np.any(m1):
        p_in  = pa.vertices[pa.btri_v_in[m1, 0]]
        pc1   = pos[pa.btri_vp1[m1]]
        pc2   = pos[pa.btri_vp2[m1]]
        d1    = pa.btri_d1[m1]        # pre-computed edge direction for VP1
        d2    = pa.btri_d2[m1]        # pre-computed edge direction for VP2
        cells = pa.btri_cell[m1]
        vp1   = pa.btri_vp1[m1]
        vp2   = pa.btri_vp2[m1]

        base_area = _triangle_areas_batch(p_in, pc1, pc2)

        # Perturb VP1: p_cut1 → p_cut1 + eps * d1
        area_p1 = _triangle_areas_batch(p_in, pc1 + eps * d1, pc2)
        dA_dlam1 = (area_p1 - base_area) / eps

        # Perturb VP2: p_cut2 → p_cut2 + eps * d2
        area_p2 = _triangle_areas_batch(p_in, pc1, pc2 + eps * d2)
        dA_dlam2 = (area_p2 - base_area) / eps

        # Scatter-add into Jacobian (only cells 0..n_c-1)
        valid = cells < n_c
        for c in range(n_c):
            cmask = valid & (cells == c)
            np.add.at(jacobian[c], vp1[cmask], dA_dlam1[cmask])
            np.add.at(jacobian[c], vp2[cmask], dA_dlam2[cmask])

    # --- 2-inside triangles (similar, with quadrilateral split) ---
    m2 = pa.btri_n_inside == 2
    if np.any(m2):
        p_in1 = pa.vertices[pa.btri_v_in[m2, 0]]
        p_in2 = pa.vertices[pa.btri_v_in[m2, 1]]
        pc1   = pos[pa.btri_vp1[m2]]
        pc2   = pos[pa.btri_vp2[m2]]
        d1    = pa.btri_d1[m2]
        d2    = pa.btri_d2[m2]
        cells = pa.btri_cell[m2]
        vp1   = pa.btri_vp1[m2]
        vp2   = pa.btri_vp2[m2]

        base_area = (_triangle_areas_batch(p_in1, pc1, pc2)
                   + _triangle_areas_batch(p_in1, pc2, p_in2))

        # Perturb VP1
        area_p1 = (_triangle_areas_batch(p_in1, pc1 + eps * d1, pc2)
                 + _triangle_areas_batch(p_in1, pc2, p_in2))
        dA_dlam1 = (area_p1 - base_area) / eps

        # Perturb VP2
        area_p2 = (_triangle_areas_batch(p_in1, pc1, pc2 + eps * d2)
                 + _triangle_areas_batch(p_in1, pc2 + eps * d2, p_in2))
        dA_dlam2 = (area_p2 - base_area) / eps

        valid = cells < n_c
        for c in range(n_c):
            cmask = valid & (cells == c)
            np.add.at(jacobian[c], vp1[cmask], dA_dlam1[cmask])
            np.add.at(jacobian[c], vp2[cmask], dA_dlam2[cmask])

    return jacobian
```

**Complexity**: O(n_btri) — one vectorized pass over ~3,930 boundary triangle
rows. No Python for-loop over VPs. Each triangle computes exactly 2 FD
perturbations (matching the old code), but the perturbation is a simple
position shift `p_cut + eps * d`, not a full recomputation of all VP positions.

**Estimated time**: ~1–3 ms (3 vectorized `_triangle_areas_batch` calls per
triangle type, each processing ~2,000 rows).


### Solution B: Analytical Area Jacobian (recommended)

Eliminate finite differences entirely by deriving ∂Area/∂λ analytically via
the chain rule on the cross-product area formula.

**Mathematical derivation for 3D triangles:**

The area of a triangle with vertices `(p₁, p₂, p₃)` in 3D is:

\[
  A = \tfrac{1}{2}\,\lVert\mathbf{u}\times\mathbf{v}\rVert,
  \qquad \mathbf{u} = p_2 - p_1,\quad \mathbf{v} = p_3 - p_1
\]

The unit normal is:

\[
  \hat{\mathbf{n}} = \frac{\mathbf{u}\times\mathbf{v}}
                          {\lVert\mathbf{u}\times\mathbf{v}\rVert}
\]

If `p₂` depends on parameter λ through `p₂ = λ·vₐ + (1−λ)·v_b`, then
`dp₂/dλ = vₐ − v_b ≡ d` (the edge direction, constant). Only **u** depends
on λ:

\[
  \frac{\partial A}{\partial\lambda}
    = \frac{1}{2}\,\frac{(\mathbf{u}\times\mathbf{v})\cdot
      (\mathbf{d}\times\mathbf{v})}
      {\lVert\mathbf{u}\times\mathbf{v}\rVert}
    = \frac{1}{2}\,\hat{\mathbf{n}}\cdot(\mathbf{d}\times\mathbf{v})
\]

This is a **scalar triple product**, computable in 9 multiplications and 5
additions per triangle.

**Case 1 — One vertex inside (triangle region):**

The cell's area contribution from this boundary triangle is
`A = area(p_in, p_cut1, p_cut2)`.

Let `u = p_cut1 − p_in`, `v = p_cut2 − p_in`, `n̂ = (u×v)/‖u×v‖`.

- `d₁ = vertices[edge1[0]] − vertices[edge1[1]]` (edge direction for VP1)
- `d₂ = vertices[edge2[0]] − vertices[edge2[1]]` (edge direction for VP2)

\[
  \frac{\partial A}{\partial\lambda_1}
    = \tfrac{1}{2}\,\hat{\mathbf{n}}\cdot(\mathbf{d}_1\times\mathbf{v})
  ,\qquad
  \frac{\partial A}{\partial\lambda_2}
    = \tfrac{1}{2}\,\hat{\mathbf{n}}\cdot(\mathbf{u}\times\mathbf{d}_2)
\]

**Case 2 — Two vertices inside (quadrilateral region):**

The cell's area is the sum of two triangles:

- `A₁ = area(p_in1, p_cut1, p_cut2)` with `u₁ = p_cut1 − p_in1`, `v₁ = p_cut2 − p_in1`
- `A₂ = area(p_in1, p_cut2, p_in2)` with `u₂ = p_cut2 − p_in1`, `v₂ = p_in2 − p_in1`

For VP1 (affects only p_cut1, which appears only in A₁):

\[
  \frac{\partial A}{\partial\lambda_1}
    = \tfrac{1}{2}\,\hat{\mathbf{n}}_1\cdot(\mathbf{d}_1\times\mathbf{v}_1)
\]

For VP2 (affects p_cut2, which appears in both A₁ and A₂):

\[
  \frac{\partial A}{\partial\lambda_2}
    = \tfrac{1}{2}\,\hat{\mathbf{n}}_1\cdot(\mathbf{u}_1\times\mathbf{d}_2)
    + \tfrac{1}{2}\,\hat{\mathbf{n}}_2\cdot(\mathbf{d}_2\times\mathbf{v}_2)
\]

**2D specialization:**

In 2D, the signed area is `A_signed = ½(u_x·v_y − u_y·v_x)`, and the
derivatives simplify to:

\[
  \frac{\partial A}{\partial\lambda_1}
    = \operatorname{sign}(A_{\text{signed}})\;\tfrac{1}{2}
      (d_{1x}\,v_y - d_{1y}\,v_x)
\]

\[
  \frac{\partial A}{\partial\lambda_2}
    = \operatorname{sign}(A_{\text{signed}})\;\tfrac{1}{2}
      (u_x\,d_{2y} - u_y\,d_{2x})
\]


**Implementation** (`vectorized_area.py`):

```python
def compute_area_jacobian_analytical(pa):
    """Area Jacobian via analytical chain-rule derivatives.

    No finite differences.  One vectorized pass over boundary triangles.

    Cost: O(n_btri) — same as compute_cell_areas itself.
    """
    n_c = pa.n_cells - 1
    pos = _compute_vp_positions(pa)
    jacobian = np.zeros((n_c, pa.n_active_vp))

    dim = pa.vertices.shape[1]

    # --- 1-inside triangles: A = area(p_in, p_cut1, p_cut2) ---
    m1 = pa.btri_n_inside == 1
    if np.any(m1):
        p_in = pa.vertices[pa.btri_v_in[m1, 0]]
        pc1  = pos[pa.btri_vp1[m1]]
        pc2  = pos[pa.btri_vp2[m1]]

        u = pc1 - p_in                               # (K, dim)
        v = pc2 - p_in                               # (K, dim)

        # Edge directions (pre-computable, but cheap to compute here)
        d1 = (pa.vertices[pa.vp_edge_v1[pa.btri_vp1[m1]]]
            - pa.vertices[pa.vp_edge_v2[pa.btri_vp1[m1]]])
        d2 = (pa.vertices[pa.vp_edge_v1[pa.btri_vp2[m1]]]
            - pa.vertices[pa.vp_edge_v2[pa.btri_vp2[m1]]])

        if dim == 3:
            cross_uv = np.cross(u, v)                # (K, 3)
            norm_uv = np.linalg.norm(cross_uv, axis=1, keepdims=True)
            norm_uv = np.maximum(norm_uv, 1e-30)
            n_hat = cross_uv / norm_uv               # unit normal

            # dA/dlam1 = 0.5 * n_hat . (d1 x v)
            dA_dl1 = 0.5 * np.sum(n_hat * np.cross(d1, v), axis=1)
            # dA/dlam2 = 0.5 * n_hat . (u x d2)
            dA_dl2 = 0.5 * np.sum(n_hat * np.cross(u, d2), axis=1)
        else:
            # 2D: signed area derivative
            signed_area = 0.5 * (u[:, 0]*v[:, 1] - u[:, 1]*v[:, 0])
            sign = np.sign(signed_area)
            sign[sign == 0] = 1.0
            dA_dl1 = sign * 0.5 * (d1[:, 0]*v[:, 1] - d1[:, 1]*v[:, 0])
            dA_dl2 = sign * 0.5 * (u[:, 0]*d2[:, 1] - u[:, 1]*d2[:, 0])

        cells = pa.btri_cell[m1]
        vp1   = pa.btri_vp1[m1]
        vp2   = pa.btri_vp2[m1]
        for c in range(n_c):
            cmask = cells == c
            np.add.at(jacobian[c], vp1[cmask], dA_dl1[cmask])
            np.add.at(jacobian[c], vp2[cmask], dA_dl2[cmask])

    # --- 2-inside triangles: A = area(p_in1,pc1,pc2) + area(p_in1,pc2,p_in2) ---
    m2 = pa.btri_n_inside == 2
    if np.any(m2):
        p_in1 = pa.vertices[pa.btri_v_in[m2, 0]]
        p_in2 = pa.vertices[pa.btri_v_in[m2, 1]]
        pc1   = pos[pa.btri_vp1[m2]]
        pc2   = pos[pa.btri_vp2[m2]]

        # Triangle 1: (p_in1, pc1, pc2)
        u1 = pc1 - p_in1
        v1 = pc2 - p_in1
        # Triangle 2: (p_in1, pc2, p_in2)
        u2 = pc2 - p_in1
        v2 = p_in2 - p_in1

        d1 = (pa.vertices[pa.vp_edge_v1[pa.btri_vp1[m2]]]
            - pa.vertices[pa.vp_edge_v2[pa.btri_vp1[m2]]])
        d2 = (pa.vertices[pa.vp_edge_v1[pa.btri_vp2[m2]]]
            - pa.vertices[pa.vp_edge_v2[pa.btri_vp2[m2]]])

        if dim == 3:
            cross1 = np.cross(u1, v1)
            norm1 = np.maximum(np.linalg.norm(cross1, axis=1, keepdims=True), 1e-30)
            n1 = cross1 / norm1

            cross2 = np.cross(u2, v2)
            norm2 = np.maximum(np.linalg.norm(cross2, axis=1, keepdims=True), 1e-30)
            n2 = cross2 / norm2

            # VP1 only affects triangle 1 (through pc1)
            dA_dl1 = 0.5 * np.sum(n1 * np.cross(d1, v1), axis=1)
            # VP2 affects triangle 1 (through pc2 in v1 position)
            #   and triangle 2 (through pc2 in u2 position)
            dA_dl2 = (0.5 * np.sum(n1 * np.cross(u1, d2), axis=1)
                    + 0.5 * np.sum(n2 * np.cross(d2, v2), axis=1))
        else:
            sa1 = 0.5 * (u1[:, 0]*v1[:, 1] - u1[:, 1]*v1[:, 0])
            s1 = np.sign(sa1); s1[s1 == 0] = 1.0
            sa2 = 0.5 * (u2[:, 0]*v2[:, 1] - u2[:, 1]*v2[:, 0])
            s2 = np.sign(sa2); s2[s2 == 0] = 1.0

            dA_dl1 = s1 * 0.5 * (d1[:, 0]*v1[:, 1] - d1[:, 1]*v1[:, 0])
            dA_dl2 = (s1 * 0.5 * (u1[:, 0]*d2[:, 1] - u1[:, 1]*d2[:, 0])
                    + s2 * 0.5 * (d2[:, 0]*v2[:, 1] - d2[:, 1]*v2[:, 0]))

        cells = pa.btri_cell[m2]
        vp1   = pa.btri_vp1[m2]
        vp2   = pa.btri_vp2[m2]
        for c in range(n_c):
            cmask = cells == c
            np.add.at(jacobian[c], vp1[cmask], dA_dl1[cmask])
            np.add.at(jacobian[c], vp2[cmask], dA_dl2[cmask])

    return jacobian
```

**Complexity**: O(n_btri) — one vectorized pass. No FD, no Python loop over VPs.
All operations are batch NumPy on arrays of length ~2,000. Estimated time:
**< 1 ms**.


## Comparison of All Three Approaches

| Approach | Triangle evals | Python loops | Memory allocs | Estimated time |
|---|---|---|---|---|
| Old per-triangle FD | 6,288 (scalar) | 3,144 calls | 3,144 × `(1977,)` arrays | ~20 s |
| Current per-VP vectorized FD | 7,769,610 (batched) | 1,977 iters | 1,977 × `(1977,3)` arrays (~1 GB) | ~20 s |
| **Sparse FD (Solution A)** | ~11,790 (batched) | 0 | 3 batch arrays | **~2 ms** |
| **Analytical (Solution B)** | 0 (pure derivatives) | 0 | 3 batch arrays | **< 1 ms** |


## Recommendation

**Implement Solution B (analytical Jacobian)** as the primary path. It is:

1. **Faster**: No FD perturbations at all — pure arithmetic on the same arrays
   already used by `compute_cell_areas`.
2. **More accurate**: Eliminates the ε-dependent FD truncation error. The
   perimeter gradient is already analytical; making the area Jacobian
   analytical too removes FD as a source of numerical noise.
3. **No new data structures needed**: The edge directions `d₁, d₂` can be
   computed on-the-fly from `pa.vp_edge_v1`, `pa.vp_edge_v2`, and
   `pa.vertices` (same data already in `PartitionArrays`).
4. **Eliminates memory pressure**: No temporary array allocation in a loop,
   fixing the progressive slowdown observed in later iterations.

Solution A (sparse FD) is a simpler, lower-risk fallback that can be
implemented first as a validation target: the analytical Jacobian should match
sparse FD to ~1e-6 tolerance.


## Expected Impact

With the area Jacobian reduced from ~20 s to < 1 ms, the per-iteration cost
becomes:

| Component | Current | After fix |
|---|---|---|
| SLSQP internals | ~0.5 s | ~0.5 s |
| Perimeter + gradient | < 0.01 s | < 0.01 s |
| Area constraints | < 0.01 s | < 0.01 s |
| **Area Jacobian** | **~20 s** | **< 0.001 s** |
| Steiner (all) | < 0.02 s | < 0.02 s |
| **Total per iteration** | **~24 s** | **~0.5 s** |
| **534-iteration run** | **~3.6 hours** | **~4.5 minutes** |

This represents a **~48× end-to-end speedup**, reducing the 5-partition torus
optimization from hours to minutes.


## Files to Modify

| File | Change |
|---|---|
| `src/core/vectorized_area.py` | Replace `compute_area_jacobian` with analytical version |
| `src/core/perimeter_optimizer.py` | No change needed (already calls `vectorized_area.compute_area_jacobian`) |
| `src/core/partition_arrays.py` | No change needed (all required data already present) |
| `testing/test_vectorized_evaluation.py` | Add test: analytical Jacobian matches sparse FD to tolerance |

## Validation Strategy

1. **Sparse FD as reference**: Implement `compute_area_jacobian_sparse_fd` as a
   test utility. This has zero redundancy so it should match the old
   `AreaCalculator.compute_area_jacobian()` to machine precision.

2. **Analytical vs sparse FD**: Compare `compute_area_jacobian_analytical`
   against `compute_area_jacobian_sparse_fd` at the same λ vector. Tolerance:
   ~1e-6 (limited by FD truncation error at ε = 1e-7).

3. **End-to-end**: Run the same 534-iteration optimization with the analytical
   Jacobian and compare final perimeter / constraint violations against the
   original run.
