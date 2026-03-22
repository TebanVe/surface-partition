# Vectorized Evaluation for Perimeter Optimization

## Problem Statement

The perimeter refinement phase (Phase 2) is the dominant computational bottleneck. A single
SLSQP optimization run on a torus mesh with 46,080 vertices, 5 cells, and ~1,982 variable
points takes **5+ hours** (18,432 seconds for 653 iterations / 1,309 function evaluations).
With 4 iterative refinement cycles, total runtime exceeds **20 hours**.

The root cause is **not** algorithmic complexity — the total arithmetic work per function
evaluation is ~680,000 floating-point operations (trivial). The problem is that these
operations are executed through **Python for-loops over dataclass objects** with per-element
method calls, dict lookups, and NumPy micro-allocations. The Python interpreter overhead
dominates by a factor of ~20,000x.

## Profiling Evidence

From `logs/ring_partition_20260321_001727.log`:

```
00:20:10 - Starting perimeter optimization with method=SLSQP
00:20:10 -   Optimizing 1982 active VPs (skipping 192 inactive)
00:28:47 - Iteration 10: Perimeter=37.492846   (~52s per iter in first batch)
05:27:28 - Optimization completed in 18431.81s
05:27:28 -   Iterations: 653
05:27:28 -   Function evaluations: 1309
```

Key metrics:
- ~14 seconds per function evaluation
- ~28 seconds per SLSQP iteration (includes gradient + Jacobian)
- 1,982 active variable points, ~1,994 boundary segments, ~3,940 boundary triangles
- Migration phase (65 migrations): 4 seconds total (0.02% of runtime)
- FEM assembly: 130 seconds (0.7% of runtime)
- SLSQP optimization: 18,432 seconds (99.9% of runtime)

## Architecture: Why Two Representations Are Needed

The codebase has two fundamentally different computational modes:

### Mode A — Mutation (migrations, topology switches)
- Irregular, graph-based, relational operations
- Small number of operations (65 migrations in 4 seconds)
- Correctness-critical, needs debuggability
- **Current object-oriented structures are correct and must not change**

### Mode B — Evaluation (objective, gradient, constraints during SLSQP)
- Regular, arithmetic, embarrassingly parallel
- Massive repetition (1,309 function evaluations × ~2,000 segments each)
- Performance-critical
- **Needs vectorized flat arrays**

The solution is a **dual representation**: keep objects for mutation, add flat arrays for
evaluation, synchronize at the boundary between modes.

```
┌─────────────────────────────────────────────────────────────┐
│  Mode A: Mutation (objects + dicts) — NO CHANGES            │
│  Used by: MigrationExecutor, OneRingRebuilder,              │
│           MigrationDetector, MigrationOrchestrator,         │
│           TopologySwitcher, SteinerHandler (mutation)        │
│  Runtime: seconds                                           │
└──────────────────────┬──────────────────────────────────────┘
                       │ partition.compile_arrays()
                       │ (called once before optimization)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Mode B: Evaluation (flat NumPy arrays) — NEW               │
│  Used by: vectorized_perimeter, vectorized_area             │
│  Runtime: milliseconds per evaluation                       │
└─────────────────────────────────────────────────────────────┘
```

## Critical Semantic Conventions to Preserve

The vectorized code must reproduce the **exact same numerical values** as the current
implementation. The following conventions are non-negotiable:

### 1. Perimeter is the sum of all cell perimeters (double-counted)

The current `PerimeterCalculator.compute_total_perimeter()` sums per-cell perimeters:

```python
for cell_idx in range(self.partition.n_cells):
    total += self.compute_cell_perimeter(cell_idx, lambda_vec)
```

Each `BoundarySegment` has a `cell_pair = (cell_a, cell_b)`. When summing over all cells,
each segment is counted **twice** — once for `cell_a`, once for `cell_b`. This is by design.
The vectorized total perimeter must therefore be `2 * sum(segment_lengths)`.

The same applies to the gradient: `compute_total_perimeter_gradient()` sums per-cell
gradients, so each segment's `∂ℓ/∂λ` is accumulated twice. The vectorized gradient must
also be `2 * sum(per_segment_gradients)`.

### 2. Triple-point triangles are excluded from AreaCalculator

In `AreaCalculator._categorize_triangles()`:

```python
if len(set(labels)) == 3:
    continue  # Skip - handled by SteinerHandler
```

The area of triple-point mesh triangles is computed **entirely** by `SteinerHandler`, which
decomposes each triple-point triangle into:
- **Corner area** per cell: triangle (mesh_vertex, VP_1, VP_2) for that cell's vertex
- **Void area** per cell: triangle (VP_1, VP_2, steiner_point) for that cell's VP pair

These are then added in `PerimeterOptimizer.constraint_area_equality()`:

```python
areas = self.area_calc.compute_all_cell_areas(full_vec)
steiner_areas = self.steiner_handler.get_total_area_contribution()
for cell_idx, area_contrib in steiner_areas.items():
    areas[cell_idx] += area_contrib
```

The vectorized `compile_arrays()` must **exclude triple-point triangles** from the boundary
triangle arrays, replicating the same `len(set(labels)) == 3` guard.

### 3. Steiner perimeter uses the same per-cell-sum convention

`SteinerHandler.get_total_perimeter_contribution()` calls `sum(contrib.values())` where
`contrib` maps each cell to its NET Steiner correction:

```python
contributions[cell_idx] = steiner_edge1 + steiner_edge2 - original_edge
```

This sums over all 3 cells per triple point. Each Steiner edge appears in 2 cells'
contributions. This is consistent with the double-counted regular perimeter convention.

### 4. Lambda convention

`VariablePoint.evaluate()` computes: `x = λ * vertices[edge[0]] + (1-λ) * vertices[edge[1]]`
with `edge[0] < edge[1]` (normalized). The vectorized position computation must use the
same formula.

### 5. SteinerHandler reads from PartitionContour VP objects

During optimization, `SteinerHandler` reads VP positions via
`partition.evaluate_variable_point(vi)`, which reads `variable_points[vi].lambda_param`.
The optimizer must call `partition.set_variable_vector()` before every SteinerHandler call
to keep the VP objects in sync with the current optimization vector.

## What to Implement

### Step 1: PartitionArrays dataclass

Create `src/core/partition_arrays.py` with a frozen array snapshot of the partition state.

```python
@dataclass
class PartitionArrays:
    """Flat array representation of partition state for vectorized evaluation."""

    # Variable point arrays (shape: n_active_vp)
    vp_edge_v1: np.ndarray       # int32 — first vertex index of edge (edge[0])
    vp_edge_v2: np.ndarray       # int32 — second vertex index of edge (edge[1])
    vp_lambda: np.ndarray        # float64 — lambda parameters (mutable during optim)

    # Boundary segment arrays (shape: n_segments)
    seg_vp1: np.ndarray          # int32 — first VP index (into active VP arrays)
    seg_vp2: np.ndarray          # int32 — second VP index (into active VP arrays)
    seg_cell_a: np.ndarray       # int32 — first cell of the pair
    seg_cell_b: np.ndarray       # int32 — second cell of the pair

    # Boundary triangle arrays (shape: n_boundary_triangles)
    # EXCLUDES triple-point triangles (handled by SteinerHandler)
    # One row per (triangle, cell) pair where cell has 1 or 2 vertices inside
    btri_idx: np.ndarray         # int32 — original triangle index in mesh
    btri_cell: np.ndarray        # int32 — cell this boundary triangle contributes to
    btri_n_inside: np.ndarray    # int32 — number of vertices inside cell (1 or 2)
    btri_v_in: np.ndarray        # int32, shape (n_btri, 2) — inside vertex indices (padded)
    btri_v_out: np.ndarray       # int32, shape (n_btri, 2) — outside vertex indices (padded)
    btri_vp1: np.ndarray         # int32 — VP index on first cut edge
    btri_vp2: np.ndarray         # int32 — VP index on second cut edge

    # Pre-computed constants
    cell_interior_area: np.ndarray  # float64, shape (n_cells,) — constant interior area per cell
    n_cells: int
    n_active_vp: int

    # Index mapping: active VP index -> absolute VP index (for syncing back)
    active_to_absolute: np.ndarray  # int32, shape (n_active_vp,)

    # Set of VP indices that affect boundary triangle areas (for sparse Jacobian)
    area_affected_vps: np.ndarray   # int32 — unique VPs in btri_vp1 ∪ btri_vp2

    # Mesh vertex coordinates (reference, not copied)
    vertices: np.ndarray         # float64, shape (N, 2 or 3) — mesh vertices
```

### Step 2: compile_arrays() method on PartitionContour

Add a method to `PartitionContour` that walks the object structures once and fills the flat
arrays. This is called once before optimization starts.

Location: `src/core/contour_partition.py`, add method to class `PartitionContour`.

The method must:

1. Build a mapping from absolute VP index to active VP index (skip inactive VPs).
   Store the reverse mapping (`active_to_absolute`) for syncing back after optimization.
2. Fill `vp_edge_v1`, `vp_edge_v2`, `vp_lambda` from `self.variable_points` (active only).
   `vp_edge_v1[i] = vp.edge[0]`, `vp_edge_v2[i] = vp.edge[1]`.
3. Fill `seg_vp1`, `seg_vp2`, `seg_cell_a`, `seg_cell_b` from `self.boundary_segments`,
   remapping VP indices through the absolute-to-active index map.
4. Fill boundary triangle arrays by iterating `self.mesh.faces`, classifying each triangle
   by `self.vertex_labels` (same logic as `AreaCalculator._categorize_triangles()`):
   - **Skip triple-point triangles** where `len(set(labels)) == 3`.
   - For each non-triple boundary triangle with `n_inside = 1 or 2` vertices in a given
     cell: record the triangle index, cell index, n_inside, inside/outside vertex indices,
     and the VP indices on the two cut edges (looked up via `self.edge_to_varpoint`).
   - Each boundary triangle may generate multiple rows (one per cell it contributes to).
5. Fill `cell_interior_area` by summing `mesh.triangle_areas[tri_idx]` for fully interior
   triangles (where all 3 vertices belong to the same cell).
6. Compute `area_affected_vps` as the unique union of all VP indices in `btri_vp1`
   and `btri_vp2`.
7. Return a `PartitionArrays` instance.

**Critical detail for boundary triangle arrays**: For each boundary triangle row, the two
VP indices (`btri_vp1`, `btri_vp2`) must correspond to the two cut edges. The cut edges
connect inside vertices to outside vertices:
- For `n_inside = 2` (one outside vertex `v_out`, two inside `v_in1`, `v_in2`):
  `btri_vp1` is the VP on edge `sorted(v_out, v_in1)`, `btri_vp2` on `sorted(v_out, v_in2)`.
  `btri_v_in = [v_in1, v_in2]`, `btri_v_out = [v_out, -1]` (padded).
- For `n_inside = 1` (one inside vertex `v_in`, two outside `v_out1`, `v_out2`):
  `btri_vp1` is the VP on edge `sorted(v_in, v_out1)`, `btri_vp2` on `sorted(v_in, v_out2)`.
  `btri_v_in = [v_in, -1]` (padded), `btri_v_out = [v_out1, v_out2]`.

### Step 3: Vectorized perimeter computation

Create `src/core/vectorized_perimeter.py` with functions (not a class) that operate on
`PartitionArrays`.

#### Helper: compute all VP positions

```python
def _compute_vp_positions(pa: PartitionArrays) -> np.ndarray:
    """Compute all VP positions from lambdas. Returns shape (n_active_vp, dim)."""
    return (pa.vp_lambda[:, None] * pa.vertices[pa.vp_edge_v1]
          + (1 - pa.vp_lambda[:, None]) * pa.vertices[pa.vp_edge_v2])
```

This matches `VariablePoint.evaluate()`: `λ * vertices[edge[0]] + (1-λ) * vertices[edge[1]]`.

#### Total perimeter (objective function)

The current `compute_total_perimeter` sums per-cell perimeters, counting each segment twice.
The vectorized version must reproduce this:

```python
def compute_total_perimeter(pa: PartitionArrays) -> float:
    """Compute total perimeter matching PerimeterCalculator.compute_total_perimeter().

    Each segment is counted once per cell it belongs to (twice total, since each
    segment separates exactly 2 cells). This matches the sum-of-cell-perimeters
    convention used throughout the codebase.
    """
    pos = _compute_vp_positions(pa)
    diff = pos[pa.seg_vp1] - pos[pa.seg_vp2]
    lengths = np.linalg.norm(diff, axis=1)
    return float(2.0 * lengths.sum())
```

#### Perimeter gradient (objective gradient)

Same double-counting convention for the gradient:

```python
def compute_perimeter_gradient(pa: PartitionArrays) -> np.ndarray:
    """Compute ∂(total_perimeter)/∂λ matching PerimeterCalculator.compute_total_perimeter_gradient().

    Each segment's gradient contribution is counted twice (once per cell).
    """
    pos = _compute_vp_positions(pa)
    diff = pos[pa.seg_vp1] - pos[pa.seg_vp2]
    lengths = np.linalg.norm(diff, axis=1)
    lengths = np.maximum(lengths, 1e-12)

    # ∂x/∂λ = vertices[edge[0]] - vertices[edge[1]] (from evaluate formula)
    dv1 = pa.vertices[pa.vp_edge_v1[pa.seg_vp1]] - pa.vertices[pa.vp_edge_v2[pa.seg_vp1]]
    dv2 = pa.vertices[pa.vp_edge_v1[pa.seg_vp2]] - pa.vertices[pa.vp_edge_v2[pa.seg_vp2]]

    # Per-segment gradients (paper equations 349-353)
    grad_vp1 = np.sum(diff * dv1, axis=1) / lengths
    grad_vp2 = np.sum(-diff * dv2, axis=1) / lengths

    # Scatter-add into gradient vector, then multiply by 2 for double-counting
    gradient = np.zeros(pa.n_active_vp)
    np.add.at(gradient, pa.seg_vp1, grad_vp1)
    np.add.at(gradient, pa.seg_vp2, grad_vp2)
    gradient *= 2.0

    return gradient
```

### Step 4: Vectorized area computation

Create `src/core/vectorized_area.py` with functions that operate on `PartitionArrays`.

**Reminder**: Triple-point triangles are NOT in the boundary triangle arrays. Their area
contribution is added separately by the SteinerHandler (unchanged).

#### Cell areas (constraint function)

The total area for cell `c` is:

```
Area_c = cell_interior_area[c]
       + sum of partial areas for boundary triangles of cell c
       + Steiner area contributions (added separately, not here)
```

For boundary triangles with 2 vertices inside (quadrilateral portion):
- The quadrilateral (p_in1, p_cut1, p_cut2, p_in2) is split into two triangles:
  (p_in1, p_cut1, p_cut2) and (p_in1, p_cut2, p_in2)
- This matches `AreaCalculator._quadrilateral_area(p_in1, p_cut1, p_cut2, p_in2)`

For boundary triangles with 1 vertex inside (small triangle):
- The triangle (p_in, p_cut1, p_cut2)
- This matches `AreaCalculator._triangle_area_3d(p_in, p_cut1, p_cut2)`

The cut point positions are computed using the VP evaluate formula, which is the same
formula used in `AreaCalculator._partial_area_one_inside` and `_partial_area_two_inside`
(both branches of their edge-orientation conditionals simplify to the evaluate formula).

```python
def compute_cell_areas(pa: PartitionArrays) -> np.ndarray:
    """Compute all cell areas (excluding Steiner contributions).

    Matches AreaCalculator.compute_all_cell_areas(). Steiner contributions
    must be added separately by the caller.
    """
    areas = pa.cell_interior_area.copy()
    pos = _compute_vp_positions(pa)

    # --- 1-inside boundary triangles: area = triangle(p_in, p_cut1, p_cut2) ---
    mask1 = pa.btri_n_inside == 1
    if np.any(mask1):
        p_in = pa.vertices[pa.btri_v_in[mask1, 0]]
        p_cut1 = pos[pa.btri_vp1[mask1]]
        p_cut2 = pos[pa.btri_vp2[mask1]]
        tri_areas = _triangle_areas_batch(p_in, p_cut1, p_cut2)
        np.add.at(areas, pa.btri_cell[mask1], tri_areas)

    # --- 2-inside boundary triangles: area = quad(p_in1, p_cut1, p_cut2, p_in2) ---
    mask2 = pa.btri_n_inside == 2
    if np.any(mask2):
        p_in1 = pa.vertices[pa.btri_v_in[mask2, 0]]
        p_in2 = pa.vertices[pa.btri_v_in[mask2, 1]]
        p_cut1 = pos[pa.btri_vp1[mask2]]
        p_cut2 = pos[pa.btri_vp2[mask2]]
        quad_areas = (_triangle_areas_batch(p_in1, p_cut1, p_cut2)
                    + _triangle_areas_batch(p_in1, p_cut2, p_in2))
        np.add.at(areas, pa.btri_cell[mask2], quad_areas)

    return areas


def _triangle_areas_batch(p1, p2, p3):
    """Batch triangle area. Matches AreaCalculator._triangle_area_3d()."""
    v1 = p2 - p1
    v2 = p3 - p1
    if p1.shape[1] == 2:
        return 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    else:
        cross = np.cross(v1, v2)
        return 0.5 * np.linalg.norm(cross, axis=1)
```

#### Area constraint Jacobian

The Jacobian `∂(Area_c)/∂λ_j` has nonzero entries only for VPs that appear in boundary
triangles. The set of affected VPs is pre-computed in `pa.area_affected_vps`.

Use finite differences on the vectorized area computation, matching the current
`AreaCalculator` which also uses finite differences:

```python
def compute_area_jacobian(pa: PartitionArrays, eps: float = 1e-7) -> np.ndarray:
    """Compute area constraint Jacobian via vectorized finite differences.

    Matches AreaCalculator.compute_area_jacobian(). Steiner Jacobian
    contributions must be added separately by the caller.
    """
    n_constraints = pa.n_cells - 1
    base_areas = compute_cell_areas(pa)

    jacobian = np.zeros((n_constraints, pa.n_active_vp))
    original_lambda = pa.vp_lambda.copy()

    for vp_idx in pa.area_affected_vps:
        pa.vp_lambda[vp_idx] = original_lambda[vp_idx] + eps
        perturbed_areas = compute_cell_areas(pa)
        pa.vp_lambda[vp_idx] = original_lambda[vp_idx]
        jacobian[:, vp_idx] = (perturbed_areas[:n_constraints] - base_areas[:n_constraints]) / eps

    return jacobian
```

### Step 5: Steiner handler integration

The `SteinerHandler` computes perimeter, gradient, area, and area Jacobian contributions
from triple-point triangles. These are added to the vectorized regular contributions in
`PerimeterOptimizer`.

#### Current cost (8 triple points — negligible)

For the current 5-cell test with 8 triple points, the Steiner handler runs in milliseconds.
No changes needed for this case.

#### Future cost (thousands of cells — will dominate)

For n cells on a closed surface, the number of triple points scales as ~2n. Each SLSQP
function evaluation currently requires per triple point:

| Operation | BFGS solves |
|---|---|
| `get_total_perimeter_contribution` | 1 |
| `compute_total_gradient_finite_difference` | 3 (one per VP perturbation) |
| `get_total_area_contribution` | 1 |
| `compute_area_gradients_finite_difference` | 3 (one per VP perturbation) |
| **Total per triple point per evaluation** | **~8** |

For 1,000 cells (~2,000 TPs): ~16,000 BFGS solves per evaluation.
For 100,000 cells (~200,000 TPs): ~1,600,000 BFGS solves per evaluation.

Additionally, the current code redundantly resets `steiner_point = None` and recomputes
the Steiner point in each of the 4 method calls per evaluation. This means each triple
point's BFGS is solved ~4× per evaluation instead of 1×.

#### Recommended improvements (future, not required for initial implementation)

1. **Cache Steiner points per evaluation**: Compute each Steiner point once per λ-vector
   and reuse across perimeter, gradient, area, and area Jacobian calls. This gives a 4×
   reduction in BFGS solves immediately.

2. **Analytical Steiner point (Fermat-Torricelli)**: The BFGS solves
   `min_S ||S - p1|| + ||S - p2|| + ||S - p3||`, which is the Fermat-Torricelli problem.
   When all angles of triangle (p1, p2, p3) are < 120°, the solution has a known
   closed-form. When any angle ≥ 120°, the solution is the obtuse vertex (which is the
   Type 2 trigger). Replacing BFGS with the analytical formula would eliminate iterative
   solves entirely and enable batch computation over all triple points.

3. **Batch vectorized Steiner**: With the analytical formula, all triple points' Steiner
   computations (positions, perimeter contributions, area contributions, gradients) can be
   expressed as array operations over flat arrays, just like the regular perimeter/area.

### Step 6: Wire into PerimeterOptimizer

Modify `src/core/perimeter_optimizer.py` to use the vectorized path.

#### Compile before optimization:

```python
def compile(self):
    """Compile flat arrays for fast evaluation. Call after migrations, before optimize()."""
    self._arrays = self.partition.compile_arrays()
```

#### Replace objective():

```python
def objective(self, lambda_vec: np.ndarray) -> float:
    self._arrays.vp_lambda[:] = lambda_vec

    regular_perimeter = vectorized_perimeter.compute_total_perimeter(self._arrays)

    # SteinerHandler reads from VP objects — sync before calling
    self.partition.set_variable_vector(self._to_full(lambda_vec))
    steiner_perimeter = self.steiner_handler.get_total_perimeter_contribution()

    total = regular_perimeter + steiner_perimeter
    self._last_objective = total
    return total
```

**Note**: `partition.set_variable_vector()` is needed because SteinerHandler calls
`partition.evaluate_variable_point()` which reads from the VP objects. This adds a
Python loop over all VPs per evaluation — acceptable for now but should be eliminated
when the Steiner handler is vectorized.

#### Replace objective_gradient():

Same pattern — vectorized regular gradient + object-based Steiner gradient.
Both use the full (absolute-indexed) gradient vector; the active-only compression
happens at the return.

#### Replace constraint_area_equality():

Vectorized area + object-based Steiner area correction.

#### Replace constraint_area_jacobian():

Vectorized area Jacobian + object-based Steiner area Jacobian.

#### After optimization completes:

Sync the optimized lambdas back to the PartitionContour objects:

```python
self.partition.set_variable_vector(self._to_full(result.x))
```

### Step 7: Fix the callback overhead

The `_callback()` method currently recomputes the objective and constraints at every
iteration — this is redundant because SLSQP already called `objective()` internally.
Cache the last value instead:

```python
def objective(self, lambda_vec):
    ...
    self._last_objective = total
    return total

def _callback(self, *args, **kwargs):
    self.iteration += 1
    if self.iteration % 10 == 0:
        self.logger.info(f"Iteration {self.iteration}: Perimeter={self._last_objective:.6f}")
```

## Files to Create

| File | Purpose |
|---|---|
| `src/core/partition_arrays.py` | `PartitionArrays` dataclass |
| `src/core/vectorized_perimeter.py` | Vectorized perimeter + gradient functions |
| `src/core/vectorized_area.py` | Vectorized area + Jacobian functions |

## Files to Modify

| File | Change |
|---|---|
| `src/core/contour_partition.py` | Add `compile_arrays()` method to `PartitionContour` |
| `src/core/perimeter_optimizer.py` | Wire vectorized path into `objective()`, `objective_gradient()`, `constraint_area_equality()`, `constraint_area_jacobian()`, fix callback |

## Files NOT Modified

| File | Reason |
|---|---|
| `src/core/migration_detector.py` | Uses object structures for graph queries — correct as-is |
| `src/core/migration_executor.py` | Mutates VP objects and partition state — correct as-is |
| `src/core/migration_orchestrator.py` | Orchestration logic — correct as-is |
| `src/core/one_ring_rebuilder.py` | One-ring rebuild — correct as-is |
| `src/core/migration_types.py` | Data containers — no computation |
| `src/core/migration_utils.py` | Helper functions — correct as-is |
| `src/core/topology_switcher.py` | Topology switch logic — correct as-is |
| `src/core/steiner_handler.py` | Unchanged for initial implementation |
| `src/core/perimeter_calculator.py` | Kept as reference / validation implementation |
| `src/core/area_calculator.py` | Kept as reference / validation implementation |

## Testing Strategy

1. **Numerical equivalence**: For a given partition state + lambda vector, the vectorized
   functions must match the original implementations to floating-point tolerance (~1e-12
   relative error):
   - `vectorized_perimeter.compute_total_perimeter()` vs `PerimeterCalculator.compute_total_perimeter()`
   - `vectorized_perimeter.compute_perimeter_gradient()` vs `PerimeterCalculator.compute_total_perimeter_gradient()`
   - `vectorized_area.compute_cell_areas()` vs `AreaCalculator.compute_all_cell_areas()`
   - `vectorized_area.compute_area_jacobian()` vs `AreaCalculator.compute_area_jacobian()`

2. **Gradient verification**: Compare vectorized analytical perimeter gradient against
   finite differences (already implemented in
   `PerimeterCalculator.verify_gradient_finite_differences`).

3. **Round-trip test**: Run one full optimize cycle with vectorized path, compare final
   perimeter and constraint violation against the original path run on the same input.

4. **Migration compatibility**: Run a detect → migrate → compile_arrays → optimize cycle.
   Verify that `compile_arrays()` correctly handles inactive VPs and post-migration state.

## Implementation Order

1. `PartitionArrays` dataclass + `compile_arrays()` method (foundation)
2. `compute_total_perimeter()` vectorized + test against `PerimeterCalculator`
3. `compute_perimeter_gradient()` vectorized + test against `PerimeterCalculator`
4. `compute_cell_areas()` vectorized + test against `AreaCalculator`
5. `compute_area_jacobian()` vectorized (finite-difference) + test against `AreaCalculator`
6. Wire into `PerimeterOptimizer` with fallback to original path via flag
7. Fix callback overhead
8. End-to-end test: full optimize cycle with identical results
9. (Future) Cache Steiner points to eliminate redundant BFGS solves
10. (Future) Analytical Steiner point formula + batch vectorized Steiner
11. (Future) Analytical area Jacobian to eliminate FD loop

## Key Invariants to Preserve

- `PartitionContour` VP objects remain the source of truth for mutations
- After optimization, `partition.set_variable_vector(result.x)` syncs back
- `compile_arrays()` is idempotent — can be called multiple times safely
- Active/inactive VP distinction is respected (inactive VPs excluded from arrays)
- Edge normalization convention: `edge[0] < edge[1]` (same as `VariablePoint`)
- Lambda convention: `x = λ * vertices[edge[0]] + (1-λ) * vertices[edge[1]]`
- Perimeter = sum of cell perimeters (each segment counted twice)
- Triple-point triangles excluded from area arrays (SteinerHandler handles them)
- Steiner area = void triangle area + corner triangle area per cell
