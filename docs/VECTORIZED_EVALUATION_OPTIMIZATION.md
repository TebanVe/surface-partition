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
│  Used by: VectorizedPerimeterCalculator,                    │
│           VectorizedAreaCalculator                           │
│  Runtime: milliseconds per evaluation                       │
└─────────────────────────────────────────────────────────────┘
```

## What to Implement

### Step 1: PartitionArrays dataclass

Create `src/core/partition_arrays.py` with a frozen array snapshot of the partition state.

```python
@dataclass
class PartitionArrays:
    """Flat array representation of partition state for vectorized evaluation."""

    # Variable point arrays (shape: n_active_vp)
    vp_edge_v1: np.ndarray       # int32 — first vertex index of edge
    vp_edge_v2: np.ndarray       # int32 — second vertex index of edge
    vp_lambda: np.ndarray        # float64 — lambda parameters (mutable during optim)

    # Boundary segment arrays (shape: n_segments)
    seg_vp1: np.ndarray          # int32 — first VP index (into active VP arrays)
    seg_vp2: np.ndarray          # int32 — second VP index (into active VP arrays)
    seg_cell_a: np.ndarray       # int32 — first cell of the pair
    seg_cell_b: np.ndarray       # int32 — second cell of the pair

    # Boundary triangle arrays (shape: n_boundary_triangles)
    # For area computation of partial triangles
    btri_idx: np.ndarray         # int32 — original triangle index in mesh
    btri_cell: np.ndarray        # int32 — cell this boundary triangle contributes to
    btri_n_inside: np.ndarray    # int32 — number of vertices inside cell (1 or 2)
    btri_v_in: np.ndarray        # int32, shape (n_btri, 2) — inside vertex indices (padded)
    btri_v_out: np.ndarray       # int32, shape (n_btri, 2) — outside vertex indices (padded)
    btri_vp1: np.ndarray         # int32 — first VP index on cut edge
    btri_vp2: np.ndarray         # int32 — second VP index on cut edge
    btri_edge1_v_first: np.ndarray  # int32 — which vertex is edge[0] for VP 1
    btri_edge2_v_first: np.ndarray  # int32 — which vertex is edge[0] for VP 2

    # Pre-computed constants
    cell_interior_area: np.ndarray  # float64, shape (n_cells,) — constant interior area per cell
    n_cells: int
    n_active_vp: int

    # Mesh vertex coordinates (reference, not copied)
    vertices: np.ndarray         # float64, shape (N, 2 or 3) — mesh vertices
```

### Step 2: compile_arrays() method on PartitionContour

Add a method to `PartitionContour` that walks the object structures once and fills the flat
arrays. This is called once before optimization starts.

Location: `src/core/contour_partition.py`, add method to class `PartitionContour`.

The method must:

1. Build a mapping from absolute VP index to active VP index (skip inactive VPs).
2. Fill `vp_edge_v1`, `vp_edge_v2`, `vp_lambda` from `self.variable_points` (active only).
3. Fill `seg_vp1`, `seg_vp2`, `seg_cell_a`, `seg_cell_b` from `self.boundary_segments`,
   remapping VP indices through the active-index map.
4. Fill boundary triangle arrays by iterating `self.mesh.faces`, classifying each triangle
   by vertex labels (same logic as `AreaCalculator._categorize_triangles()`), and for
   boundary triangles, recording the VP indices on the cut edges via `self.edge_to_varpoint`.
5. Fill `cell_interior_area` by summing `mesh.triangle_areas[tri_idx]` for fully interior
   triangles (same logic as current `AreaCalculator.__init__`).
6. Return a `PartitionArrays` instance.

Important edge case: the active-to-absolute index mapping must be stored so that the
optimized lambda vector can be written back to the VP objects after optimization completes.

### Step 3: Vectorized perimeter computation

Create `src/core/vectorized_perimeter.py` with functions (not a class) that operate on
`PartitionArrays`.

#### Total perimeter (objective function)

```python
def compute_total_perimeter(pa: PartitionArrays) -> float:
    """Compute total perimeter from flat arrays. O(n_segments) vectorized."""
    # All VP positions at once
    pos = pa.vp_lambda[:, None] * pa.vertices[pa.vp_edge_v1] \
        + (1 - pa.vp_lambda[:, None]) * pa.vertices[pa.vp_edge_v2]

    # All segment lengths at once
    diff = pos[pa.seg_vp1] - pos[pa.seg_vp2]
    lengths = np.linalg.norm(diff, axis=1)

    return float(lengths.sum())
```

#### Perimeter gradient (objective gradient)

```python
def compute_perimeter_gradient(pa: PartitionArrays) -> np.ndarray:
    """Compute ∂(total_perimeter)/∂λ for all active VPs. Vectorized."""
    pos = pa.vp_lambda[:, None] * pa.vertices[pa.vp_edge_v1] \
        + (1 - pa.vp_lambda[:, None]) * pa.vertices[pa.vp_edge_v2]

    diff = pos[pa.seg_vp1] - pos[pa.seg_vp2]
    lengths = np.linalg.norm(diff, axis=1)
    lengths = np.maximum(lengths, 1e-12)  # avoid division by zero

    # Edge direction vectors for each segment's VPs
    dv1 = pa.vertices[pa.vp_edge_v1[pa.seg_vp1]] - pa.vertices[pa.vp_edge_v2[pa.seg_vp1]]
    dv2 = pa.vertices[pa.vp_edge_v1[pa.seg_vp2]] - pa.vertices[pa.vp_edge_v2[pa.seg_vp2]]

    # Per-segment gradients: ∂ℓ/∂λ_i and ∂ℓ/∂λ_j (paper equations 349-353)
    grad_vp1 = np.sum(diff * dv1, axis=1) / lengths
    grad_vp2 = np.sum(-diff * dv2, axis=1) / lengths

    # Scatter-add into gradient vector
    gradient = np.zeros(pa.n_active_vp)
    np.add.at(gradient, pa.seg_vp1, grad_vp1)
    np.add.at(gradient, pa.seg_vp2, grad_vp2)

    return gradient
```

#### Per-cell perimeter (for diagnostics only, not in hot path)

```python
def compute_cell_perimeters(pa: PartitionArrays) -> np.ndarray:
    """Compute per-cell perimeters. Each segment contributes to both cells."""
    pos = pa.vp_lambda[:, None] * pa.vertices[pa.vp_edge_v1] \
        + (1 - pa.vp_lambda[:, None]) * pa.vertices[pa.vp_edge_v2]
    diff = pos[pa.seg_vp1] - pos[pa.seg_vp2]
    lengths = np.linalg.norm(diff, axis=1)

    perimeters = np.zeros(pa.n_cells)
    np.add.at(perimeters, pa.seg_cell_a, lengths)
    np.add.at(perimeters, pa.seg_cell_b, lengths)
    return perimeters
```

### Step 4: Vectorized area computation

Create `src/core/vectorized_area.py` with functions that operate on `PartitionArrays`.

#### Cell areas (constraint function)

The total area for cell `c` is:

```
Area_c = cell_interior_area[c] + sum of partial areas for boundary triangles of cell c
```

For boundary triangles with 2 vertices inside (quadrilateral portion):
- The area is that of the quadrilateral (p_in1, p_cut1, p_cut2, p_in2)
- Split into two triangles and sum their cross-product areas

For boundary triangles with 1 vertex inside (small triangle):
- The area is that of triangle (p_in, p_cut1, p_cut2)

These can be batched by `btri_n_inside`:
- Gather all "2-inside" boundary triangles, compute quadrilateral areas vectorized
- Gather all "1-inside" boundary triangles, compute triangle areas vectorized
- Scatter-add into per-cell area array

```python
def compute_cell_areas(pa: PartitionArrays) -> np.ndarray:
    """Compute all cell areas. Vectorized over boundary triangles."""
    areas = pa.cell_interior_area.copy()

    # Evaluate cut point positions from VP lambdas
    pos = pa.vp_lambda[:, None] * pa.vertices[pa.vp_edge_v1] \
        + (1 - pa.vp_lambda[:, None]) * pa.vertices[pa.vp_edge_v2]

    # --- 1-inside boundary triangles ---
    mask1 = pa.btri_n_inside == 1
    if np.any(mask1):
        p_in = pa.vertices[pa.btri_v_in[mask1, 0]]
        p_cut1 = pos[pa.btri_vp1[mask1]]
        p_cut2 = pos[pa.btri_vp2[mask1]]
        tri_areas = _triangle_areas_batch(p_in, p_cut1, p_cut2)
        np.add.at(areas, pa.btri_cell[mask1], tri_areas)

    # --- 2-inside boundary triangles ---
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
```

Where the helper is:

```python
def _triangle_areas_batch(p1, p2, p3):
    """Batch triangle area computation. p1, p2, p3 are (n, 2or3) arrays."""
    v1 = p2 - p1
    v2 = p3 - p1
    if p1.shape[1] == 2:
        return 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    else:
        cross = np.cross(v1, v2)
        return 0.5 * np.linalg.norm(cross, axis=1)
```

#### Area constraint Jacobian

The Jacobian ∂(Area_c)/∂λ_j has nonzero entries only for VPs on boundary triangles of
cell c. Use finite differences on the vectorized area computation:

```python
def compute_area_jacobian(pa: PartitionArrays, eps: float = 1e-7) -> np.ndarray:
    """Compute area constraint Jacobian via vectorized finite differences."""
    n_constraints = pa.n_cells - 1
    base_areas = compute_cell_areas(pa)

    # Find which VPs appear in boundary triangles (sparse set)
    affected_vps = np.unique(np.concatenate([pa.btri_vp1, pa.btri_vp2]))

    jacobian = np.zeros((n_constraints, pa.n_active_vp))
    original_lambda = pa.vp_lambda.copy()

    for vp_idx in affected_vps:
        pa.vp_lambda[vp_idx] = original_lambda[vp_idx] + eps
        perturbed_areas = compute_cell_areas(pa)
        pa.vp_lambda[vp_idx] = original_lambda[vp_idx]
        jacobian[:, vp_idx] = (perturbed_areas[:n_constraints] - base_areas[:n_constraints]) / eps

    return jacobian
```

**Important optimization**: Only VPs that appear in `btri_vp1` or `btri_vp2` have nonzero
area gradients. The set `affected_vps` is typically much smaller than `n_active_vp` (only
VPs on boundary-triangle edges). This should be computed once in `compile_arrays()`.

**Further optimization**: Replace the finite-difference loop with analytical gradients.
The partial area formulas for 1-inside and 2-inside triangles depend on λ through the cut
point positions. The chain rule gives ∂(area)/∂λ_j analytically. This eliminates the
remaining Python loop over affected VPs.

### Step 5: Steiner tree contributions (keep as-is initially)

The Steiner handler manages only ~8 triple points. Its contribution to total evaluation time
is negligible. For the initial implementation, keep the existing `SteinerHandler` methods
unchanged:

- `get_total_perimeter_contribution()` — adds Steiner perimeter to vectorized regular perimeter
- `compute_total_gradient_finite_difference()` — adds Steiner gradient to vectorized gradient
- `get_total_area_contribution()` — adds Steiner areas to vectorized cell areas
- `compute_area_gradients_finite_difference()` — adds Steiner Jacobian rows

These are called once per function evaluation and process 8 triple points via BFGS + finite
differences. Total time per call is milliseconds, so this is not a priority.

### Step 6: Wire into PerimeterOptimizer

Modify `src/core/perimeter_optimizer.py` to use the vectorized path.

#### In `__init__` or a new `compile()` method:

```python
def compile(self):
    """Compile flat arrays for fast evaluation. Call after migrations, before optimize()."""
    self._arrays = self.partition.compile_arrays()
    self._arrays_stale = False
```

#### Replace `objective()`:

```python
def objective(self, lambda_vec: np.ndarray) -> float:
    # Update lambda in the compiled arrays (no Python loop)
    self._arrays.vp_lambda[:] = lambda_vec

    regular_perimeter = vectorized_perimeter.compute_total_perimeter(self._arrays)
    steiner_perimeter = self.steiner_handler.get_total_perimeter_contribution()
    return regular_perimeter + steiner_perimeter
```

**Critical**: Note that `self._arrays.vp_lambda[:] = lambda_vec` is a single array copy
operation — no Python loop, no object attribute updates. The `PartitionContour` object
lambdas are NOT updated during optimization. They are synced back once at the end.

However, the `SteinerHandler` methods still call `partition.evaluate_variable_point()`,
which reads from the VP objects. So before calling Steiner methods, sync the lambda:

```python
def objective(self, lambda_vec: np.ndarray) -> float:
    self._arrays.vp_lambda[:] = lambda_vec

    regular_perimeter = vectorized_perimeter.compute_total_perimeter(self._arrays)

    # Steiner still uses object-based partition (only ~8 triple points)
    self.partition.set_variable_vector(self._to_full(lambda_vec))
    steiner_perimeter = self.steiner_handler.get_total_perimeter_contribution()

    return regular_perimeter + steiner_perimeter
```

The `set_variable_vector` call is cheap for ~8 Steiner VP lookups. Later, Steiner
evaluation can be moved to arrays too, but it's not a priority.

#### Replace `objective_gradient()`:

Same pattern — vectorized regular gradient + object-based Steiner gradient.

#### Replace `constraint_area_equality()`:

Vectorized area computation + object-based Steiner area correction.

#### Replace `constraint_area_jacobian()`:

Vectorized area Jacobian + object-based Steiner Jacobian correction.

#### After optimization completes:

Sync the optimized lambdas back to the PartitionContour objects:

```python
self.partition.set_variable_vector(self._to_full(result.x))
```

### Step 7: Fix the callback overhead

In `perimeter_optimizer.py`, the `_callback()` method currently recomputes the objective
and constraints at every iteration:

```python
def _callback(self, *args, **kwargs):
    ...
    obj = self.objective(xk)                          # REDUNDANT
    constraints = self.constraint_area_equality(xk)   # REDUNDANT
```

SLSQP already computed these internally. Options:
1. Remove the objective/constraint recomputation from the callback entirely, and only log
   the iteration count. This halves the per-iteration cost.
2. Keep logging but only every N iterations (e.g., every 50).
3. Cache the last objective value from `objective()` and reuse in callback.

Option 3 is cleanest:

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

## Expected Impact

### Per-evaluation time breakdown (current vs. vectorized)

| Component | Current | Vectorized | Speedup |
|---|---|---|---|
| Perimeter (2000 segments) | ~3s | ~0.1ms | ~30,000x |
| Perimeter gradient | ~4s | ~0.2ms | ~20,000x |
| Area constraints (3940 btri) | ~2s | ~0.2ms | ~10,000x |
| Area Jacobian (4 × 3940 btri) | ~4s | ~1ms (FD) or ~0.3ms (analytical) | ~4,000-13,000x |
| Steiner contributions (8 TPs) | ~1s | ~1s (unchanged) | 1x |
| Callback overhead | ~14s | ~0s (cached) | ∞ |
| **Total per evaluation** | **~28s** | **~1.5s** (FD Jacobian) or **~1.3s** (analytical) | **~20x** |

With analytical area Jacobian and Steiner still object-based:

| Component | Time |
|---|---|
| Vectorized perimeter + gradient + area + Jacobian | ~1ms |
| Steiner handler (8 triple points, BFGS + FD) | ~1s |
| **Total per evaluation** | **~1s** |

Note: The Steiner handler will become the new bottleneck. If needed, it can be optimized
later by caching Steiner points across evaluations (they change slowly) or vectorizing the
BFGS calls.

### Total optimization time

| Metric | Current | After vectorization |
|---|---|---|
| Per function evaluation | ~14s | ~1s |
| 653 iterations, 1309 fevals | 18,432s (5.1 hrs) | ~1,300s (22 min) |
| 4 refinement cycles | 20+ hours | ~1.5 hours |

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

## Files NOT Modified (critical)

| File | Reason |
|---|---|
| `src/core/migration_detector.py` | Uses object structures for graph queries — correct as-is |
| `src/core/migration_executor.py` | Mutates VP objects and partition state — correct as-is |
| `src/core/migration_orchestrator.py` | Orchestration logic — correct as-is |
| `src/core/one_ring_rebuilder.py` | One-ring rebuild — correct as-is |
| `src/core/migration_types.py` | Data containers — no computation |
| `src/core/migration_utils.py` | Helper functions — correct as-is |
| `src/core/topology_switcher.py` | Topology switch logic — correct as-is |
| `src/core/perimeter_calculator.py` | Kept as fallback / reference implementation |
| `src/core/area_calculator.py` | Kept as fallback / reference implementation |

## Testing Strategy

1. **Numerical equivalence**: For a given partition state + lambda vector, the vectorized
   perimeter/area/gradient must match the original implementation to within floating-point
   tolerance (~1e-12 relative error).

2. **Gradient verification**: Compare vectorized analytical gradient against finite
   differences (already implemented in `PerimeterCalculator.verify_gradient_finite_differences`).

3. **Round-trip test**: Run one full optimize cycle with vectorized path, compare final
   perimeter and constraint violation against the original path run on the same input.

4. **Migration compatibility**: Run a detect → migrate → compile_arrays → optimize cycle.
   Verify that `compile_arrays()` correctly handles inactive VPs and post-migration state.

## Implementation Order

1. `PartitionArrays` dataclass + `compile_arrays()` method (foundation)
2. `compute_total_perimeter()` vectorized + test against `PerimeterCalculator`
3. `compute_perimeter_gradient()` vectorized + test against `PerimeterCalculator`
4. `compute_cell_areas()` vectorized + test against `AreaCalculator`
5. `compute_area_jacobian()` vectorized (finite-difference first) + test
6. Wire into `PerimeterOptimizer` with fallback to original path via flag
7. Fix callback overhead
8. End-to-end test: full optimize cycle
9. Optional: analytical area Jacobian to eliminate FD loop
10. Optional: vectorize Steiner contributions if they become the bottleneck

## Key Invariants to Preserve

- `PartitionContour` VP objects remain the source of truth for mutations
- After optimization, `partition.set_variable_vector(result.x)` syncs back
- `compile_arrays()` is idempotent — can be called multiple times safely
- Active/inactive VP distinction is respected (inactive VPs excluded from arrays)
- Edge normalization convention: `edge[0] < edge[1]` (same as `VariablePoint`)
- Lambda convention: `x = λ * vertices[edge[0]] + (1-λ) * vertices[edge[1]]`
