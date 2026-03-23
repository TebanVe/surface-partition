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
│           TopologySwitcher, SteinerHandler (mutation only)   │
│  Runtime: seconds                                           │
└──────────────────────┬──────────────────────────────────────┘
                       │ partition.compile_arrays()
                       │ (called once before optimization)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Mode B: Evaluation (flat NumPy arrays) — NEW               │
│  Used by: vectorized_perimeter, vectorized_area,            │
│           vectorized_steiner                                 │
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

The area of triple-point mesh triangles is computed **entirely** by the Steiner component,
which decomposes each triple-point triangle into:
- **Corner area** per cell: triangle (mesh_vertex, VP_1, VP_2) for that cell's vertex
- **Void area** per cell: triangle (VP_1, VP_2, steiner_point) for that cell's VP pair

The current (pre-vectorization) code adds these in `PerimeterOptimizer`:

```python
areas = self.area_calc.compute_all_cell_areas(full_vec)
steiner_areas = self.steiner_handler.get_total_area_contribution()
for cell_idx, area_contrib in steiner_areas.items():
    areas[cell_idx] += area_contrib
```

The vectorized version preserves this same separation: `vectorized_area.compute_cell_areas()`
handles regular boundary triangles, `vectorized_steiner.compute_steiner_areas()` handles
triple-point triangles, and the optimizer adds them together.

The vectorized `compile_arrays()` must **exclude triple-point triangles** from the boundary
triangle arrays, replicating the same `len(set(labels)) == 3` guard.

### 3. Steiner perimeter uses the same per-cell-sum convention

Each cell at a triple point receives a NET Steiner correction:

```python
contributions[cell_idx] = steiner_edge1 + steiner_edge2 - original_edge
```

This sums over all 3 cells per triple point. Each Steiner edge appears in 2 cells'
contributions. This is consistent with the double-counted regular perimeter convention.
The vectorized version (`vectorized_steiner.compute_steiner_perimeter()`) replicates this
per-cell summation as a batch array operation.

### 4. Lambda convention

`VariablePoint.evaluate()` computes: `x = λ * vertices[edge[0]] + (1-λ) * vertices[edge[1]]`
with `edge[0] < edge[1]` (normalized). The vectorized position computation must use the
same formula.

### 5. Steiner evaluation uses flat arrays (not VP objects)

The vectorized Steiner module reads VP positions directly from the flat `vp_lambda` array
in `PartitionArrays`, the same way `vectorized_perimeter` and `vectorized_area` do. There
is no need to call `partition.set_variable_vector()` during the evaluation loop.

The object-based `SteinerHandler` remains for **mutation mode only** (triple-point detection,
Type 2 migration logic, topology switches). It is not called during the SLSQP optimization
loop.

## Implementation Context: Key Source Files and Data Structures

This section provides the codebase navigation context needed to implement the steps below.
Read the listed source files (in the order given) before starting implementation.

### Priority reading list

| File | What to look at | Why |
|---|---|---|
| `src/core/contour_partition.py` | `VariablePoint` dataclass (line ~48), `TriangleSegment` dataclass (line ~93), `BoundarySegment` dataclass (line ~148), `PartitionContour.__init__` (line ~202) | Core data structures that `compile_arrays()` reads from |
| `src/core/perimeter_optimizer.py` | `PerimeterOptimizer.__init__` (line ~61), `_to_full` / `_to_active` / `_to_active_2d` (line ~111), `objective` through `constraint_area_jacobian` (line ~131) | The methods being replaced — study their exact signatures and conventions |
| `src/core/steiner_handler.py` | `TriplePoint` class (line ~29) especially `cell_to_varpoint_pair` and `cell_to_mesh_vertex` dicts, `SteinerHandler._detect_triple_points` (line ~397) | The triple-point structures that `compile_arrays()` step 7 reads from |
| `src/core/area_calculator.py` | `_categorize_triangles` (line ~88), `_partial_area_two_inside` (line ~260), `_partial_area_one_inside` (line ~355) | The triangle classification and area computation logic being replicated |
| `src/core/perimeter_calculator.py` | `compute_segment_length`, `compute_segment_gradient` | The perimeter computation being replicated |
| `src/core/tri_mesh.py` | `TriMesh.__init__` (line ~18) | `vertices` is `(N, 2 or 3)` float64, `faces` is `(T, 3)` int, `triangle_areas` is a cached `(T,)` property |

### Key data structures on PartitionContour

These are the fields that `compile_arrays()` must traverse:

```python
class PartitionContour:
    mesh: TriMesh
    variable_points: List[VariablePoint]      # ALL VPs (active + inactive)
    _triangle_segments: Dict[int, TriangleSegment]  # tri_idx -> TriangleSegment
    edge_to_varpoint: Dict[Tuple[int, int], int]    # normalized_edge -> vp_idx
    boundary_segments: List[BoundarySegment]         # explicit segment list
    _vertex_labels: np.ndarray                       # (N,) int — cell label per mesh vertex
    _active_vp_indices: List[int]                    # absolute VP indices of active VPs
    _vp_idx_to_opt_idx: Dict[int, int]               # absolute -> active position
    n_cells: int

    @property
    def vertex_labels(self) -> np.ndarray            # read-only view of _vertex_labels
```

`VariablePoint` fields used by `compile_arrays()`:

```python
@dataclass
class VariablePoint:
    edge: Tuple[int, int]          # (v_small, v_large), normalized
    lambda_param: float            # λ ∈ [0, 1]
    global_idx: int                # absolute index in variable_points list
    belongs_to_cells: Set[int]     # cells this VP separates
    active: bool = True            # inactive VPs are skipped
```

`BoundarySegment` fields used by `compile_arrays()`:

```python
@dataclass
class BoundarySegment:
    vp_idx_1: int                  # first VP (absolute index)
    vp_idx_2: int                  # second VP (absolute index)
    cell_pair: Tuple[int, int]     # (cell_a, cell_b)
```

`TriplePoint` fields used by `compile_arrays()` (from `SteinerHandler.triple_points`):

```python
class TriplePoint:
    var_point_indices: List[int]                      # 3 VP absolute indices
    cell_indices: List[int]                           # 3 cell indices
    cell_to_varpoint_pair: Dict[int, Tuple[int, int]] # cell -> (vp_a, vp_b) absolute
    cell_to_mesh_vertex: Dict[int, int]               # cell -> mesh vertex index
    triangle_idx: int
```

### The active/inactive VP mapping

After migrations, some VPs are marked `active = False` but remain in `variable_points`
(preserving index stability for snapshot rollback). The optimizer works with active-only
vectors:

- `PartitionContour._active_vp_indices`: list of absolute VP indices that are active
- `PerimeterOptimizer._to_full(active_vec)`: expands active-only → full (inactive VPs
  keep current λ)
- `PerimeterOptimizer._to_active(full_vec)`: compresses full → active-only

The vectorized code eliminates this mapping at the optimizer level. `PartitionArrays`
stores only active VPs and works entirely in active-index space. The
`active_to_absolute` array in `PartitionArrays` serves the same purpose as
`_active_vp_indices` for syncing back after optimization.

### The compile_arrays() / SteinerHandler dependency

`compile_arrays()` is specified as a method on `PartitionContour` (Step 2), but it needs
to read `SteinerHandler.triple_points` (Step 2, item 7). The `SteinerHandler` is created
by `PerimeterOptimizer.__init__`, not by `PartitionContour`.

Resolution: `compile_arrays()` should accept the `SteinerHandler` as an argument:

```python
def compile_arrays(self, steiner_handler: SteinerHandler) -> PartitionArrays:
```

This is called from `PerimeterOptimizer.compile()`, which has access to both:

```python
def compile(self):
    self._arrays = self.partition.compile_arrays(self.steiner_handler)
```

### Area Jacobian: current implementation uses finite differences

The comment in `perimeter_optimizer.py` line 222 says "Regular area Jacobian (from boundary
triangles, analytical)" — this is a **misleading comment in the source code**. The actual
gradient computation inside `AreaCalculator._partial_area_two_inside()` (line ~328) and
`_partial_area_one_inside()` uses per-variable-point finite differences (eps=1e-7), not
analytical chain-rule derivatives. The code comment in `_partial_area_two_inside` confirms:
"Compute gradient (simplified - more accurate implementation would use chain rule) / For
now, use finite differences."

The vectorized area Jacobian in Step 4 correctly uses FD to match this behavior. The
"(Future) Analytical area Jacobian" item in the implementation order refers to eventually
replacing both the current and vectorized FD approaches with true analytical derivatives.

### Test infrastructure

Existing tests live in two locations:
- `testing/` — integration tests (`test_migration_and_continue.py`,
  `test_migrations_debug.py`, `test_self_healing_selection.py`)
- `examples/` — example scripts that can serve as test harnesses
  (`test_optimizer.py`, `test_mesh_matrices.py`)

There is no formal test framework (no pytest fixtures or unittest base classes). Tests are
standalone scripts that load partition state from `.h5` files and run checks.

For validation tests of the vectorized code, the recommended approach is:
1. Load an existing partition state (from an `.h5` checkpoint or by running initial setup)
2. Create both the original calculators and the vectorized arrays
3. Compare outputs at the same λ vector to floating-point tolerance
4. Place new tests in `testing/test_vectorized_evaluation.py`

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
    # EXCLUDES triple-point triangles (handled by vectorized_steiner)
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

    # --- Triple-point arrays (from SteinerHandler triple-point detection) ---

    # Per triple point (shape: n_triple_points)
    tp_vp_indices: np.ndarray      # int32, shape (n_tp, 3) — active VP indices
    n_triple_points: int

    # Per (triple_point, cell) contribution rows (shape: 3 * n_triple_points)
    # Each triple point generates exactly 3 rows (one per adjacent cell).
    tp_contrib_tp_idx: np.ndarray    # int32 — which triple point this row belongs to
    tp_contrib_cell: np.ndarray      # int32 — cell index for this contribution
    tp_contrib_vp1: np.ndarray       # int32 — first VP of the cell's pair (active index)
    tp_contrib_vp2: np.ndarray       # int32 — second VP of the cell's pair (active index)
    tp_contrib_mesh_vertex: np.ndarray  # int32 — mesh vertex for corner triangle

    # Set of VP indices involved in any triple point (for sparse FD loops)
    tp_affected_vps: np.ndarray    # int32 — unique active VPs in tp_vp_indices
```

### Step 2: compile_arrays() method on PartitionContour

Add a method to `PartitionContour` that walks the object structures once and fills the flat
arrays. This is called once before optimization starts.

Location: `src/core/contour_partition.py`, add method to class `PartitionContour`.

Signature (takes SteinerHandler for triple-point data — see "Implementation Context" above):

```python
def compile_arrays(self, steiner_handler: SteinerHandler) -> PartitionArrays:
```

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
7. Fill triple-point arrays from `SteinerHandler.triple_points`. For each `TriplePoint`:
   - Store its 3 VP indices (remapped to active indices) in `tp_vp_indices[i]`.
   - For each of its 3 cells (from `cell_to_varpoint_pair`): emit one contribution row
     with the cell index, the cell's two VP active indices, and the mesh vertex index
     (from `cell_to_mesh_vertex`).
   - Set `tp_contrib_tp_idx[k] = i` for the 3 rows belonging to triple point `i`.
8. Compute `tp_affected_vps` as the unique set of active VP indices in `tp_vp_indices`.
9. Return a `PartitionArrays` instance.

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
contribution is computed separately by `vectorized_steiner` and added by the caller.

#### Cell areas (constraint function)

The total area for cell `c` is:

```
Area_c = cell_interior_area[c]
       + sum of partial areas for boundary triangles of cell c
       + Steiner area contributions (added separately by vectorized_steiner)
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
    from vectorized_steiner must be added separately by the caller.
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
    contributions from vectorized_steiner must be added separately by the caller.
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

### Step 5: Vectorized Steiner computation (analytical Fermat-Torricelli)

The current `SteinerHandler` uses BFGS to solve the Fermat-Torricelli problem for each
triple point. This step replaces BFGS with a closed-form analytical formula and expresses
all Steiner contributions as vectorized array operations.

Create `src/core/vectorized_steiner.py` with functions that operate on `PartitionArrays`.

#### Why replace BFGS now (not later)

For n cells on a closed surface, the number of triple points scales as ~2n. Each SLSQP
function evaluation currently requires per triple point:

| Operation | BFGS solves |
|---|---|
| `get_total_perimeter_contribution` | 1 |
| `compute_total_gradient_finite_difference` | ~5 (base + 3 perturbations + restore) |
| `get_total_area_contribution` | 1 |
| `compute_area_gradients_finite_difference` | ~5 (base + 3 perturbations + restore) |
| **Total per triple point per evaluation** | **~12** |

Additionally, the current code redundantly resets `steiner_point = None` in each of the 4
aggregation methods (`get_total_perimeter_contribution`, `get_total_area_contribution`,
`compute_total_gradient_finite_difference`, `compute_area_gradients_finite_difference`),
forcing a fresh BFGS solve every time. There is no cross-call caching.

For 1,000 cells (~2,000 TPs): ~24,000 BFGS solves per evaluation.
For 100,000 cells (~200,000 TPs): ~2,400,000 BFGS solves per evaluation.

With the analytical formula, each "solve" becomes a few arithmetic operations. **Caching
becomes unnecessary** — the formula is so cheap that recomputing is faster than maintaining
a cache with invalidation logic.

#### The Fermat-Torricelli closed-form

The BFGS objective `min_S ||S - p1|| + ||S - p2|| + ||S - p3||` is the classical
Fermat-Torricelli problem. The solution is known analytically:

**Case 1 — all angles of triangle (p1, p2, p3) < 120°:**

The Fermat point (first isogonic center, X₁₃) has barycentric coordinates derived from
trilinear coordinates `csc(A + π/3) : csc(B + π/3) : csc(C + π/3)`:

```
w_i = a_i / sin(A_i + π/3)
S = (w₁ · p₁ + w₂ · p₂ + w₃ · p₃) / (w₁ + w₂ + w₃)
```

where `a_i` is the side length opposite vertex `p_i` and `A_i` is the angle at vertex
`p_i`. This formula works directly in any dimension for coplanar points — it is a weighted
average of positions, requiring only side lengths and angles (computed from dot products).
No projection to/from 2D is needed.

**Case 2 — any angle ≥ 120°:**

The Fermat point degenerates to the vertex with the obtuse angle. This is exactly the
Type 2 trigger condition detected by `MigrationDetector.detect_type2_triggers()`. In the
formula, this arises naturally: as `A_i → 120°`, `sin(A_i + 60°) → 0`, so `w_i → ∞` and
`S → p_i`. For numerical safety, the implementation explicitly checks for the ≥ 120° case
and returns the obtuse vertex rather than relying on the diverging weights.

#### Steiner point computation (batch vectorized)

```python
def compute_steiner_points(pa: PartitionArrays) -> np.ndarray:
    """Compute all Steiner points using the analytical Fermat-Torricelli formula.

    Returns shape (n_triple_points, dim). For triple points where any void
    angle >= 120°, returns the obtuse vertex position (degenerate Steiner).
    """
    if pa.n_triple_points == 0:
        return np.empty((0, pa.vertices.shape[1]))

    pos = _compute_vp_positions(pa)

    # Gather the 3 VP positions for each triple point: shape (n_tp, 3, dim)
    p = pos[pa.tp_vp_indices]  # (n_tp, 3, dim)
    p1, p2, p3 = p[:, 0], p[:, 1], p[:, 2]

    # Side lengths opposite each vertex
    a = np.linalg.norm(p2 - p3, axis=1)  # opposite p1
    b = np.linalg.norm(p1 - p3, axis=1)  # opposite p2
    c = np.linalg.norm(p1 - p2, axis=1)  # opposite p3

    # Angles at each vertex (law of cosines)
    cos_A = np.clip((b**2 + c**2 - a**2) / (2*b*c + 1e-30), -1.0, 1.0)
    cos_B = np.clip((a**2 + c**2 - b**2) / (2*a*c + 1e-30), -1.0, 1.0)
    cos_C = np.clip((a**2 + b**2 - c**2) / (2*a*b + 1e-30), -1.0, 1.0)

    A_ang = np.arccos(cos_A)
    B_ang = np.arccos(cos_B)
    C_ang = np.arccos(cos_C)

    # Barycentric weights: w_i = a_i / sin(A_i + π/3)
    w1 = a / np.maximum(np.sin(A_ang + np.pi/3), 1e-15)
    w2 = b / np.maximum(np.sin(B_ang + np.pi/3), 1e-15)
    w3 = c / np.maximum(np.sin(C_ang + np.pi/3), 1e-15)

    w_sum = w1 + w2 + w3
    steiner = (w1[:, None]*p1 + w2[:, None]*p2 + w3[:, None]*p3) / w_sum[:, None]

    # Handle degenerate case: any angle >= 120° -> Steiner = obtuse vertex
    threshold = 2*np.pi/3  # 120° in radians
    all_angles = np.stack([A_ang, B_ang, C_ang], axis=1)
    degen_mask = all_angles.max(axis=1) >= threshold
    if np.any(degen_mask):
        max_idx = np.argmax(all_angles[degen_mask], axis=1)
        steiner[degen_mask] = p[degen_mask, max_idx]

    return steiner
```

#### Steiner perimeter contribution (batch vectorized)

```python
def compute_steiner_perimeter(pa: PartitionArrays,
                              steiner_pts: np.ndarray) -> float:
    """Compute total Steiner perimeter contribution.

    Per cell per triple point: d(vp_a, S) + d(vp_b, S) - d(vp_a, vp_b).
    Summed over all 3 cells per triple point (consistent with double-counting
    convention — each Steiner edge appears in 2 cells' contributions).
    """
    if pa.n_triple_points == 0:
        return 0.0

    pos = _compute_vp_positions(pa)
    vp1_pos = pos[pa.tp_contrib_vp1]
    vp2_pos = pos[pa.tp_contrib_vp2]
    s_pos = steiner_pts[pa.tp_contrib_tp_idx]

    d_s_vp1 = np.linalg.norm(vp1_pos - s_pos, axis=1)
    d_s_vp2 = np.linalg.norm(vp2_pos - s_pos, axis=1)
    d_vp1_vp2 = np.linalg.norm(vp1_pos - vp2_pos, axis=1)

    return float(np.sum(d_s_vp1 + d_s_vp2 - d_vp1_vp2))
```

#### Steiner area contribution (batch vectorized)

```python
def compute_steiner_areas(pa: PartitionArrays,
                          steiner_pts: np.ndarray) -> np.ndarray:
    """Compute Steiner area contributions per cell.

    Each cell at a triple point gets:
      void_area   = area(vp_a, vp_b, steiner_point)
      corner_area = area(mesh_vertex, vp_a, vp_b)
    Returns shape (n_cells,) with contributions scatter-added.
    """
    areas = np.zeros(pa.n_cells)
    if pa.n_triple_points == 0:
        return areas

    pos = _compute_vp_positions(pa)
    vp1_pos = pos[pa.tp_contrib_vp1]
    vp2_pos = pos[pa.tp_contrib_vp2]
    s_pos = steiner_pts[pa.tp_contrib_tp_idx]
    mv_pos = pa.vertices[pa.tp_contrib_mesh_vertex]

    void_areas = _triangle_areas_batch(vp1_pos, vp2_pos, s_pos)
    corner_areas = _triangle_areas_batch(mv_pos, vp1_pos, vp2_pos)

    np.add.at(areas, pa.tp_contrib_cell, void_areas + corner_areas)
    return areas
```

Note: `_compute_vp_positions` and `_triangle_areas_batch` are shared helpers also used by
`vectorized_perimeter.py` and `vectorized_area.py`. They should be defined in a shared
utility module or imported from `vectorized_perimeter.py`.

#### Steiner gradients via finite differences (using the analytical formula)

The Steiner perimeter and area gradients w.r.t. λ are computed via finite differences,
but now each perturbation evaluates the analytical formula (~microseconds) instead of
running BFGS (~milliseconds). This keeps the implementation simple while being fast:

```python
def compute_steiner_perimeter_gradient(pa: PartitionArrays,
                                       eps: float = 1e-6) -> np.ndarray:
    """∂(steiner_perimeter)/∂λ via finite differences on the analytical formula."""
    gradient = np.zeros(pa.n_active_vp)
    if pa.n_triple_points == 0:
        return gradient

    base_steiner = compute_steiner_points(pa)
    base_perim = compute_steiner_perimeter(pa, base_steiner)
    original_lambda = pa.vp_lambda.copy()

    for vp_idx in pa.tp_affected_vps:
        pa.vp_lambda[vp_idx] = original_lambda[vp_idx] + eps
        pert_steiner = compute_steiner_points(pa)
        pert_perim = compute_steiner_perimeter(pa, pert_steiner)
        gradient[vp_idx] = (pert_perim - base_perim) / eps
        pa.vp_lambda[vp_idx] = original_lambda[vp_idx]

    return gradient


def compute_steiner_area_jacobian(pa: PartitionArrays,
                                  eps: float = 1e-7) -> np.ndarray:
    """∂(steiner_areas)/∂λ via finite differences on the analytical formula."""
    n_constraints = pa.n_cells - 1
    jacobian = np.zeros((n_constraints, pa.n_active_vp))
    if pa.n_triple_points == 0:
        return jacobian

    base_steiner = compute_steiner_points(pa)
    base_areas = compute_steiner_areas(pa, base_steiner)
    original_lambda = pa.vp_lambda.copy()

    for vp_idx in pa.tp_affected_vps:
        pa.vp_lambda[vp_idx] = original_lambda[vp_idx] + eps
        pert_steiner = compute_steiner_points(pa)
        pert_areas = compute_steiner_areas(pa, pert_steiner)
        jacobian[:, vp_idx] = (pert_areas[:n_constraints] - base_areas[:n_constraints]) / eps
        pa.vp_lambda[vp_idx] = original_lambda[vp_idx]

    return jacobian
```

The FD loop runs over `tp_affected_vps` — the unique set of VPs that participate in any
triple point. For 8 triple points with up to 24 VPs, this is negligible. For large
partitions, the inner evaluation (analytical formula + vectorized contributions) scales as
O(n_triple_points) per perturbation, still far cheaper than O(n_tp × BFGS_iterations).

#### Relationship to SteinerHandler (mutation mode)

The object-based `SteinerHandler` is **not removed**. It remains the implementation used
during mutation mode (Mode A) for:

- Triple-point detection (`_detect_triple_points`)
- Type 2 migration logic in `MigrationDetector`
- Topology switch operations in `TopologySwitcher`
- Migration snapshots and rollbacks

The `compile_arrays()` method reads from `SteinerHandler.triple_points` to populate the
flat triple-point arrays. During the SLSQP optimization loop (Mode B), only the vectorized
module is called.

### Step 6: Wire into PerimeterOptimizer

Modify `src/core/perimeter_optimizer.py` to use the fully vectorized path (regular +
Steiner). No `partition.set_variable_vector()` calls during the evaluation loop — all
computation reads from the flat `PartitionArrays`.

#### Compile before optimization:

```python
def compile(self):
    """Compile flat arrays for fast evaluation. Call after migrations, before optimize()."""
    self._arrays = self.partition.compile_arrays(self.steiner_handler)
```

#### Replace objective():

```python
def objective(self, lambda_vec: np.ndarray) -> float:
    self._arrays.vp_lambda[:] = lambda_vec

    regular_perimeter = vectorized_perimeter.compute_total_perimeter(self._arrays)

    steiner_pts = vectorized_steiner.compute_steiner_points(self._arrays)
    steiner_perimeter = vectorized_steiner.compute_steiner_perimeter(
        self._arrays, steiner_pts)

    total = regular_perimeter + steiner_perimeter
    self._last_objective = total
    return total
```

#### Replace objective_gradient():

```python
def objective_gradient(self, lambda_vec: np.ndarray) -> np.ndarray:
    self._arrays.vp_lambda[:] = lambda_vec

    regular_gradient = vectorized_perimeter.compute_perimeter_gradient(self._arrays)
    steiner_gradient = vectorized_steiner.compute_steiner_perimeter_gradient(self._arrays)

    return regular_gradient + steiner_gradient
```

#### Replace constraint_area_equality():

```python
def constraint_area_equality(self, lambda_vec: np.ndarray) -> np.ndarray:
    self._arrays.vp_lambda[:] = lambda_vec

    areas = vectorized_area.compute_cell_areas(self._arrays)

    steiner_pts = vectorized_steiner.compute_steiner_points(self._arrays)
    steiner_areas = vectorized_steiner.compute_steiner_areas(self._arrays, steiner_pts)
    areas += steiner_areas

    return areas[:self._arrays.n_cells - 1] - self.target_area
```

#### Replace constraint_area_jacobian():

```python
def constraint_area_jacobian(self, lambda_vec: np.ndarray) -> np.ndarray:
    self._arrays.vp_lambda[:] = lambda_vec

    regular_jacobian = vectorized_area.compute_area_jacobian(self._arrays)
    steiner_jacobian = vectorized_steiner.compute_steiner_area_jacobian(self._arrays)

    return regular_jacobian + steiner_jacobian
```

#### After optimization completes:

Sync the optimized lambdas back to the PartitionContour objects:

```python
self.partition.set_variable_vector(self._to_full(result.x))
```

This is the only `set_variable_vector()` call — it happens once at the end to bring the
object representation back in sync for subsequent mutation operations.

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
| `src/core/vectorized_steiner.py` | Analytical Steiner points + vectorized contributions + FD gradients |

## Files to Modify

| File | Change |
|---|---|
| `src/core/contour_partition.py` | Add `compile_arrays()` method to `PartitionContour` |
| `src/core/perimeter_optimizer.py` | Wire fully vectorized path (regular + Steiner) into `objective()`, `objective_gradient()`, `constraint_area_equality()`, `constraint_area_jacobian()`, fix callback |

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
| `src/core/steiner_handler.py` | Kept for mutation mode (triple-point detection, Type 2 logic); not called during SLSQP evaluation |
| `src/core/perimeter_calculator.py` | Kept as reference / validation implementation |
| `src/core/area_calculator.py` | Kept as reference / validation implementation |

## Testing Strategy

1. **Numerical equivalence (regular)**: For a given partition state + lambda vector, the
   vectorized functions must match the original implementations to floating-point tolerance
   (~1e-12 relative error):
   - `vectorized_perimeter.compute_total_perimeter()` vs `PerimeterCalculator.compute_total_perimeter()`
   - `vectorized_perimeter.compute_perimeter_gradient()` vs `PerimeterCalculator.compute_total_perimeter_gradient()`
   - `vectorized_area.compute_cell_areas()` vs `AreaCalculator.compute_all_cell_areas()`
   - `vectorized_area.compute_area_jacobian()` vs `AreaCalculator.compute_area_jacobian()`

2. **Analytical Steiner validation**: Compare `vectorized_steiner.compute_steiner_points()`
   against the existing `TriplePoint.compute_steiner_point()` (BFGS) for all triple points
   in the current partition state. Tolerance: ~1e-6 (limited by BFGS `gtol=1e-8`).
   Test cases should include:
   - Equilateral void triangles (all angles = 60°, Steiner = centroid)
   - Acute void triangles (all angles < 120°, non-trivial Fermat point)
   - Near-degenerate triangles (one angle near 120°, tests numerical stability)

3. **Numerical equivalence (Steiner contributions)**: For a given partition state:
   - `vectorized_steiner.compute_steiner_perimeter()` vs
     `SteinerHandler.get_total_perimeter_contribution()`
   - `vectorized_steiner.compute_steiner_areas()` vs
     `SteinerHandler.get_total_area_contribution()`
   - `vectorized_steiner.compute_steiner_perimeter_gradient()` vs
     `SteinerHandler.compute_total_gradient_finite_difference()`
   - `vectorized_steiner.compute_steiner_area_jacobian()` vs
     `SteinerHandler.compute_area_gradients_finite_difference()`

4. **Gradient verification**: Compare vectorized analytical perimeter gradient against
   finite differences (already implemented in
   `PerimeterCalculator.verify_gradient_finite_differences`).

5. **Round-trip test**: Run one full optimize cycle with vectorized path, compare final
   perimeter and constraint violation against the original path run on the same input.

6. **Migration compatibility**: Run a detect → migrate → compile_arrays → optimize cycle.
   Verify that `compile_arrays()` correctly handles inactive VPs and post-migration state.

## Implementation Order

1. `PartitionArrays` dataclass + `compile_arrays()` method (foundation, including
   triple-point arrays from `SteinerHandler.triple_points`)
2. `compute_total_perimeter()` vectorized + test against `PerimeterCalculator`
3. `compute_perimeter_gradient()` vectorized + test against `PerimeterCalculator`
4. `compute_cell_areas()` vectorized + test against `AreaCalculator`
5. `compute_area_jacobian()` vectorized (finite-difference) + test against `AreaCalculator`
6. `compute_steiner_points()` analytical Fermat-Torricelli + validate against BFGS
7. `compute_steiner_perimeter()` + `compute_steiner_areas()` vectorized + test against
   `SteinerHandler`
8. `compute_steiner_perimeter_gradient()` + `compute_steiner_area_jacobian()` FD-based +
   test against `SteinerHandler`
9. Wire fully vectorized path into `PerimeterOptimizer` with fallback to original via flag
10. Fix callback overhead
11. End-to-end test: full optimize cycle with identical results
12. (Future) Analytical area Jacobian to eliminate FD loop for regular boundary triangles

## Key Invariants to Preserve

- `PartitionContour` VP objects remain the source of truth for mutations
- After optimization, `partition.set_variable_vector(result.x)` syncs back (once, at end)
- During the SLSQP evaluation loop, only flat arrays are read — no VP object access
- `compile_arrays()` is idempotent — can be called multiple times safely
- Active/inactive VP distinction is respected (inactive VPs excluded from arrays)
- Edge normalization convention: `edge[0] < edge[1]` (same as `VariablePoint`)
- Lambda convention: `x = λ * vertices[edge[0]] + (1-λ) * vertices[edge[1]]`
- Perimeter = sum of cell perimeters (each segment counted twice)
- Triple-point triangles excluded from boundary triangle arrays (vectorized_steiner handles them)
- Steiner area = void triangle area + corner triangle area per cell
- Steiner point = analytical Fermat-Torricelli (no BFGS); degenerates to obtuse vertex at ≥ 120°
- `SteinerHandler` used only in mutation mode; `vectorized_steiner` used in evaluation mode
