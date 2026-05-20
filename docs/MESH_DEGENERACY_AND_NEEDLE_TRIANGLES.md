# Mesh Degeneracy and Needle Triangles in Marching-Cubes Meshes

**Status:** Problem statement + design spec for a future cleanup tool.
**Date:** 2026-05-07 (revised 2026-05-07 with data-vs-rendering clarification
and mesh-quality philosophy).
**Audience:** Future agents / developers picking up the mesh-cleanup work.
**Related:** `docs/TOPOLOGY_SWITCH_METHODOLOGY.md`, `docs/type1_several_VPs_issue.md`,
`docs/MD_SIMULATION_EXPORT_NOTES.md`.

---

## TL;DR

Marching cubes on the double-torus implicit function occasionally produces tiny
clusters of near-coincident vertices (within ~`3×10⁻⁴` of each other) connected
by extreme-aspect-ratio "needle" triangles. After Phase 1 + Phase 2 optimisation,
these needles end up with **boundary segments lying along their own mesh edges**
because IPOPT pushes VPs onto the cluster vertices. This produces:

1. Visualisation ambiguity (a rendering bug — see §1.5; this is *not* a data
   problem, just a renderer that fails on degenerate polygons).
2. Mildly jagged optimised cell boundaries that follow mesh edges instead of
   cutting smoothly through triangles, with sharp kinks at stranded-VP
   vertices. This *is* in the data and matters for MD use.
3. Stranded VPs at low-valence vertices that no Type 1 trigger catches.

**Important distinction (§1.5):** within any single boundary triangle the data
always carries a clean single straight boundary segment between two VPs.
There is *never* a hidden zigzag inside a triangle, no matter how degenerate
the triangle is. The kinks only live at the seams *between* triangles. The
pink/white rendering pathology of T11743 is purely a renderer artefact.

**Short-term mitigation:** higher marching-cubes resolution (`n_grid_z` is the
highest-leverage knob for the double torus), more outer refinement iterations,
and the rendering Config-3 fix described in §5.

**Long-term solution:** a one-shot mesh-cleanup pass that runs after
`marching_cubes` and *before* Phase 1, welding near-coincident vertex clusters
into single vertices and removing the resulting degenerate triangles. This
document specifies that tool in §6. The recommended pragmatic strategy is the
three-layer defence in §4: tune marching-cubes parameters → run the cleanup
pass → only then consider switching to a quality-controlled surface mesher.

---

## 1. Problem statement

### 1.1 Symptom (visualisation)

When visualising a Type 1 component with
`scripts/visualize_type1_vertex_collapse.py`, certain needle-shaped boundary
triangles render as either:

- Both cells claiming the entire triangle (one cell's color drawn on top of
  the other → e.g. T11743 in component 37, BEFORE state of run
  `run_20260417_175907_..._iteration_001_20260506_093157.h5`), or
- Both cells claiming a zero-area collinear "polygon", leaving the triangle
  uncoloured / white (same triangle, AFTER state).

The visualisation pathology is a *renderer bug*: the underlying partition
data in these triangles is well-defined and exact (a single straight
boundary segment between two VPs, with one cell owning the rest), but the
renderer's polygon-construction logic fails when that segment lies along a
mesh edge of the triangle. See §1.5 for the precise data-vs-rendering
distinction. The visualisation symptom is a useful *signpost* pointing to
the underlying mesh degeneracy described in §1.2, even though the rendered
triangle itself isn't a faithful picture of the data.

### 1.2 Root cause (mesh)

For the double torus implicit function

```
f(x,y,z) = (x(x-1)²(x-2) + y²)² + z² − c = 0,    c = 0.03
```

with the current grid `(n_grid_x, n_grid_y, n_grid_z) = (80, 60, 30)` over the
bounding box `((-0.5,2.5), (-1,1), (-0.3,0.3))`, the surface passes through
some grid cells very close to a corner, almost grazing it. Three of the cube's
twelve grid edges then carry surface-crossings within ~`10⁻⁴` of each other,
and `marching_cubes` outputs three near-coincident vertices.

Concrete example documented in `iteration_001_20260506_093157.h5`:

| Vertex | Position (x, y, z) | Cell |
|---|---|---|
| V5656 | (0.519417, −0.519134, −0.146809) | 2 |
| V5657 | (0.519417, −0.518987, −0.146915) | 2 |
| V5968 | (0.519730, −0.518987, −0.146809) | 2 |

All three are within `~3×10⁻⁴` of each other in every axis. The triangulation
fans needle triangles from this cluster out to the surrounding "normal-spaced"
mesh vertices (V5378, V5658, V5969, V5970), each of which is `~3×10⁻²` away —
two orders of magnitude more. The 6 triangles in the 1-ring of V5656 have
aspect ratios:

| Triangle | Face | Labels | Aspect ratio | Area |
|---|---|---|---|---|
| T11124 | (5657, 5656, 5378) | (2, 2, 2) | **176** | 2.7e-6 |
| T11128 | (5378, 5656, 5658) | (2, 2, 8) | 1.56 | 3.1e-4 |
| T11742 | (5657, 5968, 5656) | (2, 2, 2) | 1.90 | 2.9e-8 |
| T11743 | (5656, 5968, 5969) | (2, 2, 8) | **90** | 2.8e-6 |
| T11744 | (5970, 5656, 5969) | (8, 2, 8) | 10.6 | 4.5e-5 |
| T11745 | (5658, 5656, 5970) | (8, 2, 8) | 1.58 | 3.1e-4 |

Three of the six are needles (T11124, T11743, plus T11742 which is a tiny
all-cluster triangle invisible at most zoom levels).

### 1.3 What goes wrong after Phase 2

IPOPT pushes VPs to λ = 0 or 1 when that minimises perimeter, snapping them
onto mesh vertices. In the cluster region this happens routinely:

- Five VPs in the 1-ring of V5656 (VP712, VP739, VP740, VP741, VP789) all sit
  exactly on mesh vertices to within `~10⁻¹⁰`.
- Four converge onto V5656; one (VP789) lands on V5968.
- For T11743 the boundary segment runs from VP740 (= V5656) to VP789 (= V5968)
  — a segment that is literally the V5656–V5968 mesh edge itself. The
  remaining triangle vertex V5969 is on the cell-8 side of this edge, so cell
  8 owns the entire triangle interior.

The Type 1 migration system, which exists to absorb collapsed VPs at vertices,
does *not* fire on V5968 because V5968 has only one VP collapsed onto it —
below the ≥3-VP threshold required by the current Type 1 trigger criterion.
So the configuration is stable: the optimiser has no descent direction
without changing topology, and the migration system has no trigger to change
the topology.

### 1.4 Why this matters for downstream use

The user's downstream goal is molecular simulation on the partitioned surface
(see `docs/MD_SIMULATION_EXPORT_NOTES.md`). For that:

- Boundaries that lie along mesh edges are *jagged* (piecewise mesh-edge) —
  far from the smooth optimal boundary the optimisation is supposed to find.
- VPs accumulating at mesh vertices cause sharp kinks in the boundary curve.
- Both effects hurt the smoothness metric needed for clean particle
  classification near boundaries.

This is independent of any rendering/visualisation issue — it's a
property of the post-optimisation partition itself.

### 1.5 Data vs rendering: where the actual jaggedness lives

A point of confusion worth pinning down explicitly, because the visualisation
*looks* much worse than the underlying data is.

**Invariant of the data model.** Every 2-cell boundary triangle in the
partition carries **exactly two VPs**, and the boundary inside the triangle is
**always a single straight segment** between those two VPs. There are never
intermediate VPs inside a single triangle. So the boundary curve never
zigzags within a triangle — it can only kink at the *seams* where it crosses
from one triangle into the next.

**Concrete example: T11743 (cluster needle, component 37 in the reference
solution).** Labels `(2, 2, 8)`, two boundary VPs.

| State | VP positions | Boundary segment | Comment |
|---|---|---|---|
| BEFORE migration | VP740 at V5656; VP789 at V5968 | V5656 → V5968 (full mesh edge) | Both VPs collapsed onto distinct triangle vertices; segment lies along the V5656–V5968 mesh edge. Length `3.5×10⁻⁴`. |
| AFTER migration | VP2313 at midpoint(V5656, V5968); VP789 at V5968 | midpoint(V5656, V5968) → V5968 (half edge) | V5656 has flipped to cell 8, so the V5656-side half of the edge is now interior to cell 8. Length `1.7×10⁻⁴`. |

In both states the boundary in T11743 is a single straight segment lying on
*one* of T11743's mesh edges. There is no `V5656 → another VP → another VP →
V5968` zigzag — that pattern is structurally impossible inside a single
triangle.

**What the visualisation made it look like.** In the BEFORE state the renderer
draws T11743 entirely pink (cell 2 wins a depth fight); in the AFTER state it
draws T11743 white (both cells produce zero-area collinear polygons). Both are
fallout from `compute_cell_portion_in_triangle_simple` failing the side test
when the boundary segment lies on a mesh edge — they are *renderer bugs*, not
hidden zigzag in the partition. Area conservation is exact at machine
precision in both states (`~10⁻¹⁵` relative error; vectorised and scalar
calculations agree to `~10⁻¹³`).

**Where the real geometric jaggedness is.** Across the broader 1-ring around
the V5656 cluster, the boundary curve does have several sharp corners. The
optimised boundary path is something like:

```
... V5378 → midpoint(V5378,V5656) → midpoint(V5656,V5657) → midpoint(V5656,V5968) → V5968 ...
     ↑                ↑                       ↑                       ↑               ↑
   stranded       triangle seam           triangle seam           triangle seam     stranded
   VP at vertex                                                                     VP at vertex
```

Five kinks, two at stranded-VP vertices, three at triangle seams along the
cluster. Each individual segment between two of those points is straight.
This is the genuine geometric cost of the cluster after optimisation: kinks
at the cluster's "exit" vertices, not zigzag inside any single triangle.

**Practical implication for downstream code.** Any classifier that reduces
to point-in-triangle on a face-labelled mesh — Representation 1
(face-labeled, winner-take-all) or Representation 3 (subdivided mesh with
boundary cuts) in `docs/MD_SIMULATION_EXPORT_NOTES.md` — gets unambiguous
answers everywhere, including in the cluster region. The data is clean and
exhaustive: every point on the surface lies in exactly one triangle, and every
triangle has a defined ownership. Only classifiers that integrate *along the
boundary curve itself* (e.g. line tension, surface energy on the boundary)
would feel the kinks, and they would still see a piecewise-linear curve —
just with sharper-than-typical corners at the few points per cluster.

---

## 2. Available marching-cubes parameters (today)

Configured in YAML under `surface.<surface_name>` and passed to the provider's
`__init__`. For implicit surfaces:

| Parameter | Where | Effect |
|---|---|---|
| `n_grid_x`, `n_grid_y`, `n_grid_z` | YAML / `ImplicitSurfaceProvider.__init__` | Grid resolution. Higher → smaller cells → fewer "grazing" cells. |
| `n_grid_x_increment`, `n_grid_y_increment` | YAML | Per-refinement-level increments for x and y. z scales with x via `set_resolution`. |
| `c` | YAML / surface-specific subclass | Implicit-function constant. Shifts the level set. |
| `bounding_box()` | hard-coded in surface subclass (`src/surfaces/double_torus.py` etc.) | Domain extent. Not currently exposed via YAML. |

There is **no current support for**:

- Grid origin offset / jitter.
- Mesh post-processing (vertex welding, edge collapse, decimation).
- Per-axis increment tuning beyond x ↔ y ↔ z(via x ratio).

---

## 3. Short-term mitigations (no new code)

In order of effort/leverage:

1. **Bump `n_grid_z`** — the lowest-resolution axis for the double torus and
   the direction where the surface is thinnest. Try 60 (currently 30). One
   YAML edit.
2. **Bump `n_grid_x` and `n_grid_y`** proportionally — try `(120, 90)`. Phase
   1 cost goes up roughly with `n_grid_x · n_grid_y · n_grid_z`; Phase 2 cost
   grows with mesh boundary triangle count.
3. **Run more outer refinement iterations** (`max_iterations: 8` or higher in
   the refinement YAML). Stranded VPs sometimes get resolved by cascading
   migrations elsewhere.
4. **Tighten / loosen `boundary_tol`** to fire migrations more or less
   aggressively. `0.02` for a final cleanup pass after the main optimisation
   has converged.
5. **Adjust `c` slightly** (e.g. `0.029` or `0.031`). A grid-alignment lottery
   — sometimes a different value avoids the specific corner-grazing that
   produces a cluster, but a different cluster may appear elsewhere.

These reduce the *frequency* of clusters. They do not eliminate them — for
any grid resolution, some cells will graze the surface and produce
near-coincident vertices.

---

## 4. Mesh-quality philosophy: uniform vs well-shaped, and the three-layer defence

A natural impulse on seeing the cluster pathology is "the right answer is a
strictly uniform mesh." That's slightly stronger than what is actually
needed. The partition algorithm doesn't care whether triangle sizes are
identical across the surface — it cares about the absence of pathological
shapes locally. Re-stated as positive criteria, a mesh is "good enough" for
the partition pipeline if every triangle satisfies:

1. **Bounded aspect ratio.** Longest edge / shortest edge ≲ 5 (heuristic; the
   exact threshold depends on the IPOPT scaling, but anything below 10 is
   typically safe).
2. **No near-coincident vertex clusters.** No two distinct mesh vertices
   closer than some `ε_v` that is small compared to the typical edge length
   in that region.
3. **No degenerate triangles.** Triangle area above some `A_min`.

A mesh that meets these three conditions can have varying triangle sizes
(denser in high-curvature regions, coarser elsewhere) and the partition
algorithm will be perfectly happy. So the target is **well-shaped**, not
**uniform**. This matters for cost: producing a uniformly-fine mesh
everywhere is expensive, but producing a mesh with bounded aspect ratio and
no clusters is comparatively cheap, and it is the property that actually
fixes the issue.

### 4.1 The three-layer defence

For implicit-surface meshes (double torus, Banchoff-Chmutov), there are three
layers of defence against the cluster pathology, in increasing power and
implementation cost:

| Layer | Effort | Effectiveness | What it looks like in practice |
|---|---|---|---|
| **(L1)** Tune marching-cubes parameters | minutes | Reduces *frequency* of clusters but cannot eliminate them | Edit `n_grid_x/y/z`, `c`, bounding box in the YAML; see §3 |
| **(L2)** Post-process: vertex welding + sliver removal (cleanup tool) | days to implement | *Eliminates* the problem on existing marching-cubes outputs | The mesh-cleanup tool specified in §6 |
| **(L3)** Switch to a quality-controlled surface mesher | weeks; new dependency | Best-shaped meshes, with explicit minimum-angle / aspect-ratio guarantees built into the mesher | CGAL `Surface_mesh_generation`, Geogram, fTetWild + extract; replaces the marching-cubes path entirely for implicit surfaces |

Parametric meshes (`TorusMeshProvider`, `EllipsoidMeshProvider`) are already
structurally well-shaped and don't need any of these layers; the cluster
pathology is specific to the marching-cubes pipeline.

### 4.2 Pragmatic recommendation

For the immediate downstream MD pipeline:

1. Apply L1 first (lowest effort). Tune `n_grid_z` upward and try a couple of
   `c` values. This often pushes the worst clusters out of the geometry of
   interest.
2. Then implement L2 (the cleanup tool in §6). After this step, the
   marching-cubes output is *effectively* well-shaped — clusters get welded,
   slivers get removed, and IPOPT no longer has the opportunity to produce
   edge-aligned boundary segments. This is the single highest-leverage
   change in the whole defence stack.
3. Only invest in L3 if L1 + L2 still leaves artefacts that matter for the
   particular MD application. Replacing the mesher is a substantial new
   dependency (CGAL/Geogram are non-trivial to package) and should be
   considered only if we hit a wall.

L1 + L2 together should be sufficient for all current surfaces in the
codebase. L3 is an escape hatch.

### 4.3 What L2 leaves behind (post-cleanup residual jaggedness)

Even after vertex welding and sliver removal, the boundary curve in the
optimised partition can still be "kinked" — it's piecewise-linear by
construction (one straight segment per boundary triangle). What L2
*eliminates* is the special case where a triangle is so degenerate that its
two VPs collapse onto two distinct mesh vertices and the boundary segment
ends up *aligned* with a mesh edge. After L2 there are no needle triangles
and no clusters, so the IPOPT solution lives in a regime where the side test
is well-behaved and boundary segments cut cleanly through triangles.

What L2 does *not* do: smoothing of the piecewise-linear boundary curve.
That's a Phase 2.5 / post-processing job, completely separate from mesh
quality. If MD work eventually needs a `C¹`-smoothed boundary representation
that's its own problem to solve and belongs in `docs/MD_SIMULATION_EXPORT_NOTES.md`.

---

## 5. The Config-3 rendering and migration extensions (already specified)

The rendering pathology in T11743 has a clean fix that detects the
"both-VPs-at-triangle-vertices" configuration and falls back to a label-based
decision. This is documented in detail in the conversation that produced this
file (see "Your proposed rule extension — let's formalise it" in the
agent-transcript). Summary:

- **Config 3 (rendering):** in `compute_cell_portion_in_triangle_simple`, if
  both VPs evaluate to two distinct triangle-vertices of `T`, then the
  boundary segment lies along an edge of `T`, and the third triangle vertex's
  label decides which cell owns the triangle interior. This generalises the
  existing "coinciding VPs" fallback (Config 2). Small contained patch.
- **Config 3 (migration):** the same detection criterion can be a *new*
  Type 1-like trigger that fires even at low-valence vertices when two VPs
  on adjacent triangle edges have both collapsed to triangle vertices. This
  is the smoothness fix.

These fixes operate on the *post-mesh* state. They handle the boundary-state
pathology after the mesh is already built. They are complementary to — and
do not replace — the mesh-cleanup tool described below.

---

## 6. Long-term solution: a mesh-cleanup tool (layer L2)

### 6.1 Purpose

Detect and remove *mesh-level* degeneracies in the marching-cubes output,
*before* Phase 1 runs. This eliminates the root cause: needle triangles
spawned by near-coincident vertex clusters never get into the optimisation
pipeline, so IPOPT never has the opportunity to push VPs onto them, so the
boundary never collapses onto a mesh edge.

### 6.2 What constitutes a "degenerate" mesh feature

For the purposes of this tool, degenerate features are:

1. **Vertex clusters:** groups of two or more vertices within distance `ε_v`
   of each other (where `ε_v` is small enough not to merge legitimately
   distinct vertices but large enough to catch IPOPT-relevant clusters).
2. **Short edges:** edges shorter than `ε_e` (typically `ε_e ≈ ε_v`).
3. **Sliver / needle triangles:** triangles with aspect ratio exceeding
   `α_max`, or with smallest angle below `θ_min`.
4. **Zero-area triangles:** area below `A_min`. Can arise after vertex
   merging.

These are not independent — clusters cause short edges cause needles cause
near-zero areas. Cleaning up clusters typically resolves the others.

### 6.3 Recommended algorithm: vertex welding + degenerate-triangle removal

The cleanest approach is a two-stage pass:

**Stage A — vertex welding via union-find on a KD-tree:**

```
1. Build a KD-tree on mesh.vertices.
2. For every vertex v, query all neighbours within radius ε_v.
3. Use union-find to compute equivalence classes of mutually-close vertices.
4. For each class C, choose one canonical vertex:
       canonical(C) = centroid of C (or, alternatively, the lowest-index
       vertex in C — both are valid; centroid gives slightly smoother result
       but lowest-index makes the operation reproducible and bit-stable).
5. Build a remap array: remap[v] = canonical_index(class_of(v)).
6. Replace every vertex index in mesh.faces by remap[that_index].
```

**Stage B — degenerate-triangle removal:**

```
For every face (a, b, c):
    if a == b or b == c or c == a:
        drop the face (it has merged into a line or a point)
    else if triangle area < A_min:
        drop the face (numerically degenerate)
    else:
        keep the face
```

**Stage C — vertex compaction:**

```
1. Find the set of vertex indices actually referenced by surviving faces.
2. Build a compaction map old_index -> new_index.
3. Apply compaction to mesh.faces.
4. Trim mesh.vertices to only the referenced subset.
```

After this, return a fresh `TriMesh(verts, faces)` with no clusters, no
zero-area triangles, and a smaller (but still manifold) face list.

**Topology preservation:** vertex welding can theoretically change topology
(e.g. merging two vertices on opposite sides of a thin sheet would create
a non-manifold edge). For the implicit-surface meshes this codebase deals
with, this risk is low because:

- The implicit function has a single zero level set.
- `ε_v` is much smaller than the surface's minimum feature size (the tube
  thickness of the double torus, ~`2·sqrt(c) ≈ 0.35`, is `~10³ ε_v`).
- Marching cubes already produces a manifold output (each grid edge
  contributes at most one vertex).

A safety check should still be performed (count edges with > 2 incident
faces; warn if any are produced).

### 6.4 Choosing the tolerance `ε_v`

The cluster scale we measured is `~3×10⁻⁴`. The surface feature scale (e.g.
the V5969 vertex sits `~3×10⁻²` away from the V5656 cluster) is `~10²` larger.
Any `ε_v` in the geometric mean range is safe.

A robust default is

```
ε_v = max(1e-6, 1e-3 · median_edge_length / 100)
```

or expressed as a fraction of the bounding box diameter:

```
ε_v = 1e-5 · bbox_diameter
```

For the double torus with bbox diameter `~3.4`, that gives `ε_v ≈ 3.4×10⁻⁵` —
about an order of magnitude smaller than the cluster scale, so it would not
catch *every* cluster but would catch the worst ones. To be more aggressive,
use `1e-4 · bbox_diameter ≈ 3.4×10⁻⁴`, which catches the documented cluster
exactly. The right default is probably parameter-dependent and should be
exposed.

### 6.5 Where in the codebase

**New module:** `src/mesh/mesh_cleanup.py`

Functions to expose:

```python
def weld_close_vertices(
    mesh: TriMesh,
    tol: float,
    *,
    canonical: str = "centroid",   # or "lowest-index"
    min_area: float = 1e-12,
    return_stats: bool = False,
) -> TriMesh | tuple[TriMesh, dict]:
    """
    Weld vertex clusters into single vertices, then drop degenerate faces.

    Args:
        mesh:        Input TriMesh (typically just out of marching_cubes).
        tol:         Welding tolerance ε_v in world units.
        canonical:   How to pick the merged vertex position.
        min_area:    Drop triangles with area below this threshold.
        return_stats: If True, also return a dict with operation statistics.

    Returns:
        Cleaned TriMesh (and optionally statistics).
    """
    ...

def detect_needle_triangles(
    mesh: TriMesh,
    *,
    aspect_max: float = 30.0,
    min_angle_deg: float = 2.0,
) -> np.ndarray:
    """Return indices of triangles flagged as needle/sliver."""
    ...

def cleanup_report(mesh_before: TriMesh, mesh_after: TriMesh) -> str:
    """Human-readable summary: vertex/face counts, removed clusters,
    aspect-ratio histogram before/after."""
    ...
```

**Integration point 1 (recommended): provider build hook.**
In `src/surfaces/implicit.py`, extend `ImplicitSurfaceProvider.__init__` with
an optional `weld_tol: Optional[float] = None` parameter. After
`marching_cubes` returns and the world-space translation is applied, if
`weld_tol is not None`, call `weld_close_vertices(mesh, weld_tol)` before
returning. Surface YAMLs gain an optional `weld_tol` field. Default behaviour
is unchanged (no welding).

**Integration point 2 (alternative): standalone CLI.**
A script `scripts/cleanup_mesh.py` that takes an input mesh (HDF5 or OBJ),
runs `weld_close_vertices`, and writes a cleaned mesh. Useful for debugging
existing solutions without re-running the implicit-surface pipeline. Minimal
loader needed; can read from existing `.h5` solution files.

**Integration point 3 (diagnostic only): testing harness.**
A diagnostic script `testing/diagnose_mesh_quality.py` that takes a base
solution `.h5` and prints needle-triangle statistics, cluster counts, and
the worst N triangles by aspect ratio. Read-only; no mesh modification.
Useful as a pre-flight check before deciding whether to run the welding pass.

### 6.6 Tests / validation

The cleanup tool needs validation against:

1. **Synthetic test cases.** Create a tiny mesh by hand with a known cluster
   (three vertices at `(0, 0, 0)`, `(0, 0, 1e-9)`, `(1e-9, 0, 0)`, and
   surrounding triangles). Verify weld merges them into one and removes the
   resulting zero-area triangles.
2. **The reference double-torus mesh.** Run on the mesh underlying
   `iteration_001_20260506_093157.h5`. Expected outcome:
   - Cluster (V5656, V5657, V5968) is welded into a single vertex.
   - T11124, T11742, T11743 are eliminated or reduced (T11742 should vanish
     entirely, T11124 and T11743 may survive as degenerate triangles to
     remove).
   - Vertex count drops by ~tens (one per cluster).
   - Face count drops by ~tens (welded triangles).
3. **Regression on Phase 1.** Run Phase 1 on the cleaned mesh from the same
   bounding box / `c` / grid resolution. Final perimeter should be no worse
   than on the un-cleaned mesh, and ideally slightly better. Confirm the
   partition topology (number of triple points, total perimeter) is sensible.
4. **Manifold preservation check.** After welding, every edge should be
   shared by exactly 2 triangles (closed manifold) or 1 (boundary, but our
   surfaces are closed). Any edge with 3+ incident triangles indicates a
   topological merge that shouldn't have happened — log a warning and
   probably abort the welding (with a detailed message identifying the
   problematic vertices).

### 6.7 Open design questions for the implementer

The following decisions are deferred to whoever picks this up:

a) **Tolerance schema.** Is `weld_tol` a single absolute value, a fraction of
   bounding-box diameter, or a fraction of median-edge-length? Probably
   accept any of those (with sensible heuristics) and have one configuration
   point.

b) **Per-cluster behaviour.** Should the canonical vertex be the centroid
   (smoother result, but the cluster's centroid may not lie exactly on the
   surface), the lowest-index vertex (deterministic and exactly on-surface),
   or the vertex closest to the cluster centroid (smoothest of the three)?

c) **Order of operations relative to refinement levels.** The pipeline does
   multi-level refinement (`set_resolution` is called at each level). Should
   welding be redone at every level, or only at the final mesh? If the
   welding tolerance scales with the grid spacing, it should be re-applied
   per level.

d) **Interaction with the Phase 2 base-solution path.** The Phase 1 HDF5
   stores `vertices` and `faces`. If welding changes those, any pre-existing
   Phase 2 checkpoint that referenced the un-cleaned mesh must be reloaded
   or invalidated. The simplest policy: welding is *opt-in*, and it changes
   the mesh signature (e.g. via a hash in the run directory name) so
   pre-cleanup runs are clearly distinguished.

e) **Implicit vs parametric surfaces.** Parametric surfaces (`TorusMeshProvider`,
   `EllipsoidMeshProvider`) generally don't have this problem — their meshes
   are structured. The welding hook should still be available there for
   completeness but its default should be "off."

---

## 7. Cross-references and follow-up work

| Issue | Location | Status |
|---|---|---|
| Type 1 ≥3-VP threshold blind spot | `src/migration/migration_detector.py::detect_type1_triggers` | Documented in `docs/type1_several_VPs_issue.md`. Not yet addressed. |
| Triple-point safety guard | same | Implemented (commit `f9b...` series). |
| Rendering "degenerate boundary segment" fallback (Config 2 — coinciding VPs / both VPs collapsed onto the same vertex) | `scripts/visualize_type1_vertex_collapse.py::compute_cell_portion_in_triangle_simple` | Implemented (commit `ac8f90d`). |
| Rendering Config 3 (boundary segment along a triangle edge: VPs at two distinct triangle vertices) | same | **Pending.** Specified above in §5. |
| Migration Config 3 (low-valence trigger when boundary segment aligns with a mesh edge) | `src/migration/migration_detector.py` | **Pending.** Specified above in §5 (and Group B option 3 in the agent transcript). |
| Mesh cleanup tool (L2 in §4) | `src/mesh/mesh_cleanup.py` (does not exist) | **Pending.** Specified in §6. |
| Quality-controlled surface mesher (L3 in §4) | external — would replace `ImplicitSurfaceProvider.build()` | Not started; only consider if L1+L2 are insufficient. |
| MD export representation | `docs/MD_SIMULATION_EXPORT_NOTES.md` | Discussion only, deferred. |

---

## 8. Implementation order recommendation

If a future agent picks up this work, a reasonable order is (mapping back to
the L1/L2/L3 layers in §4):

1. **(L1, immediate)** Tune `n_grid_z` and `c` in
   `parameters/double_torus_10part.yaml` and re-run the reference solution.
   Capture the cluster count for the resulting mesh as a baseline. No code
   changes; minutes of effort. Often pushes the worst clusters out of the
   region of interest.
2. **(L2, diagnostic only first)** Implement `detect_needle_triangles` and a
   `testing/diagnose_mesh_quality.py` that reports cluster statistics on
   existing meshes. No mesh modification yet. This validates the choice
   of tolerance against real data.
3. **(L2, welding implementation)** Implement `weld_close_vertices` and unit
   tests on synthetic meshes (see §6.6).
4. **(L2, reference run validation)** Apply welding to the double-torus
   reference mesh; confirm clusters disappear (§6.6 case 2) and Phase 1
   still converges with sensible results.
5. **(L2, opt-in via YAML)** Wire `weld_tol` through
   `ImplicitSurfaceProvider` and the YAML loader. Default off.
6. **(L2, default-on rollout)** Once validated on multiple surfaces,
   consider making welding default-on for all implicit surfaces (with a
   sensible bounding-box-fraction-based tolerance).
7. **(Defensive in-pipeline fixes — orthogonal to mesh cleanup.)**
   Independently of L2, two narrower in-pipeline fixes are still worth
   keeping in the toolbox for cases where L2 is opt-in and disabled, or for
   meshes where some clusters slip through the welding tolerance:
   - **Rendering Config 3 fix** (§5). One-line rendering fallback. Cheap
     and contained; recommended as a permanent visualisation safety net.
   - **Migration Config 3 / low-valence guarded migration** (§5). Larger
     code change in `migration_detector`; only worth investing in if L2
     alone leaves problematic stranded VPs.
8. **(L3, escape hatch)** Switch to a quality-controlled surface mesher.
   Substantial new dependency (CGAL / Geogram / fTetWild). Only consider
   after L1 + L2 are deployed and shown to be insufficient for the MD
   pipeline's smoothness requirements.

Steps 1–6 are the core deliverable. Steps 7 and 8 are independent
follow-ups.

---

## 9. References

- Bogosel & Oudet (2017), "Partitions of Minimal Length on Manifolds", §5.3
  for the double-torus example.
- scikit-image marching-cubes documentation (no built-in degenerate-vertex
  filtering).
- Standard mesh-processing references (e.g. Botsch et al., *Polygon Mesh
  Processing*) for vertex-welding and edge-collapse algorithms.
- Conversation transcript that produced this document:
  `agent-transcripts/8b940734-41d9-4a16-95ab-4e9d2fc63a1a` — see the
  diagnostic outputs for component 37 and the discussion of the Config 3
  rule.
