# Exporting the Partition for Molecular Simulations — Discussion Notes

**Status:** Discussion / planning. No implementation decisions made yet.
**Date:** 2026-05-07
**Context:** End goal is to use the optimised partition (Phase 1 + Phase 2 output)
as input to molecular dynamics simulations on the surface. Particles are placed
on the surface and need to be classified by which cell they belong to. The exact
shape of the cell boundaries matters because it determines how particles near
boundaries behave.

This document collects the export-strategy options we discussed and the open
questions that need answers before committing to a representation.

---

## What the partition data looks like internally

For reference when reasoning about export, this is what the post-Phase-2 HDF5
checkpoint actually stores (see `iteration_NNN_*.h5`):

| Quantity | Shape | Semantics |
|---|---|---|
| `vertex_labels` | `int[N_vertices]` | Single cell index ∈ {0,…,N-1} per mesh vertex. No fractional values, no ties. |
| `lambda_parameters` | `float[N_active_VPs]` | Per-VP barycentric position in [0, 1] on its mesh edge. λ=1 → at smaller-index endpoint. |
| `vp_edges` | `int[N_active_VPs, 2]` | Mesh edge each active VP lives on. |
| `indicator_functions` | `int[N_vertices, N_cells]` | Per-vertex one-hot indicator (redundant with `vertex_labels` but useful for FEM-style ops). |
| Triple-point bookkeeping | (in `steiner_handler`) | Per-triangle Steiner geometry where 3 cells meet. |

Every mesh triangle falls into exactly one of three categories:

1. **Pure interior:** all three vertex labels equal. Triangle is fully in that cell.
2. **2-cell boundary:** two labels match, one differs. Two of the triangle's
   three edges carry one VP each. The boundary inside the triangle is a single
   line segment connecting those two VPs.
3. **Triple-point:** all three labels differ. The boundary is three segments
   meeting at a Steiner point inside the triangle.

These three cases are exhaustive and unambiguous. Any export strategy is just
a different way of materialising this information for downstream consumers.

---

## Export representations considered

### Representation 1 — Face-labeled mesh (winner-takes-all)

Each triangle gets a single integer label (whichever cell owns the largest
area portion).

```
triangle_labels: int[N_faces]   # one cell per triangle
```

| | |
|---|---|
| **Particle placement** | Sample uniformly on the surface (random barycentric coords, weighted by triangle area). |
| **Classification** | `cell_of(point) = triangle_labels[triangle_containing(point)]` — O(1) once the triangle is known. |
| **Pros** | Trivially fast. Tiny dataset. Compatible with MD codes that already use mesh-aligned regions. No floating-point edge cases. |
| **Cons** | Boundary becomes a zigzag along mesh edges instead of the smooth optimised curve. A 51/49 boundary triangle is forced 100% to one cell. Resolution limited by `mesh_edge_length`. |
| **Best for** | MD where each particle needs a single cell ID and the optimisation is used only to specify *partition topology*, not boundary geometry. Most robust against any geometric edge cases. |

### Representation 2 — Vertex labels + per-edge VP barycentric

Store the canonical partition data as-is.

```
vertex_labels: int[N_vertices]
boundary_edges: list of (v_a, v_b, lambda)    # one VP per crossing edge
triple_points: list of (triangle_idx, cell_a, cell_b, cell_c, steiner_xyz)
```

Classification of a point at barycentric coords `(α, β, γ)` in triangle
`(va, vb, vc)`:

1. If all three vertex labels match → that cell, done.
2. If two labels match and one differs → the boundary is a single segment
   between the two VPs on the two boundary edges. Sign-test which side of
   that segment the point is on.
3. If all three differ → triple-point triangle: build the three boundary
   segments to the Steiner point, find which wedge the point is in.

| | |
|---|---|
| **Pros** | Reproduces the optimised boundary exactly (no zigzag). Compact (~1 int per vertex + 1 float per boundary edge ≈ 18k ints + 2.3k floats for double-torus 10-part). |
| **Cons** | Classification is O(1) but with branching and floating-point sign tests. Vulnerable to numerical edge cases — points exactly on a boundary segment, or boundary segments collinear with triangle edges (e.g. T11743 needle case), can flip cells under tiny perturbations. Triple-point triangles need separate code. |
| **Best for** | MD that needs the *exact* optimised boundary, where particles are unlikely to spend long residence times exactly on a boundary edge. |

### Representation 3 — Subdivided mesh (boundary-cut mesh)

Pre-process: for every 2-cell boundary triangle, cut it along its VP1–VP2
segment to produce two smaller triangles, each entirely in one cell. For every
triple-point triangle, cut it into three pieces around the Steiner point.

```
new_vertices = original_vertices + VP_positions + Steiner_positions
new_faces    = original_interior_faces + split_boundary_pieces
new_face_labels: int[N_new_faces]    # single label per face
```

| | |
|---|---|
| **Pros** | Combines (1) and (2): exact optimised boundary AND single-label-per-face semantics. No floating-point edge cases at runtime. Particle classification reduces to point-in-triangle, which has well-known robust algorithms. |
| **Cons** | Mesh size grows by ~`N_boundary_triangles` (a few thousand). One-time pre-processing cost. Some sliver triangles introduced when a VP is close to a vertex. |
| **Best for** | Most MD workflows. Default recommendation unless there's a specific reason to avoid it. |

### Representation 4 — P1 indicator-function interpolation

Each cell has indicator `u_k(v) ∈ {0, 1}` at every mesh vertex. Within a
triangle, P1-interpolate. Classify by argmax over `k`.

| | |
|---|---|
| **Pros** | Smooth, no branching. Already present in the HDF5 checkpoint. Very compact code. Naturally handles triple-points. |
| **Cons** | The boundary it defines is *not* the optimised boundary — it's the level set where two indicators tie at 0.5. For pure {0,1} indicators this is exactly Representation 1's zigzag. So you lose the optimisation work that placed VPs at non-trivial λ values. |
| **Best for** | Quick prototyping where exact boundary geometry is not critical, or as a fallback classifier when other representations fail numerically. |

### Representation 5 — Geodesic distance / heat-method classifier

For each cell, compute a geodesic-distance (or heat-method) field on the
surface from the cell's interior. Classify by argmin distance.

| | |
|---|---|
| **Pros** | Smoothest possible classification. No boundary edge cases. Naturally extends off the mesh. |
| **Cons** | Expensive to compute (heat method works well but takes minutes). Doesn't exactly match the optimised partition — boundary shifts toward median geodesic between cells. Storage is `N_cells × N_vertices` floats. |
| **Best for** | Cases where you want very smooth boundaries and are willing to deviate slightly from the optimisation result. Probably overkill for an initial export. |

---

## Architectural recommendation (representation-agnostic)

Decouple into three pieces:

1. **Snapshot layer.** Save the post-Phase-2 state in a representation-agnostic
   form: `vertex_labels`, `boundary_edges` with λ values, `triple_points` with
   cells and Steiner positions. This is what the HDF5 already contains; treat
   it as the canonical record of the optimisation result.
2. **Exporter / classifier layer.** Generate one or more of Representations
   1–5 on demand from that snapshot. The choice of representation can be
   revisited without redoing the optimisation.
3. **MD-side classifier API.** A small function `cell_of(point) → int` that
   consumes whichever representation is chosen. This is the interface the MD
   code touches.

Why this matters: the MD requirements are still being defined. We don't want
to bake assumptions into the optimisation output that lock us in.

---

## Open questions to sharpen the recommendation

These need answers before committing to a representation. We should revisit
when the MD side is more concrete.

1. **Where do particles live?**
   - On the surface itself (2D manifold)?
   - In a thin shell around it (3D)? If 3D, do we need to project to the
     surface for classification?
2. **Do particles move during simulation, and can they cross boundaries?**
   - Static placement: classifier is cold, performance is irrelevant.
   - Dynamic: classifier is hot. May need a representation that supports fast
     incremental updates as a particle moves between triangles.
3. **What input format does the MD code accept?**
   - Face-labeled mesh? `(position, cell_id)` pairs? Signed-distance field?
   - LAMMPS regions / GROMACS index files / custom HDF5?
4. **Particle density / count.**
   - A handful per cell, or millions? Determines how much we should pre-compute
     in the classifier.
5. **Deterministic boundary tie-breaking?**
   - If a particle lands exactly on a boundary segment, does the assignment
     need to be reproducible? Some MD setups need a fixed tie-break rule.
6. **Smoothness scale relative to interaction range.**
   - If MD interaction range >> mesh edge length, boundary smoothness is
     invisible (zigzag fine).
   - If comparable, smooth boundaries matter.
7. **Surface representation outside the mesh.**
   - Does the MD code need to know the underlying smooth surface (for forces
     normal to the surface, etc.) or just the discrete mesh?
8. **Topology preservation across simulation.**
   - Is the partition fixed for the whole MD run, or does it evolve?
   - If fixed, we can pre-compute aggressively.
   - If evolving, we need a re-classification strategy after each topology
     change.

---

## Tentative initial recommendation

If the user just wants to start experimenting and refine later:

- Default to **Representation 3** (subdivided face-labeled mesh) as the export
  format.
- Snapshot in Representation 2 form for full reproducibility.
- Add the classifier as a separate module so changing representations later is
  a one-file edit, not a refactor.
- Defer the smoothness / boundary-quality discussion to a separate cleanup
  step (post-Phase-2 snap-and-clean of stranded VPs, mesh refinement, etc.) —
  the export representation should not depend on whether those cleanup steps
  have been run.

This is contingent on the answers to the open questions above. Revisit when
the MD pipeline requirements are clear.

---

## Cross-references

- `iteration_NNN_*.h5` HDF5 schema — see `src/pipeline/io.py`,
  `PipelineOrchestrator.export_checkpoint()`.
- Vertex-label / VP semantics — see `CLAUDE.md` "The λ Convention" and
  `src/partition/contour_partition.py`.
- Triple-point geometry — see `src/partition/steiner_handler.py` and
  `docs/reference/TOPOLOGY_SWITCH_METHODOLOGY.md`.
- Smoothness improvements (separate concern) — see refinement-loop tuning
  in `docs/reference/IPOPT_REFINEMENT_QUALITY.md`.
