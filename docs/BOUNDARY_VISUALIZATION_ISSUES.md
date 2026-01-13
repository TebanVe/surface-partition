# Boundary Visualization Issues (Type 1 Migration)

## Observations
- After Type 1 migration, the VP marker moves, but region boundaries often do not update to the new VP edge.
- Attempts to change boundary polygon construction produced artifacts: gaps/holes, stray disconnected patches, or unchanged boundaries.
- Interior triangles remain fine; the issue is localized to boundary triangles.

## What We Tried
1) Rebuilding topology after switches
   - Rebuilt `triangle_segments`, `boundary_segments`, and `segment_crossing_cache` after migration.
   - VP selection filtered to target region and non–triple-point VPs.
2) Changing boundary polygon construction
   - Variants that used only boundary points (VPs/crossings) and skipped or filled triangles when points < 3.
   - Variants that added static-label vertices to close polygons.
   - None yielded stable, gap-free boundaries that also moved with the VP.
3) Removing reliance on `indicator_functions`
   - Attempted to avoid static vertex labels for boundary geometry; fell back to labels to close polygons; still led to holes or unchanged boundaries.
4) Region rendering scope
   - Rendering all regions vs. target-only, with and without zoom and highlighting.

## Current Difficulties
- Boundary triangles frequently have only 1–2 boundary points (VPs/crossings). Naive “<3 → skip or fill by labels” either leaves holes or freezes the boundary.
- Using static vertex labels from `indicator_functions` to fill boundary triangles pins the boundary to its pre-migration state.
- Using only boundary points (VPs/crossings) without robust closure logic leaves gaps or produces disconnected polygons.
- Visual results remain inconsistent: boundaries not updating, gaps, and artifacts.

## What We Have Not Solved
- A robust, gap-free boundary polygon builder that:
  - Uses current VP/crossing geometry so the boundary moves with migrated VPs.
  - Gracefully handles the common case of only 1–2 boundary points in a triangle without introducing holes or reverting to static-label fills.

## Notes
- Triangle boundary/interior classification after migration is VP-based (`rebuild_triangle_segments_from_current_vps` and `get_triangles_by_cell_involvement` when `use_vp_based=True`), so boundary membership is dynamic and correct.
- Vertex labels from `indicator_functions` are static; safe only to identify which triangle vertices belong to a cell, but not to decide boundary geometry after migration.
- The main remaining challenge is constructing boundary polygons from sparse boundary points (VPs/crossings) without falling back to stale label-based fills or creating gaps.

