# Future Fix: Remove `_initialize_from_boundary_topology` Path

**Status:** DEFERRED -- implement after confirming lambda-edge fix  
**Priority:** Low (simplification, not correctness)  
**Date:** 2026-03-09  

## Background

`PartitionContour` has two initialization paths:

1. **`_initialize_from_indicators`** -- scans all mesh triangles directly using `indicator_functions` to find boundary edges and create VPs.
2. **`_initialize_from_boundary_topology`** -- takes pre-computed `boundary_topology` (from `ContourAnalyzer.extract_contours_with_topology()`) and creates VPs from sorted edges.

Path 2 was introduced when `indicator_functions` were considered read-only and the contour extraction pipeline (`extract_contours_with_topology`) was the only way to identify boundaries. Once we implemented direct modification of `indicator_functions` after migrations (updating the target vertex's cell label), Path 2 became unnecessary -- `indicator_functions` are always up-to-date and Path 1 can be used directly.

## Why Remove It

1. **Unnecessary complexity.** Path 2 requires a multi-step pipeline: create a temporary `ContourAnalyzer`, set `indicator_functions` as `densities`, call `extract_contours_with_topology()` (which runs `compute_indicator_functions` + level-set crossing checks per-region), produce a `boundary_topology` dict, then process it through a 3-pass initialization (edge collection, triangle segment creation, triple-point fix-up). Path 1 does all of this in a single triangle scan.

2. **Triple-point fix-up fragility.** Because `extract_contours_with_topology` processes per-region, a triple-point triangle (3 vertices in 3 different cells) only reports 2 of its 3 boundary edges from any single region's perspective. Path 2 needs a Pass 3 fix-up to detect and complete these. Path 1 naturally finds all 3 edges because it scans each triangle's edges directly.

3. **VP ordering difference.** Path 2 creates VPs in sorted-edge (lexicographic) order. Path 1 creates them in triangle-scan order. This ordering difference was the root cause of the lambda-edge mismatch bug (see `BUG_INDICATOR_FUNCTIONS_CORRUPTION.md`). While the edge-keyed save/load fix resolves this, eliminating the ordering discrepancy removes a source of confusion.

4. **The `raw_contours` return value is discarded.** All call sites that use `extract_contours_with_topology` for partition initialization ignore the `raw_contours` output. The contour extraction is wasted work.

## Call Sites to Update

| File | Current Usage | Change |
|------|--------------|--------|
| `examples/data_loader.py` | Creates temp `ContourAnalyzer`, calls `extract_contours_with_topology`, passes `boundary_topology` to `PartitionContour` | Remove the `ContourAnalyzer` creation and `extract_contours_with_topology` call. Pass `PartitionContour(mesh, indicator_functions)` without `boundary_topology`. |
| `testing/refine_perimeter_iterative.py` | Calls `analyzer.extract_contours_with_topology()`, passes result to `PartitionContour` | Remove the `extract_contours_with_topology` call. Pass `PartitionContour(mesh, indicators)` without `boundary_topology`. |
| `examples/refine_perimeter.py` | Same pattern as above | Same change as above. |
| `testing/test_lambda_edge_roundtrip.py` | Creates temp `ContourAnalyzer` for fresh partition reconstruction | Remove the `ContourAnalyzer` usage. Use `PartitionContour(mesh, saved_indicators)` directly. |
| `src/core/contour_partition.py` | Contains both `_initialize_from_indicators` and `_initialize_from_boundary_topology` | Remove `_initialize_from_boundary_topology`, remove `boundary_topology` parameter from `__init__`. |

## What to Keep

- `extract_contours_with_topology` in `src/find_contours.py` should NOT be deleted. It is used by `surface_visualization.py` for drawing raw contour polylines (a visualization-only use case independent of `PartitionContour`).
- `extract_contours` (the simpler wrapper) is also used independently.

## Considerations

- **Performance**: `_initialize_from_indicators` scans all mesh triangles (~92,160 faces). This takes a few seconds vs. the boundary_topology path which only processes boundary triangles (~3,900). However, this is negligible compared to hours of optimization per iteration.
- **VP ordering**: After this change, VPs will be in triangle-scan order everywhere (instead of sorted-edge order). With edge-keyed save/load, this has no correctness impact. VP index numbers in logs will differ from previous runs.
- **Backward compatibility**: Old HDF5 files without `vp_edges` will still load via the index-based fallback in `data_loader.py`. Their lambda assignment may be wrong (same as before the fix), but at least they won't crash.

## Verification

After implementing this change, run the lambda-edge roundtrip test:
```
python testing/test_lambda_edge_roundtrip.py --solution <iteration2_file.h5>
```
The test should pass with all lambdas correctly matched by edge key.
