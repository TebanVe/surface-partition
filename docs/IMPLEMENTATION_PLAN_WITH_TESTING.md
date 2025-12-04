# Implementation Plan: Topology Switching with Incremental Testing

**Date:** November 23, 2025 (Updated: December 4, 2025)  
**Goal:** Implement and test topology switching incrementally

---

## Overview: 5-Phase Incremental Plan

| Phase | Goal | Status |
|---|---|---|
| **0** | Verify dependencies | ✅ Complete |
| **1** | Test Type 1 & Type 2 switching | ✅ Complete |
| **2** | Fix Issue 1 (stale triangle_segments) | ✅ Complete |
| **3** | Test switching + calculations | ✅ Complete |
| **4** | Fix Issues 2+3 (cross-triangle segments) | 🔄 In Progress |
| **5** | Full optimization loop | ⏳ Pending |

---

## Critical Issues Discovered (December 4, 2025)

### Issue A: VP Index Ordering Discrepancy ⚠️ CRITICAL - RESOLVED

**Problem:** Two initialization paths create the **same VPs** but in **different order**:

| Initialization | VP on edge (9076, 9267) | Total VPs |
|----------------|-------------------------|-----------|
| Without `boundary_topology` | Index **228** | 1977 |
| With `boundary_topology` | Index **230** | 1977 |

**Root Cause:** 
- `PartitionContour(mesh, indicators)` - scans triangles in mesh.faces order
- `PartitionContour(mesh, indicators, boundary_topology=...)` - iterates by region, then sorts edges

**Critical Impact on Lambda Loading:**
The refined_contours.h5 file stores lambda values **BY VP INDEX**. When loaded with different VP indices:

| Path | VP on edge (9075, 9267) | Assigned Lambda |
|------|-------------------------|-----------------|
| Without boundary_topology | Index 227 | lambda_opt[227] = **0.8906** ✓ |
| With boundary_topology | Index 229 | lambda_opt[229] = **0.1819** ✗ |

**The same VP gets completely different lambda values!**

This causes:
1. Wrong lambda positions on boundary edges
2. Different Steiner point positions
3. **Different VP selected for Type 2 switch**
4. **Incorrect target triangle and geometry**

**Resolution:** 
All code loading refined lambdas MUST use `_initialize_from_indicators()` (NO boundary_topology) to match the VP indices used when saving.

**Files Fixed:**
- `examples/visualize_topology_switch.py` - uses indicators only
- `examples/test_topology_switcher_basic.py` - uses indicators only

**Future Fix:** Store lambdas with edge keys instead of VP indices to avoid ordering dependency.

---

### Issue B: Indicator Functions Become Stale

**Problem:** After perimeter refinement, indicator functions no longer reflect actual boundaries.

**What happens:**
- Indicator functions are computed from initial relaxation (winner-takes-all from densities)
- Perimeter refinement moves λ values, changing actual boundary positions
- But indicator functions remain frozen at initial state

**Impact:**
- VP structure (which VPs exist, their indices) determined from stale data
- `vertex_labels` in `TriangleSegment` become stale after switches
- Only works because refined λ values are loaded via `set_variable_vector()`

**Current Status:** The `boundary_topology` path preserves VP structure from contour extraction. Without it, VPs are created from stale indicators but positions are corrected via lambda loading.

---

### Issue C: Mesh Geometry and Collinearity

**Problem:** Mesh triangles are **planar approximations** of curved surface (torus).

**Geometric Reality:**
- Adjacent triangles sharing an edge form a **dihedral angle**
- A segment from VP1 to VP2 that "passes through" a shared vertex:
  - Does NOT pass through the vertex in 3D
  - Passes **above or below** (edges are in different planes)
  
**Implication for Area Calculation:**
For cross-triangle segments, we should trace:
```
VP1 → shared_vertex → VP2
```
Two sub-segments on the mesh surface, NOT a straight 3D line.

**Current Status:** Intersection calculations use 3D line-segment math, which doesn't account for mesh surface curvature.

---

### Issue D: Edge-Cutting Intersection Calculation

**Problem:** When a segment crosses a triangle edge (not at a vertex), how do we compute the intersection?

**Current Approach:**
- `_compute_segment_crossings()` traces segment through mesh
- Computes entry/exit points assuming **straight 3D line**
- Stores in `SegmentCrossingInfo`

**What Should Happen:**
1. Find intersection point on shared edge
2. Compute partial areas on each side
3. Account for mesh surface geometry (not straight-line distance)

**Status:** Not correctly implemented. Need to revise for curved mesh surfaces.

---

## Completed Phases

### Phase 0-3: Summary

All basic functionality working:
- ✅ Type 1 switches (VP edge migration)
- ✅ Type 2 switches (triple point migration) 
- ✅ Triangle segment rebuilding
- ✅ Area/perimeter calculations produce valid results
- ✅ Triple point perimeter contributions computed

### Key Implementation Details

**Type 2 Switch Algorithm:**
1. Find shared edge (closest to Steiner point)
2. Identify anchor VP (on shared edge)
3. Select moving VP (closest to shared vertex)
4. Find target triangle (shares anchor edge)
5. Move VP to free edge with λ=0.5
6. Update `boundary_segments` (destroy old, create new)

**Triangle Segment Rebuild:**
- `rebuild_triangle_segments_from_current_vps()` creates fresh segments
- Includes 1-VP triangles (for cross-triangle segment tracking)
- Uses `is_triple_point()` based on VP count, not stale `vertex_labels`

---

## Phase 4: Cross-Triangle Segment Handling (IN PROGRESS)

### Objective
Handle segments that cross multiple triangles after topology switching.

### Key Data Structures

```python
@dataclass
class BoundarySegment:
    """Explicit representation of a segment between two VPs."""
    vp_idx_1: int
    vp_idx_2: int
    cell_pair: Tuple[int, int]
    segment_type: str  # "normal", "edge_following", "edge_cutting"
    crossing_info: Optional[List['SegmentCrossingInfo']] = None

@dataclass  
class SegmentCrossingInfo:
    """Precomputed geometric intersection for segment crossing triangle."""
    segment: Tuple[int, int]
    triangle_idx: int
    entry_point: np.ndarray
    exit_point: np.ndarray
    entry_edge: Tuple[int, int]
    exit_edge: Tuple[int, int]
    cell_pair: Tuple[int, int]
```

### Segment Classification

| Type | Definition | Handling |
|------|------------|----------|
| **normal** | Both VPs in same triangle | Standard calculation |
| **edge_following** | Different triangles, edges share vertex + collinear | Path through vertex |
| **edge_cutting** | Different triangles, no shared vertex | Compute mesh intersections |

### Outstanding Questions

1. **Collinearity on curved surface:** Two edges sharing a vertex are NEVER truly collinear in 3D. How do we define "edge-following" for area calculation?

2. **Intersection path:** Should segment path follow mesh surface (through vertices) or straight 3D line?

3. **VP ordering standardization:** Which initialization path should be canonical?

---

## Files Modified

### Core Implementation
- `src/core/topology_switcher.py` - Type 1/2 switch logic, segment classification
- `src/core/contour_partition.py` - BoundarySegment, SegmentCrossingInfo, rebuild methods
- `src/core/steiner_handler.py` - Uses `identify_triple_points_from_current_vps()`
- `src/core/area_calculator.py` - Cache-aware partial area computation
- `src/core/perimeter_calculator.py` - Uses boundary_segments

### Test/Visualization
- `examples/test_topology_switcher_basic.py` - Phase 0-3 tests
- `examples/visualize_topology_switch.py` - 3D visualization with VP labels

---

## Next Steps

### Immediate (Phase 4 completion)
1. **Investigate VP ordering discrepancy** - Why do two paths create different orderings?
2. **Fix intersection calculation** - Account for mesh surface geometry
3. **Verify edge-following segments** - Ensure correct area calculation

### Future (Phase 5)
- Integration with optimization loop
- Multi-iteration topology switching
- Production testing

---

## Test Commands

```bash
# Run basic tests
python examples/test_topology_switcher_basic.py \
  --relaxed-file results/.../surface_part5_..._20251027_233612.h5 \
  --refined-file results/.../..._refined_contours.h5

# Visualize Type 2 switch  
python examples/visualize_topology_switch.py \
  --solution results/.../surface_part5_..._20251027_233612.h5 \
  --switch-type type2 --state both --vp-size 0.0005
```

---

## Success Criteria

| Phase | Criteria | Status |
|-------|----------|--------|
| 0-3 | Basic switching works | ✅ |
| 4 | Cross-triangle segments handled | 🔄 |
| 5 | Full optimization converges | ⏳ |

**Critical blocker:** VP ordering discrepancy must be resolved before Phase 5.
