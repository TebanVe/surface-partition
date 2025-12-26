# CRITICAL BUG: Missing Corner Areas in Triple Point Triangles

**Discovered**: December 25, 2025  
**Status**: Identified, Fix Planned, Not Yet Implemented  
**Severity**: HIGH - Affects optimization accuracy

---

## Summary

Triple point triangles are missing corner area contributions in the optimization. The code only counts the void interior (Steiner subdivisions) but completely ignores the corner regions between variable points and mesh vertices.

---

## The Bug

### What Should Happen (Per Paper):

A triple point triangle with:
- Mesh vertices: v1, v2, v3 (belonging to cells c1, c2, c3)
- Variable points: VP_A, VP_B, VP_C (forming void triangle)
- Steiner point: X (inside void)

Should contribute to each cell:
1. **Void interior**: Sub-triangle from Steiner point (e.g., Area(VP_A, VP_B, X) for cell c3)
2. **Corner region**: Triangle from mesh vertex to adjacent VPs (e.g., Area(v1, VP_B, VP_C) for cell c1)

### What Actually Happens:

**AreaCalculator** (lines 123-127 in `area_calculator.py`):
```python
if len(set(labels)) == 3:
    # Skip triple-point triangles - they're handled by SteinerHandler
    continue
```
→ Entire triple point triangle is excluded from area calculation

**SteinerHandler** (`get_area_contribution()`, lines 230-285):
```python
# Only computes void interior
area = triangle(vp_pos1, vp_pos2, steiner_point)
contributions[cell_idx] = area  # Missing corners!
```
→ Only void interior is added, corners are completely missing

**Result**: Corner areas contribute ZERO to any cell!

---

## Impact

### On Optimization:
- Area constraints are violated (cells have less area than calculated)
- Optimization thinks cells need more area than they actually have
- Results in inaccurate partitions

### On Visualization:
- `visualize_precise_region.py` shows white gaps (missing corners)
- Visualization correctly reflects what optimization calculates
- Both suffer from the same bug

---

## Root Cause

The code was written under the assumption that:
> "Triple point triangles = void triangles"

But the reality is:
> "Triple point triangles = void interior + 3 corner regions"

The paper (Figure 7) says "the area of ABX is added to Cell 3" which refers to the void interior subdivision. The paper **implicitly assumes** that corners are handled by normal boundary triangle logic. However, since `AreaCalculator` **skips the entire triangle**, corners are never calculated.

---

## Verification That Perimeter Is Correct

**Good news**: Perimeter calculation is correct!

The Steiner perimeter contribution (lines 193-228 in `steiner_handler.py`):
```python
contributions[cell_idx] = steiner_edge1 + steiner_edge2 - original_edge
```

This NET formula means:
- Regular perimeter counts: ||VP_A - VP_B|| (void edge)
- Steiner adds: ||VP_A - X|| + ||VP_B - X|| - ||VP_A - VP_B||
- NET result: ||VP_A - X|| + ||VP_B - X|| (two Steiner branches)

The void edge cancels out, and the cell gets the correct Steiner branches. ✅

---

## Fix Plan

### Phase 1: Fix Calculation (Critical)

**File**: `src/core/steiner_handler.py`

1. Store mesh vertex for each cell in `TriplePoint.__init__()`:
   ```python
   self.cell_to_mesh_vertex: Dict[int, int] = {}
   ```

2. Modify `get_area_contribution()` to include corners:
   ```python
   # Void interior (existing)
   void_area = triangle(pos1, pos2, self.steiner_point)
   
   # Corner area (NEW)
   vertex_pos = mesh.vertices[self.cell_to_mesh_vertex[cell_idx]]
   corner_area = triangle(vertex_pos, pos1, pos2)
   
   # Total
   contributions[cell_idx] = void_area + corner_area
   ```

3. Gradients computed via finite differences should automatically handle the change

### Phase 2: Fix Visualization (Validation)

**File**: `examples/visualize_precise_region.py`

Modify `compute_triple_point_cell_portion()` to return BOTH void and corner polygons instead of just void.

---

## Testing Strategy

1. **Unit Test**: Simple triple point with known geometry, verify total area matches mesh triangle
2. **Integration Test**: Run optimization, verify area constraints are satisfied
3. **Visual Test**: No white gaps in `visualize_precise_region.py`

---

## Related Files

- `src/core/steiner_handler.py` - Contains the bug
- `src/core/area_calculator.py` - Skips triple point triangles (correct, but needs Steiner to be complete)
- `examples/visualize_precise_region.py` - Shows white gaps due to missing corners
- `manifold_partition.md` (line 364) - Paper's Figure 7 description

---

## Next Steps

1. Implement fix in `SteinerHandler`
2. Create unit tests
3. Verify with full optimization run
4. Update visualization (optional, for validation)

---

**Note**: This bug does NOT affect perimeter calculation, which is implemented correctly according to the paper.

