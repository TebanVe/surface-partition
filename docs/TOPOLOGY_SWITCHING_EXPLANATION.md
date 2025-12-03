# Topology Switching: Explanation and Implementation Plan

**Date:** November 14, 2025 (Revised with visual analysis)  
**Based on:** Paper Section 5 (lines 370-375) and visual diagram analysis  
**Status:** Not yet implemented

---

## What is Topology Switching?

Topology switching allows the optimization to **change which mesh edges contain variable points** when the current configuration is suboptimal. This enables contours to "flow around corners" and adapt to mesh geometry.

**Critical features:**
1. Initial contours (from indicator functions) might not have optimal topology
2. Variable points can move to adjacent edges during optimization
3. Triple points can migrate between mesh triangles via variable point movement
4. Total number of variable points and boundary segments remains constant

---

## Paper Description (Lines 370-375)

> "It may be the case that some vertices of the contour would 'like' to switch to another side. This can be the case if at the end of the optimization one of the parameters λᵢ is close to 0 or 1 or a triple point in one of the constructed Steiner trees is on the boundary of the corresponding mesh triangle. In this cases we modify the initial contours taking into the account these results and we restart the optimization procedure. The modification is done in the following way:
>
> (1) If one of the λᵢ is equal to 0 or 1 then we add the corresponding point to the adjacent cell and restart the algorithm.
> (2) If one of the triple points arrives on the edge of its corresponding mesh triangle then we allow it to move to the adjacent triangle.
>
> After a finite number of switches the configuration stabilizes and a local minimum is found."

---

## Two Types of Topology Switches

### **Type 1: Variable Point Edge Switching**

**What triggers it:**
- Variable point has λ < tol (near 0) or λ > 1-tol (near 1)
- Point wants to move past current edge's endpoint

**What happens:**
- Variable point moves from current edge to an adjacent mesh edge
- New λ positioned away from endpoint (default: 0.1 or 0.9) to prevent immediate re-trigger

**Selection algorithm (triangle-local):**
1. Find two triangles T_A, T_B sharing the current edge
2. In each triangle, identify the free edge incident to target vertex
3. Test both candidate edges: compute total segment length to neighboring variable points
4. Choose edge that minimizes: Σ distance(VP_neighbor, VP_new_position)

**Example:**
```
Current: VP2 on edge (v3,v4) at λ≈1 (near v4)
         - VP1 on (v1,v3), VP3 on (v3,v7)
         - Current edge shared by T1=(v1,v2,v3) and T3=(v3,v4,v7)

Candidates: 
         - (v1,v4) from T1 (free edge through v4)
         - (v4,v7) from T3 (free edge through v4)

Test:    For each candidate, calculate |VP1-VP2_test| + |VP2_test-VP3|

After:   VP2 moves to edge with minimum total distance, λ=0.1
```

**Fallback:** If triangle-local fails, test all edges incident to target vertex.

### **Type 2: Triple Point Migration Between Triangles**

**What triggers it:**
- Steiner point in triangle T1 approaches an edge (near boundary of T1)
- Indicates void triangle is poorly positioned

**What happens (4-step process):**

```
BEFORE (Triangle T1):
- Void triangle vertices: VP1, VP2, VP3 (on edges of T1)
- Steiner point X near edge containing VP3
- Cells meeting: 1, 2, 3

Step 1: DETECT which VP to move
        └─> One VP (e.g., VP3) already on shared edge T1↔T2 → stays in place (anchor)
        └─> From remaining 2 VPs (VP1, VP2): select closest to a shared vertex
        └─> VP1 on edge (v_a, v_b), VP2 on edge (v_c, v_d)
        └─> If shared edge is (v_a, v_c): compare VP1's dist to v_a vs VP2's dist to v_c
        └─> Choose VP that minimizes distance to its respective shared vertex

Step 2: IDENTIFY target edge in T2 (DETERMINISTIC - not optimized!)
        └─> T2 shares edge with T1 (the edge containing anchor VP3)
        └─> T2 has 3 edges: one is shared (has VP3), one has existing VP4
        └─> VP1 moves to the ONE FREE edge in T2 (no other VPs on it)
        └─> New λ = 0.5 (center of edge)
        └─> This ensures T2 ends up with exactly 3 VPs (one per edge)

Step 3: UPDATE segments - CONCEPTUAL view (what "really" happens)
        
        Segment Role Changes:
        └─> VP4-VP3: Boundary segment → Void triangle edge (absorbed into triple point)
        └─> VP2-VP3: Void triangle edge → Boundary segment (triple point dissolved in T1)
        
        Segments Destroyed:
        └─> VP1-VP2: Connection broken (VP1 left T1)
        
        Segments Persist:
        └─> VP1-VP3: Still connects moving VP to anchor (now in T2)
        
        Segments Created:
        └─> VP1-VP4: New void triangle edge in T2
        
        Net Effect:
        └─> Boundary segments: unchanged count (VP4-VP3 lost, VP2-VP3 gained)
        └─> Void edges: 3 in T1 → 0, 0 in T2 → 3 (reorganized)

Step 3b: IMPLEMENTATION approach (how code handles it)
        
        └─> Don't track individual segment transformations (too complex!)
        └─> Instead: Destroy & Rebuild strategy
        └─> Process:
            1. Destroy all 3 void triangle edges in T1 (SteinerHandler cleanup)
            2. Move VP1 to free edge in T2
            3. Rebuild triangle_segments from current VPs
               → T1 now 2 VPs → normal boundary (VP2-VP3)
               → T2 now 3 VPs → triple point detected
            4. Re-initialize SteinerHandler
               → Creates 3 new void edges in T2 (VP3-VP1, VP1-VP4, VP4-VP3)
               → Computes new Steiner point position
        └─> Why this works: Void edges are virtual (Steiner tree), recomputed from VP positions

AFTER (Triangle T2):
- New void triangle: VP3, VP1, VP4 (on edges of T2)
- Steiner point X now inside T2
- Same cells meeting: 1, 2, 3

Conservation laws:
✓ Total variable points: unchanged
✓ Total boundary segments (non-void): unchanged (1 destroyed, 1 created)
✓ Total void segments: unchanged (3 old destroyed, 3 new created)
✓ Net segment change: 0 (void segments reorganized around new Steiner point)
✓ Each edge has ≤1 variable point: maintained
✓ Triple point has exactly 3 variable points: maintained
```

**Key insight:** Type 2 is **enabled by Type 1** - moving VP1 to T2 allows the triple point to follow.

---

## Current Implementation Status

### ✅ **Detection** (Implemented)

1. `VariablePoint.on_boundary(tol)` - Detects λ≈0 or λ≈1
2. `PartitionContour.get_boundary_variable_points(tol)` - Lists points needing switches
3. `TriplePoint.is_on_triangle_boundary(tol)` - Detects Steiner point near edge
4. `SteinerHandler.get_boundary_triple_points(tol)` - Lists triple points near boundaries
5. `PerimeterOptimizer.check_topology_switches_needed(tol)` - Aggregates both types

### ❌ **Execution** (Not Implemented)

1. Moving variable points to adjacent edges (Type 1)
2. Updating segments (destroy/create) after variable point moves
3. Re-detecting triple points in new triangles (Type 2)
4. Iterative optimization loop with topology changes

---

## Selection Criteria for Type 2 Switching

When triple point in T1 wants to move to adjacent triangle T2:

**Criterion 1: Which variable point to move?**
- One VP already on shared edge T1↔T2 → **stays in place**
- From remaining 2 VPs: choose one **closest to a vertex of T2**
- Minimizes "jump" distance and likely reduces perimeter

**Criterion 2: Which edge in T2?**
- T2 has 3 edges, 2 already occupied → **deterministic choice**
- Move to the **only free edge**

---

## Implementation Algorithm

### **High-Level Loop**

```python
converged = False
topology_iteration = 0

while not converged and topology_iteration < max_iterations:
    # 1. Run optimization
    result = optimizer.optimize(...)
    
    # 2. Check for switches
    switches_needed, switch_info = check_topology_switches_needed(tol)
    
    if not switches_needed:
        converged = True
    else:
        # 3. Apply Type 1 (variable point moves)
        for vp in switch_info['boundary_points']:
            target_vertex = get_target_vertex(vp)  # Vertex vp is approaching
            
            # Triangle-local selection
            adjacent_triangles = mesh.get_triangles_sharing_edge(vp.edge)
            candidate_edges = get_free_edges_at_vertex(adjacent_triangles, target_vertex)
            
            # Distance minimization
            best_edge = min(candidate_edges, 
                           key=lambda e: total_segment_length(vp, e, target_vertex))
            
            # Move variable point
            move_vp_to_edge(vp, best_edge, target_vertex, tol=0.1)
        
        # 4. Apply Type 2 (re-detect triple points after VP moves)
        if switch_info['boundary_triple_points']:
            re_detect_triple_points()
        
        # 5. Continue
        topology_iteration += 1
```

---

## Key Implementation Notes

### **Triangle-Local Switching**
- Variable point moves within triangles adjacent to current edge
- Minimizes topological disruption and preserves partition structure
- Distance-based selection ensures perimeter reduction

### **Conservation Laws**
- Total variable points: **constant**
- Total boundary segments: **constant** (segments destroyed = segments created)
- Each mesh edge has at most 1 variable point
- Triple points always have exactly 3 variable points

### **Steiner Trees**
- Never dissolved - they minimize perimeter (120° angles)
- Only re-assigned to correct mesh triangles via re-detection

### **Re-detection After Type 1**
- After variable points move to new edges
- Triple points may now be in different mesh triangles
- Re-scan mesh to update `TriplePoint.triangle_idx`

---

## Expected Outcomes

**After implementation:**
- ✅ Adaptive topology: contours flow naturally around mesh
- ✅ Better local minima: escape suboptimal initial configurations
- ✅ Lower perimeters: match or improve current results
- ✅ Paper-accurate: implements Section 5 methodology

**Typical convergence:**
```
Iteration 1: Perimeter = 38.78, 8 switches → apply
Iteration 2: Perimeter = 38.42, 3 switches → apply  
Iteration 3: Perimeter = 38.18, 1 switch → apply
Iteration 4: Perimeter = 38.12, 0 switches → CONVERGED ✓
```

Expected: **2-5 topology iterations** for most cases

---

## References

- **Paper Section 5**, lines 370-375
- **Paper Figure 8**: Void triangles (red), Steiner trees (blue), mesh (white)
- **Paper Figure 7**: Steiner tree area and perimeter treatment
- **Visual analysis**: Hand-drawn diagrams of triple point migration (November 14, 2025)

---

## Document History

**November 18, 2025 (Type 1 Triangle-Local Strategy):**
- Updated Type 1 with triangle-local edge selection algorithm
- Added distance-based selection: minimize Σ distance(VP_neighbor, VP_new_position)
- Concrete example: VP2 switching between (v1,v4) and (v4,v7) based on segment lengths
- Removed cell membership matching (superseded by triangle-local approach)
- Updated implementation pseudocode with triangle-aware logic

**November 14, 2025 (Visual Analysis Integration):**
- Incorporated insights from hand-drawn diagrams of triple point movement
- Clarified: Type 2 requires Type 1 (variable point movement enables triple point migration)
- Added selection criteria: proximity for VP choice, deterministic for target edge
- Emphasized conservation: total VPs constant, segments destroyed = segments created
- Removed incorrect "dissolution" terminology
- Shortened document, removed outdated implementation details

**November 12, 2025:**
- Deep analysis of triple point movement mechanics
- Explained relationship between Type 1 and Type 2

**November 7, 2025:**
- Corrected Figure 8 interpretation
- Clarified void triangles are inside mesh triangles

**November 7, 2025:**
- Initial document based on paper Section 5
