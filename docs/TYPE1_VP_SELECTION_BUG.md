# Type 1 Migration: VP Selection Bug

**Date**: January 26, 2026
**Status**: IDENTIFIED - Solution In Progress

## Problem Statement

Type 1 migration produces white polygons (rendering artifacts) in certain cases due to incorrect selection of the migrating VP. The current distance-based selection can choose an **endpoint VP** in a component, whose external neighbor does not approach the same target vertex, causing migration to affect VPs outside the component scope.

### Two-Level Problem Structure

1. **Component-level selection** (WORKING CORRECTLY):
   - Deferral mechanism correctly resolves conflicts between neighboring components
   - Distance-based priority and risk assessment work as designed
   - Correctly identifies which component should migrate

2. **VP-level selection within component** (CURRENTLY BROKEN):
   - Distance-based selection can choose an endpoint VP instead of middle VP
   - Endpoint VP's external neighbor may not be in the component
   - Leads to migration affecting VPs outside component scope → white polygons

**Critical insight**: Even when the deferral mechanism correctly selects a component for migration, the wrong VP can still be chosen within that component, causing the same failure mode.

## Symptoms

- **White polygons** appear in AFTER state visualization
- **Observed in 3-VP components** where migrating VP is selected as an endpoint (could potentially occur in other component sizes if migrating VP is not selected correctly)
- Migration affects VPs that are not part of the component
- Console warnings:
  ```
  New edge (X, Y) does not contain target vertex Z, using midpoint
  ```

**CRITICAL**: These warning messages are diagnostic indicators of the bug. **Implementation must guarantee** that these warnings are produced whenever:
- Neighbor VPs don't contain the target vertex
- External neighbors are included in migration
- Validation fails for any reason

Warning visibility requirements:
- **Visualization scripts**: Terminal output + visual annotation on figures
- **Optimization scripts**: Terminal output + halt execution

## Root Cause

### Current VP Selection Logic

In `examples/visualize_type1_vertex_collapse.py` (line 763):
```python
migrating_vp_idx = min(component_vps, 
                       key=lambda vp: switcher.compute_boundary_distance(vp))
```

**Selects VP with minimum distance** `min(λ, 1-λ)` to target vertex.

### Why This Fails

Type 1 migration mechanics require:
```
left_neighbor — migrating_VP — right_neighbor
      ↓               ↓              ↓
   target_v       target_v       target_v
```

All three VPs must:
1. Form a connected chain
2. Converge to the **same target vertex**

When an **endpoint VP** is selected, its external neighbor may:
- Approach a different vertex
- Be part of a different component
- Not be close enough to the target vertex

## Concrete Example: Component 47

### Component Configuration
```
Component 47: [1804, 1805, 1926]
Target vertex: 44102
Min distance: 0.000021
```

### Actual Topology
```
1926 — 1804 — 1805 — [1806]
(end)  (middle) (end)  (EXTERNAL)
 ↓       ↓       ↓       ↓
44102   44102   44102   ??? (different vertex)
```

### What Happened

1. **VP 1805 selected** (λ = 0.000021, closest to vertex 44102)
2. VP 1805 is an **endpoint** in component chain
3. `_get_two_neighbors(1805)` returns `[1804, 1806]`
4. **VP 1806 is NOT in component 47**
5. VP 1806 likely approaches a different vertex (not 44102)

### Migration Execution
```
Migrating VP: 1805 ✓ (in component)
Left neighbor: 1804 ✓ (in component)  
Right neighbor: 1806 ✗ (NOT in component, wrong target vertex)
```

### Warnings Generated
```
New edge (43910, 44101) does not contain target vertex 44102, using midpoint
New edge (43911, 43912) does not contain target vertex 44102, using midpoint
```

These edges belong to VP 1806's geometry, not vertex 44102's neighborhood.

### Result

- VP 1806 is moved (even though it's outside component 47)
- Triangles affected by VP 1806 are NOT in the rebuild list (only 6 triangles around vertex 44102)
- Those triangles have stale `triangle_segments` → **white polygons**

## Component Size Context

### Component Types in Practice

From analysis of 72 components:

#### 3-VP Components (Majority)
- **Labeled size**: 3 VPs
- **Actual**: All 3 VPs meet `boundary_tol = 0.1` threshold
- **Behavior**: Complete component, all VPs approach same target vertex
- **Risk**: Endpoint VP selection can pick VP with external neighbor approaching different vertex

#### 2-VP Components
- **Labeled size**: 2 VPs
- **Actual**: Third VP exists but didn't quite meet distance threshold
- **Behavior**: Most have third VP also approaching same target vertex (just slightly outside threshold)
- **Exception**: Special 2-VP pairs (see below)
- **Risk**: Similar to 3-VP if endpoint selected

#### 1-VP Components
- **Labeled size**: 1 VP
- **Actual**: Other 2 VPs exist but didn't meet distance threshold
- **Behavior**: The two extra VPs (neighbors) that failed the distance criteria should also approach the same target vertex as the labeled VP
- **Risk**: Only one choice for migrating VP, but the two neighbors found via `_get_two_neighbors()` must be validated to ensure they approach the same target vertex

### Special Case: Connected Component Pairs with Shared Non-Boundary Neighbors

**Important Note**: Components 49 and 50 are **not neighbors of each other**. Each has its own neighboring component, and when compared to their respective neighbors, they were deferred.

**Example Pattern**: Two 2-VP components, each with a neighbor

**Topology**:
```
Component A (2 VPs):  VP_a1 — VP_a2 — VP_shared_ab
                        ↓       ↓           ↓
                    vertex_A  vertex_A  (middle, approaches neither vertex)

Component B (2 VPs):  VP_shared_ab — VP_b1 — VP_b2
                           ↓           ↓       ↓
                      (middle)     vertex_B  vertex_B
```

**Characteristics**:
- Two components connected by a shared non-boundary neighbor VP (`VP_shared_ab`)
- `VP_shared_ab` sits between two vertices, doesn't approach either
- Deferral logic correctly picks the component that is closer to convergence
- **Observation** (tentative): The chosen component also has `VP_shared` slightly closer to its target vertex (though not as close as the other 2 VPs in that component)

**Current behavior**: Deferral logic handles conflict resolution via distance-based priority for components with < 3 VPs.

**Generalization**: This pattern can potentially occur with other component size combinations:
- 2-VP and 1-VP components
- 3-VP and 2-VP components
- 1-VP and 1-VP components
- Any pair where components share a non-boundary neighbor

**Important Caveat**: Even when the correct component is selected via deferral, **the wrong migrating VP can still be chosen** within that component, leading to the same white polygon issue observed in Components 47-48.

## Why Most Migrations Work

Despite incomplete component detection:

1. **1-VP and 2-VP components**: External neighbors typically still approach the same target vertex (just didn't meet threshold by small margin)
2. **3-VP components**: Usually correct, but fails when:
   - Endpoint VP is closest to target vertex
   - Distance-based selection picks the endpoint VP (not the middle VP)
   - External neighbor approaches different vertex or is part of different component
3. **Critical Issue**: Even when **all three VPs approach the same target vertex**, selecting the wrong VP (an endpoint instead of the middle) causes `_get_two_neighbors()` to identify neighbors outside the component scope, leading to incorrect migration and white polygons (as seen in Components 47-48)

## Component Conflict Detection and Deferral Mechanism

### Overview

Before Type 1 migrations are executed, components are analyzed for potential conflicts (shared neighbors) and a deferral mechanism determines which components should migrate and which should wait. This is crucial for maintaining topological consistency when multiple components are converging simultaneously.

### Stage 1: Conflict Detection

**Function**: `detect_proximity_conflicts(components)` (line 2293)

**Purpose**: Identify components that share non-boundary neighbor VPs (topological conflicts)

**Process**:
1. **Pairwise conflict detection**:
   - For each pair of components (i, j)
   - Find shared non-boundary neighbors: `shared_vps = non_boundary_neighbors_i ∩ non_boundary_neighbors_j`
   - If `shared_vps` is non-empty → conflict exists

2. **Proximity check**:
   - Calculate `min_dist_i` and `min_dist_j` (minimum boundary distance in each component)
   - Both near convergence: `min_dist_i < 0.01 AND min_dist_j < 0.01`

3. **Conflict record**:
   ```python
   {
       'component_i': i,
       'component_j': j,
       'size_i': comp_i['size'],
       'size_j': comp_j['size'],
       'shared_vps': list(shared_non_boundary),
       'min_dist_i': min_dist_i,
       'min_dist_j': min_dist_j,
       'both_near_convergence': both_near
   }
   ```

4. **Chain detection**:
   - Identify components with 2+ neighboring components (part of a chain)
   - Example: Component A shares VP_ab with B, B shares VP_bc with C → chain A-B-C

**Key insight**: A conflict means two components share a neighbor VP. If both migrate, that shared VP could be adjusted twice, causing topological inconsistencies.

### Stage 2: Component Selection and Deferral

**Function**: `select_components_for_migration(components, conflicts)` (line 3029)

**Purpose**: Decide which components migrate now vs wait for future iterations

#### Critical Principles

1. **Components WITHOUT conflicts** → Migrate immediately (always safe)
2. **Components WITH conflicts** → Check risk level and apply deferral rules

#### Triple Point Exclusion (Pre-screening)

**Rule**: Components with < 3 VPs that are near triple points are **EXCLUDED**
- Migrating them could damage triple point topology
- 3-VP components are safe even near triple points (neighbors are internal)

```python
if component.get('near_triple_point', False):
    # EXCLUDED from migration
    deferred.append(component)
```
#### Deferral Criteria (ALL must be true for deferral)

1. **Conflict exists**: Components share a non-boundary neighbor VP
2. **Both near convergence**: Both have `min_dist < 0.01`
3. **BOTH components have < 3 VPs**: Risky configuration (if at least one has 3 VPs, both migrate safely)

If any criterion is false → both components can migrate safely.

**Key distinction**: 
- At least one 3-VP → **migrate both** (3-VP has internal neighbors, no risk)
- Both < 3-VP → **apply deferral strategy** ("defer_one" or "defer_both")

### CRITICAL ISSUE: Deferred Components in Subsequent Iterations

**Problem discovered**: When a component is deferred (in the < 3-VP + < 3-VP case), the other component is allowed to migrate. This creates a topology change that affects the deferred component in subsequent iterations of perimeter refinement.

**Scenario** (Both components < 3-VP):
```
Iteration N:
  Component A (2-VP, closer) → MIGRATED
  Component B (2-VP, farther) → DEFERRED

Iteration N+1:
  Component B still converging toward target vertex
  But: Shared neighbor VP was moved when Component A migrated
  Result: Migration of Component B will FAIL
          - Neighbor VP is on wrong edge
          - Cannot settle into required edges for migration
```

**Why this happens**:
- Deferral assumes topology remains stable
- But migration of the preferred component changes the shared neighbor VP
- Deferred component's migration logic expects original topology
- Mismatch causes migration failure

**Note**: This issue **only occurs when both components have < 3 VPs**. When at least one has 3 VPs, both migrate (no deferral), so no tracking is needed for those cases.

**Solutions**:

**Option 1: "defer_one" - Track and Block Deferred Components**
- Create data structure to track deferred components across iterations
- **Challenge**: Component indices are unstable across iterations
  - VPs move during optimization
  - Components can split, merge, or change membership
  - Cannot rely on index alone
- **Solution**: Store component signature:
  ```python
  {
      'target_vertex': int,  # Common vertex
      'vp_edges': Set[Tuple[int, int]],  # Edges of VPs in component
      'shared_neighbor_vp': int,  # VP shared with migrated component
      'migrated_component_target': int,  # Target vertex of migrated component
      'iteration_deferred': int,  # Iteration when deferred
  }
  ```
- Block migration in future iterations **unless**:
  - Migrated component returns to previous configuration (topology restored)
  - Deferred component moves away from target vertex (convergence reversed)
- **Use case**: Want to maximize migration throughput, accept tracking complexity

**Option 2: "defer_both" - Conservative Deferral**
- When conflict detected and both < 3-VP: defer **both** components
- Wait for future iteration when conflict may be resolved
- Simpler implementation, no tracking needed
- Slower convergence but safer
- **Use case**: Want simplicity, test if slower convergence produces better perimeter optimization

**Testing Goal**: Implement both options and compare which produces better perimeter optimization results.

**Implementation Requirements**:
1. **Fix deferral logic**: At least one 3-VP → migrate both
2. **Add configuration parameter**: Choose between "defer_one" and "defer_both"
3. **Implement Option 1 ("defer_one")**:
   - `DeferredComponentTracker` class
   - Component signature matching
   - Blocking logic for subsequent iterations
4. **Implement Option 2 ("defer_both")**:
   - Simple deferral of both components
   - No tracking needed
5. **Detailed logging** for both strategies:
   - Which components were in conflict
   - Deferral decision (one or both)
   - Shared neighbor VP information
   - Target vertices for both components
6. **Testing framework**: Compare optimization results between strategies

#### Selection Logic for Conflicting Components

**CURRENT IMPLEMENTATION (HAS BUG)**:

**Case 1: No risk (both are 3-VP components)**
```python
if component['size'] >= 3 AND other_component['size'] >= 3:
    # Both are 3-VP → Migrate both
    # Internal neighbors don't cause issues
    to_migrate.append(component)
    to_migrate.append(other_component)
```

**Case 2: BOTH components have < 3 VPs (Risky)**

Note: This case should NOT include "one is 3-VP" scenarios - those are handled in Case 1 above.

The old code incorrectly had a "Sub-case 2a: One component is 3-VP" here, which is the bug we're fixing.

**Correct code** only handles the Both < 3-VP case:
```python
# Distance-based priority
if component['min_distance'] < other_component['min_distance']:
    to_migrate.append(component)  # Migrate closer
    deferred.append(other_component)  # Defer farther
else:
    deferred.append(component)
    to_migrate.append(other_component)
```

---

**CORRECTED LOGIC (TO BE IMPLEMENTED)**:

**Case 1: Safe - At least one component has 3 VPs**
```python
if component['size'] >= 3 OR other_component['size'] >= 3:
    # At least one is 3-VP → MIGRATE BOTH
    # Reasoning: 3-VP component has internal neighbors, won't affect the other
    to_migrate.append(component)
    to_migrate.append(other_component)
```

**Case 2: Risky - Both components have < 3 VPs**
```python
# Both < 3-VP → Apply deferral strategy
if deferral_strategy == 'defer_one':
    # Distance-based priority
    if component['min_distance'] < other_component['min_distance']:
        to_migrate.append(component)  # Migrate closer
        deferred.append(other_component)  # Defer farther
        # ⚠️ MUST TRACK deferred component for subsequent iterations
    else:
        deferred.append(component)
        to_migrate.append(other_component)
        # ⚠️ MUST TRACK deferred component for subsequent iterations
elif deferral_strategy == 'defer_both':
    # Conservative approach: defer both, wait for future iteration
    deferred.append(component)
    deferred.append(other_component)
```

### Example: Components 49-50 (Deferred)

**Scenario**: Each is a 2-VP component with its own neighbor

- **Component 49**: Has neighbor component X, shares non-boundary VP
  - When compared to X: farther from convergence → **DEFERRED**
  
- **Component 50**: Has neighbor component Y, shares non-boundary VP
  - When compared to Y: farther from convergence → **DEFERRED**

**Result**: Components 49 and 50 both appear as DEFERRED in the component table, not because they conflict with each other, but because each lost to its respective neighbor in distance-based priority.

### Deferral Mechanism Strengths

1. **Prevents double adjustment** of shared neighbor VPs
2. **Prioritizes safer configurations** (3-VP over < 3-VP)
3. **Handles chains** (components with multiple neighbors)
4. **Distance-based tie-breaking** when both are risky

### Current Limitation

**The deferral mechanism correctly selects which component should migrate**, but **does not validate the migrating VP selection within that component**.

**Example**: Component 47 was correctly selected for migration (not deferred), but the distance-based VP selection within the component chose endpoint VP 1805 instead of middle VP 1804, leading to external neighbors being included in the migration.

**Solution**: After deferral resolves component-level conflicts, a **topology-based VP selection** must be used within the selected component to ensure the middle VP is chosen.

## Proposed Solution

### Strategy

**For ALL component sizes, select the topologically middle VP**:

1. **Identify actual 3-VP migration chain**:
   - Component VPs (labeled as boundary)
   - External neighbors (via `_get_two_neighbors()`)

2. **Build component-local adjacency**:
   - Count in-component neighbors for each VP
   - Middle VP: degree 2 (two neighbors within component)
   - Endpoint VP: degree 1 (one neighbor within component)

3. **Select middle VP using topology-based criteria**:
   
   **3-VP components**:
   - Build adjacency within component
   - Select VP with degree 2 (has both neighbors in component)
   
   **2-VP components** (construct auxiliary component):
   - Start with labeled 2 VPs: `[VP_a, VP_b]` approaching `target_vertex`
   - Find third VP (missing from component due to distance threshold):
     - Get left neighbor of VP_a (if exists)
     - Get right neighbor of VP_b (if exists)
     - Select the one **closest to `target_vertex`** (minimum distance `min(λ, 1-λ)`)
   - Auxiliary component: `[VP_left/missing, VP_a, VP_b]` or `[VP_a, VP_b, VP_right/missing]`
   - Apply topology-based selection on auxiliary 3-VP component
   - **This prevents selecting an endpoint VP** that would include external neighbors
   
   **1-VP components** (construct auxiliary component - more complex):
   - Labeled VP: `VP_c` approaching `target_vertex`
   - Need to find 2 missing neighbors that should participate in migration
   - **Multiple topological possibilities**:
     - Two neighbors to left: `(VP_a, VP_b, VP_c)`
     - Middle: `(VP_b, VP_c, VP_d)`
     - Two neighbors to right: `(VP_c, VP_d, VP_e)`
   - **Selection algorithm**:
     1. Get first and second level neighbors on both sides:
        - Left: `VP_a` (first level), `VP_b` (second level)
        - Right: `VP_d` (first level), `VP_e` (second level)
     2. Evaluate each candidate auxiliary component:
        - `(VP_a, VP_b, VP_c)`: Compute total distance to `target_vertex`
        - `(VP_b, VP_c, VP_d)`: Compute total distance to `target_vertex`
        - `(VP_c, VP_d, VP_e)`: Compute total distance to `target_vertex`
     3. Select auxiliary component where all 3 VPs have minimum total distance to `target_vertex`
     4. Apply topology-based selection on auxiliary component
     5. **Estimate**: `VP_c` will be chosen as migrating VP in vast majority of cases, but must verify topologically

4. **Validate target vertex**:
   - Before migration, verify `_get_two_neighbors()` returns VPs approaching same target vertex
   - If validation fails:
     - **In visualization scripts** (`visualize_type1_vertex_collapse.py`): Display highly visible warnings in terminal and on the AFTER figure, but continue execution to show the problem
     - **In main optimization script** (`refine_perimeter.py`): Halt execution and warn user, as proceeding would propagate errors

### Implementation Approach

#### Option 1: Fix Selection in Visualization Script
```python
def select_migrating_vp_safe(component_info, switcher):
    """Select middle VP for migration to avoid external neighbor issues."""
    component_vps = set(component_info['vp_indices'])
    
    if len(component_vps) == 1:
        # Only one choice
        return list(component_vps)[0]
    
    # Build adjacency within component
    from collections import defaultdict
    adjacency = defaultdict(set)
    for segment in switcher.partition.boundary_segments:
        vp1, vp2 = segment.vp_idx_1, segment.vp_idx_2
        if vp1 in component_vps and vp2 in component_vps:
            adjacency[vp1].add(vp2)
            adjacency[vp2].add(vp1)
    
    # Find VP with highest in-component degree (middle VP)
    middle_vp = max(component_vps, key=lambda vp: len(adjacency[vp]))
    
    # Validate: check if it has 2 in-component neighbors (for 3-VP components)
    if len(adjacency[middle_vp]) == 2:
        return middle_vp
    
    # Fallback to distance-based (for edge cases)
    return min(component_vps, key=lambda vp: switcher.compute_boundary_distance(vp))
```

#### Option 2: Add Method to TopologySwitcher
```python
def select_migrating_vp(self, component: Dict) -> int:
    """
    Select the topologically middle VP in a component.
    
    For Type 1 migration to work correctly, the migrating VP must have
    both neighbors within the component scope (or approaching same target vertex).
    
    Args:
        component: Component info dict from analyze_component()
        
    Returns:
        VP index to use as migrating VP
    """
    # Implementation similar to Option 1
    pass
```

#### Option 3: Add Validation in apply_type1_switch_v2
```python
def apply_type1_switch_v2(self, component: Dict, distance_preservation: str = 'preserve') -> bool:
    # Before migration, validate neighbors
    migrating_vp_idx = self._select_migrating_vp(component)
    left, right = self._get_two_neighbors(migrating_vp_idx)
    target_vertex = component['target_vertex']
    
    # Validate all 3 VPs approach same target vertex
    if not self._validate_migration_trio(migrating_vp_idx, left, right, target_vertex):
        self.logger.warning(f"Migration validation failed: neighbors don't share target vertex")
        return False
    
    # Proceed with migration...
```

## Open Questions

1. **Selection strategy for incomplete components (1-VP and 2-VP)**:
   - First, ensure the component is not part of a conflictive case with a neighboring component
   - If deferral mechanism has been used, it's safe to proceed with that component
   - **For 2-VP components**: Construct auxiliary component by finding third VP:
     - The third VP should be the neighbor (left or right) closest to the target vertex
     - Once identified, use **topology-based selection** on the auxiliary 3-VP component
     - This ensures the migrating VP has both neighbors within the auxiliary component scope
   - **For 1-VP components**: Construct auxiliary component from candidate triplets:
     - Evaluate all possible 3-VP configurations (left-left-center, left-center-right, center-right-right)
     - Select the configuration where all 3 VPs are closest to the target vertex
     - Use **topology-based selection** on the auxiliary component
   - **Critical**: The same problem as Component 47 can occur in 1-VP and 2-VP components if distance-based selection is used directly

2. **Validation strictness - Context-dependent behavior**:
   - **Visualization scripts**: Continue execution with highly visible warnings (terminal + figure annotations) to demonstrate the problem
   - **Main optimization script** (`refine_perimeter.py`): Halt and warn user, preventing error propagation

3. **Component definition and boundary threshold**:
   - **No adjustment to `boundary_tol`** - will use even smaller values in subsequent simulations
   - Method must be robust enough to identify correct component members even when some neighbor VPs fail the distance criteria
   - Rely on dynamic neighbor discovery and artificial component construction

4. **Deferral mechanism and topology-based selection**:
   - Current deferral logic correctly selects which component to migrate when conflicts exist
   - **Critical**: Once the correct component has been selected via deferral, **topology-based criteria must be used** to select the migrating VP within that component
   - This prevents selecting an endpoint VP that would include external neighbors outside the component scope

## Testing Strategy

**All testing can be performed visually using the existing visualization script** (`visualize_type1_vertex_collapse.py`). No additional test scripts needed.

1. **Test Component 47** with middle VP selection (VP 1804 instead of VP 1805)
2. **Test Component 46** (similar 3-VP configuration, also likely affected)
3. **Test 1-VP and 2-VP components** with topology-based selection and validation
4. **Test components involved in deferral** (components that had conflicts resolved)
5. **Test edge cases**: Connected component pairs with different size combinations (2-1, 3-2, 1-1)
6. **Verify no new white polygons** across all component sizes and configurations

## Related Files

- `examples/visualize_type1_vertex_collapse.py`: VP selection logic (line 763)
- `src/core/topology_switcher.py`: 
  - `apply_type1_switch_v2()`: Migration execution
  - `_get_two_neighbors()`: Neighbor identification
  - `analyze_component()`: Component metadata
  - `find_connected_components()`: Component detection
- `src/core/contour_partition.py`: `get_boundary_variable_points()` (boundary detection)

## Design Goals: Robust Migration Criteria

The solution must create a **robust, general-purpose migration strategy** capable of handling:

### 1. Unseen Scenarios
- Component size combinations beyond current observations (e.g., 3-3 conflicts, 3-1 conflicts)
- Complex chains with multiple conflicting components
- Edge cases where boundary detection threshold varies

### 2. Strict Boundary Thresholds
- Method must work correctly even with `boundary_tol` values smaller than current 0.1
- Cannot rely on all VPs meeting the threshold
- Must dynamically discover and validate the full 3-VP migration chain

### 3. Component-Level Correctness
- Deferral mechanism must continue to correctly resolve conflicts
- No changes needed to conflict detection logic (works well)
- **New requirement**: Track deferred components across iterations to prevent migration failures when neighbor component has already migrated
- **New requirement**: Detailed logging of all deferral decisions with component signatures (not just indices)
- **Configuration option**: Choose between "defer one" (with tracking) vs "defer both" (safer, slower)

### 4. VP-Level Correctness
- **After deferral selects the correct component**, topology-based selection must identify the correct migrating VP
- Middle VP selection prevents external neighbor inclusion
- Validation ensures all 3 VPs approach the same target vertex

### 5. Error Handling
- **Visualization scripts**: Visible warnings, continue execution to demonstrate issues
- **Optimization scripts**: Halt execution to prevent error propagation
- Clear, actionable error messages for debugging

### 6. Generality
- Single unified approach for all component sizes (1-VP, 2-VP, 3-VP)
- Artificial component construction for incomplete components
- Topology-based selection as the universal strategy

## Implementation Plan

### Overview

This implementation plan addresses two critical bugs and adds robustness to Type 1 migration:

**Bug 1: Incorrect Deferral Logic** (Phase 0 - CRITICAL)
- **Current**: When one component is 3-VP and other is < 3-VP, only 3-VP migrates
- **Correct**: If at least one is 3-VP, **both should migrate**
- **Impact**: Simpler logic, more components can migrate safely
- **Key Insight**: 3-VP components have internal neighbors, so their migration doesn't affect external components

**KEY INSIGHT: When Is Tracking Needed?**

| Component Pair | Current Behavior | Correct Behavior | Tracking Needed? |
|---|---|---|---|
| 3-VP + 3-VP | Migrate both ✓ | Migrate both ✓ | No |
| 3-VP + 2-VP | Migrate 3-VP, defer 2-VP ✗ | Migrate both ✓ | No (after fix) |
| 3-VP + 1-VP | Migrate 3-VP, defer 1-VP ✗ | Migrate both ✓ | No (after fix) |
| 2-VP + 2-VP | Defer one | Defer one OR defer both | **Yes (if defer_one)** |
| 2-VP + 1-VP | Defer one | Defer one OR defer both | **Yes (if defer_one)** |
| 1-VP + 1-VP | Defer one | Defer one OR defer both | **Yes (if defer_one)** |

**Tracking is only needed when BOTH components have < 3 VPs and we choose "defer_one" strategy.**

**Bug 2: Wrong Migrating VP Selection** (Phases 1-3)
- **Current**: Distance-based selection can choose endpoint VP
- **Correct**: Topology-based selection ensures middle VP is chosen
- **Impact**: Prevents external neighbors from being included in migration

**Enhancement: Deferred Component Tracking** (Phase 4)
- **Problem**: Deferred components (both < 3-VP case) can fail in subsequent iterations
- **Solution**: Two strategies with configuration parameter:
  - "defer_one": Track deferred components, block if topology changed
  - "defer_both": Defer both components, simpler but slower
- **Goal**: Test which strategy produces better optimization results

### Phase 0: Fix Deferral Logic (CRITICAL)

**Current Bug**: When one component has 3 VPs and the other has < 3 VPs, only the 3-VP component migrates and the other is deferred. This is incorrect.

**Correct Logic**: If at least one component has 3 VPs, **both should migrate**.

**0.1 Update `select_components_for_migration()` in `topology_switcher.py`**

Current code (lines ~3105-3139):
```python
# Check if at least one has < 3 VPs (risky)
is_risky = (component['size'] < 3) or (other_component['size'] < 3)

if not is_risky:
    # Both are 3-VP → safe, migrate both
    if comp_idx < other_idx:
        to_migrate.append(component)
        to_migrate.append(other_component)
else:
    # Risky conflict → apply deferral
    if component['size'] == 3 or other_component['size'] == 3:
        # Case 1: One is 3-VP → migrate 3-VP, defer other ← BUG!
        ...
    else:
        # Case 2: Both < 3-VP → distance-based priority
        ...
```

**New code**:
```python
# Check if at least one has 3 VPs (safe to migrate both)
at_least_one_3vp = (component['size'] >= 3) or (other_component['size'] >= 3)

if at_least_one_3vp:
    # At least one is 3-VP → MIGRATE BOTH (safe, internal neighbors)
    if comp_idx < other_idx:  # Process once per pair
        to_migrate.append(component)
        to_migrate.append(other_component)
        self.logger.info(
            f"Conflict {comp_idx}-{other_idx}: At least one 3-VP → migrate both"
        )
        processed.add(comp_idx)
        processed.add(other_idx)
else:
    # BOTH < 3-VP → Risky, apply deferral strategy
    if self.deferral_strategy == 'defer_one':
        # Distance-based priority
        if component['min_distance'] < other_component['min_distance']:
            to_migrate.append(component)
            deferred.append(other_component)
            self._track_deferred_component(other_component, component, shared_vps)
        else:
            deferred.append(component)
            to_migrate.append(other_component)
            self._track_deferred_component(component, other_component, shared_vps)
        
        self.logger.warning(
            f"Conflict {comp_idx}-{other_idx}: Both < 3-VP → deferred one (strategy: defer_one)"
        )
    
    elif self.deferral_strategy == 'defer_both':
        # Conservative: defer both
        deferred.append(component)
        deferred.append(other_component)
        self.logger.warning(
            f"Conflict {comp_idx}-{other_idx}: Both < 3-VP → deferred both (strategy: defer_both)"
        )
    
    processed.add(comp_idx)
    processed.add(other_idx)
```

**0.2 Add Configuration Parameter**

Update `src/config.py` or read from YAML:
```python
class TopologySwitcher:
    def __init__(self, mesh, partition, mesh_topology, deferral_strategy='defer_one'):
        self.deferral_strategy = deferral_strategy  # 'defer_one' or 'defer_both'
        self.deferred_tracker = DeferredComponentTracker() if deferral_strategy == 'defer_one' else None
```

In YAML (`parameters/input.yaml`):
```yaml
type1_migration:
  deferral_strategy: "defer_one"  # Options: "defer_one", "defer_both"
```

### Phase 1: Auxiliary Component Construction

**1.1 Helper Function: `_construct_auxiliary_component_2vp(component)`**
```python
def _construct_auxiliary_component_2vp(self, component: Dict) -> List[int]:
    """
    Construct auxiliary 3-VP component for a 2-VP labeled component.
    
    Returns: [VP_idx_1, VP_idx_2, VP_idx_3] ordered by topology
    """
    vp_indices = component['vp_indices']
    target_vertex = component['target_vertex']
    
    # Get neighbors of the 2 VPs
    # Find the third VP (left or right) closest to target_vertex
    # Return ordered list [left/middle/right VP indices]
```

**1.2 Helper Function: `_construct_auxiliary_component_1vp(component)`**
```python
def _construct_auxiliary_component_1vp(self, component: Dict) -> List[int]:
    """
    Construct auxiliary 3-VP component for a 1-VP labeled component.
    
    Evaluates three candidate triplets:
    - (VP_a, VP_b, VP_c): Two left neighbors
    - (VP_b, VP_c, VP_d): Middle configuration
    - (VP_c, VP_d, VP_e): Two right neighbors
    
    Returns: [VP_idx_1, VP_idx_2, VP_idx_3] for best triplet
    """
    vp_c = component['vp_indices'][0]
    target_vertex = component['target_vertex']
    
    # Get first and second level neighbors on both sides
    # Evaluate total distance to target_vertex for each triplet
    # Return ordered list for triplet with minimum total distance
```

### Phase 2: Topology-Based VP Selection

**2.1 Main Function: `select_migrating_vp_topology_based(component)`**
```python
def select_migrating_vp_topology_based(self, component: Dict) -> int:
    """
    Select migrating VP using topology-based criteria for all component sizes.
    
    - 3-VP: Direct topology analysis
    - 2-VP: Construct auxiliary component, then analyze
    - 1-VP: Construct auxiliary component, then analyze
    
    Returns: VP index that is topologically middle
    """
    size = len(component['vp_indices'])
    
    if size == 3:
        # Build adjacency, find VP with degree 2
        pass
    elif size == 2:
        # Construct auxiliary component
        auxiliary_vps = self._construct_auxiliary_component_2vp(component)
        # Find middle VP
        pass
    elif size == 1:
        # Construct auxiliary component
        auxiliary_vps = self._construct_auxiliary_component_1vp(component)
        # Find middle VP (likely the original 1-VP, but verify)
        pass
```

### Phase 3: Validation and Warning System

**3.1 Function: `_validate_migration_trio(migrating_vp, left, right, target_vertex)`**
```python
def _validate_migration_trio(self, migrating_vp: int, left: int, right: int, 
                             target_vertex: int) -> Tuple[bool, str]:
    """
    Validate that all 3 VPs approach the same target vertex.
    
    Returns: (is_valid, error_message)
    """
    # Check each VP's edge contains target_vertex
    # Check lambda values indicate convergence to target_vertex
    # Return detailed error message if validation fails
```

**3.2 Context-Aware Warning System**
- **In visualization scripts**: Print warnings, annotate figure, continue
- **In optimization scripts**: Print warnings, log to file, halt execution
- **Guaranteed warning production**: Unit tests to verify warnings are generated

### Phase 4: Deferred Component Tracking (Only for "defer_one" strategy)

**4.1 Data Structure: `DeferredComponentTracker`**
```python
class DeferredComponentTracker:
    """
    Track deferred components across iterations (for 'defer_one' strategy).
    
    Stores component signature (not index) to handle unstable indices.
    Only activated when both conflicting components have < 3 VPs.
    """
    def __init__(self):
        self.deferred_components = []
    
    def add_deferred(self, deferred_component: Dict, migrated_component: Dict,
                     shared_neighbor_vp: int, iteration: int):
        """
        Record a deferred component with full context.
        
        Args:
            deferred_component: Component that was deferred
            migrated_component: Component that was allowed to migrate
            shared_neighbor_vp: VP shared between the two components
            iteration: Iteration number when deferral occurred
        """
        # Extract VPs and their edges
        deferred_vps = deferred_component['vp_indices']
        deferred_vp_edges = set()
        for vp_idx in deferred_vps:
            vp = self.partition.variable_points[vp_idx]
            deferred_vp_edges.add(tuple(sorted(vp.edge)))
        
        signature = {
            'target_vertex': deferred_component['target_vertex'],
            'vp_edges': deferred_vp_edges,
            'shared_neighbor_vp': shared_neighbor_vp,
            'shared_neighbor_edge_at_deferral': tuple(sorted(
                self.partition.variable_points[shared_neighbor_vp].edge
            )),
            'migrated_component_target': migrated_component['target_vertex'],
            'iteration_deferred': iteration,
            'size': deferred_component['size']
        }
        self.deferred_components.append(signature)
        
        self.logger.info(
            f"Tracked deferred component: target_vertex={signature['target_vertex']}, "
            f"size={signature['size']}, shared_vp={shared_neighbor_vp}"
        )
    
    def is_blocked(self, component: Dict, current_iteration: int) -> Tuple[bool, str]:
        """
        Check if component should be blocked from migration.
        
        Returns:
            (blocked, reason)
        """
        target_vertex = component['target_vertex']
        component_vps = component['vp_indices']
        
        # Build current component signature
        current_vp_edges = set()
        for vp_idx in component_vps:
            vp = self.partition.variable_points[vp_idx]
            current_vp_edges.add(tuple(sorted(vp.edge)))
        
        # Check against all deferred components
        for def_sig in self.deferred_components:
            # Match by target vertex and VP edges
            if def_sig['target_vertex'] != target_vertex:
                continue
            
            # Check if VP edges match (component identity)
            if def_sig['vp_edges'] == current_vp_edges:
                # This is a previously deferred component
                # Check if shared neighbor has moved
                shared_vp = def_sig['shared_neighbor_vp']
                original_edge = def_sig['shared_neighbor_edge_at_deferral']
                current_edge = tuple(sorted(
                    self.partition.variable_points[shared_vp].edge
                ))
                
                if current_edge != original_edge:
                    reason = (
                        f"Component deferred at iteration {def_sig['iteration_deferred']}. "
                        f"Shared neighbor VP {shared_vp} has moved "
                        f"(edge {original_edge} → {current_edge}). "
                        f"Blocking migration to prevent topology mismatch."
                    )
                    return (True, reason)
        
        return (False, "")
    
    def cleanup_old_deferrals(self, current_iteration: int, max_age: int = 5):
        """Remove deferrals older than max_age iterations."""
        self.deferred_components = [
            sig for sig in self.deferred_components
            if current_iteration - sig['iteration_deferred'] <= max_age
        ]
```

**4.2 Integration in `select_components_for_migration()`**

```python
def select_components_for_migration(self, components: List[Dict], 
                                    conflicts: List[Dict],
                                    current_iteration: int = 0) -> Tuple[List[Dict], List[Dict]]:
    """... existing docstring ..."""
    
    # Check if component is blocked by previous deferral
    if self.deferral_strategy == 'defer_one' and self.deferred_tracker:
        is_blocked, reason = self.deferred_tracker.is_blocked(component, current_iteration)
        if is_blocked:
            self.logger.warning(f"Component {comp_idx} blocked: {reason}")
            deferred.append(component)
            processed.add(comp_idx)
            continue
    
    # ... rest of logic ...
```

**4.3 Tracking Deferred Components**

```python
def _track_deferred_component(self, deferred_comp: Dict, migrated_comp: Dict, 
                              shared_vps: List[int], iteration: int):
    """Helper to track deferred component (only for 'defer_one' strategy)."""
    if self.deferral_strategy == 'defer_one' and self.deferred_tracker:
        # Assume single shared VP (most common case)
        shared_vp = shared_vps[0] if shared_vps else None
        if shared_vp:
            self.deferred_tracker.add_deferred(
                deferred_comp, migrated_comp, shared_vp, iteration
            )
```

**4.4 Enhanced Logging**
```python
def _log_deferral_decision(self, component_a: Dict, component_b: Dict, 
                          action: str, reason: str):
    """
    Log detailed information about deferral decision.
    
    Args:
        action: "migrate_both", "defer_one", or "defer_both"
        reason: Explanation for the decision
    """
    self.logger.info("="*80)
    self.logger.info(f"DEFERRAL DECISION: {action}")
    self.logger.info(f"Reason: {reason}")
    self.logger.info(f"Component A: idx={component_a['index']}, size={component_a['size']}, "
                    f"target_vertex={component_a['target_vertex']}, "
                    f"min_dist={component_a['min_distance']:.6f}")
    self.logger.info(f"Component B: idx={component_b['index']}, size={component_b['size']}, "
                    f"target_vertex={component_b['target_vertex']}, "
                    f"min_dist={component_b['min_distance']:.6f}")
    self.logger.info(f"Shared neighbors: {component_a.get('shared_neighbors', [])}")
    self.logger.info("="*80)
```

### Phase 5: Integration and Testing

**5.1 Update `visualize_type1_vertex_collapse.py`**
- Replace distance-based selection with `select_migrating_vp_topology_based()`
- Add validation before migration
- Display warnings if validation fails

**5.2 Update `apply_type1_switch_v2()`**
- Add validation call before migration execution
- Return validation failure gracefully

**5.3 Update `refine_perimeter.py` (future)**
- Initialize `DeferredComponentTracker`
- Check blocking before migration
- Halt on validation failure

## Next Steps

### Immediate Priority

1. ✓ Document and review problem with stakeholder
2. ✓ Identify deferral logic bug (at least one 3-VP should migrate both)
3. ✓ Identify deferred component tracking issue
4. ✓ Design auxiliary component construction for 1-VP and 2-VP cases
5. ✓ Design both deferral strategies ("defer_one" vs "defer_both")

### Implementation Order

**CRITICAL - Phase 0: Fix Deferral Logic**
6. **Fix `select_components_for_migration()`**: Change "one is 3-VP" logic to "at least one is 3-VP → migrate both"
7. **Add configuration parameter**: `deferral_strategy` ("defer_one" or "defer_both")
8. **Add enhanced logging**: Detailed deferral decision logging
9. **Test deferral logic fix** with existing components

**Phase 1: Auxiliary Component Construction**
10. **Implement `_construct_auxiliary_component_2vp()`**: Find 3rd VP for 2-VP components
11. **Implement `_construct_auxiliary_component_1vp()`**: Evaluate 3 candidate triplets for 1-VP components
12. **Test auxiliary component construction** on sample components

**Phase 2: Topology-Based VP Selection**
13. **Implement `select_migrating_vp_topology_based()`**: Unified selection for all component sizes
14. **Update visualization script**: Replace distance-based selection
15. **Test Component 47** with topology-based selection (expect VP 1804, not VP 1805)
16. **Test Component 46** (similar 3-VP configuration)
17. **Test 1-VP and 2-VP components** with auxiliary component construction

**Phase 3: Validation and Warning System**
18. **Implement `_validate_migration_trio()`**: Check all 3 VPs approach same target vertex
19. **Add context-aware warnings**: Visualization (continue + warn) vs Optimization (halt + warn)
20. **Add unit tests**: Guarantee warning production on validation failure
21. **Test validation** on edge cases and mismatched target vertices

**Phase 4: Deferred Component Tracking (for "defer_one" strategy)**
22. **Implement `DeferredComponentTracker` class**: Component signature storage and matching
23. **Add `is_blocked()` check**: Prevent migration when shared neighbor has moved
24. **Integrate with `select_components_for_migration()`**: Check blocking before migration
25. **Add cleanup logic**: Remove old deferrals after N iterations

**Phase 5: Testing and Comparison**
26. **Test "defer_one" strategy**: With tracking, allow aggressive migration
27. **Test "defer_both" strategy**: Conservative approach, simpler implementation
28. **Run multiple optimization iterations**: Verify no failures in subsequent iterations
29. **Compare perimeter optimization results**: Which strategy produces better final perimeter?
30. **Verify area conservation**: Both strategies maintain area across iterations
31. **Performance testing**: Measure convergence speed for both strategies

**Phase 6: Integration and Documentation**
32. **Update `refine_perimeter.py`**: Add iteration counter, deferral strategy config
33. **Update YAML configuration**: Add `type1_migration` section with `deferral_strategy`
34. **Update documentation**: Implementation details, configuration guide, testing results
35. **Code review**: Ensure robustness and maintainability

### Testing Goals

**Primary Goal**: Determine which deferral strategy produces better perimeter optimization:
- **"defer_one"**: Maximize migration throughput, complex tracking
- **"defer_both"**: Slower but simpler, no tracking needed

**Success Criteria**:
- No white polygons in any component size (1-VP, 2-VP, 3-VP)
- No migration failures in subsequent iterations
- Area conservation maintained
- Detailed logging for all deferral decisions
- Clear warnings on validation failures
