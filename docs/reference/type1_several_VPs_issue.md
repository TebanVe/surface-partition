# Type 1 Migration: Triggers with >3 Approaching VPs

## Issue Summary

During Phase 2 refinement on the double torus (marching cubes mesh, `ngx=64`),
Type 1 migration triggers are detected with 4 or 5 approaching variable points
instead of the expected 3. This document records the investigation findings and
outlines the safety guard that should be implemented.

## Resolution (2026-07 — implemented)

**This issue is now RESOLVED.** The triple-point exclusion guard has been ported
to the new detector path: `detect_type1_triggers()` in
`src/migration/migration_detector.py` takes a `steiner_handler` argument and
rejects any Type 1 candidate whose vertex participates in an existing
triple-point triangle, or whose approaching VPs belong to one. It is wired in
`MigrationOrchestrator` (`migration_orchestrator.py`, passing
`steiner_handler=self.steiner_handler`) and documented in CLAUDE.md's "Migration
Types" section. The investigation below is retained as the record of *why* the
guard exists; note that the legacy `Type1ComponentAnalyzer` it references has
since been **removed** from the codebase. The "What's missing", "Recommended
Implementation", and "Status" sections below describe the pre-fix state and are
superseded by this note.

## Context

A Type 1 migration is a 2-cell operation: a mesh vertex transitions from its
current cell to a neighboring target cell. In the standard case, the boundary
"closes in" around the vertex on 3 edges of its one-ring, producing exactly 3
approaching VPs (all with `min(λ, 1-λ) < delta`). The vertex is then absorbed
by the target cell.

## Observed Behavior

In the refinement log for:
```
results/double_torus_npart10/run_20260417_175907_.../refinement/ipopt_btol0.001_lbfgs30_hess_bestiter_partial/
```

Triggers were logged with 4 and 5 approaching VPs:
```
Type 1 trigger: vertex 850 (7 -> 5), 5 approaching VPs, min_dist=0.0000
Type 1 trigger: vertex 5378 (2 -> 8), 5 approaching VPs, min_dist=0.0000
Type 1 trigger: vertex 11836 (8 -> 6), 4 approaching VPs, min_dist=0.0000
...
```

## Diagnostic Investigation

A diagnostic script (`testing/test_type1_triple_point_overlap.py`) was created
to cross-reference >3 VP triggers against triple-point triangle vertices.

### Running the diagnostic

```bash
python testing/test_type1_triple_point_overlap.py \
    --solution <path_to_refined_iteration.h5> \
    --boundary-tol 0.001
```

### Key Findings

On the iteration_001 file from the above run:

- **Total Type 1 triggers detected**: 111
- **With exactly 3 VPs**: 86
- **With >3 VPs**: 25 (including 2 with 5 VPs)
- **Of >3 VP triggers in triple-point triangles**: 0 / 25
- **Of 3 VP triggers in triple-point triangles**: 0 / 86

**Conclusion**: None of the >3 VP triggers are at triple-point vertices.
All of them have only **1 distinct non-current cell** in their one-ring,
meaning they are genuine 2-cell Type 1 situations — not misidentified
triple points.

### Root Cause

The >3 VP triggers occur on **high-valence vertices** produced by marching
cubes meshing. Unlike structured parametric meshes (e.g., torus) where vertex
valence is uniformly 6, marching cubes meshes have irregular connectivity.
Some vertices have valence 7-8, and when such a vertex is deeply enclosed
by the target cell (4-5 of its neighbors already belong to cell B while
the vertex is still labeled cell A), 4-5 boundary edges produce 4-5
approaching VPs.

These are valid Type 1 candidates — the vertex is simply more isolated
than a typical 3-VP case. The migration at these vertices is geometrically
correct (only 2 cells involved), just more aggressive.

## Triple-Point Exclusion Guard

### What exists (legacy path)

The legacy `Type1ComponentAnalyzer` (in `src/migration/type1_component_analyzer.py`)
implements a triple-point exclusion guard:

- `_get_triple_point_vps()` (line 155): collects all VP indices belonging to
  triple-point triangles via `SteinerHandler`.
- `_component_near_triple_point()` (line 172): for components with < 3 VPs,
  checks if non-boundary neighbors connect to triple-point VPs.
- `select_components_for_migration()` (line 998): excludes components marked
  `near_triple_point = True`.

Note: this guard only protects components with **< 3 VPs** (line 192:
`if component['size'] >= 3: return False, []`). Larger components are
considered safe because their internal neighbors won't affect triple-point VPs.

### What's missing (new orchestrator path)

The refactored `MigrationOrchestrator` + `detect_type1_triggers()` path in
`src/migration/migration_detector.py` does **not** implement any triple-point
exclusion. The only interaction between Type 1 and Type 2 is the conflict
resolution in `resolve_conflicts()`, which gives Type 2 priority when both
triggers fire at the same vertex/neighborhood — but does NOT preemptively
exclude Type 1 candidates near triple points.

### Recommended Implementation

A safety guard should be added to `detect_type1_triggers()` in
`src/migration/migration_detector.py` to:

1. Accept a `SteinerHandler` or set of triple-point VP indices as input.
2. For each Type 1 candidate vertex, check if any of its approaching VPs
   are also triple-point VPs (i.e., belong to a triple-point triangle).
3. If so, reject the trigger (or at minimum, log a warning and skip it).

This mirrors the logic in the legacy `_component_near_triple_point()` but
adapted for the trigger-based detection model.

## Related Files

| File | Role |
|------|------|
| `src/migration/migration_detector.py` | Type 1 detection — **now has the triple-point guard** (`detect_type1_triggers(steiner_handler=…)`) |
| `src/migration/migration_orchestrator.py` | Orchestrator — passes `steiner_handler` into the detector |
| `testing/test_type1_triple_point_overlap.py` | Diagnostic script created for this investigation |

(The legacy `src/migration/type1_component_analyzer.py`, which originally carried
the guard, has been removed.)

## Status

- **RESOLVED** (see the Resolution note at the top): the triple-point exclusion
  guard is implemented in `detect_type1_triggers(steiner_handler=…)` and wired in
  the orchestrator.
- **Diagnostic complete**: confirmed the >3 VP triggers were NOT at triple points
  for this run — they are valid 2-cell migrations on high-valence marching-cubes
  vertices.
- The latent risk the guard protects against — a vertex that IS part of a
  triple-point triangle being handled as a Type 1 migration — is now guarded.
