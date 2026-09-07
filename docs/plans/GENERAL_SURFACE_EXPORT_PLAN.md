# General-Surface Partition Export — Implementation Plan

**Status:** Not Started

The schema this implements is specified in
`docs/reference/PARTITION_EXPORT_SCHEMA_GENERAL.md`. Read that first — this
document is only the steps, and deliberately does not restate the format.

## Background

Phase 1 (approach B), Phase 2, all three validity gates, the balanced readout and
both viewers now work on any of the four surfaces. Two valid non-torus partitions
already exist and are reproducible bit-identically:

| run | surface | N | V | worst cell | fragmented |
|---|---|---|---|---|---|
| `results/other_surfaces/arm_mbo_20260907_164351_npart10_V12448_seed84172851` | double torus | 10 | 12,448 | 0.0468% | 0 |
| `results/other_surfaces/arm_mbo_20260907_164358_npart10_V26928_seed84172851` | Banchoff-Chmutov | 10 | 26,928 | 0.0169% | 0 |

**Export is the only remaining torus-only stage.** `src/export/writer.py` reads
`config["surface"]["torus"]`, writes `R`/`r`/`grid_shape`/`vertex_order`, and
asserts `n_theta * n_phi == V`.

The work is small — `src/export/rep3_builder.py` is already entirely
surface-agnostic, and all nine datasets in the exported file are general. What
takes care is not the code but *not disturbing the torus path*, which a
downstream project consumes as a stable contract.

## Phase 1 — Writer accepts a surface descriptor
**Status:** Not Started

In `src/export/writer.py`:

1. Replace the `config["surface"]["torus"]` read with `src/surfaces/factory.py`
   (`surface_name_from_config`, `resolve_surface_params`, `is_structured`).
2. Branch on the surface:
   - **torus** → emit exactly what it emits today, including
     `schema_version = "1.1"`, `R`, `r`, `grid_shape`, `vertex_order`, and keep
     the `n_theta * n_phi == V` assertion **enforced**.
   - **anything else** → `schema_version = "2.0"`, no `R`/`r`/`grid_shape`/
     `vertex_order`, plus the `/surface` group from the spec; skip the product
     assertion (it is meaningless on a marching-cubes mesh).
3. Compute `genus` / `euler_characteristic` from the exported mesh rather than
   hardcoding per surface — it is a few lines and doubles as a validity check
   that the mesh is a closed surface.
4. `scripts/export_partition.py`: drop the torus-only `--config` assumption in
   its docstring/usage and pass the surface through. `source_run_id` already
   falls back to the `arm_*` directory name, so arm runs need no change.

**Acceptance:** re-export an existing torus deliverable and require the result to
be **byte-identical** to the committed file. This is the same regression pattern
that verified the surface-agnostic MBO driver (commit `386b7a7`), and it is the
gate that protects the downstream contract. If a re-export is not byte-identical,
the change is wrong — do not rationalise a diff.

## Phase 2 — Export the two existing partitions
**Status:** Not Started

Export both runs above at their **best** Phase 2 iterate (not the last — both sit
on the migration-cycling plateau with `pending_migration=True` throughout, so
`--force-finalised` is required):

| surface | best iterate | perimeter |
|---|---|---|
| double torus | `iteration_015_20260907_170820.h5` | 25.491911 |
| Banchoff-Chmutov | `iteration_020_20260907_170922.h5` | 61.449188 |

These become the fixtures the downstream general-surface worktree develops its
reader against — considerably more useful than a hypothetical file.

## Phase 3 — Hand-off
**Status:** Not Started

Give the downstream worktree the spec plus the two files. Both items that were
open when this plan was first written have since been **measured and closed** —
they are recorded here so the downstream side inherits the answers, not the
questions.

1. **Residual tolerance — ANSWERED.** Test `|f|/|grad f|` (a distance, comparable
   across surfaces), never raw `|f|` (arbitrary per-surface scaling: 2.7e-3 vs
   2.4e-2 at comparable density). The normalised residual obeys `C * h^2` with `C`
   constant to 5% down both ladders and O(1) on both surfaces (3.2 and 0.95), so
   the check is `max |f|/|grad f| < K * voxel_size^2` with **`K = 10`**. This needs
   `voxel_size` in the file — added to the `/surface` group in the spec. The
   torus's `1e-10` cannot be reused: an exactly-parametrised torus sits at
   **4.4e-16** (machine precision, resolution-independent), a marching-cubes vertex
   near **1e-3**. Thirteen orders apart. Full table in the spec, §5.1.

2. **Mesh-quality thresholds — NOT AN ISSUE.** `_check_mesh_quality` **warns, it
   does not raise**, and *the accepted torus deliverables already breach both
   thresholds*: on the Rep-3 subdivided mesh the shipped n=25 / n=50 / n=200
   partitions all read `min_rel_area = 0.0` and `min_angle = 0.0 deg` (4 of 248,826
   and 2 of 269,832 sub-triangles exactly degenerate) against `1e-6` / `1.0 deg`.
   That is a property of Rep-3 subdivision — a variable point landing on a mesh
   vertex gives a zero-area sub-triangle — not of the surface. General surfaces are
   no worse than production torus files here. Recalibrating the thresholds (they
   look calibrated for a base mesh, not a subdivided one) is downstream cosmetics.

## Deferred — convergence of 1.1 into 2.0
**Status:** Not planned

Whether the torus eventually migrates to 2.0 (with `R`/`r`/`resolution` living in
`/surface/params` like every other surface) is a cross-repo decision and is
deliberately left open. The spec is written so that convergence is possible later
without redesign: the torus's own fields have a defined 2.0 spelling. Until both
sides want it, torus stays on 1.1 and nothing moves.

## Related documents

- `docs/reference/PARTITION_EXPORT_SCHEMA_GENERAL.md` — the schema this implements.
- `docs/plans/PIPELINE_EXPORT_INTEGRATION.md` — how the existing exporter was integrated.
- `docs/plans/MESH_DEGENERACY_AND_NEEDLE_TRIANGLES.md` — the mesh-quality background behind Phase 3's open questions.
- Code: `src/export/writer.py`, `src/export/rep3_builder.py`, `scripts/export_partition.py`, `src/surfaces/factory.py`.
