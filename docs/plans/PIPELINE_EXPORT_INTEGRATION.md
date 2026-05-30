# Pipeline Integration of the Partition Export

**Status:** Planned. Prerequisite: `scripts/export_partition.py` must be
production-tested on at least one run before pipeline integration is attempted.
**Audience:** Future agent / developer adding automatic export at the end of
Phase 2.
**Related:** `src/export/`, `scripts/export_partition.py`,
`scripts/refine_perimeter.py`,
`../link-list-torus/docs/design/PARTITION_INPUT_ASSESSMENT.md`.

---

## TL;DR

`src/export/export_partition()` was deliberately written to accept already-
loaded Python objects (mesh, partition, Steiner handler, config) rather than
opening files. That makes it callable from both the standalone CLI
(`scripts/export_partition.py`, available today) and the Phase-2 pipeline
itself (this document). No refactoring of `src/export/` is required.

This document specifies the wiring that turns the standalone CLI into an
opt-in pipeline step.

---

## What to implement

1. **CLI flag** on `scripts/refine_perimeter.py`:

   ```
   --export / --no-export
   ```

   Default off. When on, the export runs once at the end of the refinement
   loop. The flag also accepts a YAML value via `RefinementConfig`.

2. **Config field** on `RefinementConfig`
   (`src/pipeline/pipeline_orchestrator.py`):

   ```python
   export: bool = False
   ```

   Read in `from_yaml_dict()` from the `refinement.export` key; overridable
   on the CLI.

3. **End-of-loop call** in
   `PipelineOrchestrator.run_refinement_loop()`. After the loop exits
   (whether converged or hit `max_iterations`), if `self.config.export`:

   - Resolve the run YAML by reading `<run_dir>/experiment.yaml`
     (or carry it through from `find_surface_partition.py`).
   - Read `R`, `r`, `n_theta`, `n_phi` from `config['surface']['torus']`.
   - Resolve `source_run_id = Path(run_dir).name`.
   - Resolve `source_iteration` from the final-iteration counter the loop
     already maintains.
   - Read `seed` from the base solution HDF5 attributes (already loaded
     during the run; cache it at start of `run_refinement_loop`).
   - Call:

     ```python
     export_partition(
         partition=self.partition,
         mesh=self.mesh,
         steiner_handler=self.steiner_handler,
         config=experiment_config,
         output_path=output_path,
         source_run_id=source_run_id,
         source_iteration=source_iteration,
         seed=seed,
         final_perimeter=final_perimeter,
         pending_migration=pending_migration,
         strict=False,
     )
     ```

   `strict=False` is intentional: a run that exhausts `max_iterations` may
   still have `pending_migration=True` at the final checkpoint, and we still
   want a file written (with `finalised=False`) for downstream inspection.

4. **Output path convention:**

   ```
   <run_dir>/refinement/<campaign>/torus_partition_<run_id>.h5
   ```

   This places the export alongside the iteration checkpoints in the same
   campaign directory. The standalone CLI already defaults to this layout
   when invoked without `--output`.

---

## What NOT to do

- Do **not** refactor `src/export/` — its signature and split (rep3 builder
  vs. writer) was chosen to support this integration without changes.
- Do **not** make the export non-optional. The standalone CLI must keep
  working for ad-hoc exports of older runs.
- Do **not** silently overwrite an existing file at the output path; either
  add a `--force` flag or fail with a clear message. (Pick one when
  implementing — `--force` is the convention used in the standalone CLI's
  parent scripts.)
- Do **not** import `cyipopt`, `pyvista`, or other optional dependencies
  from the export module. The export should run in the core install.

---

## Verification

The change is verified end-to-end by a Phase-2 run with `--export` that
produces a `torus_partition_<run-id>.h5` alongside the iteration checkpoints
and passes the structural check from §11 of the original task prompt
(`prompts/prompt.md`).
