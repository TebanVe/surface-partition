# `docs/plans/` — design plans for work not yet implemented

A plan describes intended work. When it is fully implemented it should be
**deleted**, or have its lasting explanation moved to `docs/reference/` — a plan
left standing after the fact reads as current intent.

Implemented and removed: **general-surface export** (`0035dde`, 2026-09-08) —
its findings and remaining hand-off now live in
[`../reference/PARTITION_EXPORT_SCHEMA_GENERAL.md`](../reference/PARTITION_EXPORT_SCHEMA_GENERAL.md) §1a–1b.

| Plan | Status |
|---|---|
| [`PHASE1_BC_REPLACEMENT_PLAN.md`](PHASE1_BC_REPLACEMENT_PLAN.md) | **The active forward plan.** Shared balanced-assignment solver and evaluation harness (Phase 0), then approach C, then approach B, with pre-registered falsifiers anchored on the N=100 deliverable (Phase 2 perimeter 185.2546144718457). Phase 0 and Phase A are done and measured |
| [`PUBLICATION_READINESS_PLAN.md`](PUBLICATION_READINESS_PLAN.md) | Whether this is an independently publishable project: what is established and what may not be claimed, plus a prioritised programme. Phase 4 holds the finding that **Phase 2 is now the bottleneck** |
| [`MESH_DEGENERACY_AND_NEEDLE_TRIANGLES.md`](MESH_DEGENERACY_AND_NEEDLE_TRIANGLES.md) | Marching-cubes mesh cleanup. ⚠ Its suggested L1 knob (raising `n_grid_z`) is **measured to make things worse** — see `.claude/rules/surfaces.md` |
| [`PIPELINE_EXPORT_INTEGRATION.md`](PIPELINE_EXPORT_INTEGRATION.md) | Integrating the export stage into the pipeline |
| [`PHASE1_N1000_SCALING_PLAN.md`](PHASE1_N1000_SCALING_PLAN.md) | Sparse representation, GPU backend, projection redesign. Predates approach B |
| [`PHASE1_N1000_SCALING_PLAN_REVIEW.md`](PHASE1_N1000_SCALING_PLAN_REVIEW.md) | Adversarial review of the above |
| [`PHASE1_N1000_VALIDITY_PLAN.md`](PHASE1_N1000_VALIDITY_PLAN.md) | Territory-aware relaxation (P1–P5). ⚠ **The approach was implemented, measured and removed** — see `.claude/rules/phase1-pgd.md` and `experiments/04-territory-aware-highn-validation/` |
| [`AI_AGENT_INTEGRATION_PLAN.md`](AI_AGENT_INTEGRATION_PLAN.md) | Agent integration |
