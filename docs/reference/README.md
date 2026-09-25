# `docs/reference/` — permanent explanations

Standing documents: methodology, known-issue investigations, primers, and
schema specifications. A reference doc explains something that remains true;
the *measurement* behind it usually lives in `docs/experiments/` and the two
cross-link.

| Document | What it holds |
|---|---|
| [`winner_take_all_partition_gap.md`](winner_take_all_partition_gap.md) | **The central high-N document.** Continuous mass vs discrete territory; dormant / runt / split cells; §4b the split-cell mechanism and the N=300 ladder reconstruction; §4c the locality criterion any replacement must pass; §9b the balanced readout |
| [`PHASE1_HIGHN_APPROACHES_ABCDE.md`](PHASE1_HIGHN_APPROACHES_ABCDE.md) | The standing taxonomy of high-N approaches A / A2 / B / C / D / E, with status. **Read before proposing anything in this space.** Do not confuse A–E with the P1–P5 enumeration in `plans/PHASE1_N1000_VALIDITY_PLAN.md` |
| [`phase1_highn_proposal_ABCDE_original.md`](phase1_highn_proposal_ABCDE_original.md) | The source proposal behind the taxonomy, archived verbatim |
| [`phase1_energy_discretization_bug.md`](phase1_energy_discretization_bug.md) | The mis-discretized double well (`u²(1−u)²` instead of `q=u(1−u)`), a typo copied from the paper; fixed in `6ff71a0` |
| [`phase1_dual_projection_negative_result.md`](phase1_dual_projection_negative_result.md) | Exact dual projection measured and found *not* to be a speedup |
| [`PHASE1_PGD_SERIAL_OPTIMIZATION_AUDIT.md`](PHASE1_PGD_SERIAL_OPTIMIZATION_AUDIT.md) | Audit of the Phase 1 PGD serial optimizations (Changes A/B/C) |
| [`PHASE1_RELAXATION_TIMING_PROFILE.md`](PHASE1_RELAXATION_TIMING_PROFILE.md) | Where Phase 1 wall time goes (the projection bottleneck) |
| [`TOPOLOGY_SWITCH_METHODOLOGY.md`](TOPOLOGY_SWITCH_METHODOLOGY.md) | Type 1 and Type 2 migrations: detection, execution, rollback |
| [`type1_several_VPs_issue.md`](type1_several_VPs_issue.md) | Type 1 triggers with more than three approaching VPs |
| [`IPOPT_REFINEMENT_QUALITY.md`](IPOPT_REFINEMENT_QUALITY.md) | Phase 2 refinement quality: diagnosis and remediation options |
| [`OPTIMIZATION_METHODS_PRIMER.md`](OPTIMIZATION_METHODS_PRIMER.md) | SLSQP, IPOPT, L-BFGS, exact Hessian — background primer |
| [`SCALABILITY_ANALYSIS.md`](SCALABILITY_ANALYSIS.md) | How perimeter optimization with IPOPT scales |
| [`PARTITION_EXPORT_SCHEMA_GENERAL.md`](PARTITION_EXPORT_SCHEMA_GENERAL.md) | The general-surface export schema (`schema_version` 2.0). The namespace is **forked, not bumped** — torus stays on 1.1 byte-identical |
| [`MD_SIMULATION_EXPORT_NOTES.md`](MD_SIMULATION_EXPORT_NOTES.md) | Exporting the partition for molecular simulations |
| [`DOCUMENTATION_SYNC_RULE.md`](DOCUMENTATION_SYNC_RULE.md) | How the doc-sync rule actually triggers, the one case it misses, and the decision to accept that gap |
| [`DOWNSTREAM_CONSUMERS.md`](DOWNSTREAM_CONSUMERS.md) | **Who consumes the exported partitions**, what each reader enforces, and what must therefore not change. Holds the collapsed-faces decision (2026-09-25: do not clean the export) |
| [`deliverables.yaml`](deliverables.yaml) | The record of every exported partition. Verify with `python scripts/check_deliverables.py` |
| [`figures/`](figures/) | Shared figures referenced by the documents above |
