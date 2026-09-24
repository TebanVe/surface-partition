---
paths:
  - "src/partition/balanced_readout.py"
  - "scripts/balanced_readout.py"
  - "testing/test_balanced_assignment_solver.py"
  - "testing/test_disconnected_cells_detection.py"
---

# The balanced readout — the Phase 1 → Phase 2 bridge

Closes the winner-take-all gap **at extraction time**, in two stages, without
touching the source solution (it only ever creates a new campaign directory
under `readout/`, mirroring `refinement/`).

1. **Dual shifts.** Replace `argmax_k u_ik` with `argmax_k [log u_ik + psi_k]`,
   the N per-cell offsets `psi` solving the transportation dual by subgradient
   ascent so every cell's discrete area hits target — balanced to one-vertex
   granularity **at any N** (tail-immune, unlike any soft or variance-reducing
   mechanism). Raising `psi_k` grows cell k outward from its core, so the
   correction is **local** — the property the discrete-area trim lacked, which
   is why the trim manufactured islands and this does not.
2. **Connectivity repair.** Stray components are absorbed by the neighbour
   sharing the longest boundary, then equal areas are restored by single-vertex
   boundary transfers (improving iff `T_donor - T_receiver > v_i`, ranked by
   `log u_ir - log u_id`), each gated by an exact articulation check so no move
   disconnects a donor.

Both invariants hold by construction on exit. The stage reports its own strain —
`n_moves`, `n_blocked_by_connectivity`, `sweeps_used`, `hit_sweep_cap` — so a run
beyond what repair can fix says so.

Derivation: `docs/math/09-balanced-readout/`. Plain-language:
`docs/explanations/balanced_readout_explained.md`. Standing explanation:
`docs/reference/winner_take_all_partition_gap.md` §9b.

## `solve_dual_offsets` is the SHARED assignment primitive (A, B and C)

It takes any additive per-vertex score matrix — A: `log u`; B: the diffused `y`;
C: `-d^2` — and touches it only through `argmax(scores + psi)`.

Correctness is scale-free but **convergence is not**, so `normalize_scores`
(default **off**, keeping approach A byte-identical) divides by a robust
assignment-margin scale, applies `normalized_eta0` (2.0, against `dual_eta0` 0.5
which was tuned for log-density scale) and rescales `psi` back.

⚠ The gate for this solver (`testing/test_balanced_assignment_solver.py`) once
contained a **1.25x strong-reference term computed with the solver under test**;
six crippled solvers passed it, including one returning `psi = 0`. It was
removed. Gate 3 now asserts a null solver FAILS. Any new bar for this primitive
must be computed by something other than the primitive.

## The dual shifts can WORSEN connectivity en route

In `run_20260806_123326` the two fragmented cells go **2 -> 4 components** after
the shifts, before repair removes them. **The repair stage is necessary, not
cosmetic** — never run `--no-repair` and treat the output as a deliverable.

## `max_repair_sweeps` default is 200 (raised from 50 on 2026-08-15)

It is a **ceiling, never a schedule** — the sweep loop breaks as soon as no
candidate move improves — so any readout that converged in under 50 sweeps is
byte-identical across the change. Only readouts that reported
`hit_sweep_cap: true` differ, and for those the extra sweeps can only move
further toward equal area.

**Reproducing an archived capped readout requires `--max-repair-sweeps 50`.**
Each campaign's `readout.yaml` records the value actually used.

## Relabelings are encoded as density SWAPS

Downstream consumers read densities only through `argmax`
(`compute_indicator_functions` builds a hard 0/1 indicator), so the stage swaps
the winner's and target cell's density values at each relabeled vertex. Row sums
— and hence partition-of-unity — are untouched, and the source densities are
**exactly** recoverable by swapping back using the stored
`labels_source`/`labels_final` (verified: max abs diff 0.0). `psi` is stored for
provenance.

Output uses the **Phase 1 solution schema**, so `refine_perimeter.py`,
`visualize_partition_fast.py`, `export_partition.py` and
`testing/check_fragmentation.py` consume it unchanged, with no new flags.

## Measured at torus N=300

`run_20260806_123326`, lambda=11.5: **10 imbalanced / worst 36.15% / 2
fragmented -> 0 / 1.63% / 0**, +1.88% label-boundary length, 1,840 vertices
relabeled, ~17 s. Its 19-iteration Phase 2 reaches perimeter **323.3192**.

## Exact balance is generically unattainable, and that is fine

Vertex masses are real, so the transportation LP optimum is fractional and
splits at most N-1 vertices (`docs/math/09-balanced-readout/`). ⚠ The vertex
mass `max v_i / Abar` is a **scale for calibrating bars, NOT a lower bound** — an
earlier draft claimed it was a floor and that was false, with achievable
deviation 7 orders of magnitude below the claimed floor.

Note also that **disconnection is admissible** for an argmax super-level set —
nothing in the relaxed energy penalizes it — which is why connectivity needs an
explicit repair rather than falling out of the optimization.
