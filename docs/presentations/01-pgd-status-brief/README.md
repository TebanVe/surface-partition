# 01 — PGD status brief (up to N = 300)

**Status:** draft (2026-09-16), not yet given.

**Audience and occasion.** A briefing for a research collaborator who is not
working in the codebase: what the pipeline does, where the relaxation (PGD)
works, what it costs, which artefacts appear as the cell count grows and why,
and how the balanced readout corrects them to reach a valid 300-cell partition.
Deliberately stops there — nothing about the Phase 1 replacement work (approach
B / auction-dynamics MBO) or any n above 300. Intended to be presented live
(13 slides); the slides carry enough text to be read alone.

**Notation and provenance policy.** The slides use the notation of the paper the
repository implements (Bogosel & Oudet, *Partitions of minimal length on
manifolds*, `docs/papers/manifolds_perimeter.pdf`): `n` cells `(ω_i)`, `N` mesh
vertices, nodal vectors `φ^i`, the relaxed functional `F_ε` on `X`, the readout
(5.1), variable points `x_i = λ_i v_1 + (1−λ_i) v_2`. The FEM slide states the
three energy terms and their gradients exactly as `ProjectedGradientOptimizer`
computes them (the penalty is the code's normalised form of the paper's
`λ(std − s_target)²`; the potential uses `w = φ ⊙ (1−φ)`, not the paper's
squared-twice typo). The deck is illustrative and carries **no run IDs and no
source lines on the slides** — the mapping from every claim to its source is
kept here instead, as the folder's rule requires.

**Build.** `make` here (or `make 01-pgd-status-brief` in `docs/presentations/`)
produces `main.pdf`; `make png` renders one PNG per slide for review;
`make figures` regenerates every `fig_*` from `results/` (needs the project
environment with PyVista; ~15 min, dominated by contour extraction on the raw
N = 300 solution).

## Provenance

Every number on the slides traces to one of these. Wall times are **full-ladder
sums** of `level_wall_s` over `solution/timing_profile.yaml`, across the two
`run_*` directories where a ladder was resumed (the `run_time_seconds` field is
the last level only — see the CLAUDE.md gotcha).

| slide | claim | source |
|---|---|---|
| 2, 3, 4, 5 | problem statement, relaxed functional, FEM terms and gradients, pipeline | `docs/math/06-phase1-energy-discretization/`, `docs/reference/winner_take_all_partition_gap.md` §1, CLAUDE.md "Data Flow" |
| 2 | N = 100 render | `results/run_20260709_081548…/refinement/ipopt_btol0.001_lbfgs30_hess_bestiter_partial/iteration_020_20260709_222553.h5` (best iterate) |
| 5 | N = 200 render | `results/run_20260722_175451…/refinement/ipopt_btol0.001_lbfgs30_hess_bestiter_partial/iteration_016_20260723_152237.h5` (best iterate) |
| 6, 13 | N = 100 gates, worst cell 0.78 %, perimeter 185.2546 | `results/run_20260709_081548…` (`solution/metadata.yaml`; gates re-run 2026-09-16 with `testing/check_fragmentation.py`) |
| 6, 13 | N = 150 gates, worst 1.24 %, perimeter 228.1566 | `results/run_20260710_215525…` (levels 0–2) + `run_20260711_165615…` (levels 3–4); gates re-run as above |
| 6, 13 | N = 200 gates, worst 1.65 %, perimeter 262.1096 | `results/run_20260713_211827…` (levels 0–2) + `run_20260722_175451…` (levels 3–4); gates re-run as above |
| 6 | seeded init, λ window, seed dependence | `docs/experiments/02-corrected-energy-highn-validation/`; CLAUDE.md "Phase 1 Initial Condition", "λ window" gotcha |
| 7 | Phase 1 wall 13.4 / 17.0 / 35.8 / 79.3 h | `solution/timing_profile.yaml` of the runs above plus `run_20260806_123326…` + `run_20260808_191030…` for N = 300; computed by `make_figures.py` |
| 7 | projection = 93 % of Phase 1, 46 inner iterations | `docs/math/04-phase1-timing-profile/` (N = 300, λ = 11.5, first three levels) |
| 7 | level-0 cap waste, 13.4 h → 9.7 h with the structure trigger | CLAUDE.md "Structure-Based Refinement Trigger" (measured end to end at N = 100) |
| 9 | dormant / runt / split table | `docs/reference/winner_take_all_partition_gap.md` §2, §4b; `docs/experiments/01-winner-take-all-partition-gap/` |
| 10 | territory-vs-mass figure | `results/run_20260806_123326…/solution/surface_part300….h5` and its `readout/dualshift_gate0.05_repair/solution_balanced.h5`, gates computed by `make_figures.py` via `src/partition/find_contours.py` |
| 10 | fragmentation born mid-level, λ ceiling | CLAUDE.md "N=300's ladder" and "λ window" gotchas; reference doc §9 |
| 11 | N = 300 raw: 10 imbalanced / −36.1 % / 2 fragmented (47,488 V); 3 / −24.8 % / 2 (114,144 V) | `solution/metadata.yaml` of `run_20260806_123326…` and `run_20260808_191030…`; signs checked from `discrete_areas` |
| 11 | coarse levels dying at 32 / 83 verts per cell | CLAUDE.md "N=300's ladder shows decelerating returns" |
| 11, 12 | N = 300 renders (raw, readout) | same two `.h5` files as slide 10; highlighted cells = `imbalanced` ∪ `fragmented` of the raw gates |
| 12 | readout method and 10/−36.1 %/2 → 0/–/2 → 0/1.63 %/0, 1,840 relabelled, +1.9 %, ~17 s | `readout/dualshift_gate0.05_repair/metadata.yaml` in `run_20260806_123326…`; `docs/math/09-balanced-readout/`; CLAUDE.md `BalancedReadoutConfig` row |
| 12, 13 | Phase 2 323.3192 (47k) and 322.9622 (114k), worst 0.64 % | CLAUDE.md deliverable tables; `run_20260808_191030…/readout/…/metadata.yaml` |

## Figures

All produced by `make_figures.py` (committed beside this file):

| file | what | source |
|---|---|---|
| `fig_partition_n100.png` | N = 100 after Phase 2, exact contours, neighbour-distinct fills | slide 2 row above |
| `fig_partition_n200.png` | N = 200 after Phase 2 | slide 5 row above |
| `fig_phase1_wall.svg` | Phase 1 wall per N, full ladder | slide 7 row above |
| `fig_territory_n300.svg` | per-cell territory / target, raw vs readout, with continuous mass | slide 10 row above |
| `fig_n300_raw.png` | raw n = 300 readout, imbalanced (red) and fragmented (orange) cells on grey | slide 11 row above |
| `fig_n300_readout.png` | same cells after the balanced readout | slide 12 row above |

Renders use `scripts/visualize_partition_fast.py`'s exact-geometry region
builder (cell portions of boundary triangles, Steiner triangles), so the drawn
boundaries are the Phase 2 contours, not a per-vertex colouring. The N = 300
pair shares one fixed camera (azimuth −50°, elevated) chosen to face the two
fragmented cells and the worst runts, so before and after are directly
comparable; on the raw figure the stray piece of one fragmented cell is the
small orange patch beside the red cells near the hole. Renders are auto-cropped
to their alpha bounding box.
