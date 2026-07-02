# Adversarial Review: PHASE1_N1000_SCALING_PLAN.md

**Reviewed document:** `docs/plans/PHASE1_N1000_SCALING_PLAN.md` (commit 563e905)
**Stance:** adversarial — every load-bearing assumption independently re-derived
from the two cited timing profiles, the source code, doc 05, and (for the
time-sensitive framework/cluster claims) current public documentation.
**Review date:** 2026-07-02

**Evidence base:**
`results/run_20260627_234511_.../solution/timing_profile.yaml` (N=50),
`results/run_20260629_141012_.../solution/timing_profile.yaml` (N=100),
`src/optimization/pgd_optimizer.py`, `src/optimization/projection.py`,
`src/pipeline/relaxation.py`, `src/profiling.py`,
`docs/math/05-phase1-nregion-scaling/main.tex`, `parameters/torus_50part.yaml`.

---

## 0. Overall verdict

The plan's **go/no-go conclusion survives this review**: dense V×N is dead at
N=1000 (arithmetic verified), the projection redesign is genuinely mandatory
for sparsity (verified in code), and N=1000 lands in the hours-to-days range
under every K scenario the data admits. But the review **refutes the plan's
headline empirical finding as stated**: "K did not grow from N=50 to N=100" is
an artifact of iteration-count redistribution across refinement levels, driven
by a coarse-level line-search breakdown at N=100 that the plan's own per-level
data reveals but the plan does not confront. The like-for-like local exponent
is **α ≈ 2.0, not 1.8**, and per-level K grows as roughly **N^0.5–0.8** — the
"K saturates" scenario (and its 0.5–1.5 h headline row) should be discarded as
a planning basis. Several secondary numbers are wrong in ways that happen to
cancel in the totals but corrupt derived claims (most notably the BB stepping
win: **~1.9×, not ~2.9×** — Phase 4's own gate is unreachable by its claimed
mechanism). The Pelle GPU "unverified assumption" is publicly verifiable and
resolves favorably (GPUs exist), though the realistic workhorse is L40s, not
H100, which stretches the wall-time table by ~3–4×.

Per-assumption verdicts:

| # | Assumption | Verdict |
|---|---|---|
| 1a | N=50/N=100 is a controlled same-mesh/same-seed pair | **SOUND** |
| 1b | "K did not grow" (total K flat ⇒ saturation) | **WRONG as stated** (redistribution artifact; per-level K grew ×1.39–1.71) |
| 1c | α_local ≈ 1.81 | **OPTIMISTIC** (arithmetically correct; like-for-like ≈ 2.0; two points support no exponent) |
| 2a | Memory wall: 9.13 GB/array, 73–91 GB working set, 18.3 GB file | **SOUND** (reproduced exactly) |
| 2b | ~11 GB/s bandwidth-bound projection kernel | **SOUND** (reproduced: 35.2 ms/inner-iter, 38 MB matrix) |
| 2c | Per-call means (211 ms / 779 ms) and calls = K·(mBT+1) | **WRONG** (calls = K·mBT; means are 322 ms / 1,177 ms; errors cancel in totals) |
| 2d | BB stepping removes a ×2.9 multiplier | **WRONG** (×1.96 max) |
| 2e | H100 20–50× end-to-end; N=1000 table | **OPTIMISTIC but padded** (internally consistent; contingent on unmeasured dual-Newton behavior at N=1000) |
| 3 | Sparse top-k≈8 representation | **PLAUSIBLE at equilibrium; UNSUPPORTED for dynamics** (and one mechanism claim is wrong — §3) |
| 4 | Framework picks (PyTorch; reject JAX/MLX/CuPy) | **SOUND, verified current** (new gap: repo Python 3.9 vs torch ≥ 2.9 needing 3.10) |
| 5 | Dual semismooth-Newton projection mandatory + 5–15 steps | **Mandatory: SOUND. Step count flat in N: UNSUPPORTED** |
| 6 | Gates sufficient; 10–14 weeks | **OPTIMISTIC** (gate gap above N=100; Phase 3 underscoped; realistic 14–20 wk) |
| 7 | Pelle GPU assumption | **RESOLVED FAVORABLY** (verified: 40× L40s, 4× H100 — but H100 scarcity matters) |

---

## 1. The K(N) finding (most load-bearing) — verdict: WRONG as headlined, OPTIMISTIC as hedged

### 1.1 The controlled-pair claim is genuine

Verified from the two timing profiles: identical level schedule (V = 9,600 /
24,948 / 47,488 / 77,220 / 114,144), identical nnz(K), identical ε per level,
same λ=2.1 and seed 84172851 (run dir names), both 5-level seeded-init runs,
neither hitting the 30,000-iteration per-level cap (`max_iter: 30000` in
`torus_50part.yaml`; max observed 22,603). Totals reproduce: T₅₀ = 41,754 s,
T₁₀₀ = 146,240 s, K = 54,154 → 54,536, α_local = ln(3.502)/ln 2 = **1.808** ✓.

### 1.2 But total K is the wrong quantity, and its flatness is an artifact

The plan quotes N=100 per-level K (62 / 89 / 22,603 / 16,444 / 15,338) but
**omits the N=50 per-level K, which is in the same file it cites**:

| Level | V | K (N=50) | K (N=100) | ratio | per-level α_K |
|---|---|---|---|---|---|
| 0 | 9,600 | 76 | 62 | 0.82 | — |
| 1 | 24,948 | **18,348** | **89** | **0.005** | — |
| 2 | 47,488 | 13,233 | 22,603 | **×1.71** | 0.77 |
| 3 | 77,220 | 11,444 | 16,444 | **×1.44** | 0.52 |
| 4 | 114,144 | 11,053 | 15,338 | **×1.39** | 0.47 |
| Σ | | 54,154 | 54,536 | ×1.007 | |

At every matched fine level, **K grew by 39–71 % when N doubled** — a
per-level exponent of N^0.47–0.77. The total is flat only because level 1
collapsed from 18,348 iterations to 89. "K appears to saturate" is therefore
not what the data shows; the data shows **K per level grows ≈ N^0.5–0.8, with
the total conserved by a one-off coarse-level collapse**.

### 1.3 The level-1 collapse is a line-search breakdown, not fast convergence

The mechanism is visible in the profiles: at N=100, level-1 mBT = **14.6**
(vs 1.88 at N=50) and level-0 mBT = 20.3. In `pgd_optimizer.optimize()`, when
every Armijo trial is rejected the step decays below 1e-12, no step is
accepted, E is frozen, and the plateau trigger (`refine_delta_energy` — an
**absolute** threshold, 1e-8 in this config — over `refine_patience = 30`
iterations, plus gnorm/feas plateau deltas, also absolute) fires at the
patience floor. At 9,600–24,948 vertices, N=100 means 96–249 vertices/cell —
below or near the ≥ ~400/cell guideline hard-coded into the dormant-cell
warning in `relaxation.py`. The coarse levels at N=100 were **stuck, not
done**. Consequences:

- K measured through these triggers confounds landscape difficulty with
  trigger heuristics and mesh under-resolution. It is not a clean measurement
  of "iterations to converge" at either N.
- At N=1000, no level of a schedule built this way is well-resolved until
  V ≈ 400k. Either the schedule is redesigned (level-0 at ~400k vertices —
  eliminating cheap coarse levels entirely, inverting §3.3 item 3's lever) or
  every coarse level exhibits this breakdown. Either way the N=50/100 total-K
  behavior does not transfer.
- The plan cannot confirm from the cited files that the N=100 run passed
  `detect_dormant_cells()` (that result lives in `solution/metadata.yaml`,
  outside this review's read scope). Given 96 vertices/cell at level 0, this
  **must be checked and stated**: if the N=100 run has dead or weak cells, the
  K comparison is between a valid and an invalid partition trajectory.

### 1.4 Corrected local exponent

Excluding the anomalous coarse levels (like-for-like, fine levels 2–4 only):
wall 37,222 s → 145,857 s = ×3.92 → **α_fine = 1.97**; projection wall
30,192 s → 125,509 s = ×4.16 → **α = 2.06**. The plan's 1.81 is flattered by
N=50 having spent 4,454 s at level 1 that N=100 skipped by being stuck. The
defensible statement is: *local α ≈ 2.0–2.1 between N=50 and N=100, on one
mesh schedule, one seed, through trigger-mediated stopping*. Two points fit a
line exactly; they support **no** exponent claim beyond "well below doc-05's
3.1 locally, and above the plan's 1.8."

### 1.5 How much confidence does α ≈ 1.8 earn?

Little. It is one seed (doc 05's own N=30 sweep shows a 1.8× wall spread
across seeds — larger than the entire 50→100 disagreement between α=1.8 and
α=2.1), one mesh schedule, two points, an aggregate confounded by trigger
redistribution, and possibly hardware variance (doc 05 records a 3.8× spread
from host contention on identical config+seed; nothing in the timing files
identifies the hosts). What the pair *does* establish robustly: doc-05's
K ∝ N² (from the random-init N=10 point) does not describe the seeded-init
regime at these N — the plan is right to retire the "centuries" verdict.
Extrapolating per-level growth of ×1.4–1.7 per doubling over the 3.3 doublings
from N=100 to N=1000 gives K_fine ×3.0–5.9, i.e. **K(1000) ≈ 150k–350k** —
inside the plan's [55k, 550k] planning range, centered near its K ∝ N row.
**The planning range is fine; the "saturation holds" scenario and its
0.5–1.5 h headline should be dropped or demoted to an explicit best case.**

---

## 2. The arithmetic — independently recomputed

### 2.1 Reproduced exactly (SOUND)

- **Memory wall:** 1,141,440 × 1000 × 8 B = **9.13 GB** per array ✓. Code
  inspection confirms ≥ 10 live V×N arrays in the loop (`x`, `g`, `best_x`,
  `prev_x`, `curr_x` — the last two are copies the plan didn't even list —
  `A_trial`, `x_trial`, the projection's internal copy, `A_orth`, clip
  temporaries) → 73–91 GB is if anything slightly understated ✓. Solution file
  2 × 9.13 GB ≈ 18.3 GB ✓. Dense is dead at N=1000. **Holds regardless of K.**
- **Bandwidth-bound claim:** level 2, N=100: 712.8 ms/call ÷ 20.23 inner iters
  = 35.2 ms/inner-iter ✓; matrix 47,488×100×8 = 38.0 MB ✓; ~10–14 full-array
  touches per inner iteration confirmed by code walk of
  `orthogonal_projection_iterative` → **~11 GB/s effective** ✓. The kernel is
  bandwidth-bound; the GPU headroom argument is sound.

### 2.2 Wrong but self-cancelling (fix before anyone builds on them)

- **Projection calls per iteration.** The code performs exactly **one
  projection per line-search trial**, and `mean_backtracks_per_iter` counts
  *trials* (increment at the top of the trial loop, accepted trial included).
  Verified: K·mBT = 54,154×1.9301 = 104,523 ≈ measured 104,531 invocations
  (N=50); 106,891 ≈ 106,902 (N=100). The plan's (and doc 05's Eq. cost-model)
  **calls = K·(mBT+1) ≈ 161k is wrong**; actual ≈ 107k.
- **Per-call means.** 33,625 s ÷ 104,531 = **322 ms** (N=50) and 125,791 s ÷
  106,902 = **1,177 ms** (N=100), not 211/779 ms. The plan divided measured
  wall by a fabricated call count. The ratio (×3.66) and hence every *total*
  built from wall × calls survives — the ×1.5 errors cancel — but the per-call
  figures in the foundation table are wrong.
- **Consequence — the BB claim (§3.3 item 1, "the single most certain win"):**
  eliminating the line search takes calls/iter from mBT ≈ 1.96 to 1, a
  **×1.96** reduction, **not ×2.9**. Phase 4's gate ("projection-call counts
  required to drop ≥ 2.5×") is unreachable by the claimed mechanism alone; it
  implicitly requires BB to also cut K by ≥ 1.3×, which is plausible but is a
  different, unstated bet. Doc 05 §"What remains" already stated the honest
  version ("remaining headroom here is modest, ≲2×"); the plan regressed it.

### 2.3 Optimistic extrapolations

- **mPII at N=1000.** The plan extrapolates 21 → ~30 (×1.4). The measured
  50→100 trend is mPII ∝ N^0.62, which gives **mPII(1000) ≈ 88**. This only
  makes the (already dead) dense-CPU baseline deader, but it matters for
  Phase 1: a verbatim GPU port of the *alternating* projection inherits this
  growth, so Phase-1-only wall-time projections at large N are unstable — a
  point in *favor* of the plan's insistence that Phase 2 is mandatory, but one
  the plan should state with its own numbers.
- **H100 20–50× and the ~2–5 ms/call sparse estimate.** From 11 GB/s to even
  30–50 % of H100 NVL bandwidth is >50×, so 20–50× end-to-end is defensible
  *for the fine levels*; gather/scatter-heavy ELL access and per-row top-k
  sorts will not stream at peak, and the estimate assumes the dual-Newton
  outer count (~10) holds at N=1000 with 1000 coupled area duals —
  **unmeasured**. The wall-time table carries 3–10× internal padding (55k ×
  10 ms = 0.15 h quoted as 0.5–1.5 h), which absorbs realistic kernel
  inefficiency. Verdict: table rows are internally consistent and padded, but
  the honest center after §1 is the **5–15 h row**, and on L40s (see §7)
  multiply by ~3–4.
- **Untracked at N=1000: the per-iteration trace writer.** `optimize()` writes
  `x` into `*_internal_data.hdf5` every `h5_save_stride` iterations (~30–46
  saves/level in these runs). At N=1000 dense that is ~9 GB per save; even
  sparse, the trace schema needs the same treatment as `x_opt` — the plan's
  §3.5 covers the solution file only.

---

## 3. The sparse representation (the plan's crux) — verdict: right risks flagged, one mechanism claim wrong, dynamics unsupported

- **Is k ≈ 8 justified?** At *equilibrium*, yes: the Γ-limit is near-one-hot,
  interfaces are O(ε) wide, and generic points see 2–3 cells (triple points
  3–4); k=8 is ~2× headroom and the occupancy histogram is the right
  diagnostic. But k=8 is **arbitrary for transients**: immediately after
  level interpolation and during coarse-level churn (exactly where the N=100
  run shows pathology), support widths are not equilibrium widths. The
  mandatory histogram must be watched *per iteration early in each level*, not
  only as a per-level summary.
- **The "same physical mechanism" claim is wrong.** §3.1 asserts regions
  grow/shrink only by interface motion "the same physical mechanism as the
  dense algorithm... rather than by teleporting mass, which the energy forbids
  anyway." Not so: the dense algorithm's area step (`A *= d/(vᵀA)`, projection
  step 12) is a **global per-column rescale** — it adds mass to a starving
  cell at *every* vertex where that cell has any density, however remote,
  every inner iteration. The dense algorithm can and does maintain/resurrect
  cells non-locally; the energy penalizes but does not forbid it, and the
  projection does it unconditionally. The sparse + restricted-dual design
  removes this channel entirely, making **cell death absorbing**: once a
  cell's support shrinks below viability, no mechanism can re-nucleate it.
  This is a strictly stronger version of the plan's "area-budget stranding"
  risk, and it directly interacts with §1.3's finding that coarse levels at
  high N are under-resolved — the regime where cells get pinched is exactly
  the regime the sparse representation handles worst.
- **Two unaddressed failure modes:**
  1. **Capacity/active-set degeneracy in the restricted dual.** If a starving
     cell's every candidate entry is at a simplex face (or its support is at
     capacity k), the generalized Jacobian row for that cell vanishes and
     Newton cannot restore its area — the plan's "exact feasibility by
     construction" claim fails precisely when it matters. A support-expansion
     fallback on Newton failure must be designed in, not discovered in
     Phase 3 debugging.
  2. **The zero-row fallback.** The dense projection maps empty rows to the
     uniform density 1/N — a dense row, unrepresentable at k=8. The sparse
     design must define this case (nearest-support? winner-only?); it is
     currently undefined.
- **Does sparsity survive the projection?** Under the *current* scheme: no —
  confirmed in code; `A_orth = η⊗1 + v⊗λ` adds a dense rank-one correction to
  every row, and the column rescale is global. The plan is right that the
  alternating scheme is structurally incompatible with sparsity (this is its
  strongest single argument). Under the restricted dual-Newton: sparsity
  survives **by construction**, at the cost that the computed projection is no
  longer the true projection onto the feasible set (support-restricted). The
  plan acknowledges trajectory divergence; it should also acknowledge that
  the restriction makes the *constraint geometry itself* N-local, which is a
  model change, not an implementation detail.

---

## 4. Framework claims — verified current (July 2026)

- **jax-metal:** confirmed experimental, effectively unmaintained (community
  threads note stagnation; install failures on recent macOS; community forks
  `jax-mps`/`applejax` are alpha), and **no float64 on Metal**. Rejection
  **SOUND** — if anything stronger than the plan states.
- **MLX:** CUDA backend shipping and improving fast (v0.31.x, early 2026,
  initial quantized-GEMM CUDA work) but young and LLM-inference-focused; MLX
  also lacks fp64 on GPU. Rejection **SOUND** today; correctly framed as a
  bet-timing issue.
- **CuPy:** CUDA/ROCm only, no Metal — stable fact. Rejection **SOUND**.
- **PyTorch MPS:** confirmed **no float64** (Metal framework limitation, not a
  PyTorch gap — will not be fixed). The fp64-gates-on-CPU/CUDA-only
  workaround is therefore load-bearing and **holds**: it is the only possible
  arrangement, and it is workable because the gates run at N ≤ 100 where CPU
  fp64 is affordable. The residual risk the plan correctly implies but should
  state plainly: **MPS runs gate nothing** — dev-box GPU work is limited to
  logic/shape testing, so most numerical debugging happens on CPU (slow) or
  on Pelle (queued).
- **New gap the plan misses:** PyTorch **2.9 (Oct 2025) dropped Python 3.9**;
  minimum is now 3.10. This repo targets Python 3.9 (`pyenv` env, Black
  `py39`). "PyTorch ≥ 2.4" is installable on 3.9 only up to torch 2.8 —
  freezing the project out of current MPS/`torch.compile` improvements the
  plan's dev story leans on. A Python ≥ 3.10 environment migration is an
  unstated Phase-1 prerequisite and belongs in the plan.
- **`torch.compile` on MPS** remains partial/experimental; the plan's fusion
  claims are solid for CUDA (inductor mature, CSR spmv supported) but should
  not be assumed for the MPS dev loop. Verify at Phase-1 start.

---

## 5. The projection redesign — mandatory: yes; convergence story: incomplete

- **Genuinely mandatory for sparsity?** **Yes** (verified in code, §3 above).
  No adaptation of the alternating scheme preserves row sparsity: the
  rank-one correction is dense *in the algorithm's fixed-point structure*, not
  incidentally; restricting it to supports changes the fixed point, i.e., is
  itself a redesign. The dual decomposition (row simplex projections + N area
  duals) is also the standard, well-trodden formulation for this constraint
  intersection, and doc-05 §7.2 / audit #3 independently reached it.
- **Convergence vs the current scheme:** the dual is a concave maximization
  with piecewise-smooth gradient; semismooth Newton with damping is
  well-founded and locally superlinear, and *any* convergent dual method
  yields exact row feasibility by construction — a real improvement over the
  current loop's 10·tol stall-validation failure mode. But the plan's
  "expected 5–15 outer steps, each one row-projection pass" is **asserted,
  not measured**, and assumed flat in N. The current scheme's own coupling
  difficulty demonstrably grows (mPII ∝ N^0.62 measured); assuming the dual
  Newton count does not grow between N=100 and N=1000 is unsupported. Missing
  from the design: globalization (line search/bisection safeguard), the
  degenerate-Jacobian case (§3), and a warm-start policy for y across PGD
  iterations (the audit-#13 rejection concerned warm-starting the *old*
  scheme's internal state; the dual-Newton y warm-start is a fresh question
  that should cut the outer count to ~1–3 near convergence — a cheap,
  legitimate win the plan leaves on the table).

---

## 6. Validation gates and roadmap — pattern right, coverage gap above N=100, Phase 3 underscoped

- **Gates.** Reusing the Mode-1/Mode-2 stage-A/B/C machinery is the right
  call, and stage-C for the verbatim port is appropriately strict. Two gaps:
  1. **Nothing is A/B-validated above N=100.** Phase 5's per-rung checks
     (dormant cells, area feasibility, occupancy) are *self-consistency*
     checks, not equivalence checks — a support-dynamics bug that manifests
     only at N ≥ 250 (capacity overflow, Newton degeneracy, absorbing cell
     death) passes every gate in the plan. Fix is cheap: **dense-vs-sparse
     A/B at N=250 on GPU is feasible** (dense state at V≈285k, N=250 is
     0.57 GB/array, ~6 GB working set — fits both L40s and H100) and should
     be a Phase 5 entry gate.
  2. Stage-A/B partition-agreement gates compare *trajectories through a
     nonconvex landscape*; divergence to a different-but-valid local minimum
     would fail the 99.5 % agreement gate spuriously. The plan should
     pre-commit to how it distinguishes "different valid minimum" from
     "broken port" (e.g., perimeter within tolerance + clean dormant check +
     energy within tolerance ⇒ acceptable), or Phase 3/4 gates will generate
     unresolvable arguments.
- **Timeline.** Phases 1, 2, 4 estimates are credible. **Phase 3 (3–5 weeks)
  is underscoped**: it contains all the research risk the plan itself
  identifies (support dynamics, truncation, two HDF5 schemas, a
  `ContourAnalyzer` loader, plus the §3 failure modes above), and its Mode-2
  gates each cost an N=100-scale A/B pair. Phase 5 queue time on **4
  cluster-wide H100s** is not under the project's control. Realistic total:
  **14–20 weeks**, not 10–14.

---

## 7. The Pelle GPU assumption — resolved favorably, with a twist

The plan treats GPU availability as unverifiable before Phase 1. It is
publicly documented: **Pelle has GPU nodes** — 4 nodes × 10 NVIDIA **L40s**
(48 GB) , 2 nodes × 2 **H100** (94 GB, NVL-class), plus ~34 legacy T4 nodes
(UPPMAX Pelle hardware docs). Consequences:

- The hard-dependency scenario ("no GPUs at all") is off the table; the
  NAISS/Alvis contingency can be deleted.
- But there are only **4 H100s cluster-wide**. The realistic steady-state
  workhorse is the L40s pool — the plan's own L40s contingency (bandwidth
  864 GB/s, ~1/4 of H100-NVL; fp64 FLOPs poor but kernels bandwidth-bound)
  becomes the *primary* plan, not a fallback. All H100 wall-time rows stretch
  ~3–4×: the defensible N=1000 estimate is **~1–3 days (L40s, K ≈ 150k–350k)**
  with H100 opportunistic. Still feasible; state it this way.
- Collapse analysis (per the review brief): if GPUs were somehow unusable,
  the plan degrades gracefully and better than it advertises — sparse (N/k ≈
  125 per-pass win) + dual-Newton (~5×) + BB (~1.9×) are all CPU-valid, and
  the plan's own CPU-torch estimate (N=1000 in weeks) checks out. The crux of
  the plan (representation + projection) is hardware-independent; only the
  timeline multiplier is at stake. This is a strength the plan undersells.

---

## 8. Contingency tagging

**Contingent on the K(N)/α~1.8 finding (weakened by this review):**
- The "K saturates → 0.5–1.5 h at N=1000" headline row — **discard** (§1).
- "α_local ≈ 1.81" as a quotable exponent — replace with "α ≈ 2.0–2.1
  like-for-like; total-α 1.8 confounded" (§1.4).
- The implicit claim that Wall 2 uncertainty is narrow. It remains the
  dominant uncertainty, now with a data-supported center (K(1000) ≈
  150k–350k) rather than a saturation hope.

**Hold regardless of K(N):**
- Wall 3: dense V×N is infeasible at N=1000 (9.13 GB/array, 73–91 GB working
  set) — verified arithmetic, independent of iteration counts.
- Wall 1: dense-CPU N=1000 is infeasible under *every* K scenario (135–204
  days even at flat K; years under measured mPII growth).
- The alternating projection is structurally incompatible with sparsity —
  code-verified; hence Phase 2 before Phase 3 is a hard ordering.
- Framework selection (PyTorch), the fp64-on-CPU/CUDA gate policy, and the
  in-repo/two-backend strategy.
- The feasibility *verdict* itself: even the pessimistic K ∝ N² tail lands at
  days on GPU. The plan's conclusion is robust to losing its favorite data
  point — the plan should say so explicitly, because right now the optimistic
  K story reads as load-bearing when it is not.

---

## 9. Cheapest experiments to confirm/overturn α ≈ 1.8 (ranked)

The plan's Phase-0 proposal (one full profiled N=75 run, ~25 h+) is **not the
sharpest test**: it adds a third point to the same confounded aggregate — if
N=75's coarse levels land in between the N=50 grind and the N=100 collapse
(likely), total K will be noise.

1. **Re-analyze the existing traces (zero compute, ~half a day).** The
   per-iteration summary files
   (`traces/pgd_part{50,100}_*_level{2,3,4}_summary.out`) for both runs
   already contain per-iteration energy/gnorm/feas. Compute, per fine level,
   iterations-to-reach-matched *relative* energy decrease (e.g. fraction of
   that level's total decrease) for N=50 vs N=100. This directly separates
   "landscape genuinely needs ×1.4–1.7 more iterations" from "trigger fired
   at different points on similar curves" — the exact confound in §1. Also
   read both runs' `metadata.yaml` dormant-cell blocks (§1.3). **Do this
   first; it may settle the question outright.**
2. **Fixed-mesh single-level K(N) micro-sweep (~2–4 days unattended CPU).**
   Run `refinement_levels: 1` at the level-2 mesh (V=47,488, well-resolved
   down to ~120 cells) with seeded init, N ∈ {25, 50, 100} (+200 if the mesh
   supports it), same seed, and a **scale-invariant stopping rule** (relative
   energy plateau or fixed gnorm target — not the absolute
   `refine_delta_energy`). Three-four points, one mesh, no level
   redistribution, no trigger confound: this is the clean K(N) curve the
   whole plan needs, at a fraction of one production run's cost (level-2
   N=100 cost ≈ 10 h; N=25/50 are 2–5 h).
3. **Only then, optionally, the N=75 full run** — as a *validation* of the
   prediction from (2), not as the primary measurement. Also keep the plan's
   level-schedule experiment at N=50 (config-only, cheap, and §1.3 suggests
   schedule design is a bigger lever at high N than the plan realizes).

**Recommendation:** replace Phase 0's "one profiled N=75 run" with (1) + (2);
total cost roughly *half* of the N=75 run, and the resulting K(N) estimate is
strictly more informative.

---

## 10. Minor corrections (fix in the plan)

- §1 foundation table: per-call means 211 ms → **322 ms**, 779 ms →
  **1,177 ms**; "K·(mBT+1)" → **K·mBT** (also fix doc 05 Eq. cost-model and
  the `mean_backtracks_per_iter` naming, which counts *trials*).
- §3.3(1): "~2.9× fewer projection calls" → **~1.9×**; adjust Phase 4's
  ≥2.5× gate or restate it as (calls/iter × K) combined.
- §2/§4 H100 bandwidth: 2.5 vs 3.35 TB/s inconsistency; Pelle's H100s are
  94 GB NVL-class per UPPMAX docs.
- §4: replace the "believed to provide" paragraph with the verified Pelle GPU
  inventory (40× L40s, 4× H100, T4 legacy) and re-center the wall-time table
  on L40s.
- §3.5: extend the sparse HDF5 schema to the per-iteration trace writer
  (`*_internal_data.hdf5`), or disable `x` saving at N=1000.
- §2 framework table / §8: add the Python ≥ 3.10 migration prerequisite
  (torch 2.9+ dropped 3.9).
- §1: state whether the N=100 run passed `detect_dormant_cells()`.
