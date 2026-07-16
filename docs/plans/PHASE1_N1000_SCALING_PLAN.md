# Phase 1 Scaling to N = 1000+ Regions: Sparse Representation, GPU Backend, and Projection Redesign

**Status:** Not Started
**Scope:** Phase 1 (PGD relaxation) only. Phase 2 is touched only at the
Phase 1 → Phase 2 bridge (HDF5 schema / `ContourAnalyzer` ingestion).
**Deliverable of this document:** a decision-oriented design and phased
roadmap; no production code accompanies it.

---

## 1. Problem statement — the three walls

`docs/math/05-phase1-nregion-scaling/main.tex` concluded that Phase 1 at
N = 1000 costs ~14 years on a clean host (α ≈ 3.1 anchored on the measured
post-optimization N=50 point). Since that document, a controlled N=100 run
has completed and sharpens the picture. Three independent walls stand between
today's N ≈ 100 and the N = 1000 target. GPU hardware attacks only the first.

> **Scope caveat — this plan addresses *performance*, not *partition validity*.**
> The three walls below are compute/iteration/memory. There is a separate,
> orthogonal wall: at high N, Phase 1's winner-take-all *discrete* cell areas can
> drift far from equal even though the *continuous* equal-area constraint holds,
> producing diffuse "runt" cells that make the Phase 2 equal-area constraint
> infeasible. The old N=100 timing anchor (`run_20260629_141012`) exhibited exactly
> this (worst cell 22% off target). **Update (2026-07-09): resolved at N ≤ 100.**
> The runt was largely an artifact of a mis-discretized double well (~25× too weak;
> `docs/reference/phase1_energy_discretization_bug.md`); on the *corrected* energy
> with a moderate `lambda_penalty=5.1`, the N=100 worst cell is 0.8% and Phase 2
> refines cleanly (`run_20260709_081548`, perimeter −13.6%). So the timing anchors
> here (measured on the buggy energy) remain valid as *performance* references, but
> the partition-validity wall they flag is no longer open at N=100. It may recur at
> N=1000; its detection stays a required gate (§6). See
> `docs/reference/winner_take_all_partition_gap.md` §8 and
> `docs/experiments/02-corrected-energy-highn-validation/`.

### Empirical foundation (all measured, this repo)

| Quantity | N=50 (doc 05, `run_20260627_234511`) | N=100 (`run_20260629_141012`) |
|---|---|---|
| Mesh (identical schedule & seed) | V = 114,144, seed 84172851, λ=2.1, 5 levels, seeded init | same |
| Total wall | 41,754 s = 11.6 h | 146,240 s = 40.6 h |
| Major iterations K | 54,154 | 54,536 |
| Mean backtracks mBT | 1.93 | 1.96 |
| Mean projection inner iters mPII | 13.76 | 21.15 |
| projection / energy / gradient / backtrack(net) % | 80.5 / 8.2 / 7.4 / 2.5 | 86.0 / 6.1 / 5.5 / 1.7 |
| h5_save | ~0 | 0.01 % |
| Mean wall per projection call | 33,612 s ÷ (54,154×2.93) ≈ 211 ms | 125,766 s ÷ (54,536×2.96) ≈ 779 ms |

Two consequences of this controlled pair (same mesh, same seed, seeded init;
neither run hit the 30,000-iteration per-level cap — N=100 per-level K was
62 / 89 / 22,603 / 16,444 / 15,338):

1. **K did not grow from N=50 to N=100** (×1.007). The doc-05 K ∝ N² claim
   came from the N=10 (random init) → N=30 (seeded) pair. With seeded init
   K appears to saturate at ~55k for this mesh schedule, at least locally.
2. The local exponent between the two post-optimization points is
   T₁₀₀/T₅₀ = 3.50 → **α_local = ln 3.50 / ln 2 ≈ 1.81**, decomposing as
   ×2 (per-call O(N)) × 1.54 (mPII 13.76 → 21.15) × ~1.14 (residual), with
   K contributing ×1.0. The doc-05 α = 3.1 fit remains the pessimistic
   envelope; the truth for seeded init at large N is unknown between the two.

### Wall 1 — projection compute (86 % of wall, O(mPII·V·N) per call)

Extrapolate the dense NumPy implementation to N = 1000 at scientific
resolution (V must grow with N — see Wall 3 — take V = 1.14 M, i.e. today's
1,141 vertices/cell):

- per-call cost scales by (N: ×10) × (V: ×10) × (mPII: 21 → ~30, ×1.4) ≈
  **×140** → 779 ms → ~109 s per projection call;
- calls = K·(mBT+1) ≈ 54.5k × 2.96 ≈ 161k even if K stays flat →
  1.76×10⁷ s ≈ **204 days of projection alone**, ≈ 237 days total;
- if K ∝ N: ×10 → **6.5 years**; if K ∝ N²: ×100 → **65 years**.

Dense CPU is infeasible at N = 1000 under *every* K scenario. Note the
per-call kernel is memory-bandwidth-bound, not FLOP-bound: at level 2
(V = 47,488, N = 100) one inner iteration takes ~35 ms while touching a
38 MB matrix ~10 times ≈ 380 MB of traffic → **~11 GB/s effective**, i.e.
NumPy with temporaries exploits ~10 % of a laptop's bandwidth and ~0.5 % of
an H100's. That headroom is what the GPU buys.

### Wall 2 — iteration count K (the factor GPU cannot fix)

Every projection call is a serial dependency; K·(mBT+1) is a wall-clock
multiplier no amount of per-call parallelism removes. Evidence:

| N | init | K (all levels) |
|---|---|---|
| 10 | random | ~3.5k (estimated, doc 05) |
| 30 | seeded | 27k–50k across seeds |
| 50 | seeded | 54.2k (measured) |
| 100 | seeded | 54.5k (measured, same mesh/seed as N=50) |

Planning range for N = 1000: **K ∈ [55k (saturation holds), 550k (K ∝ N)]**,
with K ∝ N² (5.5 M) as the pessimistic tail. Discriminating this early is
Phase 0 of the roadmap; every feasibility number below is quoted against the
range. Independent of K's growth, two algorithmic levers cut the multiplier:
removing the (mBT+1) ≈ 3 factor (one projection per iteration, §5.3) and
accelerating the first-order iteration itself (§5.3).

### Wall 3 — memory: the dense V × N field

Current: V = 114,144, N = 100 → one float64 array is 114,144×100×8 B =
**91.3 MB**; the solution HDF5 is 188 MB (x_opt + x0 + mesh ≈ 2×91.3 + 5).

At N = 1000 the mesh must refine with N. Total interface length grows ∝ √N
and ε ∝ h must resolve cell diameters ∝ N^(-1/2); holding today's finest-level
1,141 vertices/cell gives **V ≈ 1.14 M**. Then:

- one dense array = 1.14M × 1000 × 8 B = **9.13 GB**;
- the PGD loop keeps ~8–10 such arrays live (`x`/A, g, g_post, best_x,
  A_trial, x_trial, the projection's internal copy, the rank-one correction
  `A_orth`, clip temporaries) → **73–91 GB working set**, before NumPy's
  expression temporaries;
- the solution file (x_opt + x0) would be **18.3 GB**.

That exceeds the 24 GB M4 Pro dev box outright and is borderline-infeasible
even on an H100-94GB. Freezing V at 114k (114 vertices/cell) would fit
(~9 GB) but is 10× under-resolved versus current practice — not a
scientifically meaningful N = 1000 run. **Dense V × N is dead at N = 1000;
the sparse representation of §5.1 is the crux of this plan, not the GPU.**

The physical solution is (near-)one-hot: a vertex has non-negligible density
only for its own cell and cells whose interface passes within O(ε). With
k ≈ 8 stored entries per vertex, the state is 1.14M × 8 × 12 B (8 B value +
4 B column) ≈ **110 MB** — a ~100× reduction that also cuts per-call
projection work from O(V·N) to O(V·k).

---

## 2. Recommended target architecture

**Sparse row-support density field + PyTorch compute backend (CUDA prod /
MPS+CPU dev) + dual-Newton exact projection + spectral (BB) stepping**, built
in this repo behind a `relaxation.backend` config switch with the current
NumPy path kept as the frozen numerical reference.

| Component | Decision | Rejected alternatives (why) |
|---|---|---|
| Data representation | Fixed-capacity sparse row support (top-k per vertex, k≈8, ELL-style layout) with 1-ring support exchange | Dense V×N (Wall 3: 73–91 GB); dynamic CSR (rebuild cost + poor GPU coalescing) |
| Framework | **PyTorch** ≥ 2.4 | JAX (jax-metal experimental, lags releases, no fp64 on Metal → dev loop broken); MLX (excellent on Apple, CUDA backend too young to bet prod on); CuPy (CUDA-only, fails the Metal dev requirement) |
| Projection | Dual semismooth-Newton: row-wise closed-form simplex projection + Newton on the N area duals (doc 05 §7.2 already identifies this as the highest-merit fix) | Keep alternating scheme (mPII grows with N: 13.8→21.2 from N=50→100, and its rank-one correction is dense — incompatible with sparsity); dual warm-start of the current loop (ruled out on correctness, audit #13) |
| Iteration count | Barzilai–Borwein spectral step + nonmonotone (Grippo) safeguard; optional Nesterov acceleration with function-value restart | Second-order/Newton on the energy (per-iteration cost and memory explode; the energy is nonconvex and the Hessian is V·N-dimensional); Anderson acceleration (fragile through the nonsmooth projection) |
| Parallelism | Single-GPU data parallelism over the V×k state; multi-GPU deferred | CPU region-parallel (doc 05 §7.3): P cores → ÷P but P≈8–16 gives 40.6 h → ~4 h at N=100 and *years → months* at N=1000; strictly dominated by GPU on every axis. Spatial domain decomposition: high complexity, only needed if a single GPU can't hold the state — it can (110 MB) |
| Precision | fp64 on CUDA and CPU (all validation gates); fp32 on MPS for dev smoke runs only | fp32 everywhere (projection tol 1e-8/1e-9 unreachable at fp32 eps ≈ 1.2e-7) |

### Why PyTorch specifically

- **Portability requirement is Metal + CUDA.** PyTorch is the only mature
  framework with first-class CUDA *and* a production-quality Metal (MPS)
  backend. One codebase, `device`/`dtype` parametrized.
- **Known limitation, planned around:** MPS has **no float64**. Dev-box GPU
  runs are fp32 behavioral smoke tests (relaxed projection tol ~3e-6);
  all numerical-equivalence gates run in fp64 on CPU (torch) or CUDA.
  This is acceptable because the kernels are bandwidth-bound and small
  configs (N=10–50, coarse levels) run fine on CPU fp64.
- Sparse building blocks needed here are gathers/scatters/segment-reductions
  and batched tiny sorts (top-k, simplex projection) — all first-class,
  fusable via `torch.compile`; sparse CSR spmv (`K @ u`) is supported on CUDA.
- Ecosystem maturity on the cluster: prebuilt CUDA wheels, no source builds
  on Pelle.

### What the GPU buys — and what it does not

Projection/energy/gradient are elementwise + row/column reductions + spmv on
a ~100 MB–1 GB state: bandwidth-bound. Measured effective bandwidth today is
~11 GB/s (NumPy, temporaries). Achievable: M4 Pro ~200 GB/s (fp32), H100 SXM
~2.5 TB/s. With kernel fusion (fewer passes), a **20–50× end-to-end
per-iteration speedup on H100 fp64 is realistic**; the dense-state N=100 run
at 40.6 h becomes ~1–2 h with no algorithm change. What the GPU does **not**
change: K and the serial chain of projection calls (Wall 2), and it cannot
make the dense N=1000 state fit (Wall 3). Hence sparse + projection redesign
+ stepping are co-equal parts of the architecture, not optional extras.

---

## 3. Component designs

### 3.1 Sparse localized density field (first-class, the crux)

**Layout.** ELL-style fixed capacity: `values: float64[V, k]`,
`cols: int32[V, k]`, k ≈ 8 (configurable), padded with sentinel −1. Fixed
shape → coalesced GPU access, no reallocation, deterministic memory.
Budget at V = 1.14 M, k = 8: 110 MB per field copy; the whole optimizer
working set (~10 copies) ≈ 1.1 GB — fits the 24 GB dev box and any prod GPU.

**Invariant.** Row i's support = the cells with meaningful density at vertex
i: its winning cell plus cells whose interface is within O(ε). Interior
vertices are one-hot (1 stored entry); interface vertices carry 2–3; triple
points 3–4. k = 8 is ~2× headroom; a per-level histogram of support occupancy
is a mandatory diagnostic (if the 95th percentile approaches k, raise k).

**Initialization.** The seeded Voronoi init (`create_seeded_initial_condition`)
is *exactly* one-hot before projection — sparse from iteration 0. Random init
is dense (~1/N everywhere) and is **incompatible** with this representation;
`init_method: seeded` becomes mandatory for the sparse backend (it is already
mandatory in practice for N ≥ 30 due to dormant cells).

**Support evolution.** The gradient of the Modica–Mortola energy at a sparse
iterate is supported on support ∪ 1-ring (only `K @ u` couples neighbors).
Each outer iteration: (a) candidate support = current support ∪ cells present
in the vertex's 1-ring neighbors' supports (bounded, local, gather-based);
(b) gradient step + projection over the candidate set; (c) truncate entries
below a floor (~1e-10, the existing regularization scale) back to capacity k,
tracking the truncated mass as a diagnostic. Regions therefore grow/shrink by
moving interfaces — the same physical mechanism as the dense algorithm —
rather than by teleporting mass, which the energy forbids anyway (a remote
sliver of cell j at vertex i pays full interface cost).

**Impact on the three kernels.**
- *Energy*: per-cell terms become segment-reductions over stored entries;
  `u_i^T K u_i` uses the sparse K restricted to rows in cell i's support —
  O(nnz(K)·k/N) per cell, O(nnz(K)·k) total instead of O(nnz(K)·N).
- *Gradient*: same structure, O(V·k) elementwise + spmv on candidate support.
- *Projection*: per-row work drops from O(N) to O(k log k); the column (area)
  reductions become segment-sums over V·k entries. Total O(V·k) per pass —
  **at N = 1000 the per-pass work equals today's N = 100 dense per-pass work**
  (1.14M×8 = 9.1M entries vs 114k×100 = 11.4M).

**Risk (honest):** this changes the algorithm's reachable set — a cell can
only gain density where it has 1-ring adjacency. Pathologies (a cell pinched
to zero width, topology events during relaxation) need the truncated-mass and
support-occupancy diagnostics plus the dense-equivalence gate at N = 50/100
(§6, Phase 3) before it is trusted at N = 1000.

### 3.2 Projection: dual semismooth-Newton (exact, sparse-compatible)

Problem: project A onto {X ≥ 0, X·1 = 1 (rows), vᵀX = d (areas)}. The
current alternating scheme needs mPII ≈ 21 sweeps at N=100 (growing with N),
each sweep ~10 full passes over the state, and its correction
`A_orth = η⊗1 + v⊗λ` is a **dense rank-one update** — it alone would
destroy any sparse layout. Replacement (doc 05 §7.2, made concrete):

- Dualize only the N area constraints with y ∈ ℝᴺ. For fixed y, the problem
  decouples into V independent simplex projections of rows
  `a_i − v_i·y` — closed form via the sort/threshold algorithm, O(k log k)
  per row on the candidate support (§3.1 restricts the candidate set; the
  dense backend uses all N).
- Outer loop: semismooth Newton on g(y) = vᵀX(y) − d. The generalized
  Jacobian is diagonally dominant (entry (j,j) sums v_i² over rows where cell
  j is in the active simplex face; off-diagonals only where two cells share
  interface rows) → diagonal-preconditioned CG or even a damped diagonal
  Newton; expected 5–15 outer steps, each one row-projection pass.
- Termination: exact primal feasibility by construction of the simplex step;
  area residual driven below tol by Newton — removes the 10·tol
  stall-validation failure mode of the current loop entirely.

Expected win independent of GPU: mPII·(~10 passes) ≈ 210 passes → ~10
Newton steps × ~3 passes ≈ 30 passes: **~5–7× on the 86 % bucket**, plus it
is the only projection formulation compatible with §3.1.

**Equivalence caveat:** this computes the *true* orthogonal projection; the
current iterative scheme returns an approximation. Trajectories will differ
→ validated with the stage-A/B gate pattern (partition agreement, perimeter,
areas), not bit-identity (see §6).

### 3.3 Iteration-count reduction (attacks Wall 2)

1. **Spectral / Barzilai–Borwein step with nonmonotone safeguard** (doc 05
   Tier 2). Kills the residual (mBT+1) ≈ 2.96 multiplier — exactly one
   projection per major iteration → **~2.9× fewer projection calls**, the
   single most certain win in this plan. Needs the Grippo-type nonmonotone
   acceptance rule and the existing [1e-8, 1−1e-8] clamp.
2. **Nesterov/FISTA acceleration with function-value restart.** For the
   convex pieces, K drops from O(1/τ) to O(1/√τ); on this nonconvex energy
   the honest expectation is a **1.5–3× reduction in K**, seed-dependent.
   Restart-on-increase keeps it safe; it composes with BB (use as alternate
   mode, pick empirically at Phase 4).
3. **Level-schedule rebalancing.** Measured N=100 per-level wall: 117 s /
   267 s / 35,307 s / 45,521 s / 65,028 s — levels 0–1 exit after 62/89
   iterations (refinement triggers fire almost immediately) while level 2
   grinds 22,603 iterations at V = 47k. Inserting intermediate coarse levels
   and loosening trigger patience so more of K is spent where iterations are
   cheap is a config-only lever worth **~1.5–2×**, measurable in Phase 0.

Combined honest projection for §3.3: **3–6×**, multiplicative with the GPU
and with §3.1/§3.2. None of it fixes a genuinely K ∝ N² landscape — only
data at N = 250/500 will tell (Phase 5 gates).

### 3.4 Region parallelism / domain decomposition (doc 05 §7.3, revisited)

The doc-05 recommendation predates the GPU decision. On a GPU, "parallelize
across regions" is subsumed: the V×k (or V×N dense) state is already
processed data-parallel; region columns are not a useful partitioning axis.
Multi-GPU column sharding would need two all-reduces per inner iteration (row
sums: V-vector; area sums: N-vector) — cheap, but pointless while the whole
sparse state is 110 MB on one device. Spatial domain decomposition composes
naturally with the multi-level scheme (coarse global solve → fine local
solves with interface exchange) but is a research project of its own.
**Decision: defer both; revisit only if N ≥ 10⁴ or V ≥ 10⁷ becomes a target.**
CPU region-parallelism (multiprocessing over columns) is rejected: bounded by
core count (~8–16×) at much higher complexity than the GPU path it duplicates.

### 3.5 Backend integration in this codebase

- New module `src/optimization/backends/` — `numpy_backend.py` (thin wrapper
  over the existing code paths, stays the reference), `torch_backend.py`
  (dense first, then sparse). Selected by a new `relaxation.backend:
  numpy | torch` and `relaxation.representation: dense | sparse` +
  `relaxation.device: auto | cpu | cuda | mps` config fields
  (`RelaxationConfig.from_yaml_dict` already ignores unknown keys in old
  configs; defaults preserve current behavior exactly).
- `ProjectedGradientOptimizer.optimize()` keeps the outer loop, trigger
  logic, tracing, and profiling contract; the backend supplies
  energy/gradient/projection/step as device-resident operations with a
  single host sync per iteration (for the summary line and trigger check).
  `RelaxationProfilingState` gains `device` timing via `torch.cuda.Event`
  wrappers where applicable — the `if profile is not None` zero-overhead
  contract is unchanged.
- HDF5: dense backend keeps the current schema. Sparse backend writes
  `x_opt_sparse/{values,cols}` + winner-take-all `labels: int32[V]` and,
  for N ≤ 100 compatibility, can optionally materialize dense `x_opt`.
  At N = 1000 dense x_opt (9.1 GB) is *not* written; `ContourAnalyzer`
  gains a loader for the sparse schema (indicator functions are argmax —
  directly available from `labels`; contour extraction needs per-edge
  density pairs, available from the sparse rows). This is the one Phase 2
  contact point.

---

## 4. Hardware and cluster reality check

**Dev — Apple M4 Pro, 24 GB unified, Metal.** MPS: fp32 only, ~273 GB/s.
Fine for: kernel development, sparse-layout debugging, fp32 smoke runs at
N ≤ 100 on coarse meshes. Not valid for: any numerical gate (fp64 CPU torch
covers that, slower but correct). Dense N=1000 does not fit (Wall 3); sparse
N=1000 state (~1 GB) fits comfortably — full-scale *logic* tests can run on
the laptop at fp32.

**Prod — UPPMAX Pelle.** `cluster/pelle_config.sh` and the three submit
scripts contain **no GPU directives** (no `--gres`/`--gpus`, no partition
selection, no CUDA module) — the pipeline is CPU-only on Pelle today.
**Assumption to verify before Phase 1 begins** (5 minutes on the login node:
`sinfo -o "%P %G %D"`, `module spider CUDA`): Pelle, as UPPMAX's 2025
Rackham replacement, is believed to provide NVIDIA GPU nodes (L40s/H100
class). Contingencies:
- H100 nodes available → plan as written (fp64 ~34 TFLOPS, 3.35 TB/s).
- Only L40s → fp64 FLOPs are poor (1:64) but the kernels are
  bandwidth-bound (864 GB/s): expect ~1/3 of H100 throughput, still ≥10×
  over CPU; alternatively mixed precision with fp64 reductions.
- No GPUs at all → apply for a NAISS GPU allocation (e.g. Alvis) or fall
  back to CPU-torch: sparse + dual-Newton + BB still deliver ~10–30×
  algorithmically (N/k × mPII × mBT wins), putting N = 1000 at weeks, not
  years — degraded but not dead.

SLURM changes when verified: `--gpus 1` + GPU partition + CUDA module in
`pelle_config.sh` / `submit_relaxation.sh`, plus a torch-CUDA wheel in the
venv recipe. `DEFAULT_MEM=16G` must rise to ~32 GB for N=1000 host-side
(mesh + HDF5 staging).

---

## 5. Feasibility arithmetic at N = 1000 (target architecture)

Per projection call (sparse, V = 1.14 M, k = 8, GPU): ~9.1 M stored entries,
~30 fused passes (dual-Newton, §3.2) over ~110 MB ≈ 3.3 GB of traffic →
**~2–5 ms/call on H100 fp64** including Newton solves and launch overhead
(vs 109 s/call dense CPU — the ×140 of Wall 1 is cancelled by N/k = 125 and
the bandwidth gap absorbs the rest). Energy + gradient per iteration: same
order (spmv-dominated, nnz(K) ≈ 7V ≈ 8 M). With BB stepping (1 projection
per iteration) and ~10 ms/iteration all-in:

| K scenario | Iterations | Projected N=1000 wall (H100) |
|---|---|---|
| K saturates (measured 50→100) | ~55k | **~0.5–1.5 h** |
| K ∝ N | ~550k | **~5–15 h** |
| K ∝ N² (pessimistic tail) | ~5.5M | **~2–6 days** |

Verdict: **N = 1000 becomes feasible — hours to days — in every K scenario**,
with the uncertainty dominated entirely by Wall 2, not by hardware.
Intermediate checkpoint: the same architecture puts N = 100 (40.6 h today)
at **~5–15 min**, which is the Phase 3/4 acceptance measurement.

---

## 6. Phased roadmap

Validation gates reuse the two-mode pattern of
`testing/validate_pgd_optimizations.py`: **Mode 1** in-process kernel
equivalence against the frozen NumPy reference; **Mode 2** A/B comparison of
two completed runs on identical config+seed. Gate tiers follow its stage
convention: *stage-C gates* (bit-level: max|dx| < 1e-12, energy rel < 1e-10)
for result-preserving ports; *stage-A/B gates* (permutation-invariant
partition agreement ≥ 99.5 %, contour perimeter rel < 1e-3, v-weighted areas
rel < 1e-3; energy/max|dx| reported, not gated) for trajectory-altering
changes. Every phase also requires a clean `detect_dormant_cells()` result
**and** a clean `detect_area_imbalance()` result (worst discrete cell area
within `AREA_IMBALANCE_REL_THRESHOLD` of target; both in
`src/partition/find_contours.py`). The second gate is not redundant: a runt
cell passes the dormant check (peak density 1.0) while failing area balance,
and it is the failure mode that grows with N — see
`docs/reference/winner_take_all_partition_gap.md`.

### Phase 0 — Scaling reconnaissance and doc-05 re-anchor
**Status:** Not Started · **Effort:** ~2–3 days · **Win:** de-risks everything downstream
- Fold the measured N=100 point (40.6 h, K = 54,536, α_local ≈ 1.8) into
  `docs/math/05-phase1-nregion-scaling/main.tex` (sync rule).
- One profiled N=75 run + (config-only) level-schedule experiment at N=50
  (§3.3 item 3) to bound K(N) curvature and the cheap schedule win.
- Record `detect_area_imbalance()` worst-cell deviation at N=50/75/100 to map
  the onset of the discrete-area wall (N=50: 0.8%, N=100: 22%) — this locates
  where partition validity, not just speed, breaks. Cheap: reads existing
  solution files (no rerun for 50/100).
- Verify Pelle GPU inventory (§4); record findings here.
- **Gate:** none (measurement only).

### Phase 1 — PyTorch dense backend port (CPU/CUDA/MPS), no algorithm change
**Status:** Not Started · **Effort:** 1–2 weeks · **Win:** 10–30× on CUDA at N=100 (40.6 h → ~1.5–4 h); unlocks all later phases
- `backends/torch_backend.py`: projection (current alternating algorithm,
  verbatim), energy, gradient, constraint check as fused device ops; one host
  sync per major iteration.
- **Gate (Mode 1):** torch-CPU-fp64 projection vs NumPy reference over the
  size/seed grid — stage-C tolerances. **Gate (Mode 2):** full N=50 A/B run
  (torch-CPU-fp64 vs main): stage-C on energy/partition (reduction-order
  differences may force stage-A/B fallback on max|dx| — decide on evidence;
  perimeter rel < 1e-9 expected). Then N=50 CUDA-fp64 vs CPU-fp64: stage-A/B.

### Phase 2 — Dual semismooth-Newton projection (dense)
**Status:** Not Started · **Effort:** 2–3 weeks · **Win:** ~5–7× on the projection bucket; exact feasibility; prerequisite for sparsity
- Implement per-row simplex projection + Newton outer loop in the backend;
  keep the alternating projection as a fallback config choice.
- **Gate (Mode 1):** on random matrices, output is feasible to 1e-12 and its
  distance to input ≤ the iterative scheme's (true projection dominates).
  **Gate (Mode 2):** N=50 and N=100 A/B vs Phase-1 backend: stage-A/B gates.

### Phase 3 — Sparse localized representation (the crux)
**Status:** Not Started · **Effort:** 3–5 weeks · **Win:** memory 9.13 GB → 110 MB per field at N=1000; per-pass work at N=1000 ≈ today's N=100; N=100 wall → ~5–15 min on GPU
- ELL layout, seeded-only init, 1-ring support growth, truncation +
  diagnostics (support-occupancy histogram, truncated-mass counter) per §3.1;
  sparse HDF5 schema + `ContourAnalyzer` sparse loader (§3.5).
- **Gate (Mode 2):** N=50 *and* N=100 sparse-vs-dense A/B on identical
  config+seed: stage-A/B gates + truncated mass < 1e-8 of total + identical
  winner-take-all label sets. **Memory gate:** peak device memory ≤ 2 GB at
  a synthetic N=1000-shaped problem (V=1.14M, coarse iterations only).

### Phase 4 — Stepping: BB/spectral + optional Nesterov restart + schedule retune
**Status:** Not Started · **Effort:** 1–2 weeks · **Win:** ~3–6× fewer projection calls (2.9× from mBT alone)
- **Gate (Mode 2):** N=100 A/B vs Phase-3 backend: stage-A/B gates; K and
  projection-call counts reported and required to drop ≥ 2.5×.

### Phase 5 — Scale ladder and production hardening on Pelle
**Status:** Not Started · **Effort:** ~2 weeks engineering + queue time
- GPU SLURM support in `cluster/*.sh`; N = 250 → 500 → 1000 ladder (torus
  first), each with: dormant-cell check, area feasibility, support-occupancy
  histogram, measured K(N) appended to doc 05, and Phase-2 pipeline ingestion
  smoke test (contour extraction + one refinement iteration) at N=250.
- **Gate per rung:** valid N-region partition — zero dead/weak cells *and*
  `detect_area_imbalance()` worst-cell deviation within threshold (a runt cell
  passes the former but fails Phase 2); K and wall within 3× of the §5
  projection for the measured K scenario; explicit go/no-go before the next
  rung. A rung that scales fast but produces an area-imbalanced partition is a
  fail, not a pass.

**Total effort: ~10–14 weeks** of focused work, front-loaded with the
highest-certainty wins (Phase 1's GPU port and Phase 4's mBT elimination are
near-mechanical; Phase 3 carries the research risk).

---

## 7. This repo vs a new repo

**Recommendation: this repo, on feature branches, behind config flags.**
- The validation strategy *is* A/B execution of two backends on identical
  config+seed through the same pipeline — `validate_pgd_optimizations.py`
  Mode 2 depends on both living in one tree.
- Everything outside the hot loop is reused unchanged: mesh providers,
  multi-level orchestration, traces/profiling/analyzers, sweep tooling,
  cluster scripts, Phase 2.
- The NumPy path remains the frozen reference (as `reference_projection_iterative`
  already demonstrates the pattern in-repo).
- CLAUDE.md / docs sync rules keep one source of truth; a second repo would
  fork the documentation contract immediately.
A new repo would be justified only if the rewrite abandoned the pipeline
contract (HDF5 schemas, run layout) — this plan deliberately preserves both.
Torch stays an *optional* dependency (`pip install -e ".[gpu]"`), so the core
install is unaffected.

## 8. Dev (Mac) → prod (Pelle) workflow

1. Develop on M4 Pro: unit/kernel work + Mode-1 gates on CPU fp64
   (`device: cpu`); fast behavioral smoke runs on MPS fp32 with relaxed
   projection tol (never gated); small configs (`torus_10part`, truncated
   `torus_50part`).
2. Mode-2 N=50 gate locally on CPU fp64 (hours, tolerable), or directly on a
   Pelle CPU node with the identical config+seed.
3. Push branch → Pelle: `pip install -e ".[gpu]"` with the CUDA wheel;
   `submit_relaxation.sh --gpus 1` (new flag); first job re-runs the N=50
   Mode-2 gate CUDA-vs-CPU on the *same host pair* documented in
   `experiment.yaml`.
4. Scale ladder (Phase 5) exclusively on Pelle; results under
   `/proj/.../results/` as today; `--profile` mandatory on every ladder rung
   so `sweep/timing_analyzer.py --phase relaxation` tracks the breakdown.
5. Numerical policy: fp64 for anything gated or published; MPS fp32 output
   is never promoted to a result.

## 9. Top risks

1. **K(N) growth (Wall 2)** — flat from 50→100, but two points on one mesh
   schedule prove little; if K ∝ N² reasserts itself, N=1000 is days (still
   feasible) but N=10⁴ ambitions die. Mitigation: Phase 0/5 measurements
   gate each rung.
2. **Sparse support dynamics (Phase 3)** — truncation could bias interfaces
   or strand a cell's area budget; the dual-Newton area coupling only reaches
   cells through interface rows. Mitigation: diagnostics + dense-equivalence
   gates at N=50/100 before any trust at N=1000.
3. **Platform gaps** — no fp64 on MPS (dev gates confined to CPU), and Pelle
   GPU availability is an unverified assumption with the CPU-torch fallback
   costing ~an order of magnitude.

## Related documents
- Math: `docs/math/05-phase1-nregion-scaling/main.tex` (scaling fit this plan
  re-anchors), `docs/math/04-phase1-timing-profile/` (projection cost model)
- Reference: `docs/reference/PHASE1_PGD_SERIAL_OPTIMIZATION_AUDIT.md`
  (Changes A/B/C; audit #3 = dual-Newton projection, #13 = rejected dual
  warm-start), `docs/reference/winner_take_all_partition_gap.md` (dormant &
  runt cells — the discrete-area validity wall the §6 gates now enforce),
  `docs/reference/SCALABILITY_ANALYSIS.md`
- Plans: `docs/plans/PHASE1_SEEDED_INITIALIZATION_PLAN.md` (seeded init the
  sparse representation depends on)
- Code: `src/optimization/pgd_optimizer.py`, `src/optimization/projection.py`,
  `src/pipeline/relaxation.py`, `src/profiling.py`,
  `testing/validate_pgd_optimizations.py` (gate harness), `cluster/pelle_config.sh`
- Data: `results/run_20260629_141012_.../solution/timing_profile.yaml`
  (N=100 measured profile this plan is anchored on)
