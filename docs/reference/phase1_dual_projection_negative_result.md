# Phase 1 Exact Dual Projection — Negative Result (Not a Speedup)

**Status:** Closed investigation. Measured 2026-07-20. The **documentation** of the
attempt is on `main` — the derivation `docs/math/08-dual-newton-projection/` and the
measured study `docs/experiments/03-dual-projection-verification/` — so the reasoning
survives without checking out a branch. The **implementation**
(`orthogonal_projection_newton`, its flag-gating, tests and benchmark) lives only on
the branch **`feat/newton-projection`** (pushed to origin as a reference; **not
merged** to `main` and not intended to be, on these grounds). This document supersedes
the former `docs/plans/PHASE1_DUAL_NEWTON_PROJECTION_PLAN.md`, which has been removed.

---

## 1. Verdict

The exact dual projection (`orthogonal_projection_newton`) was built to replace the
incumbent `orthogonal_projection_iterative` on the premise that a faster projection
is the highest-leverage Phase 1 speedup (projection is 86–91 % of wall time). That
premise does **not** hold:

- **As a speedup: refuted.** The exact projection is **3–12× slower per call** than
  the iterative method across all mesh sizes measured, and **~parity at best** after
  the most generous fairness correction. It is never faster.
- **The gap widens with N** — the opposite of what the N=1000 goal needs. Fine-mesh
  warm ratio: 0.30× at N=10 → 0.16× at N=100.
- **As a correctness fix: still valid but not obviously worth it.** It computes the
  *true* Euclidean projection (machine precision) where the iterative method does not
  (see §6). At 5–6× the cost, correctness alone does not justify it as a drop-in.
- **As a prerequisite for a future sparse representation: possibly valid, unbuilt.**
  See §6.

Do not re-open this as a dense-projection speedup. If the projection is revisited, it
must be inside a sparse/localized representation (a different piece of work), or to
test whether an *exact* projection reduces the PGD **iteration count** (§7, untested).

---

## 2. What was attempted, and the premise

Phase 1 PGD projects the nodal density onto the feasible set
`{row-sum=1} ∩ {equal areas vᵀU=d} ∩ {box 0≤U≤1}` every gradient step and every
backtrack. Empirical profiling (`docs/math/04-phase1-timing-profile`,
`docs/reference/PHASE1_RELAXATION_TIMING_PROFILE.md`) shows this projection consumes
**86–91 % of wall time**. The natural conclusion was: make the projection faster and
Phase 1 gets dramatically faster.

The chosen replacement solves the projection **exactly** via its concave dual: a
per-vertex probability-simplex projection (inner) wrapped in an L-BFGS-B ascent on the
`N`-dimensional area dual `β`, with an optional semismooth-Newton polish for the last
digits of feasibility. The full derivation is `docs/math/08-dual-newton-projection`;
the disproof that the *iterative* method is not the Euclidean projection is
`docs/experiments/03-dual-projection-verification`.

The correctness half of that thesis was **confirmed** — the iterative method really
is not the projection, and the dual method really is exact (§6). The speed half was
**wrong**, for the reason in §4.

---

## 3. What is not doable / does not improve efficiency

### N=10 — the standard low-N config (torus_10part refines to V≈1.7×10⁵)

Per-call wall time (ms), synthetic crisp candidates, warm = β threaded across steps.
`× warm` = iterative / newton-warm (>1 would mean newton is faster).

| V (nt×np) | iter ms | newton-warm ms | × warm | newton-cold ms | × cold |
|---:|---:|---:|---:|---:|---:|
| 5,280 (80×66)    | 3.9  | 8.4   | 0.47× | 12.6  | 0.31× |
| 37,128 (204×182) | 29.7 | 49.1  | 0.60× | 71.4  | 0.42× |
| 97,744 (328×298) | 81.9 | 315.1 | 0.26× | 149.9 | 0.55× |
| **187,128 (452×414)** | **173.8** | **766.4** | **0.23×** | 263.8 | 0.66× |

*(newton tol = 1×10⁻¹⁰, the optimizer's real setting.)*
Tolerance-matching newton to the iterative method's real `1×10⁻⁸` barely moves it —
fine-mesh warm ratio 0.23× → 0.30×.

### N=100 — the target-scale regime, tolerance-matched (newton tol = 1×10⁻⁸)

| V (nt×np) | iter ms | newton-warm ms | × warm | newton-cold ms | newton outer iters | peak RSS |
|---:|---:|---:|---:|---:|---:|---:|
| 5,280    | 31.4   | 398.0   | 0.08× | 540.7    | 36 | 305 MB |
| 37,128   | 174.7  | 1,663.6 | 0.10× | 2,376.4  | 22 | 605 MB |
| 97,744   | 479.9  | 3,948.2 | 0.12× | 5,410.0  | 19 | 1,609 MB |
| **187,128** | **1,181.5** | **7,245.6** | **0.16×** | 10,037.4 | 16 | **3,292 MB** |

Raw, newton is **6–12× slower** at N=100. Memory scales ~linearly with N; at N=100
fine the process holds ~3.3 GB of V×N arrays (extrapolates to ~6–8 GB at N=200 fine —
the fine+high-N cell is memory-bound, not time-bound).

### The fairness correction (and why it still isn't a win)

The synthetic candidates above make the iterative method converge in
**mPII ≈ 6–9 inner iterations**, but *real* Phase 1 runs average **mPII ≈ 33**
(measured: `run_20260709_081548`, N=100, `mean_projection_inner_iters` = 32.6–34.0,
`projection_pct_wall` = 91.55). So the tables **understate** the iterative method's
real per-call cost by ~5.5×. Applying that (generous) handicap to the fine N=100 cell:

```
corrected iterative ≈ 1181 ms × 5.5 ≈ 6,500 ms   vs   newton-warm 7,246 ms
                                                  →  still ~1.1× slower
```

So **even under the most generous correction, the best case is rough parity — never a
speedup** — and that is the *good* end of the trend, which worsens with N.

---

## 4. Why it fails

Two independent reasons, both structural (not tuning):

**(a) The premise mis-read the bottleneck.** Projection is 86–91 % of wall time
because it is **called ~60,000 times per run** (every PGD step, every backtrack), each
call *cheap* (~33 iterations). It is death by call *count*, not by expensive calls.
The dual method attacks per-call cost and does not touch the call count — it optimizes
the wrong quantity.

**(b) The dual method makes each call cost *more*, and scales worse with N.** Per call
it runs L-BFGS-B for ~16–36 outer iterations (each an `O(V·N·logN)` simplex sweep over
all V rows), then a semismooth-Newton polish whose Jacobian assembly is `O(V·N²)`:

```
J = diag(v²·a) − aᵀ·((v²/m)·a)      # a is the V×N active-set mask  →  O(V·N²)
```

That `N²` term is the killer as N grows. The iterative method is `O(V·N)·mPII` with
mPII ≈ 33; the dual method is `~20 · O(V·N·logN) + O(V·N²)`. For the N=1000 target the
`O(V·N²)` polish dominates and the method scales the wrong way. Tolerance-matching to
`1×10⁻⁸` (which can skip the polish) does not rescue it — the L-BFGS outer count and
the `logN` simplex sorts still lose to a 33-iteration `O(V·N)` sweep.

---

## 5. The test and how to reproduce

The measurements above come from a self-contained micro-benchmark on the branch:

- **`testing/benchmark_newton_projection_speed.py`** (`feat/newton-projection`).
  Builds a torus mesh + lumped mass, synthesizes a crisp near-one-hot density,
  generates a deterministic PGD-like candidate trajectory, and times
  `orthogonal_projection_iterative` (with the optimizer's `1e-8` pre-clip) against
  `orthogonal_projection_newton` cold and warm, per resolution. Reports per-call ms,
  speedup ratios, iterative inner-iteration count (`mPII_it`), newton outer-iteration
  count, feasibility, and peak RSS. Flags: `--n`, `--resolutions`, `--newton-tol`,
  `--drift`, `--steps`.

```bash
git fetch origin && git checkout feat/newton-projection
# N=100 target-scale ladder, tolerance-matched (the decisive run):
python testing/benchmark_newton_projection_speed.py \
    --resolutions "80,66 204,182 328,298 452,414" --n 100 --steps 6 --newton-tol 1e-8
# N=10 ladder at the optimizer's real tol:
python testing/benchmark_newton_projection_speed.py \
    --resolutions "80,66 204,182 328,298 452,414" --n 10 --steps 15
```

**Methodology caveat (important):** the synthetic candidates are *easier* for the
iterative method than real PGD candidates (mPII ≈ 6–9 vs real ≈ 33), so the benchmark
**understates iterative** — i.e. it is biased *toward* newton, and newton still loses.
Raising `--drift` does not close the mPII gap (it pushes mPII down, not up); reproducing
the real ≈33 would require real gradient-step candidates from a live relaxation.

**Correctness is separately verified** (both on the branch, both pass):
`testing/test_newton_projection_equivalence.py` (agreement with an independent Dykstra
projection ≤ 2×10⁻¹⁴; KKT via the solver's own duals ≤ 2×10⁻¹⁶; idempotency ≈ machine
eps; FD checks of `∇q=−R` and `J`) and `testing/test_newton_projection_pathological.py`
(empty cell, one-hot rows, deficient rows, breakpoint ties).

---

## 6. What remains valid

The negative result is about *speed*, not correctness. Two things survive:

- **It is the true Euclidean projection.** Verified exact to machine precision against
  Dykstra and a solver-own-dual KKT certificate. The incumbent iterative method is
  *not* the projection — it returns a feasible but objective-suboptimal, path-dependent
  point (`docs/experiments/03-dual-projection-verification`: KKT residual 0.17–0.31 vs
  ~10⁻¹⁵; objective excess up to +64 %). Concretely the dual method attains a strictly
  lower `½‖U−Y‖²` (e.g. 8.12 vs 10.81 binding; 1.55 vs 2.07 crisp). If exactness is
  ever required, this is the correct tool.
- **It may be a prerequisite for a sparse representation.** The N=1000 scaling path
  (`docs/plans/PHASE1_N1000_SCALING_PLAN.md`) needs a projection that decouples per
  vertex so the density field can be stored sparsely/localized; the iterative method's
  global linear solve couples all vertices and cannot localize. The dual formulation
  (per-vertex simplex + N-dim dual) is structurally compatible with that. If realized,
  the value is **architectural**, in a future sparse rewrite — **not** as a drop-in
  dense speedup, which this document refutes.

---

## 7. The one unfalsified path (weak bet)

The only way the dual method could still reduce *total* wall time is if the exact
projection yields better PGD search directions and cuts the ~60,000 call **count**
enough to overcome a per-call deficit that *grows* with N. This is untested. It is a
weak bet — there is no clear mechanism by which a truer projection slashes iteration
count, and the per-call deficit worsens toward the target scale. It would be settled
by a short capped full-loop run (a few hundred PGD iterations) comparing
iterations-to-a-fixed-tolerance for `projection_method: iterative` vs `newton`, not by
any per-call benchmark. Not pursued here.

---

## 8. Pointers

- Branch (all artifacts): `feat/newton-projection`
- Implementation: `src/optimization/projection.py::orthogonal_projection_newton`
- Benchmark: `testing/benchmark_newton_projection_speed.py`
- Correctness tests: `testing/test_newton_projection_equivalence.py`,
  `testing/test_newton_projection_pathological.py`
- Derivation: `docs/math/08-dual-newton-projection`
- Correctness disproof of the incumbent: `docs/experiments/03-dual-projection-verification`
- Real timing anchor: `docs/reference/PHASE1_RELAXATION_TIMING_PROFILE.md`,
  `docs/math/04-phase1-timing-profile`
