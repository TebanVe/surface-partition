# Optimization Methods Primer — SLSQP, IPOPT, L-BFGS, Exact Hessian

## Purpose

This document is a **self-contained conceptual primer** on the optimization
choices exposed in `scripts/refine_perimeter.py` for Phase 2 (perimeter
refinement). It assumes **no prior optimization background** and builds
up from first principles to the specific trade-offs that determine
performance and solution quality at every problem scale.

Read this document when:

- You are choosing between `--method slsqp`, `--method ipopt`, and
  `--method ipopt --exact-hessian` and want to know why the options exist
  and what each one does.
- You are reviewing the reference documents
  (`docs/reference/SCALABILITY_ANALYSIS.md`,
  `docs/math/03-analytical-steiner-derivatives`) and need a non-technical
  lens for the mathematical statements in them.
- You are trying to decide whether the bottleneck in your runs is
  Python code, FD computations, or IPOPT internals.

The companion reference for empirical timings and cross-method benchmarks
is `docs/reference/SCALABILITY_ANALYSIS.md`.

---

## Table of Contents

1. [What the problem actually is](#1-problem)
2. [SLSQP vs IPOPT — the solver choice](#2-slsqp-vs-ipopt)
3. [L-BFGS memory vs exact Hessian — the curvature choice](#3-lbfgs-vs-exact)
4. [How this maps onto the current code](#4-current-code)
5. [Scalability to thousands of cells](#5-scalability)
6. [Summary in one table](#6-summary)
7. [Practical takeaway](#7-takeaway)
8. [Related documents](#8-related)

---

## 1. What the problem actually is <a id="1-problem"></a>

The perimeter-refinement step (Phase 2) is a **constrained nonlinear
optimization problem** of the following shape:

$$
\min_{\lambda} \; P(\lambda) \quad \text{subject to} \quad A_k(\lambda) = \bar A \quad \text{for each cell } k = 1, \ldots, N-1.
$$

- $\lambda \in \mathbb{R}^n$ is the vector of all variable-point parameters
  (one λ per VP).
- $P(\lambda)$ is the total boundary perimeter (a scalar).
- $A_k(\lambda)$ is the area of cell $k$, constrained to equal the target
  area $\bar A = A_{\text{total}} / N$.

There are no inequality constraints in your formulation — just $N-1$
equality constraints (one per cell, with the last one redundant because
the total area is conserved).

All the numerical optimization methods below are **iterative**: they start
at some initial λ, and at each step they use local information about $P$
and the $A_k$'s to produce a new λ that is closer to a local minimum.
They stop when some measure of "gradient and constraint violation are
small" falls below a tolerance.

Two things distinguish the available methods:

- **How they build the next step** — the "solver" choice: **SLSQP** vs
  **IPOPT**.
- **How much curvature information they use** — the "Hessian" choice:
  **L-BFGS approximation** vs **exact Hessian**.

These two choices are mostly orthogonal. IPOPT supports either L-BFGS or
exact Hessian; SLSQP is essentially always run with its own built-in
quasi-Newton approximation.

---

## 2. SLSQP vs IPOPT — the solver choice <a id="2-slsqp-vs-ipopt"></a>

### 2.1 Core vocabulary: gradient, Hessian, Lagrangian

- **Gradient** $\nabla P$ — vector of first derivatives. Points in the
  direction of steepest increase. Moving in $-\nabla P$ decreases the
  objective locally.
- **Hessian** $\nabla^2 P$ — matrix of second derivatives. Describes the
  local curvature of $P$. If you know $\nabla P$ and $\nabla^2 P$ at a
  point, Newton's method tells you the "ideal" step:
  $\Delta\lambda = -(\nabla^2 P)^{-1} \nabla P$.
- **Lagrangian**
  $L(\lambda, \mu) = P(\lambda) + \sum_k \mu_k (A_k(\lambda) - \bar A)$.
  The scalar obtained by adding, to the objective, each constraint
  multiplied by its own auxiliary variable $\mu_k$ (the **Lagrange
  multiplier**). For constrained optimization the natural target is a
  *saddle point* of $L$ — simultaneously a minimum over $\lambda$ and a
  maximum over $\mu$.
- **KKT conditions** — the system $\nabla_\lambda L = 0$ and
  $A_k - \bar A = 0$ for all $k$. Every constrained optimum satisfies
  these; methods differ in how they drive toward them.

### 2.2 SLSQP — Sequential Least-Squares Quadratic Programming

**Intuition:** at the current point, approximate the problem locally by
something simple, solve the simple problem exactly, take a step, repeat.

At each iteration SLSQP constructs:

- A **quadratic** model of the Lagrangian around the current λ (the
  objective plus a linear-in-λ constraint penalty, with a matrix that
  approximates $\nabla^2 L$).
- A **linear** model of each constraint $A_k$.

It solves this small quadratic-programming (QP) subproblem exactly,
obtaining a step direction, then does a line search along that direction.

Key properties for your project:

- It maintains its curvature matrix (the approximation of $\nabla^2 L$)
  internally using **BFGS rank-one updates** from successive gradient
  evaluations. You never provide a Hessian callback.
- Its QP solver and curvature matrix are **dense**. Memory cost is
  $O(n^2)$; factorisation cost is $O(n^3)$ per iteration in the worst
  case.
- SciPy's SLSQP implementation (Fortran, from the 1980s) caps out around
  $n \approx 500$–$1000$ variables before it becomes impractical. Beyond
  that its dense linear algebra is the bottleneck.
- Handles equality and inequality constraints equally well, including
  bound constraints.
- Generally robust and fast on *small* smooth problems.

### 2.3 IPOPT — Interior-Point OPTimizer

**Intuition:** instead of tracking the feasible / infeasible boundary
explicitly, **smear** the constraints into the objective and solve a
sequence of smooth approximate problems, gradually tightening.

At a high level:

1. Replace bound / inequality constraints with a **log-barrier penalty**
   in the objective (your problem has no explicit inequalities, but IPOPT
   adds them implicitly through variable bounds and slack variables —
   this is the "interior-point" machinery).
2. With a fixed barrier parameter $\mu$, solve
   $\min_\lambda L_\mu(\lambda, \mu)$ by a Newton-like iteration.
3. Decrease $\mu$ and repeat until it is essentially zero.

At the heart of each iteration is solving a **KKT linear system** of the
form

$$
\begin{bmatrix} H + \Sigma & J^\top \\ J & -\delta I \end{bmatrix}
\begin{bmatrix} \Delta\lambda \\ \Delta\mu \end{bmatrix}
= \begin{bmatrix} r_\lambda \\ r_\mu \end{bmatrix}
$$

where $H$ is the Hessian of the Lagrangian, $J$ is the constraint
Jacobian, and $\Sigma$, $\delta$ are small regularisation terms. This is
a **sparse** linear system.

Key properties for your project:

- IPOPT uses **sparse** linear algebra throughout. It can handle $n$ in
  the hundreds of thousands as long as $H$ and $J$ are sparse.
- It accepts **user-supplied Hessian and Jacobian callbacks** that return
  sparse (row, col, value) triplets.
- It is an open-source project (COIN-OR) with a long track record; IPOPT
  stands for "Interior Point OPTimizer" and is the standard for
  large-scale nonlinear programming.
- It supports either an **exact Hessian** (supplied by you) or an
  internal **L-BFGS approximation** — see §3.
- It is available in Python through `cyipopt`.

### 2.4 Side-by-side

| Aspect | SLSQP | IPOPT |
|---|---|---|
| Algorithm family | Sequential QP (SQP) | Primal-dual interior-point |
| How constraints are handled | Linearised constraints + active-set | Log-barrier + Newton on KKT |
| Hessian representation | Built internally via BFGS (dense) | User-supplied sparse callback, OR internal L-BFGS |
| Linear algebra | Dense | Sparse |
| Scales to | ~500–1 000 variables | $10^5$–$10^6$ variables |
| Quality on small smooth problems | Very good | Very good |
| Quality on large problems | Impractical | Designed for it |
| Robustness near infeasible starts | OK | Better (barrier keeps iterates interior) |
| Availability in your stack | `scipy.optimize` | `cyipopt` (optional dependency) |

**Summary.** SLSQP is a small, dense, classical Sequential-QP algorithm
suitable for problems with up to ~1 000 variables. IPOPT is a modern,
sparse, large-scale interior-point algorithm. For thousands-of-cells
ambitions, SLSQP is simply out of the running — it would run out of
memory or wall-clock before reaching a useful answer. Everything
interesting happens inside IPOPT, and the remaining decision is the
Hessian strategy.

---

## 3. L-BFGS memory vs exact Hessian — the curvature choice <a id="3-lbfgs-vs-exact"></a>

Here "exact Hessian" and "L-BFGS" are **two ways IPOPT can populate the
$H$ block in its KKT system**. Both result in the same kind of
algorithm (an interior-point iteration); they differ only in how much
second-derivative information each iteration uses.

### 3.1 Why the Hessian matters

Close to a minimum, the objective looks like a quadratic bowl. If you
knew the exact bowl (the exact Hessian), Newton's method would solve a
single linear system and jump to the minimum in one step. This behaviour
is called **quadratic convergence**: the error squares each iteration,
so a handful of iterations drives you to machine precision.

If you only know the gradient (not the curvature), you are doing
something closer to steepest descent, and the error decays only linearly
— many more iterations are needed, and the step sizes must be controlled
carefully via line searches.

Real-world optimisers sit in between: they use approximate curvature that
gets better as they run.

### 3.2 L-BFGS ("Limited-memory BFGS")

L-BFGS never stores $H$ explicitly. Instead:

- It remembers the last **$m$** pairs of $(\Delta\lambda,\, \Delta\text{gradient})$
  from past iterations (your config uses $m = 20$).
- A clever recursive formula (the "two-loop recursion") uses those $m$
  pairs to multiply any vector by an implicit approximation of $H^{-1}$
  in $O(mn)$ time.
- Memory cost: $O(mn)$ — in your case, a few hundred kilobytes even for
  $n = 10\,000$.
- Per-iteration cost: one gradient evaluation, no Hessian evaluation,
  one cheap linear-system-like application.
- Convergence rate: **superlinear** in the best case, but **linear** in
  practice on ill-conditioned problems — i.e., many more iterations than
  true Newton.

Crucially, L-BFGS uses only **first-derivative** information. The
second-derivative structure of your problem (how neighbouring variable
points couple through shared boundary segments, through Steiner points,
through the equal-area constraints) is only *inferred* from how past
gradients have changed. When different pieces of the problem have very
different curvature (as in the perimeter-vs-area balance your problem
has), L-BFGS's inferred curvature is often a blur of the true one, and
it takes many iterations to distinguish them.

### 3.3 Exact Hessian

"Exact Hessian" means that at each iteration, you explicitly compute and
hand IPOPT the matrix

$$
H = \sigma \, \nabla^2 P + \sum_k \mu_k \, \nabla^2 A_k
$$

— the Hessian of the Lagrangian. $\sigma$ is `obj_factor` and $\mu_k$ are
the current Lagrange multipliers; IPOPT tells you both before calling
your callback.

Because this is the true curvature, each iteration is a true Newton step
in the interior-point framework. You get local **quadratic convergence**:
once you are in a neighbourhood of the optimum, the iteration count
drops dramatically (often to 10–30 total iterations from a good start).

The trade is cost per iteration:

- **You must compute $H$.** In your codebase this is the work in
  `compute_perimeter_hessian_sparse`, `compute_area_hessian_sparse`, and
  the two Steiner Hessian FD routines.
- **IPOPT must factor the full KKT matrix.** The KKT matrix has more
  non-zeros with an exact $H$ than with a low-rank L-BFGS correction, so
  the internal sparse factorisation is more expensive per iteration.

### 3.4 The core trade-off

| | Per-iteration cost | Iteration count | Robustness | Solution quality near optimum |
|---|---|---|---|---|
| L-BFGS | Cheap | Many | Gentle, good for rough descent | Can stall before reaching tight tolerance |
| Exact Hessian | Expensive | Few (near optimum) | Sharper — exposes ill-conditioning if present | Reaches tight tolerance, triggers sharper VP motion |

The quality consequence is visible in your benchmark
(`docs/reference/SCALABILITY_ANALYSIS.md`, 10-partition run):

- L-BFGS reached the same perimeter value 24× faster but **triggered zero
  Type-1 migrations**. Reason: L-BFGS stops short of pushing VPs all the
  way to the triangle edge boundary because it never "sees" the sharp
  second-order coupling that would justify that; its inferred curvature
  is too diffuse.
- Exact Hessian **triggered 77 Type-1 migrations** and produced smoother
  boundaries, at the cost of being an order of magnitude slower overall.

For your application, where topology migrations are a first-class
concern, that quality gap is not decorative — it is structural. L-BFGS
gives you a "rough" optimum; exact Hessian gives you a "sharp" optimum.

---

## 4. How this maps onto the current code <a id="4-current-code"></a>

Your implementation already mixes analytical and finite-difference (FD)
computations. The map between "what IPOPT needs" and "what your code
provides" is the following.

### 4.1 What each mode asks of your code

| IPOPT needs | L-BFGS mode | Exact-Hessian mode |
|---|---|---|
| Objective $P(\lambda)$ | yes | yes |
| Objective gradient $\nabla P$ | yes | yes |
| Constraint values $A_k - \bar A$ | yes | yes |
| Constraint Jacobian $J$ (one row per constraint) | yes | yes |
| Lagrangian Hessian $H$ | **no** | **yes** |

The first four are always required. L-BFGS simply skips the last one;
IPOPT fabricates its own $H$ from the history of gradient evaluations.

### 4.2 What is analytical vs FD in your code

| Quantity | Regular (segment / boundary-triangle) | Steiner / triple-point |
|---|---|---|
| Perimeter value | analytical | analytical |
| Perimeter gradient $\nabla P$ | analytical (`vectorized_perimeter.py`) | analytical (closed-form from Fermat construction) |
| Area Jacobian $J$ | analytical (`vectorized_area.py`) | analytical (`vectorized_steiner.py`) |
| Perimeter Hessian $\nabla^2 P$ | analytical (`vectorized_perimeter.py`) | analytical (`vectorized_steiner.py`) |
| Area Hessian $\nabla^2 A_k$ | analytical (`vectorized_area.py`) | analytical (`vectorized_steiner.py`) |

Every entry is now analytical: the closed-form Steiner derivatives are
derived in `docs/math/03-analytical-steiner-derivatives` (the
finite-difference versions are retained as `*_fd_reference` regression
oracles).

### 4.3 How each mode actually exercises this

**L-BFGS mode** uses only the top four rows of §4.1 (value, gradient,
constraints, Jacobian). The Steiner area-Jacobian it uses every iteration
is now an analytical closed form, so its cost is negligible.
Per-iteration cost is dominated by IPOPT internals and gradient
evaluation, not Hessian work (there is none).

**Exact-Hessian mode** uses everything in §4.2. In particular:

- The bottom three rows apply at every single IPOPT iteration.
- The Steiner perimeter and area Hessians are now analytical closed
  forms (`compute_steiner_*_hessian` in `vectorized_steiner.py`),
  $\mathcal{O}(T)$ per iteration where $T$ is the number of triple
  points — the earlier nested-FD $\mathcal{O}(T^2)$ cost is gone.
- The analytical perimeter and area Hessians are vectorised in NumPy
  and accumulate via `np.add.at` on pre-computed offset arrays, so there
  is no Python-level loop left in the Hessian path.

With every Hessian piece analytical, the exact-Hessian path is now
*faster* per iteration than L-BFGS on the 10-partition benchmark, while
still doing the curvature work L-BFGS skips.

---

## 5. Scalability to thousands of cells <a id="5-scalability"></a>

Let me track each of the four possible solver × Hessian combinations
against your target $N \approx 1000$ using the scaling picture in
`docs/reference/SCALABILITY_ANALYSIS.md`.

Four rough scaling estimates first (sizes at $N = 1000$, based on §2–§3
of that document):

- Number of variables (VPs): on the order of a few tens of thousands.
- Number of triple points $T$: $\approx 2N = 2\,000$.
- Number of equality constraints: $N - 1 = 999$.
- The dense "Schur complement block" in IPOPT's KKT solve is
  $(N-1) \times (N-1)$, i.e. $1000 \times 1000$.

### 5.1 SLSQP + anything

**Not viable.** SLSQP's internal dense Hessian approximation is
$n \times n$, which at $n \approx 20\,000$ is $\sim 3$ GB. Its dense QP
factorisation is $O(n^3)$. Cannot reach $N = 1000$ at all.

### 5.2 IPOPT + L-BFGS

Per iteration:

- Cheap: $O(n)$ gradient + $O(nm)$ quasi-Newton update with $m = 20$.
- KKT system factorisation with a low-rank correction — manageable,
  dominated by Jacobian sparsity.
- **No Hessian evaluation.** The FD Steiner *Jacobian* is still there and
  costs $O(T)$ per iteration, but this is small compared to the KKT
  solve.

Iteration count: **this is where it breaks.** At 10 partitions L-BFGS
already failed to converge within 1000 iterations. Extrapolating,
$N = 1000$ would need vastly more iterations and produce correspondingly
worse boundaries (no Type-1 migrations, jagged boundaries). L-BFGS alone
is a fast but *incomplete* solver at scale.

### 5.3 IPOPT + exact Hessian

Per iteration:

- Full Hessian evaluation: analytical parts cost $O(n)$; FD Steiner parts
  cost **$O(T^2)$** per iteration. At $T = 2000$, that is
  $\sim 4 \times 10^6$ cheap evaluations per iteration.
- KKT factorisation with the full Hessian: the $(N-1) \times (N-1)$
  Schur complement block is **dense** and costs $O(N^3) \approx 10^9$
  floating-point operations per iteration just for that.

Iteration count: fewer than L-BFGS, but "fewer" here means maybe 50–100
instead of 1000+. That is not enough to offset the per-iteration cost at
this scale.

### 5.4 What the analytical-Steiner work buys

Two per-iteration optimisations of the exact-Hessian mode attacked the
$H$ evaluation cost; both are now done:

- **Vectorised Hessian accumulation**: the Python for-loop accumulation
  in the analytical perimeter / area Hessian was replaced by `np.add.at`
  on pre-computed offset arrays, flattening the `perimeter_hess` /
  `area_hess` buckets in the per-component profile of
  `compare_hessian_modes.py`.
- **Analytical Steiner**: the $O(T^2)$ nested-FD Steiner Hessian was
  replaced by an $O(T)$ analytical closed form
  (`docs/math/03-analytical-steiner-derivatives`). This was the single
  largest per-iteration saving — on the 10-partition reference the
  Steiner Hessian dropped from $\sim 1.5$ s/call to $\sim 1$ ms/call.

With analytical Steiner done, the per-iteration exact-Hessian work is
dominated by the **IPOPT KKT linear solve**, not by anything you can do
in Python — the exact-Hessian solve is now faster per iteration than
L-BFGS.

### 5.5 The two walls those plans do *not* break

Even after vectorised accumulation + analytical Steiner:

1. **The dense $(N-1) \times (N-1)$ Schur complement factorisation inside
   IPOPT** is $O(N^3)$. At $N = 1000$, this is $10^9$ ops per iteration,
   on IPOPT's side. You cannot touch this from Python. Mitigations are
   swapping the IPOPT linear solver (MA57, MA97 via the HSL library), or
   restructuring the problem so the Schur complement is sparse or
   block-structured.
2. **The iteration count** grows with $N$ because the problem becomes
   worse-conditioned as more constraints interact. Exact Hessian helps
   but does not eliminate this; the 10-partition run taking 525
   iterations (vs 55 for 5-partitions) is the canary. No per-iteration
   speedup fixes this.

These two walls are what `docs/reference/SCALABILITY_ANALYSIS.md` labels
**Tier 3.** Breaking them requires not faster code but a *different
problem formulation*:

- **Multigrid / coarse-to-fine (Tier 3B):** solve at $N = 10$, subdivide,
  solve only locally near new boundaries, repeat. Each individual solve
  is small.
- **Augmented Lagrangian / ADMM decomposition (Tier 3A):** split the
  global NLP into per-cell subproblems coupled by the area multipliers;
  each sub-solve is small and parallelisable.
- **Lloyd CVT initialiser (Tier 3C):** replace the monolithic solver by
  a geometric iteration that produces a near-optimal starting partition
  cheaply; use IPOPT only for a local polishing pass on the fine
  structure.

---

## 6. Summary in one table <a id="6-summary"></a>

| Choice | What it does | Wall-clock cost | Solution quality | Scales to |
|---|---|---|---|---|
| **SLSQP + BFGS** | Dense Sequential QP, classical | Fast at $n < 1000$, impossible beyond | Good | $n \lesssim 1000$ variables, $N \lesssim 10$ cells |
| **IPOPT + L-BFGS** | Sparse interior-point, curvature approximated from gradient history | Cheap per iter, many iters, sometimes many too many | Rough — misses sharp VP motion, jagged boundaries, few migrations | $N \approx 100$, with caveats |
| **IPOPT + exact Hessian** | Sparse interior-point, user-supplied $H$ — now all-analytical and vectorised | Per-iter dominated by IPOPT internal KKT solve; faster per iter than L-BFGS | Sharp — full migration triggering, smooth boundaries, tight validation guarantees | $N \approx 100$–$300$ practically before Schur / iter-count walls |
| **Tier 3** (multigrid, AL, Lloyd) | Restructures the problem; each sub-solve is one of the above at small $N$ | Per sub-solve is small; global cost scales as $N \log N$ or $N$ | Depends on scheme; polishing pass can recover full quality | **$N \approx 1\,000+$** |

---

## 7. Practical takeaway <a id="7-takeaway"></a>

- For **research-scale problems** ($N \lesssim 30$), IPOPT + exact
  Hessian with your current code is correct and adequate. The
  analytical-Steiner work has made it tighter and faster.
- For **medium-scale problems** ($N \approx 30$–$300$), the
  analytical-Steiner work (now done) is necessary but not sufficient —
  a Tier 3 outer loop is also needed. You can probably also get material gains from
  swapping IPOPT's default MUMPS linear solver for MA57 through HSL —
  that is a few lines of configuration, no code.
- For **thousands of cells**, none of the four solver × Hessian
  combinations on the monolithic formulation will work well enough. The
  Tier-2 work (validation harness + analytical Steiner) is a prerequisite
  (the exact-Hessian path needs to be as fast as possible, since it will
  be used repeatedly inside a hierarchical scheme), but the actual
  unlock is a Tier 3 outer loop —
  multigrid or augmented-Lagrangian decomposition being the most
  pragmatic candidates because they reuse your existing exact-Hessian
  solver as the inner kernel.

The right mental model is: the Tier 2 work (analytical Steiner) makes
**one IPOPT solve** fast enough to be affordable; Tier 3 plans arrange
the problem so that only **many small IPOPT solves at low $N$** are ever
needed, instead of one giant solve at $N = 1000$.

---

## 8. Related documents <a id="8-related"></a>

Cross-referenced planning and reference documents, in the recommended
reading order:

- **`docs/reference/SCALABILITY_ANALYSIS.md`** — empirical benchmarks,
  per-iteration cost breakdown by $N$, and the Tier 1 / 2 / 3 mitigation
  ladder. The authoritative reference for measured numbers.
- **`docs/reference/IPOPT_REFINEMENT_QUALITY.md`** — detailed analysis of
  the quality gap between L-BFGS and exact Hessian (jagged boundaries,
  migration deficit, restoration-phase losses) and when each matters.
- **`docs/math/03-analytical-steiner-derivatives`** — the closed-form
  derivation of the analytical Steiner first and second derivatives that
  replaced the FD Steiner code ($O(N^2) \to O(N)$ on the Steiner Hessian
  at $N \approx 1000$). The exact-Hessian path is now fully analytical;
  its validation harness is in `testing/` (`compare_hessian_modes.py`,
  `test_exact_hessian_vs_fd.py`, `test_steiner_gradient_analytical.py`,
  `test_steiner_hessian_analytical.py`).
- **`docs/math/01-phase2-derivatives`** — the mathematical derivations of
  every currently-implemented Phase 2 quantity (perimeter / area value,
  gradient, Jacobian, Hessian; Steiner FD schemes; Lagrangian assembly).
