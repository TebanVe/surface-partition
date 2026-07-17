# Implementation Plan — Territory-Aware Phase 1 Relaxation

**A winner-take-all (WTA) balance energy term + a discrete-area trim + a projected-gradient
optimizer fix, to close the high-N validity gap and scale toward N≈1000.**

**Status:** SPECIFICATION — ready to implement. Not started.
**Target branch:** `feat/phase1-territory-aware-relaxation` (this document lives on it).
**Audience:** an implementing agent with NO prior conversation context. This document is
self-contained for implementation; where it says "read", read before writing code.
**Genre:** executable implementation spec. The *why* (diagnosis, measured evidence) lives in
`docs/plans/PHASE1_N1000_VALIDITY_PLAN.md`; do not duplicate it — reference it.

---

## 0. Orientation — read these before writing any code

**Design rationale & measured diagnosis (the WHY):**
- `docs/plans/PHASE1_N1000_VALIDITY_PLAN.md` — the accounting identity, the √N scaling law,
  the two measured findings (frozen optimizer / artificial λ ceiling; and that fixing the
  optimizer alone does not close the gap). This plan implements its P1 + P2 + P3.
- `docs/reference/winner_take_all_partition_gap.md` §9 — the empirical λ-window / seed-lottery
  / two-sub-type picture the fix targets.

**The existing energy this term extends:**
- `docs/reference/phase1_energy_discretization_bug.md` and
  `docs/math/06-phase1-energy-discretization/main.tex` — the corrected Γ-convergence energy
  `E₀ = ε·uᵀKu + (1/ε)·qᵀMq` (q = u(1−u)) + λ·penalty. Understand it fully before adding to it.

**Codebase to analyze (what to extract from each):**
- `src/optimization/pgd_optimizer.py` — `compute_energy`, `compute_gradient`, `optimize`
  (the line search ~lines 306–341, the refinement triggers ~408–429). Extract the EXACT
  current energy assembly, the density-matrix layout/shape convention (V×N vs N×V), how the
  gradient is assembled and returned, and how/where the penalty is added. The new term must
  match these conventions exactly.
- `src/optimization/projection.py` — `orthogonal_projection_iterative`. Confirm it accepts an
  arbitrary per-cell area-target vector `d` (the trim updates `d`); note the equal-mass +
  sum-to-one alternating structure and the stall logic.
- `src/pipeline/relaxation.py` — the multi-level loop, `RelaxationConfig` (add fields here),
  `ε = √(mean triangle area)` assignment, `init_method` dispatch, and where the projection
  targets `d` are set/carried across levels.
- `src/partition/find_contours.py` — `detect_area_imbalance` and `detect_dormant_cells`.
  The WTA (argmax) territory computation here is REUSED by the trim (§4) and by the acceptance
  gate; do not re-implement it.
- `src/mesh/tri_mesh.py` — `M`, `K`, `v` (lumped mass = row-sum of M). `v` is the vertex area
  weight used throughout below.
- `testing/test_steiner_gradient_analytical.py` — the finite-difference gradient-check pattern
  to MIRROR for the new term's test.
- `testing/validate_pgd_optimizations.py` — the N=50 stage-A/B regression harness for P2.

**Conventions & standards (mandatory):**
- `CLAUDE.md` — style (Black, line length 88, Python 3.9), relative imports within `src/`,
  `get_logger(__name__)` + `@log_performance`, **no `print` in library code**, dataclasses for
  config, and the documentation-sync rules. Also the "zero-overhead when a feature is off"
  pattern used by `src/profiling.py` (guard all new work behind `if flag:`), and the
  "seeded init is mandatory at high N" rule.
- `docs/math/AUTHORING_GUIDE.md` — you WILL create `docs/math/07-phase1-wta-balance/`.

---

## 1. Notation and the existing objective (precise)

Let the mesh have `V` vertices and the partition have `N` cells.
- `U ∈ ℝ^{V×N}`, entries `u_{i,k}` = density of cell `k` at vertex `i` (the "claim"/"paint").
  Constraints, enforced by the projection: `Σ_k u_{i,k} = 1` ∀i (sum-to-one / simplex),
  `0 ≤ u_{i,k} ≤ 1` (box), and the equal-area constraint `Σ_i v_i u_{i,k} = d_k` (default
  `d_k = Ā`).
- `v ∈ ℝ^V`, `v_i > 0` = lumped mass (vertex area weight); `A = Σ_i v_i` = total area;
  `Ā = A/N` = fair share.
- Existing energy (unchanged): `E₀(U) = ε·Σ_k u_kᵀ K u_k + (1/ε)·Σ_k q_kᵀ M q_k + λ·P_cris(U)`,
  `q_k = u_k∘(1−u_k)`. Its gradient is in `compute_gradient`; do not modify it.

**This plan adds `E = E₀ + P_bal` (§2) and a periodic update to the targets `d` (§3), plus an
optimizer change that does not touch `E` (§4).** All three are independently flag-gated.

---

## 2. The WTA balance term — full specification and derivation

### 2.1 Soft territory (a smooth, differentiable surrogate for winner-take-all area)

The hard WTA area of cell `k` is `Σ_{i: k=argmax_l u_{i,l}} v_i`, which is non-differentiable.
Replace the hard indicator by a **power-normalized (Gibbs-type) soft assignment** with exponent
`p ≥ 1` (default `p = 2`):

    w_{i,k} = u_{i,k}^p / S_i,    S_i = Σ_l u_{i,l}^p,          (Σ_k w_{i,k} = 1)          (2.1)
    T_k    = Σ_i v_i · w_{i,k}                                  (Σ_k T_k = A, exactly)     (2.2)

`p = 1` gives `w = u` (no sharpening; the current blind spot). `p → ∞` → the hard argmax.
`p = 2` is the default: the smallest integer power that (a) sharpens toward argmax, (b) keeps
`{w_{i,k}}_k` a smooth partition of unity, and (c) yields a well-conditioned closed-form
gradient (§2.3). The math doc (§7) must justify `p = 2` and treat `p` as an exposed knob.
(Alternative surrogate to discuss but not default: temperature softmax
`w_{i,k} = e^{u_{i,k}/τ}/Σ_l e^{u_{i,l}/τ}`; power-normalization is preferred for its exact
`Σ_k T_k = A` and bounded gradient.)

### 2.2 The penalty

    r_k    = (T_k − Ā) / Ā            (dimensionless relative territory deviation)          (2.3)
    P_bal  = (γ/2) · Σ_k r_k²          (γ ≥ 0 the strength; §2.5 for calibration)           (2.4)

### 2.3 The gradient (full derivation — reproduce in the math doc)

Only vertex `i`'s own weights depend on `u_{i,k}`. With `w_{i,k}=u_{i,k}^p/S_i`:

    ∂w_{i,l}/∂u_{i,k} = (p·u_{i,k}^{p-1}/S_i)·(δ_{lk} − w_{i,l})                            (2.5)

(For `p=2`: `∂w_{i,l}/∂u_{i,k} = (2u_{i,k}/S_i)(δ_{lk} − w_{i,l})`.) Then

    ∂T_l/∂u_{i,k} = v_i · (p·u_{i,k}^{p-1}/S_i)·(δ_{lk} − w_{i,l})                          (2.6)
    ∂P_bal/∂u_{i,k} = (γ/Ā)·Σ_l r_l·∂T_l/∂u_{i,k}
                    = (γ·v_i/Ā)·(p·u_{i,k}^{p-1}/S_i)·( r_k − Σ_l r_l w_{i,l} )              (2.7)

Define the per-vertex scalar `m_i = Σ_l r_l w_{i,l}`. Final form (p=2):

    ∂P_bal/∂u_{i,k} = (2γ/Ā) · v_i · (u_{i,k}/S_i) · (r_k − m_i)                            (2.8)

**Vectorized computation (two reductions, O(V·N) — the mandated implementation):**
```
# U: (V,N) feasible density; v: (V,); Abar = A/N; p (default 2); gamma
Up   = U**p                       # (V,N)
S    = Up.sum(axis=1)             # (V,)   ; note S_i ∈ [N^(1-p)... , 1] (§2.4 stability)
W    = Up / S[:, None]            # (V,N)  partition of unity over cells
T    = (v[:, None] * W).sum(0)    # (N,)   soft territory ; sum(T)=A exactly
r    = (T - Abar) / Abar          # (N,)
P_bal = 0.5 * gamma * (r @ r)                                   # scalar energy
m    = W @ r                      # (V,)   per-vertex reduction over cells
grad_bal = (gamma * p / Abar) * v[:, None] * (U**(p-1) / S[:, None]) * (r[None, :] - m[:, None])   # (V,N)
```
Match `compute_energy`/`compute_gradient`'s actual array orientation; transpose if the code
stores `N×V`.

### 2.4 Demonstrated properties (the math doc must prove each)

1. **Total-area consistency:** `Σ_k T_k = Σ_i v_i Σ_k w_{i,k} = Σ_i v_i = A` exactly (2.2). So
   `P_bal` is compatible with the fixed total area and does not fight the sum-to-one constraint.
2. **Self-deactivation (no λ-style ceiling):** at balance `T_k = Ā ⇒ r = 0 ⇒ P_bal = 0` and,
   by (2.8), `grad_bal = 0`. The force fades to zero at the solution — contrast the λ crispness
   penalty whose gradient is nonzero at balance (the mechanism of its ceiling).
3. **Interface-band localization:** in the interior of cell `k` at vertex `i` (`u_{i,k}≈1`,
   others ≈0): `S_i≈1`, `W_{i,·}` ≈ one-hot at `k`, so `m_i ≈ r_k` and `(r_k − m_i) ≈ 0` ⇒
   `grad_bal ≈ 0`. The force is supported only on the blurry boundary band where `W_{i,·}` is
   spread. (This is the "move fences, not interiors" property.)
4. **No bias at the symmetric state:** `u_{i,k} = 1/N ∀k ⇒ w = 1/N, T_k = Ā, r = 0 ⇒ grad = 0`.
   It cannot deepen the random-init symmetric trap nor perturb seeded symmetry-breaking.
5. **Restorative for a starving cell:** `T_k → 0 ⇒ r_k → −1`; at vertices where cell `k` has
   some presence and `m_i > r_k` (neighbours less in deficit), the descent direction
   `−∂P_bal/∂u_{i,k} > 0` raises `u_{i,k}` — a built-in reward for winning territory (the thing
   `winner_take_all_partition_gap.md` §3 notes no existing term provides).
6. **Γ-consistency:** the soft-vs-hard territory gap is `O(ε·Perimeter_k)` (band-confined), so
   as `ε→0`, `P_bal → (γ/2)Σ(true-area deviation)²/Ā²`, which vanishes on the equal-area
   minimizer. Adding `P_bal` therefore does not change the perimeter Γ-limit
   (Modica–Mortola; Braides 2002 — see §11). The math doc must state this precisely, not merely
   assert it.

### 2.5 Strength `γ`, and N-invariance

Per-entry gradient magnitude (2.8) scales as `γ·v_i·(u/S)/Ā ∼ γ / (vertices-per-cell)` because
`v_i ∼ A/V` and `Ā = A/N`. Under the §5/P3 mesh policy (vertices-per-cell held fixed as N
grows), this magnitude is **N-invariant** — one `γ` works across N. Calibrate `γ` ONCE: pick it
so that a 5% territory deficit produces a band-vertex force comparable to a fixed fraction
(target ≈ 5–15%) of the local double-well gradient magnitude at that vertex; record the
procedure and the chosen value in the math doc. Do NOT retune `γ` per N.

---

## 3. The discrete-area trim (exact-equality controller — no gradient)

Periodically retarget the projection so its fixed point is exact **discrete** equality.

- Every `J` accepted iterations (default `J = 200`), compute the hard WTA territories
  `T_wta ∈ ℝ^N` via the existing `detect_area_imbalance` machinery (reuse; do not
  re-implement the argmax).
- Update the per-cell projection targets:
  `d_k ← clip( d_k + β·(Ā − T_wta,k),  Ā·(1−c),  Ā·(1+c) )`, defaults `β = 0.5`, `c = 0.20`.
  Then renormalize `d ← d · (A / Σ_k d_k)` to preserve `Σ_k d_k = A` (the projection requires
  a feasible target sum).
- **State/lifecycle:** carry `d` within a level; **reset `d ← Ā·1` at the start of each new
  level** (after interpolation), since the mesh — and thus `T_wta` — changes.
- This deliberately unpins the *continuous* masses by `O(ε·Perimeter_k)`. Phase 2 is
  indifferent: its iteration-0 feasibility is exactly the *discrete* equality this delivers.
- Cost: `O(V)` for the argmax + `O(N)` update, only every `J` iters — negligible.

---

## 4. Optimizer/trigger fix (P2 — enabling prerequisite; does not touch `E`)

Rationale and measured evidence: `PHASE1_N1000_VALIDITY_PLAN.md` §2.3. The current line search
freezes on a non-stationary iterate (≈61% of the gradient is feasible descent left unused) and
the plateau trigger reads that freeze as convergence.

1. **Reduced (projected) gradient step.** Instead of `x⁺ = P(clip(x − s·g))`, step along the
   gradient projected onto the tangent space of the *active* constraints restricted to the free
   set (entries not pinned at a box bound with outward gradient): `g_t = g − α⊗1 − v⊗βᵀ`, where
   `α ∈ ℝ^V` (sum-to-one duals) and `β ∈ ℝ^N` (area duals) solve the two coupled normal
   equations, obtained by a few (≈5–10) Gauss–Seidel sweeps over the (α, β) blocks with
   warm-started duals. Then `x⁺ = P(clip(x − s·g_t))`.
2. **Acceptance test:** proximal-form sufficient decrease `E(x⁺) ≤ E(x) − (c/s)‖x⁺ − x‖²`
   (equivalently Armijo against `‖g_t‖²`). Empirically the achievable rate matches
   `dE ≈ −s‖g_t‖²`.
3. **Trigger fix:** replace raw gradient-norm criteria with the projected stationarity measure
   `‖g_t‖` (or `‖x⁺ − x‖/s`); make `refine_delta_energy` relative to the level's cumulative
   decrease; and classify **"no accepted step within the patience window" as STALLED, not
   converged** — log a warning and do NOT fire the refinement trigger on it.
4. (Future, out of scope here) an exact dual semismooth-Newton projection would remove the
   measured projection non-idempotency; note it in the doc, do not implement now.

Validate P2 with `testing/validate_pgd_optimizations.py` stage-A/B at N=50: it must preserve the
documented serial-optimization equivalence within tolerance when the WTA term is off.
**Expected (and required) caveat:** P2 alone must NOT be expected to fix validity — it is
necessary, not sufficient. Confirm the gate metric is essentially unchanged by P2 alone.

---

## 5. Mesh-budget validity floor (P3 — policy, minimal code)

Do NOT decouple `ε` below `h`: `ε = √(mean triangle area)` is the FEM-resolvability coupling
(`ε < h` under-integrates the well and mesh-pins interfaces). The correct policy is **hold
vertices-per-cell fixed, i.e. `V ∝ N`**. This is a *config/experiment* policy (choose base mesh
and levels so the finest level gives ≥ ~600 verts/cell for the gate margin; with the WTA term in
the loop this floor may relax to ~250–300 — measure, don't assume). Code change is optional: an
optional spatially varying `ε(x) = √(local triangle area)` (per-triangle weights in K/M assembly;
the Modica–Mortola constant is ε-independent so the perimeter limit is unchanged) to remove the
torus 4:1 anisotropy. Treat the ε(x) variant as a stretch goal, gated off by default.

---

## 6. Implementation steps (in order)

**Stage 0 — Pre-implementation verification (no energy changes; cheap):**
- Re-verify the frozen-optimizer finding on the existing solution
  `results/run_20260714_224821_.../solution/*.h5`: compute the reduced-gradient KKT residual and
  the projection non-idempotency energy cost at the final iterate. Confirm the §2.1 accounting
  identity `T_k − Ā = gain_k − lost_k` holds. Write a short throwaway script under
  `scripts/debug_archive/` or `testing/`. This gates whether P2 is worth building. Report numbers.

**Stage 1 — Math document** `docs/math/07-phase1-wta-balance/` (follow AUTHORING_GUIDE):
`main.tex` deriving (2.1)–(2.8), proving the six properties of §2.4, justifying `p=2` and the
`γ` calibration, and stating Γ-consistency. Add any new bib keys to
`docs/math/shared/references.bib`. Build the PDF (`make -C docs/math/07-phase1-wta-balance`).

**Stage 2 — WTA balance term** in `src/optimization/pgd_optimizer.py`: add `P_bal` to
`compute_energy` and `grad_bal` to `compute_gradient`, exactly as §2.3, vectorized, guarded by a
config flag (default off). Add the numerical guard `S = maximum(S, tiny)` (though `Σu=1, u≥0`
⇒ `S ≥ N^{1-p} > 0`, guard defensively since the gradient may be evaluated pre-projection).

**Stage 3 — Finite-difference gradient test** `testing/test_wta_balance_gradient_analytical.py`
(mirror `test_steiner_gradient_analytical.py`): on a small deterministic feasible `U`
(e.g. V=200, N=8, fixed `np.random` seed, projected to the simplex), central-difference
`(P_bal(U+δE)−P_bal(U−δE))/(2δ)` vs `⟨grad_bal, E⟩` for several random directions `E` and several
`δ` (Richardson); relative error must be `< 1e-6`. Also assert `grad_bal ≈ 0` at a balanced `U`
and exactly `0` at `U ≡ 1/N`. **This is the correctness gate — nothing downstream proceeds until
it passes.**

**Stage 4 — Discrete-area trim** in `src/pipeline/relaxation.py` per §3, flag-gated, reusing
`detect_area_imbalance`. Reset `d` per level.

**Stage 5 — Optimizer/trigger fix (P2)** in `src/optimization/pgd_optimizer.py` per §4,
flag-gated; regression-test with `validate_pgd_optimizations.py` at N=50.

**Stage 6 — Confirming experiment** (§9). **Stage 7 — scale + experiment writeup** (§9).

**Config schema** — add to `RelaxationConfig` (`src/pipeline/relaxation.py`), read by
`from_yaml_dict`, all defaulted so existing configs are byte-for-byte unchanged:
```
wta_balance_enabled: bool  = False
wta_balance_gamma:   float = 0.0
wta_balance_power:   float = 2.0
wta_trim_enabled:    bool  = False
wta_trim_period:     int   = 200
wta_trim_damping:    float = 0.5
wta_trim_clamp:      float = 0.20
pgd_reduced_gradient: bool = False   # P2 master flag (step + acceptance + trigger fix)
pgd_dual_sweeps:      int  = 8
```

---

## 7. New documents to create

- `docs/math/07-phase1-wta-balance/{main.tex, Makefile, main.pdf}` — the derivation (Stage 1);
  update `docs/math/Makefile` `DOCS` list and `docs/math/shared/references.bib`.
- `docs/experiments/03-territory-aware-highn-validation/` — AFTER Stage 6/7, the measured
  before/after (mirror `02-corrected-energy-highn-validation/`: provenance block, `make_figures.py`,
  `main.tex`, status label). Records whether the fix worked.
- Update `docs/plans/PHASE1_N1000_VALIDITY_PLAN.md` status as stages complete; per the repo
  convention, when fully implemented move its lasting explanation to reference / delete the plan.
- Update `CLAUDE.md` on landing: the new energy term + config flags + the new test (doc-sync rule).
- Update `docs/reference/winner_take_all_partition_gap.md` §9 with the resolution IF Stage 6 confirms.

---

## 8. Scientific-computing & efficiency requirements (non-negotiable)

- **Vectorization:** no Python loops over `V` or `N` in `compute_energy`/`compute_gradient`/the
  trim; pure NumPy as shown in §2.3. The term must stay `O(V·N)` (same class as `E₀`) and must not
  regress the PGD hot loop when the flag is off (zero-overhead: guard with `if enabled:`).
- **Correctness gate:** the analytical gradient MUST match finite differences (Stage 3) before any
  production run. Treat a failed FD check as a hard stop.
- **Numerical stability:** demonstrate `S_i` bounded away from 0 under the constraints; still guard.
  Avoid catastrophic cancellation in `r`; prefer the `(T−Ā)/Ā` form.
- **Reproducibility:** deterministic tests (fixed seeds); no wall-clock/random in library code.
- **Style:** Black (88), relative imports in `src/`, `get_logger`, `@log_performance` on the hot
  functions if consistent with the file, **no `print` in `src/`**, dataclass config.
- **Backward compatibility:** with all new flags off, an existing config (e.g.
  `parameters/torus_100part_coarse_seeded.yaml`) must produce a result identical to `main` — add a
  quick equivalence check to the PR notes.

---

## 9. Validation protocol

**Stage 6 — the cheapest confirming experiment (a step toward N=1000, not a N=300 one-off):**
Re-run the KNOWN-FAILING N=200 configuration and change ONLY the new flags:
- Base: `parameters/torus_200part_coarse_seeded.yaml` semantics but λ=9, seed 84172851, 3 levels
  (this reproduces `run_20260712_224424`: two entrenched runts at −34%/−38%, immune to λ 7→9).
- Enable: `wta_balance_enabled: true` (γ = calibrated value), `wta_trim_enabled: true`,
  `pgd_reduced_gradient: true`. Nothing else changed (same seed, same λ).
- **Success** = `detect_area_imbalance` reports `n_imbalanced: 0` (worst < 5%) WITHOUT touching
  seed or λ — i.e. the mechanism rescues the entrenched-runt failure on a *bad* seed (the
  anti-lottery property N=1000 needs). **Refute** = runts persist ⇒ fall back to P2+P3 and the
  hard-constraint (augmented-Lagrangian territory-equality) variant.
- Cost ≈ 6–15 h CPU against an existing failed control.

**On success:** N=300 A/B vs `run_20260714_224821` (predict 9 → ≤1 over gate); then the
N=250/500/1000 ladder under the §5 mesh budget; write `docs/experiments/03-…`.

---

## 10. Definition of done

- [ ] Stage 0 numbers reported (frozen-optimizer + identity verified).
- [ ] `docs/math/07-…/main.pdf` built; (2.1)–(2.8) derived; six properties proven; `p=2` and `γ`
      justified; Γ-consistency stated.
- [ ] WTA term + gradient implemented, vectorized, flag-gated; **FD test passes < 1e-6**.
- [ ] Trim implemented, flag-gated, reusing `detect_area_imbalance`, `d` reset per level.
- [ ] P2 implemented, flag-gated; N=50 regression passes; P2-alone confirmed non-sufficient.
- [ ] Backward-compat: flags-off run identical to `main`.
- [ ] Stage 6 experiment run; result (confirm/refute) recorded in `docs/experiments/03-…`.
- [ ] `CLAUDE.md` + affected `docs/` updated per the sync rule.

---

## 11. References

- B. Bogosel, É. Oudet. *Partitions of Minimal Length on Manifolds.* Experimental Mathematics
  31(3), 2023. arXiv:1606.02873. — the method, the ε∝h coupling, the equal-area constraint.
- L. Modica, S. Mortola. *Un esempio di Γ-convergenza.* Boll. Un. Mat. Ital. B (5) 14, 1977.
- L. Modica. *The gradient theory of phase transitions and the minimal interface criterion.*
  Arch. Rational Mech. Anal. 98, 1987, 123–142. — the Modica–Mortola Γ-limit constant.
- A. Braides. *Γ-convergence for Beginners.* Oxford University Press, 2002. — for the
  Γ-consistency argument of §2.4(6).
- J. Nocedal, S. J. Wright. *Numerical Optimization*, 2nd ed., Springer, 2006. — gradient
  projection & KKT (Ch. 12, 16–18); finite-difference gradient checking (§8.1) for Stage 3.
- D. P. Bertsekas. *Nonlinear Programming*, 2nd ed., Athena Scientific, 1999. — the gradient
  projection method (§4) and projection onto convex sets.
- Q. Du, V. Faber, M. Gunzburger. *Centroidal Voronoi Tessellations: Applications and
  Algorithms.* SIAM Review 41(4), 1999. — for the optional deterministic Lloyd/CVT seed step.
- Existing internal derivations to match in style/rigor: `docs/math/01-phase2-derivatives/`,
  `docs/math/03-analytical-steiner-derivatives/`, `docs/math/06-phase1-energy-discretization/`.
