# Phase 1 Energy: Mis-Discretized Double-Well (a Typo Copied From the Paper)

The Phase 1 Γ-convergence energy in `src/optimization/pgd_optimizer.py` originally
implemented the double-well interface term **incorrectly** (now **fixed** — see the
Status line below): it used `interface_vec = u²(1−u)²` where the correct quantity is
`q = u(1−u)`. As a result the coded energy computed `∫u⁴(1−u)⁴` (an 8th-degree well,
~25× too small in magnitude) instead of the intended `∫u²(1−u)²`, **and** the coded
gradient was not the gradient of the coded energy. This is a faithful transcription of a **typo in the source paper**
(Bogosel & Oudet, arXiv:1606.02873, §3): the paper's prose and its gradient formula
both require `v = u(1−u)`, but its printed definition of `v` has an extra squaring.
The method still produces valid partitions at low N (the buggy term is still a
descent-driving binarization penalty), but the effective well and the descent
direction are both wrong, which distorts quantitative behavior — and every high-N
conclusion drawn so far (see `docs/reference/winner_take_all_partition_gap.md`) was
measured under these distorted dynamics.

**Status: FIXED and validated** (found 2026-07-06; fixed 2026-07-08 in commit
`6ff71a0`, branch `fix/phase1-energy-discretization`). The one-token change
`interface_vec = u*(1−u)` is in place in both `compute_energy` and `compute_gradient`
(`src/optimization/pgd_optimizer.py:95,138`); the gradient-vs-energy FD check now reads
~1e-8. Because the fix shifts the energy scale ~25×, it was re-validated with a re-tuned
λ. At N=30, **seeded** init gives a clean partition (worst-cell 0.7%) — but random init
now *traps* on the corrected (steeper) well, so **seeded init is mandatory** (see below,
and `docs/reference/winner_take_all_partition_gap.md` §3). At N=100, the corrected well
plus a moderate `lambda_penalty=5.1` resolves the high-N "runt" (worst-cell **22.5% → 0.8%**)
so that Phase 2 refinement now *decreases* perimeter (**−13.6%**) instead of crashing on
infeasibility. Full validation: `docs/experiments/02-corrected-energy-highn-validation/`;
formal derivation of the corrected discretization: `docs/math/06-phase1-energy-discretization/`.

## The intended energy (Bogosel & Oudet §3)

The relaxed Modica–Mortola-type functional the paper minimizes (per phase) is

```
F_ε(u) = ε ∫_S |∇_τ u|²  +  (1/ε) ∫_S u²(1−u)²
```

with the standard 4th-degree double well `W(u) = u²(1−u)²`. The paper's FEM
discretization (P1 elements, mass matrix M, stiffness matrix K) is:

- `∫|∇_τ u|² = uᵀ K u` — correct in the code.
- `∫u²(1−u)²` — write `W(u) = [u(1−u)]² = q²` with `q = u(1−u)`; the integral of the
  square of the P1 interpolant of `q` is `qᵀ M q`. So the **intended** discretization is
  `(1/ε) qᵀ M q` with `q = u(1−u)`, and its gradient is `(2/ε)(M q)·(1−2u)` (since
  `dq/du = 1−2u`).

## The typo, in the paper and in the code

The paper (§3, page 6) prints:

- energy: `∫u²(1−u)² = vᵀMv`, with **`v = u.²·(1−u).²`**  (i.e. `v = u²(1−u)²`)
- gradient: `∇(vᵀMv) = 2Mv·(1−2u)`

These three statements are mutually inconsistent, and the inconsistency has a unique
resolution:

- `vᵀMv` equals the target `∫u²(1−u)²` **only if `v = u(1−u)`** (with `v = u²(1−u)²`
  it equals `∫u⁴(1−u)⁴`).
- `2Mv·(1−2u)` is the true gradient of `vᵀMv` **only if `v = u(1−u)`** (`d(u(1−u))/du
  = 1−2u`; for `v = u²(1−u)²` the chain rule gives an extra factor `2u(1−u)`).

So **two** independent signals (the prose's target integral and the printed gradient
formula) require `v = u(1−u)`; only the printed *definition* of `v` says `u²(1−u)²`.
The printed `v = u.²·(1−u).²` is a **typo** (extra squares). The paper's method and
theory (the Γ-convergence Theorem 2.2) are correct; only this one discretization
formula is mis-printed.

`src/optimization/pgd_optimizer.py` transcribed the printed formulas verbatim:

```python
# compute_energy (interface term)
interface_vec = u ** 2 * (1 - u) ** 2                 # = the paper's printed v (the typo)
interface_term = (1 / eps) * (interface_vec.T @ (M @ interface_vec))   # = ∫u⁴(1−u)⁴, not ∫u²(1−u)²

# compute_gradient (interface term)
interface_vec = u ** 2 * (1 - u) ** 2
grad_interface = (2 / eps) * (M @ interface_vec) * (1 - 2 * u)   # not ∇ of the above
```

Both functions use `interface_vec = u²(1−u)²`, inheriting the typo, which turns one
printed error into two coupled problems: a wrong energy **and** a gradient that no
longer matches it.

## Numerical proof

Verified directly against the real code and the actual mass matrix (torus, 20×16,
ε=0.05):

| definition of `v` | is `2Mv·(1−2u)` the true gradient of `vᵀMv`? | does `ε·vᵀMv` equal ∫u²(1−u)²? |
|---|---|---|
| **`v = u(1−u)`** (correct) | **yes** — FD rel. err **1e-8** | **yes** — 0.718 vs 0.783 ✓ |
| `v = u²(1−u)²` (code / printed) | **no** — FD rel. err **1.23 (123%)** | **no** — 0.031 vs 0.783 (≈25× too small) |

And on the full `compute_gradient` vs `compute_energy` of the real optimizer, the
interface term alone is off by **152%** (the gradient is the wrong vector, not a
rounding artifact); the corrected pair is consistent to machine precision.

```python
# reproduce (from repo root)
import numpy as np, sys; sys.path.insert(0, '.')
from src.surfaces.torus import TorusMeshProvider
prov = TorusMeshProvider(20, 16, 1.0, 0.6); mesh = prov.build(); mesh.compute_matrices()
M = np.asarray(mesh.M.todense()) if hasattr(mesh.M, 'todense') else mesh.M
eps = 0.05; rng = np.random.default_rng(1); u = rng.random(len(mesh.v))
for name, v in [("u(1-u)", u*(1-u)), ("u^2(1-u)^2", u**2*(1-u)**2)]:
    g = (2/eps)*(M @ v)*(1-2*u)                       # paper's/code's stated gradient
    Ef = lambda uu, sq=(name!="u(1-u)"): (1/eps)*float(
        ((uu**2*(1-uu)**2) if sq else uu*(1-uu)) @ (M @ ((uu**2*(1-uu)**2) if sq else uu*(1-uu))))
    d = rng.standard_normal(len(u)); d /= np.linalg.norm(d); h = 1e-6
    fd = (Ef(u+h*d) - Ef(u-h*d))/(2*h)
    print(name, "grad-vs-FD rel err =", abs(float(g@d)-fd)/abs(fd))
```

## The fix

A single change, applied identically in `compute_energy` and `compute_gradient`:

```python
interface_vec = u * (1 - u)          # was:  u ** 2 * (1 - u) ** 2
```

The surrounding formulas — `(1/ε)·interface_vecᵀ M interface_vec` and
`(2/ε)·(M @ interface_vec)·(1−2u)` — are then correct and mutually consistent, and
match the paper's intent.

**This is not cosmetic.** The correct well is ~25× larger in magnitude than the
buggy one (measured above), so the balance among the three energy terms —
`ε·uᵀKu`, `(1/ε)·well`, and `λ·penalty` — shifts substantially, and re-tuning was
required. This was done (small/fast proxy first: coarse mesh, low N — no cluster):
the post-fix FD consistency check reads ~1e-8; `ε = √mean_triangle_area` was kept;
`lambda_penalty` was raised 2.1 → 5.1 for N=100; and clean partitions at N=30
(seeded) and N=100 confirm the re-tuning. Two behavioural changes followed from the
steeper well — **seeded init became mandatory** (random now traps; see below) and λ
became an effective lever (it was inert under the buggy well).

## Impact and why it went unnoticed

- **It still works at low N.** `∫u⁴(1−u)⁴` is still minimized only when `u∈{0,1}`, so
  it drives binarization; and the biased gradient keeps positive cosine (~0.89) with
  the true descent direction, so Armijo accepts steps and the energy decreases. The
  N≤50 partitions (crisp, downstream-consumed) are unaffected qualitatively.
- **It distorts the quantitative regime — and made λ a dead lever.** The effective
  well is flatter (8th vs 4th degree), so intermediate densities are under-penalized
  and the gradient under-restores low-density "halo" mass. This is directly relevant
  to the high-N "runt" failure (`docs/reference/winner_take_all_partition_gap.md`):
  the runt's diffuse halo was priced ~25× too cheaply. **Re-measured after the fix**
  (`docs/experiments/02-corrected-energy-highn-validation/`): correcting the well
  *alone*, at the original `lambda_penalty=2.1`, did **not** cure the runt (N=100
  worst-cell 21.8%, essentially unchanged from the buggy 22.5%) — but on the corrected
  well **λ becomes an effective lever again**, and a moderate `lambda_penalty=5.1`
  drives the runt to 0.8%, after which Phase 2 refinement converges (perimeter −13.6%).
  Under the buggy 25×-too-weak well, sweeping λ∈{1…10} had *no* effect on the runt;
  the mis-scaled interface term simply dominated the crispness reward. The fix restores
  the intended balance among `ε·uᵀKu`, `(1/ε)·well`, and `λ·penalty`, so λ can do its job.
- **No derivation existed to catch it.** There is no `docs/math` derivation of the
  Phase-1 energy; the only record of the form was CLAUDE.md, which documents the
  code (hence the typo). The finding surfaced from an independent second-opinion pass
  and was confirmed by the FD check above and by reading the paper's §3.

## New operating requirement after the fix: seeded init is mandatory

The corrected well is ~25× steeper, which fixed the runt but **introduced a new
trap**: on the corrected energy the *symmetric diffuse state* (every cell at the
uniform density `1/N` everywhere) is a genuine local minimum. There the well's
second derivative is large enough (`W''(1/N) = 2(1−6/N+6/N²)/ε ≫ 0` for the coarse
mesh) that the projected gradient damps small perturbations instead of amplifying
them, so a symmetry-respecting start never breaks symmetry. **Random** level-0
initialization (uniform-random densities, then projected) sits essentially at this
symmetric state and freezes: at N=30, `lambda_penalty=2.1`, random init ends with
23 imbalanced cells and a 43% worst-cell area error, whereas **seeded** (Voronoi)
init from the identical config ends clean (0.7% worst-cell). Under the *buggy*
(flatter) well this trap was much weaker, so random init used to "work" at low N.

Consequences, now documented in the configs and CLAUDE.md:

- **`relaxation.init_method: seeded` is mandatory** for the corrected energy at every
  N we run (it was already recommended for N ≥ 30 to avoid dormant cells; it is now
  required for a different reason — escaping the symmetric trap).
- Random init can still escape *if* λ is pushed hard (N=30 level-0 sweep: random
  escapes cleanly at λ ≥ ~10), but large λ over-crisps and costs perimeter (N=30
  full run at λ=50: perimeter 174.2 vs 117.8 seeded-λ=2.1, +48%). Seeded + moderate
  λ is the operating point.

See `docs/reference/winner_take_all_partition_gap.md` §3 and the head-to-head in
`docs/experiments/02-corrected-energy-highn-validation/`.

## The formal derivation

The `docs/math` derivation of the corrected Phase-1 Γ-convergence energy —
`∫|∇u|² = uᵀKu`, `∫u²(1−u)² = qᵀMq` with `q = u(1−u)`, both gradients, the λ variance
penalty, and the Modica–Mortola constant — now exists at
`docs/math/06-phase1-energy-discretization/`. It derives the corrected, *implemented*
code per the `docs/math` scope policy (`docs/math/AUTHORING_GUIDE.md`). This reference
doc carries only the essential math needed to justify the finding; the full derivation
lives there.

## Related documents, code, and references

- Reference: `docs/reference/winner_take_all_partition_gap.md` (the high-N "runt"
  failure — now **resolved** on the corrected energy with moderate λ; its earlier
  measurements were taken under these distorted dynamics).
- Experiment: `docs/experiments/02-corrected-energy-highn-validation/` (the N=30/N=100
  head-to-head validating the fix: runt 22.5%→0.8%, Phase 2 −13.6%, random-init trap).
- Math: `docs/math/06-phase1-energy-discretization/` (the formal derivation of the
  corrected discretization).
- Code: `src/optimization/pgd_optimizer.py:95,138` (`compute_energy`, `compute_gradient`
  — the corrected `interface_vec = u*(1−u)`), `src/pipeline/relaxation.py`
  (`ε = √mean_triangle_area`; the energy scale the re-tuned λ rebalances).
- Paper: Bogosel & Oudet, *Partitions of Minimal Length on Manifolds*, Experimental
  Mathematics (2023); arXiv:1606.02873, §3 (the mis-printed `v` definition and the
  gradient formula).
