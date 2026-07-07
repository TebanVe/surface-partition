# Phase 1 Energy: Mis-Discretized Double-Well (a Typo Copied From the Paper)

The Phase 1 Γ-convergence energy in `src/optimization/pgd_optimizer.py` implements
the double-well interface term **incorrectly**: it uses `interface_vec = u²(1−u)²`
where the correct quantity is `q = u(1−u)`. As a result the coded energy computes
`∫u⁴(1−u)⁴` (an 8th-degree well, ~25× too small in magnitude) instead of the
intended `∫u²(1−u)²`, **and** the coded gradient is not the gradient of the coded
energy. This is a faithful transcription of a **typo in the source paper**
(Bogosel & Oudet, arXiv:1606.02873, §3): the paper's prose and its gradient formula
both require `v = u(1−u)`, but its printed definition of `v` has an extra squaring.
The method still produces valid partitions at low N (the buggy term is still a
descent-driving binarization penalty), but the effective well and the descent
direction are both wrong, which distorts quantitative behavior — and every high-N
conclusion drawn so far (see `docs/reference/winner_take_all_partition_gap.md`) was
measured under these distorted dynamics.

**Status: found, not yet fixed** (discovered 2026-07-06). The fix is a one-token
change; because it shifts the energy scale ~25×, it requires re-tuning ε, λ, and
the refinement-trigger thresholds and a full re-validation.

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
`ε·uᵀKu`, `(1/ε)·well`, and `λ·penalty` — shifts substantially. After the fix,
**ε, `lambda_penalty`, and the refinement-trigger thresholds must be re-tuned**, and
the change must be re-validated (small/fast proxy first: coarse mesh, low N — no
cluster). The FD consistency check above should read ~1e-8 post-fix, and a clean
partition at N=30/50 confirms the re-tuning.

## Impact and why it went unnoticed

- **It still works at low N.** `∫u⁴(1−u)⁴` is still minimized only when `u∈{0,1}`, so
  it drives binarization; and the biased gradient keeps positive cosine (~0.89) with
  the true descent direction, so Armijo accepts steps and the energy decreases. The
  N≤50 partitions (crisp, downstream-consumed) are unaffected qualitatively.
- **It distorts the quantitative regime.** The effective well is flatter (8th vs 4th
  degree), so intermediate densities are under-penalized, and the gradient
  under-restores low-density "halo" mass. This is directly relevant to the high-N
  "runt" failure (`docs/reference/winner_take_all_partition_gap.md`): the runt's
  diffuse halo is priced ~25× too cheaply and its restoring force is attenuated, so
  the runt could soften once the well is corrected — the current runt measurements
  were taken under the buggy dynamics and must be re-taken after the fix.
- **No derivation existed to catch it.** There is no `docs/math` derivation of the
  Phase-1 energy; the only record of the form was CLAUDE.md, which documents the
  code (hence the typo). The finding surfaced from an independent second-opinion pass
  and was confirmed by the FD check above and by reading the paper's §3.

## Follow-up: a proper derivation

A `docs/math` derivation of the Phase-1 Γ-convergence energy — `∫|∇u|² = uᵀKu`,
`∫u²(1−u)² = qᵀMq` with `q = u(1−u)`, the gradients, the penalty term, and the
Modica–Mortola constant — is warranted (none exists today) and should be authored
**together with the fix**, so it derives the corrected, implemented code per the
`docs/math` scope policy (`docs/math/AUTHORING_GUIDE.md`). This reference doc carries
only the essential math needed to justify the finding; the formal derivation belongs
in `docs/math`.

## Related documents, code, and references

- Reference: `docs/reference/winner_take_all_partition_gap.md` (the high-N "runt"
  failure whose measurements were taken under these distorted dynamics).
- Code: `src/optimization/pgd_optimizer.py` (`compute_energy`, `compute_gradient` —
  the `interface_vec` definitions), `src/pipeline/relaxation.py`
  (`ε = √mean_triangle_area`; the energy scale the re-tuning must rebalance).
- Math (to be added with the fix): `docs/math/NN-phase1-energy-discretization/`.
- Paper: Bogosel & Oudet, *Partitions of Minimal Length on Manifolds*, Experimental
  Mathematics (2023); arXiv:1606.02873, §3 (the mis-printed `v` definition and the
  gradient formula).
