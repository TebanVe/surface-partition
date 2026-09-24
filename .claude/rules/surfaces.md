---
paths:
  - "src/surfaces/**"
  - "parameters/double_torus*.yaml"
  - "parameters/banchoff*.yaml"
  - "parameters/ellipsoid*.yaml"
---

# Surface providers

## Adding one

**Parametric:** subclass `SurfaceProvider` (`src/surfaces/base.py`) and implement
`surface_name`, `resolution_labels`, `get_resolution`, `set_resolution`,
`get_initial_resolution`, `get_resolution_increment`, `resolution_summary`,
`build`, `theoretical_total_area` (may return `None`). See
`EllipsoidMeshProvider` for a parametric example with polar cap handling.

**Implicit (zero level sets):** subclass `ImplicitSurfaceProvider`
(`src/surfaces/implicit.py`) and implement only `surface_name`,
`implicit_function(x, y, z)` (vectorized; surface is `f = 0`) and
`bounding_box()`. The base class handles marching cubes, resolution tracking and
refinement scaling. Override `build()` for post-processing — e.g.
`BanchoffChmutovMeshProvider` filters to the largest connected component.

Then: add to `src/surfaces/__init__.py`, add a branch in
`scripts/find_surface_partition.py`, and create a YAML under `parameters/`.

`src/surfaces/factory.py` (`build_provider(cfg)` / `is_structured(name)`) is the
config -> provider path used by `run_mbo_arm.py`. `find_surface_partition.py`
deliberately keeps its own equivalent branch — consolidating them is a separate
change with its own regression burden. **`is_structured` is a whitelist of one**:
only the torus satisfies `V == res1*res2`; marching-cubes meshes do not.

## Status: PGD results on non-torus surfaces are INVALID, not merely unvalidated

Every archived ellipsoid / double-torus / Banchoff-Chmutov PGD run predates the
energy-discretization fix (`6ff71a0`) and used `init_method: random`. Graded with
the three gates on 2026-09-07, **all 21 FAIL**: 16 double-torus runs give 4-10 of
10 cells fragmented, and all 5 Banchoff-Chmutov runs give 12 of 12 fragmented at
14-30 components per cell.

**Approach B handles them cleanly** — `mbo_auction.py` was always mesh-generic
and the driver is now surface-agnostic. At N=10, all three gates pass on raw
labels with 0 fragmented: double torus 12,448 verts, worst cell 0.0468%, **7 s**;
Banchoff-Chmutov 26,928 verts, worst 0.0169%, **17 s**. Configs:
`parameters/{double_torus,banchoff_chmutov}_10part_seeded.yaml`.

⚠ `scripts/export_partition.py` is **still torus-only** — it writes the
link-list-torus schema, reads `config["surface"]["torus"]`, and asserts
`n_theta*n_phi == V`, which no marching-cubes mesh satisfies. Phase 1, Phase 2,
all three gates and the balanced readout work unchanged; only the export does
not. The general schema is specified in
`docs/reference/PARTITION_EXPORT_SCHEMA_GENERAL.md`; the implementation steps in
`docs/plans/GENERAL_SURFACE_EXPORT_PLAN.md`.

## Choose a marching-cubes grid by STIFFNESS CONDITIONING, not vertex count

Measured 2026-09-07. Marching cubes emits needle triangles and near-coincident
vertex clusters at **4-7% of triangles at every resolution tried** — the
fraction is irreducible, so refining does not clean it up. What varies, by
**140x between neighbouring grids**, is whether the worst sliver lands on the
surface: `max(K_ii)/median(K_ii)` is 29 at double-torus `88x59x18` and **4,064**
at `124x83x26`.

This matters to **Phase 1, not only to Phase 2 rendering** as
`docs/plans/MESH_DEGENERACY_AND_NEEDLE_TRIANGLES.md` frames it: on grid
`104x70x22` the largest `||g||` component sits on a triangle of aspect ratio
9,927 carrying `K_ii = 10,509` against a mesh median of **5.13**, and PGD's
Armijo line search is limited by exactly that component — one sliver throttles a
level.

Two consequences:

1. **The doc's suggested knob — raising `n_grid_z` — makes it worse**, because
   it breaks the cubic voxel `ImplicitSurfaceProvider` tries to preserve (median
   AR 1.63 -> 3.88 going `104x80x30` -> `104x80x120`). Size the grid to the
   bounding box instead: double torus 3.0x2.0x0.6 => z ~ 1 + 0.2(x-1);
   Banchoff's bbox is cubic => nx=ny=nz (the old `60x40x20` was 3x anisotropic).
2. **Select the ladder by scanning candidates**, requiring every rung clean. The
   shipped ones are `84x56x18 +16/+11` (8 outliers, all on level 0) and `48^3 +6`
   (**zero at every level**).

⚠ Do not extend either upward: a 4th double-torus rung reaches 199.8 with 20
outliers, and Banchoff degrades sharply above ~80^3 (82^3 -> 793, 86^3 -> 4,700).

Approach B is the less exposed method here — it has **no line search**, doing one
prefactorized LU solve per level, which tolerates a few stiff rows far better
than a max-norm-limited gradient step.

## lambda cannot be calibrated from ||g||_inf on a marching-cubes mesh

The force-parity rule used for `soft_area_mu` (`_calibrate_soft_area_mu`) returns
**lambda = 54.9** for the torus N=10 config whose known-good value is **3.25**,
and swings 109 -> 2,979 between two meshes *of the same surface* — because it
keys on `||g||_inf`, which is the sliver artefact above.

What works instead is matching the **energy** ratio
`|E_penalty| / (E_grad + E_interface)` at the seeded init, since energy is an
integral and robust to a few bad triangles: lambda = 1.2 (double torus) and 2.4
(Banchoff) against the torus N=10 reference.

⚠ Carry **±60%** — the target ratio is itself seed-dependent (0.1441 at seed
13001502 vs 0.0886 at 84172851 for the *identical* torus config).

This whole problem is a reason to prefer approach B on a new surface: **B has no
lambda**, tau being derived per level from the mesh and N.
