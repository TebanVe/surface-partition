# General-Surface Partition Export Schema (`schema_version` 2.0)

This document specifies how a finalised partition on a **non-torus** surface is
stored, and how a downstream consumer discovers which surface it is holding. It
is the contract a general-surface reader should be written against.

It exists because the current export format (`schema_version` 1.1, specified in
`../link-list-torus/docs/reference/PARTITION_FILE_FORMAT.md`) is torus-only in
four attributes and in four reader-side checks, while the partition payload
itself is already entirely surface-agnostic. This document defines the smallest
change that admits other surfaces **without altering the torus contract at all**.

**Status:** specification, not yet implemented. The implementation steps are in
`docs/plans/GENERAL_SURFACE_EXPORT_PLAN.md`.

## 1. The governing constraint: the torus path does not move

`schema_version` 1.1 is treated by the downstream consumer as *"a stable external
contract"*, and its reader validates `schema_version == "1.1"` **and**
`surface == "torus"`. Therefore:

- **Torus exports keep writing `schema_version = "1.1"`, byte-identical, indefinitely.**
  Nothing in this document changes what a torus run produces.
- **General surfaces write `schema_version = "2.0"`.**

The version namespace is *forked*, not bumped. The consequence is the property we
want: the existing torus reader cannot be affected even by accident, because it
refuses a 2.0 file at its first check with a clear message. That refusal is
correct behaviour, not breakage — a reader built for analytic-torus ground truth
genuinely cannot consume a genus-5 surface.

A shared `1.2` with optional fields was considered and rejected: it would require
editing the stable reader to relax checks it currently relies on.

## 2. What is already general (do not redesign it)

Measured against a real exported deliverable
(`results/mc_study/.../torus_partition_n25_V114144.h5`): **all nine datasets are
surface-agnostic**, and exactly **four attributes** are torus-specific.

| Item | Kind | General? |
|---|---|---|
| `/mesh/vertices`, `/mesh/faces` | data | ✅ |
| `/partition/sub_vertices`, `/sub_faces`, `/face_labels` | data | ✅ |
| `/snapshot/vertex_labels`, `/vp_edges`, `/vp_lambda` | data | ✅ |
| `/snapshot/triple_points/{triangle_index,cell_triple,steiner_xyz}` | data | ✅ |
| `/:schema_version`, `surface`, `n_cells`, `finalised`, `finalised_note`, `source_run_id`, `source_iteration`, `seed`, `final_perimeter`, `surface_partition_git_sha`, `created` | attrs | ✅ |
| `/partition:n_original_vertices`, `/snapshot:n_variable_points` | attrs | ✅ |
| **`/:R`, `/:r`** | attrs | ❌ torus geometry |
| **`/mesh:grid_shape`, `/mesh:vertex_order`** | attrs | ❌ structured-grid indexing |

`src/export/rep3_builder.py` — the Representation-3 subdivided-mesh builder, and
the substantive part of the exporter — contains **no torus reference at all**.

So 2.0 is 1.1 with the four torus attributes replaced by a self-describing
`/surface` group. Every dataset keeps its name, shape convention and dtype, which
means a downstream reader can share essentially all of its loading code between
the two versions.

## 3. The `/surface` group (the substantive addition)

A consumer must be able to answer "what surface is this, and what may I assume
about it?" from the file alone. `/surface` carries that.

| Attribute | Type | Required | Meaning |
|---|---|---|---|
| `name` | str | yes | `"torus"` \| `"ellipsoid"` \| `"double_torus"` \| `"banchoff_chmutov"`. Matches `experiment.surface` upstream and the key under `surface:` in the config. |
| `kind` | str | yes | `"parametric"` or `"implicit"`. Says which of the two blocks below is populated. |
| `structured` | bool | yes | Whether `V == resolution[0] * resolution[1]` holds and a `(u,v)` grid indexing of `/mesh/vertices` is meaningful. **True only for the torus.** See §4. |
| `genus` | int | yes | Topological genus, computed from the exported mesh as `(2 − χ)/2` with `χ = V − E + F`. |
| `euler_characteristic` | int | yes | `χ`. Written alongside `genus` so a consumer can re-derive rather than trust. |
| `bbox` | float64 (3,2) | yes | Axis-aligned bounding box of `/mesh/vertices`, `[[xmin,xmax],[ymin,ymax],[zmin,zmax]]`. |
| `params` | subgroup | yes | The surface's own parameters as scalar attributes. May be empty. |
| `implicit_expr` | str | if `kind=="implicit"` | The defining equation as a human-readable string, zero level set. See §5. |
| `voxel_size` | float64 | if `kind=="implicit"` | Marching-cubes grid spacing `h` along x. **Required** — the residual tolerance scales with it (§5). |
| `resolution` | int (2,) | if `structured` | The `(res1, res2)` pair; the 2.0 spelling of `grid_shape`. |
| `resolution_labels` | str (2,) | if `structured` | Axis names, e.g. `["n_theta","n_phi"]`; the 2.0 spelling of `vertex_order`'s content. |

Per-surface values, with the topology **measured from the exported meshes** (the
torus row is the method's own sanity check — χ = 0, genus 1, as it must be):

| `name` | `kind` | `structured` | χ | `genus` | `params` |
|---|---|---|---|---|---|
| `torus` | parametric | **true** | 0 | 1 | `R`, `r` |
| `ellipsoid` | parametric | false ⚠ | 2 | 0 | `a`, `b`, `c` |
| `double_torus` | implicit | false | −2 | **2** | `c` (tube thickness, 0.03) |
| `banchoff_chmutov` | implicit | false | −8 | **5** | *(none)* |

⚠ The ellipsoid is marked `structured: false` deliberately. `EllipsoidMeshProvider`
builds its own vertex array with polar caps and the identity
`V == n_theta * n_phi` has **never been checked**. Claiming a guarantee that has
not been verified is worse than declining to claim it; if it is ever needed,
verify first and then promote it.

## 4. `structured` is the load-bearing flag

The torus's `grid_shape` + `vertex_order` are not decoration. They let the
consumer compute analytic `(θ_i, φ_j)` for any vertex in O(1) directly from the
index, which is what its smooth-surface classifier and its analytic geodesic
distances are built on.

**No general surface has an equivalent.** A marching-cubes mesh's resolution pair
is a *sampling grid*, and the vertex count is whatever the level set happens to
intersect — a real double-torus solution carries `200 × 150 = 30,000` against
`V = 56,700`. There is no `(u,v)` per vertex to recover, and no analytic geodesic
distance.

A general-surface consumer must therefore plan for **mesh-based distance from the
outset**. This is intrinsic to the problem, not a gap in the export; the
downstream project already anticipates it ("future projects targeting general
surfaces lose analytic-distance tractability and are *mesh-only* by necessity").
`structured: false` is the file saying so explicitly, so a reader fails loudly at
load rather than silently mis-indexing.

## 5. `implicit_expr` is the generalisation of `R`/`r`

`R` and `r` currently let a consumer do three things: verify that the mesh lies on
the intended surface (`_check_mesh_on_torus`), compute exact surface normals, and
project a particle back onto the surface.

For an implicit surface, `f(x,y,z)` and `∇f` give **all three** — so recording the
defining function is the true analogue, not a placeholder:

| `name` | `implicit_expr` (zero level set) |
|---|---|
| `double_torus` | `(x*(x-1)**2*(x-2) + y**2)**2 + z**2 - c` |
| `banchoff_chmutov` | `T4(x) + T4(y) + T4(z)`, with `T4(t) = 8*t**4 - 8*t**2 + 1` |

The residual check that replaces `_check_mesh_on_torus` is then
`max |f(v)| < tol` over `/mesh/vertices`, with the same shape as the torus one.

### 5.1 The residual tolerance — measured, with a scaling law

**Do not test raw `|f|`.** `f` carries arbitrary scaling per surface: at comparable
mesh density the double torus reads `max|f| = 2.7e-3` while Banchoff-Chmutov reads
`2.4e-2`, a 9x spread that says nothing about mesh accuracy — the double torus's
`f` has units of length^4, Banchoff's is dimensionless.

**Test `|f| / |grad f|`**, which is the first-order distance from the vertex to the
level set and is therefore in length units and comparable across surfaces. On the
same two meshes it reads `4.1e-3` and `1.9e-3` — the same order, as it should.

Measured across all three levels of both shipped ladders, the normalised residual
follows `C * h^2` with `h` the voxel spacing, i.e. the **second-order** accuracy of
marching cubes' linear interpolation:

| surface | grid | `h` | `max |f|/|grad f|` | `/ h^2` |
|---|---|---|---|---|
| double torus | 84x56x18 | 0.03614 | 4.110e-03 | 3.146 |
| double torus | 100x67x21 | 0.03030 | 3.022e-03 | 3.291 |
| double torus | 116x78x24 | 0.02609 | 2.196e-03 | 3.227 |
| Banchoff | 48^3 | 0.04681 | 1.943e-03 | 0.887 |
| Banchoff | 54^3 | 0.04151 | 1.656e-03 | 0.961 |
| Banchoff | 60^3 | 0.03729 | 1.383e-03 | 0.994 |

`C` is constant to within 5% down each ladder and is **O(1)** on both surfaces
(3.2 and 0.95). So the recommended check is

```
max |f(v)| / |grad f(v)|  <  K * voxel_size**2 ,   K = 10
```

`K = 10` clears the measured `C` by 3x on the double torus and 10x on Banchoff —
enough margin for a surface with sharper curvature, tight enough to still catch a
genuinely wrong mesh (which fails by orders of magnitude, not by a factor of three).

**For scale, why the torus's `1e-10` cannot be reused:** an exactly-parametrised
torus mesh has `max|f|` of **4.4e-16 at 100x96 and 5.0e-16 at 348x328** — machine
precision, six orders inside its tolerance, and resolution-independent because the
vertices are placed by the analytic parametrisation. A marching-cubes vertex sits
on a *linear interpolant* of the level set, so its error is `O(h^2)` and lands
near **1e-3**. That is thirteen orders of magnitude apart. `1e-10` is not a
tolerance these meshes miss narrowly; it is the wrong kind of test for them.

## 6. What a general-surface reader must not assume

1. **No `(u,v)` per vertex** unless `structured` is true (§4).
2. **No analytic geodesic distance** on any 2.0 surface.
3. **Mesh quality: already advisory, and the torus already trips it.** The
   downstream `_check_mesh_quality` **warns, it does not raise** ("Check 8: mesh
   quality on `/partition`. Warn (don't raise)"), and *the currently accepted
   torus deliverables already breach both thresholds*: measured on the Rep-3
   subdivided mesh, `min_rel_area` is **exactly 0.0** and `min_interior_angle`
   **0.0 deg** for the shipped n=25, n=50 and n=200 torus partitions (4 of 248,826
   and 2 of 269,832 sub-triangles are exactly degenerate), against thresholds of
   `1e-6` and `1.0 deg`. This is a property of Rep-3 subdivision — a variable
   point sitting on a mesh vertex yields a zero-area sub-triangle — not of the
   surface. So mesh quality is **not a barrier** for general surfaces: they are no
   worse than production torus files on this measure. The thresholds appear
   calibrated for a base mesh rather than a subdivided one; recalibrating them is
   a downstream cosmetic matter, not a blocker.

4. **Genus is not 1.** Anything assuming a single handle, or two periodic
   directions, is torus-specific. The exported surfaces are genus 2 and genus 5.
5. **`final_perimeter` counts every interface twice.** It is
   `Σ_k Per(cell_k)`, so it is exactly `2 ×` the total interface length. This is
   true of 1.1 as well and is not a 2.0 change, but it is the single easiest
   number in the file to misread — see the perimeter-convention gotcha in
   `CLAUDE.md`.

## 7. Provenance for arm-produced partitions

Partitions on these surfaces are produced by approach B (`scripts/run_mbo_arm.py`),
whose run directories are named `arm_{mode}_{timestamp}_npart{N}_V{V}_seed{S}`
rather than `run_*`. `source_run_id` therefore records the **arm** directory name.
This is already the behaviour for the N=400–1000 torus deliverables and needs no
special handling; it is noted only so a consumer does not expect a `run_*` prefix.

## Related documents

- `docs/plans/GENERAL_SURFACE_EXPORT_PLAN.md` — the implementation steps for this spec.
- `docs/plans/MESH_DEGENERACY_AND_NEEDLE_TRIANGLES.md` — why marching-cubes mesh quality is what it is, and the unimplemented cleanup tool that would improve it.
- `docs/reference/MD_SIMULATION_EXPORT_NOTES.md` — what the downstream simulation needs from an exported partition.
- `../link-list-torus/docs/reference/PARTITION_FILE_FORMAT.md` — the 1.1 contract this extends. **Not ours to edit.**
- Code: `src/export/writer.py`, `src/export/rep3_builder.py`, `src/surfaces/factory.py`.
