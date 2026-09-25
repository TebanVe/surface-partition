# Reply to surface-partition#4 — collapsed faces in the subdivided mesh

**Issue:** [TebanVe/surface-partition#4](https://github.com/TebanVe/surface-partition/issues/4)
— *"Consolidated partition export contains collapsed (zero-area) faces and
coincident vertices in the subdivided mesh"*. Opened 2026-06-01, **no comments**.

**Reporter's draft** (with the evidence and provenance):
`link-list-general-surface/docs/upstream/surface-partition-collapsed-faces.md`.
⚠ That file still says *"Draft to be filed"* — it **was** filed, as #4. Worth
updating there so it does not read as outstanding.

**Numbers below:** every figure comes from
`python scripts/check_degenerate_faces.py`, whose output is committed as
[`degenerate_faces.yaml`](degenerate_faces.yaml). Deterministic, no seeds.

**Recommended action:** post the body below as a comment on #4, then close it as
*answered, no change* — the decision and its revisit trigger are recorded in
`docs/reference/DOWNSTREAM_CONSUMERS.md`.

---

## Confirmed, and the export will not be changed

Thanks for the report — it holds up, and the measurement it prompted turned out
to say something more useful than the original count.

It reproduces exactly: on the file you cite I get **4 zero-area faces and 3
coincident vertices** out of 499,154 sub-faces. Your snippet is correct as
written.

### Q1: Are *finalised* exports guaranteed free of collapsed faces?

**No.** Measured across all 26 exports this repository has produced, counting
exactly-zero-area sub-faces the way you did, and alongside them the faces your
own relative mask excludes:

| export | `finalised` | `F_sub` | exact-zero | `≤ 1e-10 × median` | coincident V |
|---|---|---:|---:|---:|---:|
| your source file (N=10) | False | 499,154 | 4 | 329 | 3 |
| N=100, V=114,144 (flagship) | True | 258,454 | **0** | 1,050 | 2 |
| N=200, V=114,144 | True | 270,886 | 12 | 1,815 | 30 |
| N=300, V=114,144 | True | 281,192 | 8 | 2,346 | 13 |
| N=400, V=114,144 | True | 288,462 | **0** | 2,437 | 5 |
| N=1000, V=114,144 | True | 324,006 | **0** | 907 | 17 |
| n=50 (B1) | True | 248,826 | 4 | 680 | 3 |
| n=50 (B3) | True | 250,970 | **32** | 759 | 30 |
| n=200 (B2) | True | 269,832 | 2 | 1,645 | 17 |
| double torus, schema 2.0 | True | 27,604 | 6 | 107 | 3 |

**15 of the 24 finalised exports contain at least one exactly-zero face**, up to
32 of them. So the answer is no — and note the converse too: nine contain none.
**A consumer can rely on neither their presence nor their absence.** `finalised`
records that the Phase 2 best iterate was accepted at the migration-cycling
plateau; it says nothing about subdivision degeneracy.

### Q2: Is `pending_migration` expected to remove these?

**No.** That flag marks the Phase 2 migration-cycling plateau — the topology
oscillates so the flag never clears, which is why every high-N export is written
with `--force-finalised`. It is unrelated to subdivision.

Your diagnosis of the origin is right: Representation 3 inserts variable points
and Steiner points along cell boundaries, and a variable point landing on, or
arbitrarily near, a mesh vertex yields a zero-area sub-triangle. That is a
property of the representation, so it will recur in any export.

### The exactly-zero count is the wrong target

This is what decided it. Your `_DEGENERATE_AREA_REL = 1e-10` relative mask
catches **two to three orders of magnitude more faces** than the exact-zero
count, in every single file — the comparison held across all 26.

The sharpest case is the N=100 flagship deliverable: **exact-zero is 0, and your
mask still excludes 1,050 faces.** An exporter fix that dropped exactly-zero
faces would remove *none* of what your code already handles there, and under 1%
of it at N=200. It would look like a fix and change nothing that matters.

Your relative, median-normalised threshold is the right instrument — invariant to
surface size and mesh resolution, and it catches the sliver continuum you noted
but did not act on. It is strictly better than anything we could put in the file.

### Decision: resolution 3. The exporter is unchanged.

**(1) Drop/merge faces and reindex** is the option that would cause harm, and it
would fail silently. Simulation checkpoints persist `tri` as an `int32` face
index per particle (`io/checkpoint.py`), and `rebuild_state()` reattaches
`(tri, bary)` to a live mesh with no validation. A re-exported partition with
different face indexing would place every particle of every saved run on a
different triangle — no crash, no warning, just wrong answers.
`_assert_off_sliver` would not fire either, since there would be no slivers left
to sit on. It also fails our own acceptance gate by construction: any change to
the writer must re-export an existing torus deliverable **byte-identically**.

**(2) Ship a degenerate-face mask** is safe but creates a second source of truth.
Ours would mean "exactly zero"; yours means "≤ 1e-10 × median". Per the table
above the two disagree by a factor of ~1,000, and yours is the better definition.

**(3) Document it** — done, on our side:

- `docs/reference/PARTITION_EXPORT_SCHEMA_GENERAL.md` §6.3 — that shipped
  deliverables carry degenerate sub-triangles, and that the downstream
  mesh-quality check is advisory rather than a gate.
- `docs/reference/DOWNSTREAM_CONSUMERS.md` — the full reasoning, and the revisit
  trigger below.
- `scripts/check_degenerate_faces.py` — so the counts above are reproducible
  rather than quoted.

### Revisit trigger

Two circumstances reopen this: **a consumer that does not compute its own
degeneracy mask**, or **a point at which every existing simulation checkpoint is
being regenerated anyway** — which removes the reindexing hazard and makes
cleaning cheap. Absent either, the export stays as it is.

### One note on the format doc

Your resolution (3) asks for the caveat in `PARTITION_FILE_FORMAT`. That document
belongs to `link-list-torus` and is not edited from here, so if you want
"`sub_faces` may contain zero-area faces and coincident vertices" stated there
too, that is a change on your side. The measurement above is yours to quote.
