# Downstream consumers of the exported partition

Exported partitions leave this repository. This document records **who consumes
them, what each consumer relies on, and what therefore must not change**. It is
the per-relationship record; the per-file record is
[`deliverables.yaml`](deliverables.yaml), verified by
`python scripts/check_deliverables.py`.

A contract is a property of a repository *pair*, not of a file, which is why it
lives here and not in the YAML. Adding a consumer means adding a section here and
one block in `deliverables.yaml` — it does not mean editing every row.

## Why this matters

Three ordinary-looking actions are externally visible:

1. **Changing the export schema** — even adding an attribute. A consumer's reader
   validates a fixed set.
2. **Renaming or moving a deliverable** — consumers reference files by path.
3. **Deleting a run** — `results/` is gitignored, so a deleted run cannot be
   recovered from version control, and its export goes with it.

Check this document before any of the three.

---

## `link-list-torus` — the active consumer

**Repo:** https://github.com/TebanVe/link-list-torus
**Reader:** `src/lnk_list_torus/partition/reader.py`

Runs **Monte Carlo, Brownian dynamics and molecular dynamics** of particles on
the partitioned torus. Particles are classified by which cell they occupy, so the
exact boundary geometry matters — that requirement is what selected
Representation 3 (subdivided boundary-cut mesh with per-face cell labels); the
options analysis is in [`MD_SIMULATION_EXPORT_NOTES.md`](MD_SIMULATION_EXPORT_NOTES.md).

**Consumes:** every finalised torus export — groups `common-mesh`, `finer-mesh`
and `mc-study`.

### What its reader enforces

| Check | Behaviour | Consequence for us |
|---|---|---|
| `surface == "torus"` | **raises** | The hard gate. This, not the version check, is what keeps a non-torus file out |
| `schema_version in ("1.0", "1.1")` | **warns only** — *"attempting to read anyway"* | An unrecognised version is not refused. Do not rely on the version check to protect this reader |
| `finalised == True` | gated on | Every high-N export needs `--force-finalised` (the Phase 2 plateau leaves `pending_migration=True` on the best iterate) |
| Mesh quality (check 8) | **warns, does not raise** | Accepted torus deliverables already breach both thresholds — a property of Rep-3 subdivision, not of the partition. See [`PARTITION_EXPORT_SCHEMA_GENERAL.md`](PARTITION_EXPORT_SCHEMA_GENERAL.md) §6.3 |

⚠ **The version check is advisory, not protective.** An earlier description in
this repo said the reader "refuses a 2.0 file at its first check". It does not —
it warns and reads on. The fork is still safe because 2.0 is only ever written
for non-torus surfaces, which then fail the surface check. But the protection
comes from the surface check alone.

### What must not change

- **`schema_version` 1.1 stays byte-identical for the torus**, indefinitely-for-now.
  The acceptance gate for any change to `src/export/writer.py` is that
  re-exporting an existing torus deliverable produces a **byte-identical** file.
  A diff means the change is wrong; do not rationalise it.
- The 1.1 format itself is specified in
  `../link-list-torus/docs/reference/PARTITION_FILE_FORMAT.md` — **not ours to edit.**

### Which partition to hand over

- **Cross-N studies need the `common-mesh` group** (V = 114,144, N = 50→1000).
  Per-cell quantities are only comparable on one mesh.
- **`finer-mesh` gives better partitions** but each sits on a different mesh, so
  those are for single-N work, not comparison.
- **`mc-study`** is the requested A/B/C set: group A varies surface size at fixed
  shape, group B varies shape at constant area, C1 brackets the perimeter minimum
  from below. Each config carries its design in its own header under `parameters/`.

The cost of choosing the common mesh is measured and small — see the header of
`deliverables.yaml`.

---

## `link-list-general-surface` — hand-off open

**Repo:** https://github.com/TebanVe/link-list-general-surface
**Reader:** **not built.** No `schema_version` 2.0 partition reader exists yet.

An **independent repository**, not a worktree of `link-list-torus`: it was cloned
from that lineage (they share root commit `ad19abd`) and has since diverged.

**Consumes:** group `implicit-surface` — the double-torus and Banchoff-Chmutov
partitions, the only two 2.0 files in existence. They are the fixtures its reader
will be developed against.

**Status:** the export side is done (`0035dde`); the hand-off is not. Both
questions that were open when the general-surface work started are now answered
and travel with the spec — the residual tolerance in
[`PARTITION_EXPORT_SCHEMA_GENERAL.md`](PARTITION_EXPORT_SCHEMA_GENERAL.md) §5.1
and mesh quality in §6.3 — so the hand-off delivers answers, not questions.

⚠ **It is not waiting on us.** That repository is *project 3 of three*, split
from `link-list-torus` on 2026-09-21, and its active thread is the self-approach
study (`docs/experiments/09-self-approach/`), not partition reading. Its
`partition/reader.py` is still **byte-identical** to `link-list-torus`'s — the
torus-only 1.1 reader, inherited at the split. A 2.0 reader is planned, not
imminent.

**Transferring the fixtures is a physical copy.** Both 2.0 files live under
`results/`, which is gitignored here, and the consumer gitignores
`data/partitions/*.h5`. Cloning gets neither. They are 1.4 MB and 3.0 MB.

---

## Open: an upstream request we have not answered

`link-list-general-surface/docs/upstream/surface-partition-collapsed-faces.md`
is a drafted-but-never-filed issue against this repository: the Rep-3 subdivided
mesh contains **collapsed (exactly zero-area) faces and coincident vertices**,
which break point-in-triangle classification. Its evidence is an N=10
`finalised=False` checkpoint from 2026-05-27 (`643eb3f`).

Both of its questions can be answered from work done since:

- *"Are finalised exports guaranteed free of collapsed faces?"* — **No.** The
  shipped n=25 / n=50 / n=200 finalised torus partitions read `min_rel_area`
  exactly 0.0 and `min_interior_angle` 0.0 deg, with 4 of 248,826 and 2 of
  269,832 sub-triangles exactly degenerate (§6.3 of the schema spec). The defect
  is not confined to intermediate checkpoints.
- *"Is `pending_migration` expected to remove these?"* — **No.** That flag marks
  the Phase 2 migration-cycling plateau and has nothing to do with subdivision
  degeneracy. The draft's own guess at the origin is right: a variable point
  landing on a mesh vertex yields a zero-area sub-triangle, so this is a property
  of Rep-3 subdivision.

The consumer has already hardened against it (it excludes zero-area and
non-finite-barycentric faces from its triangle lookup), so nothing is broken.
What is outstanding is a **decision**, not a fix: drop/merge degenerate faces in
the exporter, record a degenerate-face mask in the file, or stand on documenting
it. Changing the exporter is gated on the byte-identical re-export rule above.

---

## Not consumers

`manifold-linked-list` and `geodesic-distance` sit in the surrounding ecosystem
and read no partition file. `link-list-torus-mesh` and `link-list-torus-research`
are **worktrees** of `link-list-torus` (branches `dev/mesh-framework` and
`research/surface-native-partition`), not separate consumers.

## Related documents

- [`deliverables.yaml`](deliverables.yaml) — the per-file record, with the
  `consumers` and `group_consumers` blocks this document explains.
- [`PARTITION_EXPORT_SCHEMA_GENERAL.md`](PARTITION_EXPORT_SCHEMA_GENERAL.md) —
  the 1.1 / 2.0 fork, and the decision to converge on one schema eventually.
- [`MD_SIMULATION_EXPORT_NOTES.md`](MD_SIMULATION_EXPORT_NOTES.md) — why the
  exported representation is what it is.
- `.claude/rules/run-layouts.md` — the export command and its flags.
