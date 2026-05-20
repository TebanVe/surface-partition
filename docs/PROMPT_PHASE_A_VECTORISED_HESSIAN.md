# Prompt — Phase A: Vectorise Hessian Accumulation

> **Use this file as a single self-contained briefing for a fresh agent.**
> Copy everything from "BEGIN PROMPT" to "END PROMPT" below into the new
> agent's first message. The agent does not need to read anything else
> first; the prompt cites the exact file paths, line ranges, and plan
> sections it needs.

---

## BEGIN PROMPT

You are a coding agent in the **surface-partition** repository. Read
`CLAUDE.md` first if you have not already — it has the project context
and conventions.

### Your task

Implement **Phase A** ("Vectorised Hessian accumulation") from
`docs/EXACT_HESSIAN_VALIDATION_AND_PERF_PLAN.md`. Section 3 of that
document is your specification. Read it before touching any code.

The current code computes the analytical Hessian arithmetic (`H_aa`,
`H_bb`, `H_ab`, etc.) in vectorised NumPy, but then accumulates the
results into a flat sparse `values` array using a **Python `for` loop
with three dictionary lookups per row**. This loop runs every time
IPOPT calls the Hessian callback (once per outer iteration), and is the
single most obvious Python-level bottleneck in the exact-Hessian path.

You will replace these per-row Python loops with vectorised
`np.add.at(values, offsets, contributions)` calls. The required
"offsets" arrays are pre-computed once during partition compilation and
stored as new fields on `PartitionArrays`.

### Files you will edit (exactly three)

1. **`src/partition/partition_arrays.py`** — add nine new
   `Optional[np.ndarray]` fields with `= None` defaults (see plan §3.1
   for the exact list and dtypes).
2. **`src/partition/contour_partition.py`** — populate those fields in
   `compile_arrays()` and pass them to the `PartitionArrays(...)`
   constructor (see plan §3.2 for a complete code block; the insertion
   point is between lines 1078 and 1081).
3. **`src/partition/vectorized_perimeter.py`** and
   **`src/partition/vectorized_area.py`** — replace three Python loops
   with `np.add.at` calls (see plan §3.3 and §3.4 for exact replacement
   snippets).

The exact code blocks to insert are written out in the plan. You should
not need to derive any new logic — this is a mechanical refactor.

### Three Python loops you are removing

Each shows the canonical "vectorised NumPy → Python scatter loop"
pattern. After Phase A, only the vectorised arithmetic above each loop
remains; the loops themselves are replaced.

1. `src/partition/vectorized_perimeter.py:105-115` —
   `compute_perimeter_hessian_sparse()`. Three dict lookups + three `+=`
   per perimeter segment.
2. `src/partition/vectorized_area.py:440-451` —
   `compute_area_hessian_sparse()`, `m1` (1-inside) block. Same pattern,
   plus a multiplication by `multipliers[c]` per row.
3. `src/partition/vectorized_area.py:532-543` —
   `compute_area_hessian_sparse()`, `m2` (2-inside) block. Same as #2.

### Files you must NOT touch

- `src/partition/vectorized_steiner.py` — the FD Steiner Hessian loops
  there are out of scope for Phase A (plan §3.5). Leaving them alone is
  required.
- `src/optimization/perimeter_optimizer.py` — the IPOPT adapter is
  read-only for this phase (plan §2.4).
- Any IPOPT options, the L-BFGS / exact-Hessian toggle, or any
  derivative formula. Phase A is purely an internal refactor.

### Correctness requirement

The replacement must produce **bit-for-bit-equivalent or
floating-tolerance-equivalent (`atol=1e-12`)** Hessian values compared
to the current implementation, on the same problem with the same input
$\lambda$. This is non-negotiable: a regression in numerical agreement
indicates a bug in the offset arrays, not a tolerance issue.

To verify:

1. Pick any existing Phase-2 checkpoint or base solution. The torus
   problem is fastest to iterate on:
   `parameters/torus_10part.yaml`. If you do not have a base solution,
   run Phase 1 first:
   `python scripts/find_surface_partition.py --config parameters/torus_10part.yaml`.
2. Capture the Hessian values from the **current** code on a single
   IPOPT iteration. The cleanest way is to add a temporary `np.save`
   inside `IPOPTProblemAdapter._hessian_impl` (in
   `src/optimization/perimeter_optimizer.py`) for the very first call,
   then run
   `python scripts/refine_perimeter.py --solution <base.h5> --config <yaml> --method ipopt --exact-hessian --max-iterations 1`.
   Record the saved `H_old.npy`.
3. Apply your Phase A changes.
4. Re-run the same command and capture `H_new.npy` with the same
   technique.
5. Assert `np.allclose(H_old, H_new, atol=1e-12, rtol=0)`. Print also
   `np.max(np.abs(H_old - H_new))` for the record.
6. Remove the temporary `np.save` after both captures are done.

This procedure is essentially the bench described in plan §3.6, item 1.
Use it as your acceptance test.

### Stronger validation — write a CLI test

In addition to the one-off comparison above, write
`testing/test_phase_a_vectorised_hessian.py`. The script must:

- Take `--solution <path.h5>` (and optionally `--config <yaml>` if
  needed for `build_optimizer`).
- Build a compiled `PartitionArrays` (see the `build_optimizer` helper
  in plan §4 around line 415-460 for the canonical setup).
- Call `compute_perimeter_hessian_sparse(pa)` once, record the result
  as `H_perim_new`.
- Re-run the same call after **monkey-patching the new offsets to
  `None`**: this forces the fallback Python-loop path included in plan
  §3.3 (the `else:` branch). Record as `H_perim_old`.
- Likewise for `compute_area_hessian_sparse(pa, multipliers)` with both
  `multipliers = np.zeros(pa.n_cells - 1)` and a deterministic random
  vector (`np.random.default_rng(42).standard_normal(pa.n_cells - 1)`).
- Assert `np.allclose(H_new, H_old, atol=1e-12, rtol=0)` on every pair.
- Also assert `np.max(np.abs(H_new - H_old)) < 1e-12`.
- Print a single PASS / FAIL line, exit code 0 / 1.

This script is the regression test that locks in Phase A. Future agents
working on Phase B (`testing/test_exact_hessian_vs_fd.py`,
`testing/compare_hessian_modes.py`) and on the analytical Steiner plan
will rely on it. Add a one-paragraph entry for it in
`testing/README_testing.md`.

The fallback path is the **same code that exists today** — that is the
reason the plan deliberately keeps the `else:` branches in §3.3 and
§3.4. Do not remove those fallbacks; they exist precisely so this test
can compare new vs old in a single process without git stashing.

### Stretch correctness check (optional but cheap)

The new offset arrays must satisfy
`0 <= offset < len(pa.hess_row)` for every entry. Add a single assertion
(`assert (offsets >= 0).all() and (offsets < len(hess_row)).all()`)
inside `compile_arrays()` during development — see plan §3.2 last
paragraph. Remove the assertion before committing if it is in a hot
path; if it is at compile time only, leave it in.

### Acceptance criteria (verbatim from plan §6, with the test script
added)

- [ ] All nine new fields on `PartitionArrays` exist, have `Optional`
      type annotations, `= None` defaults, and are placed after
      `hess_offset_map` (otherwise the dataclass default-after-non-default
      rule will reject the file).
- [ ] `compile_arrays()` populates every new field and includes each
      one in the `return PartitionArrays(...)` call.
- [ ] `compute_perimeter_hessian_sparse()` uses `np.add.at` on the
      pre-computed offsets; the `else:` fallback path is preserved
      verbatim.
- [ ] Both blocks (`m1`, `m2`) of `compute_area_hessian_sparse()` use
      `np.add.at` with the pre-computed offsets and the
      `btri{1,2}_cell_active` mask applied; both `else:` fallbacks are
      preserved verbatim.
- [ ] On at least one reference checkpoint, the Hessian values returned
      by `_hessian_impl` before and after the refactor agree within
      `atol=1e-12, rtol=0`.
- [ ] `testing/test_phase_a_vectorised_hessian.py` exists, runs
      end-to-end, and prints PASS with `max |H_new − H_old| < 1e-12`.
- [ ] `scripts/refine_perimeter.py --method ipopt --exact-hessian`
      completes end-to-end without errors on at least one Phase-1
      solution.

### Reference reading (in order)

1. **`docs/EXACT_HESSIAN_VALIDATION_AND_PERF_PLAN.md` §1, §2, §3, §6** —
   your specification. §3 is the implementation; everything else is
   context.
2. **`docs/OPTIMIZATION_METHODS_PRIMER.md` §3-§4** — explains *why* this
   refactor matters and where it sits in the larger story. Read this if
   "exact Hessian", "Lagrangian", or "KKT system" need clarification.
3. **`CLAUDE.md`** — project conventions: snake_case, no narrative
   comments, no emojis, `get_logger(__name__)` for logging.
4. **`docs/SCALABILITY_ANALYSIS.md` §1-§2** — empirical timings showing
   why this loop matters at scale. Read only if you want context for
   why the project cares about this work.

### What you should NOT do

- Do not change the derivative formulas (`H_aa`, `H_bb`, etc.). Only
  the accumulation pattern.
- Do not modify `compute_steiner_*_hessian_fd` (the FD Steiner code).
  Plan §3.5 is explicit on this.
- Do not change the IPOPT adapter, the L-BFGS toggle, or any IPOPT
  option.
- Do not delete the `else:` fallback branches in
  `compute_perimeter_hessian_sparse` or `compute_area_hessian_sparse`.
  They are required for the test in `testing/test_phase_a_vectorised_hessian.py`
  to compare the two paths in a single process.
- Do not add pytest fixtures or `pytest.ini`. This project does not use
  pytest (see `CLAUDE.md`, "Gotchas and Known Issues"). The validation
  script is a plain CLI script, exit code 0 / 1, that the user runs
  manually.
- Do not commit or push without the user explicitly asking. Phase A is
  one logical change set; if the user wants it committed, they will say
  so.

### Order of operations

1. Read plan §3 fully.
2. Edit `partition_arrays.py` (add the nine fields).
3. Edit `contour_partition.py` (populate them in `compile_arrays`, pass
   them to the `return` call). Verify with the assertion from §3.2.
4. Edit `vectorized_perimeter.py` (replace one loop, keep fallback).
5. Edit `vectorized_area.py` (replace two loops, keep fallbacks).
6. Run `python scripts/refine_perimeter.py --solution <base.h5> --config
   <yaml> --method ipopt --exact-hessian --max-iterations 1` to confirm
   the pipeline still runs end-to-end.
7. Write `testing/test_phase_a_vectorised_hessian.py` (use the
   `build_optimizer` helper from plan §4). Run it, see PASS.
8. Run the temporary `np.save` capture procedure described above to
   double-check vs the pre-Phase-A code via git stash. Once both
   confirm agreement to `1e-12`, remove the temporary save.
9. Add the new test to `testing/README_testing.md` (one paragraph, same
   style as the existing entries — do not invent a new style).
10. Update plan frontmatter — flip these todos in
    `docs/EXACT_HESSIAN_VALIDATION_AND_PERF_PLAN.md`:
    - `precomp-seg-offsets`, `precomp-btri-offsets`,
      `vectorize-perim-hess`, `vectorize-area-hess` →
      `status: completed`.
    Leave the Phase B and Phase C todos untouched — those are the next
    agent's work.
11. Report back to the user with: the diff summary, the test output, and
    the measured `max |H_new − H_old|`.

### When to stop and ask

- If the offset assertion in §3.2 fails, stop. The
  `hess_offset_map` build is wrong, and that is out of scope here.
- If `np.allclose(H_old, H_new, atol=1e-12)` fails on any tested
  Hessian, stop and report the discrepancy. Do not try to "fix" by
  loosening the tolerance — a >1e-12 disagreement means an indexing
  bug, which the user must see.
- If the existing fallback Python-loop path produces an unexpected
  result on its own (i.e., the test fails on the `else:` branch alone),
  stop — there is a pre-existing bug, not your problem.

### Expected size of the change

- `partition_arrays.py`: ~12 lines added.
- `contour_partition.py`: ~50 lines added.
- `vectorized_perimeter.py`: ~10 lines changed (loop body replaced with
  3 `np.add.at` calls; fallback preserved as `else`).
- `vectorized_area.py`: ~30 lines changed (two blocks).
- `testing/test_phase_a_vectorised_hessian.py`: ~80 lines new.
- `testing/README_testing.md`: ~5 lines added.
- `docs/EXACT_HESSIAN_VALIDATION_AND_PERF_PLAN.md`: 4 frontmatter
  status flips.

Total: under 200 lines of new / changed code, plus one new test file.

### Performance expectations

You are not required to benchmark, but as a sanity check: the
`np.add.at` calls should be roughly 10–50× faster than the Python loop
on a 5 000-segment problem. If you observe a *slowdown*, something is
miswritten — the most likely culprit is duplicating work outside the
loop instead of replacing it. The `compare_hessian_modes.py` script
(plan §4.4, written by the next agent) will provide formal performance
measurements with per-component breakdown.

### Final note

This task is ~1 day of work for an experienced Python engineer, ~2 days
otherwise. The plan is thorough; if any part feels under-specified,
re-read plan §3 — the snippet you need is almost certainly there
already.

## END PROMPT
