#!/usr/bin/env python3
"""Apply the PRE-REGISTERED Phase A scorecard to one or more arm reports.

    python scripts/score_mbo_arm.py results/arm_mbo_*/arm_report.yaml \
        --init-report results/arm_init_*/arm_report.yaml

Every threshold below is a literal copied from the pre-registration in
``docs/plans/PHASE1_BC_REPLACEMENT_PLAN.md`` (commits 9ffdefb, e1e6288, adc42b7),
committed before any scored run existed. **Nothing here is computed by judgement
after the fact** -- that is the entire point of scoring with a script rather than
by reading a log. If a verdict looks wrong, the fix is to argue with the
pre-registration, not to edit the constants.
"""

import argparse
import glob
import sys

import yaml

# --- literals from the pre-registration -----------------------------------
N100_ANCHOR = 185.2546144718457  # CENSORED: best iterate is its last => a bound
N300_ANCHOR = 323.319247087254  # V=47,488, plateaued => the safe anchor
SUCCESS_REL, REFUTED_REL = 0.01, 0.05
# PGD Phase 1 wall per configuration, from timing_profile total_wall_s (never
# metadata run_time_seconds). Keyed by N: using the N=100 figure for an N=300 arm
# understates the speedup by 1.6x, which is how the first N=300 scorecard printed
# 194x for a run that is actually 319x.
CONTROL_WALL_S = {100: 48132.0, 300: 79068.6}
NC3_K, NC3_F_MIN = 79.2482, 5.634e-3
EXCURSION_FLOOR = 2e-3  # or 1.5 x the run's own slack, whichever larger
P13_MIN_REL = 0.003


def load(pattern):
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(pattern)
    return [(h, yaml.safe_load(open(h))) for h in hits]


def perimeter_verdict(p, anchor):
    rel = (p - anchor) / anchor
    if rel <= SUCCESS_REL:
        return rel, "SUCCESS"
    if rel <= REFUTED_REL:
        return rel, "PARTIAL"
    return rel, "REFUTED ON PERIMETER"


def _speed(n_partitions, wall):
    cw = CONTROL_WALL_S.get(n_partitions)
    return f"{cw/max(wall,1e-9):.1f}x vs {cw:.0f}s PGD" if cw else "n/a (no PGD wall)"


def score(path, r, init_p2=None, init_n=None):
    N, V = r["n_partitions"], r["arm_final_V"]
    anchor = N100_ANCHOR if N == 100 else N300_ANCHOR
    print(
        f"\n{'='*78}\n{path}\n  N={N} V={V} seed={r['seed']} anchor={r['anchor_run']}\n{'='*78}"
    )

    g = r["gates_raw"]
    a, c = g["area_imbalance"], g["connectivity"]
    print(
        f"P3  area worst dev : {a['worst_rel_dev']*100:.4f}%  bar {a['harness_fault_threshold']*100:.4f}%"
        f"   {'HARNESS FAULT (fix 0a; NOT evidence against B)' if a['suspect_solver_failure'] else 'ok'}"
    )
    print(
        f"P4  fragmented     : {c['n_fragmented']}  {c['fragmented'][:10]}"
        f"   {'-> lane: compose with E' if c['n_fragmented'] else '-> lane: CLEAN'}"
    )
    print(
        f"P5  ladder wall    : {r['wall_seconds']:.0f}s = {r['wall_seconds']/60:.1f} min"
        f"   speedup {_speed(N, r['wall_seconds'])}"
    )

    # P10 -- descent inequality and E_tau excursions, ACTIVE steps only.
    worst_viol, worst_slack, incs, active = -1e30, 0.0, 0, 0
    for lv in r["levels"]:
        steps = lv["steps"]
        act = [s for s in steps if s["churn"] > 0]
        active += len(act)
        for s in act:
            worst_viol = max(worst_viol, s["violation_rel"])
            worst_slack = max(worst_slack, s["slack_rel"])
        e = [s["E_lumped"] for s in steps]
        incs += sum(1 for i in range(1, len(e)) if e[i] > e[i - 1])
    ceiling = max(EXCURSION_FLOOR, 1.5 * worst_slack)
    print(
        f"P10 inequality     : worst violation {worst_viol:.3e} (<=1e-12)  "
        f"max slack {worst_slack:.3e}  ceiling {ceiling:.3e}"
    )
    print(
        f"    E_tau          : {incs} increases over {active} active steps "
        f"({incs/max(active,1)*100:.1f}%, <25%)"
    )

    # P8 -- both detector clauses, per level, reported separately.
    print(
        "P8  pinning        : level | c/c_lo | dE/tail clause | moved-frac clause | AND"
    )
    dE_levels, and_levels = [], []
    for lv in r["levels"]:
        pr = lv.get("probe") or {}
        probes = pr.get("probes", [])
        if not probes:
            continue
        best_dE = max(p["dE_over_tail"] for p in probes)
        best_mf = max(p["moved_frac"] for p in probes)
        dE_hit, mf_hit = best_dE > NC3_K, best_mf > NC3_F_MIN
        if dE_hit:
            dE_levels.append(lv["level"])
        if dE_hit and mf_hit:
            and_levels.append(lv["level"])
        ratio = 4.0 / lv["tau"]["c_lo_hmean"]
        print(
            f"      L{lv['level']} | {ratio:6.3f} | {best_dE:9.2f} {'HIT ' if dE_hit else '    '}"
            f"| {best_mf:.3e} {'HIT ' if mf_hit else '    '}| "
            f"{'PINNED' if pr.get('pinned') else '-'}"
        )
    agree = set(dE_levels) == set(and_levels)
    print(f"    dE clause flags {dE_levels}; AND flags {and_levels}")
    if not agree:
        print(
            "    => CLAUSES DISAGREE: P8 UNTESTABLE AS WRITTEN (detector not "
            "mesh-transferable). Both lists reported above."
        )
    elif len(and_levels) > 2:
        print(
            f"    => CLAUSES AGREE and {len(and_levels)} > 2 levels flagged: **P8 FAILED** "
            "(recorded as a miss, per the committed scoring rule)"
        )
    else:
        print(f"    => CLAUSES AGREE and {len(and_levels)} <= 2 flagged: P8 HELD")

    p2 = r.get("phase2")
    if r.get("phase2_aborted"):
        print(
            "P12 Phase 2       : **ABORTED -> REFUTATION ON GEOMETRY**, not scored "
            "against +1%/+5%"
        )
        return
    if not p2:
        print("P12 Phase 2       : not run")
        return
    if p2.get("short_campaign"):
        print(
            f"P12 Phase 2       : **SHORT CAMPAIGN {p2['n_iterations']} of "
            f"{p2.get('expected_iterations')} -> SOFT ABORT, refutation on geometry, "
            "NOT scored against +1%/+5%**"
        )
    else:
        print(
            f"P12 Phase 2       : {p2['n_iterations']} of "
            f"{p2.get('expected_iterations', '?')} iterates, no abort"
        )
    rel, v = perimeter_verdict(p2["best_perimeter"], anchor)
    print(
        f"P1  perimeter     : {p2['best_perimeter']:.10f} vs anchor {anchor:.10f}"
        f"  = {rel*100:+.3f}%   -> {v}"
    )
    if p2["censored"]:
        print(
            "    ** CENSORED (best iterate is the last) -> NOT scored against a threshold **"
        )
    # P13 only means anything when the init control is the SAME configuration.
    # Applying an N=100 init to an N=300 report printed "-68.704% ... B's descent
    # claim is unearned" -- a nonsense verdict from a cross-N comparison, and the
    # same unit-bug class as the 194x speedup.
    if init_p2 is not None and init_n is not None and init_n != N:
        print(
            f"P13 ATTRIBUTION   : SKIPPED -- init control is N={init_n}, this report "
            f"is N={N}. Cross-N attribution is meaningless."
        )
    elif init_p2 is not None:
        rel13 = (init_p2 - p2["best_perimeter"]) / init_p2
        print(
            f"P13 ATTRIBUTION   : init {init_p2:.6f} -> MBO {p2['best_perimeter']:.6f} "
            f"= {rel13*100:+.3f}%  (need >= {P13_MIN_REL*100:.1f}%)"
        )
        if rel13 < P13_MIN_REL:
            print(
                "    ** MBO ADDS < 0.3% OVER ITS OWN INIT: the N=100 number is a PREVIEW OF C, "
                "not a result about B. B's descent claim is unearned. **"
            )
        else:
            print("    MBO earns its place over the balanced geodesic init.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+")
    ap.add_argument(
        "--init-report",
        default=None,
        help="SF-a init-only arm_report.yaml, for the P13 attribution test",
    )
    args = ap.parse_args()

    init_p2 = init_n = None
    if args.init_report:
        for _, ir in load(args.init_report):
            if ir.get("phase2"):
                init_p2 = ir["phase2"]["best_perimeter"]
                init_n = ir["n_partitions"]

    loaded = []
    for pat in args.reports:
        loaded.extend(load(pat))
    for path, r in loaded:
        score(path, r, init_p2, init_n)

    p2s = [
        (r["seed"], r["phase2"]["best_perimeter"])
        for _, r in loaded
        if r.get("phase2") and r["n_partitions"] == 100
    ]
    if len(p2s) >= 2:
        lo, hi = min(v for _, v in p2s), max(v for _, v in p2s)
        spread = (hi - lo) / lo
        print(
            f"\nP2  SEED SPREAD (N=100): {spread*100:.3f}% over {len(p2s)} seeds "
            f"{[(s, round(v,4)) for s, v in p2s]}"
        )
        if spread > 0.01:
            print(
                "    ** SPREAD > 1%: the primary falsifier CANNOT be scored at +/-1%. "
                "Report that instead of a verdict. **"
            )
        elif spread >= 0.005:
            print("    ** SPREAD >= 0.5%: P2 MISSED (recorded as a miss). **")
        else:
            print("    P2 held (< 0.5%).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
