#!/usr/bin/env python3
"""Extract report 08's figure data from the scored runs into a COMMITTED data.yaml.

    python docs/experiments/08-mbo-auction-dynamics/extract_data.py

``results/`` is gitignored, so a figure script that reads it directly cannot be
re-run by anyone who does not already hold the 12.6 GB of run output. This script
distils the ~3 MB of arm reports into a small committed file, and ``make_figures.py``
reads only that. The provenance chain is therefore:

    scored run  ->  results/<run>/arm_report.yaml  ->  data.yaml (committed)
                ->  make_figures.py  ->  fig_*.pdf (committed)  ->  main.tex

Re-running this script against the same runs must reproduce data.yaml byte for
byte; re-running it after new runs is how the report is updated.
"""

import glob
import os
import sys

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.yaml")

RUNS = {
    "n100_s84172851": "results/arm_mbo_20260819_173411_npart100_V114144_seed84172851",
    "n100_s61803399": "results/arm_mbo_20260819_185131_npart100_V114144_seed61803399",
    "n300_s61803399": "results/arm_mbo_20260819_192659_npart300_V47488_seed61803399",
    "init_n100": "results/arm_init_20260819_181203_npart100_V114144_seed84172851",
}

# Anchors, from the runs named in the report's provenance block.
ANCHORS = {
    100: {
        "perimeter": 185.2546144718457,
        "run": "run_20260709_081548",
        "pipeline": "raw PGD (no readout)",
        "best_iteration": 20,
        "n_iterations": 20,
        "pgd_wall_s": 48132.12,
        "label_boundary_length": 107.34097586,  # recomputed in-session from x_opt
    },
    300: {
        "perimeter": 323.319247087254,
        "run": "run_20260806_123326",
        "pipeline": "PGD + balanced readout (A+E)",
        "best_iteration": 18,
        "n_iterations": 19,
        "pgd_wall_s": 79068.56,
        "label_boundary_length": 193.3024,  # repaired labels the anchor consumed
    },
}


# ---------------------------------------------------------------------------
# Measurements that do not live in an arm_report.yaml. Transcribed here with the
# command that reproduces each, and an explicit self_reproduced flag: this
# programme's standing rule is that a number produced by a review subagent is
# never restated in the first person until the author has reproduced it.
# ---------------------------------------------------------------------------
MEASUREMENTS = {
    "gates": {
        "provenance": "python testing/test_mbo_auction.py   (G1-G8, ~257 s)",
        "self_reproduced": True,
        "G4": {
            "n100_V9600_full_budget": dict(
                active=40,
                worst_violation=-4.314e-07,
                max_slack=1.665e-04,
                e_increases=0,
            ),
            "n300_V47488_early_stop": dict(
                active=40,
                worst_violation=-2.914e-08,
                max_slack=2.075e-04,
                e_increases=1,
                worst_increase=2.542e-05,
            ),
        },
        "G7_tau_mutation_rejected_at": 1.626e-01,
        "G1_mass_err": 3.58e-15,
    },
    "negative_controls": {
        "provenance": "python testing/test_mbo_auction.py --negative-controls  (~800 s)"
        " and --only NC5b,NC2 --nc2-k 79.2482 --nc2-f-min 5.634e-3",
        "self_reproduced": True,
        "NC1_cold_init": dict(
            cold_voronoi=0.4555,
            cold_after_diffusion=0.5067,
            cold_after_balancing=0.506742,
            warm=0.016156,
            bar=0.020216,
        ),
        "NC4_wrong_operator": dict(wrong=4.060e2, correct=3.577e-15),
        "NC3_CAL": dict(
            frozen_dE_over_tail=[154.15, 260.72],
            frozen_moved_frac=[3.75e-3, 7.81e-3],
            converged_dE_over_tail=[20.99, 24.09],
            converged_moved_frac=[3.23e-3, 4.06e-3],
            K=79.2482,
            f_min=5.634e-3,
            margin=10.82,
        ),
        "NC2_freeze_V114144_c2": dict(
            pinned=True, moved_frac=[4.60e-3, 9.76e-3], dE_over_tail=[42.99, 89.82]
        ),
        # NC5 / NC5b: the over-merge instrument. c is swept with rho lifted so the
        # cap does not clamp c=8 back inside the window.
        "NC5_V47488_N300": [
            dict(
                c=2,
                sqrt_tau_over_r_cell=0.335,
                fragmented=0,
                q_mean=1.6149,
                q_worst=2.036,
                core_loss=0,
                boundary=189.71,
            ),
            dict(
                c=4,
                sqrt_tau_over_r_cell=0.671,
                fragmented=0,
                q_mean=1.5649,
                q_worst=1.928,
                core_loss=0,
                boundary=186.79,
            ),
            dict(
                c=8,
                sqrt_tau_over_r_cell=1.341,
                fragmented=0,
                q_mean=1.5543,
                q_worst=1.874,
                core_loss=0,
                boundary=186.18,
            ),
        ],
        "NC5b_V9600_N300": [
            dict(
                c=2,
                sqrt_tau_over_r_cell=0.747,
                fragmented=0,
                q_mean=1.6269,
                q_worst=2.381,
                core_loss=0,
                boundary=190.26,
            ),
            dict(
                c=4,
                sqrt_tau_over_r_cell=1.495,
                fragmented=0,
                q_mean=1.5949,
                q_worst=2.194,
                core_loss=0,
                boundary=188.37,
            ),
            dict(
                c=8,
                sqrt_tau_over_r_cell=2.990,
                fragmented=0,
                q_mean=1.5770,
                q_worst=2.174,
                core_loss=0,
                boundary=187.33,
            ),
        ],
    },
    "per_level_connectivity": {
        "provenance": "python scripts/run_mbo_arm.py --anchor-run "
        "'results/run_20260806_123326*' --out-root <scratch>  (~255 s), "
        "after per-level gates were added to mbo_level",
        "self_reproduced": True,
        "n300_fragmented_by_level": [1, 0, 0],
        "n300_L0_fragmented_cells": [295],
        "n100_fragmented_by_level": [0, 0, 0, 0, 0],
        "note": "Bit-exact ladder reproduction: worst areas 6.7063/2.5394/1.4172%, "
        "steps 16/21/66, boundary 187.2574.",
    },
    "review6": {
        "provenance": "sixth adversarial review (Fable subagent), 2026-08-19",
        "self_reproduced": False,
        "control_phase2_extension": dict(
            iterations="21-40 resumed from the control's own iteration_020",
            best_over_40=185.2546144718457,
            best_iteration=20,
            extension_min=185.2577,
            extension_max=185.3038,
            final=185.2896,
        ),
        "n300_seed_stress_ladder_only": [
            dict(
                seed=27182818, fragmented=0, worst_rel_dev=0.012531, boundary=186.0847
            ),
            dict(
                seed=13131313, fragmented=0, worst_rel_dev=0.014467, boundary=186.5474
            ),
            dict(
                seed=84172851, fragmented=0, worst_rel_dev=0.013887, boundary=186.9009
            ),
        ],
        "readout_boundary_decomposition": dict(
            raw_pgd_argmax=189.7310,
            after_dual_shifts=190.7114,
            after_repair=193.3024,
            repair_damage_rel=0.01882,
        ),
        "raw_pgd_n300_gates": dict(
            n_imbalanced=10, worst_rel_dev=0.3615, n_fragmented=2
        ),
    },
    "predictions": {
        "provenance": "pre-registered in commit 9ffdefb (before any scored run); "
        "scored by scripts/score_mbo_arm.py",
        "self_reproduced": True,
        "rows": [
            dict(
                id="P1",
                predicted="188.0 (+1.5%), PARTIAL",
                actual="184.4118 / 184.1615, SUCCESS",
                verdict="MISS",
            ),
            dict(id="P2", predicted="spread < 0.5%", actual="0.136%", verdict="HELD"),
            dict(
                id="P3",
                predicted="<= 0.28%, point 0.15%",
                actual="0.1504 / 0.1493%",
                verdict="HELD",
            ),
            dict(id="P4", predicted="<= 3, point 0", actual="0", verdict="HELD"),
            dict(
                id="P5",
                predicted=">= 30x, point 53x",
                actual="210.9 / 225.2x",
                verdict="HELD (point missed)",
            ),
            dict(
                id="P6",
                predicted="331.4, 80% CI [323.3, 341.0]",
                actual="319.9428",
                verdict="MISS",
            ),
            dict(
                id="P7",
                predicted="<= 2.02%, <= 6 frag, >= 88x",
                actual="1.4172%, 0, 318.8x",
                verdict="HELD",
            ),
            dict(
                id="P8",
                predicted="<= 2 of 5 flagged",
                actual="0, clauses agree",
                verdict="HELD",
            ),
            dict(
                id="P8'",
                predicted="dE flags L2,L3,L4 (falsifying P8)",
                actual="dE flat 14-22, flags none",
                verdict="MISS",
            ),
            dict(
                id="P9",
                predicted="frag not death; worst at L0",
                actual="1 frag at L0 (295), 0 elsewhere",
                verdict="HIT",
            ),
            dict(
                id="P10",
                predicted="viol <= 1e-12; excursions <= ceiling",
                actual="-6.13e-08; 4/96 within ceiling",
                verdict="HELD",
            ),
            dict(id="P12", predicted="no abort", actual="20/20, 19/19", verdict="HELD"),
            dict(id="P13", predicted=">= 0.3%", actual="+2.761%", verdict="HELD"),
        ],
    },
}


def summarize(rep):
    lv = []
    for L in rep["levels"]:
        steps = L["steps"]
        active = [s for s in steps if s["churn"] > 0]
        e = [s["E_lumped"] for s in steps]
        lv.append(
            {
                "level": L["level"],
                "V": L["n_vertices"],
                "verts_per_cell": L["n_vertices"] / L["n_cells"],
                "n_steps": L["n_steps"],
                "active_steps": L["active_steps"],
                "worst_rel_dev": L["final_worst_rel_dev"],
                "quality_bar": L["tau"].get("quality_bar"),
                "q_mean": L["geometry"]["q_mean"],
                "q_worst": L["geometry"]["q_worst"],
                "core_loss": L["geometry"]["core_loss"],
                "label_boundary_length": L["label_boundary_length"],
                "wall_s": L["wall_seconds"],
                "sqrt_tau_over_h_max": L["tau"]["sqrt_tau_over_h_max"],
                "sqrt_tau_over_r_cell": L["tau"]["sqrt_tau_over_r_cell"],
                "c_eff": L["tau"]["c_eff"],
                "c_lo_hmean": L["tau"]["c_lo_hmean"],
                "c_hi_hmean": L["tau"]["c_hi_hmean"],
                "cap_active": L["tau"]["cap_active"],
                "n_fragmented_level": (L.get("gates") or {}).get("n_fragmented"),
                "churn_frac": [s["churn_frac"] for s in steps],
                "E_lumped": e,
                "worst_violation_rel": max(
                    (s["violation_rel"] for s in active), default=None
                ),
                "max_slack_rel": max((s["slack_rel"] for s in active), default=0.0),
                "n_E_increases": sum(1 for i in range(1, len(e)) if e[i] > e[i - 1]),
                "probe_best_dE_over_tail": max(
                    (
                        p["dE_over_tail"]
                        for p in (L.get("probe") or {}).get("probes", [])
                    ),
                    default=None,
                ),
                "probe_best_moved_frac": max(
                    (p["moved_frac"] for p in (L.get("probe") or {}).get("probes", [])),
                    default=None,
                ),
                "pinned": (L.get("probe") or {}).get("pinned"),
            }
        )
    g = rep["gates_raw"]
    out = {
        "run_dir": os.path.basename(
            os.path.dirname(rep["solution"]).rsplit("/solution", 1)[0]
        ),
        "arm": rep["arm"],
        "N": rep["n_partitions"],
        "V": rep["arm_final_V"],
        "seed": rep["seed"],
        "phase1_wall_s": rep["wall_seconds"],
        "init_wall_s": rep["init"]["wall_seconds"],
        "init_worst_rel_dev": rep["init"]["worst_rel_dev"],
        "label_boundary_length": rep["label_boundary_length"],
        "gates": {
            "n_dead": g["dormant"]["n_dead"],
            "n_weak": g["dormant"]["n_weak"],
            "min_peak_density": g["dormant"]["min_peak_density"],
            "n_imbalanced": g["area_imbalance"]["n_imbalanced"],
            "worst_rel_dev": g["area_imbalance"]["worst_rel_dev"],
            "quality_bar": g["area_imbalance"]["harness_fault_threshold"],
            "granularity": g["area_imbalance"]["vertex_granularity_rel"],
            "n_fragmented": g["connectivity"]["n_fragmented"],
        },
        "levels": lv,
    }
    p2 = rep.get("phase2")
    if p2:
        out["phase2"] = {
            "best_perimeter": p2["best_perimeter"],
            "best_iteration": p2["best_iteration"],
            "n_iterations": p2["n_iterations"],
            "censored": p2["censored"],
            "initial_perimeter": p2["initial_perimeter"],
            "wall_s": p2["wall_seconds"],
            "trajectory": p2["trajectory"],
        }
    return out


def main():
    data = {"anchors": ANCHORS, "runs": {}, "measurements": MEASUREMENTS}
    missing = []
    for key, rel in RUNS.items():
        hits = sorted(glob.glob(os.path.join(ROOT, rel, "arm_report.yaml")))
        if not hits:
            missing.append(rel)
            continue
        data["runs"][key] = summarize(yaml.safe_load(open(hits[0])))
    if missing:
        print("MISSING (results/ is gitignored; these runs are not on this disk):")
        for m in missing:
            print("   ", m)
        if not data["runs"]:
            return 1
    with open(OUT, "w") as f:
        yaml.safe_dump(data, f, sort_keys=True, width=100, default_flow_style=False)
    print(
        f"wrote {OUT}  ({os.path.getsize(OUT)/1024:.1f} kB, {len(data['runs'])} runs)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
