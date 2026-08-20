#!/usr/bin/env python3
"""Run approach B (auction-dynamics MBO) as a Phase 1 replacement arm, and score it.

    # scored arm, N=100, seed matched to the control
    python scripts/run_mbo_arm.py --anchor-run results/run_20260709_081548* --phase2

    # SF-a attribution control: the balanced geodesic init alone, no MBO
    python scripts/run_mbo_arm.py --anchor-run results/run_20260709_081548* \
        --init-only --phase2

    # N=300 secondary; asserts the arm's final mesh matches the anchor's
    python scripts/run_mbo_arm.py --anchor-run results/run_20260806_123326* --phase2

**The ladder is read from the ANCHOR RUN'S ``experiment.yaml``, never from
``parameters/``.** ``parameters/torus_300part_seeded_lam11p5_original_energy.yaml``
says ``refinement_levels: 5`` (V=114,144) -- raised post-hoc for a resume -- while
the anchor ``run_20260806_123326`` was produced with 3 (V=47,488). Reading the live
parameters file would score a V=114,144 arm against a V=47,488 anchor: a mesh
effect in the arm's favour, breaking the harness's mesh-matching rule. The driver
therefore asserts ``arm_final_V == anchor_V`` and refuses to score otherwise.

Everything downstream is the shared harness (``src/partition/arm_harness.py``) and
nothing else: the arm's labels are written in the Phase 1 solution schema, the three
validity gates run on the RAW labels first, and Phase 2 runs as a subprocess against
the anchor campaign's ENTIRE ``refinement.yaml``.
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.logging_config import get_logger, setup_logging  # noqa: E402
from src.partition.arm_harness import (  # noqa: E402
    labels_to_one_hot,
    run_gates,
    run_phase2,
    write_arm_solution,
)
from src.partition.balanced_readout import label_boundary_length  # noqa: E402
from src.partition.mbo_auction import MBOConfig, run_mbo_ladder  # noqa: E402
from src.surfaces.torus import TorusMeshProvider  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
logger = get_logger(__name__)


def resolve_anchor(pattern: str) -> Path:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no run matches {pattern!r}")
    if len(matches) > 1:
        raise ValueError(f"{pattern!r} is ambiguous: {matches}")
    return Path(matches[0])


def anchor_spec(run_dir: Path) -> dict:
    """Mesh ladder, N and seed, taken from the anchor run's OWN experiment.yaml."""
    with open(run_dir / "experiment.yaml") as f:
        cfg = yaml.safe_load(f)
    relax = cfg.get("relaxation", cfg)
    torus = cfg.get("surface", {}).get("torus", {})
    return {
        "n_partitions": int(relax["n_partitions"]),
        "seed": int(relax["seed"]),
        "levels": int(relax["refinement_levels"]),
        "n_theta": int(torus["n_theta"]),
        "n_phi": int(torus["n_phi"]),
        "d_theta": int(torus["n_theta_increment"]),
        "d_phi": int(torus["n_phi_increment"]),
        "R": float(torus.get("R", 1.0)),
        "r": float(torus.get("r", 0.6)),
    }


def anchor_solution_vertices(run_dir: Path) -> int:
    """Vertex count of the solution the anchor's Phase 2 campaign actually consumed."""
    cands = sorted(run_dir.glob("readout/*/solution_balanced.h5")) or sorted(
        run_dir.glob("solution/surface_*.h5")
    )
    if not cands:
        raise FileNotFoundError(f"no anchor solution under {run_dir}")
    with h5py.File(cands[0], "r") as f:
        return int(f["vertices"].shape[0])


def anchor_campaign(run_dir: Path) -> Path:
    cands = sorted(run_dir.glob("readout/*/refinement/*/refinement.yaml")) or sorted(
        run_dir.glob("refinement/*/refinement.yaml")
    )
    if not cands:
        raise FileNotFoundError(f"no campaign refinement.yaml under {run_dir}")
    return cands[0]


def spec_from_config(path: Path) -> dict:
    """Ladder / N / seed from a plain parameters/*.yaml, for a configuration that
    has no baseline. Same fields and key names as ``anchor_spec``.

    Used only in exploratory mode. With no anchor there is no mesh to match and
    no perimeter to score against -- the run answers "does B produce a valid
    partition here, and how fast", not "is B better".
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    relax = cfg.get("relaxation", cfg)
    torus = cfg.get("surface", {}).get("torus", {})
    return {
        "n_partitions": int(relax["n_partitions"]),
        "seed": int(relax["seed"]),
        "levels": int(relax["refinement_levels"]),
        "n_theta": int(torus["n_theta"]),
        "n_phi": int(torus["n_phi"]),
        "d_theta": int(torus["n_theta_increment"]),
        "d_phi": int(torus["n_phi_increment"]),
        "R": float(torus.get("R", 1.0)),
        "r": float(torus.get("r", 0.6)),
    }


def build_ladder(spec: dict):
    meshes = []
    nt, nphi = spec["n_theta"], spec["n_phi"]
    for _ in range(spec["levels"]):
        meshes.append(
            TorusMeshProvider(n_theta=nt, n_phi=nphi, R=spec["R"], r=spec["r"]).build()
        )
        nt, nphi = nt + spec["d_theta"], nphi + spec["d_phi"]
    return meshes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--anchor-run",
        default=None,
        help="glob for the anchor run directory; supplies the ladder, "
        "N, seed, the V assertion and the Phase 2 campaign",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override the arm's init seed (default: the anchor's)",
    )
    ap.add_argument(
        "--init-only",
        action="store_true",
        help="SF-a: balanced geodesic init on the finest mesh, no MBO",
    )
    ap.add_argument("--tau-c", type=float, default=4.0)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--max-iters", type=int, default=200)
    ap.add_argument("--nc3-k", type=float, default=None)
    ap.add_argument("--nc3-f-min", type=float, default=None)
    ap.add_argument("--phase2", action="store_true")
    ap.add_argument("--out-root", type=str, default=str(ROOT / "results"))
    ap.add_argument("--tag", type=str, default="")
    # --- exploratory mode: no baseline exists for this N ------------------
    ap.add_argument(
        "--config",
        default=None,
        help="parameters/*.yaml to run from when NO anchor run exists (e.g. a new "
        "N). Mutually exclusive with --anchor-run. Perimeter is then NOT "
        "scored against anything -- there is no baseline to score against.",
    )
    ap.add_argument(
        "--campaign",
        default=None,
        help="refinement.yaml (or a config with a `refinement:` section) for "
        "Phase 2. Defaults to the anchor's campaign, or to --config itself.",
    )
    ap.add_argument(
        "--levels",
        type=int,
        default=None,
        help="truncate the mesh ladder to this many levels (smoke tests)",
    )
    args = ap.parse_args()
    if bool(args.anchor_run) == bool(args.config):
        ap.error("give exactly one of --anchor-run or --config")

    setup_logging(log_level="INFO")
    if args.config:
        # EXPLORATORY: no anchor, so no mesh-match assertion and no scored
        # perimeter. The V assertion exists to stop an arm being compared against
        # a baseline built on a different mesh; with no baseline there is nothing
        # to protect, and nothing to claim either.
        cfg_path = Path(args.config).resolve()
        spec = spec_from_config(cfg_path)
        anchor = None
        anchor_V = None
        campaign = Path(args.campaign).resolve() if args.campaign else cfg_path
        print(f"config        : {cfg_path.name}   [EXPLORATORY -- no anchor]")
    else:
        anchor = resolve_anchor(args.anchor_run)
        spec = anchor_spec(anchor)
        anchor_V = anchor_solution_vertices(anchor)
        campaign = (
            Path(args.campaign).resolve() if args.campaign else anchor_campaign(anchor)
        )
        print(f"anchor run    : {anchor.name}")
    if args.levels is not None:
        spec["levels"] = int(args.levels)
    seed = args.seed if args.seed is not None else spec["seed"]
    N = spec["n_partitions"]
    print(
        f"ladder        : {spec['levels']} levels from {spec['n_theta']}x{spec['n_phi']} "
        f"(+{spec['d_theta']}/+{spec['d_phi']}), N={N}"
    )
    print(
        f"anchor V      : {anchor_V if anchor_V is not None else 'n/a (exploratory)'}"
    )
    print(f"campaign      : {campaign}")
    print(f"arm seed      : {seed}{'  (overridden)' if args.seed is not None else ''}")

    meshes = build_ladder(spec)
    arm_V = len(meshes[-1].vertices)
    print(f"arm final V   : {arm_V}")
    if anchor_V is not None and arm_V != anchor_V:
        print(f"\nREFUSING TO SCORE: arm final V={arm_V} != anchor V={anchor_V}.")
        print("The ladder and the anchor disagree; scoring across meshes would put a")
        print(
            "mesh-refinement effect into the comparison. Fix the ladder, not this check."
        )
        return 2

    cfg = MBOConfig(
        tau_c=args.tau_c,
        rho=args.rho,
        max_iters=args.max_iters,
        seed=seed,
        nc3_K=args.nc3_k,
        nc3_f_min=args.nc3_f_min,
    )
    mode = "init" if args.init_only else "mbo"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    run_root = (
        Path(args.out_root) / f"arm_{mode}_{stamp}_npart{N}_V{arm_V}_seed{seed}{tag}"
    )
    (run_root / "solution").mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    labels, reports, info = run_mbo_ladder(meshes, N, cfg, init_only=args.init_only)
    wall = time.perf_counter() - t0

    mesh = meshes[-1]
    gates = run_gates(labels_to_one_hot(labels, N), mesh, N)
    lbl = label_boundary_length(labels, mesh.vertices, mesh.faces)

    print("\n--- RAW arm labels, all three gates ---")
    print(
        f"  dormant      : {gates.n_dead} dead / {gates.n_weak} weak "
        f"(vacuous for arms: one-hot)"
    )
    print(
        f"  area         : {gates.n_imbalanced} imbalanced, worst "
        f"{gates.worst_rel_dev*100:.4f}%  bar {gates.quality_bar*100:.4f}%  "
        f"granularity {gates.vertex_granularity_rel*100:.4f}%"
    )
    if gates.worst_rel_dev > gates.quality_bar:
        print(
            "    ^ ABOVE THE BAR => HARNESS FAULT (fix the assignment), "
            "NOT evidence against B"
        )
    print(f"  connectivity : {gates.n_fragmented} fragmented {gates.fragmented[:10]}")
    print(f"  boundary len : {lbl:.4f}   arm wall {wall:.0f}s")

    sol = str(run_root / "solution" / f"arm_{mode}_part{N}_V{arm_V}_seed{seed}.h5")
    write_arm_solution(
        sol,
        labels,
        mesh.vertices,
        mesh.faces,
        N,
        extra_attrs={
            "arm": mode,
            "tau_c": args.tau_c,
            "rho": args.rho,
            "seed": seed,
            # HDF5 attrs cannot hold None, so exploratory runs record the config
            # they came from instead of an anchor that does not exist.
            "anchor_run": anchor.name if anchor is not None else "",
            "provenance": (
                anchor.name if anchor is not None else f"exploratory:{campaign.name}"
            ),
        },
    )

    report = {
        "arm": mode,
        "anchor_run": anchor.name if anchor is not None else None,
        "anchor_V": anchor_V,
        "exploratory": anchor is None,
        "arm_final_V": arm_V,
        "n_partitions": N,
        "seed": seed,
        "config": cfg.to_dict(),
        "init": info,
        "wall_seconds": wall,
        "levels": [r.to_dict() for r in reports],
        "gates_raw": gates.to_dict(),
        "label_boundary_length": float(lbl),
        "solution": sol,
        "campaign_config": str(campaign),
    }

    if args.phase2:
        print("\nrunning Phase 2 through the harness (pinned campaign) ...")
        try:
            p2, _ = run_phase2(sol, str(campaign), str(ROOT))
            report["phase2"] = p2.to_dict()
            # A HARD abort raises (non-zero exit) and is caught below. A SOFT one
            # does not: report 05's A2 arm stopped at topology iterate 3 of 20 and
            # the script still exited cleanly, so the only evidence is a short
            # campaign. Without this check P12 would be scored "no abort" on a run
            # that died two thirds of the way through.
            with open(campaign) as cf:
                expected = int(yaml.safe_load(cf).get("max_iterations", 0))
            report["phase2"]["expected_iterations"] = expected
            short = expected and p2.n_iterations < expected
            report["phase2"]["short_campaign"] = bool(short)
            if short:
                print(
                    f"  ** SHORT CAMPAIGN: {p2.n_iterations} iterates of {expected} "
                    "expected -> treat as a Phase 2 abort (lane 6a: REFUTATION ON "
                    "GEOMETRY), NOT scored against +1%/+5% **"
                )
            print(
                f"  best perimeter : {p2.best_perimeter:.13f} "
                f"(iterate {p2.best_iteration} of {p2.n_iterations})"
            )
            print(
                f"  censored       : {p2.censored}"
                f"{'   <- still falling; NOT scored against a threshold' if p2.censored else ''}"
            )
            print(f"  wall           : {p2.wall_seconds:.0f}s")
        except Exception as exc:  # noqa: BLE001
            # Pre-registered lane 6a: an abort is REFUTATION ON GEOMETRY, reported
            # with the iterate reached, and explicitly NOT scored against the
            # +1%/+5% bars -- a solver that died is not a perimeter measurement.
            report["phase2_aborted"] = {"error": str(exc)[:4000]}
            print("\n  PHASE 2 ABORTED -- lane 6a: REFUTATION ON GEOMETRY.")
            print("  Not scored against the +1%/+5% bars. See the report YAML.")

    out = run_root / "arm_report.yaml"

    def _plain(o):
        # numpy scalars are not YAML-serializable. Cast bools to bool FIRST:
        # np.bool_ passed through float() becomes 0.0/1.0, which silently turns
        # every flag in the report -- censored, cap_active, annealed -- into a
        # number nobody reading the YAML would recognise as a flag.
        if isinstance(o, (bool, np.bool_)):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        raise TypeError(type(o))

    with open(out, "w") as f:
        yaml.safe_dump(
            json.loads(json.dumps(report, default=_plain)),
            f,
            sort_keys=False,
            width=100,
        )
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
