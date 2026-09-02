#!/usr/bin/env python3
"""Run one study config end to end: Phase 1 -> Phase 2 -> exported partition.

The manual sequence has one error-prone step -- the deliverable must come from
the BEST iterate, which is frequently not the last one (N=500 best at 10/20,
N=750 at 9/20, N=1000 at 15/20). This driver picks it by scanning
``final_perimeter`` rather than trusting position, and refuses to export if any
Phase 1 validity gate failed.

    python scripts/run_study_config.py --config parameters/torus_B3_n50.yaml
    python scripts/run_study_config.py --config ... --skip-phase1   # resume

Writes everything under --out-root (default results/mc_study) and prints a
one-line summary per stage so a tail of the log tells you where it is.
"""
import argparse, glob, os, re, subprocess, sys, time
import h5py, numpy as np, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sh(cmd, log):
    """Run a stage, streaming to a log file; raise on failure."""
    with open(log, "ab") as f:
        p = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        raise SystemExit(f"FAILED ({p.returncode}): {' '.join(cmd)}\n  see {log}")


def newest_run(out_root, n, V):
    cands = sorted(glob.glob(os.path.join(out_root, f"arm_mbo_*_npart{n}_V{V}_*")))
    if not cands:
        raise SystemExit(f"no run directory found for n={n} V={V} under {out_root}")
    return cands[-1]


def best_iterate(run):
    """Minimum final_perimeter across the campaign -- NOT the last iterate."""
    fs = glob.glob(os.path.join(run, "refinement", "*", "iteration_*.h5"))
    if not fs:
        raise SystemExit(f"no Phase 2 iterates in {run}")
    best = (float("inf"), None, None)
    for f in fs:
        with h5py.File(f) as h:
            p = float(h.attrs["final_perimeter"])
            if p < best[0]:
                best = (p, f, int(h.attrs["iteration_number"]))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-root", default="results/mc_study")
    ap.add_argument("--skip-phase1", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(ROOT, a.config)))
    n = cfg["relaxation"]["n_partitions"]
    t = cfg["surface"]["torus"]
    nt, npi = t["n_theta"], t["n_phi"]
    for _ in range(cfg["relaxation"]["refinement_levels"] - 1):
        nt, npi = nt + t["n_theta_increment"], npi + t["n_phi_increment"]
    V = nt * npi
    tag = os.path.basename(a.config).replace("torus_", "").replace(".yaml", "")
    os.makedirs(os.path.join(ROOT, a.out_root), exist_ok=True)
    log = os.path.join(ROOT, a.out_root, f"{tag}.log")
    t0 = time.time()
    print(f"[{tag}] n={n} V={V:,}  log -> {log}", flush=True)

    if not a.skip_phase1:
        print(f"[{tag}] phase 1 ...", flush=True)
        sh([sys.executable, "scripts/run_mbo_arm.py", "--config", a.config,
            "--out-root", a.out_root], log)
    run = newest_run(os.path.join(ROOT, a.out_root), n, V)
    rel = os.path.relpath(run, ROOT)

    rep = yaml.safe_load(open(os.path.join(run, "arm_report.yaml")))
    g = rep["gates_raw"]
    bad = (g["dormant"]["n_dead"] or g["dormant"]["n_weak"]
           or g["area_imbalance"]["n_imbalanced"] or g["connectivity"]["n_fragmented"])
    print(f"[{tag}] phase 1 done {rep['wall_seconds']:.0f}s  "
          f"worst {g['area_imbalance']['worst_rel_dev']*100:.4f}% "
          f"(bar {g['area_imbalance']['harness_fault_threshold']*100:.4f}%)  "
          f"frag {g['connectivity']['n_fragmented']}", flush=True)
    if bad:
        raise SystemExit(f"[{tag}] GATE FAILURE -- refusing to continue. See {run}/arm_report.yaml")

    sol = glob.glob(os.path.join(run, "solution", "*.h5"))[0]
    if not glob.glob(os.path.join(run, "refinement", "*", "iteration_*.h5")):
        print(f"[{tag}] phase 2 ...", flush=True)
        sh([sys.executable, "scripts/refine_perimeter.py", "--solution",
            os.path.relpath(sol, ROOT), "--config", a.config, "--profile"], log)

    per, bf, bit = best_iterate(run)
    nit = len(glob.glob(os.path.join(run, "refinement", "*", "iteration_*.h5")))
    cens = " CENSORED (best is last -- still improving)" if bit == nit else ""
    print(f"[{tag}] phase 2 done  best {per:.6f} at iterate {bit}/{nit}{cens}", flush=True)

    out = os.path.join(run, "partition", f"torus_partition_{tag}_V{V}.h5")
    print(f"[{tag}] exporting ...", flush=True)
    sh([sys.executable, "scripts/export_partition.py", "--solution",
        os.path.relpath(bf, ROOT), "--config", a.config,
        "--output", os.path.relpath(out, ROOT), "--force-finalised"], log)
    with h5py.File(out) as h:
        print(f"[{tag}] DONE in {(time.time()-t0)/60:.0f} min  "
              f"n_cells={h.attrs['n_cells']} finalised={h.attrs['finalised']} "
              f"perimeter={h.attrs['final_perimeter']:.6f}", flush=True)
        print(f"[{tag}] -> {os.path.relpath(out, ROOT)}", flush=True)


if __name__ == "__main__":
    main()
