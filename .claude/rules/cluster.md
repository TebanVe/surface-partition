---
paths:
  - "cluster/**"
---

# Running on Pelle (UPPMAX)

Code and scripts live in `$HOME` (small, backed up). Large output data (results,
HDF5) goes under `/proj/<allocation>/`.

**First-time setup:**

1. Clone the repo to `$HOME`.
2. Edit `cluster/pelle_config.sh` — set `PROJECT_ID`, `PROJECT_BASE`, verify
   `PYTHON_MODULE` (check with `module spider python` on Pelle).
3. Create the venv on Pelle per the instructions in `pelle_config.sh`.

```bash
# Phase 1
bash cluster/submit_relaxation.sh --config parameters/torus_100part_coarse_seeded.yaml
bash cluster/submit_relaxation.sh --config <cfg> --time 24:00:00 --cpus 8
bash cluster/submit_relaxation.sh --config <cfg> --resume-from results/run_.../solution/surface_....h5
bash cluster/submit_relaxation.sh --config <cfg> --dry-run

# Phase 2
bash cluster/submit_refinement.sh --solution <solution.h5> --config <cfg>
bash cluster/submit_refinement.sh --solution <solution.h5> --config <cfg> --method ipopt --exact-hessian

# Sweep (one job per combination)
bash cluster/submit_sweep.sh --sweep sweep/parameters/sweep_torus_lambda.yaml [--dry-run|--auto-collect]

# Collect after the jobs finish
python sweep/parameter_sweep.py --sweep <spec> --mode collect
```

These scripts target Pelle specifically — `cluster/pelle_config.sh` is the one
file to edit before first use.
