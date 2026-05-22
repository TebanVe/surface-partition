# Surface Partition Optimization Framework

This project implements a framework for finding minimal-perimeter partitions on closed triangulated surfaces, based on the method from "Partitions of Minimal Length on Manifolds" by Bogosel and Oudet. The framework supports any triangulated surface (2D or 3D) and provides a two-phase pipeline: Gamma-convergence relaxation followed by direct perimeter refinement.

## Mathematical Framework

The project uses a Gamma-convergence approach with the energy functional:

$$J_ε(u) = ε \int_S |∇_τ u|^2 + \frac{1}{ε} \int_S u^2(1 - u)^2$$

For partitions, we minimize:

$$\sum_{i=1}^n J_ε(u_i)$$

Subject to constraints:
- **Partition constraint**: $\sum_{i=1}^n u_i = 1$ at each vertex
- **Area constraint**: $\int_S u_i = A/n$ for each partition

Where $S$ is any triangulated surface and $∇_τ$ denotes the tangential gradient.

## Key Features

### Surfaces
Four surface types are implemented:
- **Torus** (`TorusMeshProvider`) — structured parametric mesh
- **Ellipsoid** (`EllipsoidMeshProvider`) — parametric spherical-coordinate mesh
- **Double torus** (`DoubleTorusMeshProvider`) — implicit surface via marching cubes
- **Banchoff-Chmutov order 4** (`BanchoffChmutovMeshProvider`) — implicit surface via marching cubes

### Phase 1: Gamma-Convergence Relaxation
- Custom Projected Gradient Descent (PGD) with per-step projection onto partition and area constraints
- FEM-weighted penalty for constant-phase regularization (fixed or adaptive target mode)
- Armijo backtracking line search
- Multi-level mesh refinement with nearest-neighbor solution interpolation
- Automatic refinement triggers based on convergence metrics

### Phase 2: Perimeter Refinement
- Direct perimeter minimization on extracted contours
- Equal-area constraints with analytical gradients (SLSQP, trust-constr, or IPOPT)
- Steiner tree handling for triple points (where 3 regions meet)
- Automatic topology migrations: Type 1 (vertex collapse) and Type 2 (triple-point)
- Vectorized kernels for fast perimeter/area/Steiner evaluation
- Exact analytical Hessian for IPOPT (in addition to L-BFGS)
- Optional timing profiling via `--profile`

### Parameter Sweeps
- Grid or paired sweep over arbitrary YAML parameters
- Local sequential, local parallel, generate-only, and collect modes
- Automatic `experiment_index.yaml` with per-run metrics
- Sweep analysis plots (heatmaps, convergence overlays, sensitivity)
- Timing analysis for profiled runs

## Installation

Requires Python >= 3.9. The project uses pyenv for environment management.

```bash
# Activate the pyenv virtualenv (see .python-version)
pyenv activate surface-partition   # or: pyenv activate ringtest-3.9

# Install core dependencies
pip install -e .

# Install optional extras
pip install -e ".[viz]"       # core + PyVista (3D visualisation)
pip install -e ".[ipopt]"     # core + IPOPT solver (cyipopt)
pip install -e ".[implicit]"  # core + scikit-image (implicit surfaces)
pip install -e ".[all]"       # all of the above
pip install -e ".[dev]"       # adds pytest, black, flake8
```

## Project Structure

```
surface-partition/
├── src/                          # Core library
│   ├── mesh/                     # Mesh and FEM
│   │   ├── tri_mesh.py           # TriMesh: vertices, faces, FEM matrices (M, K, v)
│   │   ├── mesh_topology.py      # Edge/triangle adjacency for migrations
│   │   └── interpolation.py      # Nearest-neighbor interpolation between levels
│   ├── surfaces/                 # Surface providers
│   │   ├── base.py               # SurfaceProvider abstract base class
│   │   ├── torus.py              # TorusMeshProvider
│   │   ├── ellipsoid.py          # EllipsoidMeshProvider
│   │   ├── implicit.py           # ImplicitSurfaceProvider (marching-cubes base)
│   │   ├── double_torus.py       # DoubleTorusMeshProvider
│   │   └── banchoff_chmutov.py   # BanchoffChmutovMeshProvider
│   ├── optimization/             # Optimizers
│   │   ├── pgd_optimizer.py      # Phase 1: Projected Gradient Descent
│   │   ├── perimeter_optimizer.py # Phase 2: constrained perimeter minimization
│   │   ├── projection.py         # Iterative constraint projection
│   │   └── exceptions.py         # RefinementTriggered exception
│   ├── partition/                # Contour extraction and partition data
│   │   ├── find_contours.py      # ContourAnalyzer: HDF5 → boundary topology
│   │   ├── contour_partition.py  # PartitionContour, VariablePoint, TriangleSegment
│   │   ├── perimeter_calculator.py # Per-segment perimeter with analytical gradients
│   │   ├── area_calculator.py    # Per-cell FEM area with analytical gradients
│   │   ├── steiner_handler.py    # Steiner/triple-point contributions
│   │   ├── partition_arrays.py   # Sparse Jacobian/Hessian sparsity (IPOPT)
│   │   ├── vectorized_perimeter.py
│   │   ├── vectorized_area.py
│   │   └── vectorized_steiner.py # Analytical first and second Steiner derivatives
│   ├── migration/                # Topology migration subsystem
│   │   ├── migration_orchestrator.py
│   │   ├── migration_detector.py
│   │   ├── migration_executor.py
│   │   ├── migration_types.py
│   │   ├── migration_utils.py
│   │   ├── type2_migration_io.py
│   │   ├── type2_migration_history.py
│   │   └── one_ring_rebuilder.py
│   ├── pipeline/                 # Workflow orchestration
│   │   ├── relaxation.py         # run_relaxation(): multi-level PGD pipeline
│   │   ├── pipeline_orchestrator.py # Phase 2 iterative refinement loop
│   │   └── io.py                 # HDF5 loaders, run-layout detection
│   ├── visualization/            # Plotting helpers
│   │   ├── plot_utils.py
│   │   ├── partition_helpers.py
│   │   └── partition_screenshots.py
│   ├── profiling.py              # ProfilingState: opt-in timing accumulator
│   └── logging_config.py         # Logging setup, @log_performance decorator
├── scripts/                      # CLI entry points
│   ├── find_surface_partition.py # Phase 1: Gamma-convergence relaxation
│   ├── refine_perimeter.py       # Phase 2: iterative perimeter refinement
│   ├── optimization_analyzer.py  # Per-run analysis and plotting
│   ├── visualize_partition_fast.py # Fast production viewer (PyVista, vectorized)
│   ├── visualize_partition.py    # Original viewer — debugging (PyVista)
│   ├── visualize_type1_vertex_collapse.py
│   ├── visualize_type2_triple_point.py
│   └── debug_archive/
├── testing/                      # CLI diagnostic tools (not pytest)
│   ├── README_testing.md
│   └── test_*.py / diagnose_*.py / compare_*.py
├── parameters/                   # Experiment configuration files
│   ├── torus_10part.yaml
│   ├── ellipsoid_6part.yaml
│   ├── double_torus_10part.yaml
│   └── banchoff_chmutov_12part.yaml
├── sweep/                        # Parameter sweep tool
│   ├── parameter_sweep.py        # Sweep orchestrator
│   ├── sweep_analyzer.py         # Experiment-wide analysis plots
│   ├── timing_analyzer.py        # Scaling figures from profiled runs
│   └── parameters/
│       ├── sweep_torus_lambda.yaml
│       └── sweep_double_torus_lambda.yaml
├── cluster/                      # HPC job submission (UPPMAX Pelle)
│   ├── pelle_config.sh           # Shared config — edit REPO_DIR and VENV_DIR
│   ├── submit_relaxation.sh      # Submit Phase 1 job
│   ├── submit_refinement.sh      # Submit Phase 2 job
│   ├── submit_sweep.sh           # Submit parameter sweep (one job per combination)
│   └── cleanup_sweep_results.py  # Prune worst sweep runs
├── docs/                         # Documentation
│   ├── math/                     # LaTeX derivations (compiled PDFs)
│   ├── guides/                   # LaTeX user guides (compiled PDFs)
│   ├── plans/                    # Design plans for future work
│   └── reference/                # Permanent explanatory documents
└── pyproject.toml                # Package config, Black, pytest
```

## Usage

### Phase 1: Gamma-Convergence Relaxation

```bash
# Run from experiment config
python scripts/find_surface_partition.py --config parameters/torus_10part.yaml
python scripts/find_surface_partition.py --config parameters/ellipsoid_6part.yaml
python scripts/find_surface_partition.py --config parameters/double_torus_10part.yaml
python scripts/find_surface_partition.py --config parameters/banchoff_chmutov_12part.yaml

# Warm-start from a prior solution (increase refinement_levels in YAML first)
python scripts/find_surface_partition.py \
    --config parameters/torus_10part.yaml \
    --resume-from results/prior/solution/surface_....h5
```

### Phase 2: Perimeter Refinement

```bash
# Default run (reads refinement: section from config)
python scripts/refine_perimeter.py \
    --solution results/run_xyz/solution/surface_....h5 \
    --config parameters/torus_10part.yaml

# Override solver and iteration count
python scripts/refine_perimeter.py \
    --solution results/run_xyz/solution/surface_....h5 \
    --max-iterations 10 --method ipopt --exact-hessian

# Resume from checkpoint (auto-detected by the pipeline)
python scripts/refine_perimeter.py \
    --solution results/run_xyz/refinement/slsqp_btol0.001/iteration_003_....h5 \
    --config parameters/torus_10part.yaml

# Enable timing profiling (writes timing_profile.yaml per campaign)
python scripts/refine_perimeter.py \
    --solution results/run_xyz/solution/surface_....h5 \
    --config parameters/torus_10part.yaml --profile
```

### Analysis and Visualisation

```bash
# Analyse a single run
python scripts/optimization_analyzer.py --results-dir results/run_<timestamp>_...

# Visualise partition — fast renderer (production, scales to fine meshes)
python scripts/visualize_partition_fast.py \
    --solution results/run_xyz/refinement/slsqp_btol0.001/iteration_005_....h5 \
    --show-steiner

# Visualise partition — original renderer (debugging)
python scripts/visualize_partition.py --solution results/run_xyz/solution/surface_....h5

# Migration debugging viewers
python scripts/visualize_type1_vertex_collapse.py \
    --solution results/run_xyz/refinement/slsqp_btol0.001/iteration_005_....h5 \
    --region 2
python scripts/visualize_type2_triple_point.py \
    --solution results/run_xyz/refinement/slsqp_btol0.001/iteration_005_....h5 \
    --region 2
```

### Parameter Sweeps

```bash
# Preview what will run (no jobs submitted)
python sweep/parameter_sweep.py \
    --sweep sweep/parameters/sweep_torus_lambda.yaml --mode generate-only

# Run sequentially
python sweep/parameter_sweep.py --sweep sweep/parameters/sweep_torus_lambda.yaml

# Run in parallel (4 workers)
python sweep/parameter_sweep.py \
    --sweep sweep/parameters/sweep_torus_lambda.yaml --mode local-parallel --workers 4

# Collect results and build experiment_index.yaml
python sweep/parameter_sweep.py \
    --sweep sweep/parameters/sweep_torus_lambda.yaml --mode collect

# Experiment-wide analysis plots
python sweep/sweep_analyzer.py --experiment-dir results/torus_npart10/

# Timing analysis (for --profile runs)
python sweep/timing_analyzer.py --experiment-dir results/torus_npart10/
```

## Configuration

Each experiment is defined by a single YAML file (e.g. `parameters/torus_10part.yaml`) with four sections:

- **`experiment`**: name and surface type
- **`relaxation`**: Phase 1 PGD parameters (`n_partitions`, `lambda_penalty`, `seed`, convergence thresholds, etc.)
- **`surface`**: geometry parameters keyed by surface name (e.g. `torus: {n_theta, n_phi, R, r}`)
- **`refinement`**: Phase 2 parameters (`method`, `max_iterations`, `boundary_tol`, IPOPT options, etc.)

Both scripts accept `--config <experiment.yaml>`. CLI flags override YAML values. A verbatim copy of the config is saved as `experiment.yaml` in the run directory for reproducibility.

## Cluster Support (UPPMAX Pelle)

The cluster submission scripts target UPPMAX Pelle. See `docs/guides/01-pelle-user-guide/main.pdf` for the full collaborator guide, including first-time setup, directory conventions, and all submission options.

Quick reference:

```bash
# Edit cluster/pelle_config.sh first: set REPO_DIR and VENV_DIR to your username

# Phase 1 relaxation
bash cluster/submit_relaxation.sh --config parameters/torus_10part.yaml
bash cluster/submit_relaxation.sh --config parameters/torus_10part.yaml \
    --time 24:00:00 --cpus 8

# Phase 2 refinement
bash cluster/submit_refinement.sh \
    --solution /proj/.../results/run_.../solution/surface_....h5 \
    --config parameters/torus_10part.yaml

# Parameter sweep (one SLURM job per combination)
bash cluster/submit_sweep.sh \
    --sweep sweep/parameters/sweep_torus_lambda.yaml --auto-collect
```

Note: IPOPT is not available on Pelle. Phase 2 on the cluster uses `--method slsqp` (default) or `--method trust-constr`.

## References

- Bogosel, B., & Oudet, É. "Partitions of Minimal Length on Manifolds"
- Wächter, A., & Biegler, L.T. "On the Implementation of an Interior-Point Filter Line-Search Algorithm for Large-Scale Nonlinear Programming" (IPOPT)
