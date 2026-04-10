# Surface Partition Optimization Framework

This project implements a framework for finding minimal-perimeter partitions on closed triangulated surfaces, based on the optimization method described in "Partitions of Minimal Length on Manifolds" by Bogosel and Oudet. The framework supports any triangulated surface (2D or 3D) and provides a two-phase pipeline: Γ-convergence relaxation followed by direct perimeter refinement.

## Mathematical Framework

The project uses a Γ-convergence approach with the energy functional:

$$J_ε(u) = ε \int_S |∇_τ u|^2 + \frac{1}{ε} \int_S u^2(1 - u)^2$$

For partitions, we minimize:

$$\sum_{i=1}^n J_ε(u_i)$$

Subject to constraints:
- **Partition constraint**: $\sum_{i=1}^n u_i = 1$ at each vertex
- **Area constraint**: $\int_S u_i = A/n$ for each partition

Where $S$ is any triangulated surface and $∇_τ$ denotes the tangential gradient.

## Key Features

### Perimeter Refinement (Phase 2)
- **Section 5 Implementation**: Direct perimeter minimization on extracted contours
- **Constrained Optimization**: Equal-area constraints with analytical gradients (SLSQP, trust-constr, or IPOPT)
- **Steiner Trees**: Optimal handling of triple points (3 regions meet)
- **Topology Migrations**: Type 1 (vertex collapse) and Type 2 (triple-point) automatic detection and execution
- **Vectorized Kernels**: Fast perimeter/area/Steiner evaluation paths

### Surface-Agnostic Design
- **TriMesh**: Universal triangle mesh class supporting both 2D and 3D surfaces
- **Surface Providers**: Modular system for different surface types (currently: torus)
- **P1 FEM Assembly**: Automatic mass and stiffness matrix computation for any triangulation

### Projected Gradient Descent (PGD) — Phase 1
- **Custom gradient descent** with per-step projection onto partition and area constraints
- **FEM-weighted penalty** for constant-phase regularization (fixed or adaptive target mode)
- **Armijo backtracking** line search
- **Mesh refinement triggers**: Plateau detection on energy, gradient norm, and feasibility

### Mesh Refinement System
- **Multi-level Optimization**: Progressive mesh refinement with nearest-neighbor solution interpolation
- **Automatic Refinement**: Smart triggers based on convergence metrics and plateaus
- **Warm-start Resume**: Continue from a prior solution HDF5 at a higher refinement level

### Analysis & Visualization
- **Optimization Analyzer**: Multi-level result analysis with energy component and constraint evolution plots
- **Partition Viewer**: 3D visualization via PyVista (base or refined solutions)
- **HDF5 Output**: Efficient storage of optimization iterates, solutions, and refined contour checkpoints

## Installation

Requires Python >= 3.9. The project uses pyenv for environment management.

```bash
# Activate the pyenv virtualenv (see .python-version)
pyenv activate ringtest-3.9

# Install core dependencies
pip install -e .

# Or with optional extras:
pip install -e ".[viz]"       # core + PyVista (3D visualization)
pip install -e ".[ipopt]"     # core + IPOPT solver
pip install -e ".[all]"       # core + PyVista + IPOPT
```

## Project Structure

```
surface-partition/
├── src/                              # Core library
│   ├── mesh/                         # Mesh and FEM
│   │   ├── tri_mesh.py              # TriMesh class (vertices, faces, M, K, v)
│   │   ├── mesh_topology.py         # Edge/triangle adjacency for migrations
│   │   └── interpolation.py         # Nearest-neighbor interpolation between levels
│   ├── surfaces/                     # Surface providers
│   │   └── torus.py                 # TorusMeshProvider (R3 torus of revolution)
│   ├── optimization/                 # Optimizers
│   │   ├── pgd_optimizer.py         # Projected gradient descent (Phase 1)
│   │   ├── perimeter_optimizer.py   # Constrained perimeter minimization (Phase 2)
│   │   ├── projection.py           # Iterative constraint projection
│   │   └── exceptions.py           # RefinementTriggered exception
│   ├── partition/                    # Contour extraction and partition data
│   │   ├── find_contours.py         # ContourAnalyzer: HDF5 → indicators → boundary topology
│   │   ├── contour_partition.py     # PartitionContour, VariablePoint, TriangleSegment
│   │   ├── perimeter_calculator.py  # Per-segment perimeter with analytical gradients
│   │   ├── area_calculator.py       # Per-cell FEM area with analytical gradients
│   │   ├── steiner_handler.py       # Steiner/triple-point perimeter + area contributions
│   │   ├── partition_arrays.py      # PartitionArrays: sparse Jacobian/Hessian structure
│   │   ├── vectorized_perimeter.py  # Fast vectorized perimeter evaluation
│   │   ├── vectorized_area.py       # Fast vectorized area evaluation
│   │   └── vectorized_steiner.py    # Fast vectorized Steiner evaluation
│   ├── migration/                    # Topology migration subsystem
│   │   ├── migration_orchestrator.py # Top-level detect → execute loop
│   │   ├── migration_detector.py    # Type 1 + Type 2 trigger detection
│   │   ├── migration_executor.py    # Execute migrations on partition state
│   │   ├── migration_types.py       # DetectionResult, MigrationResult dataclasses
│   │   ├── migration_utils.py       # Shared helpers
│   │   ├── type1_component_analyzer.py  # Connected-component analysis for Type 1
│   │   ├── type2_migration_io.py    # Type 2 snapshot save/restore
│   │   ├── type2_migration_history.py   # Type 2 rollback history
│   │   └── one_ring_rebuilder.py    # One-ring mesh rebuilding after migration
│   ├── pipeline/                     # Workflow orchestration
│   │   ├── relaxation.py           # run_relaxation(): multi-level PGD pipeline
│   │   ├── pipeline_orchestrator.py # PipelineOrchestrator: iterative refinement loop
│   │   └── io.py                   # HDF5 loaders (base + refined files)
│   ├── visualization/               # Plotting helpers
│   │   ├── plot_utils.py           # Matplotlib utilities
│   │   └── partition_helpers.py    # Partition-specific viz helpers
│   ├── config.py                    # Legacy Config class
│   └── logging_config.py           # Logging setup with performance decorators
├── scripts/                          # CLI entry points
│   ├── find_surface_partition.py    # Phase 1: Γ-convergence relaxation
│   ├── refine_perimeter.py         # Phase 2: iterative perimeter refinement
│   ├── optimization_analyzer.py     # Result analysis and plotting
│   ├── visualize_partition.py       # Original partition viewer — debugging (PyVista)
│   ├── visualize_partition_fast.py  # Fast partition viewer — production (PyVista, vectorized)
│   ├── visualize_type1_vertex_collapse.py  # Type 1 migration viewer — debugging
│   ├── visualize_type2_triple_point.py     # Type 2 migration viewer — debugging
│   └── debug_archive/              # Archived diagnostic scripts
├── testing/                          # Test scripts and debug tools
│   ├── README_testing.md           # Test registry documentation
│   └── test_migrations_debug.py    # Migration debug CLI
├── parameters/                       # Configuration
│   └── input.yaml                  # Default run parameters
├── cluster/                          # HPC job submission
│   └── submit.sh                   # SLURM script for UPPMAX (Rackham)
├── pyproject.toml                   # Package config, Black, pytest
└── requirements.txt                 # Core dependencies
```

## Usage

### Phase 1: Γ-Convergence Relaxation

```bash
# Run with default parameters on the torus
python scripts/find_surface_partition.py --input parameters/input.yaml --surface torus

# Custom output directory
python scripts/find_surface_partition.py --input parameters/input.yaml --solution-dir results/my_run

# Warm-start from a prior solution (must increase refinement_levels in YAML)
python scripts/find_surface_partition.py --input parameters/input.yaml --resume-from results/prior/solution.h5
```

### Phase 2: Perimeter Refinement

After obtaining a relaxed solution, refine the contours to get accurate perimeter values:

```bash
# Fresh run from base solution
python scripts/refine_perimeter.py \
    --solution results/run_xyz/solution.h5 \
    --max-iterations 10

# Resume from checkpoint (auto-detected)
python scripts/refine_perimeter.py \
    --solution results/run_xyz/solution_btol0.001_iteration3_refined_contours.h5 \
    --max-iterations 10

# With IPOPT solver and exact Hessian
python scripts/refine_perimeter.py \
    --solution results/run_xyz/solution.h5 \
    --max-iterations 10 --method ipopt --exact-hessian

# Save all intermediate checkpoints
python scripts/refine_perimeter.py \
    --solution results/run_xyz/solution.h5 \
    --max-iterations 20 --save-iterations
```

### Analysis and Visualization

```bash
# Analyze optimization results
python scripts/optimization_analyzer.py --results-dir results/run_20250101_120000_surftorus_npart2_...

# Visualize partition — fast renderer (vectorized, scales to fine meshes)
python scripts/visualize_partition_fast.py --solution results/run_xyz/*_refined_contours.h5 --show-steiner

# Visualize partition — original renderer (slower on fine meshes, useful for debugging)
python scripts/visualize_partition.py --solution results/run_xyz/solution.h5

# Visualize Type 1 migration (vertex collapse) — debugging tool
python scripts/visualize_type1_vertex_collapse.py \
    --solution results/run_xyz/*_refined_contours.h5 \
    --region 2 --state before --show-vps --show-steiner

# Visualize Type 2 migration (triple-point) — debugging tool
python scripts/visualize_type2_triple_point.py \
    --solution results/run_xyz/*_refined_contours.h5 \
    --region 2 --state before --show-vps --show-steiner

# Debug migrations step-by-step
python testing/test_migrations_debug.py --solution results/run_xyz/*_refined_contours.h5
```

All visualization scripts require PyVista. The fast renderer (`visualize_partition_fast.py`) uses vectorized NumPy indexing for interior triangles and is the recommended choice for fine meshes. The original renderer and the two migration viewers (`visualize_type1_vertex_collapse.py`, `visualize_type2_triple_point.py`) are slower but invaluable for testing and debugging topology switches at small scales.

### Visualization Flags

```
--region N             Highlight a specific cell
--show-steiner         Show Steiner points and void triangles
--show-vps             Show all variable points
--use-initial          Visualize x0 instead of x_opt (base solutions only)
--opacity 0.8          Cell color opacity
--save <path>          Save image to file (otherwise interactive window)
--no-fill / --no-mesh  Toggle filled partitions or mesh overlay (2D)
--color-partition       Light per-face region colors with strong contour lines (3D)
```

### Torus Configuration

Torus-specific YAML parameters (used when `--surface torus`):

```yaml
n_theta: 80          # samples along major circle
n_phi: 66            # samples along tube circle
R: 1.0               # major radius
r: 0.6               # minor radius
n_theta_increment: 62 # resolution increase per refinement level (major)
n_phi_increment: 58   # resolution increase per refinement level (minor)
```

## Configuration

The project is configured through `parameters/input.yaml`. Key parameter groups:

### Optimization
- `n_partitions`: Number of equal-area partitions
- `lambda_penalty`: Constant-phase penalty weight
- `max_iter`: Maximum PGD iterations per refinement level
- `seed`: Random seed for initial conditions

### PGD Tuning
- `pgd_step0`, `pgd_armijo_c`, `pgd_backtrack_rho`: Step size and backtracking
- `penalty_target_mode`: `fixed` (paper default) or `adaptive`
- `penalty_eps`: Stabilizer for penalty target denominators

### Mesh Refinement
- `refinement_levels`: Number of multi-level refinement steps
- `enable_refinement_triggers`: Enable/disable automatic early refinement
- `refine_gnorm_patience`, `refine_feas_patience`: Plateau detection windows
- `refine_trigger_mode`: `full` (energy + gradient + feasibility) or `energy` only

### HDF5 Output Control
- `h5_save_stride`: Save every N-th iteration
- `h5_save_vars`: Variables to save (`x`, `constraints`)
- `run_log_frequency`: Console logging interval

## Cluster Support

The SLURM submission script targets UPPMAX (Rackham) but can be adapted:

```bash
bash cluster/submit.sh --input parameters/input.yaml --surface torus
bash cluster/submit.sh --input parameters/input.yaml --time 24:00:00 --solution-dir /proj/.../SOLUTIONS
```

## References

- Bogosel, B., & Oudet, É. "Partitions of Minimal Length on Manifolds"
- Γ-convergence theory for surface partitioning problems
