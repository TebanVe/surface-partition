"""Γ-convergence relaxation pipeline (Phase 1).

Provides `run_relaxation()` — the multi-level PGD relaxation workflow —
and `compute_initial_perimeter()` for post-processing any base solution.
"""

import os
import gc
import time
import datetime
import logging
import getpass
import platform
import socket
import yaml
import h5py
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

from ..logging_config import get_logger
from ..mesh.interpolation import nearest_neighbor_interpolate
from ..optimization.exceptions import RefinementTriggered
from ..optimization.pgd_optimizer import ProjectedGradientOptimizer
from ..optimization.projection import (
    orthogonal_projection_iterative,
    create_initial_condition_with_projection,
)
from ..optimization.initialization import create_seeded_initial_condition
from ..partition.find_contours import (
    ContourAnalyzer, detect_dormant_cells, detect_area_imbalance,
    detect_disconnected_cells,
)
from ..partition.contour_partition import PartitionContour
from ..optimization.perimeter_optimizer import PerimeterOptimizer


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RelaxationConfig:
    """Configuration for the Γ-convergence relaxation pipeline.

    Focused subset of the parameters from src/config.py that are relevant
    to the PGD relaxation workflow.
    """
    n_partitions: int = 3
    seed: int = 42
    lambda_penalty: float = 1.0
    refinement_levels: int = 1
    max_iter: int = 1000
    use_analytic: bool = True
    use_discrete_area_for_constraints: bool = True

    refine_patience: int = 30
    refine_delta_energy: float = 1e-4
    refine_grad_tol: float = 1e-2
    refine_constraint_tol: float = 1e-2

    pgd_step0: float = 1.0
    pgd_armijo_c: float = 1e-4
    pgd_backtrack_rho: float = 0.5
    pgd_projection_max_iter: int = 100
    pgd_projection_tol: float = 1e-8
    run_log_frequency: int = 100
    h5_save_stride: int = 1
    h5_save_vars: List[str] = field(default_factory=lambda: ['x'])
    h5_always_save_first_last: bool = True
    refine_trigger_mode: str = 'full'
    refine_gnorm_patience: int = 30
    refine_gnorm_delta: float = 1e-4
    refine_feas_patience: int = 30
    refine_feas_delta: float = 1e-6
    enable_refinement_triggers: bool = True
    penalty_target_mode: str = 'fixed'
    penalty_eps: float = 1e-8
    profile: bool = False
    init_method: str = 'random'

    # Per-level resume checkpoint: after each level completes, write that level's
    # solution to solution/checkpoint_level{L}.h5 so a killed multi-day run can be
    # restarted with --resume-from instead of losing every completed level.
    # Only the most recent checkpoint is kept; all are removed once the final
    # solution is written.
    checkpoint_per_level: bool = True

    # Soft continuous equal-area constraint (A2). Default OFF -- byte-for-byte
    # backward compatible. ON replaces the iterative alternating projection
    # (93.3% of Phase 1 wall time) with a closed-form per-vertex simplex
    # projection, moving equal area into the objective as a quadratic penalty
    # of weight `soft_area_mu`. See the ProjectedGradientOptimizer docstring.
    soft_area_constraint: bool = False
    soft_area_mu: float = 0.0

    # Structure-based refinement trigger (stuck detector). Default OFF.
    # Fires when the winner-take-all label field has been frozen for
    # `struct_window` iterations, at a flip rate under `struct_rate_tol` per
    # iteration per vertex. Catches levels whose partition stopped changing but
    # whose energy still creeps above the absolute refine_delta_energy -- 27.1%
    # of the N=100 deliverable's Phase 1. See the ProjectedGradientOptimizer
    # docstring for why the window is long rather than the rate tight.
    struct_trigger_enabled: bool = False
    struct_window: int = 2000
    struct_rate_tol: float = 1e-6

    # Structure GATE: the same signal used the other way -- blocks refinement
    # while the label field is still moving, so a level cannot be stopped
    # mid-churn and have a half-settled configuration frozen by the next mesh.
    struct_gate_enabled: bool = False
    struct_gate_window: int = 500
    struct_gate_rate_tol: float = 1e-5
    # Both structure rules are evaluated on this stride -- the same granularity
    # they were validated at offline. Not a per-iteration rule.
    struct_sample_stride: int = 500

    @classmethod
    def from_yaml_dict(cls, params: dict) -> 'RelaxationConfig':
        """Construct from a YAML-loaded parameter dict.

        Accepts both the sectioned format (looks for a ``relaxation`` key)
        and the legacy flat format where all keys live at the top level.
        Unknown keys are silently ignored. Type coercion is applied so
        that values like ``1e-8`` (parsed as str by PyYAML) become float.
        """
        import dataclasses
        section = params.get('relaxation', params)
        field_map = {f.name: f for f in dataclasses.fields(cls)}
        filtered = {}
        for k, v in section.items():
            if k in field_map and v is not None:
                ft = field_map[k].type
                if ft == 'float' or ft is float:
                    v = float(v)
                elif ft == 'int' or ft is int:
                    v = int(v)
                elif ft == 'bool' or ft is bool:
                    v = bool(v)
                filtered[k] = v
        return cls(**filtered)


@dataclass
class RelaxationResult:
    """Structured result from the full relaxation pipeline."""
    energy: float
    initial_perimeter: Optional[float]
    solution_path: str
    output_dir: str
    elapsed_seconds: float
    converged: bool
    n_partitions: int
    seed: int
    lambda_penalty: float
    levels: list
    metadata: dict
    dormant_cells: dict = field(default_factory=dict)
    area_imbalance: dict = field(default_factory=dict)
    disconnected_cells: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a JSON-serializable summary for the AI agent layer."""
        return {
            'energy': self.energy,
            'initial_perimeter': self.initial_perimeter,
            'solution_path': self.solution_path,
            'output_dir': self.output_dir,
            'elapsed_seconds': self.elapsed_seconds,
            'converged': self.converged,
            'n_partitions': self.n_partitions,
            'seed': self.seed,
            'lambda_penalty': self.lambda_penalty,
        }


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def run_relaxation(provider, config: RelaxationConfig,
                   output_dir: str = None,
                   logger=None,
                   warm_start_path: Optional[str] = None) -> RelaxationResult:
    """Run the full multi-level Γ-convergence relaxation pipeline.

    Args:
        provider: A MeshProvider (e.g. TorusMeshProvider) that builds meshes.
        config: RelaxationConfig with all optimization parameters.
        output_dir: Directory for results. If None, auto-generated with
            timestamp and config parameters.
        logger: Logger instance. If None, uses get_logger(__name__).
        warm_start_path: Optional path to a prior solution HDF5 produced by
            run_relaxation(). When provided, the run resumes from the level
            after the last completed level stored in that file. The
            config.refinement_levels must be greater than completed_levels
            in the checkpoint.

    Returns:
        RelaxationResult with energy, solution path, metadata, etc.
    """
    if logger is None:
        logger = get_logger(__name__)

    surface = provider.surface_name()
    label1, label2 = provider.resolution_labels()
    n1_init, n2_init = provider.get_resolution()
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    v1_info, v2_info = provider.resolution_summary(config.refinement_levels)

    if output_dir is None:
        output_dir = (
            f"results/run_{timestamp}_surf{surface}"
            f"_npart{config.n_partitions}"
            f"_v1{label1}{v1_info}_v2{label2}{v2_info}"
            f"_lam{config.lambda_penalty}_seed{config.seed}"
        )
    os.makedirs(output_dir, exist_ok=True)

    solution_dir = os.path.join(output_dir, 'solution')
    traces_dir = os.path.join(output_dir, 'traces')
    logs_dir = os.path.join(output_dir, 'logs')
    for d in (solution_dir, traces_dir, logs_dir):
        os.makedirs(d, exist_ok=True)

    logfile_path = os.path.join(logs_dir, 'relaxation.log')
    root_logger = logging.getLogger()
    file_handler = logging.FileHandler(logfile_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    root_logger.addHandler(file_handler)

    prof = None
    if config.profile:
        from ..profiling import RelaxationProfilingState
        prof = RelaxationProfilingState(n_partitions=config.n_partitions)
    t_run_start = time.time()

    try:
        results = []
        levels_meta = []
        prev_vertices = None
        prev_x_opt = None
        start_level = 0

        ws = None
        if warm_start_path is not None:
            ws = _load_warm_start(warm_start_path)
            prev_vertices = ws['prev_vertices']
            prev_x_opt = ws['prev_x_opt']
            start_level = ws['completed_levels']
            logger.info(
                f"Warm start: resuming from level {start_level} "
                f"using checkpoint {warm_start_path}"
            )
            if start_level >= config.refinement_levels:
                raise ValueError(
                    f"Warm-start checkpoint already has {start_level} completed "
                    f"levels but config.refinement_levels="
                    f"{config.refinement_levels}. Increase refinement_levels "
                    f"to run additional levels beyond the checkpoint."
                )

        logger.info(
            f"Starting relaxation with {config.refinement_levels} "
            f"refinement levels"
        )

        for level in range(start_level, config.refinement_levels):
            logger.info("=" * 80)
            logger.info(
                f"Refinement Level {level+1}/{config.refinement_levels}"
            )
            logger.info("=" * 80)

            if prof is not None:
                prof.start_level(level)

            level_ctx = _setup_level(provider, config, level, logger,
                                     profile=prof)
            mesh = level_ctx['mesh']

            if prof is not None:
                prof.set_level_mesh_stats(
                    n_vertices=int(mesh.vertices.shape[0]),
                    n_triangles=int(mesh.faces.shape[0]),
                    nnz_K=int(mesh.K.nnz),
                    nnz_M=int(mesh.M.nnz),
                    epsilon=float(level_ctx['epsilon']),
                )

            x0 = _create_initial_condition(
                mesh, config, level, prev_vertices, prev_x_opt,
                profile=prof, logger=logger
            )

            level_result = _optimize_level(
                level_ctx['optimizer'], config, x0, level,
                provider, traces_dir, logger, profile=prof
            )
            level_result['epsilon'] = level_ctx['epsilon']

            results.append(level_result)
            level_info = {
                'level': level,
                label1: int(provider.get_resolution()[0]),
                label2: int(provider.get_resolution()[1]),
                'N': int(level_ctx['N']),
                'v_sum': float(np.sum(mesh.v)),
                'epsilon': float(level_ctx['epsilon']),
                'files': level_result['optimizer_files'],
                'iters': {
                    'num_iterations': int(level_result['iterations'])
                },
            }
            if hasattr(provider, 'n_grid_z'):
                level_info['ngz'] = int(provider.n_grid_z)
            levels_meta.append(level_info)

            logger.info(f"Results for level {level+1}:")
            logger.info(f"  Energy: {level_result['energy']:.6e}")
            logger.info(f"  Iterations: {level_result['iterations']}")
            logger.info(f"  Time: {level_result['elapsed']:.2f}s")
            logger.info(f"  Success: {level_result['success']}")

            if prof is not None:
                prof.finalize_level()

            prev_vertices = mesh.vertices.copy()
            prev_x_opt = level_result['x_opt'].copy()

            if level < config.refinement_levels - 1:
                # Persist this level before the mesh is freed, so an
                # interrupted run resumes here instead of from level 0. The
                # final level needs none: _save_solution() follows the loop.
                if config.checkpoint_per_level:
                    _save_level_checkpoint(
                        mesh, level_result['x_opt'], x0, config, provider,
                        levels_meta, level + 1, solution_dir, logger,
                        extra_attrs={'checkpoint': True},
                    )

                # Release this level's FEM matrices before building the finer
                # one. K/M/v are read-only properties; the backing store is
                # mass_matrix/stiffness_matrix (setting mesh.K/M silently raised
                # AttributeError and freed nothing). Also drop the level_ctx
                # references so the finished mesh + optimizer are collectable
                # now, not one level late (level_ctx is rebound next iteration).
                mesh.mass_matrix = None
                mesh.stiffness_matrix = None
                del level_ctx['optimizer']
                level_ctx['mesh'] = None
                del mesh
                gc.collect()

        if prof is not None:
            prof.total_wall_s = time.time() - t_run_start
            prof.finalize()
            _write_timing_profile(prof, solution_dir, provider, config, logger)

        final = results[-1]

        final_densities = final['x_opt'].reshape(
            mesh.vertices.shape[0], config.n_partitions
        )
        dormant = detect_dormant_cells(final_densities)
        area_imbalance = detect_area_imbalance(
            final_densities, mesh.v, config.n_partitions
        )
        disconnected = detect_disconnected_cells(
            final_densities, mesh.faces, mesh.v
        )

        solution_path = _save_solution(
            mesh, final['x_opt'], x0, config, provider,
            levels_meta, timestamp, solution_dir,
            completed_levels=start_level + len(levels_meta),
        )
        _cleanup_level_checkpoints(solution_dir, logger)

        initial_perimeter = compute_initial_perimeter(
            solution_path, config.n_partitions, logger
        )

        metadata = _collect_metadata(
            config, provider, results, levels_meta, mesh,
            timestamp, solution_path, logfile_path, initial_perimeter,
            warm_start_path=warm_start_path, dormant_cells=dormant,
            area_imbalance=area_imbalance,
            disconnected_cells=disconnected,
        )

        with open(os.path.join(solution_dir, 'metadata.yaml'), 'w') as f:
            yaml.dump(metadata, f)

        logger.info("Refinement Summary:")
        logger.info("=" * 80)
        logger.info(
            " Level    Mesh Size       Energy Iterations   Time (s)"
        )
        logger.info("-" * 80)
        for i, res in enumerate(results):
            lv = levels_meta[i]
            level_num = int(lv.get('level', i)) + 1
            mesh_size = (
                f"{int(lv.get(label1, 0))}x{int(lv.get(label2, 0))}"
            )
            logger.info(
                f"{level_num:6d} {mesh_size:>11s} {res['energy']:12.6e} "
                f"{res['iterations']:10d} {res['elapsed']:10.2f}"
            )
        if initial_perimeter is not None:
            logger.info(
                f"Initial perimeter (contour extraction): "
                f"{initial_perimeter:.10f}"
            )
        else:
            logger.info("Initial perimeter: could not be computed")

        _warn_if_dormant_cells(dormant, levels_meta, config, logger)
        _warn_if_area_imbalance(area_imbalance, config, logger)
        _warn_if_disconnected_cells(disconnected, config, logger)

        overall_success = all(r['success'] for r in results)
        total_elapsed = sum(r['elapsed'] for r in results)

        return RelaxationResult(
            energy=float(final['energy']),
            initial_perimeter=(
                float(initial_perimeter)
                if initial_perimeter is not None else None
            ),
            solution_path=os.path.abspath(solution_path),
            output_dir=os.path.abspath(output_dir),
            elapsed_seconds=total_elapsed,
            converged=overall_success,
            n_partitions=config.n_partitions,
            seed=config.seed,
            lambda_penalty=config.lambda_penalty,
            levels=results,
            metadata=metadata,
            dormant_cells=dormant,
            area_imbalance=area_imbalance,
            disconnected_cells=disconnected,
        )

    finally:
        root_logger.removeHandler(file_handler)
        file_handler.close()


def compute_initial_perimeter(solution_path: str,
                              n_partitions: int,
                              logger=None) -> Optional[float]:
    """Compute perimeter from a base solution via contour extraction.

    This is independently useful as a post-processing step on any base
    solution HDF5 file.

    Returns the perimeter value, or None if computation fails.
    """
    if logger is None:
        logger = get_logger(__name__)
    try:
        analyzer = ContourAnalyzer(solution_path)
        analyzer.load_results(use_initial_condition=False)

        indicators = analyzer.compute_indicator_functions()
        _, boundary_topology = analyzer.extract_contours_with_topology()

        from ..mesh.tri_mesh import TriMesh
        with h5py.File(solution_path, 'r') as f:
            vertices = np.array(f['vertices'])
            faces = np.array(f['faces'])
        mesh = TriMesh(vertices, faces)
        mesh.compute_matrices()

        partition = PartitionContour(
            mesh, indicators, boundary_topology=boundary_topology
        )

        total_area = float(np.sum(mesh.v))
        target_area = total_area / n_partitions
        perim_optimizer = PerimeterOptimizer(partition, mesh, target_area)
        x0_perim = partition.get_variable_vector()
        initial_perimeter = perim_optimizer.objective(x0_perim)

        logger.info(
            f"Initial perimeter (contour extraction): "
            f"{initial_perimeter:.10f}"
        )
        return initial_perimeter
    except Exception as e:
        logger.warning(f"Could not compute initial perimeter: {e}")
        return None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_warm_start(solution_path: str) -> dict:
    """Load resume state from a prior solution HDF5.

    Returns dict with keys:
        'prev_x_opt'        np.ndarray — optimized solution from last completed level
        'prev_vertices'     np.ndarray — mesh vertices from last completed level
        'completed_levels'  int        — number of levels already completed
    Raises ValueError if the file is missing required datasets or attributes.
    """
    with h5py.File(solution_path, 'r') as f:
        if 'x_opt' not in f or 'vertices' not in f:
            raise ValueError(
                f"Warm-start file {solution_path} is missing 'x_opt' or "
                f"'vertices' datasets. Was it produced by run_relaxation()?"
            )
        if 'completed_levels' not in f.attrs:
            raise ValueError(
                f"Warm-start file {solution_path} has no 'completed_levels' "
                f"attribute. Re-run the original relaxation to regenerate it."
            )
        return {
            'prev_x_opt': np.array(f['x_opt']),
            'prev_vertices': np.array(f['vertices']),
            'completed_levels': int(f.attrs['completed_levels']),
        }


def _setup_level(provider, config, level, logger, profile=None) -> dict:
    """Build mesh and PGD optimizer for one refinement level."""
    n1, n2 = provider.get_initial_resolution()
    dn1, dn2 = provider.get_resolution_increment()
    n1 = n1 + level * dn1
    n2 = n2 + level * dn2
    provider.set_resolution(n1, n2)

    if profile is not None:
        _t_mb = time.perf_counter()
    mesh = provider.build()
    if profile is not None:
        profile.record('mesh_build', time.perf_counter() - _t_mb)
    mesh.compute_matrices(_prof=profile)
    stats = mesh.get_mesh_statistics()
    epsilon = (np.sqrt(stats['mean_triangle_area'])
               if stats['mean_triangle_area'] > 0 else 1e-2)
    logger.info(
        f"epsilon set to sqrt(mean_triangle_area) = {epsilon:.3e}"
    )

    if config.use_discrete_area_for_constraints:
        total_area = float(np.sum(mesh.v))
    else:
        theoretical = getattr(provider, 'theoretical_total_area', None)
        val = provider.theoretical_total_area() if callable(theoretical) else None
        total_area = float(val) if val is not None else float(np.sum(mesh.v))

    optimizer = ProjectedGradientOptimizer(
        K=mesh.K, M=mesh.M, v=mesh.v,
        n_partitions=config.n_partitions,
        epsilon=epsilon, total_area=total_area,
        lambda_penalty=config.lambda_penalty,
        refine_patience=int(config.refine_patience),
        refine_delta_energy=float(config.refine_delta_energy),
        refine_grad_tol=float(config.refine_grad_tol),
        refine_constraint_tol=float(config.refine_constraint_tol),
        soft_area_constraint=bool(config.soft_area_constraint),
        soft_area_mu=float(config.soft_area_mu),
        struct_trigger_enabled=bool(config.struct_trigger_enabled),
        struct_window=int(config.struct_window),
        struct_rate_tol=float(config.struct_rate_tol),
        struct_gate_enabled=bool(config.struct_gate_enabled),
        struct_gate_window=int(config.struct_gate_window),
        struct_gate_rate_tol=float(config.struct_gate_rate_tol),
        struct_sample_stride=int(config.struct_sample_stride),
        logger=logger,
    )
    if hasattr(optimizer, 'penalty_target_mode'):
        optimizer.penalty_target_mode = str(config.penalty_target_mode)
    if hasattr(optimizer, 'penalty_eps'):
        optimizer.penalty_eps = float(config.penalty_eps)

    return {
        'mesh': mesh,
        'optimizer': optimizer,
        'epsilon': epsilon,
        'total_area': total_area,
        'N': len(mesh.v),
        'n1': n1,
        'n2': n2,
    }


def _create_initial_condition(mesh, config, level,
                              prev_vertices, prev_x_opt, profile=None,
                              logger=None) -> np.ndarray:
    """Create initial condition (level-0 builder, interpolated otherwise).

    Level 0 (``prev_vertices is None``) dispatches on ``config.init_method``:
    ``'seeded'`` builds a Voronoi initial condition; any other value (default
    ``'random'``) uses the legacy uniform-random builder, warning if the value
    is unrecognized. Finer levels always warm-start by interpolation.
    """
    if logger is None:
        logger = get_logger(__name__)
    N = len(mesh.v)
    if prev_vertices is None:
        if profile is not None:
            _t_ic = time.perf_counter()
        method = str(config.init_method).lower()
        if method == 'seeded':
            x0 = create_seeded_initial_condition(
                mesh, config.n_partitions, mesh.v,
                seed=config.seed, logger=logger
            )
        else:
            if method != 'random':
                logger.warning(
                    f"Unrecognized init_method '{config.init_method}'; "
                    f"falling back to 'random'."
                )
            x0 = create_initial_condition_with_projection(
                N, config.n_partitions, mesh.v,
                seed=config.seed, method="iterative", _prof=profile
            )
        if profile is not None:
            profile.record('init_condition', time.perf_counter() - _t_ic)
        return x0

    x0 = nearest_neighbor_interpolate(
        prev_vertices, mesh.vertices, prev_x_opt, config.n_partitions, _prof=profile
    )
    A = x0.reshape(N, config.n_partitions)
    area_targets = (np.sum(mesh.v) / config.n_partitions
                    * np.ones(config.n_partitions))
    A = orthogonal_projection_iterative(
        A, np.ones(config.n_partitions), area_targets,
        mesh.v, max_iter=100, tol=1e-8, _prof=profile
    )
    return A.flatten()


def _optimize_level(optimizer, config, x0, level,
                    provider, traces_dir, logger, profile=None) -> dict:
    """Run PGD optimizer for one level. Handles RefinementTriggered."""
    label1, label2 = provider.resolution_labels()
    n1, n2 = provider.get_resolution()

    start = time.time()
    try:
        x_opt, success = optimizer.optimize(
            x0=x0,
            maxiter=config.max_iter,
            step0=float(config.pgd_step0),
            armijo_c=float(config.pgd_armijo_c),
            backtrack_rho=float(config.pgd_backtrack_rho),
            projection_max_iter=int(config.pgd_projection_max_iter),
            projection_tol=float(config.pgd_projection_tol),
            results_dir=traces_dir,
            run_name=(
                f"pgd_part{config.n_partitions}"
                f"_v1{label1}{n1}_v2{label2}{n2}_level{level}"
            ),
            is_mesh_refinement=(level > 0),
            data_save_stride=int(config.h5_save_stride),
            data_save_vars=config.h5_save_vars,
            save_first_last=bool(config.h5_always_save_first_last),
            log_frequency=int(config.run_log_frequency),
            refine_trigger_mode=str(config.refine_trigger_mode),
            refine_gnorm_patience=int(config.refine_gnorm_patience),
            refine_gnorm_delta=float(config.refine_gnorm_delta),
            refine_feas_patience=int(config.refine_feas_patience),
            refine_feas_delta=float(config.refine_feas_delta),
            enable_refinement_triggers=bool(config.enable_refinement_triggers),
            profile=profile,
        )
    except RefinementTriggered:
        logger.info(f"Refinement triggered early at level {level+1}")
        x_opt = getattr(optimizer, 'prev_x', x0)
        success = False
    elapsed = time.time() - start

    energy = optimizer.compute_energy(x_opt)
    n_iterations = int(optimizer.log.get('iterations', 0))

    return {
        'level': level,
        'x_opt': x_opt,
        'energy': energy,
        'iterations': n_iterations,
        'elapsed': elapsed,
        'success': success,
        'optimizer_files': {
            'internal_data': os.path.abspath(
                getattr(optimizer, 'internal_data_file', '')
            ),
            'summary': os.path.abspath(
                getattr(optimizer, 'summary_file', '')
            ),
        },
    }


def _write_timing_profile(prof, solution_dir, provider, config, logger):
    """Write the per-run Phase 1 timing_profile.yaml into the solution dir."""
    tp_path = os.path.join(solution_dir, 'timing_profile.yaml')
    data = prof.to_yaml_dict(
        surface=provider.surface_name(),
        n_partitions=config.n_partitions,
        refinement_levels=config.refinement_levels,
    )
    with open(tp_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Timing profile written to: {tp_path}")


def _write_solution_h5(path, mesh, x_opt, x0, config, provider,
                       levels_meta, completed_levels,
                       extra_attrs=None) -> None:
    """Write a solution/checkpoint HDF5 at ``path``.

    ``completed_levels`` is the ABSOLUTE number of levels completed across the
    whole ladder (start_level + levels run in this invocation), which is what
    ``_load_warm_start`` resumes from — not ``len(levels_meta)``, which counts
    only the levels this invocation ran.
    """
    surface = provider.surface_name()
    label1, label2 = provider.resolution_labels()

    with h5py.File(path, 'w') as f:
        f.create_dataset('x_opt', data=x_opt)
        f.create_dataset('x0', data=x0)
        f.create_dataset('vertices', data=mesh.vertices)
        f.create_dataset('faces', data=mesh.faces, dtype='i4')
        f.attrs['n_partitions'] = config.n_partitions
        f.attrs['surface'] = surface
        f.attrs['resolution_labels'] = [label1, label2]
        if levels_meta:
            last_level = levels_meta[-1]
            f.attrs['var1'] = int(
                last_level.get(label1, provider.get_resolution()[0])
            )
            f.attrs['var2'] = int(
                last_level.get(label2, provider.get_resolution()[1])
            )
        else:
            f.attrs['var1'] = int(provider.get_resolution()[0])
            f.attrs['var2'] = int(provider.get_resolution()[1])
        f.attrs['lambda_penalty'] = float(config.lambda_penalty)
        f.attrs['seed'] = int(config.seed)
        f.attrs['optimizer'] = 'PGD'
        f.attrs['use_analytic'] = bool(config.use_analytic)
        f.attrs['completed_levels'] = int(completed_levels)
        for k, v in (extra_attrs or {}).items():
            f.attrs[k] = v


def _save_solution(mesh, x_opt, x0, config, provider,
                   levels_meta, timestamp, solution_dir,
                   completed_levels, extra_attrs=None) -> str:
    """Write final solution HDF5 file. Returns path."""
    surface = provider.surface_name()
    label1, label2 = provider.resolution_labels()
    v1_info, v2_info = provider.resolution_summary(config.refinement_levels)

    solution_path = os.path.join(
        solution_dir,
        f"surface_part{config.n_partitions}_surf{surface}"
        f"_v1{label1}{v1_info}_v2{label2}{v2_info}"
        f"_lam{config.lambda_penalty}_seed{config.seed}_{timestamp}.h5"
    )
    _write_solution_h5(
        solution_path, mesh, x_opt, x0, config, provider,
        levels_meta, completed_levels, extra_attrs=extra_attrs
    )
    return solution_path


def _save_level_checkpoint(mesh, x_opt, x0, config, provider, levels_meta,
                           completed_levels, solution_dir, logger,
                           extra_attrs=None) -> Optional[str]:
    """Write the per-level resume checkpoint, replacing any earlier one.

    Written to a temporary file and moved into place, so a run killed mid-write
    still leaves the previous checkpoint intact. Only the newest checkpoint is
    kept; a failure here is logged and swallowed — losing a checkpoint must
    never kill a multi-day run.
    """
    path = os.path.join(
        solution_dir, f"checkpoint_level{completed_levels:02d}.h5"
    )
    tmp_path = path + '.tmp'
    try:
        _write_solution_h5(
            tmp_path, mesh, x_opt, x0, config, provider,
            levels_meta, completed_levels, extra_attrs=extra_attrs
        )
        os.replace(tmp_path, path)
    except Exception as e:
        logger.warning(f"Could not write level checkpoint {path}: {e}")
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return None

    _cleanup_level_checkpoints(solution_dir, logger, keep=path)
    logger.info(
        f"Level checkpoint written: {path} "
        f"(resume with --resume-from {path})"
    )
    return path


def _cleanup_level_checkpoints(solution_dir, logger, keep=None) -> None:
    """Remove per-level checkpoints in ``solution_dir`` except ``keep``."""
    import glob
    for p in glob.glob(os.path.join(solution_dir, 'checkpoint_level*.h5')):
        if keep is not None and os.path.abspath(p) == os.path.abspath(keep):
            continue
        try:
            os.remove(p)
        except OSError as e:
            logger.warning(f"Could not remove stale checkpoint {p}: {e}")


def _warn_if_dormant_cells(dormant, levels_meta, config, logger) -> None:
    """Log a prominent warning when the partition has dormant cells.

    A dormant solution satisfies the equal-area and sum-to-one constraints but
    loses regions under winner-take-all, so it is not a usable N-region
    partition. The root cause is an under-resolved coarsest mesh, so the
    warning points the user at the initial mesh resolution.
    """
    dead = dormant.get('dead', [])
    weak = dormant.get('weak', [])
    if not dead and not weak:
        return

    coarsest_N = int(levels_meta[0].get('N', 0)) if levels_meta else 0
    per_cell = (coarsest_N / config.n_partitions) if config.n_partitions else 0.0

    logger.warning("=" * 80)
    logger.warning(
        f"DORMANT CELLS DETECTED - this is NOT a valid {config.n_partitions}-region "
        f"partition."
    )
    logger.warning(
        "The solution is a consistent continuous minimizer (equal areas and "
        "sum-to-one are satisfied), but some density columns never win the "
        "per-vertex argmax, so the discrete partition is missing regions."
    )
    if dead:
        logger.warning(
            f"  Dead cells (0 winning vertices): {dead}  ->  effective regions: "
            f"{dormant.get('n_effective')}/{dormant.get('n_cells')}"
        )
    if weak:
        logger.warning(
            f"  Weak cells (peak density < {dormant.get('weak_threshold')}): {weak}"
        )
    logger.warning(
        f"Likely cause: coarsest-level mesh too sparse (level 1 had {coarsest_N} "
        f"vertices = {per_cell:.0f}/cell; guideline >= ~400/cell)."
    )
    logger.warning(
        "Fix: increase the initial mesh resolution (n_theta/n_phi) or "
        "refinement_levels, then re-run."
    )
    logger.warning("=" * 80)


def _warn_if_area_imbalance(area_imbalance, config, logger) -> None:
    """Log a prominent warning when discrete cell areas are grossly unequal.

    Phase 1 satisfies the *continuous* equal-area constraint, but Phase 2
    inherits the *discrete* winner-take-all areas. When those are far from equal
    (a diffuse "runt" cell that is not dormant), Phase 2's equal-area constraint
    starts infeasible: perimeter rises every iteration and IPOPT stalls at a
    point of local infeasibility. Distinct from the dormant-cell warning, which
    only catches cells that vanish entirely. See
    docs/reference/winner_take_all_partition_gap.md.
    """
    if not area_imbalance or not area_imbalance.get('imbalanced'):
        return

    worst = area_imbalance.get('worst_cell')
    logger.warning("=" * 80)
    logger.warning(
        f"DISCRETE AREA IMBALANCE - Phase 2 equal-area constraint will start "
        f"infeasible for this {config.n_partitions}-region solution."
    )
    logger.warning(
        "Phase 1's continuous equal-area constraint is satisfied, but the "
        "winner-take-all (discrete) cell areas are not equal, so Phase 2 "
        "refinement is likely to raise the perimeter and converge to a point "
        "of local infeasibility rather than reduce it."
    )
    logger.warning(
        f"  Worst cell {worst}: discrete area deviates "
        f"{area_imbalance.get('worst_rel_dev', 0.0) * 100:.1f}% from target "
        f"(abs {area_imbalance.get('worst_abs_dev', 0.0):.4g} = the Phase 2 "
        f"iter-0 constraint violation)."
    )
    logger.warning(
        f"  {area_imbalance.get('n_imbalanced')} cell(s) exceed the "
        f"{area_imbalance.get('rel_threshold', 0.0) * 100:.0f}% threshold: "
        f"{area_imbalance.get('imbalanced')}"
    )
    logger.warning(
        f"  Area spread (std/target) = "
        f"{area_imbalance.get('area_std_over_target', 0.0) * 100:.1f}%."
    )
    logger.warning(
        "Likely cause: at high N cells shrink toward the mesh-tied interface "
        "width, so winner-take-all loses a large fraction of a cell's area. A "
        "finer mesh does NOT reliably help; try other seeds and/or tune "
        "lambda_penalty. See docs/reference/winner_take_all_partition_gap.md."
    )
    logger.warning("=" * 80)


def _warn_if_disconnected_cells(disconnected, config, logger) -> None:
    """Log a prominent warning when cells split into disconnected pieces.

    A cell whose winner-take-all territory breaks into two or more disconnected
    islands passes both the dormant-cell and discrete-area-imbalance checks -- its
    pieces sum to the target area and each is crisp -- yet it is not a physical
    minimal-perimeter cell, since optimal cells are connected. This is a
    relaxation local minimum: nothing in the energy, the WTA balance term, or the
    discrete-area trim penalizes disconnection, so an area-balanced fragmented
    cell is a stable fixed point. Distinct from the dormant and imbalance
    warnings, both of which pass for a fragmented cell. See
    docs/reference/winner_take_all_partition_gap.md.
    """
    if not disconnected or not disconnected.get('fragmented'):
        return

    worst = disconnected.get('worst_cell')
    logger.warning("=" * 80)
    logger.warning(
        f"DISCONNECTED CELLS - {disconnected.get('n_fragmented')} cell(s) of this "
        f"{config.n_partitions}-region solution split into disconnected pieces."
    )
    logger.warning(
        "Each flagged cell's winner-take-all territory breaks into 2+ islands on "
        "the surface. The pieces sum to the target area and are crisp, so the "
        "dormant and area-imbalance checks pass -- but a minimal-perimeter cell "
        "is connected, so this is a non-physical relaxation local minimum."
    )
    logger.warning(
        f"  Worst cell {worst}: stray (non-largest) pieces total "
        f"{disconnected.get('worst_stray_rel', 0.0) * 100:.1f}% of a cell's target "
        f"area (abs {disconnected.get('worst_stray_abs', 0.0):.4g})."
    )
    logger.warning(f"  Fragmented cells: {disconnected.get('fragmented')}")
    for d in disconnected.get('details', [])[:8]:
        logger.warning(
            f"    cell {d['cell']}: {d['n_components']} pieces, areas "
            f"{[round(a, 5) for a in d['component_areas']]}"
        )
    logger.warning(
        "Likely cause: a cell pinched apart on the coarse mesh (large epsilon) "
        "and refinement could not reconnect it; connectivity is not in the "
        "objective. Try a different seed, or a connectivity-repair post-process. "
        "Do NOT hand a fragmented cell to Phase 2 (it yields multi-loop contours)."
    )
    logger.warning("=" * 80)


def _collect_metadata(config, provider, results, levels_meta, mesh,
                      timestamp, solution_path, logfile_path,
                      initial_perimeter,
                      warm_start_path: Optional[str] = None,
                      dormant_cells: Optional[dict] = None,
                      area_imbalance: Optional[dict] = None,
                      disconnected_cells: Optional[dict] = None) -> dict:
    """Assemble comprehensive metadata dictionary."""
    surface = provider.surface_name()
    label1, label2 = provider.resolution_labels()

    cum = 0
    for lm in levels_meta:
        ni = int(lm.get('iters', {}).get('num_iterations', 0))
        lm['iters']['start_index_global'] = int(cum)
        lm['iters']['end_index_global'] = int(cum + max(ni - 1, 0))
        cum += ni

    theoretical_total_area = None
    if (hasattr(provider, 'theoretical_total_area')
            and callable(provider.theoretical_total_area)):
        val = provider.theoretical_total_area()
        if val is not None:
            theoretical_total_area = float(val)

    surface_params = {}
    for attr in ('R', 'r', 'a', 'b', 'c'):
        if hasattr(provider, attr):
            surface_params[attr] = float(getattr(provider, attr))

    meta = {
        'input_parameters': {
            'refinement_levels': int(config.refinement_levels),
            'use_analytic': bool(config.use_analytic),
            'lambda_penalty': float(config.lambda_penalty),
            'seed': int(config.seed),
            'surface': surface,
            'resolution_labels': [label1, label2],
            'resolution_summary': list(
                provider.resolution_summary(config.refinement_levels)
            ),
            'use_discrete_area_for_constraints': bool(
                config.use_discrete_area_for_constraints
            ),
            'n_partitions': int(config.n_partitions),
            'optimizer_type': 'pgd',
            'surface_params': surface_params,
            'run_log_frequency': int(config.run_log_frequency),
            'h5_save_stride': int(config.h5_save_stride),
            'h5_save_vars': list(config.h5_save_vars),
            'h5_always_save_first_last': bool(config.h5_always_save_first_last),
            'refine_trigger_mode': str(config.refine_trigger_mode),
            'refine_gnorm_patience': int(config.refine_gnorm_patience),
            'refine_gnorm_delta': float(config.refine_gnorm_delta),
            'refine_feas_patience': int(config.refine_feas_patience),
            'refine_feas_delta': float(config.refine_feas_delta),
            'enable_refinement_triggers': bool(config.enable_refinement_triggers),
        },
        'levels': levels_meta,
        'final_mesh_stats': mesh.get_mesh_statistics(),
        'final_epsilon': float(results[-1]['epsilon']),
        'final_energy': float(results[-1]['energy']),
        'final_iterations': int(results[-1]['iterations']),
        'run_time_seconds': float(results[-1]['elapsed']),
        'success': bool(results[-1]['success']),
        'datetime': timestamp,
        'user': getpass.getuser(),
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'solution_path': os.path.abspath(solution_path),
        'optimizer': 'PGD',
        'theoretical_total_area': theoretical_total_area,
        'initial_perimeter': (
            float(initial_perimeter) if initial_perimeter is not None else None
        ),
        'files': {
            'run_log': os.path.abspath(logfile_path),
            'solution_path': os.path.abspath(solution_path),
        },
        'warm_start_path': (
            os.path.abspath(warm_start_path)
            if warm_start_path is not None else None
        ),
        'dormant_cells': dormant_cells,
        'area_imbalance': area_imbalance,
        'disconnected_cells': disconnected_cells,
    }
    return meta
