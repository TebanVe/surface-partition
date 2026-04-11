"""
Pipeline orchestrator for iterative perimeter refinement.

Provides decomposed, independently callable stage methods (optimize, detect,
migrate, export_checkpoint) plus a convenience run_refinement_loop() that
chains them with automatic file-type detection and resume support.
"""

import os
import re
import time
import dataclasses
import numpy as np
import h5py
import yaml
from dataclasses import dataclass
from typing import Optional, Dict, List

from ..logging_config import get_logger
from ..mesh.tri_mesh import TriMesh
from ..mesh.mesh_topology import MeshTopology
from ..partition.find_contours import ContourAnalyzer
from ..partition.contour_partition import PartitionContour
from ..partition.perimeter_calculator import PerimeterCalculator
from ..partition.steiner_handler import SteinerHandler
from ..optimization.perimeter_optimizer import PerimeterOptimizer
from ..migration.migration_orchestrator import MigrationOrchestrator, MigrationConfig
from ..migration.migration_types import DetectionResult, MigrationResult
from .io import (
    load_partition_from_base_file,
    load_partition_from_refined_file,
    detect_run_layout,
)


def detect_file_type(path):
    """Auto-detect whether an HDF5 file is a base solution or iteration checkpoint.

    Returns:
        'base' -- file contains x_opt, enter at Optimize
        'checkpoint_pending' -- iteration file with pending_migration=True, enter at Migrate
        'checkpoint_converged' -- iteration file with pending_migration=False, done
    """
    with h5py.File(path, 'r') as f:
        if 'x_opt' in f:
            return 'base'
        pending = bool(f.attrs.get('pending_migration', True))
        return 'checkpoint_pending' if pending else 'checkpoint_converged'


def build_campaign_name(config):
    """Construct a campaign directory name from a RefinementConfig.

    Encodes the method and boundary tolerance as the base, plus optional
    IPOPT-specific suffixes for non-default settings.

    Examples:
        slsqp_btol0.001
        ipopt_btol0.001_lbfgs20_hess
        trust-constr_btol0.01_midpoint
    """
    parts = [config.method.lower(), f"btol{config.boundary_tol}"]

    if config.method.lower() == 'ipopt':
        if config.lbfgs_memory != 6:
            parts.append(f"lbfgs{config.lbfgs_memory}")
        if config.exact_hessian:
            parts.append("hess")
        if config.best_iterate:
            parts.append("bestiter")
        if config.allow_partial_convergence:
            parts.append("partial")

    if config.distance_preservation != 'preserve':
        if config.distance_preservation == 'midpoint':
            parts.append("midpoint")
        else:
            parts.append(f"dist{config.distance_preservation}")

    return "_".join(parts)


def derive_output_paths(solution_path, file_type, config=None, output_override=None):
    """Derive the output directory for checkpoint files.

    Supports both the structured layout (solution/ + refinement/{campaign}/)
    and the legacy flat layout.

    When *config* is provided and the layout is structured, output paths
    are placed under ``refinement/{campaign_name}/`` inside the run directory.

    Args:
        solution_path: Path to the input HDF5 file.
        file_type: One of 'base', 'checkpoint_pending', 'checkpoint_converged'.
        config: Optional RefinementConfig; required for structured-layout
            base files to determine the campaign directory name.
        output_override: If provided, use this directory as the output.

    Returns:
        output_dir: The directory where checkpoint files should be written.
    """
    if output_override is not None:
        return os.path.dirname(output_override) or '.'

    layout, run_dir = detect_run_layout(solution_path)

    if layout == 'structured' and file_type == 'base' and config is not None:
        campaign_name = build_campaign_name(config)
        return os.path.join(run_dir, 'refinement', campaign_name)

    if layout == 'structured' and file_type in ('checkpoint_pending', 'checkpoint_converged'):
        return os.path.dirname(os.path.abspath(solution_path))

    # Legacy flat layout — put checkpoints next to the solution file
    return os.path.dirname(os.path.abspath(solution_path))


def _read_iteration_number(checkpoint_path):
    """Read iteration_number from an HDF5 checkpoint's attributes.

    Falls back to filename regex for old-style filenames, then to 1.
    """
    try:
        with h5py.File(checkpoint_path, 'r') as f:
            val = f.attrs.get('iteration_number')
            if val is not None:
                return int(val)
    except Exception:
        pass
    m = re.search(r'iteration[_]?(\d+)', os.path.basename(checkpoint_path))
    return int(m.group(1)) if m else 1


def _check_indicator_vp_consistency(partition, mesh, logger):
    """Roundtrip consistency check: verify that partition.indicator_functions
    produces exactly the same set of boundary edges (and thus VP count) as
    the live partition.variable_points.

    Returns True if consistent, False if a mismatch is detected.
    """
    logger.info("  Running indicator_functions <-> VP roundtrip consistency check...")

    vertex_labels = np.argmax(partition.indicator_functions, axis=1)
    reconstructed_edges = set()
    for face in mesh.faces:
        v0, v1, v2 = face
        l0, l1, l2 = vertex_labels[v0], vertex_labels[v1], vertex_labels[v2]
        if l0 != l1:
            reconstructed_edges.add(tuple(sorted((v0, v1))))
        if l1 != l2:
            reconstructed_edges.add(tuple(sorted((v1, v2))))
        if l0 != l2:
            reconstructed_edges.add(tuple(sorted((v0, v2))))

    live_edges = {vp.edge for vp in partition.variable_points if getattr(vp, 'active', True)}

    only_in_reconstructed = reconstructed_edges - live_edges
    only_in_live = live_edges - reconstructed_edges

    if only_in_reconstructed or only_in_live:
        logger.error("  CONSISTENCY CHECK FAILED: indicator_functions do not match live VPs")
        logger.error(f"    Live VPs      : {len(live_edges)} unique edges")
        logger.error(f"    Reconstructed : {len(reconstructed_edges)} unique edges from indicator_functions")
        logger.error(f"    Edges in indicator_functions but NOT in live VPs ({len(only_in_reconstructed)}): "
                     f"{sorted(only_in_reconstructed)[:10]}{'...' if len(only_in_reconstructed) > 10 else ''}")
        logger.error(f"    Edges in live VPs but NOT in indicator_functions ({len(only_in_live)}): "
                     f"{sorted(only_in_live)[:10]}{'...' if len(only_in_live) > 10 else ''}")
        logger.error("  *** This will cause a VP count mismatch when the file is reloaded! ***")
        return False
    else:
        logger.info(f"  Consistent: {len(live_edges)} boundary edges match in both live VPs and indicator_functions")
        return True


def _coerce(value, type_hint):
    """Coerce a YAML-loaded value to the expected dataclass field type.

    Handles cases like ``1e-7`` being parsed as str instead of float.
    """
    if type_hint == 'float' or type_hint is float:
        return float(value)
    if type_hint == 'int' or type_hint is int:
        return int(value)
    if type_hint == 'bool' or type_hint is bool:
        return bool(value)
    return value


@dataclass
class RefinementConfig:
    """Configuration for the iterative refinement pipeline."""
    max_iterations: int = 10
    max_opt_iter: int = 1000
    tolerance: float = 1e-7
    boundary_tol: float = 1e-3
    method: str = 'SLSQP'
    lbfgs_memory: int = 6
    best_iterate: bool = False
    exact_hessian: bool = False
    allow_partial_convergence: bool = False
    use_vectorized: bool = True
    save_iterations: bool = False
    distance_preservation: str = 'preserve'

    @classmethod
    def from_yaml_dict(cls, params: dict) -> 'RefinementConfig':
        """Construct from a YAML-loaded parameter dict.

        Accepts both the sectioned format (looks for a ``refinement`` key)
        and a flat dict of refinement fields. Unknown keys are ignored.
        Type coercion is applied based on the dataclass field types so
        that values like ``1e-7`` (parsed as str by PyYAML) become float.
        """
        section = params.get('refinement', params)
        field_map = {f.name: f for f in dataclasses.fields(cls)}
        filtered = {}
        for k, v in section.items():
            if k in field_map and v is not None:
                filtered[k] = _coerce(v, field_map[k].type)
        return cls(**filtered)


class PipelineOrchestrator:
    """Decomposed iterative refinement pipeline.

    Exposes individual stage methods that can be called independently,
    plus a convenience run_refinement_loop() that chains them.
    """

    def __init__(self, mesh: TriMesh, partition: PartitionContour,
                 config: RefinementConfig = None, logger=None):
        self.mesh = mesh
        self.partition = partition
        self.config = config or RefinementConfig()
        self.logger = logger or get_logger(__name__)

        self.target_area = float(mesh.M.sum()) / partition.n_cells

        self._migration_orchestrator: Optional[MigrationOrchestrator] = None

    # ── Individual stage methods ──────────────────────────────────────

    def optimize(self, **kwargs) -> dict:
        """Run constrained perimeter optimization on the current partition.

        Any RefinementConfig field can be overridden via kwargs:
        method, max_opt_iter, tolerance, lbfgs_memory, best_iterate,
        exact_hessian, use_vectorized.

        Returns:
            opt_info dict from PerimeterOptimizer.get_optimization_info(),
            augmented with 'initial_perimeter', 'perimeter_reduction',
            'percent_reduction', 'final_constraint_violations', and 'result'
            (the raw OptimizeResult).
        """
        cfg = self.config
        use_vectorized = kwargs.get('use_vectorized', cfg.use_vectorized)
        method = kwargs.get('method', cfg.method)
        max_opt_iter = kwargs.get('max_opt_iter', cfg.max_opt_iter)
        tolerance = kwargs.get('tolerance', cfg.tolerance)
        lbfgs_memory = kwargs.get('lbfgs_memory', cfg.lbfgs_memory)
        best_iterate = kwargs.get('best_iterate', cfg.best_iterate)
        exact_hessian = kwargs.get('exact_hessian', cfg.exact_hessian)

        optimizer = PerimeterOptimizer(
            self.partition, self.mesh, self.target_area,
            use_vectorized=use_vectorized)

        x0 = self.partition.get_variable_vector()
        initial_perimeter = optimizer.objective(x0)
        self.logger.info(f"Perimeter at start of iteration: {initial_perimeter:.10f}")

        opt_start_time = time.time()
        result = optimizer.optimize(
            max_iter=max_opt_iter,
            tol=tolerance,
            method=method,
            lbfgs_memory=lbfgs_memory,
            best_iterate=best_iterate,
            exact_hessian=exact_hessian,
        )
        opt_elapsed = time.time() - opt_start_time

        opt_info = optimizer.get_optimization_info(result)

        final_perimeter = opt_info['final_perimeter']
        perimeter_reduction = initial_perimeter - final_perimeter
        percent_reduction = (perimeter_reduction / initial_perimeter * 100) if initial_perimeter > 0 else 0.0
        final_areas = np.array(opt_info['final_areas'])
        constraint_violations = final_areas - self.target_area

        opt_info['initial_perimeter'] = float(initial_perimeter)
        opt_info['perimeter_reduction'] = float(perimeter_reduction)
        opt_info['percent_reduction'] = float(percent_reduction)
        opt_info['final_constraint_violations'] = constraint_violations.tolist()
        opt_info['result'] = result

        self.logger.info(f"\nOptimization completed in {opt_elapsed:.2f}s")
        self.logger.info(f"  Success: {opt_info['success']}")
        self.logger.info(f"  Iterations: {opt_info['n_iterations']}")
        self.logger.info(f"  Final perimeter: {final_perimeter:.10f}")
        self.logger.info(f"  Improvement: {perimeter_reduction:.10f} ({percent_reduction:.4f}%)")
        self.logger.info(f"  Max area violation: {np.max(np.abs(constraint_violations)):.2e}")

        return opt_info

    def detect(self) -> DetectionResult:
        """Detect topology migration triggers.

        Creates a MigrationOrchestrator internally, calls
        detect_all_triggers(), and returns the DetectionResult.
        Also stores the orchestrator instance for use by migrate().
        """
        mesh_topology = MeshTopology(self.mesh)
        orchestrator = MigrationOrchestrator(
            self.partition, self.mesh, mesh_topology,
            MigrationConfig(delta=self.config.boundary_tol))
        detection = orchestrator.detect_all_triggers(delta=self.config.boundary_tol)

        self._migration_orchestrator = orchestrator
        return detection

    def migrate(self) -> MigrationResult:
        """Execute migrations from the most recent detect() call.

        Calls execute_migrations(mode='batch') on the internally stored
        MigrationOrchestrator. Runs post-migration consistency check.

        Returns MigrationResult.

        Raises RuntimeError if detect() was not called first.
        """
        if self._migration_orchestrator is None:
            raise RuntimeError("detect() must be called before migrate()")

        mig_result = self._migration_orchestrator.execute_migrations(mode='batch')
        _check_indicator_vp_consistency(self.partition, self.mesh, self.logger)
        self._migration_orchestrator = None
        return mig_result

    def export_checkpoint(self, iteration_number, output_dir,
                          opt_info, pending_migration=True,
                          base_solution_path=None) -> str:
        """Export partition state to HDF5 checkpoint.

        Includes pre-write consistency check and post-write roundtrip
        perimeter verification. Returns the path to the exported file.

        Args:
            output_dir: Directory in which the checkpoint file is created.
                For structured layouts this is the campaign directory.
            base_solution_path: If provided, stored as a relative path
                in the HDF5 attrs so loaders can find the base solution
                without filename-based heuristics.
        """
        ckpt_ts = time.strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(
            output_dir, f"iteration_{iteration_number:03d}_{ckpt_ts}.h5")

        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info(f"EXPORTING ITERATION {iteration_number} STATE")
        self.logger.info("=" * 80)
        self.logger.info(f"Output file: {output_path}")
        self.logger.info(f"pending_migration: {pending_migration}")

        _check_indicator_vp_consistency(self.partition, self.mesh, self.logger)

        active_lambdas = []
        active_edges = []
        for vp in self.partition.variable_points:
            if getattr(vp, 'active', True):
                active_lambdas.append(vp.lambda_param)
                active_edges.append(vp.edge)

        lambda_opt = np.array(active_lambdas)
        vp_edges_arr = np.array(active_edges, dtype=np.int64)
        indicator_functions = self.partition.indicator_functions

        with h5py.File(output_path, 'w') as f:
            f.create_dataset('lambda_parameters', data=lambda_opt)
            f.create_dataset('vp_edges', data=vp_edges_arr)
            f.create_dataset('indicator_functions', data=indicator_functions)

            f.attrs['n_variable_points'] = len(lambda_opt)
            f.attrs['n_cells'] = self.partition.n_cells
            f.attrs['final_perimeter'] = opt_info['final_perimeter']
            f.attrs['optimization_success'] = opt_info['success']
            f.attrs['optimization_iterations'] = opt_info['n_iterations']
            f.attrs['iteration_number'] = iteration_number
            f.attrs['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
            f.attrs['pending_migration'] = bool(pending_migration)

            if base_solution_path is not None:
                try:
                    rel = os.path.relpath(
                        base_solution_path, os.path.dirname(output_path))
                except ValueError:
                    rel = os.path.abspath(base_solution_path)
                f.attrs['base_solution_path'] = rel

            opt_grp = f.create_group('optimization_info')
            opt_grp.attrs['initial_perimeter'] = opt_info['initial_perimeter']
            opt_grp.attrs['final_perimeter'] = opt_info['final_perimeter']
            opt_grp.attrs['perimeter_reduction'] = opt_info['perimeter_reduction']
            opt_grp.attrs['percent_reduction'] = opt_info['percent_reduction']
            opt_grp.create_dataset('constraint_violations', data=opt_info['final_constraint_violations'])

        self.logger.info("State exported successfully")
        self.logger.info(f"  {len(lambda_opt)} lambda parameters saved")
        self.logger.info(f"  Indicator functions saved: {indicator_functions.shape}")
        self.logger.info(f"  Perimeter: {opt_info['final_perimeter']:.10f}")
        self.logger.info(f"  pending_migration: {pending_migration}")

        try:
            mesh_r, partition_r = load_partition_from_refined_file(output_path, verbose=False)

            lambda_r = partition_r.get_variable_vector()
            partition_r.set_variable_vector(lambda_r)

            perim_calc_r = PerimeterCalculator(mesh_r, partition_r)
            regular_perimeter_r = perim_calc_r.compute_total_perimeter(lambda_r)

            steiner_handler_r = SteinerHandler(mesh_r, partition_r)
            steiner_perimeter_r = steiner_handler_r.get_total_perimeter_contribution()

            perimeter_r = regular_perimeter_r + steiner_perimeter_r

            in_memory_perimeter = opt_info['final_perimeter']
            rel_diff = abs(perimeter_r - in_memory_perimeter) / max(in_memory_perimeter, 1e-12)

            if rel_diff < 1e-4:
                self.logger.info(f"  Roundtrip perimeter check PASSED: "
                                 f"in-memory={in_memory_perimeter:.6f}, reloaded={perimeter_r:.6f} "
                                 f"(rel_diff={rel_diff:.2e})")
                self.logger.info(f"    Regular={regular_perimeter_r:.6f}, Steiner={steiner_perimeter_r:.6f}")
            else:
                self.logger.warning(f"  Roundtrip perimeter check FAILED: "
                                    f"in-memory={in_memory_perimeter:.6f}, reloaded={perimeter_r:.6f} "
                                    f"(rel_diff={rel_diff:.2e})")
                self.logger.warning(f"    Regular={regular_perimeter_r:.6f}, Steiner={steiner_perimeter_r:.6f}")
                self.logger.warning(f"    VP count in-memory: {len(lambda_opt)}, reloaded: {len(lambda_r)}")
        except Exception as e:
            self.logger.warning(f"  Roundtrip perimeter check skipped: {e}")

        self.logger.info("=" * 80)

        return output_path

    # ── Convenience loop ──────────────────────────────────────────────

    def run_refinement_loop(self, solution_path, output_dir=None) -> dict:
        """Full iterative refinement loop with auto-detection and resume.

        This method:
        1. Calls detect_file_type() on solution_path
        2. Loads mesh/partition via the appropriate loader
        3. Determines the entry point (optimize or migrate)
        4. Runs the optimize -> detect -> export -> migrate loop
        5. Returns a summary dict

        The three execution paths (base, checkpoint_pending,
        checkpoint_converged) are handled automatically.
        """
        file_type = detect_file_type(solution_path)
        self.logger.info(f"File type detected: {file_type}")

        if file_type == 'checkpoint_converged':
            self.logger.info("Input file is a converged result (pending_migration=False). Nothing to do.")
            return {'converged': True, 'iterations': 0, 'message': 'Already converged'}

        if file_type == 'base':
            self.logger.info("")
            self.logger.info("=" * 80)
            self.logger.info("Loading base solution via ContourAnalyzer")
            self.logger.info("=" * 80)

            mesh, partition = load_partition_from_base_file(solution_path, verbose=True)
            starting_iteration = 0
            enter_at = 'optimize'
        else:
            self.logger.info("")
            self.logger.info("=" * 80)
            self.logger.info("Loading iteration checkpoint (pending migration)")
            self.logger.info("=" * 80)

            mesh, partition = load_partition_from_refined_file(solution_path, verbose=True)
            starting_iteration = _read_iteration_number(solution_path)
            enter_at = 'migrate'
            self.logger.info(f"Will resume from iteration {starting_iteration} (migrate first, then optimize)")

        self.mesh = mesh
        self.partition = partition
        self.target_area = float(mesh.M.sum()) / partition.n_cells

        if output_dir is None:
            output_dir = derive_output_paths(
                solution_path, file_type, config=self.config)

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        self._write_refinement_yaml(output_dir, solution_path)

        abs_solution = os.path.abspath(solution_path)
        if file_type == 'base':
            base_solution_for_export = abs_solution
        else:
            try:
                from .io import find_base_solution_path
                base_solution_for_export = find_base_solution_path(solution_path)
            except FileNotFoundError:
                base_solution_for_export = None

        self.logger.info(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} triangles")
        self.logger.info(f"Partition: {partition.n_cells} cells, {len(partition.variable_points)} VPs")
        self.logger.info(f"Total mesh area: {float(mesh.M.sum()):.6f}")
        self.logger.info(f"Target area per cell: {self.target_area:.6f}")
        self.logger.info(f"Output directory: {output_dir}")

        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("STARTING ITERATIVE REFINEMENT")
        self.logger.info("=" * 80)
        self.logger.info(f"Entry point: {enter_at}")
        self.logger.info(f"Starting iteration: {starting_iteration}")
        self.logger.info(f"Maximum topology iterations: {self.config.max_iterations}")
        self.logger.info(f"Save iterations: {self.config.save_iterations}")
        self.logger.info("=" * 80)

        global_start_time = time.time()
        converged = False
        topology_iteration = 0
        total_type1 = 0
        total_type2 = 0
        iteration_files = []
        first_pass = True
        final_perimeter = None

        while not converged and topology_iteration < self.config.max_iterations:
            iteration_number = starting_iteration + topology_iteration + 1

            # ── RESUME-AT-MIGRATE (first pass only) ──
            if first_pass and enter_at == 'migrate':
                first_pass = False

                self.logger.info("")
                self.logger.info("=" * 80)
                self.logger.info(f"RESUMING: Applying pending migration from iteration {starting_iteration}")
                self.logger.info("=" * 80)

                detection = self.detect()

                if not detection.type1_triggers and not detection.type2_triggers:
                    self.logger.warning("File had pending_migration=True but no switches found on resume")
                    converged = True
                    break

                self.logger.info(f"Re-detected: {len(detection.type1_triggers)} Type 1, "
                                 f"{len(detection.type2_triggers)} Type 2")

                mig_result = self.migrate()
                if mig_result.failed:
                    self.logger.error(f"Migration failure on resume: {mig_result.error_message}")
                    return {'converged': False, 'error': mig_result.error_message}

                total_type1 += mig_result.type1_applied
                total_type2 += (mig_result.type2_forward_applied +
                                mig_result.type2_rollbacks_applied)

                self.logger.info("Pending migration applied. Continuing to optimization.")
                continue

            first_pass = False

            self.logger.info("")
            self.logger.info("=" * 80)
            self.logger.info(f"ITERATION {iteration_number}/{starting_iteration + self.config.max_iterations}")
            self.logger.info("=" * 80)

            # ── PHASE 1: OPTIMIZE ──
            self.logger.info("")
            self.logger.info(f"Phase 1: Optimizing current topology...")
            self.logger.info("-" * 80)

            opt_info = self.optimize()

            if not opt_info['success']:
                IPOPT_NONFATAL = {-1, -4, 5}
                ipopt_status = opt_info.get('status', None)
                is_nonfatal = (ipopt_status in IPOPT_NONFATAL)

                if self.config.allow_partial_convergence and is_nonfatal:
                    self.logger.warning(f"\nOptimizer did not fully converge (status={ipopt_status}: "
                                        f"{opt_info['message']})")
                    self.logger.warning("--allow-partial-convergence is set: continuing to migration phase.")
                else:
                    self.logger.error("\nOptimization failed to converge!")
                    self.logger.error(f"Message: {opt_info['message']}")
                    self.logger.error("Exporting current state and stopping")

                    failure_file = self.export_checkpoint(
                        iteration_number, output_dir, opt_info,
                        pending_migration=True,
                        base_solution_path=base_solution_for_export)
                    self.logger.info(f"State saved to: {failure_file}")
                    return {'converged': False, 'error': 'Optimization failed',
                            'last_file': failure_file}

            final_perimeter = opt_info['final_perimeter']

            # ── PHASE 2: DETECT ──
            self.logger.info("")
            self.logger.info(f"Phase 2: Detecting topology switches...")
            self.logger.info("-" * 80)

            detection = self.detect()
            pending_migration = bool(detection.type1_triggers or detection.type2_triggers)

            if pending_migration:
                self.logger.info(f"Switches detected: {len(detection.type1_triggers)} Type 1, "
                                 f"{len(detection.type2_triggers)} Type 2")
            else:
                self.logger.info("No topology switches needed")

            # ── PHASE 3: EXPORT (conditional) ──
            is_last_iteration = (topology_iteration + 1 == self.config.max_iterations)
            should_export = (self.config.save_iterations
                             or not pending_migration
                             or is_last_iteration)

            iteration_file = None
            if should_export:
                self.logger.info("")
                self.logger.info(f"Phase 3: Exporting state (pending_migration={pending_migration})...")
                self.logger.info("-" * 80)

                iteration_file = self.export_checkpoint(
                    iteration_number, output_dir, opt_info,
                    pending_migration=pending_migration,
                    base_solution_path=base_solution_for_export)
                iteration_files.append(iteration_file)

            # ── CHECK CONVERGENCE ──
            if not pending_migration:
                self.logger.info("")
                self.logger.info("=" * 80)
                self.logger.info("CONVERGENCE ACHIEVED")
                self.logger.info("=" * 80)
                self.logger.info("No topology switches needed")
                self.logger.info(f"Final perimeter: {final_perimeter:.10f}")
                converged = True
                break

            # ── PHASE 4: MIGRATE ──
            self.logger.info("")
            self.logger.info(f"Phase 4: Applying migrations...")
            self.logger.info("-" * 80)

            mig_result = self.migrate()

            if mig_result.failed:
                self.logger.error("")
                self.logger.error("=" * 80)
                self.logger.error("MIGRATION FAILURE")
                self.logger.error("=" * 80)
                self.logger.error(f"Error: {mig_result.error_message}")
                diag_file = iteration_file if iteration_file else solution_path
                self.logger.error(f"State saved to: {diag_file}")
                self.logger.error("=" * 80)
                return {'converged': False, 'error': mig_result.error_message,
                        'last_file': iteration_file}

            iteration_migrations = (mig_result.type1_applied
                                    + mig_result.type2_forward_applied
                                    + mig_result.type2_rollbacks_applied)

            if iteration_migrations == 0:
                self.logger.warning("")
                self.logger.warning("=" * 80)
                self.logger.warning("NO MIGRATIONS APPLIED")
                self.logger.warning("=" * 80)
                self.logger.warning("Switches were detected but no migrations succeeded")
                self.logger.warning("Cannot make progress - stopping")
                diag_file = iteration_file if iteration_file else solution_path
                self.logger.warning(f"State saved to: {diag_file}")
                self.logger.warning("=" * 80)
                return {'converged': False,
                        'error': 'Switches detected but no migrations applied',
                        'last_file': iteration_file}

            _t1_count = mig_result.type1_applied
            _t2_count = mig_result.type2_forward_applied + mig_result.type2_rollbacks_applied
            self.logger.info(f"\nMigrations applied in iteration {iteration_number}: {iteration_migrations}")
            self.logger.info(f"  Type 1: {_t1_count}")
            self.logger.info(f"  Type 2: {_t2_count}")

            total_type1 += mig_result.type1_applied
            total_type2 += (mig_result.type2_forward_applied +
                            mig_result.type2_rollbacks_applied)

            active_vps = sum(1 for vp in self.partition.variable_points if vp.active)
            self.logger.info(f"  Total VPs: {len(self.partition.variable_points)} ({active_vps} active)")
            self.logger.info(f"  Optimizer will be rebuilt at start of next iteration")

            self.logger.info("")
            self.logger.info("=" * 80)
            self.logger.info(f"ITERATION {iteration_number} COMPLETE")
            self.logger.info("=" * 80)

            topology_iteration += 1

        if not converged:
            self.logger.warning("")
            self.logger.warning("=" * 80)
            self.logger.warning("MAX ITERATIONS REACHED")
            self.logger.warning("=" * 80)
            self.logger.warning(f"Stopped after {self.config.max_iterations} topology iterations")
            self.logger.warning("Partition has not fully converged")
            self.logger.warning("Consider increasing --max-iterations if further refinement needed")
            self.logger.warning("=" * 80)

        global_elapsed = time.time() - global_start_time
        final_file = iteration_files[-1] if iteration_files else None

        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("REFINEMENT COMPLETE")
        self.logger.info("=" * 80)
        self.logger.info(f"Total time: {global_elapsed:.2f}s")
        self.logger.info(f"Total topology iterations: {topology_iteration}")
        self.logger.info(f"Total migrations applied: {total_type1 + total_type2}")
        self.logger.info(f"  Type 1: {total_type1}")
        self.logger.info(f"  Type 2: {total_type2}")
        if final_perimeter is not None:
            self.logger.info(f"Final perimeter: {final_perimeter:.10f}")
        self.logger.info(f"Convergence: {'Yes' if converged else 'No (max iterations reached)'}")
        self.logger.info("")
        self.logger.info("Output files:")
        if iteration_files:
            for fpath in iteration_files:
                self.logger.info(f"  - {fpath}")
            self.logger.info(f"Final state: {final_file}")
        self.logger.info("=" * 80)

        return {
            'converged': converged,
            'iterations': topology_iteration,
            'total_type1_migrations': total_type1,
            'total_type2_migrations': total_type2,
            'final_perimeter': final_perimeter,
            'iteration_files': iteration_files,
        }

    # ── Private helpers ───────────────────────────────────────────────

    def _write_refinement_yaml(self, campaign_dir, solution_path):
        """Write a refinement.yaml config snapshot into the campaign directory."""
        yaml_path = os.path.join(campaign_dir, 'refinement.yaml')
        if os.path.exists(yaml_path):
            return

        cfg = self.config
        snapshot = dataclasses.asdict(cfg)
        snapshot['base_solution'] = os.path.abspath(solution_path)
        snapshot['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        snapshot['campaign_name'] = build_campaign_name(cfg)

        with open(yaml_path, 'w') as f:
            yaml.dump(snapshot, f, default_flow_style=False, sort_keys=False)
