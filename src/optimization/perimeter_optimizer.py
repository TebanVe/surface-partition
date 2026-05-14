"""
Perimeter optimization with area constraints for partition refinement.

This module implements the constrained optimization algorithm from Section 5 of the paper.
It minimizes total perimeter while maintaining equal area constraints using scipy's SLSQP.

TERMINOLOGY (following paper Section 5):
- "cell": Partition region (what we optimize for equal areas)
- "triangle": Mesh triangle element (computational discretization)
- "edge": Mesh triangle edge (computational discretization)

Main features:
- Objective: Minimize total perimeter (including Steiner tree contributions)
- Constraints: Equal area for all partition cells (within tolerance)
- Bounds: λ ∈ [0, 1] for all variable points
- Gradients: Analytical for perimeter and area
- Topology switching: Detect and handle boundary cases
"""

import numpy as np
from scipy.optimize import minimize, OptimizeResult
from typing import Dict, Optional, Tuple, List, Set
import time

from ..logging_config import get_logger
from ..profiling import ProfilingState
from ..mesh.tri_mesh import TriMesh
from ..partition.contour_partition import PartitionContour
from ..partition.area_calculator import AreaCalculator
from ..partition.perimeter_calculator import PerimeterCalculator
from ..partition.steiner_handler import SteinerHandler
from ..partition.partition_arrays import PartitionArrays
from ..partition import vectorized_perimeter
from ..partition import vectorized_area
from ..partition import vectorized_steiner


class IPOPTProblemAdapter:
    """Wraps PerimeterOptimizer callbacks into the cyipopt problem interface.

    cyipopt requires a class with specific method names (objective, gradient,
    constraints, jacobian, jacobianstructure). This adapter delegates to the
    existing PerimeterOptimizer methods, which handle the active/inactive VP
    mapping internally.

    Best-iterate tracking
    ---------------------
    When ``track_best=True``, the adapter records the best (objective, x) pair
    seen during the solve, considering only iterates with constraint violation
    below ``best_feas_tol``.  After ``problem.solve()`` returns, the caller can
    check ``best_x`` / ``best_obj`` and substitute them into the result if the
    returned iterate is worse.
    """

    def __init__(self, optimizer: 'PerimeterOptimizer',
                 track_best: bool = False, best_feas_tol: float = 1e-6,
                 exact_hessian: bool = False,
                 profile: Optional[ProfilingState] = None):
        self._opt = optimizer
        self._pa = optimizer._arrays   # PartitionArrays snapshot
        self._track_best = track_best
        self._best_feas_tol = best_feas_tol
        self._exact_hessian = exact_hessian
        self._profile = profile
        self.best_obj: float = np.inf
        self.best_x: Optional[np.ndarray] = None
        self._last_x: Optional[np.ndarray] = None

        if exact_hessian:
            self.hessianstructure = self._hessianstructure_impl
            self.hessian = self._hessian_impl

    def objective(self, x: np.ndarray) -> float:
        if self._track_best:
            self._last_x = x.copy()
        if self._profile is not None:
            t0 = time.perf_counter()
        result = self._opt.objective(x)
        if self._profile is not None:
            self._profile.record('objective', time.perf_counter() - t0)
        return result

    def gradient(self, x: np.ndarray) -> np.ndarray:
        if self._profile is not None:
            t0 = time.perf_counter()
        result = self._opt.objective_gradient(x)
        if self._profile is not None:
            self._profile.record('gradient', time.perf_counter() - t0)
        return result

    def constraints(self, x: np.ndarray) -> np.ndarray:
        if self._profile is not None:
            t0 = time.perf_counter()
        result = self._opt.constraint_area_equality(x)
        if self._profile is not None:
            self._profile.record('constraints', time.perf_counter() - t0)
        return result

    def jacobianstructure(self) -> tuple:
        """Pre-computed sparsity pattern — called once at setup."""
        return (self._pa.jac_row, self._pa.jac_col)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Return non-zero Jacobian values in jacobianstructure() order.

        Direct sparse computation — no dense matrix allocated.
        """
        if self._profile is not None:
            t0 = time.perf_counter()
        self._opt._arrays.vp_lambda[:] = x
        pa = self._opt._arrays

        if pa.nnz_lookup is not None:
            area_vals = vectorized_area.compute_area_jacobian_sparse(pa)
            steiner_vals = vectorized_steiner.compute_steiner_area_jacobian_sparse(
                pa, _prof=self._profile)
            result = area_vals + steiner_vals
        else:
            J_dense = self._opt.constraint_area_jacobian(x)
            result = J_dense[self._pa.jac_row, self._pa.jac_col]

        if self._profile is not None:
            self._profile.record('jacobian', time.perf_counter() - t0)
        return result

    def _hessianstructure_impl(self) -> tuple:
        """Return Hessian sparsity pattern (lower triangle)."""
        return (self._pa.hess_row, self._pa.hess_col)

    def _hessian_impl(self, x: np.ndarray, lagrange: np.ndarray,
                      obj_factor: float) -> np.ndarray:
        """Compute the Hessian of the Lagrangian.

        H = obj_factor * d2f/dlam^2 + sum_k lagrange[k] * d2c_k/dlam^2

        Args:
            x: current VP parameters
            lagrange: constraint multipliers (n_cells-1,)
            obj_factor: scaling for the objective (sigma)

        Returns:
            (hess_nnz,) float64 — lower triangle values.
        """
        if self._profile is not None:
            t0 = time.perf_counter()
        self._opt._arrays.vp_lambda[:] = x
        pa = self._opt._arrays

        perim_hess = vectorized_perimeter.compute_perimeter_hessian_sparse(pa)
        steiner_perim_hess = vectorized_steiner.compute_steiner_perimeter_hessian_fd(
            pa, _prof=self._profile)

        area_hess = vectorized_area.compute_area_hessian_sparse(pa, lagrange)
        steiner_area_hess = vectorized_steiner.compute_steiner_area_hessian_fd(
            pa, lagrange, _prof=self._profile)

        result = (obj_factor * (perim_hess + steiner_perim_hess)
                  + area_hess + steiner_area_hess)
        if self._profile is not None:
            self._profile.record('hessian', time.perf_counter() - t0)
        return result

    def intermediate(self, alg_mod, iter_count, obj_value,
                     inf_pr, inf_du, mu, d_norm,
                     regularization_size, alpha_du, alpha_pr, ls_trials):
        """Called by IPOPT after each iteration. Log progress and track best."""
        import logging
        logger = logging.getLogger(__name__)
        if iter_count % 10 == 0:
            logger.info(f"IPOPT iter {iter_count}: obj={obj_value:.6f}, "
                        f"constr_viol={inf_pr:.2e}")

        if self._profile is not None:
            self._profile.ipopt_iter_count += 1

        if (self._track_best
                and inf_pr < self._best_feas_tol
                and obj_value < self.best_obj
                and self._last_x is not None):
            self.best_obj = obj_value
            self.best_x = self._last_x.copy()

        return True


class PerimeterOptimizer:
    """
    Constrained perimeter minimization optimizer for partition refinement.
    
    Takes zigzagged contours from indicator functions and optimizes variable
    point positions to minimize total perimeter while preserving equal areas.
    
    Attributes:
        mesh: The underlying TriMesh
        partition: PartitionContour with variable points
        target_area: Target area for each partition cell
        area_calc: AreaCalculator for computing areas and gradients
        perim_calc: PerimeterCalculator for computing perimeters and gradients
        steiner_handler: SteinerHandler for triple point management
        logger: Logger instance
    """
    
    def __init__(self, partition: PartitionContour, mesh: TriMesh, target_area: float,
                 area_calc=None, perim_calc=None, steiner_handler=None,
                 use_vectorized: bool = True):
        """
        Initialize perimeter optimizer.
        
        Args:
            partition: PartitionContour with extracted contours
            mesh: TriMesh object
            target_area: Target area for each partition cell (total_area / n_cells)
            area_calc: Optional pre-created AreaCalculator (for efficiency)
            perim_calc: Optional pre-created PerimeterCalculator (for efficiency)
            steiner_handler: Optional pre-created SteinerHandler (for efficiency)
            use_vectorized: If True (default), compile flat arrays and use the
                vectorized evaluation path.  When False, use the original
                per-element calculators.
        """
        self.mesh = mesh
        self.partition = partition
        self.target_area = float(target_area)
        self.logger = get_logger(__name__)
        
        # Initialize calculators (reuse if provided, create if not)
        self.area_calc = area_calc if area_calc is not None else AreaCalculator(mesh, partition)
        self.perim_calc = perim_calc if perim_calc is not None else PerimeterCalculator(mesh, partition)
        self.steiner_handler = steiner_handler if steiner_handler is not None else SteinerHandler(mesh, partition)
        
        # Active VP mapping: after migrations some VPs are inactive and should
        # not be optimized.  The calculators work with full-sized vectors indexed
        # by absolute VP position, so we expand/compress at the optimizer boundary.
        self._active_indices = np.array(
            [i for i, vp in enumerate(partition.variable_points)
             if getattr(vp, 'active', True)],
            dtype=int
        )
        self._n_total = len(partition.variable_points)
        self._n_active = len(self._active_indices)
        self._use_active_mapping = (self._n_active < self._n_total)
        
        # Vectorized evaluation state
        self._use_vectorized = use_vectorized
        self._arrays: Optional[PartitionArrays] = None

        # Cached last objective for callback (avoids redundant recomputation)
        self._last_objective: Optional[float] = None
        
        # Optimization state
        self.iteration = 0
        self.objective_history = []
        self.constraint_violation_history = []
        
        self.logger.info(f"Initialized PerimeterOptimizer:")
        self.logger.info(f"  {partition.n_cells} partition cells")
        if self._use_active_mapping:
            self.logger.info(f"  {self._n_total} variable points ({self._n_active} active, "
                           f"{self._n_total - self._n_active} inactive)")
        else:
            self.logger.info(f"  {self._n_total} variable points (all active)")
        self.logger.info(f"  {len(self.steiner_handler.triple_points)} triple points")
        self.logger.info(f"  Target area per partition cell: {target_area:.6f}")
        self.logger.info(f"  Vectorized evaluation: {use_vectorized}")
    
    def _to_full(self, active_vec: np.ndarray) -> np.ndarray:
        """Expand an active-only vector to a full-sized vector (inactive VPs keep current λ)."""
        if not self._use_active_mapping:
            return active_vec
        full = self.partition.get_variable_vector()
        full[self._active_indices] = active_vec
        return full
    
    def _to_active(self, full_vec: np.ndarray) -> np.ndarray:
        """Compress a full-sized vector to active-only."""
        if not self._use_active_mapping:
            return full_vec
        return full_vec[self._active_indices]
    
    def _to_active_2d(self, full_matrix: np.ndarray) -> np.ndarray:
        """Compress columns of a full-sized matrix (e.g. Jacobian) to active-only."""
        if not self._use_active_mapping:
            return full_matrix
        return full_matrix[:, self._active_indices]
    
    def compile(self):
        """Compile flat arrays for fast vectorized evaluation.

        Call after migrations, before optimize().  When ``_use_vectorized`` is
        False this is a no-op.
        """
        if not self._use_vectorized:
            return
        self._arrays = self.partition.compile_arrays(self.steiner_handler)
        self.logger.info("Compiled PartitionArrays for vectorized evaluation")
    
    def objective(self, lambda_vec: np.ndarray) -> float:
        """
        Compute total perimeter (objective function to minimize).
        
        Includes both regular segment perimeters and Steiner tree contributions.
        
        Args:
            lambda_vec: Variable point parameters (active-only during optimization,
                       or full-sized for external calls)
            
        Returns:
            Total perimeter length
        """
        # --- Vectorized path ---
        if self._arrays is not None:
            self._arrays.vp_lambda[:] = lambda_vec
            regular_perimeter = vectorized_perimeter.compute_total_perimeter(self._arrays)
            steiner_pts = vectorized_steiner.compute_steiner_points(self._arrays)
            steiner_perim = vectorized_steiner.compute_steiner_perimeter(self._arrays, steiner_pts)
            total = regular_perimeter + steiner_perim
            self._last_objective = total
            return total

        # --- Original path (fallback) ---
        full_vec = self._to_full(lambda_vec) if len(lambda_vec) == self._n_active else lambda_vec
        self.partition.set_variable_vector(full_vec)
        
        regular_perimeter = self.perim_calc.compute_total_perimeter(full_vec)
        steiner_perimeter = self.steiner_handler.get_total_perimeter_contribution()
        
        total = regular_perimeter + steiner_perimeter
        self._last_objective = total
        return total
    
    def objective_gradient(self, lambda_vec: np.ndarray) -> np.ndarray:
        """
        Compute gradient of objective function ∂(perimeter)/∂λ.
        
        Args:
            lambda_vec: Variable point parameters (active-only during optimization)
            
        Returns:
            Gradient array (active-only during optimization)
        """
        # --- Vectorized path ---
        if self._arrays is not None:
            self._arrays.vp_lambda[:] = lambda_vec
            regular_grad = vectorized_perimeter.compute_perimeter_gradient(self._arrays)
            steiner_grad = vectorized_steiner.compute_steiner_perimeter_gradient(self._arrays)
            return regular_grad + steiner_grad

        # --- Original path (fallback) ---
        full_vec = self._to_full(lambda_vec) if len(lambda_vec) == self._n_active else lambda_vec
        self.partition.set_variable_vector(full_vec)
        
        regular_gradient = self.perim_calc.compute_total_perimeter_gradient(full_vec)
        steiner_gradient = self.steiner_handler.compute_total_gradient_finite_difference()
        
        total_gradient = regular_gradient + steiner_gradient
        
        return self._to_active(total_gradient) if len(lambda_vec) == self._n_active else total_gradient
    
    def constraint_area_equality(self, lambda_vec: np.ndarray) -> np.ndarray:
        """
        Compute area constraint violations: Area_i - target_area.
        
        For n cells, we constrain n-1 cells (last one is determined by conservation).
        
        Args:
            lambda_vec: Variable point parameters (active-only during optimization)
            
        Returns:
            Constraint violations array of shape (n_cells - 1,)
        """
        # --- Vectorized path ---
        if self._arrays is not None:
            self._arrays.vp_lambda[:] = lambda_vec
            areas = vectorized_area.compute_cell_areas(self._arrays)
            steiner_pts = vectorized_steiner.compute_steiner_points(self._arrays)
            steiner_a = vectorized_steiner.compute_steiner_areas(self._arrays, steiner_pts)
            areas += steiner_a
            return areas[:self._arrays.n_cells - 1] - self.target_area

        # --- Original path (fallback) ---
        full_vec = self._to_full(lambda_vec) if len(lambda_vec) == self._n_active else lambda_vec
        self.partition.set_variable_vector(full_vec)
        
        areas = self.area_calc.compute_all_cell_areas(full_vec)
        steiner_areas = self.steiner_handler.get_total_area_contribution()
        for cell_idx, area_contrib in steiner_areas.items():
            if cell_idx < len(areas):
                areas[cell_idx] += area_contrib
        
        return areas[:-1] - self.target_area
    
    def constraint_area_jacobian(self, lambda_vec: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of area constraints ∂(Area_i - target)/∂λ.
        
        Per paper section 5: Uses finite differences for Steiner area gradients.
        
        Args:
            lambda_vec: Variable point parameters (active-only during optimization)
            
        Returns:
            Jacobian array (columns correspond to active VPs during optimization)
        """
        # --- Vectorized path ---
        if self._arrays is not None:
            self._arrays.vp_lambda[:] = lambda_vec
            regular_jac = vectorized_area.compute_area_jacobian(self._arrays)
            steiner_jac = vectorized_steiner.compute_steiner_area_jacobian(self._arrays)
            return regular_jac + steiner_jac

        # --- Original path (fallback) ---
        full_vec = self._to_full(lambda_vec) if len(lambda_vec) == self._n_active else lambda_vec
        self.partition.set_variable_vector(full_vec)
        
        jacobian = self.area_calc.compute_area_jacobian(full_vec)
        steiner_gradients = self.steiner_handler.compute_area_gradients_finite_difference(
            self.mesh, eps=1e-7
        )
        
        for cell_idx in range(self.partition.n_cells - 1):
            jacobian[cell_idx, :] += steiner_gradients[cell_idx]
        
        return self._to_active_2d(jacobian) if len(lambda_vec) == self._n_active else jacobian
    
    def _callback(self, *args, **kwargs):
        """
        Callback function called at each optimization iteration.
        
        Uses cached ``_last_objective`` from the most recent ``objective()``
        call to avoid redundant recomputation.
        """
        self.iteration += 1
        
        if self._last_objective is not None:
            self.objective_history.append(self._last_objective)

        if self.iteration % 10 == 0:
            obj_str = f"{self._last_objective:.6f}" if self._last_objective is not None else "N/A"
            self.logger.info(f"Iteration {self.iteration}: Perimeter={obj_str}")
    
    def optimize(self, max_iter: int = 1000, tol: float = 1e-7,
                method: str = 'SLSQP',
                lbfgs_memory: int = 6,
                best_iterate: bool = False,
                exact_hessian: bool = False,
                profile: Optional[ProfilingState] = None) -> OptimizeResult:
        """
        Run constrained perimeter optimization.
        
        Uses scipy.optimize.minimize (SLSQP / trust-constr) or cyipopt.Problem
        (ipopt) to minimize total perimeter subject to equal-area constraints.
        
        - Objective: Total perimeter
        - Constraints: Equal area for each cell (n-1 constraints)
        - Bounds: λ ∈ [0, 1] for all variable points
        
        Args:
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            method: Optimization method — 'SLSQP', 'trust-constr', or 'ipopt'.
                'ipopt' requires the cyipopt package and vectorized evaluation.
            lbfgs_memory: L-BFGS history size for IPOPT (default 6).
                Higher values capture more curvature at the cost of memory.
                Ignored for SLSQP / trust-constr.
            best_iterate: If True, track the best feasible iterate during the
                IPOPT solve and return it instead of the last iterate when the
                last iterate is worse.  Prevents restoration-phase losses from
                compounding across outer iterations.  Ignored for SLSQP /
                trust-constr.
            exact_hessian: If True, provide IPOPT with an analytical Hessian
                of the Lagrangian instead of L-BFGS approximation.  Gives
                exact curvature for smoother boundaries.  Ignored for SLSQP /
                trust-constr.
            
        Returns:
            scipy OptimizeResult object
        """
        self.logger.info(f"Starting perimeter optimization with method={method}")
        self.logger.info(f"  max_iter={max_iter}, tol={tol}")
        if method == 'ipopt':
            self.logger.info(f"  L-BFGS memory={lbfgs_memory}, best_iterate={best_iterate}, "
                           f"exact_hessian={exact_hessian}")
        if self._use_active_mapping:
            self.logger.info(f"  Optimizing {self._n_active} active VPs "
                           f"(skipping {self._n_total - self._n_active} inactive)")
        
        # Compile vectorized arrays (no-op when _use_vectorized is False)
        self.compile()
        
        # Initial guess: current λ values for active VPs only
        lambda0 = self.partition.get_active_variable_vector() if self._use_active_mapping \
                   else self.partition.get_variable_vector()
        
        # Box constraints: λ ∈ [0, 1]
        bounds = [(0.0, 1.0) for _ in lambda0]
        
        # Area equality constraints
        constraints = {
            'type': 'eq',
            'fun': self.constraint_area_equality,
            'jac': self.constraint_area_jacobian
        }
        
        # Initial objective and constraint values
        obj0 = self.objective(lambda0)
        constr0 = self.constraint_area_equality(lambda0)
        max_viol0 = float(np.max(np.abs(constr0)))
        
        self.logger.info(f"Initial state:")
        self.logger.info(f"  Perimeter: {obj0:.6f}")
        self.logger.info(f"  Max constraint violation: {max_viol0:.2e}")
        
        # Run optimization
        start_time = time.time()

        if method == 'ipopt':
            # ---- IPOPT path (does NOT call scipy minimize) ----
            try:
                import cyipopt
            except ImportError:
                raise ImportError(
                    "IPOPT requested but cyipopt is not installed.\n"
                    "Install with: pip install cyipopt"
                )

            if self._arrays is None:
                raise RuntimeError(
                    "IPOPT requires vectorized evaluation. "
                    "Do not pass --no-vectorized when using --method ipopt."
                )

            n = len(lambda0)
            m = self.partition.n_cells - 1

            adapter = IPOPTProblemAdapter(
                self, track_best=best_iterate, best_feas_tol=tol * 100,
                exact_hessian=exact_hessian, profile=profile)

            problem = cyipopt.Problem(
                n=n,
                m=m,
                problem_obj=adapter,
                lb=np.zeros(n),
                ub=np.ones(n),
                cl=np.zeros(m),
                cu=np.zeros(m),
            )

            if not exact_hessian:
                problem.add_option('hessian_approximation', 'limited-memory')
                problem.add_option('limited_memory_max_history', lbfgs_memory)
            problem.add_option('mu_strategy', 'adaptive')
            problem.add_option('tol', tol)
            problem.add_option('acceptable_tol', tol * 100)
            problem.add_option('max_iter', max_iter)
            problem.add_option('print_level', 3)

            # cyipopt logs every callback invocation at INFO; raise threshold to
            # WARNING so the per-iteration callback noise is suppressed in normal
            # use and only appears with --log-level DEBUG.
            import logging as _logging
            _cyipopt_logger = _logging.getLogger('cyipopt')
            _cyipopt_prev_level = _cyipopt_logger.level
            _cyipopt_logger.setLevel(_logging.WARNING)

            x_opt, info = problem.solve(lambda0)

            _cyipopt_logger.setLevel(_cyipopt_prev_level)

            ipopt_status = info['status']
            success = ipopt_status in (0, 1)
            msg = info.get('status_msg', '')
            if isinstance(msg, bytes):
                msg = msg.decode()

            final_obj = info['obj_val']
            final_x = x_opt

            if (best_iterate
                    and adapter.best_x is not None
                    and adapter.best_obj < final_obj):
                delta = final_obj - adapter.best_obj
                self.logger.info(
                    f"Best-iterate recovery: returning iter with "
                    f"obj={adapter.best_obj:.6f} instead of final "
                    f"obj={final_obj:.6f} (recovered {delta:.6f})")
                final_x = adapter.best_x
                final_obj = adapter.best_obj

            result = OptimizeResult(
                x=final_x,
                success=success,
                message=msg,
                fun=final_obj,
                nit=-1,
                nfev=-1,
                status=ipopt_status,
            )

        else:
            # ---- scipy path (SLSQP / trust-constr) ----
            if method == 'SLSQP':
                options = {'maxiter': max_iter, 'ftol': tol, 'disp': True}
            elif method == 'trust-constr':
                options = {'maxiter': max_iter, 'gtol': tol, 'xtol': tol, 'disp': True}
            else:
                options = {'maxiter': max_iter, 'disp': True}

            result = minimize(
                fun=self.objective,
                x0=lambda0,
                method=method,
                jac=self.objective_gradient,
                bounds=bounds,
                constraints=constraints,
                callback=self._callback,
                options=options,
            )

        elapsed_time = time.time() - start_time
        if profile is not None:
            profile.total_wall_s += elapsed_time

        # Sync optimized lambdas back to the object representation
        if self._arrays is not None:
            # Vectorized path: result.x lives in active-index space; expand to full
            full_result = self._to_full(result.x)
            self.partition.set_variable_vector(full_result)
        else:
            self.partition.set_variable_vector(result.x)
        
        # Final statistics
        final_obj = self.objective(result.x)
        final_constr = self.constraint_area_equality(result.x)
        final_max_viol = float(np.max(np.abs(final_constr)))
        
        self.logger.info(f"Optimization completed in {elapsed_time:.2f}s")
        self.logger.info(f"  Success: {result.success}")
        self.logger.info(f"  Message: {result.message}")
        self.logger.info(f"  Iterations: {result.nit}")
        self.logger.info(f"  Function evaluations: {result.nfev}")
        self.logger.info(f"  Final perimeter: {final_obj:.6f}")
        self.logger.info(f"  Perimeter reduction: {obj0 - final_obj:.6f} ({100*(obj0-final_obj)/obj0:.2f}%)")
        self.logger.info(f"  Final max constraint violation: {final_max_viol:.2e}")
        
        return result
    
    def get_optimization_info(self, result: OptimizeResult) -> Dict:
        """
        Extract optimization metadata for saving.
        
        Args:
            result: scipy OptimizeResult object
            
        Returns:
            Dictionary with optimization information
        """
        final_perimeter = self.objective(result.x)
        full_x = self._to_full(result.x) if len(result.x) == self._n_active else result.x
        final_areas = self.area_calc.compute_all_cell_areas(full_x)
        
        # Add Steiner contributions to areas
        steiner_areas = self.steiner_handler.get_total_area_contribution()
        for cell_idx, area_contrib in steiner_areas.items():
            if cell_idx < len(final_areas):
                final_areas[cell_idx] += area_contrib
        
        return {
            'success': bool(result.success),
            'status': int(result.status) if hasattr(result, 'status') and result.status is not None else None,
            'n_iterations': int(result.nit),
            'n_function_evals': int(result.nfev),
            'final_perimeter': float(final_perimeter),
            'final_areas': final_areas.tolist(),
            'target_area': float(self.target_area),
            'max_area_violation': float(np.max(np.abs(final_areas - self.target_area))),
            'message': str(result.message),
            'objective_history': self.objective_history,
            'constraint_violation_history': self.constraint_violation_history
        }
    
    def check_topology_switches_needed(self, tol: float = 1e-3) -> Tuple[bool, Dict]:
        """
        Check if topology switches are needed after optimization.
        
        Returns whether any λ parameters are near boundaries (0 or 1),
        indicating that variable points want to move to adjacent edges.
        
        IMPORTANT: Boundary VPs are classified into two categories:
        1. Pure boundary VPs: Safe for Type 1 switches (not part of any triple point)
        2. Triple point boundary VPs: Should be handled via Type 2 switches
        
        This distinction is critical because applying Type 1 to a triple point VP
        would break the triple point structure instead of properly migrating it.
        
        Args:
            tol: Threshold for considering a point at boundary
            
        Returns:
            (switches_needed, switch_info) where switch_info contains details
        """
        # Step 1: Get all VPs that are part of triple points
        triple_point_vps = set()
        for tp in self.steiner_handler.triple_points:
            triple_point_vps.update(tp.var_point_indices)
        
        # Step 2: Get all boundary VPs (λ near 0 or 1)
        all_boundary_points = self.partition.get_boundary_variable_points(tol)
        
        # Step 3: Classify boundary VPs
        pure_boundary_points = [vp for vp in all_boundary_points if vp not in triple_point_vps]
        triple_point_boundary_vps = [vp for vp in all_boundary_points if vp in triple_point_vps]
        
        # Step 4: Get boundary triple points (Steiner point near mesh edge)
        boundary_triple_points = self.steiner_handler.get_boundary_triple_points(tol)
        
        # Switches needed if any boundary VPs or triple points near boundaries
        switches_needed = len(pure_boundary_points) > 0 or len(boundary_triple_points) > 0
        
        switch_info = {
            # Pure boundary VPs - safe for Type 1 switches
            'n_boundary_points': len(pure_boundary_points),
            'boundary_point_indices': pure_boundary_points,
            # Triple point boundary VPs - NOT safe for Type 1, handle via Type 2
            'n_triple_point_boundary_vps': len(triple_point_boundary_vps),
            'triple_point_boundary_vp_indices': triple_point_boundary_vps,
            # Boundary triple points - for Type 2 switches
            'n_boundary_triple_points': len(boundary_triple_points),
            'boundary_triple_point_indices': [tp.triangle_idx for tp in boundary_triple_points],
            # Total counts for logging
            'n_total_boundary_vps': len(all_boundary_points)
        }
        
        if switches_needed or len(triple_point_boundary_vps) > 0:
            self.logger.info(f"Topology switches needed:")
            self.logger.info(f"  {len(pure_boundary_points)} pure boundary VPs (Type 1 candidates)")
            if len(triple_point_boundary_vps) > 0:
                self.logger.info(f"  {len(triple_point_boundary_vps)} triple point boundary VPs (handle via Type 2)")
            self.logger.info(f"  {len(boundary_triple_points)} triple points near boundaries (Type 2 candidates)")
        
        return switches_needed, switch_info

