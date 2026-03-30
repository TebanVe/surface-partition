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

try:
    from ..logging_config import get_logger
    from .tri_mesh import TriMesh
    from .contour_partition import PartitionContour
    from .area_calculator import AreaCalculator
    from .perimeter_calculator import PerimeterCalculator
    from .steiner_handler import SteinerHandler
    from .partition_arrays import PartitionArrays
    from . import vectorized_perimeter
    from . import vectorized_area
    from . import vectorized_steiner
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from logging_config import get_logger
    from core.tri_mesh import TriMesh
    from core.contour_partition import PartitionContour
    from core.area_calculator import AreaCalculator
    from core.perimeter_calculator import PerimeterCalculator
    from core.steiner_handler import SteinerHandler
    from core.partition_arrays import PartitionArrays
    from core import vectorized_perimeter
    from core import vectorized_area
    from core import vectorized_steiner


class IPOPTProblemAdapter:
    """Wraps PerimeterOptimizer callbacks into the cyipopt problem interface.

    cyipopt requires a class with specific method names (objective, gradient,
    constraints, jacobian, jacobianstructure). This adapter delegates to the
    existing PerimeterOptimizer methods, which handle the active/inactive VP
    mapping internally.
    """

    def __init__(self, optimizer: 'PerimeterOptimizer'):
        self._opt = optimizer
        self._pa = optimizer._arrays   # PartitionArrays snapshot

    def objective(self, x: np.ndarray) -> float:
        return self._opt.objective(x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return self._opt.objective_gradient(x)

    def constraints(self, x: np.ndarray) -> np.ndarray:
        return self._opt.constraint_area_equality(x)

    def jacobianstructure(self) -> tuple:
        """Pre-computed sparsity pattern — called once at setup."""
        return (self._pa.jac_row, self._pa.jac_col)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Return non-zero Jacobian values in jacobianstructure() order.

        Phase 1: compute the dense Jacobian and extract non-zeros by indexing.
        This is correct and fast enough at current scale. Phase 2 will replace
        this with a direct sparse computation.
        """
        J_dense = self._opt.constraint_area_jacobian(x)   # (n_cells-1, n_vp)
        dense_mb = J_dense.nbytes / 1e6
        if dense_mb > 50:
            import warnings
            warnings.warn(
                f"IPOPT Jacobian: dense matrix is {dense_mb:.0f} MB. "
                f"Implement Phase 2 (sparse Jacobian) for better scaling. "
                f"See docs/IPOPT_INTEGRATION_PLAN.md, Section 4, Phase 2.",
                stacklevel=2,
            )
        return J_dense[self._pa.jac_row, self._pa.jac_col]

    def intermediate(self, alg_mod, iter_count, obj_value,
                     inf_pr, inf_du, mu, d_norm,
                     regularization_size, alpha_du, alpha_pr, ls_trials):
        """Called by IPOPT after each iteration. Log progress."""
        import logging
        logger = logging.getLogger(__name__)
        if iter_count % 10 == 0:
            logger.info(f"IPOPT iter {iter_count}: obj={obj_value:.6f}, "
                        f"constr_viol={inf_pr:.2e}")
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
                method: str = 'SLSQP') -> OptimizeResult:
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
            
        Returns:
            scipy OptimizeResult object
        """
        self.logger.info(f"Starting perimeter optimization with method={method}")
        self.logger.info(f"  max_iter={max_iter}, tol={tol}")
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

            adapter = IPOPTProblemAdapter(self)

            problem = cyipopt.Problem(
                n=n,
                m=m,
                problem_obj=adapter,
                lb=np.zeros(n),
                ub=np.ones(n),
                cl=np.zeros(m),
                cu=np.zeros(m),
            )

            problem.add_option('hessian_approximation', 'limited-memory')
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

            result = OptimizeResult(
                x=x_opt,
                success=success,
                message=msg,
                fun=info['obj_val'],
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
    
    def diagnose_boundary_triple_points(self, tol: float = 1e-3) -> None:
        """
        Print diagnostic information about triple points near mesh triangle boundaries.
        
        For each boundary triple point, shows:
        - Mesh triangle index and vertex labels
        - Distance from Steiner point to each mesh triangle edge
        - Which edge is closest
        - Lambda values of the three variable points forming the void triangle
        
        Args:
            tol: Tolerance used for boundary detection
        """
        boundary_triple_points = self.steiner_handler.get_boundary_triple_points(tol)
        
        if not boundary_triple_points:
            return
        
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("BOUNDARY TRIPLE POINT DIAGNOSTICS")
        self.logger.info("=" * 80)
        
        for i, tp in enumerate(boundary_triple_points):
            self.logger.info("")
            self.logger.info(f"Triple Point {i+1}/{len(boundary_triple_points)}:")
            self.logger.info(f"  Mesh Triangle: {tp.triangle_idx}")
            self.logger.info(f"  Cells meeting: {sorted(tp.cell_indices)}")
            
            tri_seg = self.partition.get_triangle_segment(tp.triangle_idx)
            
            if tri_seg and tri_seg.is_triple_point():
                v_indices = tri_seg.vertex_indices
                v_labels = tuple(int(self.partition.vertex_labels[v]) for v in v_indices)
                self.logger.info(f"  Triangle vertices: v{v_indices[0]} (Cell {v_labels[0]}), "
                               f"v{v_indices[1]} (Cell {v_labels[1]}), "
                               f"v{v_indices[2]} (Cell {v_labels[2]})")
                
                # Get Steiner point position
                if tp.steiner_point is None:
                    tp.compute_steiner_point(partition=self.partition)
                steiner_pos = tp.steiner_point
                
                # Check distance to each edge of the mesh triangle
                vertices = [self.mesh.vertices[v_indices[i]] for i in range(3)]
                
                edges = [
                    ((v_indices[0], v_indices[1]), vertices[0], vertices[1], f"({v_indices[0]}, {v_indices[1]})"),
                    ((v_indices[1], v_indices[2]), vertices[1], vertices[2], f"({v_indices[1]}, {v_indices[2]})"),
                    ((v_indices[2], v_indices[0]), vertices[2], vertices[0], f"({v_indices[2]}, {v_indices[0]})")
                ]
                
                closest_edge = None
                min_dist = float('inf')
                
                for edge_vertices, p_start, p_end, edge_name in edges:
                    dist = tp._point_to_segment_distance(steiner_pos, p_start, p_end)
                    self.logger.info(f"  Distance to edge {edge_name}: {dist:.6e}")
                    
                    if dist < min_dist:
                        min_dist = dist
                        closest_edge = (edge_vertices, edge_name, dist)
                
                if closest_edge:
                    self.logger.info(f"  → Steiner point closest to edge {closest_edge[1]} (dist = {closest_edge[2]:.6e})")
            else:
                self.logger.warning(f"  Could not find TriangleSegment for this triple point")
            
            # Get variable points information
            self.logger.info(f"  Variable points forming void triangle:")
            
            for vp_idx in tp.var_point_indices:
                vp = self.partition.variable_points[vp_idx]
                lambda_val = vp.lambda_param
                
                # Indicate if near boundary
                near_0 = lambda_val < tol
                near_1 = lambda_val > (1.0 - tol)
                
                boundary_indicator = ""
                if near_0:
                    boundary_indicator = f" ← NEAR 0 (within tol={tol})"
                elif near_1:
                    boundary_indicator = f" ← NEAR 1 (within tol={tol})"
                
                self.logger.info(f"    VP {vp_idx}: λ = {lambda_val:.6f}, "
                              f"edge {vp.edge}{boundary_indicator}")
        
        self.logger.info("")
        self.logger.info("=" * 80)
    
    def apply_topology_switches(self, switch_info: Dict, switch_tol: float = 1e-3) -> int:
        """
        Apply topology switches (Type 1 and/or Type 2) based on detection results.
        
        IMPORTANT: Type 2 switches are applied FIRST, then Type 1 switches.
        This ensures triple point VPs are properly migrated before any
        individual VP moves that could break triple point structure.
        
        Type 1 switches are only applied to "pure" boundary VPs (not part of any
        triple point). Triple point boundary VPs are handled via Type 2 migration.
        
        Args:
            switch_info: Dict from check_topology_switches_needed()
            switch_tol: Tolerance for switch detection (default 1e-3)
            
        Returns:
            Number of variable points moved
        """
        from .mesh_topology import MeshTopology
        from .topology_switcher import TopologySwitcher
        
        # Create topology switcher (lightweight, just stores references)
        mesh_topology = MeshTopology(self.mesh)
        switcher = TopologySwitcher(self.mesh, self.partition, mesh_topology)
        
        total_vp_moves = 0
        
        n_type1_detected = switch_info['n_boundary_points']  # Pure boundary VPs only
        n_type2_detected = switch_info['n_boundary_triple_points']
        n_triple_point_boundary = switch_info.get('n_triple_point_boundary_vps', 0)
        
        # Log if there are triple point boundary VPs that will be handled via Type 2
        if n_triple_point_boundary > 0:
            self.logger.info(f"Note: {n_triple_point_boundary} boundary VPs are part of triple points")
            self.logger.info(f"  These will be handled via Type 2 migration, not Type 1 switches")
        
        # Handle Type 2 switches FIRST (boundary triple points)
        # This ensures proper migration before any Type 1 switches
        if n_type2_detected > 0:
            self.logger.info(f"Applying Type 2 switches (triple point migration)...")
            n_type2_applied = 0
            
            # Get boundary triple points fresh (they may have been updated)
            boundary_triple_points = self.steiner_handler.get_boundary_triple_points(switch_tol)
            
            for tp in boundary_triple_points:
                # Apply proper Type 2 switch (handles segment topology correctly)
                if switcher.apply_type2_switch(tp, tol=0.1):
                    n_type2_applied += 1
                    total_vp_moves += 1
            
            self.logger.info(f"  ✓ Applied {n_type2_applied}/{n_type2_detected} Type 2 switches")
            
            # Rebuild after Type 2 switches
            if n_type2_applied > 0:
                self.partition.rebuild_triangle_segments_from_current_vps()
                self.steiner_handler = SteinerHandler(self.mesh, self.partition)
        
        # Handle Type 1 switches (pure boundary VPs only - NOT part of triple points)
        if n_type1_detected > 0:
            self.logger.info(f"Applying Type 1 switches (pure boundary VPs)...")
            
            # Filter connected VPs (keep closest in each component)
            boundary_vps = switch_info['boundary_point_indices']
            filtered_vps = self._filter_connected_boundary_vps(boundary_vps)
            
            # Sort remaining VPs by distance to target vertex (closest first)
            filtered_vps_sorted = sorted(
                filtered_vps,
                key=lambda vp_idx: self._compute_boundary_distance(vp_idx)
            )
            
            self.logger.info(f"  Processing {len(filtered_vps_sorted)} VPs (filtered from {len(boundary_vps)})")
            
            # Apply switches in priority order
            n_type1_applied = 0
            for vp_idx in filtered_vps_sorted:
                dist = self._compute_boundary_distance(vp_idx)
                self.logger.debug(f"    VP {vp_idx}: distance = {dist:.6f}")
                if switcher.apply_type1_switch(vp_idx, tol=0.1):
                    n_type1_applied += 1
                    total_vp_moves += 1
            
            self.logger.info(f"  ✓ Applied {n_type1_applied}/{len(filtered_vps_sorted)} Type 1 switches")
        
        if total_vp_moves == 0:
            self.logger.warning(f"⚠ Switches detected but none could be applied")
        
        return total_vp_moves
    
    def _compute_boundary_distance(self, vp_idx: int) -> float:
        """
        Compute how far a boundary VP is from its target vertex.
        
        For λ < 0.5: VP approaching edge[1], distance = λ
        For λ > 0.5: VP approaching edge[0], distance = (1 - λ)
        
        Args:
            vp_idx: Variable point index
            
        Returns:
            Distance in [0, 0.5], where smaller = closer to target vertex
        """
        vp = self.partition.variable_points[vp_idx]
        if vp.lambda_param < 0.5:
            return vp.lambda_param
        else:
            return 1.0 - vp.lambda_param
    
    def _find_connected_components(self, boundary_vps: Set[int]) -> List[Set[int]]:
        """
        Find connected components of boundary VPs using DFS.
        
        VPs are connected if they form a segment (edge of partition boundary).
        
        Args:
            boundary_vps: Set of boundary VP indices
            
        Returns:
            List of sets, each set is a connected component of VP indices
        """
        from collections import defaultdict
        
        # Build adjacency list (only for boundary VPs)
        adjacency = defaultdict(set)
        for segment in self.partition.boundary_segments:
            vp1, vp2 = segment.vp_idx_1, segment.vp_idx_2
            if vp1 in boundary_vps and vp2 in boundary_vps:
                adjacency[vp1].add(vp2)
                adjacency[vp2].add(vp1)
        
        # Find connected components using DFS
        visited = set()
        components = []
        
        for vp_idx in boundary_vps:
            if vp_idx in visited:
                continue
            
            # DFS to find component
            component = set()
            stack = [vp_idx]
            
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                
                visited.add(current)
                component.add(current)
                
                # Add unvisited neighbors
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        stack.append(neighbor)
            
            components.append(component)
        
        return components
    
    def _filter_connected_boundary_vps(self, boundary_vps: List[int]) -> List[int]:
        """
        Filter boundary VPs to keep only one per connected component.
        
        For each connected component of boundary VPs:
        - Find the VP closest to its target vertex
        - Keep only that VP
        - Remove all others in the component
        
        This ensures we don't switch multiple connected VPs simultaneously,
        which would invalidate the segments between them.
        
        Args:
            boundary_vps: List of pure boundary VP indices (not in triple points)
            
        Returns:
            Filtered list with one VP per connected component
        """
        if not boundary_vps:
            return []
        
        boundary_set = set(boundary_vps)
        
        # Find connected components
        components = self._find_connected_components(boundary_set)
        
        vps_to_keep = []
        vps_deferred = []
        
        self.logger.info(f"  Found {len(components)} connected component(s) among {len(boundary_vps)} boundary VPs")
        
        for i, component in enumerate(components):
            if len(component) == 1:
                # Single VP - always keep
                vp_idx = list(component)[0]
                vps_to_keep.append(vp_idx)
                self.logger.debug(f"    Component {i+1}: Single VP {vp_idx}")
            else:
                # Multiple connected VPs - keep closest
                vps_with_dist = [
                    (self._compute_boundary_distance(vp), vp)
                    for vp in component
                ]
                vps_with_dist.sort()  # Sort by distance (closest first)
                
                closest_dist, closest_vp = vps_with_dist[0]
                vps_to_keep.append(closest_vp)
                
                # Defer all others
                deferred_in_component = [vp for _, vp in vps_with_dist[1:]]
                vps_deferred.extend(deferred_in_component)
                
                self.logger.info(f"    Component {i+1}: {len(component)} connected VPs")
                self.logger.info(f"      Keeping VP {closest_vp} (distance={closest_dist:.6f})")
                if len(deferred_in_component) <= 3:
                    self.logger.info(f"      Deferring: {deferred_in_component}")
                else:
                    self.logger.info(f"      Deferring: {len(deferred_in_component)} VPs")
        
        if vps_deferred:
            self.logger.info(f"  Total deferred: {len(vps_deferred)} VPs (will be reconsidered in next iteration)")
        
        return vps_to_keep
    
    def reinitialize_after_switches(self) -> None:
        """
        Re-initialize calculators after topology switches.
        
        Must be called after apply_topology_switches() to update:
        - PartitionContour's triangle_segments (rebuilt based on current VPs)
        - Steiner handler (triple points may be in different triangles)
        - Area calculator (boundary triangles may have changed)
        - Perimeter calculator (fresh state)
        
        This is a helper method for the manual loop in refine_perimeter.py.
        """
        self.logger.info("Re-initializing calculators after topology switches...")
        
        # CRITICAL FIX (Issue 1): Rebuild partition.triangle_segments first
        self.partition.rebuild_triangle_segments_from_current_vps()
        
        # Re-initialize Steiner handler (finds triple points in new triangles)
        self.steiner_handler = SteinerHandler(self.mesh, self.partition)
        
        # Re-initialize calculators (boundary triangles may have changed)
        # IMPORTANT: Use VP-based categorization after topology switches
        from .area_calculator import AreaCalculator
        from .perimeter_calculator import PerimeterCalculator
        
        self.area_calc = AreaCalculator(self.mesh, self.partition)
        self.perim_calc = PerimeterCalculator(self.mesh, self.partition)
        
        self.logger.info(f"  New triple point count: {len(self.steiner_handler.triple_points)}")

