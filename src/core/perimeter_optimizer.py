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
    
    def __init__(self, partition: PartitionContour, mesh: TriMesh, target_area: float):
        """
        Initialize perimeter optimizer.
        
        Args:
            partition: PartitionContour with extracted contours
            mesh: TriMesh object
            target_area: Target area for each partition cell (total_area / n_cells)
        """
        self.mesh = mesh
        self.partition = partition
        self.target_area = float(target_area)
        self.logger = get_logger(__name__)
        
        # Initialize calculators
        self.area_calc = AreaCalculator(mesh, partition)
        self.perim_calc = PerimeterCalculator(mesh, partition)
        self.steiner_handler = SteinerHandler(mesh, partition)
        
        # Optimization state
        self.iteration = 0
        self.objective_history = []
        self.constraint_violation_history = []
        
        self.logger.info(f"Initialized PerimeterOptimizer:")
        self.logger.info(f"  {partition.n_cells} partition cells")
        self.logger.info(f"  {len(partition.variable_points)} variable points")
        self.logger.info(f"  {len(self.steiner_handler.triple_points)} triple points")
        self.logger.info(f"  Target area per partition cell: {target_area:.6f}")
    
    def objective(self, lambda_vec: np.ndarray) -> float:
        """
        Compute total perimeter (objective function to minimize).
        
        Includes both regular segment perimeters and Steiner tree contributions.
        
        Args:
            lambda_vec: Current variable point parameters
            
        Returns:
            Total perimeter length
        """
        self.partition.set_variable_vector(lambda_vec)
        
        # Regular perimeter from contours
        regular_perimeter = self.perim_calc.compute_total_perimeter(lambda_vec)
        
        # Steiner tree contributions from triple points
        steiner_perimeter = self.steiner_handler.get_total_perimeter_contribution()
        
        total = regular_perimeter + steiner_perimeter
        
        return total
    
    def objective_gradient(self, lambda_vec: np.ndarray) -> np.ndarray:
        """
        Compute gradient of objective function ∂(perimeter)/∂λ.
        
        Args:
            lambda_vec: Current variable point parameters
            
        Returns:
            Gradient array of shape (n_variable_points,)
        """
        self.partition.set_variable_vector(lambda_vec)
        
        # Regular perimeter gradient (analytical)
        regular_gradient = self.perim_calc.compute_total_perimeter_gradient(lambda_vec)
        
        # Steiner tree gradient (finite differences, as suggested in paper)
        steiner_gradient = self.steiner_handler.compute_total_gradient_finite_difference()
        
        total_gradient = regular_gradient + steiner_gradient
        
        return total_gradient
    
    def constraint_area_equality(self, lambda_vec: np.ndarray) -> np.ndarray:
        """
        Compute area constraint violations: Area_i - target_area.
        
        For n cells, we constrain n-1 cells (last one is determined by conservation).
        
        Args:
            lambda_vec: Current variable point parameters
            
        Returns:
            Constraint violations array of shape (n_cells - 1,)
        """
        self.partition.set_variable_vector(lambda_vec)
        
        # Compute regular areas
        areas = self.area_calc.compute_all_cell_areas(lambda_vec)
        
        # Add Steiner tree area contributions
        steiner_areas = self.steiner_handler.get_total_area_contribution()
        for cell_idx, area_contrib in steiner_areas.items():
            if cell_idx < len(areas):
                areas[cell_idx] += area_contrib
        
        # Return constraint violations for first n-1 cells
        return areas[:-1] - self.target_area
    
    def constraint_area_jacobian(self, lambda_vec: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of area constraints ∂(Area_i - target)/∂λ.
        
        Per paper section 5: Uses finite differences for Steiner area gradients.
        
        Args:
            lambda_vec: Current variable point parameters
            
        Returns:
            Jacobian array of shape (n_cells - 1, n_variable_points)
        """
        self.partition.set_variable_vector(lambda_vec)
        
        # Regular area Jacobian (from boundary triangles, analytical)
        jacobian = self.area_calc.compute_area_jacobian(lambda_vec)
        
        # Add Steiner tree area gradients (finite differences, per paper line 366)
        steiner_gradients = self.steiner_handler.compute_area_gradients_finite_difference(
            self.mesh, eps=1e-7
        )
        
        # Add Steiner contributions to jacobian (only first n_cells-1 rows)
        for cell_idx in range(self.partition.n_cells - 1):
            jacobian[cell_idx, :] += steiner_gradients[cell_idx]
        
        return jacobian
    
    def _callback(self, *args, **kwargs):
        """
        Callback function called at each optimization iteration.
        
        Handles different signatures from different optimization methods:
        - SLSQP: callback(xk) - xk is the parameter vector
        - trust-constr: callback(state) or callback(xk, res) - need to extract xk
        
        Args:
            *args: Variable arguments - first arg is typically xk or state object
            **kwargs: Keyword arguments (not used)
        """
        # Extract xk from different callback signatures
        if len(args) == 0:
            return  # No arguments, skip
        
        # For SLSQP: args[0] is xk (numpy array)
        # For trust-constr: scipy wraps and calls with (xk, res) where xk is np.ndarray
        # For trust-constr direct: args[0] might be state object with .x attribute
        if len(args) >= 2:
            # Multiple arguments: trust-constr wrapped signature (xk, res)
            # First arg should be xk (numpy array)
            if isinstance(args[0], np.ndarray):
                xk = args[0]
            else:
                # Try second argument
                xk = args[1].x if hasattr(args[1], 'x') else args[0]
        elif isinstance(args[0], np.ndarray):
            # Single numpy array: SLSQP signature
            xk = args[0]
        elif hasattr(args[0], 'x'):
            # trust-constr state object with .x attribute
            xk = args[0].x
        else:
            # Fallback: try to get x attribute
            xk = getattr(args[0], 'x', None)
            if xk is None:
                return  # Can't extract xk, skip
        
        self.iteration += 1
        
        # Compute and log progress
        obj = self.objective(xk)
        constraints = self.constraint_area_equality(xk)
        max_violation = float(np.max(np.abs(constraints)))
        
        self.objective_history.append(obj)
        self.constraint_violation_history.append(max_violation)
        
        if self.iteration % 10 == 0:
            self.logger.info(f"Iteration {self.iteration}: "
                           f"Perimeter={obj:.6f}, MaxViolation={max_violation:.2e}")
    
    def optimize(self, max_iter: int = 1000, tol: float = 1e-7,
                method: str = 'SLSQP') -> OptimizeResult:
        """
        Run constrained perimeter optimization.
        
        Uses scipy.optimize.minimize with:
        - Method: SLSQP (Sequential Least Squares Programming) by default
        - Objective: Total perimeter
        - Constraints: Equal area for each cell (n-1 constraints)
        - Bounds: λ ∈ [0, 1] for all variable points
        
        Args:
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            method: Optimization method ('SLSQP' or 'trust-constr')
            
        Returns:
            scipy OptimizeResult object
        """
        self.logger.info(f"Starting perimeter optimization with method={method}")
        self.logger.info(f"  max_iter={max_iter}, tol={tol}")
        
        # Initial guess: current λ values (all 0.5 from initialization)
        lambda0 = self.partition.get_variable_vector()
        
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
        
        # Set method-specific options
        if method == 'SLSQP':
            options = {'maxiter': max_iter, 'ftol': tol, 'disp': True}
        elif method == 'trust-constr':
            # trust-constr uses 'gtol' for gradient tolerance and 'xtol' for x tolerance
            # 'maxiter' is also supported
            options = {'maxiter': max_iter, 'gtol': tol, 'xtol': tol, 'disp': True}
        else:
            # Default options for other methods
            options = {'maxiter': max_iter, 'disp': True}
        
        result = minimize(
            fun=self.objective,
            x0=lambda0,
            method=method,
            jac=self.objective_gradient,
            bounds=bounds,
            constraints=constraints,
            callback=self._callback,
            options=options
        )
        
        elapsed_time = time.time() - start_time
        
        # Update partition with optimized parameters
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
        final_areas = self.area_calc.compute_all_cell_areas(result.x)
        
        # Add Steiner contributions to areas
        steiner_areas = self.steiner_handler.get_total_area_contribution()
        for cell_idx, area_contrib in steiner_areas.items():
            if cell_idx < len(final_areas):
                final_areas[cell_idx] += area_contrib
        
        return {
            'success': bool(result.success),
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
            
            # Get TriangleSegment for this triple point (contains vertex info)
            tri_seg = None
            for ts in self.partition.triangle_segments:
                if ts.triangle_idx == tp.triangle_idx and ts.is_triple_point():
                    tri_seg = ts
                    break
            
            if tri_seg:
                v_indices = tri_seg.vertex_indices
                v_labels = tri_seg.vertex_labels
                self.logger.info(f"  Triangle vertices: v{v_indices[0]} (Cell {v_labels[0]}), "
                               f"v{v_indices[1]} (Cell {v_labels[1]}), "
                               f"v{v_indices[2]} (Cell {v_labels[2]})")
                
                # Get Steiner point position
                if tp.steiner_point is None:
                    tp.compute_steiner_point()
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
        from .area_calculator import AreaCalculator
        from .perimeter_calculator import PerimeterCalculator
        
        self.area_calc = AreaCalculator(self.mesh, self.partition)
        self.perim_calc = PerimeterCalculator(self.mesh, self.partition)
        
        self.logger.info(f"  New triple point count: {len(self.steiner_handler.triple_points)}")

