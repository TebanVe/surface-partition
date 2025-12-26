"""
Steiner tree handling for triple points in partition contours.

This module implements the triple point treatment from Section 5 of the paper:
"The empty spaces around triple points". 

TERMINOLOGY (following paper Section 5):
- "cell": Partition region (what we optimize for equal areas)
- "triangle": Mesh triangle element (computational discretization)
- "edge": Mesh triangle edge (computational discretization)

When three different partition cells meet at a mesh triangle, small void spaces are created.
These are filled with Steiner trees that:
1. Connect three variable points on the mesh triangle's edges
2. Meet at an optimal Steiner point that minimizes total edge length
3. Divide the void area among the three adjacent partition cells

The Steiner point satisfies the Fermat point property: edges meet at 120° angles.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.optimize import minimize

try:
    from ..logging_config import get_logger
    from .tri_mesh import TriMesh
    from .contour_partition import PartitionContour
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from logging_config import get_logger
    from core.tri_mesh import TriMesh
    from core.contour_partition import PartitionContour


class TriplePoint:
    """
    Represents a triple point where three partition cells meet.
    
    At a triple point, three variable points on a mesh triangle's edges form a small
    void space. A Steiner point is computed to optimally connect these three points,
    minimizing the total perimeter contribution.
    
    Attributes:
        triangle_idx: Index of the mesh triangle containing this triple point
        var_point_indices: List of 3 variable point indices on the mesh triangle edges
        cell_indices: List of 3 partition cell indices that meet here
        steiner_point: Optimal connection point (computed)
        boundary_points: 3D/2D coordinates of the three variable points
    """
    
    def __init__(self, triangle_idx: int, var_point_indices: List[int],
                 partition: PartitionContour):
        """
        Initialize triple point.
        
        Args:
            triangle_idx: Index of triangle in mesh
            var_point_indices: List of 3 variable point indices
            partition: PartitionContour object
        """
        self.triangle_idx = triangle_idx
        self.var_point_indices = var_point_indices
        self.partition = partition
        self.logger = get_logger(__name__)
        
        # Determine which partition cells meet at this point
        self.cell_indices: List[int] = []
        for vp_idx in var_point_indices:
            vp = partition.variable_points[vp_idx]
            self.cell_indices.extend(list(vp.belongs_to_cells))
        # Get unique cells (should be exactly 3)
        self.cell_indices = list(set(self.cell_indices))
        
        if len(self.cell_indices) != 3:
            self.logger.warning(f"Triple point at mesh triangle {triangle_idx} has "
                              f"{len(self.cell_indices)} partition cells, expected 3")
        
        # Compute cell-to-variable-point mapping for perimeter calculation
        # Per Figure 7: each cell gets perimeter from 2 variable points on edges opposite its vertex
        self.cell_to_varpoint_pair: Dict[int, Tuple[int, int]] = {}
        
        # Map cell_idx -> mesh_vertex_idx for the corner vertex of each cell
        # This is needed to compute corner areas (from mesh vertex to adjacent VPs)
        self.cell_to_mesh_vertex: Dict[int, int] = {}
        
        self._compute_cell_to_varpoint_mapping()
        
        # Will be computed
        self.steiner_point: Optional[np.ndarray] = None
        self.boundary_points: Optional[List[np.ndarray]] = None
    
    def _compute_cell_to_varpoint_mapping(self):
        """
        Compute mapping from each cell to the 2 variable points that define its boundary.
        
        Per Figure 7: For a triple point with triangle ABC:
        - Cell at vertex A gets boundary from edges BC (opposite to A)
        - Cell at vertex B gets boundary from edges CA (opposite to B)
        - Cell at vertex C gets boundary from edges AB (opposite to C)
        
        This determines which 2 Steiner edges and which 1 original edge
        contribute to each cell's perimeter.
        """
        # Get the TriangleSegment for this triple point
        tri_seg = None
        for ts in self.partition.triangle_segments:
            if ts.triangle_idx == self.triangle_idx and ts.is_triple_point():
                tri_seg = ts
                break
        
        if tri_seg is None:
            self.logger.warning(f"Could not find TriangleSegment for triple point at triangle {self.triangle_idx}")
            return
        
        # Extract mesh triangle vertices and their cell labels
        vertices = tri_seg.vertex_indices  # (v1, v2, v3)
        labels = tri_seg.vertex_labels     # (label1, label2, label3)
        
        # For each vertex (and its cell), find the 2 variable points on opposite edges
        for i in range(3):
            vertex = vertices[i]
            cell_label = labels[i]
            
            # Store which mesh vertex belongs to this cell (for corner area calculation)
            self.cell_to_mesh_vertex[cell_label] = int(vertex)
            
            # The two "opposite" edges don't include this vertex
            # They connect the other two vertices
            other_idx1 = (i + 1) % 3
            other_idx2 = (i + 2) % 3
            
            # Edge between the other two vertices
            edge1 = tuple(sorted([vertices[other_idx1], vertices[other_idx2]]))
            # Edges from this vertex to each of the other vertices
            edge2 = tuple(sorted([vertices[i], vertices[other_idx1]]))
            edge3 = tuple(sorted([vertices[i], vertices[other_idx2]]))
            
            # Find which 2 of our 3 variable points are on the edges opposite to this vertex
            # The opposite edges are edge2 and edge3 (emanating from the other two vertices)
            vp_indices_for_cell = []
            for vp_idx in self.var_point_indices:
                vp = self.partition.variable_points[vp_idx]
                # Check if this variable point is on one of the edges opposite to our vertex
                if vp.edge == edge2 or vp.edge == edge3:
                    vp_indices_for_cell.append(vp_idx)
            
            if len(vp_indices_for_cell) == 2:
                self.cell_to_varpoint_pair[cell_label] = tuple(vp_indices_for_cell)
            else:
                self.logger.warning(f"Expected 2 variable points for cell {cell_label} at triple point {self.triangle_idx}, found {len(vp_indices_for_cell)}")
    
    def compute_steiner_point(self) -> np.ndarray:
        """
        Compute optimal Steiner point that minimizes total edge length.
        
        For three points p1, p2, p3, find point S that minimizes:
            f(S) = ||S - p1|| + ||S - p2|| + ||S - p3||
        
        This is the geometric median (Fermat point) problem.
        For a proper Steiner tree, edges meet at 120° angles.
        
        Returns:
            Optimal Steiner point coordinates (2D or 3D)
        """
        # Evaluate the three variable points
        self.boundary_points = [
            self.partition.evaluate_variable_point(vp_idx)
            for vp_idx in self.var_point_indices
        ]
        
        p1, p2, p3 = self.boundary_points
        
        # Initial guess: centroid of the three points
        initial_guess = (p1 + p2 + p3) / 3.0
        
        # Objective: sum of distances
        def objective(S):
            return (np.linalg.norm(S - p1) + 
                   np.linalg.norm(S - p2) + 
                   np.linalg.norm(S - p3))
        
        # Gradient of objective
        def gradient(S):
            grad = np.zeros_like(S)
            for p in [p1, p2, p3]:
                dist = np.linalg.norm(S - p)
                if dist > 1e-12:
                    grad += (S - p) / dist
            return grad
        
        # Optimize
        result = minimize(objective, initial_guess, jac=gradient, method='BFGS',
                         options={'gtol': 1e-8})
        
        self.steiner_point = result.x
        return self.steiner_point
    
    def _triangle_area_3d(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Compute area of triangle with vertices p1, p2, p3.
        
        Works for both 2D and 3D triangles.
        
        Args:
            p1, p2, p3: Triangle vertices (2D or 3D coordinates)
            
        Returns:
            Triangle area
        """
        v1 = p2 - p1
        v2 = p3 - p1
        
        if len(p1) == 2:
            # 2D case: use cross product formula
            area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
        else:
            # 3D case: use cross product magnitude
            cross = np.cross(v1, v2)
            area = 0.5 * np.linalg.norm(cross)
        
        return area
    
    def get_perimeter_contribution(self) -> Dict[int, float]:
        """
        Compute NET perimeter contribution to each adjacent cell.
        
        Per paper Figure 7: Each cell gets (2 Steiner edges - 1 original edge).
        
        For triple point with triangle ABC and Fermat point X:
        - Cell at vertex A (opposite to edge BC) gets: BX + CX - BC
        - Cell at vertex B (opposite to edge CA) gets: CX + AX - CA
        - Cell at vertex C (opposite to edge AB) gets: AX + BX - AB
        
        Returns:
            Dict mapping cell_idx -> NET perimeter contribution
        """
        if self.steiner_point is None:
            self.compute_steiner_point()
        
        contributions = {cell_idx: 0.0 for cell_idx in self.cell_indices}
        
        # For each cell, compute: dist(vp1, X) + dist(vp2, X) - dist(vp1, vp2)
        for cell_idx, (vp_idx1, vp_idx2) in self.cell_to_varpoint_pair.items():
            # Get variable point positions
            vp_pos1 = self.partition.evaluate_variable_point(vp_idx1)
            vp_pos2 = self.partition.evaluate_variable_point(vp_idx2)
            
            # Two Steiner edges (from variable points to Fermat point X)
            steiner_edge1 = np.linalg.norm(vp_pos1 - self.steiner_point)
            steiner_edge2 = np.linalg.norm(vp_pos2 - self.steiner_point)
            
            # One original edge between the two variable points (to subtract)
            original_edge = np.linalg.norm(vp_pos1 - vp_pos2)
            
            # NET contribution per Figure 7: (2 Steiner edges) - (1 original edge)
            contributions[cell_idx] = steiner_edge1 + steiner_edge2 - original_edge
        
        return contributions
    
    def get_area_contribution(self, mesh: TriMesh) -> Dict[int, float]:
        """
        Compute area contribution to each adjacent cell.
        
        CRITICAL FIX: This method now correctly computes BOTH:
        1. Void interior area (Steiner subdivisions) - as per paper Figure 7
        2. Corner area (from mesh vertex to adjacent VPs) - previously missing!
        
        Per Figure 7 of the paper: The triangular void ABC is divided into three 
        sub-triangles by the Steiner point X. Each sub-triangle is assigned to the
        corresponding cell. For example, area of ABX is added to Cell 3.
        
        HOWEVER: The paper implicitly assumes that corner areas (from mesh vertices
        to VPs) are handled elsewhere. Since AreaCalculator SKIPS entire triple point
        triangles, we must add corners here.
        
        Total area for each cell = Void interior + Corner region
        
        Returns:
            Dict mapping cell_idx -> total area contribution (void + corner)
        """
        if self.steiner_point is None:
            self.compute_steiner_point()
        
        contributions = {cell_idx: 0.0 for cell_idx in self.cell_indices}
        
        # Get the three variable points positions
        var_points_pos = [self.partition.evaluate_variable_point(vp_idx) 
                          for vp_idx in self.var_point_indices]
        
        # For each cell: compute void interior + corner area
        for cell_idx in self.cell_indices:
            if cell_idx not in self.cell_to_varpoint_pair:
                # Fallback: equal division (shouldn't happen if mapping is correct)
                triangle_area = mesh.triangle_areas[self.triangle_idx]
                contributions[cell_idx] = triangle_area / 3.0
                self.logger.warning(f"Cell {cell_idx} not in cell_to_varpoint_pair for "
                                  f"triple point at triangle {self.triangle_idx}")
                continue
            
            # Get the two variable point indices for this cell
            vp_idx1, vp_idx2 = self.cell_to_varpoint_pair[cell_idx]
            
            # Find their positions
            idx1_in_list = self.var_point_indices.index(vp_idx1)
            idx2_in_list = self.var_point_indices.index(vp_idx2)
            pos1 = var_points_pos[idx1_in_list]
            pos2 = var_points_pos[idx2_in_list]
            
            # PART 1: Void interior area (per Figure 7)
            # Triangle formed by (vp1, vp2, steiner_point)
            void_area = self._triangle_area_3d(pos1, pos2, self.steiner_point)
            
            # PART 2: Corner area (CRITICAL FIX - was missing!)
            # Triangle formed by (mesh_vertex, vp1, vp2)
            if cell_idx in self.cell_to_mesh_vertex:
                mesh_vertex_idx = self.cell_to_mesh_vertex[cell_idx]
                vertex_pos = mesh.vertices[mesh_vertex_idx]
                corner_area = self._triangle_area_3d(vertex_pos, pos1, pos2)
            else:
                corner_area = 0.0
                self.logger.warning(f"Cell {cell_idx} not in cell_to_mesh_vertex for "
                                  f"triple point at triangle {self.triangle_idx}")
            
            # TOTAL area contribution
            contributions[cell_idx] = void_area + corner_area
        
        return contributions
    
    def compute_area_gradients_finite_difference(self, mesh: TriMesh, 
                                                 eps: float = 1e-7) -> Dict[int, np.ndarray]:
        """
        Compute gradients of Steiner tree area contributions w.r.t. λ parameters.
        
        Per paper (line 366): "In order to find the gradient corresponding to the 
        lengths and area changes due to the addition of these Steiner points we use 
        a finite differences approximation."
        
        Pattern follows compute_gradients_finite_difference() for perimeter.
        
        Args:
            mesh: TriMesh object  
            eps: Perturbation size for finite differences
            
        Returns:
            Dict mapping cell_idx -> gradient array of shape (n_variable_points,)
        """
        n_vars = len(self.partition.variable_points)
        gradients = {cell_idx: np.zeros(n_vars) for cell_idx in self.cell_indices}
        
        # Base area contributions
        base_contrib = self.get_area_contribution(mesh)
        
        # Perturb each λ parameter that affects this triple point
        for vp_idx in self.var_point_indices:
            old_lambda = self.partition.variable_points[vp_idx].lambda_param
            
            # Perturb
            self.partition.variable_points[vp_idx].lambda_param = old_lambda + eps
            
            # Recompute Steiner point and areas
            self.steiner_point = None  # Force recomputation
            perturbed_contrib = self.get_area_contribution(mesh)
            
            # Gradient via finite difference for each cell
            for cell_idx in self.cell_indices:
                gradients[cell_idx][vp_idx] = (perturbed_contrib[cell_idx] - 
                                               base_contrib[cell_idx]) / eps
            
            # Restore original value
            self.partition.variable_points[vp_idx].lambda_param = old_lambda
        
        # Force recomputation with original values
        self.steiner_point = None
        self.compute_steiner_point()
        
        return gradients
    
    def get_segments(self) -> Dict[int, List[np.ndarray]]:
        """
        Get Steiner tree segments for visualization.
        
        Returns:
            Dict mapping cell_idx -> list of segment arrays (2, D)
        """
        if self.steiner_point is None:
            self.compute_steiner_point()
        
        segments_dict = {cell_idx: [] for cell_idx in self.cell_indices}
        
        # Create segments from each variable point to Steiner point
        for vp_idx in self.var_point_indices:
            vp_pos = self.partition.evaluate_variable_point(vp_idx)
            segment = np.vstack([vp_pos, self.steiner_point])
            
            # Add to partition cells that this variable point belongs to
            vp = self.partition.variable_points[vp_idx]
            for cell_idx in vp.belongs_to_cells:
                if cell_idx in segments_dict:
                    segments_dict[cell_idx].append(segment)
        
        return segments_dict
    
    def compute_gradients_finite_difference(self, eps: float = 1e-6) -> np.ndarray:
        """
        Compute gradients of Steiner tree perimeter w.r.t. λ parameters.
        
        As suggested in the paper, use finite differences for Steiner point gradients
        since analytical derivatives are complex.
        
        Args:
            eps: Perturbation size for finite differences
            
        Returns:
            Gradient array of shape (n_variable_points,)
        """
        n_vars = len(self.partition.variable_points)
        gradient = np.zeros(n_vars)
        
        # Base perimeter contribution
        base_contrib = self.get_perimeter_contribution()
        base_total = sum(base_contrib.values())
        
        # Perturb each λ parameter that affects this triple point
        for vp_idx in self.var_point_indices:
            old_lambda = self.partition.variable_points[vp_idx].lambda_param
            
            # Perturb
            self.partition.variable_points[vp_idx].lambda_param = old_lambda + eps
            
            # Recompute Steiner point and perimeter
            self.steiner_point = None  # Force recomputation
            perturbed_contrib = self.get_perimeter_contribution()
            perturbed_total = sum(perturbed_contrib.values())
            
            # Gradient via finite difference
            gradient[vp_idx] = (perturbed_total - base_total) / eps
            
            # Restore original value
            self.partition.variable_points[vp_idx].lambda_param = old_lambda
        
        # Force recomputation with original values
        self.steiner_point = None
        self.compute_steiner_point()
        
        return gradient
    
    def is_on_triangle_boundary(self, tol: float = 1e-3) -> bool:
        """
        Check if Steiner point is near the triangle boundary.
        
        If true, topology switch may be needed (expand to adjacent triangle).
        
        Args:
            tol: Distance tolerance
            
        Returns:
            True if Steiner point is within tol of any triangle edge
        """
        if self.steiner_point is None:
            self.compute_steiner_point()
        
        # Get triangle vertices
        face = self.partition.mesh.faces[self.triangle_idx]
        v1, v2, v3 = [int(i) for i in face]
        p1 = self.partition.mesh.vertices[v1]
        p2 = self.partition.mesh.vertices[v2]
        p3 = self.partition.mesh.vertices[v3]
        
        # Check distance to each edge
        edges = [(p1, p2), (p2, p3), (p3, p1)]
        for edge_start, edge_end in edges:
            dist = self._point_to_segment_distance(self.steiner_point, edge_start, edge_end)
            if dist < tol:
                return True
        
        return False
    
    def _point_to_segment_distance(self, point: np.ndarray, 
                                   seg_start: np.ndarray, 
                                   seg_end: np.ndarray) -> float:
        """
        Compute distance from point to line segment.
        
        Args:
            point: Query point
            seg_start: Segment start point
            seg_end: Segment end point
            
        Returns:
            Minimum distance
        """
        # Vector from start to end
        v = seg_end - seg_start
        # Vector from start to point
        w = point - seg_start
        
        # Projection parameter
        c1 = np.dot(w, v)
        if c1 <= 0:
            return np.linalg.norm(point - seg_start)
        
        c2 = np.dot(v, v)
        if c1 >= c2:
            return np.linalg.norm(point - seg_end)
        
        # Project onto segment
        b = c1 / c2
        proj = seg_start + b * v
        return np.linalg.norm(point - proj)


class SteinerHandler:
    """
    Manages all triple points in the partition and their Steiner trees.
    
    Attributes:
        mesh: The underlying TriMesh
        partition: PartitionContour with variable points
        triple_points: List of detected TriplePoint objects
    """
    
    def __init__(self, mesh: TriMesh, partition: PartitionContour):
        """
        Initialize Steiner handler and detect triple points.
        
        Args:
            mesh: TriMesh object
            partition: PartitionContour with variable points
        """
        self.mesh = mesh
        self.partition = partition
        self.logger = get_logger(__name__)
        
        # Detect and create triple points
        self.triple_points: List[TriplePoint] = []
        self._detect_triple_points()
    
    def _detect_triple_points(self):
        """
        Detect all triple points in the partition.
        
        Uses VP-based detection (identify_triple_points_from_current_vps) which works
        correctly after topology switches, rather than the old indicator-based method
        which relies on stale indicator_functions.
        """
        triple_point_data = self.partition.identify_triple_points_from_current_vps()
        
        for tri_idx, var_point_indices in triple_point_data:
            tp = TriplePoint(tri_idx, var_point_indices, self.partition)
            tp.compute_steiner_point()
            self.triple_points.append(tp)
        
        self.logger.info(f"Detected and initialized {len(self.triple_points)} triple points")
    
    def get_total_perimeter_contribution(self) -> float:
        """
        Compute total perimeter contribution from all Steiner trees.
        
        Returns:
            Sum of perimeter contributions from all triple points
        """
        total = 0.0
        for tp in self.triple_points:
            contrib = tp.get_perimeter_contribution()
            total += sum(contrib.values())
        
        return total
    
    def get_total_area_contribution(self) -> Dict[int, float]:
        """
        Compute total area contribution from all Steiner trees to each cell.
        
        Returns:
            Dict mapping cell_idx -> total additional area
        """
        contributions = {i: 0.0 for i in range(self.partition.n_cells)}
        
        for tp in self.triple_points:
            area_contrib = tp.get_area_contribution(self.mesh)
            for cell_idx, area in area_contrib.items():
                contributions[cell_idx] += area
        
        return contributions
    
    def compute_total_gradient_finite_difference(self, eps: float = 1e-6) -> np.ndarray:
        """
        Compute total gradient contribution from all Steiner trees.
        
        Args:
            eps: Perturbation size for finite differences
            
        Returns:
            Gradient array of shape (n_variable_points,)
        """
        gradient = np.zeros(len(self.partition.variable_points))
        
        for tp in self.triple_points:
            gradient += tp.compute_gradients_finite_difference(eps)
        
        return gradient
    
    def compute_area_gradients_finite_difference(self, mesh: TriMesh, 
                                                 eps: float = 1e-7) -> Dict[int, np.ndarray]:
        """
        Compute total area gradient contributions from all Steiner trees.
        
        Pattern follows compute_total_gradient_finite_difference() for perimeter.
        
        Args:
            mesh: TriMesh object
            eps: Perturbation size for finite differences
            
        Returns:
            Dict mapping cell_idx -> gradient array of shape (n_variable_points,)
        """
        n_vars = len(self.partition.variable_points)
        gradients = {i: np.zeros(n_vars) for i in range(self.partition.n_cells)}
        
        for tp in self.triple_points:
            tp_gradients = tp.compute_area_gradients_finite_difference(mesh, eps)
            for cell_idx, grad in tp_gradients.items():
                gradients[cell_idx] += grad
        
        return gradients
    
    def get_boundary_triple_points(self, tol: float = 1e-3) -> List[TriplePoint]:
        """
        Find triple points with Steiner point near triangle boundary.
        
        These may need topology switches (expansion to adjacent triangles).
        
        Args:
            tol: Distance tolerance
            
        Returns:
            List of TriplePoint objects near boundaries
        """
        boundary_tps = []
        for tp in self.triple_points:
            if tp.is_on_triangle_boundary(tol):
                boundary_tps.append(tp)
        
        return boundary_tps
    
    def update_after_lambda_change(self):
        """
        Recompute all Steiner points after λ parameters change.
        
        Should be called after optimization updates λ values.
        """
        for tp in self.triple_points:
            tp.steiner_point = None
            tp.compute_steiner_point()

