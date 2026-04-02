"""
Steiner tree handling for triple points in partition contours.

This module implements the triple point treatment from Section 5 of the paper:
"The empty spaces around triple points". 

When three different partition cells meet at a mesh triangle, small void spaces are created.
These are filled with Steiner trees that connect three variable points on the triangle's
edges, meeting at an optimal Steiner point that minimizes total edge length (120° angles).
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
    
    Decoupled from PartitionContour: methods receive VP positions and mesh vertices
    as arguments rather than holding a partition back-reference. This enables
    snapshotting without dragging the entire partition.
    """
    
    def __init__(self, triangle_idx: int, var_point_indices: List[int],
                 cell_indices: List[int], vertex_indices: Tuple[int, int, int],
                 vertex_labels: Tuple[int, int, int]):
        """
        Initialize triple point with explicit data (no partition reference).
        
        Args:
            triangle_idx: Index of triangle in mesh
            var_point_indices: List of 3 variable point indices
            cell_indices: List of 3 cell indices meeting at this point
            vertex_indices: Tuple of 3 mesh vertex indices of the triangle
            vertex_labels: Tuple of 3 cell labels for the vertices
        """
        self.triangle_idx = triangle_idx
        self.var_point_indices = list(var_point_indices)
        self.logger = get_logger(__name__)
        
        self.cell_indices = list(set(cell_indices))
        if len(self.cell_indices) != 3:
            self.logger.warning(f"Triple point at triangle {triangle_idx} has "
                              f"{len(self.cell_indices)} cells, expected 3")
        
        self.vertex_indices = vertex_indices
        self._vertex_labels = vertex_labels
        
        self.cell_to_varpoint_pair: Dict[int, Tuple[int, int]] = {}
        self.cell_to_mesh_vertex: Dict[int, int] = {}
        
        self.steiner_point: Optional[np.ndarray] = None
        self.boundary_points: Optional[List[np.ndarray]] = None
    
    @classmethod
    def from_partition(cls, triangle_idx: int, var_point_indices: List[int],
                       partition: PartitionContour) -> 'TriplePoint':
        """Factory: create a TriplePoint from a PartitionContour (backward-compat helper)."""
        cell_indices_set: set = set()
        for vp_idx in var_point_indices:
            vp = partition.variable_points[vp_idx]
            cell_indices_set.update(vp.belongs_to_cells)
        
        ts = partition.get_triangle_segment(triangle_idx)
        if ts is not None:
            vertex_indices = ts.vertex_indices
        else:
            face = partition.mesh.faces[triangle_idx]
            vertex_indices = (int(face[0]), int(face[1]), int(face[2]))
        
        vertex_labels = tuple(int(partition.vertex_labels[v]) for v in vertex_indices)
        
        tp = cls(triangle_idx, var_point_indices, list(cell_indices_set),
                 vertex_indices, vertex_labels)
        tp._compute_cell_to_varpoint_mapping(partition)
        return tp
    
    def _compute_cell_to_varpoint_mapping(self, partition: PartitionContour):
        """
        Compute mapping from each cell to the 2 VPs that define its boundary.
        
        Per Figure 7: For a triple point with triangle ABC:
        - Cell at vertex A gets boundary from edges emanating from A
        - Each cell's perimeter is defined by the 2 VPs on edges adjacent to its vertex
        """
        vertices = self.vertex_indices
        labels = self._vertex_labels
        
        for i in range(3):
            vertex = vertices[i]
            cell_label = int(labels[i])
            
            self.cell_to_mesh_vertex[cell_label] = int(vertex)
            
            edge2 = tuple(sorted([vertices[i], vertices[(i + 1) % 3]]))
            edge3 = tuple(sorted([vertices[i], vertices[(i + 2) % 3]]))
            
            vp_indices_for_cell = []
            for vp_idx in self.var_point_indices:
                vp = partition.variable_points[vp_idx]
                if vp.edge == edge2 or vp.edge == edge3:
                    vp_indices_for_cell.append(vp_idx)
            
            if len(vp_indices_for_cell) == 2:
                self.cell_to_varpoint_pair[cell_label] = tuple(vp_indices_for_cell)
            else:
                self.logger.warning(f"Expected 2 VPs for cell {cell_label} at triple point "
                                  f"{self.triangle_idx}, found {len(vp_indices_for_cell)}")
    
    def compute_steiner_point(self, vp_positions: Optional[List[np.ndarray]] = None,
                              partition: Optional[PartitionContour] = None) -> np.ndarray:
        """
        Compute optimal Steiner point that minimizes total edge length.
        
        Args:
            vp_positions: List of 3 VP position arrays. If None, evaluated from partition.
            partition: PartitionContour (used only if vp_positions is None).
        """
        if vp_positions is not None:
            self.boundary_points = list(vp_positions)
        elif partition is not None:
            self.boundary_points = [
                partition.evaluate_variable_point(vp_idx)
                for vp_idx in self.var_point_indices
            ]
        elif self.boundary_points is None:
            raise ValueError("Must provide vp_positions or partition")
        
        p1, p2, p3 = self.boundary_points
        initial_guess = (p1 + p2 + p3) / 3.0
        
        def objective(S):
            return (np.linalg.norm(S - p1) + 
                   np.linalg.norm(S - p2) + 
                   np.linalg.norm(S - p3))
        
        def gradient(S):
            grad = np.zeros_like(S)
            for p in [p1, p2, p3]:
                dist = np.linalg.norm(S - p)
                if dist > 1e-12:
                    grad += (S - p) / dist
            return grad
        
        result = minimize(objective, initial_guess, jac=gradient, method='BFGS',
                         options={'gtol': 1e-8})
        
        self.steiner_point = result.x
        return self.steiner_point
    
    def compute_void_angles(self, vp_positions: Optional[List[np.ndarray]] = None) -> Tuple[float, float, float]:
        """
        Compute angles of the void triangle ABC. Used for Type 2 detection.
        
        Returns angles at each of the 3 VPs (in the same order as var_point_indices).
        An angle >= 120° indicates Steiner collapse onto that VP.
        
        Args:
            vp_positions: List of 3 VP position arrays. Uses self.boundary_points if None.
            
        Returns:
            Tuple of 3 angles in degrees (at VP[0], VP[1], VP[2])
        """
        if vp_positions is not None:
            points = vp_positions
        elif self.boundary_points is not None:
            points = self.boundary_points
        else:
            raise ValueError("Must provide vp_positions or compute steiner_point first")
        
        p0, p1, p2 = points[0], points[1], points[2]
        
        def angle_at(center, a, b):
            v1 = a - center
            v2 = b - center
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-15)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            return np.degrees(np.arccos(cos_angle))
        
        return (angle_at(p0, p1, p2), angle_at(p1, p0, p2), angle_at(p2, p0, p1))
    
    def _triangle_area_3d(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Compute area of triangle with vertices p1, p2, p3."""
        v1 = p2 - p1
        v2 = p3 - p1
        if len(p1) == 2:
            return 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
        else:
            return 0.5 * np.linalg.norm(np.cross(v1, v2))
    
    def get_perimeter_contribution(self, vp_positions: Optional[List[np.ndarray]] = None,
                                   partition: Optional[PartitionContour] = None) -> Dict[int, float]:
        """
        Compute NET perimeter contribution to each adjacent cell.
        Per paper Figure 7: Each cell gets (2 Steiner edges - 1 original edge).
        """
        if self.steiner_point is None:
            self.compute_steiner_point(vp_positions=vp_positions, partition=partition)
        
        contributions = {cell_idx: 0.0 for cell_idx in self.cell_indices}
        
        for cell_idx, (vp_idx1, vp_idx2) in self.cell_to_varpoint_pair.items():
            idx1 = self.var_point_indices.index(vp_idx1)
            idx2 = self.var_point_indices.index(vp_idx2)
            vp_pos1 = self.boundary_points[idx1]
            vp_pos2 = self.boundary_points[idx2]
            
            steiner_edge1 = np.linalg.norm(vp_pos1 - self.steiner_point)
            steiner_edge2 = np.linalg.norm(vp_pos2 - self.steiner_point)
            original_edge = np.linalg.norm(vp_pos1 - vp_pos2)
            
            contributions[cell_idx] = steiner_edge1 + steiner_edge2 - original_edge
        
        return contributions
    
    def get_area_contribution(self, mesh: TriMesh,
                              vp_positions: Optional[List[np.ndarray]] = None) -> Dict[int, float]:
        """
        Compute area contribution to each adjacent cell (void interior + corner).
        """
        if self.steiner_point is None:
            self.compute_steiner_point(vp_positions=vp_positions)
        
        contributions = {cell_idx: 0.0 for cell_idx in self.cell_indices}
        var_points_pos = self.boundary_points
        
        for cell_idx in self.cell_indices:
            if cell_idx not in self.cell_to_varpoint_pair:
                triangle_area = mesh.triangle_areas[self.triangle_idx]
                contributions[cell_idx] = triangle_area / 3.0
                continue
            
            vp_idx1, vp_idx2 = self.cell_to_varpoint_pair[cell_idx]
            idx1 = self.var_point_indices.index(vp_idx1)
            idx2 = self.var_point_indices.index(vp_idx2)
            pos1 = var_points_pos[idx1]
            pos2 = var_points_pos[idx2]
            
            void_area = self._triangle_area_3d(pos1, pos2, self.steiner_point)
            
            corner_area = 0.0
            if cell_idx in self.cell_to_mesh_vertex:
                mesh_vertex_idx = self.cell_to_mesh_vertex[cell_idx]
                vertex_pos = mesh.vertices[mesh_vertex_idx]
                corner_area = self._triangle_area_3d(vertex_pos, pos1, pos2)
            
            contributions[cell_idx] = void_area + corner_area
        
        return contributions
    
    def compute_area_gradients_finite_difference(self, mesh: TriMesh,
                                                 partition: PartitionContour,
                                                 eps: float = 1e-7) -> Dict[int, np.ndarray]:
        """Compute gradients of Steiner tree area contributions w.r.t. λ parameters."""
        n_vars = len(partition.variable_points)
        gradients = {cell_idx: np.zeros(n_vars) for cell_idx in self.cell_indices}
        
        base_vp_positions = [partition.evaluate_variable_point(vi) for vi in self.var_point_indices]
        self.steiner_point = None
        base_contrib = self.get_area_contribution(mesh, vp_positions=base_vp_positions)
        
        for vp_idx in self.var_point_indices:
            old_lambda = partition.variable_points[vp_idx].lambda_param
            partition.variable_points[vp_idx].lambda_param = old_lambda + eps
            
            vp_positions = [partition.evaluate_variable_point(vi) for vi in self.var_point_indices]
            self.steiner_point = None
            perturbed_contrib = self.get_area_contribution(mesh, vp_positions=vp_positions)
            
            for cell_idx in self.cell_indices:
                gradients[cell_idx][vp_idx] = (perturbed_contrib[cell_idx] - 
                                               base_contrib[cell_idx]) / eps
            
            partition.variable_points[vp_idx].lambda_param = old_lambda
        
        self.steiner_point = None
        vp_positions = [partition.evaluate_variable_point(vi) for vi in self.var_point_indices]
        self.compute_steiner_point(vp_positions=vp_positions)
        
        return gradients
    
    def compute_gradients_finite_difference(self, partition: PartitionContour,
                                            eps: float = 1e-6) -> np.ndarray:
        """Compute gradients of Steiner tree perimeter w.r.t. λ parameters."""
        n_vars = len(partition.variable_points)
        gradient = np.zeros(n_vars)
        
        base_vp_positions = [partition.evaluate_variable_point(vi) for vi in self.var_point_indices]
        self.steiner_point = None
        base_contrib = self.get_perimeter_contribution(vp_positions=base_vp_positions)
        base_total = sum(base_contrib.values())
        
        for vp_idx in self.var_point_indices:
            old_lambda = partition.variable_points[vp_idx].lambda_param
            partition.variable_points[vp_idx].lambda_param = old_lambda + eps
            
            vp_positions = [partition.evaluate_variable_point(vi) for vi in self.var_point_indices]
            self.steiner_point = None
            perturbed_contrib = self.get_perimeter_contribution(vp_positions=vp_positions)
            perturbed_total = sum(perturbed_contrib.values())
            
            gradient[vp_idx] = (perturbed_total - base_total) / eps
            partition.variable_points[vp_idx].lambda_param = old_lambda
        
        self.steiner_point = None
        vp_positions = [partition.evaluate_variable_point(vi) for vi in self.var_point_indices]
        self.compute_steiner_point(vp_positions=vp_positions)
        
        return gradient
    
    def is_on_triangle_boundary(self, mesh: TriMesh, tol: float = 1e-3) -> bool:
        """Check if Steiner point is near the triangle boundary."""
        if self.steiner_point is None:
            return False
        
        face = mesh.faces[self.triangle_idx]
        v1, v2, v3 = [int(i) for i in face]
        p1, p2, p3 = mesh.vertices[v1], mesh.vertices[v2], mesh.vertices[v3]
        
        edges = [(p1, p2), (p2, p3), (p3, p1)]
        for edge_start, edge_end in edges:
            dist = self._point_to_segment_distance(self.steiner_point, edge_start, edge_end)
            if dist < tol:
                return True
        return False
    
    @staticmethod
    def _point_to_segment_distance(point: np.ndarray, seg_start: np.ndarray, 
                                   seg_end: np.ndarray) -> float:
        """Compute distance from point to line segment."""
        v = seg_end - seg_start
        w = point - seg_start
        c1 = np.dot(w, v)
        if c1 <= 0:
            return np.linalg.norm(point - seg_start)
        c2 = np.dot(v, v)
        if c1 >= c2:
            return np.linalg.norm(point - seg_end)
        b = c1 / c2
        proj = seg_start + b * v
        return np.linalg.norm(point - proj)


class SteinerHandler:
    """
    Manages all triple points in the partition and their Steiner trees.
    """
    
    def __init__(self, mesh: TriMesh, partition: PartitionContour):
        self.mesh = mesh
        self.partition = partition
        self.logger = get_logger(__name__)
        
        self.triple_points: List[TriplePoint] = []
        self._detect_triple_points()
    
    def _detect_triple_points(self):
        """Detect all triple points using VP-based detection."""
        triple_point_data = self.partition.identify_triple_points_from_current_vps()
        
        for tri_idx, var_point_indices in triple_point_data:
            tp = TriplePoint.from_partition(tri_idx, var_point_indices, self.partition)
            vp_positions = [self.partition.evaluate_variable_point(vi) for vi in var_point_indices]
            tp.compute_steiner_point(vp_positions=vp_positions)
            self.triple_points.append(tp)
        
        self.logger.info(f"Detected and initialized {len(self.triple_points)} triple points")
    
    def get_total_perimeter_contribution(self) -> float:
        """Compute total perimeter contribution from all Steiner trees."""
        total = 0.0
        for tp in self.triple_points:
            vp_positions = [self.partition.evaluate_variable_point(vi) for vi in tp.var_point_indices]
            tp.steiner_point = None
            contrib = tp.get_perimeter_contribution(vp_positions=vp_positions)
            total += sum(contrib.values())
        return total
    
    def get_total_area_contribution(self) -> Dict[int, float]:
        """Compute total area contribution from all Steiner trees to each cell."""
        contributions = {i: 0.0 for i in range(self.partition.n_cells)}
        for tp in self.triple_points:
            vp_positions = [self.partition.evaluate_variable_point(vi) for vi in tp.var_point_indices]
            tp.steiner_point = None
            area_contrib = tp.get_area_contribution(self.mesh, vp_positions=vp_positions)
            for cell_idx, area in area_contrib.items():
                contributions[cell_idx] += area
        return contributions
    
    def compute_total_gradient_finite_difference(self, eps: float = 1e-6) -> np.ndarray:
        """Compute total gradient contribution from all Steiner trees."""
        gradient = np.zeros(len(self.partition.variable_points))
        for tp in self.triple_points:
            gradient += tp.compute_gradients_finite_difference(self.partition, eps)
        return gradient
    
    def compute_area_gradients_finite_difference(self, mesh: TriMesh, 
                                                 eps: float = 1e-7) -> Dict[int, np.ndarray]:
        """Compute total area gradient contributions from all Steiner trees."""
        n_vars = len(self.partition.variable_points)
        gradients = {i: np.zeros(n_vars) for i in range(self.partition.n_cells)}
        for tp in self.triple_points:
            tp_gradients = tp.compute_area_gradients_finite_difference(mesh, self.partition, eps)
            for cell_idx, grad in tp_gradients.items():
                gradients[cell_idx] += grad
        return gradients
    
    def get_boundary_triple_points(self, tol: float = 1e-3) -> List[TriplePoint]:
        """Find triple points with Steiner point near triangle boundary."""
        return [tp for tp in self.triple_points if tp.is_on_triangle_boundary(self.mesh, tol)]
    
