"""
Perimeter computation and gradient calculation for partition cells.

This module implements the perimeter computation logic from Section 5 of the paper:
"Computation of the perimeters of the cells". 

TERMINOLOGY (following paper Section 5):
- "cell": Partition region (what we optimize for equal areas)
- "triangle": Mesh triangle element (computational discretization)
- "edge": Mesh triangle edge (computational discretization)

For a segment between two variable points:
  x_i = λ_i * v1 + (1-λ_i) * v2  (on mesh triangle edge (v1,v2))
  x_j = λ_j * v3 + (1-λ_j) * v4  (on mesh triangle edge (v3,v4))

Length: ℓ = ||x_i - x_j||

Gradients (paper, line 349-353):
  ∂ℓ/∂λ_i = (x_i - x_j) · (v1 - v2) / ℓ
  ∂ℓ/∂λ_j = (x_j - x_i) · (v3 - v4) / ℓ
"""

import numpy as np
from typing import Tuple

from ..logging_config import get_logger
from .tri_mesh import TriMesh
from .contour_partition import PartitionContour


class PerimeterCalculator:
    """
    Computes partition cell perimeters and gradients for optimization.
    
    The perimeter of a partition cell is the sum of lengths of all segments forming
    its contour. Each segment connects two variable points on mesh triangle edges.
    
    Attributes:
        mesh: The underlying TriMesh (for vertex coordinates)
        partition: PartitionContour with variable points
    """
    
    def __init__(self, mesh: TriMesh, partition: PartitionContour):
        """
        Initialize perimeter calculator.
        
        Args:
            mesh: TriMesh object
            partition: PartitionContour with variable points
        """
        self.mesh = mesh
        self.partition = partition
        self.logger = get_logger(__name__)
        
        self.logger.info(f"Initialized PerimeterCalculator for {partition.n_cells} cells")
    
    def compute_total_perimeter(self, lambda_vec: np.ndarray) -> float:
        """
        Compute total perimeter of all cells.
        
        Phase 2: Refactored to use cell indices instead of deprecated CellContour objects.
        
        Note: Caller must call partition.set_variable_vector(lambda_vec) before this method.
        
        Args:
            lambda_vec: Current variable point parameters (for consistency with gradient signature)
            
        Returns:
            Sum of all cell perimeters
        """
        total = 0.0
        
        for cell_idx in range(self.partition.n_cells):
            total += self.compute_cell_perimeter(cell_idx, lambda_vec)
        
        return total
    
    def compute_cell_perimeter(self, cell_idx: int, lambda_vec: np.ndarray) -> float:
        """
        Compute perimeter of one cell.
        
        Phase 4: Uses explicit boundary_segments for accurate perimeter calculation
        after topology switches. Falls back to triangle-based extraction if not available.
        
        Note: Caller must call partition.set_variable_vector(lambda_vec) before this method.
        
        Args:
            cell_idx: Index of the cell
            lambda_vec: Current variable point parameters (for consistency with gradient signature)
            
        Returns:
            Total perimeter length
        """
        perimeter = 0.0
        
        # Phase 4: Use explicit boundary_segments if available
        if self.partition.boundary_segments:
            for seg in self.partition.boundary_segments:
                # Check if this segment belongs to this cell
                if cell_idx in seg.cell_pair:
                    length = self.compute_segment_length(seg.vp_idx_1, seg.vp_idx_2)
                    perimeter += length
        else:
            # Fallback to triangle-based extraction
            segments = self.partition.get_cell_segments_from_triangles(cell_idx)
            
            for var_idx_i, var_idx_j in segments:
                length = self.compute_segment_length(var_idx_i, var_idx_j)
                perimeter += length
        
        return perimeter
    
    def compute_segment_length(self, var_idx_i: int, var_idx_j: int) -> float:
        """
        Compute length of segment between two variable points.
        
        Args:
            var_idx_i: Index of first variable point
            var_idx_j: Index of second variable point
            
        Returns:
            Euclidean distance between the two points
        """
        pos_i = self.partition.evaluate_variable_point(var_idx_i)
        pos_j = self.partition.evaluate_variable_point(var_idx_j)
        
        return float(np.linalg.norm(pos_i - pos_j))
    
    def compute_segment_gradient(self, var_idx_i: int, var_idx_j: int) -> Tuple[float, float]:
        """
        Compute gradient of segment length w.r.t. its two λ parameters.
        
        From paper (Section 5, equation at line 349-351):
        For segment between x_i and x_j:
          ∂ℓ/∂λ_i = (x_i - x_j) · (v1 - v2) / ℓ
          ∂ℓ/∂λ_j = (x_j - x_i) · (v3 - v4) / ℓ
        
        where x_i = λ_i*v1 + (1-λ_i)*v2, x_j = λ_j*v3 + (1-λ_j)*v4
        
        Args:
            var_idx_i: Index of first variable point
            var_idx_j: Index of second variable point
            
        Returns:
            (∂ℓ/∂λ_i, ∂ℓ/∂λ_j)
        """
        vp_i = self.partition.variable_points[var_idx_i]
        vp_j = self.partition.variable_points[var_idx_j]
        
        # Get positions
        pos_i = vp_i.evaluate(self.mesh.vertices)
        pos_j = vp_j.evaluate(self.mesh.vertices)
        
        # Get edge endpoints
        v1_i, v2_i = vp_i.edge
        v3_j, v4_j = vp_j.edge
        
        p1_i = self.mesh.vertices[v1_i]
        p2_i = self.mesh.vertices[v2_i]
        p3_j = self.mesh.vertices[v3_j]
        p4_j = self.mesh.vertices[v4_j]
        
        # Compute length
        diff = pos_i - pos_j
        length = np.linalg.norm(diff)
        
        if length < 1e-12:
            # Avoid division by zero
            return 0.0, 0.0
        
        # Gradient w.r.t. λ_i
        # ∂x_i/∂λ_i = v1 - v2 (by definition of x_i = λ_i*v1 + (1-λ_i)*v2)
        dx_i_dlambda_i = p1_i - p2_i
        grad_i = float(np.dot(diff, dx_i_dlambda_i) / length)
        
        # Gradient w.r.t. λ_j
        # ∂x_j/∂λ_j = v3 - v4
        dx_j_dlambda_j = p3_j - p4_j
        # Note: ∂ℓ/∂x_j = -(x_i - x_j)/ℓ = (x_j - x_i)/ℓ
        grad_j = float(np.dot(-diff, dx_j_dlambda_j) / length)
        
        return grad_i, grad_j
    
    def compute_perimeter_gradient(self, cell_idx: int, 
                                   lambda_vec: np.ndarray) -> np.ndarray:
        """
        Compute gradient of cell perimeter w.r.t. all λ parameters.
        
        Phase 4: Uses explicit boundary_segments for accurate gradient calculation
        after topology switches. Falls back to triangle-based extraction if not available.
        
        Note: Caller must call partition.set_variable_vector(lambda_vec) before this method.
        
        Args:
            cell_idx: Index of the cell
            lambda_vec: Current variable point parameters (for consistency with signature)
            
        Returns:
            Gradient array of shape (n_variable_points,)
        """
        gradient = np.zeros(len(lambda_vec))
        
        # Phase 4: Use explicit boundary_segments if available
        if self.partition.boundary_segments:
            for seg in self.partition.boundary_segments:
                if cell_idx in seg.cell_pair:
                    grad_i, grad_j = self.compute_segment_gradient(seg.vp_idx_1, seg.vp_idx_2)
                    gradient[seg.vp_idx_1] += grad_i
                    gradient[seg.vp_idx_2] += grad_j
        else:
            # Fallback to triangle-based extraction
            segments = self.partition.get_cell_segments_from_triangles(cell_idx)
            
            for var_idx_i, var_idx_j in segments:
                grad_i, grad_j = self.compute_segment_gradient(var_idx_i, var_idx_j)
                gradient[var_idx_i] += grad_i
                gradient[var_idx_j] += grad_j
        
        return gradient
    
    def compute_total_perimeter_gradient(self, lambda_vec: np.ndarray) -> np.ndarray:
        """
        Compute gradient of total perimeter w.r.t. all λ parameters.
        
        This is the objective function gradient for perimeter minimization.
        
        Phase 2: Refactored to use cell indices instead of deprecated CellContour objects.
        
        Note: Caller must call partition.set_variable_vector(lambda_vec) before this method.
        
        Args:
            lambda_vec: Current variable point parameters (for consistency with signature)
            
        Returns:
            Gradient array of shape (n_variable_points,)
        """
        gradient = np.zeros(len(lambda_vec))
        
        for cell_idx in range(self.partition.n_cells):
            gradient += self.compute_perimeter_gradient(cell_idx, lambda_vec)
        
        return gradient
    

