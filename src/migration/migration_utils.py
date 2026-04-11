"""
Shared utility functions for Type 1 migration calculations.

This module contains low-level helper functions used by both:
- Component analysis (type1_component_analyzer.py)
- Migration execution (migration_executor.py)
"""

import numpy as np
from typing import Tuple, Optional

from ..logging_config import get_logger
from ..mesh.tri_mesh import TriMesh
from ..partition.contour_partition import PartitionContour, VariablePoint


def identify_target_vertex(vp: VariablePoint) -> Optional[int]:
    """
    Identify which vertex the variable point is approaching.
    
    CRITICAL: The λ convention in VariablePoint is:
        position = λ * edge[0] + (1-λ) * edge[1]
    
    This means:
        - λ = 1 → position at edge[0] (smaller vertex index due to normalization)
        - λ = 0 → position at edge[1] (larger vertex index)
    
    So:
        - If λ > 0.5, VP is closer to edge[0], approaching edge[0]
        - If λ < 0.5, VP is closer to edge[1], approaching edge[1]
    
    Args:
        vp: VariablePoint object
        
    Returns:
        Vertex index, or None if can't determine
    """
    # λ > 0.5 means closer to edge[0] (since λ=1 is AT edge[0])
    # λ < 0.5 means closer to edge[1] (since λ=0 is AT edge[1])
    if vp.lambda_param > 0.5:
        return vp.edge[0]
    else:
        return vp.edge[1]


def compute_boundary_distance(vp: VariablePoint) -> float:
    """
    Compute distance from VP to its target vertex: min(λ, 1-λ)
    
    Args:
        vp: VariablePoint object
        
    Returns:
        Distance in [0, 0.5], where smaller = closer to target vertex
    """
    return min(vp.lambda_param, 1.0 - vp.lambda_param)


def get_two_neighbors(partition: PartitionContour, vp_idx: int) -> Tuple[int, int]:
    """
    Get the two neighbors of a VP via boundary_segments.
    
    CRITICAL: Every VP has exactly 2 neighbors (one on each side of the boundary segment).
    
    Args:
        partition: PartitionContour containing boundary segments
        vp_idx: Variable point index
        
    Returns:
        (left_neighbor_idx, right_neighbor_idx)
        Note: These might be boundary or non-boundary VPs!
        
    Raises:
        ValueError: If VP doesn't have exactly 2 neighbors
    """
    neighbors = []
    for segment in partition.boundary_segments:
        if segment.vp_idx_1 == vp_idx:
            neighbors.append(segment.vp_idx_2)
        elif segment.vp_idx_2 == vp_idx:
            neighbors.append(segment.vp_idx_1)
    
    if len(neighbors) != 2:
        raise ValueError(f"VP {vp_idx} must have exactly 2 neighbors, found {len(neighbors)}")
    
    return (neighbors[0], neighbors[1])
