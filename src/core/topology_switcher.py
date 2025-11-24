"""
Topology switching operations for partition optimization.

This module implements Type 1 and Type 2 topology switches from paper Section 5:
- Type 1: Move variable points to adjacent edges when λ ≈ 0 or 1
- Type 2: Migrate triple points to adjacent triangles (via Type 1 moves)

Key algorithms:
- Triangle-local edge selection (minimizes topological disruption)
- Distance-based candidate selection (minimizes perimeter)
- Conservation of variable points and segments
"""

import numpy as np
from typing import List, Tuple, Optional, Set, Dict

try:
    from ..logging_config import get_logger
    from .tri_mesh import TriMesh
    from .contour_partition import PartitionContour, VariablePoint
    from .mesh_topology import MeshTopology
    from .steiner_handler import TriplePoint, SteinerHandler
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from logging_config import get_logger
    from core.tri_mesh import TriMesh
    from core.contour_partition import PartitionContour, VariablePoint
    from core.mesh_topology import MeshTopology
    from core.steiner_handler import TriplePoint, SteinerHandler


class TopologySwitcher:
    """
    Handles Type 1 and Type 2 topology switches during optimization.
    
    Type 1: Variable point edge switching (direct)
    Type 2: Triple point migration (via Type 1 moves)
    
    Attributes:
        mesh: The underlying TriMesh
        partition: PartitionContour with variable points
        mesh_topology: Precomputed mesh connectivity
        logger: Logger instance
    """
    
    def __init__(self, mesh: TriMesh, partition: PartitionContour, 
                 mesh_topology: MeshTopology):
        """
        Initialize topology switcher.
        
        Args:
            mesh: TriMesh object
            partition: PartitionContour with variable points
            mesh_topology: Precomputed mesh connectivity
        """
        self.mesh = mesh
        self.partition = partition
        self.mesh_topology = mesh_topology
        self.logger = get_logger(__name__)
        
        self.logger.info(f"Initialized TopologySwitcher for {len(partition.variable_points)} variable points")
    
    def apply_type1_switch(self, vp_idx: int, tol: float = 0.1) -> bool:
        """
        Apply Type 1 switch: move variable point to adjacent edge.
        
        Algorithm (triangle-local with distance minimization):
        1. Identify target vertex (which endpoint VP is approaching)
        2. Find two triangles sharing current edge
        3. In each triangle, identify free edge through target vertex
        4. Test both candidates: compute Σ distance(VP_neighbor, VP_new_position)
        5. Move VP to edge with minimum total distance
        
        Args:
            vp_idx: Index of variable point to move
            tol: New λ position away from boundary (default 0.1)
            
        Returns:
            True if switch successful, False otherwise
        """
        vp = self.partition.variable_points[vp_idx]
        
        # 1. Identify target vertex
        target_vertex = self._identify_target_vertex(vp)
        
        if target_vertex is None:
            self.logger.warning(f"VP {vp_idx}: Could not identify target vertex")
            return False
        
        # 2. Get triangle-local candidate edges
        candidates = self._get_triangle_local_candidates(vp.edge, target_vertex)
        
        if not candidates:
            # Fallback: try all adjacent edges at target vertex
            self.logger.debug(f"VP {vp_idx}: No triangle-local candidates, trying all adjacent edges")
            candidates = self._get_all_adjacent_edges_fallback(vp.edge, target_vertex)
        
        if not candidates:
            self.logger.warning(f"VP {vp_idx}: No candidate edges found for switching")
            return False
        
        # 3. Get neighboring variable points
        neighbors = self._get_neighboring_variable_points(vp_idx)
        
        if not neighbors:
            self.logger.warning(f"VP {vp_idx}: No neighboring variable points found")
            return False
        
        # 4. Test all candidates and select minimum distance
        best_edge = None
        best_lambda = None
        min_distance = float('inf')
        
        for candidate_edge in candidates:
            # Determine lambda orientation (close to target vertex)
            test_lambda = self._get_lambda_near_vertex(candidate_edge, target_vertex, tol)
            
            # Compute total segment length with this configuration
            total_dist = self._compute_total_segment_length(
                vp_idx, candidate_edge, test_lambda, neighbors
            )
            
            if total_dist < min_distance:
                min_distance = total_dist
                best_edge = candidate_edge
                best_lambda = test_lambda
        
        if best_edge is None:
            self.logger.warning(f"VP {vp_idx}: Could not find valid target edge")
            return False
        
        # Save old state before moving (for logging)
        old_edge = vp.edge
        old_lambda = vp.lambda_param
        
        # 5. Move the variable point
        self._move_variable_point(vp_idx, best_edge, best_lambda)
        
        self.logger.info(f"VP {vp_idx}: Type 1 switch successful")
        self.logger.info(f"  Old edge: {old_edge}, λ = {old_lambda:.3f}")
        self.logger.info(f"  New edge: {best_edge}, λ = {best_lambda:.3f}")
        self.logger.info(f"  Total segment length: {min_distance:.6f}")
        
        return True
    
    def _identify_target_vertex(self, vp: VariablePoint) -> Optional[int]:
        """
        Identify which vertex the variable point is approaching.
        
        Args:
            vp: VariablePoint object
            
        Returns:
            Vertex index, or None if can't determine
        """
        # If λ < 0.5, approaching edge[0] (first vertex)
        # If λ > 0.5, approaching edge[1] (second vertex)
        if vp.lambda_param < 0.5:
            return vp.edge[0]
        else:
            return vp.edge[1]
    
    def _get_triangle_local_candidates(self, current_edge: Tuple[int, int], 
                                        target_vertex: int) -> List[Tuple[int, int]]:
        """
        Get triangle-local candidate edges (from adjacent triangles).
        
        Per discussion: Only consider edges from the 2 triangles sharing current edge.
        
        Args:
            current_edge: Current edge containing variable point
            target_vertex: Vertex that VP is approaching
            
        Returns:
            List of candidate edges (up to 2)
        """
        # Get triangles sharing the current edge
        adjacent_triangles = self.mesh_topology.get_triangles_sharing_edge(current_edge)
        
        candidates = []
        occupied_edges = set(vp.edge for vp in self.partition.variable_points)
        
        for tri_idx in adjacent_triangles:
            # Get all edges of this triangle
            tri_edges = self.mesh.get_triangle_edges(tri_idx)
            
            # Find edges that:
            # 1. Include target_vertex
            # 2. Are not the current edge
            # 3. Don't have a variable point already
            for edge in tri_edges:
                normalized_edge = tuple(sorted(edge))
                
                if (target_vertex in edge and 
                    normalized_edge != tuple(sorted(current_edge)) and
                    normalized_edge not in occupied_edges):
                    candidates.append(normalized_edge)
        
        # Remove duplicates (edge might appear in both triangles)
        candidates = list(set(candidates))
        
        return candidates
    
    def _get_all_adjacent_edges_fallback(self, current_edge: Tuple[int, int],
                                         target_vertex: int) -> List[Tuple[int, int]]:
        """
        Fallback: Get all adjacent edges at target vertex.
        
        Used when triangle-local selection finds no candidates
        (e.g., at mesh boundaries).
        
        Args:
            current_edge: Current edge
            target_vertex: Target vertex
            
        Returns:
            List of candidate edges
        """
        occupied_edges = set(vp.edge for vp in self.partition.variable_points)
        
        # Get all edges at target vertex
        adjacent_edges = self.mesh_topology.get_adjacent_edges_through_vertex(
            current_edge, target_vertex
        )
        
        # Filter out occupied edges
        return [e for e in adjacent_edges if e not in occupied_edges]
    
    def _get_neighboring_variable_points(self, vp_idx: int) -> List[int]:
        """
        Find variable points connected to this VP by segments.
        
        Uses partition.triangle_segments to find connections.
        
        Args:
            vp_idx: Index of variable point
            
        Returns:
            List of neighboring variable point indices
        """
        neighbors = set()
        
        # Scan all triangle segments to find connections
        for tri_seg in self.partition.triangle_segments:
            if vp_idx in tri_seg.var_point_indices:
                # This triangle contains our VP
                # All other VPs in this triangle are neighbors
                for other_vp_idx in tri_seg.var_point_indices:
                    if other_vp_idx != vp_idx:
                        neighbors.add(other_vp_idx)
        
        return list(neighbors)
    
    def _get_lambda_near_vertex(self, edge: Tuple[int, int], 
                                 target_vertex: int, tol: float) -> float:
        """
        Determine λ value that places VP near target vertex.
        
        Args:
            edge: Target edge
            target_vertex: Vertex to place VP near
            tol: Distance from boundary (default 0.1)
            
        Returns:
            Lambda value (0.1 if target is edge[0], 0.9 if target is edge[1])
        """
        if edge[0] == target_vertex:
            # Target is first vertex → λ = tol (small value)
            return tol
        elif edge[1] == target_vertex:
            # Target is second vertex → λ = 1 - tol (large value)
            return 1.0 - tol
        else:
            # Shouldn't happen, but fallback to middle
            self.logger.warning(f"Target vertex {target_vertex} not in edge {edge}")
            return 0.5
    
    def _compute_total_segment_length(self, vp_idx: int, 
                                       test_edge: Tuple[int, int],
                                       test_lambda: float,
                                       neighbors: List[int]) -> float:
        """
        Compute Σ distance(VP_neighbor, VP_test_position).
        
        This is the selection criterion: choose edge that minimizes total length.
        
        Args:
            vp_idx: Variable point index being tested
            test_edge: Candidate edge
            test_lambda: Lambda value on candidate edge
            neighbors: List of neighboring VP indices
            
        Returns:
            Total distance to all neighbors
        """
        # Compute test position
        v1, v2 = test_edge
        p1 = self.mesh.vertices[v1]
        p2 = self.mesh.vertices[v2]
        test_pos = test_lambda * p1 + (1 - test_lambda) * p2
        
        # Sum distances to all neighbors
        total_dist = 0.0
        for neighbor_idx in neighbors:
            neighbor_pos = self.partition.evaluate_variable_point(neighbor_idx)
            dist = np.linalg.norm(test_pos - neighbor_pos)
            total_dist += dist
        
        return total_dist
    
    def _move_variable_point(self, vp_idx: int, new_edge: Tuple[int, int], 
                             new_lambda: float) -> None:
        """
        Actually move the variable point to new edge.
        
        Updates:
        - vp.edge
        - vp.lambda_param
        - partition.edge_to_varpoint dict
        
        Args:
            vp_idx: Variable point index
            new_edge: New edge tuple
            new_lambda: New lambda value
        """
        vp = self.partition.variable_points[vp_idx]
        old_edge = vp.edge
        
        # Update edge_to_varpoint dict (remove old, add new)
        if old_edge in self.partition.edge_to_varpoint:
            del self.partition.edge_to_varpoint[old_edge]
        
        self.partition.edge_to_varpoint[new_edge] = vp_idx
        
        # Update variable point
        vp.edge = new_edge
        vp.lambda_param = new_lambda
    
    def select_variable_point_for_type2(self, triple_point: TriplePoint) -> Optional[int]:
        """
        Select which variable point to move for Type 2 switch.
        
        Per documentation (lines 66-68):
        1. One VP already on shared edge (boundary between triangles) → stays in place
        2. From remaining 2 VPs: select the one closest to a vertex
        3. Minimizes "jump" distance
        
        Args:
            triple_point: TriplePoint with Steiner point near mesh triangle boundary
            
        Returns:
            Index of variable point to move, or None if can't determine
        """
        # 1. Find which mesh triangle edge the Steiner point is closest to
        closest_edge, min_dist = self._find_closest_edge_to_steiner(triple_point)
        
        if closest_edge is None:
            self.logger.warning(f"Could not find closest edge for triple point at triangle {triple_point.triangle_idx}")
            # Fallback: pick VP closest to any vertex
            return self._select_vp_closest_to_vertex(triple_point.var_point_indices)
        
        # 2. Find which VP is already on that edge (if any)
        vp_on_shared_edge = None
        remaining_vps = []
        
        for vp_idx in triple_point.var_point_indices:
            vp = self.partition.variable_points[vp_idx]
            if vp.edge == closest_edge:
                vp_on_shared_edge = vp_idx
            else:
                remaining_vps.append(vp_idx)
        
        # 3. If one VP is on the shared edge, choose from remaining 2
        if vp_on_shared_edge is not None and len(remaining_vps) == 2:
            # Select VP closest to a vertex (smallest min(λ, 1-λ))
            return self._select_vp_closest_to_vertex(remaining_vps)
        
        # 4. Fallback: no clear shared edge, pick VP closest to any vertex
        else:
            return self._select_vp_closest_to_vertex(triple_point.var_point_indices)
    
    def _select_vp_closest_to_vertex(self, vp_indices: List[int]) -> Optional[int]:
        """
        Select variable point closest to a vertex (min distance from λ=0 or λ=1).
        
        Args:
            vp_indices: List of variable point indices to choose from
            
        Returns:
            Index of VP closest to a vertex
        """
        min_dist_to_vertex = float('inf')
        best_vp = None
        
        for vp_idx in vp_indices:
            vp = self.partition.variable_points[vp_idx]
            dist = min(vp.lambda_param, 1.0 - vp.lambda_param)
            
            if dist < min_dist_to_vertex:
                min_dist_to_vertex = dist
                best_vp = vp_idx
        
        return best_vp
    
    def _find_closest_edge_to_steiner(self, triple_point: TriplePoint) -> Tuple[Optional[Tuple[int, int]], float]:
        """
        Find which mesh triangle edge the Steiner point is closest to.
        
        Args:
            triple_point: TriplePoint object
            
        Returns:
            (closest_edge, distance) or (None, inf) if not found
        """
        if triple_point.steiner_point is None:
            triple_point.compute_steiner_point()
        
        tri_idx = triple_point.triangle_idx
        
        # Find the TriangleSegment for this triple point
        tri_seg = None
        for ts in self.partition.triangle_segments:
            if ts.triangle_idx == tri_idx and ts.is_triple_point():
                tri_seg = ts
                break
        
        if tri_seg is None:
            return (None, float('inf'))
        
        # Check all 3 edges of the mesh triangle
        vertices = tri_seg.vertex_indices
        edges = [
            (vertices[0], vertices[1]),
            (vertices[1], vertices[2]),
            (vertices[2], vertices[0])
        ]
        
        min_dist = float('inf')
        closest_edge = None
        
        for edge in edges:
            v1_pos = self.mesh.vertices[edge[0]]
            v2_pos = self.mesh.vertices[edge[1]]
            
            # Distance from Steiner point to edge
            dist = triple_point._point_to_segment_distance(
                triple_point.steiner_point, v1_pos, v2_pos
            )
            
            if dist < min_dist:
                min_dist = dist
                closest_edge = tuple(sorted(edge))  # Normalize
        
        return closest_edge, min_dist

