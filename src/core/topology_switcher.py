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
    from .contour_partition import PartitionContour, VariablePoint, BoundarySegment, SegmentCrossingInfo
    from .mesh_topology import MeshTopology
    from .steiner_handler import TriplePoint, SteinerHandler
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from logging_config import get_logger
    from core.tri_mesh import TriMesh
    from core.contour_partition import PartitionContour, VariablePoint, BoundarySegment, SegmentCrossingInfo
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
        6. Place VP at λ = 0.5 on new edge (midpoint for neutral initialization)
        
        Args:
            vp_idx: Index of variable point to move
            tol: Unused (kept for API compatibility, new λ is always 0.5)
            
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
        
        # Phase 4: Update segment classifications after the move
        # Note: Caller should call partition.rebuild_triangle_segments_from_current_vps()
        # which will rebuild boundary_segments, then call update_segment_classifications_after_switch()
        
        return True
    
    def apply_type2_switch(self, triple_point: TriplePoint, tol: float = 0.1) -> bool:
        """
        Apply Type 2 switch: migrate triple point to adjacent triangle.
        
        Algorithm:
        1. Identify anchor VP (on shared edge closest to Steiner point)
        2. Select VP to move (closest to shared vertex)
        3. Find target triangle (shares the anchor edge)
        4. Find free edge in target triangle (the one without a VP)
        5. Move VP to free edge with λ=0.5
        6. Caller must rebuild triangle_segments and re-initialize SteinerHandler
        
        Args:
            triple_point: TriplePoint with Steiner near boundary
            tol: Distance tolerance for boundary detection
            
        Returns:
            True if switch successful, False otherwise
        """
        self.logger.info(f"=== Applying Type 2 switch for triple point at triangle {triple_point.triangle_idx} ===")
        
        # Step 1: Find shared edge (where Steiner point is closest)
        shared_edge, dist_to_edge = self._find_closest_edge_to_steiner(triple_point)
        
        if shared_edge is None:
            self.logger.warning(f"Could not find shared edge for triple point at triangle {triple_point.triangle_idx}")
            return False
        
        self.logger.info(f"Shared edge (closest to Steiner): {shared_edge}, distance = {dist_to_edge:.6e}")
        
        # Step 2: Identify anchor VP (on shared edge) and remaining VPs
        anchor_vp_idx = None
        remaining_vp_indices = []
        
        for vp_idx in triple_point.var_point_indices:
            vp = self.partition.variable_points[vp_idx]
            if vp.edge == shared_edge:
                anchor_vp_idx = vp_idx
                self.logger.info(f"  Anchor VP {vp_idx}: stays on shared edge {shared_edge}")
            else:
                remaining_vp_indices.append(vp_idx)
        
        if anchor_vp_idx is None:
            self.logger.warning(f"No VP found on shared edge {shared_edge}")
            return False
        
        if len(remaining_vp_indices) != 2:
            self.logger.warning(f"Expected 2 remaining VPs, found {len(remaining_vp_indices)}")
            return False
        
        # Step 3: Select which VP to move (closest to shared vertex)
        moving_vp_idx = self._select_vp_closest_to_shared_edge(
            remaining_vp_indices, shared_edge
        )
        
        if moving_vp_idx is None:
            self.logger.warning(f"Could not select VP to move")
            return False
        
        moving_vp = self.partition.variable_points[moving_vp_idx]
        staying_vp_idx = [idx for idx in remaining_vp_indices if idx != moving_vp_idx][0]
        
        self.logger.info(f"  Moving VP {moving_vp_idx}: edge {moving_vp.edge}, λ = {moving_vp.lambda_param:.6f}")
        self.logger.info(f"  Staying VP {staying_vp_idx}: will remain in source triangle")
        
        # Save old state for logging
        old_edge = moving_vp.edge
        old_lambda = moving_vp.lambda_param
        old_triangle = triple_point.triangle_idx
        
        # Step 4: Find target triangle (shares the anchor edge)
        adjacent_triangles = self.mesh_topology.get_triangles_sharing_edge(shared_edge)
        target_triangle = None
        
        for tri_idx in adjacent_triangles:
            if tri_idx != triple_point.triangle_idx:
                target_triangle = tri_idx
                break
        
        if target_triangle is None:
            self.logger.warning(f"Could not find target triangle sharing edge {shared_edge}")
            return False
        
        self.logger.info(f"Target triangle: {target_triangle}")
        
        # Step 5: Find free edge in target triangle
        target_tri_edges = self.mesh.get_triangle_edges(target_triangle)
        
        # Normalize edges for comparison (order-independent)
        def normalize_edge(edge):
            return tuple(sorted(edge))
        
        occupied_edges_normalized = set(normalize_edge(vp.edge) for vp in self.partition.variable_points)
        
        free_edges = [edge for edge in target_tri_edges 
                     if normalize_edge(edge) not in occupied_edges_normalized]
        
        if len(free_edges) != 1:
            self.logger.warning(f"Expected 1 free edge in target triangle, found {len(free_edges)}")
            self.logger.warning(f"  Target triangle edges: {target_tri_edges}")
            self.logger.warning(f"  Occupied edges: {[e for e in target_tri_edges if normalize_edge(e) in occupied_edges_normalized]}")
            return False
        
        target_edge = free_edges[0]
        self.logger.info(f"Free edge in target triangle: {target_edge}")
        
        # Step 6: Move VP to free edge with λ=0.5
        new_lambda = 0.5  # Center of edge
        self._move_variable_point(moving_vp_idx, target_edge, new_lambda)
        
        self.logger.info(f"=== Type 2 switch successful ===")
        self.logger.info(f"Triple point migrated: triangle {old_triangle} → {target_triangle}")
        self.logger.info(f"VP {moving_vp_idx} moved:")
        self.logger.info(f"  Old: edge {old_edge}, λ = {old_lambda:.6f}")
        self.logger.info(f"  New: edge {target_edge}, λ = {new_lambda:.6f}")
        self.logger.info(f"Anchor VP {anchor_vp_idx} stayed on shared edge {shared_edge}")
        self.logger.info(f"Staying VP {staying_vp_idx} remains in source triangle {old_triangle}")
        
        return True
    
    def _select_vp_closest_to_shared_edge(self, vp_indices: List[int], 
                                           shared_edge: Tuple[int, int]) -> Optional[int]:
        """
        Select VP closest to a vertex on the shared edge.
        
        For Type 2 switches: from remaining VPs, choose the one whose edge shares
        a vertex with the shared edge AND is closest to that shared vertex.
        This minimizes the "jump" distance.
        
        Args:
            vp_indices: List of candidate VP indices
            shared_edge: The shared edge (anchor VP is here)
            
        Returns:
            Index of VP to move
        """
        shared_vertices = set(shared_edge)
        min_dist_to_shared_vertex = float('inf')
        best_vp = None
        
        for vp_idx in vp_indices:
            vp = self.partition.variable_points[vp_idx]
            vp_vertices = set(vp.edge)
            
            # Find shared vertex (if any)
            common_vertices = shared_vertices & vp_vertices
            
            if not common_vertices:
                continue
            
            # Calculate distance to shared vertex
            # Determine which end of vp.edge is the shared vertex
            shared_vertex = list(common_vertices)[0]
            
            # CRITICAL: λ convention is position = λ * edge[0] + (1-λ) * edge[1]
            # So: distance to edge[0] = 1-λ, distance to edge[1] = λ
            if vp.edge[0] == shared_vertex:
                # Shared vertex is at edge[0] → distance = 1 - λ
                dist = 1.0 - vp.lambda_param
            else:
                # Shared vertex is at edge[1] → distance = λ
                dist = vp.lambda_param
            
            if dist < min_dist_to_shared_vertex:
                min_dist_to_shared_vertex = dist
                best_vp = vp_idx
        
        if best_vp is not None:
            self.logger.info(f"Selected VP {best_vp} to move (distance to shared vertex = {min_dist_to_shared_vertex:.6f})")
        
        return best_vp
    
    def _identify_target_vertex(self, vp: VariablePoint) -> Optional[int]:
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
        Determine λ value for VP on new edge after topology switch.
        
        Following the paper's initialization strategy (Section 5), we place
        the VP at the midpoint (λ = 0.5) to give the optimizer maximum freedom
        to find the optimal position. This is more neutral than biasing toward
        either vertex.
        
        Args:
            edge: Target edge
            target_vertex: Vertex to place VP near (used for validation only)
            tol: Unused (kept for API compatibility)
            
        Returns:
            Lambda value (always 0.5 for neutral initialization)
        """
        # Validate that target vertex is actually on this edge
        if target_vertex not in edge:
            self.logger.warning(f"Target vertex {target_vertex} not in edge {edge}")
        
        # Always use midpoint for neutral initialization
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
    
    # =========================================================================
    # Phase 4: Segment Classification and Cross-Triangle Handling
    # =========================================================================
    
    def classify_segment(self, vp_idx1: int, vp_idx2: int) -> str:
        """
        Classify a segment between two VPs.
        
        Returns:
            "normal": Both VPs in same triangle (standard case)
            "edge_following": VPs in different triangles but segment follows mesh edge
            "edge_cutting": Segment cuts across triangle edges
        """
        edge1 = self.partition.variable_points[vp_idx1].edge
        edge2 = self.partition.variable_points[vp_idx2].edge
        
        # Step 1: Check if edges share a triangle (NORMAL case)
        if self._edges_share_triangle(edge1, edge2):
            return "normal"
        
        # --- VPs are in different triangles (cross-triangle segment) ---
        
        # Step 2: Check if edges share a vertex
        shared_vertices = set(edge1) & set(edge2)
        
        if not shared_vertices:
            # No shared vertex - definitely EDGE-CUTTING
            return "edge_cutting"
        
        shared_vertex = list(shared_vertices)[0]
        
        # Step 3: Edges share vertex - check COLLINEARITY
        if self._edges_are_collinear(edge1, edge2, shared_vertex):
            # Edges are on same mesh line - EDGE-FOLLOWING
            return "edge_following"
        else:
            # Edges form an angle - EDGE-CUTTING (even though they share vertex)
            return "edge_cutting"
    
    def _edges_share_triangle(self, edge1: Tuple[int, int], edge2: Tuple[int, int]) -> bool:
        """Check if two edges are on the same mesh triangle."""
        edge1_norm = tuple(sorted(edge1))
        edge2_norm = tuple(sorted(edge2))
        
        triangles1 = self.mesh_topology.get_triangles_sharing_edge(edge1_norm)
        triangles2 = self.mesh_topology.get_triangles_sharing_edge(edge2_norm)
        
        # Convert to sets for intersection
        return bool(set(triangles1) & set(triangles2))
    
    def _edges_are_collinear(self, edge1: Tuple[int, int], edge2: Tuple[int, int], 
                             shared_vertex: int) -> bool:
        """
        Check if two edges that share a vertex are collinear (on same mesh line).
        
        Uses the other vertices of each edge to form vectors from shared vertex,
        then checks if they are parallel or anti-parallel.
        """
        # Get the "other" vertex from each edge
        other1 = edge1[1] if edge1[0] == shared_vertex else edge1[0]
        other2 = edge2[1] if edge2[0] == shared_vertex else edge2[0]
        
        # Get positions
        shared_pos = self.mesh.vertices[shared_vertex]
        other1_pos = self.mesh.vertices[other1]
        other2_pos = self.mesh.vertices[other2]
        
        # Vectors from shared vertex
        vec1 = other1_pos - shared_pos
        vec2 = other2_pos - shared_pos
        
        # Normalize
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 < 1e-10 or norm2 < 1e-10:
            return False
        
        vec1 = vec1 / norm1
        vec2 = vec2 / norm2
        
        # Check if parallel or anti-parallel (dot product ≈ ±1)
        dot = np.abs(np.dot(vec1, vec2))
        return dot > 0.99  # Tolerance for numerical precision
    
    def classify_all_segments(self):
        """
        Classify ALL boundary segments and compute crossing info for edge_cutting ones.
        
        Called after rebuild_triangle_segments_from_current_vps() to set correct
        segment types and populate the segment_crossing_cache.
        """
        # Clear the crossing cache before rebuilding
        self.partition.segment_crossing_cache.clear()
        
        for seg in self.partition.boundary_segments:
            # Classify this segment
            seg.segment_type = self.classify_segment(seg.vp_idx_1, seg.vp_idx_2)
            
            # If edge_cutting, compute which triangles it crosses
            if seg.segment_type == "edge_cutting":
                seg.crossing_triangles = self._find_crossed_triangles(
                    seg.vp_idx_1, seg.vp_idx_2
                )
                
                # Compute and cache crossing info for area calculations
                self._compute_and_cache_crossings(seg)
            else:
                seg.crossing_triangles = []
        
        # Log summary
        num_normal = sum(1 for s in self.partition.boundary_segments if s.segment_type == "normal")
        num_following = sum(1 for s in self.partition.boundary_segments if s.segment_type == "edge_following")
        num_cutting = sum(1 for s in self.partition.boundary_segments if s.segment_type == "edge_cutting")
        
        self.logger.info(f"Segment classification: {num_normal} normal, "
                        f"{num_following} edge_following, {num_cutting} edge_cutting")
        
        if num_cutting > 0:
            total_crossings = sum(len(crossings) for crossings in self.partition.segment_crossing_cache.values())
            self.logger.info(f"  Cached {total_crossings} crossing infos across {len(self.partition.segment_crossing_cache)} triangles")
    
    def update_segment_classifications_after_switch(self, moved_vp_idx: int):
        """
        After a VP moves, update the classifications of all segments involving it.
        
        This is called after apply_type1_switch() or apply_type2_switch().
        Updates the partition.boundary_segments with correct segment_type values.
        
        Args:
            moved_vp_idx: Index of the VP that was moved
        """
        # Find all BoundarySegments involving this VP
        for seg in self.partition.boundary_segments:
            if seg.vp_idx_1 == moved_vp_idx or seg.vp_idx_2 == moved_vp_idx:
                # Re-classify this segment
                seg.segment_type = self.classify_segment(seg.vp_idx_1, seg.vp_idx_2)
                
                # If edge_cutting, compute which triangles it crosses
                if seg.segment_type == "edge_cutting":
                    seg.crossing_triangles = self._find_crossed_triangles(
                        seg.vp_idx_1, seg.vp_idx_2
                    )
                    
                    # Compute and cache crossing info for area calculations
                    self._compute_and_cache_crossings(seg)
                else:
                    seg.crossing_triangles = []
        
        # Log summary
        num_normal = sum(1 for s in self.partition.boundary_segments if s.segment_type == "normal")
        num_following = sum(1 for s in self.partition.boundary_segments if s.segment_type == "edge_following")
        num_cutting = sum(1 for s in self.partition.boundary_segments if s.segment_type == "edge_cutting")
        
        self.logger.info(f"Segment classification update: {num_normal} normal, "
                        f"{num_following} edge_following, {num_cutting} edge_cutting")
    
    def _find_crossed_triangles(self, vp_idx1: int, vp_idx2: int) -> List[int]:
        """
        Find all triangles that a segment crosses (for edge_cutting segments).
        
        Returns:
            List of triangle indices crossed by the segment
        """
        pos1 = self.partition.evaluate_variable_point(vp_idx1)
        pos2 = self.partition.evaluate_variable_point(vp_idx2)
        
        edge1 = self.partition.variable_points[vp_idx1].edge
        edge2 = self.partition.variable_points[vp_idx2].edge
        
        # Get triangles containing each edge
        triangles1 = self.mesh_topology.get_triangles_sharing_edge(tuple(sorted(edge1)))
        triangles2 = self.mesh_topology.get_triangles_sharing_edge(tuple(sorted(edge2)))
        
        # The segment connects these two sets of triangles
        # We need to find all triangles the line segment passes through
        crossed = []
        
        # Simple approach: check all boundary triangles for intersection
        # This is not the most efficient, but correct
        for tri_seg in self.partition.triangle_segments:
            tri_idx = tri_seg.triangle_idx
            
            # Skip triangles containing either VP
            if tri_idx in triangles1 or tri_idx in triangles2:
                continue
            
            # Check if segment intersects this triangle
            if self._segment_intersects_triangle(pos1, pos2, tri_idx):
                crossed.append(tri_idx)
        
        return crossed
    
    def _segment_intersects_triangle(self, pos1: np.ndarray, pos2: np.ndarray, 
                                     tri_idx: int) -> bool:
        """
        Check if line segment (pos1, pos2) intersects triangle.
        
        Uses edge intersection test: segment intersects triangle if it
        crosses exactly 2 of the triangle's edges.
        """
        face = self.mesh.faces[tri_idx]
        v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
        
        vertices = [
            self.mesh.vertices[v1],
            self.mesh.vertices[v2],
            self.mesh.vertices[v3]
        ]
        
        intersection_count = 0
        
        for i in range(3):
            edge_start = vertices[i]
            edge_end = vertices[(i + 1) % 3]
            
            if self._line_segments_intersect(pos1, pos2, edge_start, edge_end):
                intersection_count += 1
        
        return intersection_count == 2
    
    def _line_segments_intersect(self, p1: np.ndarray, p2: np.ndarray,
                                  q1: np.ndarray, q2: np.ndarray) -> bool:
        """
        Check if 2D/3D line segments (p1,p2) and (q1,q2) intersect.
        
        Uses parametric form and checks if intersection point is within both segments.
        """
        # Direction vectors
        d1 = p2 - p1
        d2 = q2 - q1
        
        # Build matrix for least-squares solve: p1 + t*d1 = q1 + s*d2
        # Rearrange: [d1, -d2] [t, s]^T = q1 - p1
        A = np.column_stack([d1, -d2])
        b = q1 - p1
        
        # Check if system is solvable (not parallel)
        if A.shape[0] == 3:
            # 3D case - use least squares
            try:
                result = np.linalg.lstsq(A, b, rcond=None)
                params = result[0]
                residual = result[1]
                
                # Check residual (segments may be skew in 3D)
                if len(residual) > 0 and residual[0] > 1e-6:
                    return False
                
                t, s = params[0], params[1]
            except:
                return False
        else:
            # 2D case - direct solve
            det = d1[0] * (-d2[1]) - d1[1] * (-d2[0])
            if abs(det) < 1e-10:
                return False
            
            t = (b[0] * (-d2[1]) - b[1] * (-d2[0])) / det
            s = (d1[0] * b[1] - d1[1] * b[0]) / det
        
        # Check if intersection is within both segments (with small tolerance)
        eps = 1e-6
        return (eps < t < 1 - eps) and (eps < s < 1 - eps)
    
    def _compute_and_cache_crossings(self, segment: BoundarySegment):
        """
        Compute detailed crossing information for an edge_cutting segment.
        
        For each triangle the segment crosses, compute entry/exit points
        and store in partition.segment_crossing_cache.
        """
        vp_idx1 = segment.vp_idx_1
        vp_idx2 = segment.vp_idx_2
        pos1 = self.partition.evaluate_variable_point(vp_idx1)
        pos2 = self.partition.evaluate_variable_point(vp_idx2)
        
        for tri_idx in segment.crossing_triangles:
            entry_point, exit_point, entry_edge, exit_edge = \
                self._compute_triangle_crossing_details(pos1, pos2, tri_idx)
            
            if entry_point is not None and exit_point is not None:
                # Determine which cell this crossing belongs to
                # Use the cells that both VPs separate
                cells1 = self.partition.variable_points[vp_idx1].belongs_to_cells
                cells2 = self.partition.variable_points[vp_idx2].belongs_to_cells
                shared_cells = list(cells1 & cells2)
                cell_idx = shared_cells[0] if shared_cells else 0
                
                crossing_info = SegmentCrossingInfo(
                    segment=(min(vp_idx1, vp_idx2), max(vp_idx1, vp_idx2)),
                    triangle_idx=tri_idx,
                    entry_point=entry_point,
                    exit_point=exit_point,
                    entry_edge=entry_edge,
                    exit_edge=exit_edge,
                    cell_idx=cell_idx
                )
                
                # Store in cache
                if tri_idx not in self.partition.segment_crossing_cache:
                    self.partition.segment_crossing_cache[tri_idx] = []
                self.partition.segment_crossing_cache[tri_idx].append(crossing_info)
    
    def _compute_triangle_crossing_details(self, pos1: np.ndarray, pos2: np.ndarray,
                                           tri_idx: int) -> Tuple[
                                               Optional[np.ndarray], 
                                               Optional[np.ndarray],
                                               Optional[Tuple[int, int]],
                                               Optional[Tuple[int, int]]]:
        """
        Compute where segment (pos1, pos2) enters and exits a triangle.
        
        Returns:
            (entry_point, exit_point, entry_edge, exit_edge) or (None, None, None, None)
        """
        face = self.mesh.faces[tri_idx]
        v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
        
        edges = [
            ((v1, v2), self.mesh.vertices[v1], self.mesh.vertices[v2]),
            ((v2, v3), self.mesh.vertices[v2], self.mesh.vertices[v3]),
            ((v3, v1), self.mesh.vertices[v3], self.mesh.vertices[v1])
        ]
        
        intersections = []
        
        for edge, edge_start, edge_end in edges:
            intersection = self._compute_line_edge_intersection(pos1, pos2, edge_start, edge_end)
            if intersection is not None:
                intersections.append((edge, intersection))
        
        if len(intersections) != 2:
            return None, None, None, None
        
        # Determine which is entry and which is exit based on direction from pos1 to pos2
        (edge_a, point_a), (edge_b, point_b) = intersections
        
        # Compute parameter along segment
        segment_dir = pos2 - pos1
        segment_len = np.linalg.norm(segment_dir)
        if segment_len < 1e-10:
            return None, None, None, None
        
        t_a = np.dot(point_a - pos1, segment_dir) / (segment_len ** 2)
        t_b = np.dot(point_b - pos1, segment_dir) / (segment_len ** 2)
        
        if t_a < t_b:
            return point_a, point_b, tuple(sorted(edge_a)), tuple(sorted(edge_b))
        else:
            return point_b, point_a, tuple(sorted(edge_b)), tuple(sorted(edge_a))
    
    def _compute_line_edge_intersection(self, p1: np.ndarray, p2: np.ndarray,
                                        q1: np.ndarray, q2: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute intersection point of line segment (p1, p2) with edge (q1, q2).
        
        Returns:
            Intersection point or None if no intersection
        """
        d1 = p2 - p1
        d2 = q2 - q1
        
        # Build system: p1 + t*d1 = q1 + s*d2
        A = np.column_stack([d1, -d2])
        b = q1 - p1
        
        # Check if parallel
        if A.shape[0] == 3:
            try:
                result = np.linalg.lstsq(A, b, rcond=None)
                params = result[0]
                residual = result[1]
                
                if len(residual) > 0 and residual[0] > 1e-6:
                    return None
                
                t, s = params[0], params[1]
            except:
                return None
        else:
            det = d1[0] * (-d2[1]) - d1[1] * (-d2[0])
            if abs(det) < 1e-10:
                return None
            
            t = (b[0] * (-d2[1]) - b[1] * (-d2[0])) / det
            s = (d1[0] * b[1] - d1[1] * b[0]) / det
        
        # Check if intersection is on the edge (s in [0, 1]) and within segment (t in [0, 1])
        eps = 1e-6
        if not (0 - eps <= s <= 1 + eps and 0 - eps <= t <= 1 + eps):
            return None
        
        return q1 + s * d2

