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
    
    def get_best_target_edge_for_type1(self, vp_idx: int, tol: float = 0.1) -> Optional[Tuple[int, int]]:
        """
        Determine the best target edge for a Type 1 switch WITHOUT moving the VP.
        
        This method is used by visualization tools to show the target edge.
        
        New Algorithm:
        1. Find all triangles at target vertex
        2. Filter to empty triangles (no boundary segments)
        3. Find free edges in empty triangles
        4. Select edge with minimum path length VP1→VP2→VP3 among candidates
        
        Note: This selects the best option among available edges, not necessarily
        the one that immediately reduces path length.
        
        Args:
            vp_idx: Index of variable point
            tol: Lambda tolerance (default 0.1)
            
        Returns:
            Best target edge tuple, or None if no valid edge found
        """
        vp = self.partition.variable_points[vp_idx]
        
        # 1. Identify target vertex
        target_vertex = self._identify_target_vertex(vp)
        
        if target_vertex is None:
            return None
        
        # 2. Find all triangles at target vertex
        triangles_at_vertex = self._get_all_triangles_at_vertex(target_vertex)
        
        if not triangles_at_vertex:
            return None
        
        # 3. Filter to empty triangles (no boundary segments)
        empty_triangles = []
        occupied_triangles = []
        
        for tri_idx in triangles_at_vertex:
            if self._triangle_has_boundary_segment(tri_idx):
                occupied_triangles.append(tri_idx)
            else:
                empty_triangles.append(tri_idx)
        
        if not empty_triangles:
            return None
        
        # 4. Collect all free edges from empty triangles
        candidate_edges = []
        for tri_idx in empty_triangles:
            free_edges = self._get_free_edges_in_triangle(tri_idx, target_vertex)
            candidate_edges.extend(free_edges)
        
        # Remove duplicates (edge might appear in multiple triangles)
        candidate_edges = list(set(candidate_edges))
        
        if not candidate_edges:
            return None
        
        # 4.5 NEW: Filter out edges that would create triple points
        safe_candidate_edges = [
            edge for edge in candidate_edges
            if not self._edge_would_create_triple_point(edge, occupied_triangles)
        ]
        
        if not safe_candidate_edges:
            return None
        
        # 5. Get neighboring variable points
        neighbors = self._get_neighboring_variable_points(vp_idx)
        
        if not neighbors:
            return None
        
        # 6. Test all SAFE candidates and select minimum path length
        best_edge = None
        min_path_length = float('inf')
        
        for candidate_edge in safe_candidate_edges:  # Use safe_candidate_edges
            # Determine lambda orientation (close to target vertex)
            test_lambda = self._get_lambda_near_vertex(candidate_edge, target_vertex, tol)
            
            # Compute total path length VP1 → VP_test → VP3
            path_length = self._compute_total_segment_length(
                vp_idx, candidate_edge, test_lambda, neighbors
            )
            
            if path_length < min_path_length:
                min_path_length = path_length
                best_edge = candidate_edge
        
        return best_edge
    
    def apply_type1_switch(self, vp_idx: int, tol: float = 0.1) -> bool:
        """
        Apply Type 1 switch: move variable point to adjacent edge.
        
        New Algorithm:
        1. Identify target vertex (which endpoint VP is approaching)
        2. Find all triangles at target vertex
        3. Filter to empty triangles (no boundary segments passing through)
        4. Find all free edges in empty triangles
        5. Test all candidates: compute path length VP1→VP2→VP3
        6. Move VP to edge with minimum path length among candidates
        7. Place VP at λ = 0.5 on new edge (midpoint for neutral initialization)
        
        This prevents triple point creation by avoiding triangles that already
        have boundary segments.
        
        Note: The path length VP1→VP2→VP3 may increase temporarily after migration
        (VP placed at midpoint), but subsequent optimization iterations will refine
        λ values to minimize perimeter. The goal is to select the best edge among
        available options, preventing anti-natural configurations.
        
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
        
        # 2. Find all triangles at target vertex
        triangles_at_vertex = self._get_all_triangles_at_vertex(target_vertex)
        
        if not triangles_at_vertex:
            self.logger.warning(f"VP {vp_idx}: No triangles found at target vertex {target_vertex}")
            return False
        
        # 3. Filter to empty triangles (no boundary segments)
        empty_triangles = []
        occupied_triangles = []
        
        for tri_idx in triangles_at_vertex:
            if self._triangle_has_boundary_segment(tri_idx):
                occupied_triangles.append(tri_idx)
            else:
                empty_triangles.append(tri_idx)
        
        if not empty_triangles:
            self.logger.warning(f"VP {vp_idx}: No empty triangles at target vertex {target_vertex} - Type 1 migration not possible!")
            self.logger.warning(f"  All {len(triangles_at_vertex)} triangles at vertex have boundary segments.")
            self.logger.warning(f"  This indicates a complex geometric configuration that requires investigation.")
            return False
        
        # 4. Collect all free edges from empty triangles
        candidate_edges = []
        for tri_idx in empty_triangles:
            free_edges = self._get_free_edges_in_triangle(tri_idx, target_vertex)
            candidate_edges.extend(free_edges)
        
        # Remove duplicates (edge might appear in multiple triangles)
        candidate_edges = list(set(candidate_edges))
        
        if not candidate_edges:
            self.logger.warning(f"VP {vp_idx}: No free edges found in empty triangles")
            return False
        
        # 4.5 NEW: Filter out edges that would create triple points
        # An edge creates a triple point if it's shared with an occupied triangle
        safe_candidate_edges = [
            edge for edge in candidate_edges
            if not self._edge_would_create_triple_point(edge, occupied_triangles)
        ]
        
        if not safe_candidate_edges:
            self.logger.warning(f"VP {vp_idx}: All {len(candidate_edges)} candidate edges would create triple points!")
            self.logger.warning(f"  Found {len(occupied_triangles)} occupied triangles at vertex {target_vertex}.")
            self.logger.warning(f"  No safe migration path available.")
            return False
        
        self.logger.info(f"VP {vp_idx}: Filtered {len(candidate_edges)} candidates → {len(safe_candidate_edges)} safe edges")
        
        # 5. Get neighboring variable points
        neighbors = self._get_neighboring_variable_points(vp_idx)
        
        if not neighbors:
            self.logger.warning(f"VP {vp_idx}: No neighboring variable points found")
            return False
        
        # 6. Test all SAFE candidates and select minimum path length VP1→VP2→VP3
        best_edge = None
        best_lambda = None
        min_path_length = float('inf')
        
        for candidate_edge in safe_candidate_edges:  # Use safe_candidate_edges instead of candidate_edges
            # Determine lambda orientation (close to target vertex)
            test_lambda = self._get_lambda_near_vertex(candidate_edge, target_vertex, tol)
            
            # Compute total path length VP1 → VP_test → VP3
            path_length = self._compute_total_segment_length(
                vp_idx, candidate_edge, test_lambda, neighbors
            )
            
            if path_length < min_path_length:
                min_path_length = path_length
                best_edge = candidate_edge
                best_lambda = test_lambda
        
        if best_edge is None:
            self.logger.warning(f"VP {vp_idx}: Could not find valid target edge")
            return False
        
        # Save old state before moving (for logging)
        old_edge = vp.edge
        old_lambda = vp.lambda_param
        
        # 7. Move the variable point
        self._move_variable_point(vp_idx, best_edge, best_lambda)
        
        self.logger.info(f"VP {vp_idx}: Type 1 switch successful")
        self.logger.info(f"  Old edge: {old_edge}, λ = {old_lambda:.3f}")
        self.logger.info(f"  New edge: {best_edge}, λ = {best_lambda:.3f}")
        self.logger.info(f"  Selected path length (VP1+VP2+VP3): {min_path_length:.6f}")
        self.logger.info(f"  Note: Path length may increase temporarily; subsequent optimization will refine λ values.")
        
        return True
    
    def apply_type2_switch(self, triple_point: TriplePoint, tol: float = 0.1) -> bool:
        """
        Apply Type 2 switch: migrate triple point to adjacent triangle.
        
        Algorithm:
        1. Find shared edge (where Steiner point is closest)
        2. Identify anchor VP (on shared edge) and remaining VPs
        3. Find target triangle (shares the anchor edge)
        4. Find free edge in target triangle
        5. Find new VP in target triangle (on the third edge)
        6. Select VP to move (minimizes resulting perimeter)
        7. Move VP to free edge with λ=0.5
        8. Update boundary_segments for topology change
        
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
        
        # Step 3: Find target triangle (shares the anchor edge)
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
        
        # Step 4: Find free edge in target triangle
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
        
        target_edge = normalize_edge(free_edges[0])
        self.logger.info(f"Free edge in target triangle: {target_edge}")
        
        # Step 5: Find new VP in target triangle (on the third edge, not shared, not free)
        new_vp_in_target_idx = self._find_new_vp_in_target_triangle(
            target_triangle, shared_edge, target_edge
        )
        
        if new_vp_in_target_idx is None:
            self.logger.warning(f"Could not find new VP in target triangle")
            return False
        
        # Step 6: Select which VP to move (minimizes resulting perimeter)
        moving_vp_idx = self._select_vp_minimizing_perimeter(
            remaining_vp_indices, anchor_vp_idx, target_edge, new_vp_in_target_idx
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
        
        # Step 7: Move VP to free edge with λ=0.5
        new_lambda = 0.5  # Center of edge
        self._move_variable_point(moving_vp_idx, target_edge, new_lambda)
        
        self.logger.info(f"=== Type 2 switch successful ===")
        self.logger.info(f"Triple point migrated: triangle {old_triangle} → {target_triangle}")
        self.logger.info(f"VP {moving_vp_idx} moved:")
        self.logger.info(f"  Old: edge {old_edge}, λ = {old_lambda:.6f}")
        self.logger.info(f"  New: edge {target_edge}, λ = {new_lambda:.6f}")
        self.logger.info(f"Anchor VP {anchor_vp_idx} stayed on shared edge {shared_edge}")
        self.logger.info(f"Staying VP {staying_vp_idx} remains in source triangle {old_triangle}")
        
        # Step 8: Update boundary_segments for the segment topology change
        self._update_segments_for_type2_switch(
            moving_vp_idx, staying_vp_idx, anchor_vp_idx, new_vp_in_target_idx
        )
        
        return True
    
    def _select_vp_minimizing_perimeter(self, vp_indices: List[int],
                                         anchor_vp_idx: int,
                                         target_edge: Tuple[int, int],
                                         new_vp_in_target_idx: int) -> Optional[int]:
        """
        Select VP that minimizes total segment length after the switch.
        
        For each candidate VP, compute the total perimeter contribution if that VP
        moved to target_edge at λ=0.5. Select the one with minimum total.
        
        This approach mirrors Type 1 switch logic where candidate edges are evaluated
        based on total segment length to neighbors.
        
        Considers:
        1. New void edges: (moving_vp → anchor), (moving_vp → new_vp_in_target)
        2. External boundary segments from moving_vp to its non-void neighbors
        
        Args:
            vp_indices: List of candidate VP indices (the 2 remaining VPs)
            anchor_vp_idx: VP on the shared edge (stays in place)
            target_edge: The free edge where the moving VP will land
            new_vp_in_target_idx: VP already in target triangle (joins new void)
            
        Returns:
            Index of VP to move (the one minimizing perimeter)
        """
        # Compute target position (midpoint of free edge)
        v1, v2 = target_edge
        target_pos = 0.5 * self.mesh.vertices[v1] + 0.5 * self.mesh.vertices[v2]
        
        # Get positions of VPs that will be in the new void triangle
        anchor_pos = self.partition.evaluate_variable_point(anchor_vp_idx)
        new_vp_pos = self.partition.evaluate_variable_point(new_vp_in_target_idx)
        
        min_total = float('inf')
        best_vp = None
        
        # VPs in the current void triangle (to exclude from "external" neighbors)
        void_vps = set(vp_indices) | {anchor_vp_idx}
        
        self.logger.info(f"Evaluating VP candidates for perimeter minimization:")
        
        for vp_idx in vp_indices:
            # Get all neighbors of this VP
            neighbors = self._get_neighboring_variable_points(vp_idx)
            
            # External neighbors (not in current void triangle)
            external_neighbors = [n for n in neighbors if n not in void_vps]
            
            # Compute total segment length if this VP moves to target_edge at λ=0.5:
            total = 0.0
            
            # 1. New void edge to anchor
            dist_to_anchor = np.linalg.norm(target_pos - anchor_pos)
            total += dist_to_anchor
            
            # 2. New void edge to new_vp_in_target
            dist_to_new_vp = np.linalg.norm(target_pos - new_vp_pos)
            total += dist_to_new_vp
            
            # 3. External segments (moving VP stays connected to these)
            external_dist = 0.0
            for ext_idx in external_neighbors:
                ext_pos = self.partition.evaluate_variable_point(ext_idx)
                external_dist += np.linalg.norm(target_pos - ext_pos)
            total += external_dist
            
            self.logger.info(f"  VP {vp_idx}: void edges = {dist_to_anchor:.6f} + {dist_to_new_vp:.6f}, "
                           f"external = {external_dist:.6f}, total = {total:.6f}")
            
            if total < min_total:
                min_total = total
                best_vp = vp_idx
        
        if best_vp is not None:
            self.logger.info(f"Selected VP {best_vp} to move (minimizes perimeter, total = {min_total:.6f})")
        
        return best_vp
    
    def _find_new_vp_in_target_triangle(self, target_triangle: int, 
                                         shared_edge: Tuple[int, int],
                                         free_edge: Tuple[int, int]) -> Optional[int]:
        """
        Find the VP in the target triangle that's on the third edge.
        
        The target triangle has 3 edges:
        1. shared_edge - where anchor VP is
        2. free_edge - where moving VP will land
        3. third edge - where the "new VP" is (the one that joins the new void triangle)
        
        Args:
            target_triangle: Index of the target triangle
            shared_edge: Edge shared with source triangle (anchor VP here)
            free_edge: Edge where moving VP lands
            
        Returns:
            Index of the VP on the third edge, or None if not found
        """
        target_tri_edges = self.mesh.get_triangle_edges(target_triangle)
        
        # Normalize for comparison
        shared_norm = tuple(sorted(shared_edge))
        free_norm = tuple(sorted(free_edge))
        
        # Find the third edge
        third_edge = None
        for edge in target_tri_edges:
            edge_norm = tuple(sorted(edge))
            if edge_norm != shared_norm and edge_norm != free_norm:
                third_edge = edge_norm
                break
        
        if third_edge is None:
            self.logger.warning(f"Could not find third edge in target triangle {target_triangle}")
            return None
        
        # Find VP on this edge
        if third_edge in self.partition.edge_to_varpoint:
            new_vp_idx = self.partition.edge_to_varpoint[third_edge]
            self.logger.info(f"  New VP in target triangle: VP {new_vp_idx} on edge {third_edge}")
            return new_vp_idx
        
        self.logger.warning(f"No VP found on third edge {third_edge} of target triangle")
        return None
    
    def _update_segments_for_type2_switch(self, moving_vp_idx: int, staying_vp_idx: int,
                                           anchor_vp_idx: int, new_vp_in_target_idx: int):
        """
        Update boundary_segments for Type 2 switch topology change.
        
        Type 2 switch changes segment connectivity:
        - DESTROYED: (moving_vp, staying_vp) - this connection is broken
        - CREATED: (moving_vp, new_vp_in_target) - new void edge
        - UNCHANGED: (anchor_vp, staying_vp) - changes role from void to boundary
        - UNCHANGED: (anchor_vp, moving_vp) - still void edge, now in new triangle
        
        Args:
            moving_vp_idx: VP that moved to new triangle
            staying_vp_idx: VP that stayed in old triangle (no longer in void)
            anchor_vp_idx: VP on shared edge (unchanged position)
            new_vp_in_target_idx: VP already in target triangle (joins new void)
        """
        # Step 1: Remove destroyed segment (moving, staying)
        destroyed_key = (min(moving_vp_idx, staying_vp_idx), max(moving_vp_idx, staying_vp_idx))
        
        segments_before = len(self.partition.boundary_segments)
        self.partition.boundary_segments = [
            seg for seg in self.partition.boundary_segments
            if seg.normalized_key() != destroyed_key
        ]
        segments_after_remove = len(self.partition.boundary_segments)
        
        if segments_after_remove < segments_before:
            self.logger.info(f"  Removed segment ({moving_vp_idx}, {staying_vp_idx}) - destroyed")
        else:
            self.logger.warning(f"  Segment ({moving_vp_idx}, {staying_vp_idx}) not found in boundary_segments")
        
        # Step 2: Add new segment (moving, new_vp_in_target)
        new_key = (min(moving_vp_idx, new_vp_in_target_idx), max(moving_vp_idx, new_vp_in_target_idx))
        
        # Check if segment already exists (shouldn't, but be safe)
        existing = any(seg.normalized_key() == new_key for seg in self.partition.boundary_segments)
        
        if not existing:
            # Determine cell pair from VP membership
            cells_moving = self.partition.variable_points[moving_vp_idx].belongs_to_cells
            cells_new = self.partition.variable_points[new_vp_in_target_idx].belongs_to_cells
            shared_cells = cells_moving & cells_new
            cell_pair = tuple(sorted(shared_cells)) if len(shared_cells) == 2 else (0, 0)
            
            new_segment = BoundarySegment(
                vp_idx_1=moving_vp_idx,
                vp_idx_2=new_vp_in_target_idx,
                cell_pair=cell_pair,
                segment_type="normal"  # Will be reclassified later
            )
            self.partition.boundary_segments.append(new_segment)
            self.logger.info(f"  Added segment ({moving_vp_idx}, {new_vp_in_target_idx}) - new void edge")
        else:
            self.logger.info(f"  Segment ({moving_vp_idx}, {new_vp_in_target_idx}) already exists")
        
        self.logger.info(f"  Segment count: {segments_before} → {len(self.partition.boundary_segments)}")
    
    # =========================================================================
    # Type 1 Helper Methods (New Algorithm)
    # =========================================================================
    
    def _get_all_triangles_at_vertex(self, target_vertex: int) -> List[int]:
        """
        Find all triangles that include target_vertex.
        
        Args:
            target_vertex: Vertex index
            
        Returns:
            List of triangle indices
        """
        return self.mesh_topology.get_triangles_at_vertex(target_vertex)
    
    def _triangle_has_boundary_segment(self, tri_idx: int) -> bool:
        """
        Check if a triangle has a boundary segment passing through it.
        
        A triangle is "occupied" if it has 2+ VPs on its edges, meaning
        a boundary segment crosses it.
        
        Args:
            tri_idx: Triangle index
            
        Returns:
            True if triangle has a boundary segment, False if empty
        """
        # Get triangle edges
        tri_edges = self.mesh.get_triangle_edges(tri_idx)
        
        # Count VPs on this triangle's edges
        vp_count = 0
        occupied_edges = set(tuple(sorted(vp.edge)) for vp in self.partition.variable_points)
        
        for edge in tri_edges:
            edge_norm = tuple(sorted(edge))
            if edge_norm in occupied_edges:
                vp_count += 1
        
        # Triangle has segment if 2+ VPs (forms a segment)
        return vp_count >= 2
    
    def _edge_would_create_triple_point(self, candidate_edge: Tuple[int, int], 
                                        occupied_triangles: List[int]) -> bool:
        """
        Check if moving a VP to candidate_edge would create a triple point.
        
        This happens when the candidate edge is shared with a triangle that
        already has 2+ VPs (an occupied triangle). Moving a VP to this edge
        would give that triangle 3 VPs → triple point.
        
        Args:
            candidate_edge: Edge being considered for VP migration
            occupied_triangles: List of triangle indices that already have 2+ VPs
            
        Returns:
            True if moving here would create a triple point, False otherwise
        """
        # Get triangles that share this edge
        edge_norm = tuple(sorted(candidate_edge))
        triangles_sharing_edge = self.mesh_topology.get_triangles_sharing_edge(edge_norm)
        
        # Check if any of these triangles are occupied
        for tri_idx in triangles_sharing_edge:
            if tri_idx in occupied_triangles:
                # This edge is shared with an occupied triangle
                # Moving a VP here would give that triangle 3 VPs → triple point
                return True
        
        return False
    
    def _get_free_edges_in_triangle(self, tri_idx: int, target_vertex: int) -> List[Tuple[int, int]]:
        """
        Get free edges in a triangle that include target_vertex.
        
        Args:
            tri_idx: Triangle index
            target_vertex: Required vertex
            
        Returns:
            List of free edges (no VP on them) that include target_vertex
        """
        tri_edges = self.mesh.get_triangle_edges(tri_idx)
        occupied_edges = set(tuple(sorted(vp.edge)) for vp in self.partition.variable_points)
        
        free_edges = []
        for edge in tri_edges:
            edge_norm = tuple(sorted(edge))
            if target_vertex in edge and edge_norm not in occupied_edges:
                free_edges.append(edge_norm)
        
        return free_edges
    
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
        Classify a segment between two VPs based on geometric configuration.
        
        Returns:
            "normal": Both VPs in same triangle (standard case)
            "edge_following": Segment through shared vertex (Type 2 geometry)
            "edge_cutting": Segment crosses shared edge (Type 1 geometry)
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
            # No shared vertex → definitely edge_cutting
            return "edge_cutting"
        
        # Step 3: Edges share a vertex - now check if triangles share an edge
        # Get all triangles containing each edge
        triangles1 = self.mesh_topology.get_triangles_sharing_edge(tuple(sorted(edge1)))
        triangles2 = self.mesh_topology.get_triangles_sharing_edge(tuple(sorted(edge2)))
        
        # Check if any pair of triangles share an edge
        for tri1 in triangles1:
            for tri2 in triangles2:
                if tri1 == tri2:
                    continue  # Same triangle already handled
                
                shared_edge = self._find_shared_edge_between_triangles(tri1, tri2)
                if shared_edge is not None:
                    # Triangles share an edge → Type 1 case
                    # Segment crosses the shared edge (needs dual projection)
                    return "edge_cutting"
        
        # Edges share a vertex BUT triangles don't share an edge → Type 2 case
        # Segment appears to go through shared vertex (use vertex for area calc)
        return "edge_following"
    
    def _edges_share_triangle(self, edge1: Tuple[int, int], edge2: Tuple[int, int]) -> bool:
        """Check if two edges are on the same mesh triangle."""
        edge1_norm = tuple(sorted(edge1))
        edge2_norm = tuple(sorted(edge2))
        
        triangles1 = self.mesh_topology.get_triangles_sharing_edge(edge1_norm)
        triangles2 = self.mesh_topology.get_triangles_sharing_edge(edge2_norm)
        
        # Convert to sets for intersection
        return bool(set(triangles1) & set(triangles2))
    
    def classify_all_segments(self):
        """
        Classify ALL boundary segments and compute crossing info for ALL cross-triangle ones.
        
        NEW: Computes crossing info for BOTH edge_cutting AND edge_following segments.
        Classification is now based on WHERE the crossing point lands (vertex vs edge interior).
        
        Called after rebuild_triangle_segments_from_current_vps() to set correct
        segment types and populate the segment_crossing_cache.
        """
        # Clear the crossing cache before rebuilding
        self.partition.segment_crossing_cache.clear()
        
        self.logger.info(f"DEBUG: classify_all_segments() - Processing {len(self.partition.boundary_segments)} boundary segments")
        
        for seg in self.partition.boundary_segments:
            # Initial classification (will be refined after crossing computation)
            seg.segment_type = self.classify_segment(seg.vp_idx_1, seg.vp_idx_2)
            
            # NEW: Compute crossings for ALL cross-triangle segments (not just edge_cutting)
            if seg.segment_type != "normal":  # Both edge_cutting AND edge_following
                self.logger.info(f"  Segment ({seg.vp_idx_1}, {seg.vp_idx_2}): Initial type = {seg.segment_type}")
                
                seg.crossing_triangles = self._find_crossed_triangles(
                    seg.vp_idx_1, seg.vp_idx_2, seg.segment_type
                )
                
                if seg.crossing_triangles:
                    self.logger.info(f"    Found {len(seg.crossing_triangles)} crossing triangles: {seg.crossing_triangles}")
                    # Compute and cache crossing info (includes vertex detection)
                    self._compute_and_cache_crossings(seg)
                else:
                    self.logger.info(f"    WARNING: No crossing triangles found for cross-triangle segment!")
                    
                    # NEW: Refine classification based on actual crossing geometry
                    # Check if all crossings are at vertices
                    all_at_vertices = True
                    if seg.crossing_triangles:
                        for tri_idx in seg.crossing_triangles:
                            if tri_idx in self.partition.segment_crossing_cache:
                                for crossing in self.partition.segment_crossing_cache[tri_idx]:
                                    if crossing.segment == (min(seg.vp_idx_1, seg.vp_idx_2), 
                                                           max(seg.vp_idx_1, seg.vp_idx_2)):
                                        if not crossing.is_vertex_crossing:
                                            all_at_vertices = False
                                            break
                    
                    # Update classification based on geometry
                    if all_at_vertices and seg.crossing_triangles:
                        seg.segment_type = "edge_following"
                    elif seg.crossing_triangles:
                        seg.segment_type = "edge_cutting"
            else:
                seg.crossing_triangles = []
        
        # Log summary
        num_normal = sum(1 for s in self.partition.boundary_segments if s.segment_type == "normal")
        num_following = sum(1 for s in self.partition.boundary_segments if s.segment_type == "edge_following")
        num_cutting = sum(1 for s in self.partition.boundary_segments if s.segment_type == "edge_cutting")
        
        self.logger.info(f"Segment classification: {num_normal} normal, "
                        f"{num_following} edge_following, {num_cutting} edge_cutting")
        
        total_crossings = sum(len(crossings) for crossings in self.partition.segment_crossing_cache.values())
        if total_crossings > 0:
            vertex_crossings = sum(1 for tri_crossings in self.partition.segment_crossing_cache.values()
                                  for c in tri_crossings if c.is_vertex_crossing)
            edge_crossings = total_crossings - vertex_crossings
            self.logger.info(f"  Cached {total_crossings} crossing infos across {len(self.partition.segment_crossing_cache)} triangles")
            self.logger.info(f"    {vertex_crossings} vertex crossings (edge_following), {edge_crossings} edge crossings (edge_cutting)")
    
    def _find_crossed_triangles(self, vp_idx1: int, vp_idx2: int, 
                                 segment_type: str = "normal") -> List[int]:
        """
        Find all triangles that a segment crosses.
        
        Handles three cases:
        - Case 1a: Adjacent triangles (share edge) → Type 1 geometry (edge_cutting)
        - Case 1b: Non-adjacent triangles (share vertex) → Type 2 geometry (edge_following)
        - Case 2: Completely separate triangles → Check all boundary triangles
        
        CRITICAL: Intermediate triangle detection (0 VPs) only for Type 1 (edge_cutting).
        Type 2 (edge_following) segments follow mesh edges and don't need this.
        
        Args:
            vp_idx1: First variable point index
            vp_idx2: Second variable point index
            segment_type: "normal", "edge_following", or "edge_cutting"
        
        Returns:
            List of triangle indices that need crossing info
        """
        self.logger.info(f"  DEBUG: Analyzing segment ({vp_idx1}, {vp_idx2}) for crossings")
        
        pos1 = self.partition.evaluate_variable_point(vp_idx1)
        pos2 = self.partition.evaluate_variable_point(vp_idx2)
        
        edge1 = self.partition.variable_points[vp_idx1].edge
        edge2 = self.partition.variable_points[vp_idx2].edge
        
        self.logger.info(f"    VP {vp_idx1} on edge {edge1}")
        self.logger.info(f"    VP {vp_idx2} on edge {edge2}")
        
        # Get triangles containing each edge
        triangles1 = self.mesh_topology.get_triangles_sharing_edge(tuple(sorted(edge1)))
        triangles2 = self.mesh_topology.get_triangles_sharing_edge(tuple(sorted(edge2)))
        
        self.logger.info(f"    Triangles with VP {vp_idx1}: {triangles1}")
        self.logger.info(f"    Triangles with VP {vp_idx2}: {triangles2}")
        
        # Case 1a: Check if triangles are adjacent (share an edge) - Type 1 geometry
        for tri1 in triangles1:
            for tri2 in triangles2:
                if tri1 == tri2:
                    continue
                
                shared_edge = self._find_shared_edge_between_triangles(tri1, tri2)
                if shared_edge is not None:
                    # Adjacent triangles - Type 1 case
                    # Segment crosses the shared edge (edge_cutting)
                    self.logger.debug(f"  Segment ({vp_idx1}, {vp_idx2}): Type 1 geometry, "
                                     f"triangles {tri1}, {tri2} share edge {shared_edge}")
                    return [tri1, tri2]
        
        # Case 1b: Check if triangles share a vertex - Type 2 geometry (OPTIMIZED!)
        self.logger.info(f"    Checking Case 1b: Do triangles share a vertex? (Type 2)")
        
        # OPTIMIZATION: Check if both VP edges share a vertex
        # If yes, all triangle pairs will share this vertex - test only once
        shared_vertex_on_edges = set(edge1) & set(edge2)
        
        if shared_vertex_on_edges:
            shared_vertex = shared_vertex_on_edges.pop()
            self.logger.info(f"      Both VP edges share vertex {shared_vertex} - testing once")
            
            # Use first valid triangle pair for testing
            tri1 = triangles1[0]
            tri2 = triangles2[0]
            
            if self._segment_passes_through_vertex(vp_idx1, vp_idx2, shared_vertex, tri1, tri2):
                self.logger.info(f"      ✓ Segment PASSES through vertex {shared_vertex} (Type 2 geometry)")
                
                # Check if tri1 and tri2 are adjacent (share edge) or have intermediates
                shared_edge = self._find_shared_edge_between_triangles(tri1, tri2)
                
                if shared_edge is not None:
                    # Adjacent triangles - no intermediates
                    self.logger.info(f"      Triangles {tri1} and {tri2} are ADJACENT (share edge {shared_edge})")
                    result_triangles = []
                    for t1 in triangles1:
                        for t2 in triangles2:
                            if t1 != t2:
                                result_triangles.extend([t1, t2])
                    return list(set(result_triangles))
                else:
                    # Non-adjacent at shared vertex
                    # CRITICAL: Only search for intermediates for Type 1 (edge_cutting)!
                    # Type 2 (edge_following) follows mesh edges and doesn't need this
                    
                    if segment_type == "edge_cutting":
                        # Type 1: Find intermediate triangles with 0 VPs
                        self.logger.info(f"      ⚠️  INTERMEDIATE TRIANGLES DETECTED (Type 1 migration)!")
                        self.logger.info(f"      Triangles {tri1} and {tri2} share vertex {shared_vertex} but NOT an edge")
                        self.logger.info(f"      Searching for intermediate triangles with 0 VPs at vertex {shared_vertex}...")
                        
                        # Find all intermediate triangles
                        all_triangles_in_path = []
                        for t1 in triangles1:
                            for t2 in triangles2:
                                if t1 != t2:
                                    path = self._find_intermediate_triangles_at_vertex(t1, t2, shared_vertex)
                                    all_triangles_in_path.extend(path)
                        
                        result_triangles = list(set(all_triangles_in_path))
                        
                        # Log detailed information
                        num_intermediates = len(result_triangles) - len(triangles1) - len(triangles2)
                        self.logger.info(f"      Found {len(result_triangles)} total triangles:")
                        self.logger.info(f"        - {len(triangles1)} with VP {vp_idx1}")
                        self.logger.info(f"        - {len(triangles2)} with VP {vp_idx2}")
                        self.logger.info(f"        - {num_intermediates} intermediate (0 VPs, 2 crossing points each)")
                        self.logger.info(f"      Segment ({vp_idx1}, {vp_idx2}) crosses: {result_triangles}")
                        
                        return result_triangles
                    else:
                        # Type 2 (edge_following): No intermediate triangles expected
                        # Return all triangles at endpoint edges (they share the vertex)
                        self.logger.info(f"      Type 2 (edge_following): Triangles share vertex {shared_vertex} but not edge")
                        self.logger.info(f"      Returning endpoint triangles only (no intermediate search)")
                        result_triangles = []
                        for t1 in triangles1:
                            for t2 in triangles2:
                                if t1 != t2:
                                    result_triangles.extend([t1, t2])
                        return list(set(result_triangles))
            else:
                self.logger.info(f"      ✗ Segment does NOT pass through vertex {shared_vertex} (failed geometry check)")
        else:
            # No shared vertex on edges - test each triangle pair individually
            self.logger.info(f"      VP edges don't share a vertex - testing all triangle pairs")
            for tri1 in triangles1:
                for tri2 in triangles2:
                    if tri1 == tri2:
                        continue
                    
                    shared_vertex = self._find_shared_vertex_between_triangles(tri1, tri2)
                    if shared_vertex is not None:
                        self.logger.info(f"      Found shared vertex {shared_vertex} between triangles {tri1}, {tri2}")
                        # Non-adjacent triangles sharing a vertex - Type 2 case
                        # Verify segment actually passes through this vertex
                        if self._segment_passes_through_vertex(vp_idx1, vp_idx2, shared_vertex, tri1, tri2):
                            self.logger.info(f"      ✓ Segment PASSES through vertex {shared_vertex} (Type 2 geometry)")
                            
                            # Check if adjacent or have intermediates
                            shared_edge = self._find_shared_edge_between_triangles(tri1, tri2)
                            
                            if shared_edge is not None:
                                # Adjacent - no intermediates
                                return [tri1, tri2]
                            else:
                                # Non-adjacent at shared vertex
                                # CRITICAL: Only search for intermediates for Type 1 (edge_cutting)!
                                
                                if segment_type == "edge_cutting":
                                    # Type 1: Find intermediate triangles with 0 VPs
                                    self.logger.info(f"      ⚠️  INTERMEDIATE TRIANGLES DETECTED (Type 1 migration)!")
                                    self.logger.info(f"      Triangles {tri1} and {tri2} share vertex {shared_vertex} but NOT an edge")
                                    self.logger.info(f"      Searching for intermediate triangles with 0 VPs...")
                                    
                                    path = self._find_intermediate_triangles_at_vertex(tri1, tri2, shared_vertex)
                                    num_intermediates = len(path) - 2
                                    
                                    self.logger.info(f"      Found {len(path)} total triangles:")
                                    self.logger.info(f"        - Triangle {tri1} (has VP {vp_idx1})")
                                    self.logger.info(f"        - Triangle {tri2} (has VP {vp_idx2})")
                                    if num_intermediates > 0:
                                        self.logger.info(f"        - {num_intermediates} intermediate (0 VPs, 2 crossing points each)")
                                        intermediates = path[1:-1]
                                        self.logger.info(f"      Intermediate triangles: {intermediates}")
                                    self.logger.info(f"      Full path: {path}")
                                    
                                    return path
                                else:
                                    # Type 2 (edge_following): No intermediate search needed
                                    self.logger.info(f"      Type 2 (edge_following): Returning endpoint triangles only")
                                    return [tri1, tri2]
                        else:
                            self.logger.info(f"      ✗ Segment does NOT pass through vertex {shared_vertex} (failed geometry check)")
        
        # Case 2: Completely separate triangles - check all boundary triangles for intersection
        # This is the more expensive general case (may be rare after Case 1a/1b)
        self.logger.info(f"    Checking Case 2: Completely separate triangles")
        crossed = []
        for tri_seg in self.partition.triangle_segments:
            tri_idx = tri_seg.triangle_idx
            
            # Skip triangles containing either VP
            if tri_idx in triangles1 or tri_idx in triangles2:
                continue
            
            # Check if segment intersects this triangle
            if self._segment_intersects_triangle(pos1, pos2, tri_idx):
                crossed.append(tri_idx)
        
        if crossed:
            self.logger.info(f"    Case 2 result: Found {len(crossed)} crossed triangles: {crossed}")
        else:
            self.logger.info(f"    Case 2 result: No crossed triangles found")
        
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
        Compute detailed crossing information for segments (edge_cutting or edge_following).
        
        For each triangle the segment crosses, compute entry/exit points
        and store in partition.segment_crossing_cache.
        
        For edge_following segments (through shared vertex), correctly identifies
        entry/exit as: VP → vertex OR vertex → VP
        """
        vp_idx1 = segment.vp_idx_1
        vp_idx2 = segment.vp_idx_2
        pos1 = self.partition.evaluate_variable_point(vp_idx1)
        pos2 = self.partition.evaluate_variable_point(vp_idx2)
        
        self.logger.info(f"    DEBUG: Computing crossings for segment ({vp_idx1}, {vp_idx2})")
        
        # Get VP edges to check which triangle contains which VP
        vp1_edge = tuple(sorted(self.partition.variable_points[vp_idx1].edge))
        vp2_edge = tuple(sorted(self.partition.variable_points[vp_idx2].edge))
        
        # Check which triangles contain the VPs (for logging clarity)
        triangles_with_vp1 = self.mesh_topology.get_triangles_sharing_edge(vp1_edge)
        triangles_with_vp2 = self.mesh_topology.get_triangles_sharing_edge(vp2_edge)
        
        # Use the already-computed segment type (not detecting from shared vertices!)
        # Type 1 segments can also have VPs whose edges share a vertex (the target vertex)
        is_edge_following = (segment.segment_type == "edge_following")
        
        # For edge-following segments, find the shared vertex
        shared_vertex = None
        if is_edge_following:
            shared_vertex_in_edges = set(vp1_edge) & set(vp2_edge)
            if len(shared_vertex_in_edges) > 0:
                shared_vertex = shared_vertex_in_edges.pop()
                self.logger.info(f"      Edge-following segment through vertex {shared_vertex}")
        
        self.logger.info(f"      Processing {len(segment.crossing_triangles)} triangles...")
        
        for tri_idx in segment.crossing_triangles:
            # Check if this is an intermediate triangle (0 VPs)
            has_vp1 = tri_idx in triangles_with_vp1
            has_vp2 = tri_idx in triangles_with_vp2
            is_intermediate = not has_vp1 and not has_vp2
            
            if is_intermediate:
                self.logger.info(f"      ⚠️  Triangle {tri_idx}: INTERMEDIATE (0 VPs, 2 crossing points)")
            
            # Compute initial crossing points
            entry_point, exit_point, entry_edge, exit_edge = \
                self._compute_triangle_crossing_details(
                    pos1, pos2, tri_idx, vp1_edge, vp2_edge
                )
            
            # For edge-following segments, correct entry/exit to use shared vertex
            if is_edge_following and shared_vertex is not None:
                shared_vertex_pos = self.mesh.vertices[shared_vertex]
                has_vp1 = tri_idx in triangles_with_vp1
                has_vp2 = tri_idx in triangles_with_vp2
                
                if has_vp1 and not has_vp2:
                    # Triangle has VP1: entry = VP1, exit = shared vertex
                    entry_point = pos1
                    exit_point = shared_vertex_pos
                    # Keep entry_edge as vp1_edge, set exit_edge to edge containing shared_vertex
                    face = self.mesh.faces[tri_idx]
                    tri_edges = [
                        tuple(sorted([int(face[0]), int(face[1])])),
                        tuple(sorted([int(face[1]), int(face[2])])),
                        tuple(sorted([int(face[2]), int(face[0])]))
                    ]
                    for edge in tri_edges:
                        if shared_vertex in edge and edge != vp1_edge:
                            exit_edge = edge
                            break
                elif has_vp2 and not has_vp1:
                    # Triangle has VP2: entry = shared vertex, exit = VP2
                    entry_point = shared_vertex_pos
                    exit_point = pos2
                    # Set entry_edge to edge containing shared_vertex, keep exit_edge as vp2_edge
                    face = self.mesh.faces[tri_idx]
                    tri_edges = [
                        tuple(sorted([int(face[0]), int(face[1])])),
                        tuple(sorted([int(face[1]), int(face[2])])),
                        tuple(sorted([int(face[2]), int(face[0])]))
                    ]
                    for edge in tri_edges:
                        if shared_vertex in edge and edge != vp2_edge:
                            entry_edge = edge
                            break
            
            if entry_point is not None and exit_point is not None:
                # Determine which cell this crossing belongs to
                # Use the cells that both VPs separate
                cells1 = self.partition.variable_points[vp_idx1].belongs_to_cells
                cells2 = self.partition.variable_points[vp_idx2].belongs_to_cells
                shared_cells = list(cells1 & cells2)
                cell_idx = shared_cells[0] if shared_cells else 0
                
                # NEW: Check if crossing is at a vertex (edge_following case)
                entry_vertex = self._is_crossing_at_vertex(entry_point, entry_edge)
                exit_vertex = self._is_crossing_at_vertex(exit_point, exit_edge)
                
                # For edge-following: entry OR exit is at shared vertex
                # For edge-cutting: both entry AND exit at same vertex (rare)
                if is_edge_following and shared_vertex is not None:
                    is_vertex_crossing = (entry_vertex == shared_vertex or exit_vertex == shared_vertex)
                else:
                    is_vertex_crossing = (entry_vertex is not None and exit_vertex is not None and 
                                         entry_vertex == exit_vertex)
                
                crossing_info = SegmentCrossingInfo(
                    segment=(min(vp_idx1, vp_idx2), max(vp_idx1, vp_idx2)),
                    triangle_idx=tri_idx,
                    entry_point=entry_point,
                    exit_point=exit_point,
                    entry_edge=entry_edge,
                    exit_edge=exit_edge,
                    cell_idx=cell_idx,
                    entry_vertex=entry_vertex,
                    exit_vertex=exit_vertex,
                    is_vertex_crossing=is_vertex_crossing
                )
                
                # Store in cache
                if tri_idx not in self.partition.segment_crossing_cache:
                    self.partition.segment_crossing_cache[tri_idx] = []
                self.partition.segment_crossing_cache[tri_idx].append(crossing_info)
                
                self.logger.info(f"      ✓ Cached crossing info for triangle {tri_idx}")
                
                # IMPROVED LOGGING: Distinguish VP positions from computed crossings
                # Special highlighting for intermediate triangles (0 VPs)
                if is_intermediate:
                    self.logger.info(f"        ├─ INTERMEDIATE TRIANGLE {tri_idx} (0 VPs):")
                    self.logger.info(f"        ├─ Entry edge: {entry_edge}, point: {entry_point}")
                    self.logger.info(f"        └─ Exit edge:  {exit_edge}, point: {exit_point}")
                    if is_vertex_crossing:
                        self.logger.info(f"           Vertex crossing at: {entry_vertex if entry_vertex else exit_vertex}")
                elif has_vp1 and has_vp2:
                    self.logger.debug(f"    Triangle {tri_idx}: Both VPs in triangle (normal segment)")
                elif has_vp1:
                    self.logger.debug(f"    Triangle {tri_idx}: VP {vp_idx1} position used, "
                                    f"exit computed on edge {exit_edge}")
                elif has_vp2:
                    self.logger.debug(f"    Triangle {tri_idx}: VP {vp_idx2} position used, "
                                    f"entry computed on edge {entry_edge}")
                else:
                    if is_vertex_crossing:
                        self.logger.debug(f"    Triangle {tri_idx}: Vertex crossing at vertex {entry_vertex}")
                    else:
                        self.logger.debug(f"    Triangle {tri_idx}: Computed crossing "
                                        f"entry={entry_edge}, exit={exit_edge}")
    
    def _compute_triangle_normal(self, tri_idx: int) -> np.ndarray:
        """
        Compute unit normal vector for a triangle.
        
        Args:
            tri_idx: Triangle index
            
        Returns:
            Unit normal vector (3D)
        """
        face = self.mesh.faces[tri_idx]
        v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
        p1 = self.mesh.vertices[v1]
        p2 = self.mesh.vertices[v2]
        p3 = self.mesh.vertices[v3]
        
        # Cross product of two edges
        n = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(n)
        
        if norm < 1e-12:
            # Degenerate triangle, return arbitrary normal
            return np.array([0.0, 0.0, 1.0])
        
        return n / norm
    
    def _project_point_onto_triangle_plane(self, point: np.ndarray, tri_idx: int) -> np.ndarray:
        """
        Project a 3D point onto the plane containing triangle tri_idx.
        
        Args:
            point: 3D point to project
            tri_idx: Triangle index
            
        Returns:
            Projected point (3D, lies in triangle's plane)
        """
        # Get triangle vertices
        face = self.mesh.faces[tri_idx]
        v1 = int(face[0])
        p_ref = self.mesh.vertices[v1]  # Reference point on plane
        
        # Get plane normal
        normal = self._compute_triangle_normal(tri_idx)
        
        # Project: point - ((point - p_ref) · n) * n
        offset = point - p_ref
        distance_to_plane = np.dot(offset, normal)
        projected = point - distance_to_plane * normal
        
        return projected
    
    def _find_shared_edge_between_triangles(self, tri_idx1: int, tri_idx2: int) -> Optional[Tuple[int, int]]:
        """
        Find the shared edge between two triangles, if it exists.
        
        Args:
            tri_idx1: First triangle index
            tri_idx2: Second triangle index
            
        Returns:
            Shared edge tuple (normalized) or None
        """
        face1 = self.mesh.faces[tri_idx1]
        face2 = self.mesh.faces[tri_idx2]
        
        v1_1, v1_2, v1_3 = int(face1[0]), int(face1[1]), int(face1[2])
        v2_1, v2_2, v2_3 = int(face2[0]), int(face2[1]), int(face2[2])
        
        edges1 = [
            tuple(sorted([v1_1, v1_2])),
            tuple(sorted([v1_2, v1_3])),
            tuple(sorted([v1_3, v1_1]))
        ]
        
        edges2 = [
            tuple(sorted([v2_1, v2_2])),
            tuple(sorted([v2_2, v2_3])),
            tuple(sorted([v2_3, v2_1]))
        ]
        
        # Find intersection
        for e1 in edges1:
            if e1 in edges2:
                return e1
        
        return None
    
    def _find_shared_vertex_between_triangles(self, tri_idx1: int, tri_idx2: int) -> Optional[int]:
        """
        Find vertex shared by two triangles (if any).
        
        Two triangles can share:
        - 0 vertices (disjoint)
        - 1 vertex (touch at corner) ← This is what we're looking for (Type 2 case)
        - 2 vertices (share edge) ← This is Type 1 case, handled separately
        - 3 vertices (same triangle) ← Should never happen
        
        Args:
            tri_idx1: First triangle index
            tri_idx2: Second triangle index
            
        Returns:
            Vertex index if triangles share exactly one vertex, None otherwise
        """
        face1 = self.mesh.faces[tri_idx1]
        face2 = self.mesh.faces[tri_idx2]
        
        verts1 = set([int(face1[0]), int(face1[1]), int(face1[2])])
        verts2 = set([int(face2[0]), int(face2[1]), int(face2[2])])
        
        shared_verts = verts1 & verts2
        
        if len(shared_verts) == 1:
            # Exactly one shared vertex - Type 2 geometry
            return list(shared_verts)[0]
        
        # 0 shared (disjoint), 2+ shared (edge), or same triangle
        return None
    
    def _find_intermediate_triangles_at_vertex(self, tri_start: int, tri_end: int, 
                                                shared_vertex: int) -> List[int]:
        """
        Find all intermediate triangles between tri_start and tri_end at shared_vertex.
        
        When two triangles share a vertex but NOT an edge, there may be intermediate
        triangles between them (forming a "fan" around the shared vertex).
        
        Algorithm:
        1. Get all triangles at shared_vertex
        2. Build adjacency graph (triangles sharing edges at this vertex)
        3. Find shortest path from tri_start to tri_end
        4. Return triangles in order: [tri_start, tri_mid1, tri_mid2, ..., tri_end]
        
        Args:
            tri_start: Starting triangle index
            tri_end: Ending triangle index
            shared_vertex: Vertex shared by all triangles in the path
            
        Returns:
            List of triangle indices in topological order (includes start and end)
        """
        # Get all triangles at this vertex
        all_triangles_at_vertex = self.mesh_topology.get_triangles_at_vertex(shared_vertex)
        
        if tri_start not in all_triangles_at_vertex or tri_end not in all_triangles_at_vertex:
            self.logger.warning(f"  Triangles {tri_start} or {tri_end} not at vertex {shared_vertex}")
            return [tri_start, tri_end]
        
        # Build adjacency graph: triangles that share an edge at this vertex
        from collections import defaultdict, deque
        adjacency = defaultdict(list)
        
        for i, tri1 in enumerate(all_triangles_at_vertex):
            for tri2 in all_triangles_at_vertex[i+1:]:
                shared_edge = self._find_shared_edge_between_triangles(tri1, tri2)
                if shared_edge is not None and shared_vertex in shared_edge:
                    # These triangles are adjacent at the shared vertex
                    adjacency[tri1].append(tri2)
                    adjacency[tri2].append(tri1)
        
        # BFS to find shortest path from tri_start to tri_end
        queue = deque([(tri_start, [tri_start])])
        visited = {tri_start}
        
        while queue:
            current_tri, path = queue.popleft()
            
            if current_tri == tri_end:
                # Found path!
                return path
            
            for neighbor_tri in adjacency[current_tri]:
                if neighbor_tri not in visited:
                    visited.add(neighbor_tri)
                    queue.append((neighbor_tri, path + [neighbor_tri]))
        
        # No path found - triangles are not connected at this vertex
        # This shouldn't happen if geometry is manifold
        self.logger.warning(f"  No path found from triangle {tri_start} to {tri_end} at vertex {shared_vertex}")
        return [tri_start, tri_end]
    
    def _is_crossing_at_vertex(self, crossing_point: np.ndarray, 
                                edge: Tuple[int, int], tolerance_factor: float = 1e-6) -> Optional[int]:
        """
        Check if crossing point is essentially at a vertex of the edge.
        
        Args:
            crossing_point: 3D position of crossing point
            edge: Edge tuple (v1_idx, v2_idx)
            tolerance_factor: Multiplied by edge length to get tolerance
            
        Returns:
            Vertex index if at vertex, None otherwise
        """
        v1_pos = self.mesh.vertices[edge[0]]
        v2_pos = self.mesh.vertices[edge[1]]
        edge_length = np.linalg.norm(v2_pos - v1_pos)
        tolerance = tolerance_factor * edge_length
        
        dist_to_v1 = np.linalg.norm(crossing_point - v1_pos)
        dist_to_v2 = np.linalg.norm(crossing_point - v2_pos)
        
        if dist_to_v1 < tolerance:
            return edge[0]
        elif dist_to_v2 < tolerance:
            return edge[1]
        else:
            return None
    
    def _segment_passes_through_vertex(self, vp_idx1: int, vp_idx2: int, 
                                        vertex: int, tri1: int, tri2: int) -> bool:
        """
        Verify segment (vp1, vp2) passes through shared vertex using geometric check.
        
        For Type 2 segments (edge_following), the segment appears to pass through
        a shared vertex when viewed on the curved surface. This method projects
        the segment onto each triangle's plane and checks if both projections
        pass through (or very close to) the shared vertex.
        
        Args:
            vp_idx1, vp_idx2: Segment endpoints (VP indices)
            vertex: Shared vertex to check
            tri1: First triangle (contains vp1's edge)
            tri2: Second triangle (contains vp2's edge)
            
        Returns:
            True if segment passes through vertex (within tolerance)
        """
        pos1 = self.partition.evaluate_variable_point(vp_idx1)
        pos2 = self.partition.evaluate_variable_point(vp_idx2)
        vertex_pos = self.mesh.vertices[vertex]
        
        self.logger.info(f"        Testing if segment passes through vertex {vertex}")
        self.logger.info(f"          VP {vp_idx1} pos: {pos1}")
        self.logger.info(f"          VP {vp_idx2} pos: {pos2}")
        self.logger.info(f"          Vertex {vertex} pos: {vertex_pos}")
        
        # Project segment onto tri1's plane
        pos2_proj_tri1 = self._project_point_onto_triangle_plane(pos2, tri1)
        
        # Project segment onto tri2's plane  
        pos1_proj_tri2 = self._project_point_onto_triangle_plane(pos1, tri2)
        
        # Check if projected segments pass close to the vertex
        # For tri1: check if line (pos1, pos2_proj_tri1) is close to vertex
        dist1 = self._point_to_line_distance(vertex_pos, pos1, pos2_proj_tri1)
        
        # For tri2: check if line (pos1_proj_tri2, pos2) is close to vertex
        dist2 = self._point_to_line_distance(vertex_pos, pos1_proj_tri2, pos2)
        
        # Tolerance based on segment length
        segment_length = np.linalg.norm(pos2 - pos1)
        tol = 1e-4 * segment_length  # Relaxed for curved meshes with dihedral angles
        
        self.logger.info(f"          Distance from vertex to projected line (tri1): {dist1:.6e}")
        self.logger.info(f"          Distance from vertex to projected line (tri2): {dist2:.6e}")
        self.logger.info(f"          Tolerance (1e-4 * segment_length): {tol:.6e}")
        self.logger.info(f"          Segment length: {segment_length:.6e}")
        
        passes_through = (dist1 < tol and dist2 < tol)
        
        if passes_through:
            self.logger.info(f"        ✓ PASS: Both distances < tolerance")
        else:
            self.logger.info(f"        ✗ FAIL: dist1 < tol? {dist1 < tol}, dist2 < tol? {dist2 < tol}")
        
        return passes_through
    
    def _point_to_line_distance(self, point: np.ndarray, line_start: np.ndarray, 
                                 line_end: np.ndarray) -> float:
        """
        Compute minimum distance from point to line segment in 3D.
        
        Args:
            point: 3D point position
            line_start: Start of line segment
            line_end: End of line segment
            
        Returns:
            Minimum distance from point to line segment
        """
        # Vector from line_start to line_end
        line_vec = line_end - line_start
        line_length = np.linalg.norm(line_vec)
        
        if line_length < 1e-10:
            # Degenerate line (points are same)
            return np.linalg.norm(point - line_start)
        
        # Normalized line direction
        line_dir = line_vec / line_length
        
        # Vector from line_start to point
        start_to_point = point - line_start
        
        # Project onto line direction
        t = np.dot(start_to_point, line_dir)
        
        # Clamp to line segment [0, line_length]
        t = max(0, min(line_length, t))
        
        # Closest point on line segment
        closest_point = line_start + t * line_dir
        
        # Distance from point to closest point
        return np.linalg.norm(point - closest_point)
    
    def _compute_crossing_via_dual_projection(self, pos1: np.ndarray, pos2: np.ndarray,
                                               tri_idx1: int, tri_idx2: int,
                                               shared_edge: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Compute crossing point on shared edge using dual projection method.
        
        This handles the case where two adjacent triangles form a dihedral angle,
        so the 3D segment pos1→pos2 doesn't geometrically intersect triangle faces,
        but we need to find where the boundary "appears" to cross the shared edge
        for area calculation purposes.
        
        Method:
        1. Project pos2 onto tri_idx1's plane, find intersection with shared_edge
        2. Project pos1 onto tri_idx2's plane, find intersection with shared_edge
        3. Verify both projections agree (should give same point on edge)
        
        Args:
            pos1: VP position in first triangle (3D)
            pos2: VP position in second triangle (3D)
            tri_idx1: First triangle index
            tri_idx2: Second triangle index
            shared_edge: Shared edge between triangles
            
        Returns:
            Crossing point on shared edge (3D) or None if method fails
        """
        edge_start = self.mesh.vertices[shared_edge[0]]
        edge_end = self.mesh.vertices[shared_edge[1]]
        
        # Method 1: Project from tri_idx1's perspective
        pos2_proj = self._project_point_onto_triangle_plane(pos2, tri_idx1)
        crossing_1 = self._compute_line_edge_intersection(pos1, pos2_proj, edge_start, edge_end)
        
        # Method 2: Project from tri_idx2's perspective
        pos1_proj = self._project_point_onto_triangle_plane(pos1, tri_idx2)
        crossing_2 = self._compute_line_edge_intersection(pos1_proj, pos2, edge_start, edge_end)
        
        # Check if both methods succeeded
        if crossing_1 is None or crossing_2 is None:
            return None
        
        # Verify projections agree
        distance = np.linalg.norm(crossing_1 - crossing_2)
        edge_length = np.linalg.norm(edge_end - edge_start)
        tolerance = 1e-6 * edge_length
        
        if distance > tolerance:
            self.logger.warning(
                f"Dual projections disagree: distance={distance:.2e}, "
                f"edge_length={edge_length:.2e}, tolerance={tolerance:.2e}"
            )
            # Still return average, but log warning
        
        # Return average for numerical stability
        return 0.5 * (crossing_1 + crossing_2)
    
    def _compute_triangle_crossing_details(self, pos1: np.ndarray, pos2: np.ndarray,
                                           tri_idx: int,
                                           vp1_edge: Optional[Tuple[int, int]] = None,
                                           vp2_edge: Optional[Tuple[int, int]] = None) -> Tuple[
                                               Optional[np.ndarray], 
                                               Optional[np.ndarray],
                                               Optional[Tuple[int, int]],
                                               Optional[Tuple[int, int]]]:
        """
        Compute where segment (pos1, pos2) enters and exits a triangle.
        
        Uses dual projection method for adjacent triangles with dihedral angles.
        Falls back to direct 3D intersection for non-adjacent cases.
        
        If vp1_edge or vp2_edge is provided and matches a triangle edge, uses VP position
        directly instead of computing intersection.
        
        Args:
            pos1: Position of VP1 (3D)
            pos2: Position of VP2 (3D)
            tri_idx: Triangle index
            vp1_edge: Edge that VP1 sits on (normalized), or None
            vp2_edge: Edge that VP2 sits on (normalized), or None
        
        Returns:
            (entry_point, exit_point, entry_edge, exit_edge) or (None, None, None, None)
        """
        face = self.mesh.faces[tri_idx]
        v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
        
        # Get triangle edges (normalized)
        tri_edges = [
            tuple(sorted([v1, v2])),
            tuple(sorted([v2, v3])),
            tuple(sorted([v3, v1]))
        ]
        
        # Check which VPs (if any) are in this triangle
        vp1_in_triangle = vp1_edge in tri_edges if vp1_edge is not None else False
        vp2_in_triangle = vp2_edge in tri_edges if vp2_edge is not None else False
        
        # Case 1: Both VPs in triangle (shouldn't happen for edge_cutting, but handle it)
        if vp1_in_triangle and vp2_in_triangle:
            return pos1, pos2, vp1_edge, vp2_edge
        
        # Case 2: VP1 in triangle, VP2 not
        if vp1_in_triangle and not vp2_in_triangle:
            # Entry is VP1's position on its edge
            # Exit is crossing with other edge (not VP1's edge)
            for edge_tuple in tri_edges:
                if edge_tuple == vp1_edge:
                    continue  # Skip VP1's edge
                edge_start = self.mesh.vertices[edge_tuple[0]]
                edge_end = self.mesh.vertices[edge_tuple[1]]
                exit_point = self._compute_line_edge_intersection(pos1, pos2, edge_start, edge_end)
                if exit_point is not None:
                    # Determine order based on direction from pos1 to pos2
                    segment_dir = pos2 - pos1
                    t = np.dot(exit_point - pos1, segment_dir) / np.linalg.norm(segment_dir)**2
                    if t > 0:  # Exit is in forward direction
                        return pos1, exit_point, vp1_edge, edge_tuple
                    else:  # Exit is backward (shouldn't happen, but be safe)
                        return exit_point, pos1, edge_tuple, vp1_edge
            # No exit found - try dual projection (fallback)
        
        # Case 3: VP2 in triangle, VP1 not
        if vp2_in_triangle and not vp1_in_triangle:
            # Entry is crossing with other edge (not VP2's edge)
            # Exit is VP2's position on its edge
            for edge_tuple in tri_edges:
                if edge_tuple == vp2_edge:
                    continue  # Skip VP2's edge
                edge_start = self.mesh.vertices[edge_tuple[0]]
                edge_end = self.mesh.vertices[edge_tuple[1]]
                entry_point = self._compute_line_edge_intersection(pos1, pos2, edge_start, edge_end)
                if entry_point is not None:
                    # Determine order based on direction from pos1 to pos2
                    segment_dir = pos2 - pos1
                    t = np.dot(entry_point - pos1, segment_dir) / np.linalg.norm(segment_dir)**2
                    if t < 1:  # Entry is before pos2
                        return entry_point, pos2, edge_tuple, vp2_edge
                    else:  # Entry is after pos2 (shouldn't happen)
                        return pos2, entry_point, vp2_edge, edge_tuple
            # No entry found - try dual projection (fallback)
        
        # Case 4: Neither VP in triangle - use standard intersection or dual projection
        edges = [
            ((v1, v2), self.mesh.vertices[v1], self.mesh.vertices[v2]),
            ((v2, v3), self.mesh.vertices[v2], self.mesh.vertices[v3]),
            ((v3, v1), self.mesh.vertices[v3], self.mesh.vertices[v1])
        ]
        
        intersections = []
        
        # Try to find intersections with each edge
        for edge, edge_start, edge_end in edges:
            intersection = self._compute_line_edge_intersection(pos1, pos2, edge_start, edge_end)
            if intersection is not None:
                intersections.append((edge, intersection))
        
        # If standard 3D intersection found 2 edges, use it
        if len(intersections) == 2:
            (edge_a, point_a), (edge_b, point_b) = intersections
            
            # Determine which is entry and which is exit
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
        
        # Standard method failed - try dual projection for adjacent triangles
        # This handles dihedral angle cases where 3D line misses the face
        
        # We need to identify which triangles contain pos1 and pos2
        # This is expensive, so only do it when standard method fails
        
        # Get all boundary triangles (heuristic: triangles with at least 1 VP)
        triangles_with_vps = set()
        for vp in self.partition.variable_points:
            edge_triangles = self.mesh_topology.get_triangles_sharing_edge(tuple(sorted(vp.edge)))
            triangles_with_vps.update(edge_triangles)
        
        # Find which triangles are adjacent to tri_idx
        for candidate_tri in triangles_with_vps:
            if candidate_tri == tri_idx:
                continue
            
            shared_edge = self._find_shared_edge_between_triangles(tri_idx, candidate_tri)
            if shared_edge is None:
                continue
            
            # Try dual projection between tri_idx and candidate_tri
            # Determine which VP is in which triangle (heuristic based on distance)
            face_candidate = self.mesh.faces[candidate_tri]
            centroid_tri = np.mean([self.mesh.vertices[v] for v in face], axis=0)
            centroid_candidate = np.mean([self.mesh.vertices[v] for v in face_candidate], axis=0)
            
            dist1_to_tri = np.linalg.norm(pos1 - centroid_tri)
            dist1_to_candidate = np.linalg.norm(pos1 - centroid_candidate)
            
            if dist1_to_tri < dist1_to_candidate:
                # pos1 in tri_idx, pos2 in candidate_tri
                crossing = self._compute_crossing_via_dual_projection(
                    pos1, pos2, tri_idx, candidate_tri, shared_edge
                )
            else:
                # pos2 in tri_idx, pos1 in candidate_tri
                crossing = self._compute_crossing_via_dual_projection(
                    pos2, pos1, candidate_tri, tri_idx, shared_edge
                )
            
            if crossing is not None:
                # Found crossing on this shared edge
                # For entry/exit, we need to determine direction
                # The crossing point is both entry and exit for this 1-VP triangle
                return crossing, crossing, shared_edge, shared_edge
        
        # All methods failed
        return None, None, None, None
    
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

