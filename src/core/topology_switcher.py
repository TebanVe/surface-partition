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
        
        NOTE: This does NOT update vp.belongs_to_cells. For Type 2 migrations,
        belongs_to_cells is updated in a batch AFTER the cell flip (Step 11b)
        to ensure it uses the final indicator_functions state.
        
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
        # NOTE: belongs_to_cells NOT updated here - see docstring
    
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
                # Determine BOTH cells this crossing separates
                # The segment separates exactly 2 cells (the intersection of both VP's belongs_to_cells)
                cells1 = self.partition.variable_points[vp_idx1].belongs_to_cells
                cells2 = self.partition.variable_points[vp_idx2].belongs_to_cells
                shared_cells = list(cells1 & cells2)
                
                # Store both cells for proper area attribution
                if len(shared_cells) >= 2:
                    cell_pair = tuple(sorted(shared_cells[:2]))
                elif len(shared_cells) == 1:
                    cell_pair = (shared_cells[0], shared_cells[0])
                else:
                    cell_pair = (0, 0)
                
                cell_idx = shared_cells[0] if shared_cells else 0  # Legacy field
                
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
                    cell_pair=cell_pair,
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
    
    # =========================================================================
    # Stage 0: Component Analysis Infrastructure (Vertex-Collapse Strategy)
    # =========================================================================
    
    def get_non_triple_point_boundary_vps(self, boundary_tol: float = 0.1) -> List[int]:
        """
        Get boundary VPs that are NOT part of triple point triangles.
        
        CRITICAL: Type 1 migration should only consider VPs that are NOT in triple points.
        Triple points are handled separately via Type 2 migration.
        
        Args:
            boundary_tol: Threshold for boundary detection
            
        Returns:
            List of boundary VP indices (excluding triple point VPs)
        """
        from .steiner_handler import SteinerHandler
        
        # Get all boundary VPs
        all_boundary_vps = self.partition.get_boundary_variable_points(tol=boundary_tol)
        
        # Get VPs that are part of triple points
        steiner_handler = SteinerHandler(self.mesh, self.partition)
        triple_point_vps = set()
        for tp in steiner_handler.triple_points:
            triple_point_vps.update(tp.var_point_indices)
        
        # Filter out triple point VPs
        non_triple_boundary_vps = [vp for vp in all_boundary_vps if vp not in triple_point_vps]
        
        self.logger.debug(f"Boundary VPs: {len(all_boundary_vps)} total, "
                         f"{len(triple_point_vps)} in triple points, "
                         f"{len(non_triple_boundary_vps)} available for Type 1")
        
        return non_triple_boundary_vps
    
    def compute_boundary_distance(self, vp_idx: int) -> float:
        """
        Compute distance from VP to its target vertex: min(λ, 1-λ)
        
        Args:
            vp_idx: Variable point index
            
        Returns:
            Distance in [0, 0.5], where smaller = closer to target vertex
        """
        vp = self.partition.variable_points[vp_idx]
        return min(vp.lambda_param, 1.0 - vp.lambda_param)
    
    def find_connected_components(self, boundary_vps_set: set) -> List[set]:
        """
        Find connected components of boundary VPs via DFS on boundary_segments.
        
        Args:
            boundary_vps_set: Set of boundary VP indices
            
        Returns:
            List of sets, each set is a connected component
        """
        from collections import defaultdict
        
        # Build adjacency from boundary_segments (only for boundary VPs)
        adjacency = defaultdict(set)
        for segment in self.partition.boundary_segments:
            vp1, vp2 = segment.vp_idx_1, segment.vp_idx_2
            if vp1 in boundary_vps_set and vp2 in boundary_vps_set:
                adjacency[vp1].add(vp2)
                adjacency[vp2].add(vp1)
        
        # DFS to find connected components
        visited = set()
        components = []
        
        for vp_idx in boundary_vps_set:
            if vp_idx in visited:
                continue
            
            component = set()
            stack = [vp_idx]
            
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                
                visited.add(current)
                component.add(current)
                
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        stack.append(neighbor)
            
            if component:
                components.append(component)
        
        return components
    
    def _get_triple_point_vps(self) -> Set[int]:
        """
        Get all VPs that are part of triple point triangles.
        
        Returns:
            Set of VP indices that belong to triple point triangles
        """
        from .steiner_handler import SteinerHandler
        
        steiner_handler = SteinerHandler(self.mesh, self.partition)
        triple_point_vps = set()
        for tp in steiner_handler.triple_points:
            triple_point_vps.update(tp.var_point_indices)
        
        return triple_point_vps
    
    def _component_near_triple_point(self, component: Dict) -> Tuple[bool, List[int]]:
        """
        Check if a component is too close to a triple point triangle.
        
        A component is "too close" if:
        - It shares a non-boundary VP with a triple point triangle
        - AND the component has < 3 VPs (risky migration)
        
        For 3-VP components, this is safe because both neighbors are internal,
        so they won't affect triple point VPs.
        
        Args:
            component: Component info dict from analyze_component()
            
        Returns:
            (is_near: bool, shared_vps: List[int])
            - is_near: True if component is too close to triple point (and has < 3 VPs)
            - shared_vps: List of non-boundary VPs that connect component to triple points
        """
        # 3-VP components are safe (internal neighbors)
        if component['size'] >= 3:
            return False, []
        
        # Get triple point VPs
        triple_point_vps = self._get_triple_point_vps()
        
        # Build adjacency from boundary_segments to find connections
        from collections import defaultdict
        adjacency = defaultdict(set)
        for segment in self.partition.boundary_segments:
            vp1, vp2 = segment.vp_idx_1, segment.vp_idx_2
            adjacency[vp1].add(vp2)
            adjacency[vp2].add(vp1)
        
        # Check if any non-boundary neighbor of this component is connected to a triple point VP
        shared_vps = []
        for non_boundary_vp in component['non_boundary_neighbors']:
            # Check if this non-boundary VP is connected to any triple point VP
            neighbors_of_non_boundary = adjacency.get(non_boundary_vp, set())
            if neighbors_of_non_boundary & triple_point_vps:
                # This non-boundary VP connects the component to a triple point
                shared_vps.append(non_boundary_vp)
        
        is_near = len(shared_vps) > 0
        return is_near, shared_vps
    
    def analyze_component(self, component_vps: Set[int]) -> Dict:
        """
        Analyze a component and extract metadata.
        
        Args:
            component_vps: Set of VP indices in the component
            
        Returns:
            {
                'vp_indices': List[int],
                'size': int,
                'target_vertex': int,  # Common vertex all VPs converge to
                'min_distance': float,  # Closest VP distance to target
                'non_boundary_neighbors': List[int],  # External non-boundary VPs
                'boundary_neighbors': List[int],  # External boundary VPs
                'centroid': np.ndarray  # Geometric center
            }
        """
        from collections import defaultdict
        
        # Build adjacency from boundary_segments
        adjacency = defaultdict(set)
        for segment in self.partition.boundary_segments:
            vp1, vp2 = segment.vp_idx_1, segment.vp_idx_2
            adjacency[vp1].add(vp2)
            adjacency[vp2].add(vp1)
        
        # Get all neighbors of this component
        all_neighbors = set()
        for vp_idx in component_vps:
            all_neighbors.update(adjacency.get(vp_idx, set()))
        
        # External neighbors (not in this component)
        external_neighbors = all_neighbors - component_vps
        
        # Get boundary VPs set for classification
        boundary_vps_set = set(self.partition.get_boundary_variable_points(tol=0.1))
        
        # Separate boundary and non-boundary external neighbors
        boundary_neighbors = external_neighbors & boundary_vps_set
        non_boundary_neighbors = external_neighbors - boundary_vps_set
        
        # Find target vertex using distance-based logic (same as _identify_target_vertex)
        # For each VP, identify which vertex it's approaching based on lambda
        target_vertices = []
        for vp_idx in component_vps:
            vp = self.partition.variable_points[vp_idx]
            tv = self._identify_target_vertex(vp)
            if tv is not None:
                target_vertices.append(tv)
            else:
                self.logger.warning(f"Could not identify target vertex for VP {vp_idx} in component")
        
        # Verify all VPs approach the same vertex
        target_vertex = None
        if target_vertices:
            target_vertex = target_vertices[0]
            if not all(tv == target_vertex for tv in target_vertices):
                # Warning: VPs don't converge to same vertex
                unique_targets = set(target_vertices)
                self.logger.warning(
                    f"Component VPs approach different vertices: {unique_targets}. "
                    f"Using most common vertex."
                )
                # Use most common target vertex
                from collections import Counter
                target_vertex = Counter(target_vertices).most_common(1)[0][0]
        
        # Compute min distance (closest VP to target vertex)
        min_distance = float('inf')
        for vp_idx in component_vps:
            dist = self.compute_boundary_distance(vp_idx)
            if dist < min_distance:
                min_distance = dist
        
        # Compute centroid
        positions = []
        for vp_idx in component_vps:
            vp = self.partition.variable_points[vp_idx]
            positions.append(vp.evaluate(self.mesh.vertices))
        centroid = np.mean(positions, axis=0) if positions else np.array([0.0, 0.0, 0.0])
        
        # Check proximity to triple points
        component_dict = {
            'vp_indices': list(component_vps),
            'size': len(component_vps),
            'target_vertex': target_vertex,
            'min_distance': min_distance,
            'non_boundary_neighbors': list(non_boundary_neighbors),
            'boundary_neighbors': list(boundary_neighbors),
            'centroid': centroid
        }
        
        # Check if component is too close to triple point (only risky if < 3 VPs)
        is_near, shared_vps = self._component_near_triple_point(component_dict)
        component_dict['near_triple_point'] = is_near
        component_dict['triple_point_shared_vps'] = shared_vps
        
        return component_dict
    
    def detect_proximity_conflicts(self, components: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Detect conflicts between components (shared non-boundary neighbors).
        
        A conflict exists when:
        - Components share a non-boundary neighbor VP (topological connection)
        - Both components are near convergence (min_dist < 0.01)
        
        Note: Conflict detection does NOT determine deferral. Deferral requires
        additional condition: at least one component has < 3 VPs (risky).
        
        IMPORTANT: Each conflict is between exactly 2 components (one shared VP).
        However, chains can form: Component A shares VP_ab with B, B shares VP_bc with C.
        This creates a chain: A - B - C, where B has multiple neighbors.
        
        Returns:
            (conflicts: List[Dict], chain_warnings: List[Dict])
        """
        conflicts = []
        
        # Detect pairwise conflicts
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                comp_i = components[i]
                comp_j = components[j]
                
                shared_non_boundary = set(comp_i['non_boundary_neighbors']) & set(comp_j['non_boundary_neighbors'])
                
                if shared_non_boundary:
                    # Calculate minimum distances in each component
                    min_dist_i = comp_i['min_distance']
                    min_dist_j = comp_j['min_distance']
                    
                    # Determine if both components are near convergence
                    proximity_threshold = 0.01
                    both_near = min_dist_i < proximity_threshold and min_dist_j < proximity_threshold
                    
                    conflicts.append({
                        'component_i': i,
                        'component_j': j,
                        'size_i': comp_i['size'],
                        'size_j': comp_j['size'],
                        'shared_vps': list(shared_non_boundary),
                        'min_dist_i': min_dist_i,
                        'min_dist_j': min_dist_j,
                        'both_near_convergence': both_near,
                    })
        
        # Detect chains: components with multiple neighbors
        from collections import defaultdict, deque
        
        component_neighbors = defaultdict(set)
        for conflict in conflicts:
            i, j = conflict['component_i'], conflict['component_j']
            component_neighbors[i].add(j)
            component_neighbors[j].add(i)
        
        chain_warnings = []
        for comp_idx, neighbors in component_neighbors.items():
            if len(neighbors) >= 2:
                # This component has 2+ neighbors → part of a chain
                neighbor_list = list(neighbors)
                neighbor_sizes = [components[n]['size'] for n in neighbor_list]
                
                # Find all components in the chain (connected components)
                chain_components = self._find_chain_components(comp_idx, component_neighbors)
                
                chain_warnings.append({
                    'component_index': comp_idx,
                    'neighbor_indices': neighbor_list,
                    'component_size': components[comp_idx]['size'],
                    'neighbor_sizes': neighbor_sizes,
                    'chain_components': chain_components,
                    'chain_length': len(chain_components),
                    'warning': f"CHAIN: Component {comp_idx} has {len(neighbors)} neighbors "
                              f"(indices: {neighbor_list}). Chain length: {len(chain_components)}"
                })
                self.logger.warning(
                    f"⚠️  COMPONENT CHAIN DETECTED: Component {comp_idx} (size={components[comp_idx]['size']}) "
                    f"has {len(neighbors)} neighbors: {neighbor_list} (sizes: {neighbor_sizes}). "
                    f"Total chain length: {len(chain_components)} components."
                )
        
        return (conflicts, chain_warnings)
    
    def _find_chain_components(self, start_idx: int, component_neighbors: Dict[int, Set[int]]) -> Set[int]:
        """
        Find all components in the chain starting from start_idx.
        
        Uses BFS to find all connected components.
        """
        from collections import deque
        
        chain = set()
        queue = deque([start_idx])
        visited = set()
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            chain.add(current)
            
            for neighbor in component_neighbors.get(current, set()):
                if neighbor not in visited:
                    queue.append(neighbor)
        
        return chain
    
    # =========================================================================
    # Stage 1: Core Migration Function (Vertex-Collapse Strategy)
    # =========================================================================
    
    def _find_opposite_edge(self, current_edge: Tuple[int, int], 
                            target_vertex: int) -> Optional[Tuple[int, int]]:
        """
        Find the "opposite edge" - the edge in an adjacent triangle that continues
        the boundary path through the target vertex.
        
        On a torus (curved surface), edges are NOT perfectly collinear. They belong
        to different triangles and have a small angle between them.
        
        Strategy:
        1. Find all edges at target_vertex
        2. For each candidate edge, compute angle with current edge at target_vertex
        3. Find edge with angle closest to 180° (π radians) - almost collinear continuation
        4. Must be in empty triangle (no boundary segments) and free (no VP)
        
        Args:
            current_edge: Current edge (v_a, target_vertex) or (target_vertex, v_a)
            target_vertex: The vertex the VP is approaching
            
        Returns:
            Opposite edge tuple (normalized), or None if not found
        """
        import numpy as np
        
        # Get the other endpoint of current edge
        if current_edge[0] == target_vertex:
            other_endpoint = current_edge[1]
        elif current_edge[1] == target_vertex:
            other_endpoint = current_edge[0]
        else:
            self.logger.warning(f"Target vertex {target_vertex} not in current edge {current_edge}")
            return None
        
        # Get direction vector FROM target_vertex TO other_endpoint (for current edge)
        v_target = self.mesh.vertices[target_vertex]
        v_other = self.mesh.vertices[other_endpoint]
        current_dir = v_other - v_target
        current_dir_norm = np.linalg.norm(current_dir)
        if current_dir_norm < 1e-10:
            self.logger.warning(f"Current edge has zero length")
            return None
        current_dir = current_dir / current_dir_norm
        
        # Get all edges at target_vertex
        all_edges_at_vertex = self.mesh_topology.get_edges_at_vertex(target_vertex)
        
        best_edge = None
        min_deviation_from_180 = float('inf')  # Deviation from 180° (π)
        
        for candidate_edge in all_edges_at_vertex:
            # Skip current edge
            if tuple(sorted(candidate_edge)) == tuple(sorted(current_edge)):
                continue
            
            # Get the other endpoint of candidate edge
            if candidate_edge[0] == target_vertex:
                candidate_endpoint = candidate_edge[1]
            else:
                candidate_endpoint = candidate_edge[0]
            
            # Get direction vector FROM target_vertex TO candidate_endpoint
            v_candidate = self.mesh.vertices[candidate_endpoint]
            candidate_dir = v_candidate - v_target
            candidate_dir_norm = np.linalg.norm(candidate_dir)
            if candidate_dir_norm < 1e-10:
                continue
            candidate_dir = candidate_dir / candidate_dir_norm
            
            # Compute angle between the two edges at target_vertex
            dot_product = np.dot(current_dir, candidate_dir)
            dot_product = np.clip(dot_product, -1.0, 1.0)
            angle = np.arccos(dot_product)
            
            # For almost collinear edges, angle ≈ 180° (π)
            deviation_from_180 = abs(np.pi - angle)
            
            # Check if candidate edge is in empty triangle and free
            candidate_triangles = self.mesh_topology.get_triangles_sharing_edge(candidate_edge)
            is_valid = False
            for tri in candidate_triangles:
                if not self._triangle_has_boundary_segment(tri):
                    edge_norm = tuple(sorted(candidate_edge))
                    if edge_norm not in self.partition.edge_to_varpoint:
                        is_valid = True
                        break
            
            if is_valid and deviation_from_180 < min_deviation_from_180:
                min_deviation_from_180 = deviation_from_180
                best_edge = tuple(sorted(candidate_edge))
        
        if best_edge is None:
            self.logger.debug(f"Could not find opposite edge for {current_edge} "
                             f"through target vertex {target_vertex}")
        
        return best_edge
    
    def _get_two_neighbors(self, vp_idx: int) -> Tuple[int, int]:
        """
        Get the two neighbors of a VP via boundary_segments.
        
        CRITICAL: Every VP has exactly 2 neighbors (one on each side of the boundary segment).
        
        Args:
            vp_idx: Variable point index
            
        Returns:
            (left_neighbor_idx, right_neighbor_idx)
            Note: These might be boundary or non-boundary VPs!
            
        Raises:
            ValueError: If VP doesn't have exactly 2 neighbors
        """
        neighbors = []
        for segment in self.partition.boundary_segments:
            if segment.vp_idx_1 == vp_idx:
                neighbors.append(segment.vp_idx_2)
            elif segment.vp_idx_2 == vp_idx:
                neighbors.append(segment.vp_idx_1)
        
        if len(neighbors) != 2:
            raise ValueError(f"VP {vp_idx} must have exactly 2 neighbors, found {len(neighbors)}")
        
        return (neighbors[0], neighbors[1])
    
    def _construct_auxiliary_component_2vp(self, component: Dict,
                                           strict_validation: bool = True) -> List[int]:
        """
        Construct auxiliary 3-VP component for a 2-VP labeled component.
        
        The labeled component has 2 VPs, but the Type 1 migration needs 3 VPs.
        This finds the third VP (neighbor) that approaches the target vertex
        and is closest to it.
        
        CRITICAL: Filters candidates by target vertex FIRST, then selects by distance.
        
        Args:
            component: Component dict with 'vp_indices' (2 VPs), 'target_vertex', and 'index'
            strict_validation: If False, use fallback (min distance) when no valid
                             third VP found. If True, raise error. Default True.
            
        Returns:
            List of 3 VP indices ordered by topology: [left/middle/right]
            
        Raises:
            ValueError: If component doesn't have exactly 2 VPs or (if strict_validation=True)
                       no valid third VP found
        """
        vp_indices = component['vp_indices']
        target_vertex = component['target_vertex']
        comp_idx = component.get('index', '?')
        
        self.logger.debug(f"Component {comp_idx}: Constructing auxiliary for 2-VP {vp_indices}, target vertex: {target_vertex}")
        
        if len(vp_indices) != 2:
            raise ValueError(f"Component must have exactly 2 VPs, found {len(vp_indices)}")
        
        vp_a, vp_b = vp_indices[0], vp_indices[1]
        
        # Get neighbors of each VP
        neighbors_a = self._get_two_neighbors(vp_a)
        neighbors_b = self._get_two_neighbors(vp_b)
        
        # Find candidate third VPs (neighbors not in component)
        candidates = []
        
        # Left candidate: neighbor of vp_a that's not vp_b
        left_candidate = neighbors_a[0] if neighbors_a[0] != vp_b else neighbors_a[1]
        if left_candidate not in vp_indices:
            candidates.append(('left', left_candidate))
        
        # Right candidate: neighbor of vp_b that's not vp_a
        right_candidate = neighbors_b[0] if neighbors_b[0] != vp_a else neighbors_b[1]
        if right_candidate not in vp_indices:
            candidates.append(('right', right_candidate))
        
        if not candidates:
            raise ValueError(f"Component {comp_idx}: No third VP found for 2-VP component {vp_indices}")
        
        self.logger.debug(f"Component {comp_idx}:   Initial candidates: {[(pos, vp) for pos, vp in candidates]}")
        
        # CRITICAL: Filter by target vertex FIRST
        filtered_candidates = []
        rejected_candidates = []
        
        for position, vp_idx in candidates:
            vp = self.partition.variable_points[vp_idx]
            if target_vertex in vp.edge:
                filtered_candidates.append((position, vp_idx))
                self.logger.debug(f"Component {comp_idx}:   ✓ VP {vp_idx} ({position}): edge {vp.edge} contains target vertex {target_vertex}")
            else:
                rejected_candidates.append((position, vp_idx, vp.edge))
                self.logger.warning(
                    f"Component {comp_idx}:   ✗ VP {vp_idx} ({position}): edge {vp.edge} does NOT contain target vertex {target_vertex} - REJECTED"
                )
        
        if not filtered_candidates:
            error_msg = (
                f"Component {comp_idx}: No valid third VP found for 2-VP component {vp_indices}. "
                f"All candidates rejected (don't approach target vertex {target_vertex}): "
            )
            for pos, vp_idx, edge in rejected_candidates:
                error_msg += f"\n  - VP {vp_idx} ({pos}): edge {edge}"
            
            if strict_validation:
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            else:
                # Fallback for visualization: use candidate with minimum distance regardless of validity
                self.logger.warning(f"{error_msg}\nComponent {comp_idx}: Using FALLBACK (min distance) for visualization.")
                
                best_candidate = None
                min_dist = float('inf')
                
                for position, vp_idx in candidates:
                    dist = self.compute_boundary_distance(vp_idx)
                    if dist < min_dist:
                        min_dist = dist
                        best_candidate = (position, vp_idx)
                
                position, third_vp = best_candidate
                
                # Build ordered auxiliary component
                if position == 'left':
                    auxiliary = [third_vp, vp_a, vp_b]
                else:  # position == 'right'
                    auxiliary = [vp_a, vp_b, third_vp]
                
                self.logger.warning(
                    f"Component {comp_idx}: ⚠ FALLBACK: Using auxiliary component {auxiliary} "
                    f"(third VP: {third_vp}, position: {position}, dist: {min_dist:.6f}) "
                    f"even though it doesn't approach target vertex"
                )
                
                return auxiliary
        
        if rejected_candidates:
            self.logger.info(
                f"Component {comp_idx}:   Rejected {len(rejected_candidates)} candidate(s) that don't approach target vertex {target_vertex}"
            )
        
        # THEN select by distance from FILTERED candidates
        best_candidate = None
        min_dist = float('inf')
        
        for position, vp_idx in filtered_candidates:
            dist = self.compute_boundary_distance(vp_idx)
            self.logger.debug(f"Component {comp_idx}:   VP {vp_idx} ({position}): distance = {dist:.6f}")
            if dist < min_dist:
                min_dist = dist
                best_candidate = (position, vp_idx)
        
        position, third_vp = best_candidate
        
        # Build ordered auxiliary component
        if position == 'left':
            auxiliary = [third_vp, vp_a, vp_b]
        else:  # position == 'right'
            auxiliary = [vp_a, vp_b, third_vp]
        
        self.logger.info(
            f"Component {comp_idx}: ✓ Auxiliary component for 2-VP {vp_indices}: {auxiliary} "
            f"(third VP: {third_vp}, position: {position}, dist: {min_dist:.6f})"
        )
        
        return auxiliary
    
    def _construct_auxiliary_component_1vp(self, component: Dict, 
                                           strict_validation: bool = True) -> List[int]:
        """
        Construct auxiliary 3-VP component for a 1-VP labeled component.
        
        Evaluates three candidate triplets, filters by target vertex, then
        selects the one with minimum total distance:
        - (VP_a, VP_b, VP_c): Two left neighbors
        - (VP_b, VP_c, VP_d): Middle configuration  
        - (VP_c, VP_d, VP_e): Two right neighbors
        
        CRITICAL: All VPs in triplet must approach target vertex.
        
        Args:
            component: Component dict with 'vp_indices' (1 VP), 'target_vertex', and 'index'
            strict_validation: If False, use fallback (min distance) when no valid
                             triplet found. If True, raise error. Default True.
            
        Returns:
            List of 3 VP indices for best triplet ordered by topology
            
        Raises:
            ValueError: If component doesn't have exactly 1 VP or (if strict_validation=True)
                       no valid triplet found
        """
        vp_indices = component['vp_indices']
        target_vertex = component['target_vertex']
        comp_idx = component.get('index', '?')
        
        self.logger.debug(f"Component {comp_idx}: Constructing auxiliary for 1-VP [{vp_indices[0]}], target vertex: {target_vertex}")
        
        if len(vp_indices) != 1:
            raise ValueError(f"Component {comp_idx}: Component must have exactly 1 VP, found {len(vp_indices)}")
        
        vp_c = vp_indices[0]
        
        # Get first and second level neighbors
        neighbors_c = self._get_two_neighbors(vp_c)
        vp_b, vp_d = neighbors_c[0], neighbors_c[1]
        
        # Get second level neighbors
        try:
            neighbors_b = self._get_two_neighbors(vp_b)
            vp_a = neighbors_b[0] if neighbors_b[0] != vp_c else neighbors_b[1]
        except Exception as e:
            self.logger.warning(f"Component {comp_idx}: Could not get second level left neighbor for VP {vp_c}: {e}")
            vp_a = None
        
        try:
            neighbors_d = self._get_two_neighbors(vp_d)
            vp_e = neighbors_d[0] if neighbors_d[0] != vp_c else neighbors_d[1]
        except Exception as e:
            self.logger.warning(f"Component {comp_idx}: Could not get second level right neighbor for VP {vp_c}: {e}")
            vp_e = None
        
        # Build candidate triplets
        triplet_candidates = []
        if vp_a is not None:
            triplet_candidates.append(('left-left-center', [vp_a, vp_b, vp_c]))
        triplet_candidates.append(('left-center-right', [vp_b, vp_c, vp_d]))
        if vp_e is not None:
            triplet_candidates.append(('center-right-right', [vp_c, vp_d, vp_e]))
        
        # Evaluate each triplet: ALL VPs must approach target vertex
        valid_candidates = []
        
        for config_name, triplet in triplet_candidates:
            self.logger.debug(f"Component {comp_idx}:   Evaluating triplet '{config_name}': {triplet}")
            
            valid = True
            invalid_vps = []
            total_dist = 0.0
            
            for vp_idx in triplet:
                vp = self.partition.variable_points[vp_idx]
                if target_vertex not in vp.edge:
                    valid = False
                    invalid_vps.append((vp_idx, vp.edge))
                    self.logger.warning(
                        f"Component {comp_idx}:     ✗ VP {vp_idx}: edge {vp.edge} does NOT contain target vertex {target_vertex}"
                    )
                else:
                    dist = self.compute_boundary_distance(vp_idx)
                    total_dist += dist
                    self.logger.debug(f"Component {comp_idx}:     ✓ VP {vp_idx}: approaches target vertex, dist={dist:.6f}")
            
            if valid:
                valid_candidates.append((config_name, triplet, total_dist))
                self.logger.info(f"Component {comp_idx}:   ✓ Triplet '{config_name}' is VALID (total_dist={total_dist:.6f})")
            else:
                self.logger.warning(
                    f"Component {comp_idx}:   ✗ Triplet '{config_name}' is INVALID - {len(invalid_vps)} VP(s) don't approach target vertex"
                )
        
        if not valid_candidates:
            error_msg = (
                f"Component {comp_idx}: No valid triplet found for 1-VP component [{vp_c}]. "
                f"All triplets have VPs that don't approach target vertex {target_vertex}."
            )
            
            if strict_validation:
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            else:
                # Fallback for visualization: use triplet with minimum distance regardless of validity
                self.logger.warning(f"{error_msg} Component {comp_idx}: Using FALLBACK (min distance) for visualization.")
                
                # Calculate distances for all triplets
                fallback_candidates = []
                for config_name, triplet in triplet_candidates:
                    total_dist = sum(self.compute_boundary_distance(vp) for vp in triplet)
                    fallback_candidates.append((config_name, triplet, total_dist))
                
                best_config, best_triplet, best_total = min(fallback_candidates, key=lambda x: x[2])
                self.logger.warning(
                    f"Component {comp_idx}: ⚠ FALLBACK: Using triplet '{best_config}': {best_triplet} (total_dist={best_total:.6f}) "
                    f"even though some VPs don't approach target vertex"
                )
                return best_triplet
        
        # Select triplet with minimum total distance from VALID candidates
        best_config, best_triplet, best_total = min(valid_candidates, key=lambda x: x[2])
        
        self.logger.info(
            f"Component {comp_idx}: ✓ Auxiliary component for 1-VP [{vp_c}]: {best_triplet} "
            f"(config: '{best_config}', total_dist: {best_total:.6f}, "
            f"rejected {len(triplet_candidates) - len(valid_candidates)} invalid triplet(s))"
        )
        
        return best_triplet
    
    def _get_neighbors_from_auxiliary(self, migrating_vp_idx: int, 
                                      auxiliary_component: List[int]) -> Tuple[int, int]:
        """
        Get left and right neighbors from auxiliary component using adjacency.
        
        Uses the boundary segment graph to find the two neighbors of the migrating VP
        that are also in the auxiliary component. This works regardless of list ordering.
        
        Args:
            migrating_vp_idx: Index of migrating VP
            auxiliary_component: List of 3 VPs (order doesn't matter)
            
        Returns:
            (left_neighbor, right_neighbor)
            
        Raises:
            ValueError: If migrating VP is not in auxiliary or neighbors can't be determined
        """
        self.logger.debug(
            f"Getting neighbors for migrating VP {migrating_vp_idx} from auxiliary component {auxiliary_component}"
        )
        
        # Verify migrating VP is in auxiliary component
        if migrating_vp_idx not in auxiliary_component:
            error_msg = (
                f"CRITICAL ERROR: Migrating VP {migrating_vp_idx} is NOT in auxiliary component {auxiliary_component}. "
                f"This indicates a bug in topology-based selection."
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Use adjacency graph to find neighbors (not list positions)
        try:
            all_neighbors = self._get_two_neighbors(migrating_vp_idx)
        except Exception as e:
            error_msg = f"Failed to get neighbors from boundary graph for VP {migrating_vp_idx}: {e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Filter to neighbors that are in the auxiliary component
        auxiliary_set = set(auxiliary_component)
        neighbors_in_auxiliary = [n for n in all_neighbors if n in auxiliary_set]
        
        if len(neighbors_in_auxiliary) != 2:
            error_msg = (
                f"Expected 2 neighbors in auxiliary component, found {len(neighbors_in_auxiliary)}. "
                f"Migrating VP: {migrating_vp_idx}, All neighbors: {all_neighbors}, "
                f"Auxiliary component: {auxiliary_component}"
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        left_neighbor, right_neighbor = neighbors_in_auxiliary[0], neighbors_in_auxiliary[1]
        
        self.logger.info(
            f"✓ Found neighbors for migrating VP {migrating_vp_idx} from adjacency graph: "
            f"left={left_neighbor}, right={right_neighbor}"
        )
        
        return (left_neighbor, right_neighbor)
    
    def select_migrating_vp_and_auxiliary(self, component: Dict,
                                          strict_validation: bool = True) -> Tuple[int, List[int]]:
        """
        Select migrating VP and construct auxiliary component.
        
        Returns both the migrating VP and the auxiliary component to avoid
        redundant construction calls.
        
        Args:
            component: Component info dict from analyze_component()
            strict_validation: If False, use fallback when auxiliary construction fails.
                             If True, raise error. Default True.
            
        Returns:
            Tuple of (migrating_vp_idx, auxiliary_component)
        """
        from collections import defaultdict
        
        size = len(component['vp_indices'])
        
        if size == 3:
            auxiliary_component = component['vp_indices']
        elif size == 2:
            auxiliary_component = self._construct_auxiliary_component_2vp(component, strict_validation)
        elif size == 1:
            auxiliary_component = self._construct_auxiliary_component_1vp(component, strict_validation)
        else:
            raise ValueError(f"Unexpected component size: {size}")
        
        # Find middle VP (degree 2 in auxiliary component)
        auxiliary_set = set(auxiliary_component)
        adjacency = defaultdict(set)
        for segment in self.partition.boundary_segments:
            vp1, vp2 = segment.vp_idx_1, segment.vp_idx_2
            if vp1 in auxiliary_set and vp2 in auxiliary_set:
                adjacency[vp1].add(vp2)
                adjacency[vp2].add(vp1)
        
        # Find VP with degree 2
        for vp_idx in auxiliary_component:
            if len(adjacency[vp_idx]) == 2:
                self.logger.info(
                    f"Selected migrating VP {vp_idx} (topology-based: degree 2, component size {size})"
                )
                return (vp_idx, auxiliary_component)
        
        # Fallback
        self.logger.warning(
            f"Could not find degree-2 VP in auxiliary component {auxiliary_component}, "
            f"using first VP from component"
        )
        migrating_vp = component['vp_indices'][0]
        return (migrating_vp, auxiliary_component)
    
    def select_migrating_vp_topology_based(self, component: Dict,
                                           strict_validation: bool = True) -> int:
        """
        Select migrating VP using topology-based criteria for all component sizes.
        
        This prevents selecting an endpoint VP whose external neighbor doesn't
        approach the same target vertex, which would cause migration failures.
        
        Strategy:
        - 3-VP: Find VP with degree 2 in component graph (middle VP)
        - 2-VP: Construct auxiliary 3-VP component, find middle VP
        - 1-VP: Construct auxiliary 3-VP component, find middle VP
        
        Args:
            component: Component info dict from analyze_component()
            strict_validation: If False, use fallback when auxiliary construction fails.
                             If True, raise error. Default True.
            
        Returns:
            VP index that is topologically middle
            
        Raises:
            ValueError: If middle VP cannot be determined (only if strict_validation=True)
        """
        migrating_vp, _ = self.select_migrating_vp_and_auxiliary(component, strict_validation)
        return migrating_vp
    
    def _validate_migration_trio(self, migrating_vp_idx: int, left_neighbor: int, 
                                 right_neighbor: int, target_vertex: int) -> Tuple[bool, str]:
        """
        Validate that all 3 VPs approach the same target vertex.
        
        This prevents migration failures caused by including external neighbors
        that don't share the target vertex.
        
        Args:
            migrating_vp_idx: Index of migrating VP
            left_neighbor: Index of left neighbor VP
            right_neighbor: Index of right neighbor VP
            target_vertex: Expected target vertex for migration
            
        Returns:
            (is_valid, error_message)
            - is_valid: True if all VPs approach target_vertex
            - error_message: Detailed error if validation fails, empty if valid
        """
        vps_to_check = [
            (migrating_vp_idx, "migrating VP"),
            (left_neighbor, "left neighbor"),
            (right_neighbor, "right neighbor")
        ]
        
        invalid_vps = []
        
        for vp_idx, vp_name in vps_to_check:
            vp = self.partition.variable_points[vp_idx]
            edge = vp.edge
            
            # Check if target_vertex is in the VP's edge
            if target_vertex not in edge:
                invalid_vps.append({
                    'vp_idx': vp_idx,
                    'vp_name': vp_name,
                    'edge': edge,
                    'lambda': vp.lambda_param
                })
        
        if invalid_vps:
            # Construct detailed error message
            error_lines = [
                f"Migration validation FAILED: Not all VPs approach target vertex {target_vertex}"
            ]
            for info in invalid_vps:
                error_lines.append(
                    f"  - {info['vp_name']} (VP{info['vp_idx']}): "
                    f"edge {info['edge']} does NOT contain target vertex {target_vertex} "
                    f"(λ={info['lambda']:.6f})"
                )
            error_lines.append(
                f"Valid VPs should have edges containing vertex {target_vertex}"
            )
            
            error_message = "\n".join(error_lines)
            return (False, error_message)
        
        # All VPs valid
        return (True, "")
    
    def _find_triangle_with_segment(self, vp_idx1: int, vp_idx2: int) -> Optional[int]:
        """
        Find the triangle that contains a segment between two VPs.
        
        Args:
            vp_idx1: First VP index
            vp_idx2: Second VP index
            
        Returns:
            Triangle index if found, None otherwise
        """
        for tri_seg in self.partition.triangle_segments:
            if vp_idx1 in tri_seg.var_point_indices and vp_idx2 in tri_seg.var_point_indices:
                return tri_seg.triangle_idx
        return None
    
    def _compute_distance_to_vertex(self, vp_idx: int, target_vertex: int) -> float:
        """
        Compute distance of VP to target vertex along its edge.
        
        Distance is measured as the lambda parameter distance, where:
        - λ=1 means at edge[0], λ=0 means at edge[1]
        - Distance to vertex is min(λ, 1-λ) when vertex is an endpoint
        
        Args:
            vp_idx: Variable point index
            target_vertex: Vertex index to compute distance to
            
        Returns:
            Distance in [0, 0.5] range (0 = at vertex, 0.5 = at midpoint)
        """
        vp = self.partition.variable_points[vp_idx]
        edge = vp.edge
        lambda_param = vp.lambda_param
        
        # Determine which endpoint is the target vertex
        if edge[0] == target_vertex:
            # λ=1 → at edge[0], so distance from target = 1 - lambda_param
            distance = 1.0 - lambda_param
        elif edge[1] == target_vertex:
            # λ=0 → at edge[1], so distance from target = lambda_param
            distance = lambda_param
        else:
            # Target vertex not on this edge (shouldn't happen for boundary VPs approaching target)
            self.logger.warning(f"VP {vp_idx} on edge {edge} does not contain target vertex {target_vertex}")
            return 0.5
        
        return distance
    
    def _compute_lambda_for_distance(self, new_edge: Tuple[int, int], 
                                      target_vertex: int, 
                                      target_distance: float) -> float:
        """
        Compute lambda parameter that places VP at target_distance from target_vertex.
        
        Args:
            new_edge: The new edge (normalized)
            target_vertex: The vertex to maintain distance to
            target_distance: Desired distance in [0, 0.5] range
            
        Returns:
            Lambda value in [0, 1] that achieves target_distance
        """
        # Clamp target_distance to valid range
        target_distance = max(0.0, min(0.5, target_distance))
        
        # Determine which endpoint is the target vertex
        if new_edge[0] == target_vertex:
            # λ=1 → at edge[0], so we want: 1 - lambda = target_distance
            # Therefore: lambda = 1 - target_distance
            return 1.0 - target_distance
        elif new_edge[1] == target_vertex:
            # λ=0 → at edge[1], so we want: lambda = target_distance
            return target_distance
        else:
            # Target vertex not on new edge (shouldn't happen)
            self.logger.warning(f"New edge {new_edge} does not contain target vertex {target_vertex}, using midpoint")
            return 0.5
    
    def _adjust_neighbor_to_free_edge(self, neighbor_vp_idx: int, 
                                      migrating_vp_idx: int,
                                      target_vertex: Optional[int] = None,
                                      target_distance: Optional[float] = None) -> bool:
        """
        Adjust neighbor VP to free edge in the triangle containing its segment with its OTHER neighbor.
        
        CRITICAL: The neighbor VP belongs to TWO triangles (its edge is shared by 2 triangles).
        We must move it to the free edge in the triangle that contains the segment to its
        OTHER neighbor (not the migrating VP).
        
        Example: For component VP1 — VP2 — VP3 — VP4 — VP5 where VP3 is migrating:
        - VP2's other neighbor is VP1 (not VP3)
        - We find the triangle containing segment VP1-VP2
        - Move VP2 to a free edge in THAT triangle
        
        Args:
            neighbor_vp_idx: VP to adjust (e.g., VP2)
            migrating_vp_idx: The migrating VP (e.g., VP3) - used to identify the other neighbor
            target_vertex: The vertex to maintain distance to (optional)
            target_distance: Desired distance from target_vertex (optional, default 0.5)
            
        Returns:
            True if successful, False otherwise
        """
        # Use midpoint by default if no distance specified
        if target_distance is None:
            target_distance = 0.5
        # Step 1: Find the other neighbor of neighbor_vp (the one that's NOT migrating_vp)
        neighbor_vp_neighbors = []
        for segment in self.partition.boundary_segments:
            if segment.vp_idx_1 == neighbor_vp_idx:
                neighbor_vp_neighbors.append(segment.vp_idx_2)
            elif segment.vp_idx_2 == neighbor_vp_idx:
                neighbor_vp_neighbors.append(segment.vp_idx_1)
        
        if len(neighbor_vp_neighbors) != 2:
            self.logger.error(f"Neighbor VP {neighbor_vp_idx} must have exactly 2 neighbors, found {len(neighbor_vp_neighbors)}")
            return False
        
        # Find the other neighbor (not the migrating VP)
        other_neighbor = None
        for n in neighbor_vp_neighbors:
            if n != migrating_vp_idx:
                other_neighbor = n
                break
        
        if other_neighbor is None:
            self.logger.error(f"Could not find other neighbor for VP {neighbor_vp_idx} (migrating VP: {migrating_vp_idx})")
            return False
        
        self.logger.debug(f"Neighbor VP {neighbor_vp_idx}: other neighbor is VP {other_neighbor} "
                         f"(segment {other_neighbor}-{neighbor_vp_idx})")
        
        # Step 2: Find the triangle containing the segment between neighbor_vp and its other neighbor
        target_triangle = self._find_triangle_with_segment(neighbor_vp_idx, other_neighbor)
        
        if target_triangle is None:
            self.logger.warning(f"Could not find triangle containing segment VP{other_neighbor}-VP{neighbor_vp_idx}")
            # Fallback: try any triangle containing neighbor_vp's edge
            neighbor_vp = self.partition.variable_points[neighbor_vp_idx]
            neighbor_triangles = self.mesh_topology.get_triangles_sharing_edge(neighbor_vp.edge)
            if not neighbor_triangles:
                self.logger.error(f"Neighbor VP {neighbor_vp_idx}: No triangles found for edge {neighbor_vp.edge}")
                return False
            target_triangle = neighbor_triangles[0]  # Use first triangle as fallback
            self.logger.warning(f"Using fallback triangle {target_triangle} for neighbor VP {neighbor_vp_idx}")
        else:
            self.logger.debug(f"Found triangle {target_triangle} containing segment VP{other_neighbor}-VP{neighbor_vp_idx}")
        
        # Step 3: Find a free edge in the target triangle
        neighbor_vp = self.partition.variable_points[neighbor_vp_idx]
        tri_edges = self.mesh.get_triangle_edges(target_triangle)
        
        for edge in tri_edges:
            edge_norm = tuple(sorted(edge))
            # Skip the neighbor's current edge and any edge with a VP
            if edge_norm != tuple(sorted(neighbor_vp.edge)) and edge_norm not in self.partition.edge_to_varpoint:
                # Found free edge in the correct triangle!
                # Compute lambda to preserve distance to target vertex
                if target_vertex is not None:
                    lambda_param = self._compute_lambda_for_distance(edge_norm, target_vertex, target_distance)
                else:
                    lambda_param = 0.5  # Default to midpoint if no target vertex specified
                
                self._move_variable_point(neighbor_vp_idx, edge_norm, lambda_param)
                self.logger.debug(f"Adjusted neighbor VP {neighbor_vp_idx} to free edge {edge_norm} "
                                f"in triangle {target_triangle} (contains segment VP{other_neighbor}-VP{neighbor_vp_idx}) "
                                f"with λ={lambda_param:.6f} (distance to target: {target_distance:.6f})")
                return True
        
        self.logger.warning(f"Neighbor VP {neighbor_vp_idx}: No free edge found in triangle {target_triangle} "
                          f"(contains segment VP{other_neighbor}-VP{neighbor_vp_idx})")
        return False
    
    def _determine_target_vertex_cell_flip(
        self, 
        target_vertex: int,
        migrating_vp_idx: int
    ) -> Tuple[int, int]:
        """
        Determine which cells the target vertex flips between during Type 1 migration.
        
        Simple logic:
        1. Get the 2 cells separated by the boundary (from VP.belongs_to_cells)
        2. Get target vertex's current cell (from indicator_functions BEFORE migration)
        3. Target vertex flips to the OTHER cell
        
        MUST be called BEFORE moving any VPs to capture the original cell assignment.
        
        Args:
            target_vertex: The vertex that changes cells
            migrating_vp_idx: The VP being migrated (to get cell info)
            
        Returns:
            (old_cell, new_cell) tuple
        """
        # Get the two cells involved in this boundary
        migrating_vp = self.partition.variable_points[migrating_vp_idx]
        cells = migrating_vp.belongs_to_cells
        
        if len(cells) != 2:
            raise ValueError(f"Expected boundary VP to separate 2 cells, found {len(cells)}: {cells}")
        
        # Get target vertex's current cell assignment (BEFORE migration)
        vertex_labels = np.argmax(self.partition.indicator_functions, axis=1)
        old_cell = int(vertex_labels[target_vertex])
        
        # Target vertex flips to the OTHER cell
        cells_list = list(cells)
        if cells_list[0] == old_cell:
            new_cell = cells_list[1]
        elif cells_list[1] == old_cell:
            new_cell = cells_list[0]
        else:
            # This shouldn't happen - target vertex should be in one of the boundary cells
            self.logger.warning(f"Target vertex {target_vertex} in cell {old_cell}, "
                              f"but boundary separates cells {cells_list}")
            new_cell = cells_list[0] if cells_list[0] != old_cell else cells_list[1]
        
        self.logger.debug(f"Target vertex {target_vertex} will flip: cell {old_cell} → cell {new_cell}")
        
        return (old_cell, new_cell)
    
    def _update_indicator_functions_for_target_vertex(
        self,
        target_vertex: int,
        old_cell: int,
        new_cell: int
    ) -> None:
        """
        Update indicator_functions matrix after Type 1 migration.
        
        The target vertex is the ONLY vertex that changes cells during Type 1 migration.
        This is the fundamental change that causes triangles to gain/lose segments.
        
        Updates:
            partition.indicator_functions[target_vertex, old_cell] = 0
            partition.indicator_functions[target_vertex, new_cell] = 1
        
        Args:
            target_vertex: Index of vertex that changes cells
            old_cell: Cell index before migration
            new_cell: Cell index after migration
        """
        # Flip the target vertex cell assignment
        self.partition.indicator_functions[target_vertex, old_cell] = 0
        self.partition.indicator_functions[target_vertex, new_cell] = 1
        
        self.logger.info(f"Updated indicator_functions: vertex {target_vertex} "
                        f"flipped from cell {old_cell} to cell {new_cell}")
    
    def update_data_structures_after_type1_migration(self, target_vertex: int):
        """
        Optimized update for Type 1 vertex-collapse migration.
        
        Only updates the 6 triangles affected by target vertex cell flip.
        This is MUCH faster than rebuilding all triangles.
        
        IMPORTANT: indicator_functions MUST be updated BEFORE calling this method,
        because rebuild uses vertex_labels from indicator_functions.
        
        Updates (in this order):
        1. partition.triangle_segments (rebuilds ONLY 6 affected triangles)
        2. Verifies edge_to_varpoint consistency
        
        Does NOT update:
        - boundary_segments: Connectivity unchanged (VP1 still connected to VP2)
        - segment_crossing_cache: Not used in vertex-collapse (no crossings)
        
        Args:
            target_vertex: The vertex that changed cells (shared by 6 triangles)
        """
        # Step 1: Get affected triangles (only 6)
        affected_triangles = self._get_all_triangles_at_vertex(target_vertex)
        
        self.logger.debug(f"Updating {len(affected_triangles)} triangles affected by "
                         f"target vertex {target_vertex} cell flip")
        
        # Step 2: Rebuild only affected triangle_segments
        self.partition.rebuild_triangle_segments_for_affected_triangles(affected_triangles)
        
        # Step 2.5: Rebuild segment_to_triangle map (after triangle_segments updated)
        self.partition.rebuild_segment_to_triangle_map()
        
        # Step 3: Verify edge_to_varpoint consistency
        for vp_idx, vp in enumerate(self.partition.variable_points):
            edge_norm = tuple(sorted(vp.edge))
            if edge_norm not in self.partition.edge_to_varpoint:
                self.logger.warning(f"VP {vp_idx} edge {edge_norm} not in edge_to_varpoint!")
            elif self.partition.edge_to_varpoint[edge_norm] != vp_idx:
                self.logger.warning(f"VP {vp_idx} edge {edge_norm} mapped to different VP "
                                  f"{self.partition.edge_to_varpoint[edge_norm]}!")
        
        self.logger.debug(f"Type 1 data structures updated: "
                         f"{len(self.partition.triangle_segments)} triangle segments total")
    
    def update_data_structures_after_migration(self):
        """
        General-purpose update for all migration types (Type 1, Type 2, etc.).
        
        This is CRITICAL because:
        1. Triangles that had boundary segments may no longer have them
        2. Triangles that didn't have boundary segments may now have them
        3. This affects visualization (cell colors) and area calculations
        
        IMPORTANT: indicator_functions MUST be updated BEFORE calling this method,
        because rebuild_triangle_segments_from_current_vps() reads vertex_labels
        from indicator_functions for the TriangleSegment.vertex_labels attribute.
        
        Updates (in this order):
        1. partition.triangle_segments (rebuilds from current VPs and updated indicator_functions)
        2. partition.boundary_segments (rebuilds connectivity)
        3. Verifies edge_to_varpoint consistency
        
        Note: For Type 1 vertex-collapse, use update_data_structures_after_type1_migration()
        instead - it's much faster (6 triangles vs all triangles).
        
        Note: segment_crossing_cache is NOT used in vertex-collapse strategy.
        """
        # Step 1: Rebuild triangle_segments based on current VP positions
        # This updates which triangles have boundary segments and which don't
        self.partition.rebuild_triangle_segments_from_current_vps()
        
        # rebuild_triangle_segments_from_current_vps() automatically calls
        # _build_segment_connectivity() at the end, which rebuilds boundary_segments
        
        # Step 2: Verify edge_to_varpoint consistency
        for vp_idx, vp in enumerate(self.partition.variable_points):
            edge_norm = tuple(sorted(vp.edge))
            if edge_norm not in self.partition.edge_to_varpoint:
                self.logger.warning(f"VP {vp_idx} edge {edge_norm} not in edge_to_varpoint!")
            elif self.partition.edge_to_varpoint[edge_norm] != vp_idx:
                self.logger.warning(f"VP {vp_idx} edge {edge_norm} mapped to different VP "
                                  f"{self.partition.edge_to_varpoint[edge_norm]}!")
        
        self.logger.debug(f"Data structures updated after migration: "
                         f"{len(self.partition.triangle_segments)} triangle segments, "
                         f"{len(self.partition.boundary_segments)} boundary segments")
    
    def apply_type1_switch_v2(self, component: Dict, 
                              distance_preservation: str = 'preserve',
                              strict_validation: bool = True) -> bool:
        """
        Apply Type 1 switch to entire component using vertex-collapse strategy.
        
        Uses topology-based VP selection and auxiliary components to ensure:
        - Migrating VP is the middle VP (not endpoint)
        - Neighbors approach the same target vertex
        - Migration doesn't affect VPs outside the component scope
        
        CRITICAL: This function should only be called with components that contain
        VPs NOT in triple points. The filtering is done at the workflow level.
        
        Args:
            component: Component info dict from analyze_component() with 'vp_indices' field
            distance_preservation: Strategy for setting lambda after migration:
                - 'preserve': Maintain original distance to target vertex (default)
                - 'midpoint': Place at midpoint (λ=0.5)
                - float as string: Use specific distance (e.g., '0.1' for close to target)
            strict_validation: If False (visualization mode), use fallback when validation fails.
                             If True (optimization mode), halt on validation errors. Default True.
            
        Returns:
            Dict with migration results (see docstring end) or False for backward compatibility
        """
        component_vps = component['vp_indices']
        component_size = len(component_vps)
        
        if not component_vps:
            self.logger.warning("Empty component - cannot migrate")
            return {'success': False, 'error': 'Empty component'}
        
        self.logger.info("="*80)
        self.logger.info(f"STARTING TYPE 1 MIGRATION")
        self.logger.info(f"Component: {component_vps} (size: {component_size})")
        self.logger.info(f"Target vertex: {component.get('target_vertex', 'unknown')}")
        self.logger.info("="*80)
        
        # Step 1: Use topology-based VP selection
        self.logger.info("Step 1: Selecting migrating VP using TOPOLOGY-BASED criteria...")
        try:
            migrating_vp_idx = self.select_migrating_vp_topology_based(component, strict_validation)
            self.logger.info(f"✓ Selected migrating VP: {migrating_vp_idx}")
        except Exception as e:
            self.logger.error(f"❌ Failed to select migrating VP: {e}")
            return {'success': False, 'error': f'VP selection failed: {e}'}
        
        migrating_vp = self.partition.variable_points[migrating_vp_idx]
        old_edge = migrating_vp.edge
        
        # Step 1.5: Construct/identify auxiliary component
        self.logger.info("Step 1.5: Determining auxiliary component for neighbor selection...")
        try:
            if component_size == 3:
                auxiliary_component = component_vps
                self.logger.info(f"✓ Using 3-VP component directly: {auxiliary_component}")
            elif component_size == 2:
                auxiliary_component = self._construct_auxiliary_component_2vp(component, strict_validation)
            elif component_size == 1:
                auxiliary_component = self._construct_auxiliary_component_1vp(component, strict_validation)
            else:
                raise ValueError(f"Unexpected component size: {component_size}")
        except Exception as e:
            self.logger.error(f"❌ Failed to construct auxiliary component: {e}")
            return {'success': False, 'error': f'Auxiliary component construction failed: {e}'}
        
        # Step 2: Get neighbors from auxiliary component (NOT blind _get_two_neighbors)
        self.logger.info("Step 2: Getting neighbors from auxiliary component...")
        try:
            left_neighbor, right_neighbor = self._get_neighbors_from_auxiliary(
                migrating_vp_idx, auxiliary_component
            )
        except Exception as e:
            self.logger.error(f"❌ Failed to get neighbors: {e}")
            return {'success': False, 'error': f'Neighbor selection failed: {e}'}
        
        # Step 2.5: Validate migration trio
        self.logger.info("Step 2.5: Validating migration trio...")
        target_vertex = self._identify_target_vertex(migrating_vp)
        if target_vertex is None:
            self.logger.warning(f"VP {migrating_vp_idx}: Could not identify target vertex")
            return {'success': False, 'error': 'Could not identify target vertex'}
        
        is_valid, error_msg = self._validate_migration_trio(
            migrating_vp_idx, left_neighbor, right_neighbor, target_vertex
        )
        
        if not is_valid:
            if strict_validation:
                # OPTIMIZATION MODE: Abort migration on validation failure
                self.logger.error("="*80)
                self.logger.error("❌ MIGRATION VALIDATION FAILED")
                self.logger.error("="*80)
                self.logger.error(error_msg)
                self.logger.error("="*80)
                self.logger.error("Migration ABORTED - would cause topology errors")
                return {
                    'success': False,
                    'error': 'Validation failed',
                    'validation_message': error_msg,
                    'migrating_vp_idx': migrating_vp_idx,
                    'left_neighbor': left_neighbor,
                    'right_neighbor': right_neighbor,
                    'target_vertex': target_vertex
                }
            else:
                # VISUALIZATION MODE: Log warning but continue migration
                self.logger.warning("="*80)
                self.logger.warning("⚠ MIGRATION VALIDATION FAILED (continuing for visualization)")
                self.logger.warning("="*80)
                self.logger.warning(error_msg)
                self.logger.warning("="*80)
                self.logger.warning("⚠ Proceeding with migration despite validation failure (visualization mode)")
                self.logger.warning("="*80)
        else:
            self.logger.info(f"✓ Validation PASSED: All VPs approach target vertex {target_vertex}")
        
        # Step 3: Continue with migration
        self.logger.info(f"Step 3: Proceeding with migration...")
        
        # Step 4: Find target edge (opposite edge)
        target_edge = self._find_opposite_edge(old_edge, target_vertex)
        if target_edge is None:
            self.logger.warning(f"VP {migrating_vp_idx}: Could not find opposite edge "
                              f"for {old_edge} through target vertex {target_vertex}")
            return {'success': False, 'error': 'Could not find target edge'}
        
        self.logger.debug(f"VP {migrating_vp_idx}: Target edge {target_edge} found")
        
        # Step 4: Determine which cells the target vertex flips between
        # CRITICAL: Must be done BEFORE moving VPs to capture original cell assignment
        try:
            old_cell, new_cell = self._determine_target_vertex_cell_flip(
                target_vertex, migrating_vp_idx
            )
        except ValueError as e:
            self.logger.error(f"Failed to determine cell flip: {e}")
            return {'success': False, 'error': f'Cell flip determination failed: {e}'}
        
        # Step 4.6: Calculate distances to target vertex BEFORE migration
        # This preserves the VP positions relative to target vertex
        if distance_preservation == 'preserve':
            dist_migrating = self._compute_distance_to_vertex(migrating_vp_idx, target_vertex)
            dist_left = self._compute_distance_to_vertex(left_neighbor, target_vertex)
            dist_right = self._compute_distance_to_vertex(right_neighbor, target_vertex)
            self.logger.debug(f"Preserving distances to target vertex {target_vertex}: "
                            f"migrating={dist_migrating:.6f}, left={dist_left:.6f}, right={dist_right:.6f}")
        elif distance_preservation == 'midpoint':
            dist_migrating = dist_left = dist_right = 0.5
            self.logger.debug(f"Using midpoint placement (λ=0.5) for all VPs")
        else:
            # Custom distance provided as string
            try:
                custom_dist = float(distance_preservation)
                dist_migrating = dist_left = dist_right = custom_dist
                self.logger.debug(f"Using custom distance {custom_dist:.6f} for all VPs")
            except ValueError:
                self.logger.warning(f"Invalid distance_preservation value '{distance_preservation}', using midpoint")
                dist_migrating = dist_left = dist_right = 0.5
        
        # Step 6: Move migrating VP to target edge with preserved distance
        lambda_migrating = self._compute_lambda_for_distance(target_edge, target_vertex, dist_migrating)
        self._move_variable_point(migrating_vp_idx, target_edge, lambda_migrating)
        self.logger.debug(f"VP {migrating_vp_idx}: Moved to edge {target_edge} with λ={lambda_migrating:.6f}")
        
        # Step 6: Adjust left neighbor (move to free edge in triangle containing its segment with its OTHER neighbor)
        if not self._adjust_neighbor_to_free_edge(left_neighbor, migrating_vp_idx, 
                                                  target_vertex, dist_left):
            self.logger.error(f"Failed to adjust left neighbor {left_neighbor}")
            return {'success': False, 'error': f'Left neighbor adjustment failed'}
        
        # Step 8: Adjust right neighbor (move to free edge in triangle containing its segment with its OTHER neighbor)
        if not self._adjust_neighbor_to_free_edge(right_neighbor, migrating_vp_idx,
                                                  target_vertex, dist_right):
            self.logger.error(f"Failed to adjust right neighbor {right_neighbor}")
            return {'success': False, 'error': f'Right neighbor adjustment failed'}
        
        # Step 7.5: Update indicator_functions matrix (target vertex cell flip)
        # CRITICAL: Must be done AFTER moving VPs but BEFORE rebuilding triangle_segments
        self._update_indicator_functions_for_target_vertex(target_vertex, old_cell, new_cell)
        
        self.logger.debug(f"Target vertex {target_vertex} cell flip: {old_cell} → {new_cell}")
        
        # Step 10: Update belongs_to_cells for all moved VPs
        # CRITICAL: Must happen AFTER cell flip so _determine_cells_for_edge() uses final state
        # This ensures VPs' belongs_to_cells reflect the cells their edges actually separate
        moved_vps = [
            (migrating_vp_idx, "migrating_VP"),
            (left_neighbor, "left_neighbor"),
            (right_neighbor, "right_neighbor")
        ]
        
        for vp_idx, vp_name in moved_vps:
            vp = self.partition.variable_points[vp_idx]
            old_cells = vp.belongs_to_cells.copy()
            new_cells = self._determine_cells_for_edge(vp.edge)
            vp.belongs_to_cells = new_cells
            
            if old_cells != new_cells:
                self.logger.debug(f"  {vp_name} (VP{vp_idx}): "
                                 f"belongs_to_cells updated {old_cells} → {new_cells}")
        
        # Step 8: Update data structures (OPTIMIZED for Type 1)
        # Only rebuilds 6 affected triangles, skips boundary_segments (connectivity unchanged)
        # CRITICAL: This reads indicator_functions, so it must come AFTER Step 9
        self.update_data_structures_after_type1_migration(target_vertex)
        
        self.logger.info("="*80)
        self.logger.info(f"✓ TYPE 1 MIGRATION COMPLETED SUCCESSFULLY")
        self.logger.info(f"  Migrated VP: {migrating_vp_idx}")
        self.logger.info(f"  Neighbors: {left_neighbor}, {right_neighbor}")
        self.logger.info(f"  Target vertex: {target_vertex}")
        self.logger.info(f"  Component size: {component_size}")
        self.logger.info("="*80)
        
        return {
            'success': True,
            'migrating_vp_idx': migrating_vp_idx,
            'left_neighbor': left_neighbor,
            'right_neighbor': right_neighbor,
            'target_vertex': target_vertex,
            'old_edge': old_edge,
            'target_edge': target_edge,
            'auxiliary_component': auxiliary_component,
            'validation_passed': True
        }
    
    def get_migration_plan(self, component: Dict) -> Dict:
        """
        Get migration plan for a component without executing the migration.
        
        This is useful for visualization scripts that need to know which VPs
        will be migrated before actually performing the migration.
        
        Args:
            component: Component info dict from analyze_component()
            
        Returns:
            Dict with migration plan:
            {
                'success': bool,
                'migrating_vp': int,
                'left_neighbor': int,
                'right_neighbor': int,
                'auxiliary_component': List[int],
                'target_vertex': int,
                'current_edge': Tuple[int, int],
                'target_edge': Tuple[int, int],
                'validation_passed': bool,
                'validation_message': str
            }
        """
        result = {
            'success': False,
            'error': None
        }
        
        component_vps = component['vp_indices']
        component_size = len(component_vps)
        
        try:
            # Select migrating VP
            migrating_vp_idx = self.select_migrating_vp_topology_based(component)
            migrating_vp = self.partition.variable_points[migrating_vp_idx]
            
            # Construct auxiliary component
            if component_size == 3:
                auxiliary_component = component_vps
            elif component_size == 2:
                auxiliary_component = self._construct_auxiliary_component_2vp(component)
            elif component_size == 1:
                auxiliary_component = self._construct_auxiliary_component_1vp(component)
            else:
                raise ValueError(f"Unexpected component size: {component_size}")
            
            # Get neighbors
            left_neighbor, right_neighbor = self._get_neighbors_from_auxiliary(
                migrating_vp_idx, auxiliary_component
            )
            
            # Identify target vertex and edge
            target_vertex = self._identify_target_vertex(migrating_vp)
            target_edge = self._find_opposite_edge(migrating_vp.edge, target_vertex)
            
            # Validate
            is_valid, error_msg = self._validate_migration_trio(
                migrating_vp_idx, left_neighbor, right_neighbor, target_vertex
            )
            
            result.update({
                'success': True,
                'migrating_vp': migrating_vp_idx,
                'left_neighbor': left_neighbor,
                'right_neighbor': right_neighbor,
                'auxiliary_component': auxiliary_component,
                'target_vertex': target_vertex,
                'current_edge': migrating_vp.edge,
                'target_edge': target_edge,
                'validation_passed': is_valid,
                'validation_message': error_msg if not is_valid else ''
            })
            
        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Failed to create migration plan: {e}")
        
        return result
    
    def _get_conflict_for_component(self, component: Dict, conflicts: List[Dict]) -> Optional[Dict]:
        """
        Get the conflict (if any) for a given component.
        
        Args:
            component: Component dict with 'index' field
            conflicts: List of conflict dicts
            
        Returns:
            Conflict dict if found, None otherwise
        """
        comp_idx = component['index']
        for conflict in conflicts:
            if conflict['component_i'] == comp_idx or conflict['component_j'] == comp_idx:
                return conflict
        return None
    
    def can_form_valid_auxiliary_component(self, component: Dict) -> Tuple[bool, str]:
        """
        Check if a component can form a valid auxiliary component for migration.
        
        A valid auxiliary component has 3 VPs that all approach the same target vertex.
        - 3-VP components: Already valid (verified during component analysis)
        - 2-VP components: Must find a third VP that approaches the same target vertex
        - 1-VP components: Must find two more VPs that form a valid triplet
        
        Args:
            component: Component dictionary with 'vp_indices', 'target_vertex', 'size'
        
        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        size = component['size']
        
        if size == 3:
            # 3-VP components are already validated during component analysis
            return (True, "")
        
        elif size == 2:
            # Try to construct auxiliary component for 2-VP
            try:
                auxiliary = self._construct_auxiliary_component_2vp(component, strict_validation=True)
                return (True, "")
            except ValueError as e:
                reason = f"Cannot find valid third VP: {str(e)}"
                return (False, reason)
        
        elif size == 1:
            # Try to construct auxiliary component for 1-VP
            try:
                auxiliary = self._construct_auxiliary_component_1vp(component, strict_validation=True)
                return (True, "")
            except ValueError as e:
                reason = f"Cannot find valid triplet: {str(e)}"
                return (False, reason)
        
        else:
            return (False, f"Unexpected component size: {size}")
    
    def select_components_for_migration(self, components: List[Dict], 
                                        conflicts: List[Dict],
                                        conflict_strategy: str = 'exclude_one') -> Tuple[List[Dict], List[Dict]]:
        """
        Select which components to migrate and which to exclude.
        
        NEW ARCHITECTURE (Self-Healing, No Cross-Iteration Tracking):
        1. Pre-filter: Check if valid auxiliary component can be formed
        2. Find conflicts among valid components
        3. Resolve conflicts based on strategy
        4. Return components to migrate (excluded components forgotten)
        
        Args:
            components: List of component dictionaries
            conflicts: List of conflict dictionaries
            conflict_strategy: How to resolve conflicts
                - 'exclude_one': Keep closer component, exclude farther one
                - 'exclude_all_conflicts': Exclude all conflicting components (for testing)
        
        Returns:
            (components_to_migrate, components_excluded)
            
        Note: Excluded components are NOT tracked across iterations. Each iteration
        re-evaluates all components from scratch (self-healing system).
        """
        self.logger.info("="*80)
        self.logger.info("COMPONENT SELECTION FOR MIGRATION")
        self.logger.info("="*80)
        
        # ====================================================================
        # STEP 1: PRE-FILTER - Check if valid auxiliary component can be formed
        # ====================================================================
        self.logger.info(f"\nStep 1: Pre-filtering {len(components)} components...")
        
        valid_components = []
        excluded_prefilter = []
        excluded_triple_point = []
        
        for component in components:
            comp_idx = component['index']
            comp_size = component['size']
            comp_vps = component['vp_indices']
            
            # Check 1: Near triple point exclusion (safety check)
            if component.get('near_triple_point', False):
                excluded_triple_point.append(component)
                self.logger.warning(
                    f"  ✗ Component {comp_idx} ({comp_size}-VP): EXCLUDED - near triple point "
                    f"(shared VPs: {component.get('triple_point_shared_vps', [])})"
                )
                continue
            
            # Check 2: Can form valid auxiliary component?
            self.logger.info(f"\n  Component {comp_idx} ({comp_size}-VP {comp_vps}): Checking auxiliary component...")
            is_valid, reason = self.can_form_valid_auxiliary_component(component)
            
            if is_valid:
                valid_components.append(component)
                self.logger.info(f"  Component {comp_idx}: ✓ VALID - can form auxiliary component")
            else:
                excluded_prefilter.append(component)
                self.logger.warning(f"  Component {comp_idx}: ✗ EXCLUDED - {reason}")
        
        self.logger.info(
            f"\nPre-filter results: {len(valid_components)} valid, "
            f"{len(excluded_prefilter)} excluded (auxiliary), "
            f"{len(excluded_triple_point)} excluded (triple point)"
        )
        
        if not valid_components:
            self.logger.warning("No valid components found for migration")
            return ([], excluded_prefilter + excluded_triple_point)
        
        # ====================================================================
        # STEP 2: RESOLVE CONFLICTS
        # ====================================================================
        self.logger.info(f"\nStep 2: Resolving conflicts (strategy: {conflict_strategy})...")
        
        to_migrate = []
        excluded_conflict = []
        processed = set()
        
        for component in valid_components:
            comp_idx = component['index']
            if comp_idx in processed:
                continue
            
            # Check if component has a conflict
            conflict = self._get_conflict_for_component(component, conflicts)
            
            if conflict is None:
                # No conflict → migrate immediately
                to_migrate.append(component)
                processed.add(comp_idx)
                self.logger.debug(f"  Component {comp_idx}: No conflict → migrate")
            else:
                # Has conflict → resolve based on strategy
                other_idx = conflict['component_j'] if conflict['component_i'] == comp_idx else conflict['component_i']
                
                # Check if other component is also valid (not excluded in pre-filter)
                other_component = next((c for c in valid_components if c['index'] == other_idx), None)
                
                if other_component is None:
                    # Other component was excluded in pre-filter → migrate this one
                    to_migrate.append(component)
                    processed.add(comp_idx)
                    self.logger.info(f"  Component {comp_idx}: Conflict with {other_idx} (excluded) → migrate")
                    continue
                
                # Both components are valid → apply conflict strategy
                if comp_idx < other_idx:  # Process each pair only once
                    if conflict_strategy == 'exclude_all_conflicts':
                        # Exclude both conflicting components
                        excluded_conflict.append(component)
                        excluded_conflict.append(other_component)
                        self.logger.info(
                            f"  Conflict {comp_idx} vs {other_idx}: Excluding BOTH (strategy: exclude_all_conflicts)"
                        )
                    elif conflict_strategy == 'exclude_one':
                        # Exclude farther component, keep closer one
                        if component['min_distance'] < other_component['min_distance']:
                            to_migrate.append(component)
                            excluded_conflict.append(other_component)
                            self.logger.info(
                                f"  Conflict {comp_idx} vs {other_idx}: Keeping {comp_idx} "
                                f"(dist {component['min_distance']:.6f} < {other_component['min_distance']:.6f})"
                            )
                        else:
                            to_migrate.append(other_component)
                            excluded_conflict.append(component)
                            self.logger.info(
                                f"  Conflict {comp_idx} vs {other_idx}: Keeping {other_idx} "
                                f"(dist {other_component['min_distance']:.6f} < {component['min_distance']:.6f})"
                            )
                    else:
                        raise ValueError(f"Unknown conflict_strategy: {conflict_strategy}")
                    
                    processed.add(comp_idx)
                    processed.add(other_idx)
        
        # ====================================================================
        # STEP 3: SUMMARY
        # ====================================================================
        all_excluded = excluded_prefilter + excluded_triple_point + excluded_conflict
        
        self.logger.info("="*80)
        self.logger.info("SELECTION SUMMARY:")
        self.logger.info(f"  Components to migrate: {len(to_migrate)}")
        self.logger.info(f"  Components excluded:")
        self.logger.info(f"    - Pre-filter (no valid auxiliary): {len(excluded_prefilter)}")
        self.logger.info(f"    - Triple point proximity: {len(excluded_triple_point)}")
        self.logger.info(f"    - Conflict resolution: {len(excluded_conflict)}")
        self.logger.info(f"  Total excluded: {len(all_excluded)}")
        self.logger.info("="*80)
        
        return (to_migrate, all_excluded)
    
    def apply_type2_switch_v3(self, steiner_handler, triple_point_idx: int) -> Dict:
        """
        Apply Type 2 switch using improved topological VP selection strategy.
        
        New Strategy:
        1. Identify anchor VP (sits on edge between triple triangle and target triangle)
        2. Identify target edge (free edge in target triangle)
        3. Select migrating VP based on topological connectivity:
           - Choose VP whose edge shares a vertex with target edge
           - This is more "closely connected" than VP requiring multiple edges to reach target
        
        Args:
            steiner_handler: SteinerHandler object containing triple points
            triple_point_idx: Index of triple point to migrate
            
        Returns:
            Dict with migration analysis information:
            {
                'success': bool,
                'triple_triangle_idx': int,
                'target_triangle_idx': int,
                'shared_edge': tuple,
                'target_edge': tuple,
                'anchor_vp_idx': int,
                'migrating_vp_idx': int,
                'non_migrating_vp_idx': int,
                'migrating_vp_edge': tuple,
                'non_migrating_vp_edge': tuple,
                'shared_vertex': int (vertex shared between migrating VP edge and target edge),
                'connectivity_analysis': dict
            }
        """
        result = {
            'success': False,
            'error': None
        }
        
        # Step 1: Get triple point information
        if triple_point_idx >= len(steiner_handler.triple_points):
            result['error'] = f"Invalid triple point index {triple_point_idx}"
            self.logger.error(result['error'])
            return result
        
        triple_point = steiner_handler.triple_points[triple_point_idx]
        triple_triangle_idx = triple_point.triangle_idx
        triple_vp_indices = triple_point.var_point_indices
        
        if len(triple_vp_indices) != 3:
            result['error'] = f"Triple point {triple_point_idx} does not have exactly 3 VPs"
            self.logger.error(result['error'])
            return result
        
        result['triple_triangle_idx'] = triple_triangle_idx
        result['triple_vp_indices'] = triple_vp_indices
        
        self.logger.info(f"Analyzing Type 2 migration for triple point {triple_point_idx}")
        self.logger.info(f"  Triple triangle: {triple_triangle_idx}")
        self.logger.info(f"  VPs in triple: {triple_vp_indices}")
        
        # Step 2: Find target triangle - the adjacent triangle sharing edge that Steiner point approaches
        # Use Steiner tree distances: the closest VP to Steiner point is the anchor VP
        steiner_pt = triple_point.compute_steiner_point()
        
        # Calculate distances from Steiner point to each VP
        vp_distances = {}
        for vp_idx in triple_vp_indices:
            vp_pos = self.partition.evaluate_variable_point(vp_idx)
            dist = np.linalg.norm(steiner_pt - vp_pos)
            vp_distances[vp_idx] = dist
        
        # The anchor VP is the one CLOSEST to the Steiner point (Steiner approaching that edge)
        anchor_vp_idx = min(vp_distances.keys(), key=lambda k: vp_distances[k])
        anchor_vp = self.partition.variable_points[anchor_vp_idx]
        shared_edge = tuple(sorted(anchor_vp.edge))
        
        self.logger.info(f"  Steiner point distances to VPs:")
        for vp_idx in sorted(vp_distances.keys()):
            marker = " ← ANCHOR" if vp_idx == anchor_vp_idx else ""
            self.logger.info(f"    VP {vp_idx}: {vp_distances[vp_idx]:.6f}{marker}")
        
        # Find target triangle: adjacent triangle sharing the anchor edge
        target_triangle_idx = None
        if shared_edge in self.mesh_topology.edge_to_triangles:
            adj_triangles = self.mesh_topology.edge_to_triangles[shared_edge]
            for adj_tri in adj_triangles:
                if adj_tri != triple_triangle_idx:
                    target_triangle_idx = adj_tri
                    break
        
        if not shared_edge or target_triangle_idx is None or anchor_vp_idx is None:
            result['error'] = "Could not identify shared edge, target triangle, or anchor VP"
            self.logger.error(result['error'])
            return result
        
        result['shared_edge'] = shared_edge
        result['target_triangle_idx'] = target_triangle_idx
        result['anchor_vp_idx'] = anchor_vp_idx
        
        self.logger.info(f"  Shared edge (Steiner approaching): {shared_edge}")
        self.logger.info(f"  Target triangle: {target_triangle_idx}")
        self.logger.info(f"  Anchor VP: {anchor_vp_idx} (on shared edge)")
        
        # Step 3: Identify the two candidate VPs (not the anchor)
        candidate_vps = [vp_idx for vp_idx in triple_vp_indices if vp_idx != anchor_vp_idx]
        
        if len(candidate_vps) != 2:
            result['error'] = f"Expected 2 candidate VPs, found {len(candidate_vps)}"
            self.logger.error(result['error'])
            return result
        
        self.logger.info(f"  Candidate VPs for migration: {candidate_vps}")
        
        # Step 4: Find target edge (free edge in target triangle)
        target_tri_vertices = self.mesh.faces[target_triangle_idx]
        target_tri_edges = [
            tuple(sorted([target_tri_vertices[0], target_tri_vertices[1]])),
            tuple(sorted([target_tri_vertices[1], target_tri_vertices[2]])),
            tuple(sorted([target_tri_vertices[2], target_tri_vertices[0]]))
        ]
        
        target_edge = None
        for edge in target_tri_edges:
            # Skip shared edge
            if edge == shared_edge:
                continue
            # Check if edge is free (no VPs)
            if edge not in self.partition.edge_to_varpoint:
                target_edge = edge
                break
        
        if not target_edge:
            result['error'] = "Could not find free edge in target triangle"
            self.logger.error(result['error'])
            return result
        
        result['target_edge'] = target_edge
        self.logger.info(f"  Target edge (free): {target_edge}")
        
        # Step 5: Topological selection - find VP whose edge shares a vertex with target edge
        connectivity_analysis = {}
        
        for vp_idx in candidate_vps:
            vp = self.partition.variable_points[vp_idx]
            vp_edge = tuple(sorted(vp.edge))
            
            # Find shared vertices between VP edge and target edge
            shared_vertices = set(vp_edge) & set(target_edge)
            num_shared = len(shared_vertices)
            
            connectivity_analysis[vp_idx] = {
                'edge': vp_edge,
                'shared_vertices': list(shared_vertices),
                'num_shared': num_shared
            }
            
            self.logger.info(f"  VP {vp_idx}: edge={vp_edge}, shared vertices with target={list(shared_vertices)} (count={num_shared})")
        
        result['connectivity_analysis'] = connectivity_analysis
        
        # Select migrating VP: the one with exactly 1 shared vertex
        migrating_vp_idx = None
        non_migrating_vp_idx = None
        
        vps_with_one_shared = [vp for vp, info in connectivity_analysis.items() if info['num_shared'] == 1]
        vps_with_zero_shared = [vp for vp, info in connectivity_analysis.items() if info['num_shared'] == 0]
        
        if len(vps_with_one_shared) == 1 and len(vps_with_zero_shared) == 1:
            # Perfect case: one VP shares 1 vertex, other shares 0
            migrating_vp_idx = vps_with_one_shared[0]
            non_migrating_vp_idx = vps_with_zero_shared[0]
            result['success'] = True
        elif len(vps_with_one_shared) == 2:
            # WARNING: Both VPs share a vertex with target edge
            self.logger.warning(f"⚠️  UNUSUAL: Both candidate VPs share a vertex with target edge!")
            self.logger.warning(f"    VP {candidate_vps[0]}: {connectivity_analysis[candidate_vps[0]]['shared_vertices']}")
            self.logger.warning(f"    VP {candidate_vps[1]}: {connectivity_analysis[candidate_vps[1]]['shared_vertices']}")
            self.logger.warning(f"    This requires special treatment - using first VP for now")
            migrating_vp_idx = candidate_vps[0]
            non_migrating_vp_idx = candidate_vps[1]
            result['success'] = True
            result['warning'] = "Both VPs share vertex with target edge"
        elif len(vps_with_one_shared) == 0:
            # WARNING: Neither VP shares a vertex with target edge
            self.logger.warning(f"⚠️  UNUSUAL: Neither candidate VP shares a vertex with target edge!")
            self.logger.warning(f"    VP {candidate_vps[0]}: {connectivity_analysis[candidate_vps[0]]['shared_vertices']}")
            self.logger.warning(f"    VP {candidate_vps[1]}: {connectivity_analysis[candidate_vps[1]]['shared_vertices']}")
            self.logger.warning(f"    This requires special treatment - using first VP for now")
            migrating_vp_idx = candidate_vps[0]
            non_migrating_vp_idx = candidate_vps[1]
            result['success'] = True
            result['warning'] = "No VP shares vertex with target edge"
        else:
            result['error'] = f"Unexpected connectivity pattern: {connectivity_analysis}"
            self.logger.error(result['error'])
            return result
        
        result['migrating_vp_idx'] = migrating_vp_idx
        result['non_migrating_vp_idx'] = non_migrating_vp_idx
        result['migrating_vp_edge'] = connectivity_analysis[migrating_vp_idx]['edge']
        result['non_migrating_vp_edge'] = connectivity_analysis[non_migrating_vp_idx]['edge']
        result['shared_vertex'] = connectivity_analysis[migrating_vp_idx]['shared_vertices'][0] if connectivity_analysis[migrating_vp_idx]['num_shared'] == 1 else None
        
        self.logger.info(f"✓ Selected migrating VP: {migrating_vp_idx}")
        self.logger.info(f"  Reason: Edge {result['migrating_vp_edge']} shares vertex with target edge {target_edge}")
        self.logger.info(f"  Non-migrating VP: {non_migrating_vp_idx}")
        
        return result
    
    def _identify_type2_migration_triangles(self, migration_result: Dict) -> Dict:
        """
        Identify all key triangles for Type 2 migration using new methodology.
        
        Triangles identified:
        1. T_second_VP: Triangle containing segment between direct and second-level neighbors
        2. T_adjacent_to_T_second: Triangle adjacent to T_second_VP via its free edge
        3. T_shared_edge_with_target: Triangle adjacent to target triangle via target edge
        
        Args:
            migration_result: Result dict from apply_type2_switch_v3() containing:
                - migrating_vp_idx
                - target_triangle_idx
                - target_edge
                
        Returns:
            Dict with triangle identification results:
            {
                'success': bool,
                'T_second_VP': int,
                'T_second_VP_free_edge': tuple,
                'T_adjacent_to_T_second': int,
                'T_shared_edge_with_target': int,
                'vp_context': {
                    'direct_outer_neighbor': int,
                    'second_level_neighbor': int
                }
            }
        """
        result = {
            'success': False,
            'error': None
        }
        
        migrating_vp_idx = migration_result['migrating_vp_idx']
        triple_vp_set = set(migration_result['triple_vp_indices'])
        target_triangle_idx = migration_result['target_triangle_idx']
        target_edge = migration_result['target_edge']
        
        self.logger.info(f"Identifying key triangles for Type 2 migration...")
        
        # Helper function to get neighbors
        def get_neighbors(vp_idx):
            neighbors = []
            for seg in self.partition.boundary_segments:
                if vp_idx == seg.vp_idx_1:
                    neighbors.append(seg.vp_idx_2)
                elif vp_idx == seg.vp_idx_2:
                    neighbors.append(seg.vp_idx_1)
            return neighbors
        
        # Step 1: Find direct outer neighbor (Level 1)
        all_neighbors = get_neighbors(migrating_vp_idx)
        direct_outer_neighbors = [vp for vp in all_neighbors if vp not in triple_vp_set]
        
        if len(direct_outer_neighbors) != 1:
            result['error'] = f"Expected 1 direct outer neighbor, found {len(direct_outer_neighbors)}: {direct_outer_neighbors}"
            self.logger.error(result['error'])
            return result
        
        direct_outer_neighbor = direct_outer_neighbors[0]
        self.logger.info(f"  Direct outer neighbor (Level 1): VP {direct_outer_neighbor}")
        
        # Step 2: Find second-level neighbor (Level 2)
        neighbor_neighbors = get_neighbors(direct_outer_neighbor)
        # Filter out migrating VP and all triple triangle VPs
        second_level_neighbors = [vp for vp in neighbor_neighbors 
                                 if vp != migrating_vp_idx and vp not in triple_vp_set]
        
        if len(second_level_neighbors) != 1:
            result['error'] = f"Expected 1 second-level neighbor, found {len(second_level_neighbors)}: {second_level_neighbors}"
            self.logger.error(result['error'])
            return result
        
        second_level_neighbor = second_level_neighbors[0]
        self.logger.info(f"  Second-level neighbor (Level 2): VP {second_level_neighbor}")
        
        result['vp_context'] = {
            'direct_outer_neighbor': direct_outer_neighbor,
            'second_level_neighbor': second_level_neighbor
        }
        
        # Step 3: Find T_second_VP (triangle containing segment Level1--Level2)
        seg_key = tuple(sorted([direct_outer_neighbor, second_level_neighbor]))
        T_second_VP = self.partition.segment_to_triangle.get(seg_key)
        
        if T_second_VP is None:
            result['error'] = f"Could not find triangle containing segment {seg_key}"
            self.logger.error(result['error'])
            return result
        
        result['T_second_VP'] = T_second_VP
        self.logger.info(f"  T_second_VP: {T_second_VP} (contains segment VP{direct_outer_neighbor}--VP{second_level_neighbor})")
        
        # Step 4: Find free edge in T_second_VP
        tri_vertices = self.mesh.faces[T_second_VP]
        tri_edges = [
            tuple(sorted([tri_vertices[0], tri_vertices[1]])),
            tuple(sorted([tri_vertices[1], tri_vertices[2]])),
            tuple(sorted([tri_vertices[2], tri_vertices[0]]))
        ]
        
        free_edge = None
        for edge in tri_edges:
            if edge not in self.partition.edge_to_varpoint:
                free_edge = edge
                break
        
        if free_edge is None:
            result['error'] = f"Could not find free edge in T_second_VP {T_second_VP}"
            self.logger.error(result['error'])
            return result
        
        result['T_second_VP_free_edge'] = free_edge
        self.logger.info(f"  T_second_VP free edge: {free_edge}")
        
        # Step 5: Find T_adjacent_to_T_second
        if free_edge not in self.mesh_topology.edge_to_triangles:
            result['error'] = f"Free edge {free_edge} not in mesh topology"
            self.logger.error(result['error'])
            return result
        
        adj_triangles = self.mesh_topology.edge_to_triangles[free_edge]
        T_adjacent_candidates = [tri for tri in adj_triangles if tri != T_second_VP]
        
        if len(T_adjacent_candidates) != 1:
            result['error'] = f"Expected 1 adjacent triangle, found {len(T_adjacent_candidates)}"
            self.logger.error(result['error'])
            return result
        
        T_adjacent_to_T_second = T_adjacent_candidates[0]
        result['T_adjacent_to_T_second'] = T_adjacent_to_T_second
        self.logger.info(f"  T_adjacent_to_T_second: {T_adjacent_to_T_second}")
        
        # Step 6: Find T_shared_edge_with_target
        if target_edge not in self.mesh_topology.edge_to_triangles:
            result['error'] = f"Target edge {target_edge} not in mesh topology"
            self.logger.error(result['error'])
            return result
        
        adj_to_target = self.mesh_topology.edge_to_triangles[target_edge]
        T_shared_candidates = [tri for tri in adj_to_target if tri != target_triangle_idx]
        
        if len(T_shared_candidates) != 1:
            result['error'] = f"Expected 1 triangle sharing target edge, found {len(T_shared_candidates)}"
            self.logger.error(result['error'])
            return result
        
        T_shared_edge_with_target = T_shared_candidates[0]
        result['T_shared_edge_with_target'] = T_shared_edge_with_target
        self.logger.info(f"  T_shared_edge_with_target: {T_shared_edge_with_target}")
        
        result['success'] = True
        self.logger.info("✓ Successfully identified all key triangles")
        
        return result
    
    # ============================================================================
    # Type 2 Migration Implementation (v4 - New Strategy)
    # ============================================================================
    
    def _identify_vp_close_to_steiner(
        self, 
        steiner_handler, 
        triple_point_idx: int
    ) -> Optional[int]:
        """
        Identify VP closest to Steiner point (vp_close_to_steiner).
        
        This VP sits on the shared edge between triple triangle and target triangle,
        and is the one the Steiner point is approaching.
        
        Strategy: Use Steiner point distances already computed in SteinerHandler.
        
        Args:
            steiner_handler: SteinerHandler object
            triple_point_idx: Index of triple point
            
        Returns:
            VP index closest to Steiner point, or None if error
        """
        if triple_point_idx >= len(steiner_handler.triple_points):
            self.logger.error(f"Invalid triple point index {triple_point_idx}")
            return None
        
        triple_point = steiner_handler.triple_points[triple_point_idx]
        vp_indices = triple_point.var_point_indices
        steiner_pt = triple_point.compute_steiner_point()
        
        # Compute distances from Steiner point to each VP
        vp_distances = {}
        for vp_idx in vp_indices:
            vp = self.partition.variable_points[vp_idx]
            vp_pos = self._get_vp_position(vp)
            dist = np.linalg.norm(vp_pos - steiner_pt)
            vp_distances[vp_idx] = dist
        
        # Find VP with minimum distance to Steiner point
        vp_close_to_steiner_idx = min(vp_distances.keys(), key=lambda k: vp_distances[k])
        min_dist = vp_distances[vp_close_to_steiner_idx]
        
        self.logger.info(f"VP close to Steiner: VP {vp_close_to_steiner_idx} "
                        f"(distance={min_dist:.6f})")
        
        return vp_close_to_steiner_idx
    
    def _get_vp_position(self, vp) -> np.ndarray:
        """
        Get 3D position of a variable point.
        
        Args:
            vp: VariablePoint object
            
        Returns:
            3D position (x, y, z)
        """
        edge = vp.edge
        lambda_param = vp.lambda_param
        
        # position = λ * edge[0] + (1-λ) * edge[1]
        v0 = self.mesh.vertices[edge[0]]
        v1 = self.mesh.vertices[edge[1]]
        
        return lambda_param * v0 + (1 - lambda_param) * v1
    
    def _identify_migrating_and_stationary_vps(
        self,
        triple_vp_indices: List[int],
        vp_close_to_steiner_idx: int,
        target_edge: Tuple[int, int]
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Identify migrating_VP and stationary_VP from triple triangle VPs.
        
        Selection criteria:
        - Candidates: The two VPs that are NOT vp_close_to_steiner
        - migrating_VP: VP whose edge shares a vertex with target_edge (topologically connected)
        - stationary_VP: The remaining VP
        
        Args:
            triple_vp_indices: List of 3 VP indices in triple triangle
            vp_close_to_steiner_idx: Index of vp_close_to_steiner
            target_edge: Target edge (free edge in target triangle)
            
        Returns:
            (migrating_vp_idx, stationary_vp_idx) or (None, None) if error
        """
        # Get the two candidate VPs (exclude vp_close_to_steiner)
        candidates = [vp_idx for vp_idx in triple_vp_indices 
                     if vp_idx != vp_close_to_steiner_idx]
        
        if len(candidates) != 2:
            self.logger.error(f"Expected 2 candidate VPs, found {len(candidates)}")
            return None, None
        
        # Check which candidate's edge shares a vertex with target_edge
        target_vertices = set(target_edge)
        
        migrating_vp_idx = None
        stationary_vp_idx = None
        
        for vp_idx in candidates:
            vp = self.partition.variable_points[vp_idx]
            vp_edge_vertices = set(vp.edge)
            
            # Check if edges share exactly 1 vertex
            shared_vertices = vp_edge_vertices & target_vertices
            
            if len(shared_vertices) == 1:
                # This VP's edge shares a vertex with target edge
                migrating_vp_idx = vp_idx
            elif len(shared_vertices) == 0:
                # This VP's edge does NOT share a vertex
                stationary_vp_idx = vp_idx
            else:
                # Should not happen: edge shares 2 vertices means same edge
                self.logger.warning(f"VP {vp_idx} edge {vp.edge} shares 2 vertices "
                                   f"with target edge {target_edge} - unusual!")
        
        # Verify we found both
        if migrating_vp_idx is None or stationary_vp_idx is None:
            self.logger.error(f"Failed to identify migrating/stationary VPs. "
                            f"Candidates: {candidates}, target_edge: {target_edge}")
            return None, None
        
        # WARNING check: Both VPs share a vertex with target edge (should not happen)
        vp1 = self.partition.variable_points[candidates[0]]
        vp2 = self.partition.variable_points[candidates[1]]
        if (set(vp1.edge) & target_vertices) and (set(vp2.edge) & target_vertices):
            self.logger.warning(f"⚠️  BOTH candidate VPs share vertices with target edge! "
                              f"VP {candidates[0]} edge {vp1.edge}, "
                              f"VP {candidates[1]} edge {vp2.edge}, "
                              f"target edge {target_edge}. Unusual topology!")
        
        self.logger.info(f"Migrating VP: {migrating_vp_idx}, Stationary VP: {stationary_vp_idx}")
        
        return migrating_vp_idx, stationary_vp_idx
    
    def _identify_outer_neighbors(
        self,
        migrating_vp_idx: int,
        triple_vp_indices: List[int]
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Identify first_level_VP and second_level_VP (outer neighbors of migrating_VP).
        
        Strategy:
        1. Find direct neighbor of migrating_VP (excluding triple triangle VPs) = first_level
        2. Find neighbor of first_level (excluding migrating_VP) = second_level
        
        Args:
            migrating_vp_idx: Index of migrating VP
            triple_vp_indices: List of 3 VP indices in triple triangle
            
        Returns:
            (first_level_vp_idx, second_level_vp_idx) or (None, None) if error
        """
        triple_vp_set = set(triple_vp_indices)
        
        # Step 1: Find first_level_VP (direct neighbor, excluding triple VPs)
        neighbors_of_migrating = []
        for segment in self.partition.boundary_segments:
            if segment.vp_idx_1 == migrating_vp_idx:
                neighbor = segment.vp_idx_2
                if neighbor not in triple_vp_set:
                    neighbors_of_migrating.append(neighbor)
            elif segment.vp_idx_2 == migrating_vp_idx:
                neighbor = segment.vp_idx_1
                if neighbor not in triple_vp_set:
                    neighbors_of_migrating.append(neighbor)
        
        if len(neighbors_of_migrating) != 1:
            self.logger.error(f"Expected 1 outer neighbor for migrating VP {migrating_vp_idx}, "
                            f"found {len(neighbors_of_migrating)}: {neighbors_of_migrating}")
            return None, None
        
        first_level_vp_idx = neighbors_of_migrating[0]
        
        # Step 2: Find second_level_VP (neighbor of first_level, excluding migrating_VP)
        neighbors_of_first_level = []
        for segment in self.partition.boundary_segments:
            if segment.vp_idx_1 == first_level_vp_idx:
                neighbor = segment.vp_idx_2
                if neighbor != migrating_vp_idx and neighbor not in triple_vp_set:
                    neighbors_of_first_level.append(neighbor)
            elif segment.vp_idx_2 == first_level_vp_idx:
                neighbor = segment.vp_idx_1
                if neighbor != migrating_vp_idx and neighbor not in triple_vp_set:
                    neighbors_of_first_level.append(neighbor)
        
        if len(neighbors_of_first_level) != 1:
            self.logger.error(f"Expected 1 second-level neighbor for first_level VP {first_level_vp_idx}, "
                            f"found {len(neighbors_of_first_level)}: {neighbors_of_first_level}")
            return None, None
        
        second_level_vp_idx = neighbors_of_first_level[0]
        
        self.logger.info(f"Outer neighbors: first_level={first_level_vp_idx}, "
                        f"second_level={second_level_vp_idx}")
        
        return first_level_vp_idx, second_level_vp_idx
    
    def _find_common_vertex_all_triangles(
        self,
        triple_tri: int,
        target_tri: int,
        t_first: int,
        t_second: int,
        t_adjacent: int,
        t_shared: int
    ) -> Optional[int]:
        """
        Find vertex shared by ALL 6 triangles.
        
        This vertex forms a "fan" of triangles around it and will change cell
        during migration (like target_vertex in Type 1).
        
        Args:
            triple_tri: Triple point triangle
            target_tri: Target triangle
            t_first: T_first_VP triangle
            t_second: T_second_VP triangle
            t_adjacent: T_adjacent_to_T_second triangle
            t_shared: T_shared_edge_with_target triangle
            
        Returns:
            Common vertex index, or None if not found
        """
        triangles = [triple_tri, target_tri, t_first, t_second, t_adjacent, t_shared]
        
        # Get vertices from first triangle
        face0 = self.mesh.faces[triangles[0]]
        candidate_vertices = set(face0)
        
        # Check which vertices appear in ALL triangles
        for tri_idx in triangles[1:]:
            face = self.mesh.faces[tri_idx]
            candidate_vertices &= set(face)
        
        if len(candidate_vertices) == 0:
            self.logger.error(f"No common vertex found among 6 triangles")
            return None
        
        if len(candidate_vertices) > 1:
            self.logger.warning(f"Multiple common vertices found: {candidate_vertices}. "
                              f"Using first one.")
        
        common_vertex = int(list(candidate_vertices)[0])
        
        # Verify using _get_all_triangles_at_vertex
        triangles_at_vertex = self._get_all_triangles_at_vertex(common_vertex)
        
        all_present = all(t in triangles_at_vertex for t in triangles)
        
        if not all_present:
            missing = [t for t in triangles if t not in triangles_at_vertex]
            self.logger.warning(f"Common vertex {common_vertex} not present in all triangles. "
                              f"Missing: {missing}")
        
        self.logger.info(f"Common vertex (all 6 triangles): {common_vertex}")
        
        return common_vertex
    
    def _determine_cells_for_edge(self, edge: Tuple[int, int]) -> Set[int]:
        """
        Determine which two cells are separated by an edge.
        
        An edge separates exactly the cells of its two endpoint vertices.
        This is the most direct and correct approach, especially after cell flips.
        
        Args:
            edge: Edge tuple (v1, v2)
            
        Returns:
            Set of cell indices separated by this edge (typically 2, or 1 if interior)
        """
        # Get vertex labels directly from the edge vertices
        vertex_labels = np.argmax(self.partition.indicator_functions, axis=1)
        
        v1, v2 = edge
        cell1 = int(vertex_labels[v1])
        cell2 = int(vertex_labels[v2])
        
        # Edge separates the cells of its two vertices
        cells = {cell1, cell2}
        
        if len(cells) == 1:
            # Both vertices have same cell - edge is interior to that cell
            # This shouldn't happen for edges with VPs, but handle gracefully
            self.logger.debug(f"Edge {edge} has both vertices in same cell {cells}")
        
        self.logger.debug(f"Edge {edge} separates cells: {cells}")
        
        return cells
    
    def _create_steiner_vp(
        self,
        old_edge: Tuple[int, int],
        old_lambda: float
    ) -> int:
        """
        Create new VP (steiner_VP) at the exact position where vp_close_to_steiner was.
        
        Args:
            old_edge: Edge where vp_close_to_steiner was before moving
            old_lambda: Lambda value vp_close_to_steiner had before moving
            
        Returns:
            Index of newly created steiner_VP
        """
        from src.core.contour_partition import VariablePoint
        
        # Determine which cells are separated by this edge
        cells_on_edge = self._determine_cells_for_edge(old_edge)
        
        # Create new VariablePoint
        steiner_VP = VariablePoint(
            edge=old_edge,
            lambda_param=old_lambda,
            global_idx=len(self.partition.variable_points),  # New index
            belongs_to_cells=cells_on_edge
        )
        
        # Append to partition
        self.partition.variable_points.append(steiner_VP)
        steiner_VP_idx = steiner_VP.global_idx
        
        # Update edge_to_varpoint (old_edge now has steiner_VP)
        self.partition.edge_to_varpoint[old_edge] = steiner_VP_idx
        
        self.logger.info(f"Created steiner_VP {steiner_VP_idx} at edge {old_edge} "
                        f"with λ={old_lambda:.6f}, cells={cells_on_edge}")
        
        return steiner_VP_idx
    
    def apply_type2_switch_v4(
        self,
        steiner_handler,
        triple_point_idx: int,
        distance_preservation: str = 'preserve'
    ) -> Dict:
        """
        Apply Type 2 triple point collapse migration (NEW STRATEGY).
        
        Converts Steiner point into a new VP, migrates VPs, and forms new triple point.
        
        14-Step Process:
        Phase 1: Identification (Steps 1-6)
        Phase 2: VP Migrations (Steps 7-10)
        Phase 3: Cell Update (Step 11)
        Phase 4: Segment Updates (Steps 12-14)
        
        Args:
            steiner_handler: SteinerHandler object
            triple_point_idx: Index of triple point to collapse
            distance_preservation: 'preserve', 'midpoint', or custom distance string
            
        Returns:
            Dict with migration result:
            {
                'success': bool,
                'vp_count_change': int (+1),
                'segment_count_change': int (+1),
                'steiner_vp_idx': int,
                'common_vertex': int,
                ...
            }
        """
        self.logger.info("="*80)
        self.logger.info(f"TYPE 2 MIGRATION v4 (Triple Point {triple_point_idx})")
        self.logger.info("="*80)
        
        result = {
            'success': False,
            'vp_count_change': 0,
            'segment_count_change': 0
        }
        
        # Validate triple point
        if triple_point_idx >= len(steiner_handler.triple_points):
            self.logger.error(f"Invalid triple point index {triple_point_idx}")
            return result
        
        triple_point = steiner_handler.triple_points[triple_point_idx]
        triple_triangle_idx = triple_point.triangle_idx
        triple_vp_indices = triple_point.var_point_indices
        
        self.logger.info(f"Triple triangle: {triple_triangle_idx}")
        self.logger.info(f"Triple VPs: {triple_vp_indices}")
        
        # ========================================================================
        # PHASE 1: IDENTIFICATION (Steps 1-6)
        # ========================================================================
        
        # Step 1: Identify vp_close_to_steiner
        vp_close_to_steiner_idx = self._identify_vp_close_to_steiner(
            steiner_handler, triple_point_idx
        )
        if vp_close_to_steiner_idx is None:
            return result
        
        vp_close_to_steiner = self.partition.variable_points[vp_close_to_steiner_idx]
        shared_edge = vp_close_to_steiner.edge
        
        # Step 2: Identify target triangle and target edge
        triangles_on_shared_edge = self.mesh_topology.edge_to_triangles.get(shared_edge, [])
        if len(triangles_on_shared_edge) != 2:
            self.logger.error(f"Shared edge {shared_edge} not shared by exactly 2 triangles")
            return result
        
        target_triangle_idx = [t for t in triangles_on_shared_edge if t != triple_triangle_idx][0]
        
        # Find target edge (free edge in target triangle)
        target_face = self.mesh.faces[target_triangle_idx]
        target_edges = [
            tuple(sorted([target_face[0], target_face[1]])),
            tuple(sorted([target_face[1], target_face[2]])),
            tuple(sorted([target_face[2], target_face[0]]))
        ]
        
        target_edge = None
        for edge in target_edges:
            if edge not in self.partition.edge_to_varpoint:
                target_edge = edge
                break
        
        if target_edge is None:
            self.logger.error(f"No free edge found in target triangle {target_triangle_idx}")
            return result
        
        self.logger.info(f"Target triangle: {target_triangle_idx}")
        self.logger.info(f"Target edge: {target_edge}")
        
        # Step 3: Identify migrating_VP and stationary_VP
        migrating_vp_idx, stationary_vp_idx = self._identify_migrating_and_stationary_vps(
            triple_vp_indices, vp_close_to_steiner_idx, target_edge
        )
        if migrating_vp_idx is None or stationary_vp_idx is None:
            return result
        
        # Step 4: Identify outer neighbors
        first_level_vp_idx, second_level_vp_idx = self._identify_outer_neighbors(
            migrating_vp_idx, triple_vp_indices
        )
        if first_level_vp_idx is None or second_level_vp_idx is None:
            return result
        
        # Step 5: Identify triangles (using segment_to_triangle map)
        seg_migrating_first = tuple(sorted([migrating_vp_idx, first_level_vp_idx]))
        seg_first_second = tuple(sorted([first_level_vp_idx, second_level_vp_idx]))
        
        T_first_VP_idx = self.partition.segment_to_triangle.get(seg_migrating_first)
        T_second_VP_idx = self.partition.segment_to_triangle.get(seg_first_second)
        
        if T_first_VP_idx is None or T_second_VP_idx is None:
            self.logger.error(f"Failed to find T_first_VP or T_second_VP using segment_to_triangle map")
            return result
        
        # Find T_second_VP free edge
        face_t_second = self.mesh.faces[T_second_VP_idx]
        edges_t_second = [
            tuple(sorted([face_t_second[0], face_t_second[1]])),
            tuple(sorted([face_t_second[1], face_t_second[2]])),
            tuple(sorted([face_t_second[2], face_t_second[0]]))
        ]
        
        t_second_free_edge = None
        for edge in edges_t_second:
            if edge not in self.partition.edge_to_varpoint:
                t_second_free_edge = edge
                break
        
        if t_second_free_edge is None:
            self.logger.error(f"No free edge in T_second_VP {T_second_VP_idx}")
            return result
        
        # Find T_adjacent_to_T_second
        triangles_on_t_second_free = self.mesh_topology.edge_to_triangles.get(t_second_free_edge, [])
        T_adjacent_to_T_second_idx = [t for t in triangles_on_t_second_free if t != T_second_VP_idx]
        if len(T_adjacent_to_T_second_idx) != 1:
            self.logger.error(f"Could not find unique T_adjacent_to_T_second")
            return result
        T_adjacent_to_T_second_idx = T_adjacent_to_T_second_idx[0]
        
        # Find T_shared_edge_with_target
        triangles_on_target_edge = self.mesh_topology.edge_to_triangles.get(target_edge, [])
        T_shared_edge_with_target_idx = [t for t in triangles_on_target_edge if t != target_triangle_idx]
        if len(T_shared_edge_with_target_idx) != 1:
            self.logger.error(f"Could not find unique T_shared_edge_with_target")
            return result
        T_shared_edge_with_target_idx = T_shared_edge_with_target_idx[0]
        
        self.logger.info(f"T_first_VP: {T_first_VP_idx}")
        self.logger.info(f"T_second_VP: {T_second_VP_idx}")
        self.logger.info(f"T_adjacent_to_T_second: {T_adjacent_to_T_second_idx}")
        self.logger.info(f"T_shared_edge_with_target: {T_shared_edge_with_target_idx}")
        
        # Step 6: Find common_vertex_all_triangles
        common_vertex = self._find_common_vertex_all_triangles(
            triple_triangle_idx, target_triangle_idx,
            T_first_VP_idx, T_second_VP_idx,
            T_adjacent_to_T_second_idx, T_shared_edge_with_target_idx
        )
        if common_vertex is None:
            return result
        
        result['common_vertex'] = common_vertex
        
        # ========================================================================
        # PHASE 2: VP MIGRATIONS (Steps 7-10)
        # ========================================================================
        
        self.logger.info("-" * 80)
        self.logger.info("PHASE 2: VP MIGRATIONS")
        self.logger.info("-" * 80)
        
        # Step 7: Move vp_close_to_steiner to target edge
        old_edge_vp_close = vp_close_to_steiner.edge
        old_lambda_vp_close = vp_close_to_steiner.lambda_param
        
        self.logger.info(f"Step 7: Moving vp_close_to_steiner {vp_close_to_steiner_idx} "
                        f"from {old_edge_vp_close} to target edge {target_edge}")
        
        new_lambda_vp_close = 0.5  # Can be optimized later
        self._move_variable_point(vp_close_to_steiner_idx, target_edge, new_lambda_vp_close)
        
        # Step 8: Create steiner_VP at old position
        self.logger.info(f"Step 8: Creating steiner_VP at {old_edge_vp_close} λ={old_lambda_vp_close:.6f}")
        steiner_vp_idx = self._create_steiner_vp(old_edge_vp_close, old_lambda_vp_close)
        result['steiner_vp_idx'] = steiner_vp_idx
        result['vp_count_change'] = 1
        
        # Step 9: Move migrating_VP along mesh line (opposite to common_vertex)
        self.logger.info(f"Step 9: Moving migrating_VP {migrating_vp_idx} opposite to vertex {common_vertex}")
        
        migrating_vp = self.partition.variable_points[migrating_vp_idx]
        target_edge_migrating = self._find_opposite_edge(migrating_vp.edge, common_vertex)
        
        if target_edge_migrating is None:
            self.logger.error(f"Could not find opposite edge for migrating VP")
            return result
        
        if distance_preservation == 'preserve':
            dist_migrating = self._compute_distance_to_vertex(migrating_vp_idx, common_vertex)
            lambda_migrating = self._compute_lambda_for_distance(
                target_edge_migrating, common_vertex, dist_migrating
            )
        else:
            lambda_migrating = 0.5
        
        self._move_variable_point(migrating_vp_idx, target_edge_migrating, lambda_migrating)
        
        # Step 10: Move first_level_VP to free edge in T_second_VP
        self.logger.info(f"Step 10: Moving first_level_VP {first_level_vp_idx} "
                        f"to free edge in T_second_VP")
        
        if distance_preservation == 'preserve':
            dist_first_level = self._compute_distance_to_vertex(first_level_vp_idx, common_vertex)
            lambda_first_level = self._compute_lambda_for_distance(
                t_second_free_edge, common_vertex, dist_first_level
            )
        else:
            lambda_first_level = 0.5
        
        self._move_variable_point(first_level_vp_idx, t_second_free_edge, lambda_first_level)
        
        # ========================================================================
        # PHASE 3: CELL UPDATE (Step 11)
        # ========================================================================
        
        self.logger.info("-" * 80)
        self.logger.info("PHASE 3: CELL UPDATE")
        self.logger.info("-" * 80)
        
        # Step 11: Determine cell flip BEFORE migration for common_vertex
        # But we need to do this AFTER determining common_vertex but BEFORE moving VPs
        # Since we already moved VPs, we'll use the migrating VP's original cell info
        
        old_cell, new_cell = self._determine_target_vertex_cell_flip(
            common_vertex, migrating_vp_idx
        )
        
        self._update_indicator_functions_for_target_vertex(
            common_vertex, old_cell, new_cell
        )
        
        self.logger.info(f"Step 11: Common vertex {common_vertex} flipped: "
                        f"cell {old_cell} → cell {new_cell}")
        
        # Step 11b: Update belongs_to_cells for all moved VPs
        # CRITICAL: This must happen AFTER the cell flip (Step 11) so that
        # _determine_cells_for_edge() uses the FINAL indicator_functions state
        self.logger.info(f"Step 11b: Updating belongs_to_cells for moved VPs")
        
        moved_vps = [
            (vp_close_to_steiner_idx, "vp_close_to_steiner"),
            (migrating_vp_idx, "migrating_VP"),
            (first_level_vp_idx, "first_level_VP"),
            (steiner_vp_idx, "steiner_VP")  # Also verify steiner_VP
        ]
        
        for vp_idx, vp_name in moved_vps:
            vp = self.partition.variable_points[vp_idx]
            old_cells = vp.belongs_to_cells.copy()
            new_cells = self._determine_cells_for_edge(vp.edge)
            vp.belongs_to_cells = new_cells
            
            if old_cells != new_cells:
                self.logger.info(f"  {vp_name} (VP{vp_idx}): "
                                f"belongs_to_cells updated {old_cells} → {new_cells}")
            else:
                self.logger.debug(f"  {vp_name} (VP{vp_idx}): "
                                 f"belongs_to_cells unchanged {new_cells}")
        
        # ========================================================================
        # PHASE 4: SEGMENT UPDATES (Steps 12-14)
        # ========================================================================
        
        self.logger.info("-" * 80)
        self.logger.info("PHASE 4: SEGMENT UPDATES")
        self.logger.info("-" * 80)
        
        # Step 12: Delete void triangle segments
        self.logger.info(f"Step 12: Deleting 2 void triangle segments")
        
        seg1_key = tuple(sorted([migrating_vp_idx, stationary_vp_idx]))
        seg2_key = tuple(sorted([vp_close_to_steiner_idx, stationary_vp_idx]))
        segments_to_remove = [seg1_key, seg2_key]
        
        original_seg_count = len(self.partition.boundary_segments)
        
        self.partition.boundary_segments = [
            seg for seg in self.partition.boundary_segments
            if tuple(sorted([seg.vp_idx_1, seg.vp_idx_2])) not in segments_to_remove
        ]
        
        deleted_count = original_seg_count - len(self.partition.boundary_segments)
        self.logger.info(f"Deleted {deleted_count} segments")
        
        # Step 13: Create new segments for new triple point
        self.logger.info(f"Step 13: Creating 3 new segments for new triple point")
        
        # Need to find outer_neighbor_of_vp_close_to_steiner in target triangle
        # This is a neighbor of vp_close_to_steiner that is NOT in triple_vp_indices
        outer_neighbor_vp_close_idx = None
        for segment in self.partition.boundary_segments:
            if segment.vp_idx_1 == vp_close_to_steiner_idx:
                neighbor = segment.vp_idx_2
                if neighbor not in triple_vp_indices:
                    outer_neighbor_vp_close_idx = neighbor
                    break
            elif segment.vp_idx_2 == vp_close_to_steiner_idx:
                neighbor = segment.vp_idx_1
                if neighbor not in triple_vp_indices:
                    outer_neighbor_vp_close_idx = neighbor
                    break
        
        if outer_neighbor_vp_close_idx is None:
            self.logger.error(f"Could not find outer_neighbor of vp_close_to_steiner")
            return result
        
        from src.core.contour_partition import BoundarySegment
        
        # Helper to determine cell pair
        def determine_seg_cells(vp_idx1, vp_idx2):
            vp1 = self.partition.variable_points[vp_idx1]
            vp2 = self.partition.variable_points[vp_idx2]
            cells = vp1.belongs_to_cells & vp2.belongs_to_cells
            if len(cells) != 2:
                cells = vp1.belongs_to_cells if len(vp1.belongs_to_cells) == 2 else vp2.belongs_to_cells
            return tuple(sorted(cells))
        
        new_segments = [
            BoundarySegment(
                vp_idx_1=steiner_vp_idx,
                vp_idx_2=vp_close_to_steiner_idx,
                cell_pair=determine_seg_cells(steiner_vp_idx, vp_close_to_steiner_idx),
                segment_type="normal"
            ),
            BoundarySegment(
                vp_idx_1=outer_neighbor_vp_close_idx,
                vp_idx_2=steiner_vp_idx,
                cell_pair=determine_seg_cells(outer_neighbor_vp_close_idx, steiner_vp_idx),
                segment_type="normal"
            ),
            BoundarySegment(
                vp_idx_1=steiner_vp_idx,
                vp_idx_2=stationary_vp_idx,
                cell_pair=determine_seg_cells(steiner_vp_idx, stationary_vp_idx),
                segment_type="normal"
            )
        ]
        
        self.partition.boundary_segments.extend(new_segments)
        
        created_count = len(new_segments)
        new_seg_count = len(self.partition.boundary_segments)
        result['segment_count_change'] = new_seg_count - original_seg_count
        
        self.logger.info(f"Created {created_count} new segments (net change: {result['segment_count_change']})")
        
        # Step 14: Rebuild data structures
        self.logger.info(f"Step 14: Rebuilding data structures")
        
        # Collect affected triangles
        affected_triangles = [
            triple_triangle_idx, target_triangle_idx,
            T_first_VP_idx, T_second_VP_idx,
            T_adjacent_to_T_second_idx, T_shared_edge_with_target_idx
        ]
        
        # Add triangles at common_vertex
        common_vertex_triangles = self._get_all_triangles_at_vertex(common_vertex)
        affected_triangles.extend(common_vertex_triangles)
        affected_triangles = list(set(affected_triangles))  # Deduplicate
        
        # Rebuild triangle_segments (optimized)
        self.partition.rebuild_triangle_segments_for_affected_triangles(affected_triangles)
        self.logger.info(f"Rebuilt triangle_segments for {len(affected_triangles)} affected triangles")
        
        # Rebuild segment_to_triangle map
        self.partition.rebuild_segment_to_triangle_map()
        self.logger.info(f"Rebuilt segment_to_triangle map")
        
        # ========================================================================
        # DEBUG: Inspect rebuilt triangle_segments for the 6 methodology triangles
        # ========================================================================
        self.logger.info("="*80)
        self.logger.info("DEBUG: Inspecting rebuilt triangle_segments (6 methodology triangles)")
        self.logger.info("="*80)
        
        methodology_triangles = [
            triple_triangle_idx, target_triangle_idx,
            T_first_VP_idx, T_second_VP_idx,
            T_adjacent_to_T_second_idx, T_shared_edge_with_target_idx
        ]
        
        for tri_idx in sorted(methodology_triangles):
            # Find the triangle_segment for this triangle
            tri_segs = [ts for ts in self.partition.triangle_segments if ts.triangle_idx == tri_idx]
            
            if tri_segs:
                ts = tri_segs[0]
                self.logger.info(f"Triangle {tri_idx}:")
                self.logger.info(f"  VP indices: {ts.var_point_indices}")
                self.logger.info(f"  Boundary edges: {ts.boundary_edges}")
                self.logger.info(f"  Vertex labels: {ts.vertex_labels}")
                
                # Check if VPs are actually on those edges
                for vp_idx in ts.var_point_indices:
                    vp = self.partition.variable_points[vp_idx]
                    self.logger.info(f"    VP {vp_idx}: edge={vp.edge}, λ={vp.lambda_param:.6f}")
                    
                    # Verify edge is in boundary_edges
                    if vp.edge not in ts.boundary_edges:
                        self.logger.warning(f"      ⚠️  VP edge {vp.edge} NOT in triangle's boundary_edges!")
            else:
                self.logger.warning(f"Triangle {tri_idx}: NO triangle_segment found!")
        
        self.logger.info("="*80)
        
        # ========================================================================
        # DEBUG: Check edge_to_varpoint consistency for migrated VPs
        # ========================================================================
        self.logger.info("DEBUG: Checking edge_to_varpoint consistency")
        self.logger.info("="*80)
        
        key_vps = [
            (vp_close_to_steiner_idx, "vp_close_to_steiner"),
            (steiner_vp_idx, "steiner_VP"),
            (migrating_vp_idx, "migrating_VP"),
            (first_level_vp_idx, "first_level_VP"),
            (stationary_vp_idx, "stationary_VP")
        ]
        
        for vp_idx, vp_name in key_vps:
            vp = self.partition.variable_points[vp_idx]
            mapped_vp = self.partition.edge_to_varpoint.get(vp.edge)
            
            if mapped_vp == vp_idx:
                self.logger.info(f"✓ {vp_name} (VP{vp_idx}): edge {vp.edge} → VP{mapped_vp}")
            else:
                self.logger.error(f"✗ {vp_name} (VP{vp_idx}): edge {vp.edge} → VP{mapped_vp} (MISMATCH!)")
        
        self.logger.info("="*80)
        
        # ========================================================================
        # DEBUG: Verify new segments
        # ========================================================================
        self.logger.info("DEBUG: New segments verification")
        self.logger.info("="*80)
        
        for i, seg in enumerate(new_segments):
            vp1 = self.partition.variable_points[seg.vp_idx_1]
            vp2 = self.partition.variable_points[seg.vp_idx_2]
            self.logger.info(f"New segment {i+1}: VP{seg.vp_idx_1}--VP{seg.vp_idx_2}")
            self.logger.info(f"  VP{seg.vp_idx_1}: edge={vp1.edge}, λ={vp1.lambda_param:.6f}")
            self.logger.info(f"  VP{seg.vp_idx_2}: edge={vp2.edge}, λ={vp2.lambda_param:.6f}")
            self.logger.info(f"  Cells: {seg.cell_pair}")
        
        self.logger.info("="*80)
        
        # ========================================================================
        # COMPLETE
        # ========================================================================
        
        result['success'] = True
        result['affected_triangles'] = affected_triangles
        result['triple_triangle_idx'] = triple_triangle_idx
        result['target_triangle_idx'] = target_triangle_idx
        result['vp_close_to_steiner_idx'] = vp_close_to_steiner_idx
        result['migrating_vp_idx'] = migrating_vp_idx
        result['stationary_vp_idx'] = stationary_vp_idx
        result['steiner_vp_idx'] = steiner_vp_idx
        result['outer_neighbor_vp_close_idx'] = outer_neighbor_vp_close_idx
        
        self.logger.info("="*80)
        self.logger.info(f"✓ TYPE 2 MIGRATION COMPLETE")
        self.logger.info(f"  VP count change: +{result['vp_count_change']}")
        self.logger.info(f"  Segment count change: +{result['segment_count_change']}")
        self.logger.info(f"  New triple point in triangle: {target_triangle_idx}")
        self.logger.info("="*80)
        
        return result

