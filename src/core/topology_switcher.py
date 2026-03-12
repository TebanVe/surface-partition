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
    from .contour_partition import PartitionContour, VariablePoint, BoundarySegment
    from .mesh_topology import MeshTopology
    from .steiner_handler import TriplePoint, SteinerHandler
    from . import migration_utils
    from .type1_component_analyzer import Type1ComponentAnalyzer
    from .type2_migration_history import Type2MigrationHistory
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from logging_config import get_logger
    from core.tri_mesh import TriMesh
    from core.contour_partition import PartitionContour, VariablePoint, BoundarySegment
    from core.mesh_topology import MeshTopology
    from core.steiner_handler import TriplePoint, SteinerHandler
    import migration_utils
    from type1_component_analyzer import Type1ComponentAnalyzer
    from type2_migration_history import Type2MigrationHistory


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
        
        # Initialize Type 2 migration history (empty by default)
        # This will be populated by the workflow (e.g., refine_perimeter_iterative.py)
        self.type2_migration_history = Type2MigrationHistory()
        
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
        Wrapper for migration_utils.identify_target_vertex().
        Kept for backward compatibility.
        """
        return migration_utils.identify_target_vertex(vp)
    
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
    
    def _edges_share_triangle(self, edge1: Tuple[int, int], edge2: Tuple[int, int]) -> bool:
        """Check if two edges are on the same mesh triangle."""
        edge1_norm = tuple(sorted(edge1))
        edge2_norm = tuple(sorted(edge2))
        
        triangles1 = self.mesh_topology.get_triangles_sharing_edge(edge1_norm)
        triangles2 = self.mesh_topology.get_triangles_sharing_edge(edge2_norm)
        
        # Convert to sets for intersection
        return bool(set(triangles1) & set(triangles2))
    
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
        Wrapper for migration_utils.compute_boundary_distance().
        Kept for backward compatibility.
        """
        vp = self.partition.variable_points[vp_idx]
        return migration_utils.compute_boundary_distance(vp)
    
    # =============================================================================
    # Component Analysis Methods - Delegated to Type1ComponentAnalyzer
    # =============================================================================
    # These methods are kept as wrappers for backward compatibility
    # but delegate to the Type1ComponentAnalyzer module
    
    def _get_analyzer(self) -> Type1ComponentAnalyzer:
        """Create analyzer instance (lazy initialization pattern)."""
        return Type1ComponentAnalyzer(self.mesh, self.partition, self.mesh_topology)
    
    def find_connected_components(self, boundary_vps_set: set) -> List[set]:
        """Wrapper - delegates to Type1ComponentAnalyzer."""
        return self._get_analyzer().find_connected_components(boundary_vps_set)
    
    def analyze_component(self, component_vps: Set[int], boundary_tol: float = 0.1) -> Dict:
        """Wrapper - delegates to Type1ComponentAnalyzer."""
        return self._get_analyzer().analyze_component(component_vps, boundary_tol=boundary_tol)
    
    def detect_proximity_conflicts(self, components: List[Dict], boundary_tol: float = 0.1) -> Tuple[List[Dict], List[Dict]]:
        """Wrapper - delegates to Type1ComponentAnalyzer."""
        return self._get_analyzer().detect_proximity_conflicts(components, boundary_tol=boundary_tol)
    
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
        Wrapper for migration_utils.get_two_neighbors().
        Kept for backward compatibility.
        """
        return migration_utils.get_two_neighbors(self.partition, vp_idx)
    
    def _construct_auxiliary_component_2vp(self, component: Dict,
                                           strict_validation: bool = True) -> List[int]:
        """Wrapper - delegates to Type1ComponentAnalyzer."""
        return self._get_analyzer()._construct_auxiliary_component_2vp(component, strict_validation)
    
    def _construct_auxiliary_component_1vp(self, component: Dict, 
                                           strict_validation: bool = True) -> List[int]:
        """Wrapper - delegates to Type1ComponentAnalyzer."""
        return self._get_analyzer()._construct_auxiliary_component_1vp(component, strict_validation)
    
    def _get_neighbors_from_auxiliary(self, migrating_vp_idx: int, 
                                      auxiliary_component: List[int]) -> Tuple[int, int]:
        """Wrapper - delegates to Type1ComponentAnalyzer."""
        return self._get_analyzer()._get_neighbors_from_auxiliary(migrating_vp_idx, auxiliary_component)
    
    def select_migrating_vp_and_auxiliary(self, component: Dict,
                                          strict_validation: bool = True) -> Tuple[int, List[int]]:
        """Wrapper - delegates to Type1ComponentAnalyzer."""
        return self._get_analyzer().select_migrating_vp_and_auxiliary(component, strict_validation)
    
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
            Lambda value in [0, 1] that achieves target_distance,
            or -1.0 if the edge does not contain target_vertex (signals caller to abort)
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
            self.logger.warning(f"New edge {new_edge} does not contain target vertex {target_vertex}, "
                              f"aborting VP placement (would create non-boundary edge)")
            return -1.0
    
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
                # Verify the free edge contains the target vertex
                if target_vertex is not None and target_vertex not in edge_norm:
                    self.logger.warning(f"Free edge {edge_norm} in triangle {target_triangle} "
                                      f"does not contain target vertex {target_vertex} — "
                                      f"skipping (would place VP on non-boundary edge)")
                    continue
                
                # Compute lambda to preserve distance to target vertex
                if target_vertex is not None:
                    lambda_param = self._compute_lambda_for_distance(edge_norm, target_vertex, target_distance)
                    if lambda_param < 0:
                        self.logger.warning(f"Neighbor VP {neighbor_vp_idx}: "
                                          f"_compute_lambda_for_distance returned sentinel for edge {edge_norm}")
                        return False
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
        - boundary_segments: VP-VP connectivity changes when VPs move, but this is
          corrected by rebuild_triangle_segments_from_current_vps (force_rebuild=True)
          which runs in reinitialize_after_switches after all migrations complete.
        
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
        
        # Step 2: Rebuild edge_to_varpoint from the authoritative source (vp.edge)
        # Migrations (especially reverse) can leave this dict out of sync with
        # actual VP edges. Rebuilding is O(n_vps) and guarantees consistency.
        rebuilt_map = {}
        for vp_idx, vp in enumerate(self.partition.variable_points):
            edge_norm = tuple(sorted(vp.edge))
            if edge_norm in rebuilt_map:
                self.logger.warning(
                    f"Duplicate edge {edge_norm}: VP {rebuilt_map[edge_norm]} and VP {vp_idx}")
            rebuilt_map[edge_norm] = vp_idx
        self.partition.edge_to_varpoint = rebuilt_map
        
        self.logger.debug(f"Data structures updated after migration: "
                         f"{len(self.partition.triangle_segments)} triangle segments, "
                         f"{len(self.partition.boundary_segments)} boundary segments, "
                         f"{len(rebuilt_map)} edge_to_varpoint entries")
    
    def apply_type1_switch_v2(self, component: Dict, 
                              distance_preservation: str = 'preserve',
                              strict_validation: bool = True,
                              migrating_vp: int = None,
                              auxiliary_component: List[int] = None,
                              left_neighbor: int = None,
                              right_neighbor: int = None) -> bool:
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
            migrating_vp: Pre-computed migrating VP index (optional). If provided, skips
                         VP selection step. Improves performance when calling in batch.
            auxiliary_component: Pre-computed auxiliary component (optional). Must be provided
                               if migrating_vp is provided.
            left_neighbor: Pre-computed left neighbor (optional). Must be provided if
                          migrating_vp is provided.
            right_neighbor: Pre-computed right neighbor (optional). Must be provided if
                           migrating_vp is provided.
            
        Returns:
            Dict with migration results (see docstring end) or False for backward compatibility
        """
        component_vps = component['vp_indices']
        component_size = len(component_vps)
        
        if not component_vps:
            self.logger.warning("Empty component - cannot migrate")
            return {'success': False, 'error': 'Empty component'}
        
        # Check if pre-computed parameters are provided
        use_precomputed = (migrating_vp is not None and 
                          auxiliary_component is not None and
                          left_neighbor is not None and
                          right_neighbor is not None)
        
        if use_precomputed:
            # Use pre-computed values (efficient path for batch migrations)
            self.logger.debug("="*80)
            self.logger.debug(f"STARTING TYPE 1 MIGRATION (using pre-computed plan)")
            self.logger.debug(f"Component: {component_vps} (size: {component_size})")
            self.logger.debug(f"Target vertex: {component.get('target_vertex', 'unknown')}")
            self.logger.debug(f"Migrating VP: {migrating_vp}")
            self.logger.debug("="*80)
            
            migrating_vp_idx = migrating_vp
            
        else:
            # Compute values on-the-fly (backward compatibility for visualization scripts)
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
        
        # Common path continues here (whether pre-computed or computed on-the-fly)
        migrating_vp_obj = self.partition.variable_points[migrating_vp_idx]
        old_edge = migrating_vp_obj.edge
        
        # Step 2.5: Validate migration trio
        log_fn = self.logger.debug if use_precomputed else self.logger.info
        log_fn("Step 2.5: Validating migration trio...")
        target_vertex = self._identify_target_vertex(migrating_vp_obj)
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
            log_fn(f"✓ Validation PASSED: All VPs approach target vertex {target_vertex}")
        
        # Step 3: Continue with migration
        log_fn(f"Step 3: Proceeding with migration...")
        
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
        
        # Save original VP state for rollback if neighbor adjustment fails
        vp_migrating = self.partition.variable_points[migrating_vp_idx]
        vp_left = self.partition.variable_points[left_neighbor]
        vp_right = self.partition.variable_points[right_neighbor]
        orig_migrating = (vp_migrating.edge, vp_migrating.lambda_param)
        orig_left = (vp_left.edge, vp_left.lambda_param)
        orig_right = (vp_right.edge, vp_right.lambda_param)
        
        # Step 6: Move migrating VP to target edge with preserved distance
        lambda_migrating = self._compute_lambda_for_distance(target_edge, target_vertex, dist_migrating)
        if lambda_migrating < 0:
            self.logger.warning(f"VP {migrating_vp_idx}: target edge {target_edge} does not contain "
                              f"target vertex {target_vertex}, aborting migration")
            return {'success': False, 'error': 'Migrating VP target edge invalid'}
        self._move_variable_point(migrating_vp_idx, target_edge, lambda_migrating)
        self.logger.debug(f"VP {migrating_vp_idx}: Moved to edge {target_edge} with λ={lambda_migrating:.6f}")
        
        # Step 6: Adjust left neighbor (move to free edge in triangle containing its segment with its OTHER neighbor)
        if not self._adjust_neighbor_to_free_edge(left_neighbor, migrating_vp_idx, 
                                                  target_vertex, dist_left):
            self.logger.warning(f"Left neighbor {left_neighbor} adjustment failed — rolling back migration")
            self._move_variable_point(migrating_vp_idx, orig_migrating[0], orig_migrating[1])
            return {'success': False, 'error': f'Left neighbor adjustment failed (rolled back)'}
        
        # Step 8: Adjust right neighbor (move to free edge in triangle containing its segment with its OTHER neighbor)
        if not self._adjust_neighbor_to_free_edge(right_neighbor, migrating_vp_idx,
                                                  target_vertex, dist_right):
            self.logger.warning(f"Right neighbor {right_neighbor} adjustment failed — rolling back migration")
            self._move_variable_point(migrating_vp_idx, orig_migrating[0], orig_migrating[1])
            self._move_variable_point(left_neighbor, orig_left[0], orig_left[1])
            return {'success': False, 'error': f'Right neighbor adjustment failed (rolled back)'}
        
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
        
        if use_precomputed:
            self.logger.debug("="*80)
            self.logger.debug(f"✓ TYPE 1 MIGRATION COMPLETED SUCCESSFULLY")
            self.logger.debug(f"  Migrated VP: {migrating_vp_idx}")
            self.logger.debug(f"  Neighbors: {left_neighbor}, {right_neighbor}")
            self.logger.debug(f"  Target vertex: {target_vertex}")
            self.logger.debug(f"  Component size: {component_size}")
            self.logger.debug("="*80)
        else:
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
    
    def can_form_valid_auxiliary_component(self, component: Dict) -> Tuple[bool, str]:
        """Wrapper - delegates to Type1ComponentAnalyzer."""
        return self._get_analyzer().can_form_valid_auxiliary_component(component)
    
    def select_components_for_migration(self, components: List[Dict], 
                                        conflicts: List[Dict],
                                        conflict_strategy: str = 'exclude_one') -> Tuple[List[Dict], List[Dict]]:
        """Wrapper - delegates to Type1ComponentAnalyzer."""
        return self._get_analyzer().select_components_for_migration(components, conflicts, conflict_strategy)
    
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
    
    def _find_triangle_in_fan(
        self,
        common_vertex: int,
        already_assigned: List[int],
        must_share_edge_with: List[int]
    ) -> Optional[int]:
        """
        Find triangle in the fan around common_vertex that shares edges with specified triangles.
        
        This is used to find T_adjacent_to_T_second by ensuring it:
        1. Contains the common vertex (part of the fan)
        2. Shares an edge with T_second_VP (adjacent in the fan)
        3. Shares an edge with T_shared_edge_with_target (completes the fan)
        
        Args:
            common_vertex: The vertex all triangles in the fan should contain
            already_assigned: Triangles already assigned (exclude these)
            must_share_edge_with: List of triangle indices - candidate must share edge with ALL
            
        Returns:
            Triangle index, or None if not found
        """
        # Get all triangles at common_vertex
        triangles_at_vertex = self._get_all_triangles_at_vertex(common_vertex)
        
        # Get edges for triangles we must share with
        required_edge_sets = []
        for tri_idx in must_share_edge_with:
            face = self.mesh.faces[tri_idx]
            edges = [
                tuple(sorted([face[0], face[1]])),
                tuple(sorted([face[1], face[2]])),
                tuple(sorted([face[2], face[0]]))
            ]
            required_edge_sets.append(set(edges))
        
        # Find candidate triangle
        for tri_idx in triangles_at_vertex:
            if tri_idx in already_assigned:
                continue
            
            face = self.mesh.faces[tri_idx]
            tri_edges = set([
                tuple(sorted([face[0], face[1]])),
                tuple(sorted([face[1], face[2]])),
                tuple(sorted([face[2], face[0]]))
            ])
            
            # Check if shares edge with ALL required triangles
            shares_all = all(
                len(tri_edges & req_edges) > 0 
                for req_edges in required_edge_sets
            )
            
            if shares_all:
                self.logger.info(f"Found triangle {tri_idx} in fan (shares edges with {must_share_edge_with})")
                return tri_idx
        
        self.logger.error(f"No triangle found in fan around vertex {common_vertex} that shares edges with {must_share_edge_with}")
        return None
    
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
        print("="*80)
        print(f"🔥 TYPE 2 MIGRATION v4 - CODE VERSION 2026-01-26-FIX 🔥")
        print(f"TYPE 2 MIGRATION v4 (Triple Point {triple_point_idx})")
        print("="*80)
        self.logger.info("="*80)
        self.logger.info(f"🔥 TYPE 2 MIGRATION v4 - CODE VERSION 2026-01-26-FIX 🔥")
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
        
        # ===================================================================
        # CHECK FOR REVERSE MIGRATION (NEW!)
        # ===================================================================
        
        reverse_info = self.type2_migration_history.check_for_reverse(
            triple_triangle_idx,
            target_triangle_idx
        )
        
        if reverse_info is not None:
            original_triangle, target_index = reverse_info
            print("="*80)
            print("🔄 REVERSE MIGRATION DETECTED!")
            print("="*80)
            print(f"Triple point wants to return to triangle {target_triangle_idx}")
            print(f"This reverses previous migration(s)")
            print(f"Current path: {self.type2_migration_history.records[original_triangle].triangle_sequence}")
            print(f"Will truncate to index {target_index}")
            print("="*80)
            
            self.logger.info("="*80)
            self.logger.info("🔄 REVERSE MIGRATION DETECTED!")
            self.logger.info("="*80)
            self.logger.info(f"Triple point wants to return to triangle {target_triangle_idx}")
            self.logger.info(f"This reverses previous migration(s)")
            self.logger.info(f"Current path: {self.type2_migration_history.records[original_triangle].triangle_sequence}")
            self.logger.info(f"Will truncate to index {target_index}")
            self.logger.info("="*80)
            
            return self._execute_reverse_migration(
                original_triangle,
                target_index,
                distance_preservation
            )
        
        # Not a reverse - proceed with forward migration
        print("Forward migration (not a reversal)")
        self.logger.info("Forward migration (not a reversal)")
        
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
        
        # Step 3b: Compute common vertex (shared by migrating_VP edge and target_edge)
        # This is the anchor vertex that all 6 triangles should share
        common_vertex = None
        migrating_vp = self.partition.variable_points[migrating_vp_idx]
        for v in migrating_vp.edge:
            if v in target_edge:
                common_vertex = v
                break
        
        if common_vertex is None:
            self.logger.error(f"Migrating VP edge {migrating_vp.edge} shares no vertex with target edge {target_edge}")
            return result
        
        self.logger.info(f"Common vertex (migration anchor): {common_vertex}")
        result['common_vertex'] = common_vertex
        
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
        
        # Find T_shared_edge_with_target first (needed for T_adjacent_to_T_second)
        print("="*80)
        print("PHASE 1B: TRIANGLE IDENTIFICATION (v4 fan-based logic)")
        print("="*80)
        self.logger.info("="*80)
        self.logger.info("PHASE 1B: TRIANGLE IDENTIFICATION (v4 fan-based logic)")
        self.logger.info("="*80)
        
        triangles_on_target_edge = self.mesh_topology.edge_to_triangles.get(target_edge, [])
        T_shared_edge_with_target_idx = [t for t in triangles_on_target_edge if t != target_triangle_idx]
        if len(T_shared_edge_with_target_idx) != 1:
            self.logger.error(f"Could not find unique T_shared_edge_with_target")
            return result
        T_shared_edge_with_target_idx = T_shared_edge_with_target_idx[0]
        print(f"T_shared_edge_with_target: {T_shared_edge_with_target_idx}")
        self.logger.info(f"T_shared_edge_with_target: {T_shared_edge_with_target_idx}")
        
        # Find T_adjacent_to_T_second using fan geometry around common_vertex
        # It must share edges with both T_second_VP and T_shared_edge_with_target
        print(f"Finding T_adjacent_to_T_second using fan geometry around common_vertex {common_vertex}")
        print(f"Must share edges with: T_second_VP={T_second_VP_idx}, T_shared_edge_with_target={T_shared_edge_with_target_idx}")
        self.logger.info(f"Finding T_adjacent_to_T_second using fan geometry around common_vertex {common_vertex}")
        self.logger.info(f"Must share edges with: T_second_VP={T_second_VP_idx}, T_shared_edge_with_target={T_shared_edge_with_target_idx}")
        
        T_adjacent_to_T_second_idx = self._find_triangle_in_fan(
            common_vertex=common_vertex,
            already_assigned=[
                triple_triangle_idx, 
                target_triangle_idx,
                T_first_VP_idx, 
                T_second_VP_idx,
                T_shared_edge_with_target_idx
            ],
            must_share_edge_with=[T_second_VP_idx, T_shared_edge_with_target_idx]
        )
        
        if T_adjacent_to_T_second_idx is None:
            print(f"ERROR: Could not find T_adjacent_to_T_second in fan around vertex {common_vertex}")
            self.logger.error(f"Could not find T_adjacent_to_T_second in fan around vertex {common_vertex}")
            return result
        
        print(f"✓ T_first_VP: {T_first_VP_idx}")
        print(f"✓ T_second_VP: {T_second_VP_idx}")
        print(f"✓ T_adjacent_to_T_second: {T_adjacent_to_T_second_idx}")
        print(f"✓ T_shared_edge_with_target: {T_shared_edge_with_target_idx}")
        self.logger.info(f"✓ T_first_VP: {T_first_VP_idx}")
        self.logger.info(f"✓ T_second_VP: {T_second_VP_idx}")
        self.logger.info(f"✓ T_adjacent_to_T_second: {T_adjacent_to_T_second_idx}")
        self.logger.info(f"✓ T_shared_edge_with_target: {T_shared_edge_with_target_idx}")
        
        # Step 6: Validate common_vertex is in all 6 triangles (sanity check)
        validation_common_vertex = self._find_common_vertex_all_triangles(
            triple_triangle_idx, target_triangle_idx,
            T_first_VP_idx, T_second_VP_idx,
            T_adjacent_to_T_second_idx, T_shared_edge_with_target_idx
        )
        if validation_common_vertex is None:
            self.logger.error(f"Validation failed: computed common_vertex {common_vertex} not in all 6 triangles")
            return result
        
        if validation_common_vertex != common_vertex:
            self.logger.error(
                f"Validation failed: computed common_vertex {common_vertex} != "
                f"validation result {validation_common_vertex}"
            )
            return result
        
        self.logger.info(f"Validated common_vertex {common_vertex} present in all 6 triangles")
        
        # ========================================================================
        # PHASE 2: VP MIGRATIONS (Steps 7-10)
        # ========================================================================
        
        self.logger.info("-" * 80)
        self.logger.info("PHASE 2: VP MIGRATIONS")
        self.logger.info("-" * 80)
        
        # ========================================================================
        # CAPTURE VP STATE FOR HISTORY (before any moves)
        # ========================================================================
        
        # Record state of all VPs that will be moved (for future reversal)
        vp_state_record = {
            'created_vp_idx': None,  # Will be set after Step 8
            'common_vertex': common_vertex,
            'moved_vps': {}
        }
        
        # Capture vp_close_to_steiner state
        vp_close = self.partition.variable_points[vp_close_to_steiner_idx]
        old_dist_vp_close = self._compute_distance_to_vertex(vp_close_to_steiner_idx, common_vertex)
        vp_state_record['moved_vps'][vp_close_to_steiner_idx] = {
            'old_edge': vp_close.edge,
            'old_lambda': vp_close.lambda_param,
            'old_distance_to_common': old_dist_vp_close
        }
        
        # Capture migrating_VP state
        mig_vp = self.partition.variable_points[migrating_vp_idx]
        old_dist_migrating = self._compute_distance_to_vertex(migrating_vp_idx, common_vertex)
        vp_state_record['moved_vps'][migrating_vp_idx] = {
            'old_edge': mig_vp.edge,
            'old_lambda': mig_vp.lambda_param,
            'old_distance_to_common': old_dist_migrating
        }
        
        # Capture first_level_VP state
        first_vp = self.partition.variable_points[first_level_vp_idx]
        old_dist_first = self._compute_distance_to_vertex(first_level_vp_idx, common_vertex)
        vp_state_record['moved_vps'][first_level_vp_idx] = {
            'old_edge': first_vp.edge,
            'old_lambda': first_vp.lambda_param,
            'old_distance_to_common': old_dist_first
        }
        
        self.logger.debug(f"Captured VP states for reversal (before migration)")
        
        # Step 7: Move vp_close_to_steiner to target edge
        old_edge_vp_close = vp_close_to_steiner.edge
        old_lambda_vp_close = vp_close_to_steiner.lambda_param
        
        self.logger.info(f"Step 7: Moving vp_close_to_steiner {vp_close_to_steiner_idx} "
                        f"from {old_edge_vp_close} to target edge {target_edge}")
        
        # Use distance_preservation parameter (same as Steps 9 and 10)
        if distance_preservation == 'preserve':
            dist_vp_close = self._compute_distance_to_vertex(vp_close_to_steiner_idx, common_vertex)
            new_lambda_vp_close = self._compute_lambda_for_distance(
                target_edge, common_vertex, dist_vp_close
            )
            self.logger.debug(f"Preserving distance to common vertex {common_vertex}: {dist_vp_close:.6f}")
        elif distance_preservation == 'midpoint':
            new_lambda_vp_close = 0.5
            self.logger.debug(f"Using midpoint placement (λ=0.5)")
        else:
            # Custom distance provided as string
            try:
                new_lambda_vp_close = float(distance_preservation)
                self.logger.debug(f"Using custom distance {new_lambda_vp_close:.6f}")
            except ValueError:
                self.logger.warning(f"Invalid distance_preservation value '{distance_preservation}', using midpoint")
                new_lambda_vp_close = 0.5
        
        self._move_variable_point(vp_close_to_steiner_idx, target_edge, new_lambda_vp_close)
        
        # Step 8: Create steiner_VP at old position
        self.logger.info(f"Step 8: Creating steiner_VP at {old_edge_vp_close} λ={old_lambda_vp_close:.6f}")
        steiner_vp_idx = self._create_steiner_vp(old_edge_vp_close, old_lambda_vp_close)
        result['steiner_vp_idx'] = steiner_vp_idx
        result['vp_count_change'] = 1
        
        # Record steiner VP index for reversal
        vp_state_record['created_vp_idx'] = steiner_vp_idx
        
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
            self.logger.debug(f"Preserving distance to common vertex {common_vertex}: {dist_migrating:.6f}")
        elif distance_preservation == 'midpoint':
            lambda_migrating = 0.5
            self.logger.debug(f"Using midpoint placement (λ=0.5)")
        else:
            # Custom distance provided as string
            try:
                lambda_migrating = float(distance_preservation)
                self.logger.debug(f"Using custom distance {lambda_migrating:.6f}")
            except ValueError:
                self.logger.warning(f"Invalid distance_preservation value '{distance_preservation}', using midpoint")
                lambda_migrating = 0.5
        
        self._move_variable_point(migrating_vp_idx, target_edge_migrating, lambda_migrating)
        
        # Step 10: Move first_level_VP to free edge in T_second_VP
        self.logger.info(f"Step 10: Moving first_level_VP {first_level_vp_idx} "
                        f"to free edge in T_second_VP")
        
        # Find the free edge in T_second_VP
        face_t_second = self.mesh.faces[T_second_VP_idx]
        print(f"DEBUG: T_second_VP {T_second_VP_idx} vertices: {face_t_second}")
        edges_t_second = [
            tuple(sorted([face_t_second[0], face_t_second[1]])),
            tuple(sorted([face_t_second[1], face_t_second[2]])),
            tuple(sorted([face_t_second[2], face_t_second[0]]))
        ]
        
        self.logger.debug(f"T_second_VP {T_second_VP_idx} edges: {edges_t_second}")
        self.logger.debug(f"Common vertex: {common_vertex}")
        print(f"DEBUG: T_second_VP edges: {edges_t_second}")
        print(f"DEBUG: Common vertex: {common_vertex}")
        print(f"DEBUG: edge_to_varpoint map:")
        for edge in edges_t_second:
            has_vp = edge in self.partition.edge_to_varpoint
            contains_common = common_vertex in edge
            print(f"  Edge {edge}: has_VP={has_vp}, contains_common_vertex={contains_common}")
        
        # First, find ALL free edges (no VPs)
        free_edges = []
        for edge in edges_t_second:
            if edge not in self.partition.edge_to_varpoint:
                contains_common = common_vertex in edge
                free_edges.append((edge, contains_common))
                self.logger.debug(f"  Free edge {edge}: contains common_vertex={contains_common}")
        
        if not free_edges:
            self.logger.error(f"No free edge found in T_second_VP {T_second_VP_idx}")
            return result
        
        # Prefer edge containing common_vertex, but fallback to any free edge
        t_second_free_edge = None
        for edge, contains_common in free_edges:
            if contains_common:
                t_second_free_edge = edge
                self.logger.info(f"  Selected free edge {t_second_free_edge} (contains common_vertex {common_vertex})")
                print(f"  Selected free edge {t_second_free_edge} (contains common_vertex {common_vertex})")
                break
        
        if t_second_free_edge is None:
            # Fallback: use first free edge (even if it doesn't contain common_vertex)
            t_second_free_edge = free_edges[0][0]
            self.logger.warning(
                f"  No free edge contains common_vertex {common_vertex}. "
                f"Using free edge {t_second_free_edge} anyway."
            )
            print(f"  WARNING: No free edge contains common_vertex {common_vertex}. Using free edge {t_second_free_edge} anyway.")
        
        if distance_preservation == 'preserve':
            dist_first_level = self._compute_distance_to_vertex(first_level_vp_idx, common_vertex)
            lambda_first_level = self._compute_lambda_for_distance(
                t_second_free_edge, common_vertex, dist_first_level
            )
            self.logger.debug(f"Preserving distance to common vertex {common_vertex}: {dist_first_level:.6f}")
        elif distance_preservation == 'midpoint':
            lambda_first_level = 0.5
            self.logger.debug(f"Using midpoint placement (λ=0.5)")
        else:
            # Custom distance provided as string
            try:
                lambda_first_level = float(distance_preservation)
                self.logger.debug(f"Using custom distance {lambda_first_level:.6f}")
            except ValueError:
                self.logger.warning(f"Invalid distance_preservation value '{distance_preservation}', using midpoint")
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
        
        # Record cell flip in vp_state_record so reverse migrations can undo it
        vp_state_record['old_cell'] = old_cell
        vp_state_record['new_cell'] = new_cell
        
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
                self.logger.info(f"Triangle {tri_idx}: NO triangle_segment found (triangle transitioned from boundary to interior - expected after migration)")
        
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
        
        # ===================================================================
        # RECORD FORWARD MIGRATION IN HISTORY
        # ===================================================================
        
        if self.type2_migration_history.current_iteration is not None:
            self.type2_migration_history.record_forward_migration(
                current_triangle=triple_triangle_idx,
                target_triangle=target_triangle_idx,
                iteration=self.type2_migration_history.current_iteration,
                vp_record=vp_state_record
            )
            self.logger.info(f"Recorded forward migration in history: {triple_triangle_idx} → {target_triangle_idx}")
            print(f"Recorded forward migration in history: {triple_triangle_idx} → {target_triangle_idx}")
        else:
            self.logger.warning("Migration history current_iteration not set - migration not recorded")
        
        return result
    
    # ========================================================================
    # TYPE 2 REVERSE MIGRATION METHODS
    # ========================================================================
    
    def _execute_reverse_migration(
        self,
        original_triangle: int,
        target_index: int,
        distance_preservation: str
    ) -> Dict:
        """
        Reverse one or more Type 2 migrations.
        
        Moves VPs back to their old edges (preserving current distance to common vertex)
        and deletes the steiner VPs that were created during forward migrations.
        
        Args:
            original_triangle: Original triangle key in history
            target_index: Index to truncate to (reverse all migrations after this)
            distance_preservation: How to place VPs on old edges ('preserve', 'midpoint', etc.)
        
        Returns:
            Result dict with success=True, vp_count_change (negative), reversed=True
        """
        record = self.type2_migration_history.records[original_triangle]
        
        # Calculate how many migrations to reverse
        current_index = len(record.triangle_sequence) - 1
        num_to_reverse = current_index - target_index
        
        print(f"Reversing {num_to_reverse} migration(s)")
        self.logger.info(f"Reversing {num_to_reverse} migration(s)")
        
        total_vp_change = 0
        
        # Reverse in LIFO order (most recent first)
        for i in range(num_to_reverse):
            migration_index = len(record.vp_records) - 1 - i
            vp_record = record.vp_records[migration_index]
            
            from_tri = record.triangle_sequence[migration_index]
            to_tri = record.triangle_sequence[migration_index + 1]
            
            print(f"  Reversing migration {migration_index}: {from_tri} → {to_tri}")
            self.logger.info(f"  Reversing migration {migration_index}: {from_tri} → {to_tri}")
            
            vp_change = self._reverse_single_migration(vp_record, distance_preservation)
            total_vp_change += vp_change
        
        # Truncate history
        record.truncate_to_index(target_index)
        print(f"Updated path: {record.triangle_sequence}")
        self.logger.info(f"Updated path: {record.triangle_sequence}")
        
        return {
            'success': True,
            'vp_count_change': total_vp_change,  # Negative (deleted VPs)
            'segment_count_change': total_vp_change,  # Same as VP change
            'reversed': True,
            'num_reversed': num_to_reverse,
            'final_triangle': record.get_current_triangle()
        }
    
    def _reverse_single_migration(
        self,
        vp_record: Dict,
        distance_preservation: str
    ) -> int:
        """
        Reverse a single Type 2 migration.
        
        Moves VPs back to their old edges, reverses the indicator_functions
        cell flip, updates belongs_to_cells, and deletes the steiner VP.
        Data structures are updated after each reversal (incremental updates).
        
        Args:
            vp_record: Dict containing:
                - created_vp_idx: VP that was created (to delete)
                - moved_vps: Dict of {vp_idx: {old_edge, old_lambda, old_distance_to_common}}
                - common_vertex: The fan center vertex
                - old_cell: Cell before the forward migration's flip (added 2026-03-12)
                - new_cell: Cell after the forward migration's flip (added 2026-03-12)
            distance_preservation: 'preserve' or 'midpoint'
        
        Returns:
            VP count change (should be -1 for deleting steiner VP)
        """
        common_vertex = vp_record['common_vertex']
        
        print(f"    Common vertex: {common_vertex}")
        print(f"    Moving {len(vp_record['moved_vps'])} VPs back")
        self.logger.info(f"    Common vertex: {common_vertex}")
        self.logger.info(f"    Moving {len(vp_record['moved_vps'])} VPs back")
        
        # 1. Move all VPs back to their old edges
        for vp_idx, vp_data in vp_record['moved_vps'].items():
            old_edge = vp_data['old_edge']
            old_lambda = vp_data['old_lambda']
            old_distance = vp_data['old_distance_to_common']
            
            # Compute NEW lambda based on distance preservation strategy
            if distance_preservation == 'preserve':
                # Preserve CURRENT distance to common_vertex
                current_distance = self._compute_distance_to_vertex(vp_idx, common_vertex)
                new_lambda = self._compute_lambda_for_distance(
                    old_edge,
                    common_vertex,
                    current_distance  # Use current, not old
                )
                print(f"      VP {vp_idx}: {old_edge}, λ={new_lambda:.6f} (preserving distance {current_distance:.6f})")
                self.logger.info(f"      VP {vp_idx}: {old_edge}, λ={new_lambda:.6f} (preserving distance {current_distance:.6f})")
            elif distance_preservation == 'midpoint':
                new_lambda = 0.5
                print(f"      VP {vp_idx}: {old_edge}, λ=0.5 (midpoint)")
                self.logger.info(f"      VP {vp_idx}: {old_edge}, λ=0.5 (midpoint)")
            else:
                # Use old lambda as fallback
                new_lambda = old_lambda
                print(f"      VP {vp_idx}: {old_edge}, λ={new_lambda:.6f} (old lambda)")
                self.logger.info(f"      VP {vp_idx}: {old_edge}, λ={new_lambda:.6f} (old lambda)")
            
            self._move_variable_point(vp_idx, old_edge, new_lambda)
        
        # 2. Delete the steiner VP that was created
        created_vp_idx = vp_record['created_vp_idx']
        print(f"    Deleting steiner VP {created_vp_idx}")
        self.logger.info(f"    Deleting steiner VP {created_vp_idx}")
        self._delete_variable_point(created_vp_idx)
        
        # 3. Reverse the indicator_functions cell flip for common_vertex
        # The forward migration flipped common_vertex from old_cell → new_cell,
        # so the reverse must flip it back: new_cell → old_cell.
        if 'old_cell' in vp_record and 'new_cell' in vp_record:
            fwd_old_cell = vp_record['old_cell']
            fwd_new_cell = vp_record['new_cell']
            self._update_indicator_functions_for_target_vertex(
                common_vertex, fwd_new_cell, fwd_old_cell
            )
            print(f"    Reversed indicator_functions: vertex {common_vertex} "
                  f"flipped back cell {fwd_new_cell} → {fwd_old_cell}")
            self.logger.info(f"    Reversed indicator_functions: vertex {common_vertex} "
                           f"flipped back cell {fwd_new_cell} → {fwd_old_cell}")
        else:
            # Fallback for records created before this fix: derive from current state
            self.logger.warning(
                f"    vp_record missing old_cell/new_cell (pre-fix history). "
                f"Deriving cell flip from current indicator_functions."
            )
            vertex_labels = np.argmax(self.partition.indicator_functions, axis=1)
            current_cell = int(vertex_labels[common_vertex])
            # The common_vertex should border exactly the cells of the moved VPs
            neighbor_cells = set()
            for vp_idx in vp_record['moved_vps']:
                if vp_idx < len(self.partition.variable_points):
                    vp = self.partition.variable_points[vp_idx]
                    neighbor_cells.update(vp.belongs_to_cells)
            neighbor_cells.discard(current_cell)
            if len(neighbor_cells) == 1:
                target_cell = neighbor_cells.pop()
                self._update_indicator_functions_for_target_vertex(
                    common_vertex, current_cell, target_cell
                )
                print(f"    Reversed indicator_functions (derived): vertex {common_vertex} "
                      f"flipped cell {current_cell} → {target_cell}")
                self.logger.info(f"    Reversed indicator_functions (derived): vertex {common_vertex} "
                               f"flipped cell {current_cell} → {target_cell}")
            else:
                self.logger.error(
                    f"    Could not derive cell flip for vertex {common_vertex}: "
                    f"current_cell={current_cell}, neighbor_cells={neighbor_cells}"
                )
        
        # 4. Update belongs_to_cells for all moved VPs (after indicator_functions fix)
        for vp_idx in vp_record['moved_vps']:
            if vp_idx < len(self.partition.variable_points):
                vp = self.partition.variable_points[vp_idx]
                new_cells = self._determine_cells_for_edge(vp.edge)
                vp.belongs_to_cells = new_cells
        
        # 5. Update all data structures (now indicator_functions is correct)
        print(f"    Updating data structures")
        self.logger.info(f"    Updating data structures")
        self.update_data_structures_after_migration()
        
        return -1  # One VP deleted
    
    def _delete_variable_point(self, vp_idx: int):
        """
        Delete a variable point and update all affected data structures.
        
        CRITICAL: This method handles VP index shifting. When a VP is deleted,
        all VPs with higher indices must be shifted down by 1.
        
        Steps:
        1. Remove VP from edge_to_varpoint mapping
        2. Remove VP from variable_points list
        3. Update all VP indices > vp_idx in:
           - edge_to_varpoint
           - boundary_segments
           - triangle_segments (via boundary_segments)
        
        Args:
            vp_idx: Index of VP to delete
        """
        if vp_idx >= len(self.partition.variable_points):
            self.logger.error(f"Cannot delete VP {vp_idx}: index out of range")
            return
        
        vp = self.partition.variable_points[vp_idx]
        edge = vp.edge
        
        self.logger.debug(f"Deleting VP {vp_idx} on edge {edge}")
        
        # 1. Remove from edge_to_varpoint
        if edge in self.partition.edge_to_varpoint:
            if self.partition.edge_to_varpoint[edge] == vp_idx:
                del self.partition.edge_to_varpoint[edge]
            else:
                self.logger.warning(f"VP {vp_idx} not found at edge {edge} in edge_to_varpoint")
        
        # 2. Delete from variable_points list
        del self.partition.variable_points[vp_idx]
        
        # 3. Shift all higher indices down by 1
        self.logger.debug(f"Shifting VP indices > {vp_idx} down by 1")
        
        # Update edge_to_varpoint
        updated_edge_to_varpoint = {}
        for e, idx in self.partition.edge_to_varpoint.items():
            if idx > vp_idx:
                updated_edge_to_varpoint[e] = idx - 1
            else:
                updated_edge_to_varpoint[e] = idx
        self.partition.edge_to_varpoint = updated_edge_to_varpoint
        
        # Update boundary_segments (will be rebuilt by update_data_structures_after_migration)
        # But we need to shift indices now to avoid inconsistencies
        for segment in self.partition.boundary_segments:
            if segment.vp_idx_1 > vp_idx:
                segment.vp_idx_1 -= 1
            if segment.vp_idx_2 > vp_idx:
                segment.vp_idx_2 -= 1
        
        # Update triangle_segments (will also be rebuilt, but shift now)
        for tri_seg in self.partition.triangle_segments:
            updated_vp_indices = []
            for idx in tri_seg.var_point_indices:
                if idx == vp_idx:
                    self.logger.warning(f"Deleted VP {vp_idx} found in triangle_segment for triangle {tri_seg.triangle_idx}")
                    continue  # Skip deleted VP
                elif idx > vp_idx:
                    updated_vp_indices.append(idx - 1)
                else:
                    updated_vp_indices.append(idx)
            tri_seg.var_point_indices = updated_vp_indices
        
        self.logger.debug(f"VP {vp_idx} deleted, {len(self.partition.variable_points)} VPs remaining")

