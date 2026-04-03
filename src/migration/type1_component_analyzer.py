"""
Type 1 Component Analysis Module

This self-contained module handles the complete analysis pipeline for Type 1
migration candidates:
1. Finding connected components of boundary VPs
2. Analyzing each component (target vertex, distances, neighbors)
3. Constructing auxiliary components (for 1-VP and 2-VP components)
4. Detecting proximity conflicts between components
5. Selecting which components to migrate

The module provides detailed logging that matches the output in
visualize_type1_vertex_collapse.py for transparency and debugging.
"""

import numpy as np
from typing import List, Tuple, Optional, Set, Dict
from collections import defaultdict, Counter, deque

from ..logging_config import get_logger
from ..mesh.tri_mesh import TriMesh
from ..partition.contour_partition import PartitionContour, VariablePoint
from ..mesh.mesh_topology import MeshTopology
from ..partition.steiner_handler import SteinerHandler
from . import migration_utils


class Type1ComponentAnalyzer:
    """
    Analyzer for Type 1 migration component analysis.
    
    Encapsulates all logic for finding, analyzing, and selecting components
    for Type 1 vertex-collapse migration.
    """
    
    def __init__(self, mesh: TriMesh, partition: PartitionContour, 
                 mesh_topology: MeshTopology, steiner_handler=None):
        """
        Initialize the component analyzer.
        
        Args:
            mesh: TriMesh instance
            partition: PartitionContour instance
            mesh_topology: MeshTopology instance
            steiner_handler: Optional pre-created SteinerHandler (for efficiency)
        """
        self.mesh = mesh
        self.partition = partition
        self.mesh_topology = mesh_topology
        self.logger = get_logger(__name__)
        
        # Cache for triple point VPs
        if steiner_handler is not None:
            # Extract triple point VPs from provided SteinerHandler
            self._triple_point_vps_cache = set()
            for tp in steiner_handler.triple_points:
                self._triple_point_vps_cache.update(tp.var_point_indices)
        else:
            # Will create SteinerHandler when needed
            self._triple_point_vps_cache = None
    
    # =========================================================================
    # Component Finding
    # =========================================================================
    
    def find_connected_components(self, boundary_vps_set: set) -> List[set]:
        """
        Find connected components of boundary VPs, grouped by target vertex.
        
        This method first groups VPs by which vertex they approach (based on lambda),
        then finds connected components within each target vertex group. This ensures
        that components are geometrically consistent (all VPs approach the same vertex).
        
        Args:
            boundary_vps_set: Set of boundary VP indices
            
        Returns:
            List of sets, each set is a connected component with consistent target vertex
        """
        # Step 1: Group VPs by target vertex (geometric criterion)
        by_target = defaultdict(set)
        
        for vp_idx in boundary_vps_set:
            vp = self.partition.variable_points[vp_idx]
            target = migration_utils.identify_target_vertex(vp)
            if target is not None:
                by_target[target].add(vp_idx)
            else:
                self.logger.warning(f"VP {vp_idx}: Cannot identify target vertex (skipping)")
        
        # Step 2: Within each target vertex group, find connected components (topological criterion)
        all_components = []
        
        for target_vertex, vp_set in by_target.items():
            # Find connected components within this target vertex group
            sub_components = self._find_connected_components_topology(vp_set)
            all_components.extend(sub_components)
        
        return all_components
    
    def _find_connected_components_topology(self, boundary_vps_set: set) -> List[set]:
        """
        Find connected components via DFS on boundary_segments (topology only).
        
        This is a private helper method that finds connected components based purely
        on boundary segment connectivity, without considering target vertex.
        
        Args:
            boundary_vps_set: Set of boundary VP indices
            
        Returns:
            List of sets, each set is a topologically connected component
        """
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
        
        Uses caching to avoid recreating SteinerHandler for every component.
        
        Returns:
            Set of VP indices that belong to triple point triangles
        """
        # Return cached value if available
        if self._triple_point_vps_cache is not None:
            return self._triple_point_vps_cache
        
        # Create SteinerHandler once and cache result
        steiner_handler = SteinerHandler(self.mesh, self.partition)
        triple_point_vps = set()
        for tp in steiner_handler.triple_points:
            triple_point_vps.update(tp.var_point_indices)
        
        self._triple_point_vps_cache = triple_point_vps
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
    
    def analyze_component(self, component_vps: Set[int], boundary_tol: float = 0.1) -> Dict:
        """
        Analyze a component and extract metadata.
        
        Args:
            component_vps: Set of VP indices in the component
            boundary_tol: Tolerance for boundary VP detection
            
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
        boundary_vps_set = set(self.partition.get_boundary_variable_points(tol=boundary_tol))
        
        # Separate boundary and non-boundary external neighbors
        boundary_neighbors = external_neighbors & boundary_vps_set
        non_boundary_neighbors = external_neighbors - boundary_vps_set
        
        # Find target vertex using distance-based logic (same as identify_target_vertex)
        # For each VP, identify which vertex it's approaching based on lambda
        # NOTE: After the fix to find_connected_components(), all VPs in a component
        # should already approach the same vertex. This is a sanity check.
        target_vertices = []
        for vp_idx in component_vps:
            vp = self.partition.variable_points[vp_idx]
            tv = migration_utils.identify_target_vertex(vp)
            if tv is not None:
                target_vertices.append(tv)
            else:
                self.logger.warning(f"Could not identify target vertex for VP {vp_idx} in component")
        
        # Verify all VPs approach the same vertex (sanity check)
        target_vertex = None
        if target_vertices:
            target_vertex = target_vertices[0]
            if not all(tv == target_vertex for tv in target_vertices):
                # This should NOT happen after the component identification fix!
                unique_targets = set(target_vertices)
                self.logger.error(
                    f"UNEXPECTED: Component VPs approach different vertices: {unique_targets}. "
                    f"This indicates a bug in find_connected_components()!"
                )
                # Fallback: use most common target vertex
                target_vertex = Counter(target_vertices).most_common(1)[0][0]
        
        # Compute min distance (closest VP to target vertex)
        min_distance = float('inf')
        for vp_idx in component_vps:
            vp = self.partition.variable_points[vp_idx]
            dist = migration_utils.compute_boundary_distance(vp)
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
    
    # =========================================================================
    # Auxiliary Component Construction
    # =========================================================================
    
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
        
        # CRITICAL: Validate that both VPs in the component approach the same target vertex
        # This must be checked BEFORE trying to construct the auxiliary component
        # Use identify_target_vertex() which considers lambda value, not just edge containment
        from src.core.migration_utils import identify_target_vertex
        
        invalid_vps = []
        for vp_idx in vp_indices:
            vp = self.partition.variable_points[vp_idx]
            vp_target = identify_target_vertex(vp)
            if vp_target != target_vertex:
                invalid_vps.append(vp_idx)
                self.logger.debug(
                    f"Component {comp_idx}:   ✗ VP {vp_idx}: edge {vp.edge}, λ={vp.lambda_param:.6f} "
                    f"approaches vertex {vp_target}, NOT target vertex {target_vertex}"
                )
        
        if invalid_vps:
            error_msg = (
                f"Invalid 2-VP component {vp_indices}: VP(s) {invalid_vps} don't approach target vertex {target_vertex}"
            )
            if strict_validation:
                raise ValueError(error_msg)
            else:
                self.logger.warning(f"Component {comp_idx}: {error_msg} (proceeding anyway for visualization)")
        
        # Get neighbors of each VP
        neighbors_a = migration_utils.get_two_neighbors(self.partition, vp_a)
        neighbors_b = migration_utils.get_two_neighbors(self.partition, vp_b)
        
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
        # Use identify_target_vertex() which considers lambda value, not just edge containment
        filtered_candidates = []
        rejected_candidates = []
        
        for position, vp_idx in candidates:
            vp = self.partition.variable_points[vp_idx]
            vp_target = migration_utils.identify_target_vertex(vp)
            if vp_target == target_vertex:
                filtered_candidates.append((position, vp_idx))
                self.logger.debug(f"Component {comp_idx}:   ✓ VP {vp_idx} ({position}): edge {vp.edge}, λ={vp.lambda_param:.6f} approaches target vertex {target_vertex}")
            else:
                rejected_candidates.append((position, vp_idx, vp.edge, vp_target))
                self.logger.debug(
                    f"Component {comp_idx}:   ✗ VP {vp_idx} ({position}): edge {vp.edge}, λ={vp.lambda_param:.6f} "
                    f"approaches vertex {vp_target}, NOT target vertex {target_vertex} - REJECTED"
                )
        
        if not filtered_candidates:
            error_msg = (
                f"Component {comp_idx}: No valid third VP found for 2-VP component {vp_indices}. "
                f"All candidates rejected (don't approach target vertex {target_vertex}): "
            )
            for pos, vp_idx, edge, vp_target in rejected_candidates:
                error_msg += f"\n  - VP {vp_idx} ({pos}): edge {edge}, approaches {vp_target}"
            
            if strict_validation:
                self.logger.debug(error_msg)  # DEBUG: Expected filtering during validation
                raise ValueError(error_msg)
            else:
                # Fallback for visualization: use candidate with minimum distance regardless of validity
                self.logger.warning(f"{error_msg}\nComponent {comp_idx}: Using FALLBACK (min distance) for visualization.")
                
                best_candidate = None
                min_dist = float('inf')
                
                for position, vp_idx in candidates:
                    vp = self.partition.variable_points[vp_idx]
                    dist = migration_utils.compute_boundary_distance(vp)
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
            self.logger.debug(
                f"Component {comp_idx}:   Rejected {len(rejected_candidates)} candidate(s) that don't approach target vertex {target_vertex}"
            )
        
        # THEN select by distance from FILTERED candidates
        best_candidate = None
        min_dist = float('inf')
        
        for position, vp_idx in filtered_candidates:
            vp = self.partition.variable_points[vp_idx]
            dist = migration_utils.compute_boundary_distance(vp)
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
        
        self.logger.debug(
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
        neighbors_c = migration_utils.get_two_neighbors(self.partition, vp_c)
        vp_b, vp_d = neighbors_c[0], neighbors_c[1]
        
        # Get second level neighbors
        try:
            neighbors_b = migration_utils.get_two_neighbors(self.partition, vp_b)
            vp_a = neighbors_b[0] if neighbors_b[0] != vp_c else neighbors_b[1]
        except Exception as e:
            self.logger.warning(f"Component {comp_idx}: Could not get second level left neighbor for VP {vp_c}: {e}")
            vp_a = None
        
        try:
            neighbors_d = migration_utils.get_two_neighbors(self.partition, vp_d)
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
            
            # Use identify_target_vertex() which considers lambda value, not just edge containment
            for vp_idx in triplet:
                vp = self.partition.variable_points[vp_idx]
                vp_target = migration_utils.identify_target_vertex(vp)
                if vp_target != target_vertex:
                    valid = False
                    invalid_vps.append((vp_idx, vp.edge, vp_target))
                    self.logger.debug(
                        f"Component {comp_idx}:     ✗ VP {vp_idx}: edge {vp.edge}, λ={vp.lambda_param:.6f} "
                        f"approaches vertex {vp_target}, NOT target vertex {target_vertex}"
                    )
                else:
                    dist = migration_utils.compute_boundary_distance(vp)
                    total_dist += dist
                    self.logger.debug(f"Component {comp_idx}:     ✓ VP {vp_idx}: edge {vp.edge}, λ={vp.lambda_param:.6f} approaches target vertex {target_vertex}, dist={dist:.6f}")
            
            if valid:
                valid_candidates.append((config_name, triplet, total_dist))
                self.logger.debug(f"Component {comp_idx}:   ✓ Triplet '{config_name}' is VALID (total_dist={total_dist:.6f})")
            else:
                self.logger.debug(
                    f"Component {comp_idx}:   ✗ Triplet '{config_name}' is INVALID - {len(invalid_vps)} VP(s) don't approach target vertex"
                )
        
        if not valid_candidates:
            error_msg = (
                f"Component {comp_idx}: No valid triplet found for 1-VP component [{vp_c}]. "
                f"All triplets have VPs that don't approach target vertex {target_vertex}."
            )
            
            if strict_validation:
                self.logger.debug(error_msg)  # DEBUG: Expected filtering during validation
                raise ValueError(error_msg)
            else:
                # Fallback for visualization: use triplet with minimum distance regardless of validity
                self.logger.warning(f"{error_msg} Component {comp_idx}: Using FALLBACK (min distance) for visualization.")
                
                # Calculate distances for all triplets
                fallback_candidates = []
                for config_name, triplet in triplet_candidates:
                    total_dist = sum(
                        migration_utils.compute_boundary_distance(self.partition.variable_points[vp]) 
                        for vp in triplet
                    )
                    fallback_candidates.append((config_name, triplet, total_dist))
                
                best_config, best_triplet, best_total = min(fallback_candidates, key=lambda x: x[2])
                self.logger.warning(
                    f"Component {comp_idx}: ⚠ FALLBACK: Using triplet '{best_config}': {best_triplet} (total_dist={best_total:.6f}) "
                    f"even though some VPs don't approach target vertex"
                )
                return best_triplet
        
        # Select triplet with minimum total distance from VALID candidates
        best_config, best_triplet, best_total = min(valid_candidates, key=lambda x: x[2])
        
        self.logger.debug(
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
            all_neighbors = migration_utils.get_two_neighbors(self.partition, migrating_vp_idx)
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
        self.logger.debug(f"Found neighbors in auxiliary: left={left_neighbor}, right={right_neighbor}")
        
        return (left_neighbor, right_neighbor)
    
    def select_migrating_vp_and_auxiliary(self, component: Dict,
                                          strict_validation: bool = True) -> Tuple[int, List[int]]:
        """
        Select migrating VP and construct auxiliary component.
        
        Returns both the migrating VP and the auxiliary component to avoid
        redundant construction calls.
        
        CACHING: If component has 'cached_auxiliary' from pre-filter, uses it
        to avoid redundant construction and logging.
        
        Args:
            component: Component info dict from analyze_component()
            strict_validation: If False, use fallback when auxiliary construction fails.
                             If True, raise error. Default True.
            
        Returns:
            Tuple of (migrating_vp_idx, auxiliary_component)
        """
        size = len(component['vp_indices'])
        
        # Check for cached auxiliary component from pre-filter
        if 'cached_auxiliary' in component:
            auxiliary_component = component['cached_auxiliary']
            comp_idx = component.get('index', '?')
            self.logger.debug(f"Component {comp_idx}: Using cached auxiliary component {auxiliary_component}")
        else:
            # Fallback: construct if not cached (e.g., direct calls to apply_type1_switch_v2)
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
                self.logger.debug(
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
    
    # =========================================================================
    # Conflict Detection and Resolution
    # =========================================================================
    
    def detect_proximity_conflicts(self, components: List[Dict], boundary_tol: float = 0.1) -> Tuple[List[Dict], List[Dict]]:
        """
        Detect conflicts between components (shared non-boundary neighbors).
        
        A conflict exists when:
        - Components share a non-boundary neighbor VP (topological connection)
        - Both components are near convergence (min_dist < boundary_tol)
        
        Note: Conflict detection does NOT determine deferral. Deferral requires
        additional condition: at least one component has < 3 VPs (risky).
        
        IMPORTANT: Each conflict is between exactly 2 components (one shared VP).
        However, chains can form: Component A shares VP_ab with B, B shares VP_bc with C.
        This creates a chain: A - B - C, where B has multiple neighbors.
        
        Args:
            components: List of component dicts from analyze_component()
            boundary_tol: Tolerance for boundary detection and convergence check
        
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
                    # Use boundary_tol for consistency
                    both_near = min_dist_i < boundary_tol and min_dist_j < boundary_tol
                    
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
        
        CACHING: If valid, the auxiliary component is cached in component['cached_auxiliary']
        to avoid redundant construction during migration.
        
        Args:
            component: Component dictionary with 'vp_indices', 'target_vertex', 'size'
        
        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        size = component['size']
        
        if size == 3:
            # 3-VP components: Validate that all VPs approach the same target vertex
            vp_indices = component['vp_indices']
            
            # Identify target vertex from the component
            target_vertex = component.get('target_vertex')
            if target_vertex is None:
                # Fallback: compute from first VP
                from src.core.migration_utils import identify_target_vertex
                vp0 = self.partition.variable_points[vp_indices[0]]
                target_vertex = identify_target_vertex(vp0)
                if target_vertex is None:
                    return (False, "Cannot identify target vertex")
            
            # Validate each VP approaches the target vertex
            # Use identify_target_vertex() which considers lambda value, not just edge containment
            from src.core.migration_utils import identify_target_vertex
            
            invalid_vps = []
            for vp_idx in vp_indices:
                vp = self.partition.variable_points[vp_idx]
                vp_target = identify_target_vertex(vp)
                if vp_target != target_vertex:
                    invalid_vps.append((vp_idx, vp_target))
            
            if invalid_vps:
                invalid_details = ", ".join([f"VP {vp_idx} approaches {vp_target}" for vp_idx, vp_target in invalid_vps])
                reason = f"Invalid 3-VP component: {invalid_details}, but component target is {target_vertex}"
                return (False, reason)
            
            # All VPs are valid - use as auxiliary component
            component['cached_auxiliary'] = component['vp_indices']
            return (True, "")
        
        elif size == 2:
            # Try to construct auxiliary component for 2-VP
            try:
                auxiliary = self._construct_auxiliary_component_2vp(component, strict_validation=True)
                component['cached_auxiliary'] = auxiliary  # Cache for later use
                return (True, "")
            except ValueError as e:
                reason = f"Cannot find valid third VP: {str(e)}"
                return (False, reason)
        
        elif size == 1:
            # Try to construct auxiliary component for 1-VP
            try:
                auxiliary = self._construct_auxiliary_component_1vp(component, strict_validation=True)
                component['cached_auxiliary'] = auxiliary  # Cache for later use
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
        # ====================================================================
        # STEP 1: PRE-FILTER - Check if valid auxiliary component can be formed
        # ====================================================================
        # Note: Detailed auxiliary construction logging still happens (for debugging),
        # but we don't print per-component status here to avoid clutter
        
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
                self.logger.debug(
                    f"Component {comp_idx} ({comp_size}-VP): Excluded - near triple point "
                    f"(shared VPs: {component.get('triple_point_shared_vps', [])})"
                )
                continue
            
            # Check 2: Can form valid auxiliary component?
            is_valid, reason = self.can_form_valid_auxiliary_component(component)
            
            if is_valid:
                valid_components.append(component)
            else:
                excluded_prefilter.append(component)
                # Log exclusion reason (DEBUG: expected filtering)
                self.logger.debug(f"Component {comp_idx}: Excluded - {reason}")
        
        if not valid_components:
            print("\n⚠️  No valid components found for migration")
            return ([], excluded_prefilter + excluded_triple_point)
        
        # ====================================================================
        # STEP 2: RESOLVE CONFLICTS
        # ====================================================================
        print("\n" + "="*80)
        print("COMPONENT SELECTION - CONFLICT RESOLUTION")
        print("="*80)
        print(f"Step 2: Resolving conflicts (strategy: {conflict_strategy})...")
        print(f"  Total conflicts detected: {len(conflicts)}")
        
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
                    print(f"  Component {comp_idx}: Conflict with {other_idx} (excluded in pre-filter) → migrate")
                    continue
                
                # Both components are valid → check if at least one has 3 VPs
                if comp_idx < other_idx:  # Process each pair only once
                    comp_size = component['size']
                    other_size = other_component['size']
                    
                    # CRITICAL: If at least one component has 3 VPs, migrate BOTH (safe)
                    at_least_one_3vp = (comp_size >= 3) or (other_size >= 3)
                    
                    if at_least_one_3vp:
                        # Case 1: At least one is 3-VP → MIGRATE BOTH (safe, internal neighbors)
                        to_migrate.append(component)
                        to_migrate.append(other_component)
                        print(
                            f"  Conflict {comp_idx} ({comp_size}-VP) vs {other_idx} ({other_size}-VP): "
                            f"At least one 3-VP → migrate BOTH"
                        )
                    else:
                        # Case 2: BOTH < 3-VP → Risky, apply conflict strategy
                        if conflict_strategy == 'exclude_all_conflicts':
                            # Exclude both conflicting components
                            excluded_conflict.append(component)
                            excluded_conflict.append(other_component)
                            print(
                                f"  Conflict {comp_idx} ({comp_size}-VP) vs {other_idx} ({other_size}-VP): "
                                f"Both < 3-VP → Excluding BOTH (strategy: exclude_all_conflicts)"
                            )
                        elif conflict_strategy == 'exclude_one':
                            # Exclude farther component, keep closer one
                            if component['min_distance'] < other_component['min_distance']:
                                to_migrate.append(component)
                                excluded_conflict.append(other_component)
                                print(
                                    f"  Conflict {comp_idx} ({comp_size}-VP, dist={component['min_distance']:.6f}) vs "
                                    f"{other_idx} ({other_size}-VP, dist={other_component['min_distance']:.6f}): "
                                    f"Keeping {comp_idx} (closer)"
                                )
                            else:
                                to_migrate.append(other_component)
                                excluded_conflict.append(component)
                                print(
                                    f"  Conflict {comp_idx} ({comp_size}-VP, dist={component['min_distance']:.6f}) vs "
                                    f"{other_idx} ({other_size}-VP, dist={other_component['min_distance']:.6f}): "
                                    f"Keeping {other_idx} (closer)"
                                )
                        else:
                            raise ValueError(f"Unknown conflict_strategy: {conflict_strategy}")
                    
                    processed.add(comp_idx)
                    processed.add(other_idx)
        
        # ====================================================================
        # STEP 3: SUMMARY
        # ====================================================================
        all_excluded = excluded_prefilter + excluded_triple_point + excluded_conflict
        
        print("\n" + "="*80)
        print("SELECTION SUMMARY:")
        print(f"  Components to migrate: {len(to_migrate)}")
        print(f"  Components excluded:")
        print(f"    - Pre-filter (no valid auxiliary): {len(excluded_prefilter)}")
        if excluded_prefilter:
            prefilter_indices = [c['index'] for c in excluded_prefilter]
            print(f"      Indices: {prefilter_indices}")
        print(f"    - Triple point proximity: {len(excluded_triple_point)}")
        if excluded_triple_point:
            triple_indices = [c['index'] for c in excluded_triple_point]
            print(f"      Indices: {triple_indices}")
        print(f"    - Conflict resolution: {len(excluded_conflict)}")
        if excluded_conflict:
            conflict_indices = [c['index'] for c in excluded_conflict]
            print(f"      Indices: {conflict_indices}")
        print(f"  Total excluded: {len(all_excluded)}")
        print("="*80 + "\n")
        
        return (to_migrate, all_excluded)
    
    # =========================================================================
    # High-Level Orchestration
    # =========================================================================
    
    def run_full_analysis(self, boundary_tol: float = 0.1, 
                          conflict_strategy: str = 'exclude_one',
                          build_migration_plan: bool = True,
                          protect_type2: bool = True) -> Dict:
        """
        Run the complete Type 1 component analysis pipeline.
        
        This orchestrates the entire analysis process:
        1. Find boundary VPs
        2. Find connected components
        3. Analyze each component
        4. Detect conflicts
        5. Filter for Type 2 protection (NEW!)
        6. Select components for migration
        7. Build complete migration plan (if build_migration_plan=True)
        
        Args:
            boundary_tol: Tolerance for boundary VP detection (default: 0.1)
            conflict_strategy: Conflict resolution strategy (default: 'exclude_one')
            build_migration_plan: If True, pre-compute migrating VP and auxiliary for each
                                component (default: True). This avoids redundant computation
                                during the migration loop.
            protect_type2: If True, exclude Type 1 components whose VPs are outer neighbors
                          to boundary triple points (default: True). Uses topology-based
                          identification (not distance-based).
        
        Returns:
            Dict containing:
                - 'boundary_vps': List of boundary VP indices
                - 'components': List of component dicts
                - 'conflicts': List of conflict dicts
                - 'chain_warnings': List of chain warning dicts
                - 'type2_excluded': List of components excluded for Type 2 protection (NEW!)
                - 'to_migrate': List of components selected for migration
                - 'excluded': List of excluded components
                - 'migration_plan': List of dicts with pre-computed migration details
                                   (only if build_migration_plan=True)
        """
        # Step 1: Get boundary VPs (non-triple-point)
        # Use cached triple point VPs to avoid recreating SteinerHandler
        triple_point_vps = self._get_triple_point_vps()
        
        all_boundary_vps = self.partition.get_boundary_variable_points(tol=boundary_tol)
        boundary_vps = [vp for vp in all_boundary_vps if vp not in triple_point_vps]
        boundary_vps_set = set(boundary_vps)
        
        self.logger.info(f"Found {len(boundary_vps)} boundary VPs (excluding {len(triple_point_vps)} triple point VPs)")
        
        # Step 2: Find connected components
        components = self.find_connected_components(boundary_vps_set)
        self.logger.info(f"Found {len(components)} connected component(s)")
        print(f"✓ Found {len(components)} connected component(s)")
        print()
        
        # Step 3: Analyze each component
        component_info = []
        for i, comp_vps in enumerate(components):
            info = self.analyze_component(comp_vps, boundary_tol=boundary_tol)
            info['index'] = i
            component_info.append(info)
        
        # Step 4: Detect conflicts
        conflicts, chain_warnings = self.detect_proximity_conflicts(component_info, boundary_tol=boundary_tol)
        self.logger.info(f"Detected {len(conflicts)} conflict(s) between components")
        
        # Step 4.5: Filter for Type 2 protection (NEW!)
        type2_excluded = []
        available_for_migration = component_info
        
        if protect_type2:
            available_for_migration, type2_excluded = self._filter_for_type2_protection(
                component_info,
                boundary_tol=boundary_tol,
                protection_distance=None  # Unused, kept for API compatibility
            )
            self.logger.info(f"Type 2 Protection: {len(available_for_migration)} available, {len(type2_excluded)} excluded")
        
        # Step 5: Select components for migration (from available set)
        to_migrate, excluded = self.select_components_for_migration(
            available_for_migration, conflicts, conflict_strategy=conflict_strategy
        )
        
        # Summary logging (INFO level)
        self.logger.info(f"Component selection complete:")
        self.logger.info(f"  Total detected: {len(component_info)}")
        self.logger.info(f"  Available for migration: {len(to_migrate)}")
        self.logger.info(f"  Excluded (Type 2 protection): {len(type2_excluded)}")
        self.logger.info(f"  Excluded (other): {len(excluded)}")
        
        result = {
            'boundary_vps': boundary_vps,
            'components': component_info,
            'conflicts': conflicts,
            'chain_warnings': chain_warnings,
            'type2_excluded': type2_excluded,
            'to_migrate': to_migrate,
            'excluded': excluded
        }
        
        # Step 6: Build complete migration plan (if requested)
        if build_migration_plan:
            migration_plan = self._build_migration_plan(to_migrate)
            result['migration_plan'] = migration_plan
        
        return result
    
    def _filter_for_type2_protection(self, component_info: List[Dict],
                                     boundary_tol: float,
                                     protection_distance: float = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Filter Type 1 components to protect Type 2 triple point outer neighbors.
        
        Strategy: Exclude Type 1 components whose VPs are first-level outer
        neighbors to ANY triple point. The first-level VP is the one that Type 2
        actually migrates (to a free edge in T_second_VP), so it must remain in
        place. The second-level VP is only used to identify T_second_VP and is not
        itself moved during Type 2; protecting it is unnecessary and blocks valid
        Type 1 migrations.
        
        For each triple point (3 VPs):
        - Each VP has exactly 1 protected outer neighbor: first_level
        - Total: 3 VPs × 1 neighbor = 3 protected VPs per triple triangle
        - Identified topologically via boundary_segments (NOT by distance)
        
        Args:
            component_info: List of component dicts from analyze_component
            boundary_tol: UNUSED (kept for API compatibility)
            protection_distance: UNUSED (kept for API compatibility)
        
        Returns:
            Tuple of (available_components, excluded_components)
        """
        # Get ALL triple points (not just boundary ones)
        steiner_handler = self._get_steiner_handler()
        all_triple_points = steiner_handler.triple_points
        
        if not all_triple_points:
            self.logger.info("Type 2 Protection: No triple points found - all components available")
            return component_info, []
        
        self.logger.info(f"Type 2 Protection: Processing {len(all_triple_points)} triple point(s)")
        
        # For each triple point, identify first-level outer neighbor VPs (topology-based)
        protected_vps = set()
        
        for tp in all_triple_points:
            tp_vp_indices = list(tp.var_point_indices)
            triple_vp_set = set(tp_vp_indices)
            
            self.logger.debug(f"  Processing triple point: VPs {tp_vp_indices}")
            
            for vp_idx in tp_vp_indices:
                first_level_vps = []
                for segment in self.partition.boundary_segments:
                    if segment.vp_idx_1 == vp_idx:
                        neighbor = segment.vp_idx_2
                        if neighbor not in triple_vp_set:
                            first_level_vps.append(neighbor)
                    elif segment.vp_idx_2 == vp_idx:
                        neighbor = segment.vp_idx_1
                        if neighbor not in triple_vp_set:
                            first_level_vps.append(neighbor)
                
                if len(first_level_vps) != 1:
                    self.logger.debug(
                        f"    VP {vp_idx}: Expected 1 first-level outer neighbor, "
                        f"found {len(first_level_vps)}: {first_level_vps} - skipping"
                    )
                    continue
                
                first_level_vp = first_level_vps[0]
                protected_vps.add(first_level_vp)
                self.logger.debug(f"    VP {vp_idx} → first_level (protected): {first_level_vp}")
        
        # Summary logging (INFO level)
        self.logger.info(f"Type 2 Protection: {len(all_triple_points)} triple points, {len(protected_vps)} protected VPs, {len(component_info)} components analyzed")
        self.logger.debug(f"  Protected VPs: {sorted(protected_vps)}")
        
        # Filter components
        available = []
        excluded = []
        
        for comp in component_info:
            comp_vps = comp['vp_indices']  # Correct key name
            
            # Check if any VP in this component is protected
            conflicting_vps = [vp for vp in comp_vps if vp in protected_vps]
            
            if conflicting_vps:
                excluded.append(comp)
                self.logger.debug(
                    f"Type 2 Protection: Excluding Component {comp['index']} "
                    f"(size {comp['size']}, target vertex {comp['target_vertex']}) - "
                    f"contains protected VP(s): {conflicting_vps}"
                )
            else:
                available.append(comp)
        
        self.logger.info(f"Type 2 Protection: {len(available)} available, {len(excluded)} excluded")
        
        return available, excluded
    
    def _get_steiner_handler(self):
        """Get or create SteinerHandler instance."""
        # Check if we already have a steiner handler in cache
        if hasattr(self, '_steiner_handler_cache') and self._steiner_handler_cache is not None:
            return self._steiner_handler_cache
        
        # Create new SteinerHandler
        from ..partition.steiner_handler import SteinerHandler
        steiner_handler = SteinerHandler(self.mesh, self.partition)
        self._steiner_handler_cache = steiner_handler
        
        return steiner_handler
    
    def _build_migration_plan(self, to_migrate: List[Dict]) -> List[Dict]:
        """
        Build complete migration plan with pre-computed migration details.
        
        For each component to migrate, compute:
        - Migrating VP index
        - Auxiliary component
        - Left and right neighbors
        
        This avoids redundant computation during the migration loop.
        
        Args:
            to_migrate: List of component dicts from select_components_for_migration
            
        Returns:
            List of migration plan entries, each containing:
                - 'component_idx': Original component index
                - 'component': Original component dict
                - 'migrating_vp': Pre-computed migrating VP index
                - 'auxiliary_component': Pre-computed auxiliary component
                - 'left_neighbor': Left neighbor VP index
                - 'right_neighbor': Right neighbor VP index
        """
        migration_plan = []
        
        self.logger.info(f"Building complete migration plan for {len(to_migrate)} components...")
        
        for comp in to_migrate:
            try:
                # Pre-compute migrating VP and auxiliary component
                migrating_vp, auxiliary_component = self.select_migrating_vp_and_auxiliary(
                    comp, strict_validation=True
                )
                
                # Get neighbors from auxiliary component
                left_neighbor, right_neighbor = self._get_neighbors_from_auxiliary(
                    migrating_vp, auxiliary_component
                )
                
                # Log migration plan entry with component index and full auxiliary
                self.logger.info(
                    f"Component {comp['index']}: Selected migrating VP {migrating_vp} "
                    f"from auxiliary {auxiliary_component} "
                    f"(neighbors: L={left_neighbor}, R={right_neighbor})"
                )
                
                # Build migration plan entry
                plan_entry = {
                    'component_idx': comp['index'],
                    'component': comp,
                    'migrating_vp': migrating_vp,
                    'auxiliary_component': auxiliary_component,
                    'left_neighbor': left_neighbor,
                    'right_neighbor': right_neighbor,
                }
                
                migration_plan.append(plan_entry)
                
            except Exception as e:
                self.logger.warning(
                    f"Failed to build migration plan for component {comp['index']}: {e}"
                )
                # Skip this component in the migration plan
                continue
        
        self.logger.info(f"Migration plan built successfully: {len(migration_plan)} entries")
        
        return migration_plan
