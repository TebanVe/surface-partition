"""
Contour partition data structures for perimeter refinement optimization.

This module implements Section 5 of the paper "Partitions of Minimal Length on Manifolds"
by Bogosel and Oudet. It provides data structures for representing partition contours
as variable points on mesh edges, enabling direct perimeter optimization.

TERMINOLOGY (following paper Section 5):
- "cell": Partition region (what we optimize for equal areas)
- "triangle": Mesh triangle element (computational discretization)
- "edge": Mesh triangle edge (computational discretization)
- Variable points "belong to" cells they separate (not just "adjacent")

Key classes:
- VariablePoint: Point on a mesh edge parameterized by λ ∈ [0,1]
- TriangleSegment: Links mesh triangles to boundary variable points
- PartitionContour: Complete partition with global variable point management

The triangle-based approach ensures geometrically valid segment extraction that works
correctly with optimized λ values, avoiding the ordering issues of earlier implementations.
"""

import numpy as np
import h5py
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass

try:
    from ..logging_config import get_logger
    from .tri_mesh import TriMesh
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from logging_config import get_logger
    from core.tri_mesh import TriMesh

# Import BoundaryTriangleInfo for type hints (optional parameter)
try:
    from ..find_contours import BoundaryTriangleInfo
except ImportError:
    try:
        from find_contours import BoundaryTriangleInfo
    except ImportError:
        BoundaryTriangleInfo = None  # Fallback if not available


@dataclass
class VariablePoint:
    """
    A point on a mesh edge parameterized by λ ∈ [0,1].
    
    CRITICAL λ CONVENTION:
        Position: x = λ * edge[0] + (1-λ) * edge[1]
        
        This means:
        - λ = 1  → position is exactly at edge[0]
        - λ = 0  → position is exactly at edge[1]
        - λ = 0.5 → position is at midpoint
        
        Since edges are normalized (smaller vertex index first):
        - λ = 1 → at smaller vertex index
        - λ = 0 → at larger vertex index
    
    As per paper Section 5: "each of these points belongs to at least two cells"
    because they are situated on mesh triangle edges that cross cell boundaries.
    
    Attributes:
        edge: Tuple of (vertex_idx_start, vertex_idx_end), normalized so edge[0] < edge[1]
        lambda_param: Parameter value in [0, 1]. λ=1 → edge[0], λ=0 → edge[1]
        global_idx: Index in the global variable vector
        belongs_to_cells: Set of cell indices that this point belongs to (boundary between these cells)
        active: Whether this VP is active. Destroyed VPs are marked inactive but never
                removed from the list, preserving index stability for snapshot rollback.
    """
    edge: Tuple[int, int]
    lambda_param: float
    global_idx: int
    belongs_to_cells: Set[int]
    active: bool = True
    
    def evaluate(self, vertices: np.ndarray) -> np.ndarray:
        """Compute actual 3D/2D coordinates given lambda."""
        v_start = vertices[self.edge[0]]
        v_end = vertices[self.edge[1]]
        return self.lambda_param * v_start + (1 - self.lambda_param) * v_end
    
    def on_boundary(self, tol: float = 1e-3) -> bool:
        """Check if point is near edge endpoints (topology switch condition)."""
        return self.lambda_param < tol or self.lambda_param > (1 - tol)


@dataclass
class TriangleSegment:
    """
    Represents a boundary segment within a specific mesh triangle.
    
    This class links the geometric triangle to the variable points on its boundary edges,
    enabling direct re-extraction of contour segments after optimization.
    
    Attributes:
        triangle_idx: Index of the mesh triangle containing this segment
        vertex_indices: Tuple of 3 vertex indices (v1, v2, v3) of the triangle
        boundary_edges: List of edges that cross cell boundaries (normalized: smaller index first)
        var_point_indices: List of variable point indices corresponding to boundary_edges
        vertex_labels: (DEPRECATED) Kept as optional field for legacy code compatibility.
                      New code should use partition.vertex_labels[ts.vertex_indices] instead.
    """
    triangle_idx: int
    vertex_indices: Tuple[int, int, int]
    boundary_edges: List[Tuple[int, int]]
    var_point_indices: List[int]
    vertex_labels: Optional[Tuple[int, ...]] = None
    
    def num_cells(self, vertex_labels_arr: Optional[np.ndarray] = None) -> int:
        """
        Return number of distinct cells in this triangle.
        
        Args:
            vertex_labels_arr: 1D array mapping vertex index to cell label.
                              If not provided, uses the stored vertex_labels or heuristic.
        """
        if vertex_labels_arr is not None:
            labels = set(int(vertex_labels_arr[v]) for v in self.vertex_indices)
            return len(labels)
        if self.vertex_labels is not None:
            return len(set(self.vertex_labels))
        return min(len(self.var_point_indices) + 1, 3)
    
    def is_triple_point(self) -> bool:
        """
        Check if this is a triple point (3 different cells meet).
        
        A triangle is a triple point if it has exactly 3 variable points
        (one on each edge of the void triangle interior).
        """
        return len(self.var_point_indices) == 3
    
    def get_cell_indices(self, vertex_labels_arr: Optional[np.ndarray] = None) -> Set[int]:
        """Get set of unique cell indices in this triangle."""
        if vertex_labels_arr is not None:
            return set(int(vertex_labels_arr[v]) for v in self.vertex_indices)
        if self.vertex_labels is not None:
            return set(self.vertex_labels)
        return set()


@dataclass
class BoundarySegment:
    """
    Explicit representation of a boundary segment between two variable points.
    
    This replaces the implicit connectivity (inferred from TriangleSegments) with
    explicit tracking. Essential for handling cross-triangle segments after topology
    switches.
    
    Attributes:
        vp_idx_1: First variable point index
        vp_idx_2: Second variable point index
        cell_pair: Tuple of (cell_a, cell_b) - the two cells this segment separates
        segment_type: Classification of segment geometry:
            - "normal": Both VPs in same triangle (standard case)
            - "edge_following": VPs in different triangles but segment follows mesh edge
            - "edge_cutting": Segment cuts across triangle edges (needs crossing info)
        crossing_triangles: List of triangle indices this segment crosses (for edge_cutting)
    """
    vp_idx_1: int
    vp_idx_2: int
    cell_pair: Tuple[int, int]
    segment_type: str = "normal"
    crossing_triangles: List[int] = None
    
    def __post_init__(self):
        if self.crossing_triangles is None:
            self.crossing_triangles = []
    
    def normalized_key(self) -> Tuple[int, int]:
        """Return (min_vp, max_vp) for deduplication."""
        return (min(self.vp_idx_1, self.vp_idx_2), max(self.vp_idx_1, self.vp_idx_2))


class PartitionContour:
    """
    Global partition representation with variable points on mesh edges.
    
    This is the main data structure for Section 5 optimization. It manages:
    - All variable points across the partition
    - Triangle-based contour segment extraction
    - Topology information (which mesh triangle edges form which cell boundaries)
    - Conversion to/from indicator functions
    
    Internal storage uses efficient representations:
    - _vertex_labels: 1D int array (replaces N x n_cells indicator_functions matrix)
    - _triangle_segments: Dict[tri_idx, TriangleSegment] for O(1) lookup
    - _vp_adjacency: Dict[vp_idx, Set[int]] for O(1) neighbor lookup
    
    Backward-compatible properties reconstruct the old formats on demand.
    
    IMPORTANT: segment_to_triangle must be rebuilt after topology switches (Type 1 or Type 2)
    """
    
    def __init__(self, mesh: TriMesh, indicator_functions: np.ndarray, 
                 boundary_topology: Optional[Dict] = None):
        """
        Initialize partition contours from indicator functions φ_i.
        
        Args:
            mesh: TriMesh object
            indicator_functions: (N, n_cells) binary array from winner-takes-all
            boundary_topology: Optional pre-computed boundary topology from ContourAnalyzer.
                             If provided, avoids redundant triangle scanning.
                             Dict[region_idx] -> List[BoundaryTriangleInfo]
        """
        self.mesh = mesh
        self.logger = get_logger(__name__)
        self._vertex_labels: np.ndarray = np.argmax(indicator_functions, axis=1)
        self.n_cells = indicator_functions.shape[1]
        
        # Global data structures
        self.variable_points: List[VariablePoint] = []
        self._triangle_segments: Dict[int, TriangleSegment] = {}
        self.edge_to_varpoint: Dict[Tuple[int, int], int] = {}
        self.segment_to_triangle: Dict[Tuple[int, int], int] = {}
        self.triple_points: Optional[List] = None
        
        # VP adjacency: vp_idx -> set of neighbor vp_indices
        self._vp_adjacency: Dict[int, Set[int]] = {}
        
        # Active VP index management
        self._active_vp_indices: List[int] = []
        self._vp_idx_to_opt_idx: Dict[int, int] = {}
        
        # Legacy boundary_segments list (kept for transition period)
        self.boundary_segments: List[BoundarySegment] = []
        
        # Choose initialization method based on available data
        if boundary_topology is not None:
            self.logger.info("Initializing from pre-computed boundary topology (efficient path)")
            self._initialize_from_boundary_topology(boundary_topology)
        else:
            self.logger.info("Initializing from indicator functions (scanning all triangles)")
            self._initialize_from_indicators()
        
        # Build segment-to-triangle inverse map for efficient lookups
        self._build_segment_to_triangle_map()
        
        # Build VP adjacency from triangle segments
        self._rebuild_vp_adjacency()
        
        # Initialize active VP indices (all VPs active at construction)
        self.rebuild_active_vp_indices()
        
        self.logger.info(f"Initialized PartitionContour: {len(self.variable_points)} variable points, "
                        f"{self.n_cells} partition cells")
    
    # =========================================================================
    # Backward-compatible properties
    # =========================================================================
    
    @property
    def vertex_labels(self) -> np.ndarray:
        """1D array of cell labels per vertex (the canonical representation)."""
        return self._vertex_labels
    
    @vertex_labels.setter
    def vertex_labels(self, value: np.ndarray):
        self._vertex_labels = value
    
    @property
    def indicator_functions(self) -> np.ndarray:
        """Reconstruct (N, n_cells) indicator matrix on demand (for HDF5 export and legacy code)."""
        ind = np.zeros((len(self._vertex_labels), self.n_cells))
        ind[np.arange(len(self._vertex_labels)), self._vertex_labels] = 1.0
        return ind
    
    @indicator_functions.setter
    def indicator_functions(self, value: np.ndarray):
        self._vertex_labels = np.argmax(value, axis=1)
    
    @property
    def triangle_segments(self) -> List[TriangleSegment]:
        """Backward-compat: return list view of triangle segments for iteration."""
        return list(self._triangle_segments.values())
    
    @triangle_segments.setter
    def triangle_segments(self, value: List[TriangleSegment]):
        """Accept list assignment and convert to dict internally."""
        self._triangle_segments = {ts.triangle_idx: ts for ts in value}
    
    def get_triangle_segment(self, tri_idx: int) -> Optional[TriangleSegment]:
        """O(1) lookup of TriangleSegment by triangle index."""
        return self._triangle_segments.get(tri_idx)
    
    # =========================================================================
    # Active VP index management
    # =========================================================================
    
    def rebuild_active_vp_indices(self):
        """Rebuild the active VP index list after migrations."""
        self._active_vp_indices = [i for i, vp in enumerate(self.variable_points) if vp.active]
        self._vp_idx_to_opt_idx = {vp_idx: opt_idx for opt_idx, vp_idx in enumerate(self._active_vp_indices)}
    
    @property
    def active_vp_indices(self) -> List[int]:
        return self._active_vp_indices
    
    # =========================================================================
    # VP adjacency
    # =========================================================================
    
    def _rebuild_vp_adjacency(self):
        """Rebuild VP adjacency dict from current triangle segments."""
        self._vp_adjacency = {}
        for ts in self._triangle_segments.values():
            vp_indices = ts.var_point_indices
            if len(vp_indices) == 2:
                a, b = vp_indices
                self._vp_adjacency.setdefault(a, set()).add(b)
                self._vp_adjacency.setdefault(b, set()).add(a)
            elif len(vp_indices) == 3:
                for i in range(3):
                    for j in range(i + 1, 3):
                        a, b = vp_indices[i], vp_indices[j]
                        self._vp_adjacency.setdefault(a, set()).add(b)
                        self._vp_adjacency.setdefault(b, set()).add(a)
    
    def get_vp_neighbors(self, vp_idx: int) -> Set[int]:
        """Get neighbor VP indices for a given VP. O(1)."""
        return self._vp_adjacency.get(vp_idx, set())
    
    def _initialize_from_indicators(self):
        """
        Extract contours from indicator functions by finding mesh triangle edges
        that cross partition cell boundaries (where φ_i changes from 0 to 1).
        
        Uses triangle-based segment storage for geometrically valid contour extraction.
        """
        vertex_labels = self._vertex_labels
        
        for tri_idx, face in enumerate(self.mesh.faces):
            v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
            label1, label2, label3 = vertex_labels[v1], vertex_labels[v2], vertex_labels[v3]
            
            boundary_edges_in_triangle = []
            var_points_in_triangle = []
            
            edges = [(v1, v2), (v2, v3), (v3, v1)]
            labels = [(label1, label2), (label2, label3), (label3, label1)]
            
            for edge, (lab_a, lab_b) in zip(edges, labels):
                if lab_a != lab_b:
                    normalized_edge = tuple(sorted(edge))
                    
                    if normalized_edge not in self.edge_to_varpoint:
                        var_point = VariablePoint(
                            edge=normalized_edge,
                            lambda_param=0.5,
                            global_idx=len(self.variable_points),
                            belongs_to_cells={int(lab_a), int(lab_b)}
                        )
                        self.variable_points.append(var_point)
                        self.edge_to_varpoint[normalized_edge] = var_point.global_idx
                        var_point_idx = var_point.global_idx
                    else:
                        var_idx = self.edge_to_varpoint[normalized_edge]
                        self.variable_points[var_idx].belongs_to_cells.update([int(lab_a), int(lab_b)])
                        var_point_idx = var_idx
                    
                    boundary_edges_in_triangle.append(normalized_edge)
                    var_points_in_triangle.append(var_point_idx)
            
            if boundary_edges_in_triangle:
                tri_seg = TriangleSegment(
                    triangle_idx=tri_idx,
                    vertex_indices=(v1, v2, v3),
                    boundary_edges=boundary_edges_in_triangle,
                    var_point_indices=var_points_in_triangle
                )
                self._triangle_segments[tri_idx] = tri_seg
        
        num_two_cell = sum(1 for ts in self._triangle_segments.values() if ts.num_cells(vertex_labels) == 2)
        num_triple = sum(1 for ts in self._triangle_segments.values() if ts.is_triple_point())
        self.logger.info(f"Created {len(self._triangle_segments)} triangle segments: "
                        f"{num_two_cell} two-cell, {num_triple} triple-point")
        
        self._build_segment_connectivity()
    
    def _initialize_from_boundary_topology(self, boundary_topology: Dict):
        """
        Initialize from pre-computed boundary topology (efficient path).
        
        Two-pass structure:
        - Pass 1: Create unique variable points on boundary edges
        - Pass 2: Create triangle segments linking triangles to their variable points
        
        Args:
            boundary_topology: Dict[region_idx] -> List[BoundaryTriangleInfo]
        """
        all_edges_info = {}
        
        for region_idx, tri_infos in boundary_topology.items():
            for tri_info in tri_infos:
                for edge in tri_info.crossed_edges:
                    normalized_edge = tuple(sorted(edge))
                    v_start_idx = tri_info.vertices.index(edge[0])
                    v_end_idx = tri_info.vertices.index(edge[1])
                    label_start = tri_info.vertex_labels[v_start_idx]
                    label_end = tri_info.vertex_labels[v_end_idx]
                    
                    if normalized_edge not in all_edges_info:
                        all_edges_info[normalized_edge] = set()
                    all_edges_info[normalized_edge].update([label_start, label_end])
        
        sorted_edges = sorted(all_edges_info.keys())
        
        for normalized_edge in sorted_edges:
            cells = all_edges_info[normalized_edge]
            var_point = VariablePoint(
                edge=normalized_edge,
                lambda_param=0.5,
                global_idx=len(self.variable_points),
                belongs_to_cells=cells
            )
            self.variable_points.append(var_point)
            self.edge_to_varpoint[normalized_edge] = var_point.global_idx
        
        self.logger.info(f"Pass 1 complete: Created {len(self.variable_points)} unique variable points")
        
        seen_triangles = set()
        
        for region_idx, tri_infos in boundary_topology.items():
            for tri_info in tri_infos:
                if tri_info.triangle_idx in seen_triangles:
                    continue
                seen_triangles.add(tri_info.triangle_idx)
                
                var_point_indices = []
                boundary_edges_normalized = []
                
                for edge in tri_info.crossed_edges:
                    normalized_edge = tuple(sorted(edge))
                    var_point_indices.append(self.edge_to_varpoint[normalized_edge])
                    boundary_edges_normalized.append(normalized_edge)
                
                tri_seg = TriangleSegment(
                    triangle_idx=tri_info.triangle_idx,
                    vertex_indices=tri_info.vertices,
                    boundary_edges=boundary_edges_normalized,
                    var_point_indices=var_point_indices
                )
                self._triangle_segments[tri_info.triangle_idx] = tri_seg
        
        # Fix triple point triangles that may have only 2 crossed_edges from boundary_topology
        vertex_labels = self._vertex_labels
        
        for tri_seg in self._triangle_segments.values():
            if len(tri_seg.var_point_indices) == 2:
                v1, v2, v3 = tri_seg.vertex_indices
                labels = {int(vertex_labels[v1]), int(vertex_labels[v2]), int(vertex_labels[v3])}
                
                if len(labels) == 3:
                    tri_edges = [
                        tuple(sorted([v1, v2])),
                        tuple(sorted([v2, v3])),
                        tuple(sorted([v3, v1]))
                    ]
                    
                    for edge in tri_edges:
                        if edge not in tri_seg.boundary_edges and edge in self.edge_to_varpoint:
                            tri_seg.boundary_edges.append(edge)
                            tri_seg.var_point_indices.append(self.edge_to_varpoint[edge])
        
        num_two_cell = sum(1 for ts in self._triangle_segments.values() if ts.num_cells(vertex_labels) == 2)
        num_triple = sum(1 for ts in self._triangle_segments.values() if ts.is_triple_point())
        self.logger.info(f"Pass 2 complete: Created {len(self._triangle_segments)} triangle segments: "
                        f"{num_two_cell} two-cell, {num_triple} triple-point")
        
        self._build_segment_connectivity()
    
    def get_variable_vector(self) -> np.ndarray:
        """
        Return current λ parameters as optimization vector.
        
        Returns all VP lambdas (both active and inactive) for backward compatibility.
        Use get_active_variable_vector() after migrations mark VPs inactive.
        
        Returns:
            Array of shape (n_variable_points,) with λ values
        """
        return np.array([vp.lambda_param for vp in self.variable_points])
    
    def set_variable_vector(self, lambda_vec: np.ndarray):
        """
        Update all λ parameters from optimization vector.
        
        Accepts vectors sized to either all VPs or active VPs only.
        
        Args:
            lambda_vec: Array of λ values
        """
        if len(lambda_vec) == len(self.variable_points):
            for i, lam in enumerate(lambda_vec):
                self.variable_points[i].lambda_param = float(np.clip(lam, 0.0, 1.0))
        elif len(lambda_vec) == len(self._active_vp_indices):
            for k, vp_idx in enumerate(self._active_vp_indices):
                self.variable_points[vp_idx].lambda_param = float(np.clip(lambda_vec[k], 0.0, 1.0))
        else:
            raise ValueError(f"Lambda vector size {len(lambda_vec)} doesn't match "
                           f"total VPs ({len(self.variable_points)}) or "
                           f"active VPs ({len(self._active_vp_indices)})")
    
    def get_active_variable_vector(self) -> np.ndarray:
        """Return λ parameters for active VPs only (the optimization vector after migrations)."""
        return np.array([self.variable_points[i].lambda_param for i in self._active_vp_indices])
    
    def set_active_variable_vector(self, vec: np.ndarray):
        """Update λ parameters for active VPs only."""
        if len(vec) != len(self._active_vp_indices):
            raise ValueError(f"Vector size {len(vec)} doesn't match "
                           f"active VP count {len(self._active_vp_indices)}")
        for k, vp_idx in enumerate(self._active_vp_indices):
            self.variable_points[vp_idx].lambda_param = float(np.clip(vec[k], 0.0, 1.0))
    
    def rebuild_triangle_segments_from_current_vps(self):
        """
        Rebuild triangle_segments based on current variable point positions.
        
        Re-scans the mesh and rebuilds the dict based on current VP locations.
        Only active VPs are considered. Preserves existing variable_points and
        their lambda values - only updates the triangle-to-VP mapping.
        """
        self.logger.info("Rebuilding triangle_segments from current VPs...")
        
        self._triangle_segments = {}
        
        edge_to_vp = {}
        for vp_idx, vp in enumerate(self.variable_points):
            if not vp.active:
                continue
            normalized_edge = tuple(sorted(vp.edge))
            edge_to_vp[normalized_edge] = vp_idx
        
        for tri_idx, face in enumerate(self.mesh.faces):
            v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
            
            tri_edges = [
                tuple(sorted([v1, v2])),
                tuple(sorted([v2, v3])),
                tuple(sorted([v3, v1]))
            ]
            
            boundary_edges = []
            var_point_indices = []
            
            for edge in tri_edges:
                if edge in edge_to_vp:
                    boundary_edges.append(edge)
                    var_point_indices.append(edge_to_vp[edge])
            
            if len(var_point_indices) >= 1:
                tri_seg = TriangleSegment(
                    triangle_idx=tri_idx,
                    vertex_indices=(v1, v2, v3),
                    boundary_edges=boundary_edges,
                    var_point_indices=var_point_indices
                )
                self._triangle_segments[tri_idx] = tri_seg
        
        segments_list = list(self._triangle_segments.values())
        num_one_vp = sum(1 for ts in segments_list if len(ts.var_point_indices) == 1)
        num_two_vp = sum(1 for ts in segments_list if len(ts.var_point_indices) == 2)
        num_three_vp = sum(1 for ts in segments_list if len(ts.var_point_indices) == 3)
        
        self.logger.info(f"Rebuilt {len(self._triangle_segments)} triangle segments:")
        self.logger.info(f"  {num_one_vp} with 1 VP, {num_two_vp} with 2 VPs, {num_three_vp} with 3 VPs")
        
        self._build_segment_connectivity(force_rebuild=True)
        self._build_segment_to_triangle_map()
        self._rebuild_vp_adjacency()
    
    def rebuild_triangle_segments_for_affected_triangles(self, affected_triangles: List[int]):
        """
        Optimized rebuild - only updates specified triangles in the dict.
        
        Args:
            affected_triangles: List of triangle indices to rebuild
        """
        self.logger.info(f"Rebuilding triangle_segments for {len(affected_triangles)} affected triangles...")
        
        for tri_idx in affected_triangles:
            self._triangle_segments.pop(tri_idx, None)
        
        edge_to_vp = {}
        for vp_idx, vp in enumerate(self.variable_points):
            if not vp.active:
                continue
            normalized_edge = tuple(sorted(vp.edge))
            edge_to_vp[normalized_edge] = vp_idx
        
        for tri_idx in affected_triangles:
            face = self.mesh.faces[tri_idx]
            v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
            
            tri_edges = [
                tuple(sorted([v1, v2])),
                tuple(sorted([v2, v3])),
                tuple(sorted([v3, v1]))
            ]
            
            boundary_edges = []
            var_point_indices = []
            
            for edge in tri_edges:
                if edge in edge_to_vp:
                    boundary_edges.append(edge)
                    var_point_indices.append(edge_to_vp[edge])
            
            if len(var_point_indices) >= 1:
                tri_seg = TriangleSegment(
                    triangle_idx=tri_idx,
                    vertex_indices=(v1, v2, v3),
                    boundary_edges=boundary_edges,
                    var_point_indices=var_point_indices
                )
                self._triangle_segments[tri_idx] = tri_seg
        
        segments_list = list(self._triangle_segments.values())
        num_one_vp = sum(1 for ts in segments_list if len(ts.var_point_indices) == 1)
        num_two_vp = sum(1 for ts in segments_list if len(ts.var_point_indices) == 2)
        num_three_vp = sum(1 for ts in segments_list if len(ts.var_point_indices) == 3)
        
        self.logger.info(f"Rebuilt {len(affected_triangles)} affected triangles:")
        self.logger.info(f"  Total triangle_segments: {len(self._triangle_segments)}")
        self.logger.info(f"    {num_one_vp} with 1 VP, {num_two_vp} with 2 VPs, {num_three_vp} with 3 VPs")

        
    def _build_segment_connectivity(self, force_rebuild: bool = False):
        """
        Build or update explicit segment connectivity.
        
        Args:
            force_rebuild: If True, discard existing boundary_segments and rebuild
                from scratch using current triangle_segments. This is necessary after
                Type 1 migrations which move VPs to new edges without updating
                boundary_segments, causing stale VP-VP connectivity to accumulate.
        
        TWO MODES OF OPERATION:
        
        Mode 1 - Full Build (boundary_segments is empty, or force_rebuild=True):
            Build from triangle_segments. This establishes correct VP-VP connectivity
            that reflects the current VP positions.
        
        Mode 2 - Update (boundary_segments exists and force_rebuild=False):
            PRESERVE existing segment list (don't clear). Only update cell_pair attributes.
            Use only within a single migration step where connectivity is known to be
            correct (e.g., Type 2 direct segment manipulation).
        """
        if force_rebuild:
            self.boundary_segments = []

        if self.boundary_segments:
            # MODE 2: Update existing segments (after topology switch)
            # Preserve connectivity, only update cell_pair attributes
            self.logger.info(f"Updating {len(self.boundary_segments)} existing boundary segments (preserving connectivity)")
            
            for seg in self.boundary_segments:
                # Update cell_pair from current VP membership
                cells1 = self.variable_points[seg.vp_idx_1].belongs_to_cells
                cells2 = self.variable_points[seg.vp_idx_2].belongs_to_cells
                shared_cells = cells1 & cells2
                seg.cell_pair = tuple(sorted(shared_cells)) if len(shared_cells) == 2 else (0, 0)
                
                # Reset segment_type to "normal" - will be reclassified by classify_all_segments()
                # This allows segments that became cross-triangle to be properly detected
                seg.segment_type = "normal"
                seg.crossing_triangles = []
            
            self.logger.info(f"Updated cell_pair attributes for {len(self.boundary_segments)} segments")
            return
        
        # MODE 1: Initial build from scratch (boundary_segments is empty)
        self.logger.info("Building segment connectivity from scratch (initial build)")
        seen_segments = set()
        
        for tri_seg in self._triangle_segments.values():
            var_indices = tri_seg.var_point_indices
            
            if len(var_indices) == 2:
                # Normal boundary: one segment between two VPs
                vp1, vp2 = var_indices
                seg_key = (min(vp1, vp2), max(vp1, vp2))
                
                if seg_key not in seen_segments:
                    seen_segments.add(seg_key)
                    
                    # Determine cell pair from VP membership
                    cells1 = self.variable_points[vp1].belongs_to_cells
                    cells2 = self.variable_points[vp2].belongs_to_cells
                    shared_cells = cells1 & cells2
                    cell_pair = tuple(sorted(shared_cells)) if len(shared_cells) == 2 else (0, 0)
                    
                    self.boundary_segments.append(BoundarySegment(
                        vp_idx_1=vp1,
                        vp_idx_2=vp2,
                        cell_pair=cell_pair,
                        segment_type="normal"
                    ))
            
            elif len(var_indices) == 3:
                # Triple point: three segments forming void triangle edges
                for i in range(3):
                    for j in range(i + 1, 3):
                        vp1, vp2 = var_indices[i], var_indices[j]
                        seg_key = (min(vp1, vp2), max(vp1, vp2))
                        
                        if seg_key not in seen_segments:
                            seen_segments.add(seg_key)
                            
                            cells1 = self.variable_points[vp1].belongs_to_cells
                            cells2 = self.variable_points[vp2].belongs_to_cells
                            shared_cells = cells1 & cells2
                            cell_pair = tuple(sorted(shared_cells)) if len(shared_cells) == 2 else (0, 0)
                            
                            self.boundary_segments.append(BoundarySegment(
                                vp_idx_1=vp1,
                                vp_idx_2=vp2,
                                cell_pair=cell_pair,
                                segment_type="normal"
                            ))
        
        self.logger.info(f"Built {len(self.boundary_segments)} boundary segments from triangle_segments")
    
    def evaluate_variable_point(self, var_point_idx: int) -> np.ndarray:
        """Get 3D/2D coordinates of a variable point."""
        return self.variable_points[var_point_idx].evaluate(self.mesh.vertices)
    
    def _build_segment_to_triangle_map(self) -> None:
        """Build inverse map: segment (vp1, vp2) -> triangle_idx for O(1) lookup."""
        self.segment_to_triangle = {}
        
        for tri_seg in self._triangle_segments.values():
            if len(tri_seg.var_point_indices) == 2:
                seg_key = tuple(sorted(tri_seg.var_point_indices))
                self.segment_to_triangle[seg_key] = tri_seg.triangle_idx
            
        self.logger.debug(f"Built segment_to_triangle map with {len(self.segment_to_triangle)} entries")
    
    def rebuild_segment_to_triangle_map(self) -> None:
        """
        Public method to rebuild the segment-to-triangle map.
        
        Call this after:
        - Type 1 topology switches
        - Type 2 topology switches  
        - Any manual modification to triangle_segments
        """
        self._build_segment_to_triangle_map()
        self.logger.info("Rebuilt segment_to_triangle map")
    
    def validate_segment_to_triangle_map(self) -> bool:
        """Validate that segment_to_triangle map is consistent with triangle_segments."""
        expected_count = sum(1 for ts in self._triangle_segments.values() if len(ts.var_point_indices) == 2)
        
        if len(self.segment_to_triangle) != expected_count:
            self.logger.error(f"Map size mismatch: {len(self.segment_to_triangle)} vs {expected_count} (2-VP segments)")
            return False
        
        for tri_seg in self._triangle_segments.values():
            if len(tri_seg.var_point_indices) == 2:
                seg_key = tuple(sorted(tri_seg.var_point_indices))
                mapped_tri = self.segment_to_triangle.get(seg_key)
                
                if mapped_tri != tri_seg.triangle_idx:
                    self.logger.error(f"Inconsistency: segment {seg_key} maps to {mapped_tri}, expected {tri_seg.triangle_idx}")
                    return False
        
        return True
    
    def get_triangle_based_segments(self) -> List[Tuple[int, int]]:
        """
        Extract all segments from triangle_segments.
        
        Returns list of (var_idx_i, var_idx_j) tuples for all contour segments.
        Each segment is returned once, even though it may belong to multiple cells.
        """
        segments = []
        seen_segments = set()
        vertex_labels = self._vertex_labels
        
        for tri_seg in self._triangle_segments.values():
            var_indices = tri_seg.var_point_indices
            
            if tri_seg.num_cells(vertex_labels) == 2:
                if len(var_indices) == 2:
                    seg = tuple(sorted(var_indices))
                    if seg not in seen_segments:
                        segments.append(seg)
                        seen_segments.add(seg)
            
            elif tri_seg.is_triple_point():
                if len(var_indices) == 3:
                    for i in range(3):
                        for j in range(i+1, 3):
                            seg = tuple(sorted([var_indices[i], var_indices[j]]))
                            if seg not in seen_segments:
                                segments.append(seg)
                                seen_segments.add(seg)
        
        return segments
    
    def get_cell_segments_from_triangles(self, cell_idx: int) -> List[Tuple[int, int]]:
        """
        Get segments for a specific cell from triangle_segments.
        
        Args:
            cell_idx: Index of the cell
            
        Returns:
            List of segment pairs (var_point_idx1, var_point_idx2) for this cell
        """
        segments = []
        seen_segments = set()
        vertex_labels = self._vertex_labels
        
        for tri_seg in self._triangle_segments.values():
            if cell_idx not in tri_seg.get_cell_indices(vertex_labels):
                continue
            
            var_indices = tri_seg.var_point_indices
            
            if tri_seg.num_cells(vertex_labels) == 2:
                if len(var_indices) == 2:
                    seg = tuple(sorted(var_indices))
                    if seg not in seen_segments:
                        segments.append(seg)
                        seen_segments.add(seg)
            
            elif tri_seg.is_triple_point():
                continue
        
        return segments
    
    def to_visualization_format(self) -> Dict[int, List[np.ndarray]]:
        """
        Export refined contours in the same format as ContourAnalyzer.extract_contours().
        
        Phase 4: Uses explicit boundary_segments for accurate visualization after
        topology switches. This ensures cross-triangle segments are properly rendered.
        
        Returns:
            Dict[region_idx] -> List[segment arrays (2, D)]
            where D is 2 or 3 depending on mesh dimension
        """
        contours_dict = {i: [] for i in range(self.n_cells)}
        
        # Phase 4: Use explicit boundary_segments for complete coverage
        if self.boundary_segments:
            for seg in self.boundary_segments:
                # Get positions of both VPs
                p1 = self.evaluate_variable_point(seg.vp_idx_1)
                p2 = self.evaluate_variable_point(seg.vp_idx_2)
                
                segment = np.vstack([p1, p2])
                
                # Add to both cells that this segment separates
                cell_a, cell_b = seg.cell_pair
                if cell_a < self.n_cells:
                    contours_dict[cell_a].append(segment)
                if cell_b < self.n_cells and cell_b != cell_a:
                    contours_dict[cell_b].append(segment)
            
            total_segments = sum(len(segs) for segs in contours_dict.values())
            self.logger.info(f"Converted to visualization format (Phase 4 boundary_segments): "
                            f"{total_segments} total segments")
        else:
            # Fallback to triangle-based extraction (for backward compatibility)
            self.logger.warning("No boundary_segments found, using fallback triangle-based extraction")
        vertex_labels = self._vertex_labels
        for tri_seg in self._triangle_segments.values():
            if tri_seg.num_cells(vertex_labels) == 2:
                if len(tri_seg.var_point_indices) == 2:
                    vp_idx1, vp_idx2 = tri_seg.var_point_indices
                    
                    p1 = self.evaluate_variable_point(vp_idx1)
                    p2 = self.evaluate_variable_point(vp_idx2)
                    
                    segment = np.vstack([p1, p2])
                    
                    cells_in_triangle = tri_seg.get_cell_indices(vertex_labels)
                    for cell_idx in cells_in_triangle:
                        contours_dict[cell_idx].append(segment)
            
            elif tri_seg.is_triple_point():
                if len(tri_seg.var_point_indices) == 3:
                    vp_idx1, vp_idx2, vp_idx3 = tri_seg.var_point_indices
                    
                    p1 = self.evaluate_variable_point(vp_idx1)
                    p2 = self.evaluate_variable_point(vp_idx2)
                    p3 = self.evaluate_variable_point(vp_idx3)
                    
                    seg12 = np.vstack([p1, p2])
                    seg23 = np.vstack([p2, p3])
                    seg31 = np.vstack([p3, p1])
                    
                    cells_in_triangle = tri_seg.get_cell_indices(vertex_labels)
                    for cell_idx in cells_in_triangle:
                        contours_dict[cell_idx].append(seg12)
                        contours_dict[cell_idx].append(seg23)
                        contours_dict[cell_idx].append(seg31)
        
        total_segments = sum(len(segs) for segs in contours_dict.values())
        self.logger.info(f"Converted to visualization format (Phase 3 fallback): "
                        f"{total_segments} total segments")
        
        return contours_dict
    
    # =========================================================================
    # Vectorized evaluation support
    # =========================================================================

    def compile_arrays(self, steiner_handler):
        """Compile flat arrays for vectorized evaluation during SLSQP.

        Walks the object-oriented structures once and builds a
        :class:`~partition_arrays.PartitionArrays` snapshot that the vectorized
        perimeter / area / steiner modules consume without any per-element
        Python overhead.

        Args:
            steiner_handler: :class:`~steiner_handler.SteinerHandler` that owns
                the detected ``triple_points`` list.  Required because
                ``PartitionContour`` does not own the Steiner data.

        Returns:
            :class:`~partition_arrays.PartitionArrays`
        """
        from .partition_arrays import PartitionArrays

        vertex_labels = self._vertex_labels

        # 1. Build absolute → active index map
        abs_to_active: Dict[int, int] = {}
        active_to_absolute_list: List[int] = []
        for opt_idx, abs_idx in enumerate(self._active_vp_indices):
            abs_to_active[abs_idx] = opt_idx
            active_to_absolute_list.append(abs_idx)

        n_active = len(active_to_absolute_list)
        active_to_absolute = np.array(active_to_absolute_list, dtype=np.int32)

        # 2. VP edge / lambda arrays (active only)
        vp_edge_v1 = np.empty(n_active, dtype=np.int32)
        vp_edge_v2 = np.empty(n_active, dtype=np.int32)
        vp_lambda = np.empty(n_active, dtype=np.float64)
        for opt_idx, abs_idx in enumerate(self._active_vp_indices):
            vp = self.variable_points[abs_idx]
            vp_edge_v1[opt_idx] = vp.edge[0]
            vp_edge_v2[opt_idx] = vp.edge[1]
            vp_lambda[opt_idx] = vp.lambda_param

        # 3. Boundary segment arrays
        seg_vp1_list, seg_vp2_list = [], []
        seg_ca_list, seg_cb_list = [], []
        for seg in self.boundary_segments:
            a_idx = abs_to_active.get(seg.vp_idx_1)
            b_idx = abs_to_active.get(seg.vp_idx_2)
            if a_idx is None or b_idx is None:
                continue
            seg_vp1_list.append(a_idx)
            seg_vp2_list.append(b_idx)
            seg_ca_list.append(seg.cell_pair[0])
            seg_cb_list.append(seg.cell_pair[1])

        seg_vp1 = np.array(seg_vp1_list, dtype=np.int32)
        seg_vp2 = np.array(seg_vp2_list, dtype=np.int32)
        seg_cell_a = np.array(seg_ca_list, dtype=np.int32)
        seg_cell_b = np.array(seg_cb_list, dtype=np.int32)

        # 4–5. Boundary triangle arrays + cell_interior_area
        btri_idx_l, btri_cell_l, btri_nin_l = [], [], []
        btri_vin_l, btri_vout_l = [], []
        btri_vp1_l, btri_vp2_l = [], []
        cell_interior_area = np.zeros(self.n_cells, dtype=np.float64)

        tri_areas = self.mesh.triangle_areas

        for tri_idx, face in enumerate(self.mesh.faces):
            v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
            lab1, lab2, lab3 = int(vertex_labels[v1]), int(vertex_labels[v2]), int(vertex_labels[v3])
            labels_set = {lab1, lab2, lab3}

            if len(labels_set) == 3:
                continue  # triple-point triangle — handled by vectorized_steiner

            verts = [v1, v2, v3]
            labs = [lab1, lab2, lab3]

            if len(labels_set) == 1:
                cell_interior_area[lab1] += tri_areas[tri_idx]
                continue

            # Boundary triangle: contributes to one or two cells
            for cell_idx in labels_set:
                inside_mask = [l == cell_idx for l in labs]
                n_inside = sum(inside_mask)
                if n_inside == 0 or n_inside == 3:
                    continue

                inside_verts = [verts[i] for i in range(3) if inside_mask[i]]
                outside_verts = [verts[i] for i in range(3) if not inside_mask[i]]

                if n_inside == 2:
                    v_in1, v_in2 = inside_verts
                    v_out = outside_verts[0]
                    edge_a = tuple(sorted([v_out, v_in1]))
                    edge_b = tuple(sorted([v_out, v_in2]))
                    vin_row = [v_in1, v_in2]
                    vout_row = [v_out, -1]
                elif n_inside == 1:
                    v_in = inside_verts[0]
                    v_out1, v_out2 = outside_verts
                    edge_a = tuple(sorted([v_in, v_out1]))
                    edge_b = tuple(sorted([v_in, v_out2]))
                    vin_row = [v_in, -1]
                    vout_row = [v_out1, v_out2]
                else:
                    continue

                vp_a_abs = self.edge_to_varpoint.get(edge_a)
                vp_b_abs = self.edge_to_varpoint.get(edge_b)
                if vp_a_abs is None or vp_b_abs is None:
                    continue
                vp_a = abs_to_active.get(vp_a_abs)
                vp_b = abs_to_active.get(vp_b_abs)
                if vp_a is None or vp_b is None:
                    continue

                btri_idx_l.append(tri_idx)
                btri_cell_l.append(cell_idx)
                btri_nin_l.append(n_inside)
                btri_vin_l.append(vin_row)
                btri_vout_l.append(vout_row)
                btri_vp1_l.append(vp_a)
                btri_vp2_l.append(vp_b)

        n_btri = len(btri_idx_l)
        btri_idx = np.array(btri_idx_l, dtype=np.int32) if n_btri else np.empty(0, dtype=np.int32)
        btri_cell = np.array(btri_cell_l, dtype=np.int32) if n_btri else np.empty(0, dtype=np.int32)
        btri_n_inside = np.array(btri_nin_l, dtype=np.int32) if n_btri else np.empty(0, dtype=np.int32)
        btri_v_in = np.array(btri_vin_l, dtype=np.int32).reshape(-1, 2) if n_btri else np.empty((0, 2), dtype=np.int32)
        btri_v_out = np.array(btri_vout_l, dtype=np.int32).reshape(-1, 2) if n_btri else np.empty((0, 2), dtype=np.int32)
        btri_vp1 = np.array(btri_vp1_l, dtype=np.int32) if n_btri else np.empty(0, dtype=np.int32)
        btri_vp2 = np.array(btri_vp2_l, dtype=np.int32) if n_btri else np.empty(0, dtype=np.int32)

        # 6. area_affected_vps
        if n_btri:
            area_affected_vps = np.unique(np.concatenate([btri_vp1, btri_vp2]))
        else:
            area_affected_vps = np.empty(0, dtype=np.int32)

        # 7. Triple-point arrays
        triple_points = steiner_handler.triple_points
        n_tp = len(triple_points)

        if n_tp > 0:
            tp_vp_indices_l = []
            tp_ctp_l, tp_ccell_l, tp_cvp1_l, tp_cvp2_l, tp_cmv_l = [], [], [], [], []

            for tp_i, tp in enumerate(triple_points):
                active_vps = [abs_to_active[vi] for vi in tp.var_point_indices]
                tp_vp_indices_l.append(active_vps)

                for cell_idx, (vp_abs_a, vp_abs_b) in tp.cell_to_varpoint_pair.items():
                    tp_ctp_l.append(tp_i)
                    tp_ccell_l.append(cell_idx)
                    tp_cvp1_l.append(abs_to_active[vp_abs_a])
                    tp_cvp2_l.append(abs_to_active[vp_abs_b])
                    tp_cmv_l.append(tp.cell_to_mesh_vertex[cell_idx])

            tp_vp_indices = np.array(tp_vp_indices_l, dtype=np.int32)
            tp_contrib_tp_idx = np.array(tp_ctp_l, dtype=np.int32)
            tp_contrib_cell = np.array(tp_ccell_l, dtype=np.int32)
            tp_contrib_vp1 = np.array(tp_cvp1_l, dtype=np.int32)
            tp_contrib_vp2 = np.array(tp_cvp2_l, dtype=np.int32)
            tp_contrib_mesh_vertex = np.array(tp_cmv_l, dtype=np.int32)
            tp_affected_vps = np.unique(tp_vp_indices.ravel())
        else:
            dim = self.mesh.vertices.shape[1]
            tp_vp_indices = np.empty((0, 3), dtype=np.int32)
            tp_contrib_tp_idx = np.empty(0, dtype=np.int32)
            tp_contrib_cell = np.empty(0, dtype=np.int32)
            tp_contrib_vp1 = np.empty(0, dtype=np.int32)
            tp_contrib_vp2 = np.empty(0, dtype=np.int32)
            tp_contrib_mesh_vertex = np.empty(0, dtype=np.int32)
            tp_affected_vps = np.empty(0, dtype=np.int32)

        self.logger.info(f"compile_arrays: {n_active} active VPs, "
                        f"{len(seg_vp1)} segments, {n_btri} boundary-triangle rows, "
                        f"{n_tp} triple points")

        # 8. Jacobian sparsity pattern for IPOPT
        n_constrained = self.n_cells - 1

        if n_btri > 0:
            btri_mask = btri_cell < n_constrained
            rows_reg = np.concatenate([btri_cell[btri_mask], btri_cell[btri_mask]])
            cols_reg = np.concatenate([btri_vp1[btri_mask], btri_vp2[btri_mask]])
        else:
            rows_reg = np.empty(0, dtype=np.int32)
            cols_reg = np.empty(0, dtype=np.int32)

        if n_tp > 0:
            tp_rows_l, tp_cols_l = [], []
            for tp_i in range(n_tp):
                vps_i = tp_vp_indices[tp_i]
                cells_i = tp_contrib_cell[tp_contrib_tp_idx == tp_i]
                for c in cells_i:
                    if c < n_constrained:
                        for v in vps_i:
                            tp_rows_l.append(c)
                            tp_cols_l.append(v)
            rows_tp = np.array(tp_rows_l, dtype=np.int32) if tp_rows_l else np.empty(0, dtype=np.int32)
            cols_tp = np.array(tp_cols_l, dtype=np.int32) if tp_cols_l else np.empty(0, dtype=np.int32)
        else:
            rows_tp = np.empty(0, dtype=np.int32)
            cols_tp = np.empty(0, dtype=np.int32)

        all_rows = np.concatenate([rows_reg, rows_tp])
        all_cols = np.concatenate([cols_reg, cols_tp])

        if len(all_rows) > 0:
            pairs = np.unique(
                np.stack([all_rows, all_cols], axis=1), axis=0
            )
            jac_row = pairs[:, 0].astype(np.int32)
            jac_col = pairs[:, 1].astype(np.int32)
        else:
            jac_row = np.empty(0, dtype=np.int32)
            jac_col = np.empty(0, dtype=np.int32)

        self.logger.info(f"  Jacobian sparsity: {len(jac_row)} non-zeros "
                        f"(density {100 * len(jac_row) / max(1, n_constrained * n_active):.1f}%)")

        # 8b. nnz_lookup: (row, col) → offset into flat Jacobian values array
        nnz = len(jac_row)
        nnz_lookup = -np.ones((n_constrained, n_active), dtype=np.int32)
        for idx in range(nnz):
            nnz_lookup[jac_row[idx], jac_col[idx]] = idx

        # 9. Hessian sparsity pattern (lower triangle: row >= col)
        hess_pairs = set()

        for s in range(len(seg_vp1)):
            a = int(seg_vp1[s])
            b = int(seg_vp2[s])
            hess_pairs.add((a, a))
            hess_pairs.add((b, b))
            hess_pairs.add((max(a, b), min(a, b)))

        for k in range(n_btri):
            a = int(btri_vp1[k])
            b = int(btri_vp2[k])
            hess_pairs.add((a, a))
            hess_pairs.add((b, b))
            hess_pairs.add((max(a, b), min(a, b)))

        for tp_i in range(n_tp):
            vps = tp_vp_indices[tp_i]
            for i in range(3):
                for j in range(i + 1):
                    hess_pairs.add((max(int(vps[i]), int(vps[j])),
                                    min(int(vps[i]), int(vps[j]))))

        if hess_pairs:
            hess_pairs_arr = np.array(sorted(hess_pairs), dtype=np.int32)
            hess_row = hess_pairs_arr[:, 0]
            hess_col = hess_pairs_arr[:, 1]
        else:
            hess_row = np.empty(0, dtype=np.int32)
            hess_col = np.empty(0, dtype=np.int32)

        hess_offset_map = {}
        for idx in range(len(hess_row)):
            hess_offset_map[(int(hess_row[idx]), int(hess_col[idx]))] = idx

        self.logger.info(f"  Hessian sparsity: {len(hess_row)} non-zeros (lower triangle)")

        return PartitionArrays(
            vp_edge_v1=vp_edge_v1,
            vp_edge_v2=vp_edge_v2,
            vp_lambda=vp_lambda,
            seg_vp1=seg_vp1,
            seg_vp2=seg_vp2,
            seg_cell_a=seg_cell_a,
            seg_cell_b=seg_cell_b,
            btri_idx=btri_idx,
            btri_cell=btri_cell,
            btri_n_inside=btri_n_inside,
            btri_v_in=btri_v_in,
            btri_v_out=btri_v_out,
            btri_vp1=btri_vp1,
            btri_vp2=btri_vp2,
            cell_interior_area=cell_interior_area,
            n_cells=self.n_cells,
            n_active_vp=n_active,
            active_to_absolute=active_to_absolute,
            area_affected_vps=area_affected_vps,
            vertices=self.mesh.vertices,
            tp_vp_indices=tp_vp_indices,
            n_triple_points=n_tp,
            tp_contrib_tp_idx=tp_contrib_tp_idx,
            tp_contrib_cell=tp_contrib_cell,
            tp_contrib_vp1=tp_contrib_vp1,
            tp_contrib_vp2=tp_contrib_vp2,
            tp_contrib_mesh_vertex=tp_contrib_mesh_vertex,
            tp_affected_vps=tp_affected_vps,
            jac_row=jac_row,
            jac_col=jac_col,
            nnz_lookup=nnz_lookup,
            hess_row=hess_row,
            hess_col=hess_col,
            hess_offset_map=hess_offset_map,
        )

    def identify_triple_points(self) -> List[Tuple[int, List[int]]]:
        """
        Identify mesh triangles where three different partition cells meet.
        
        Uses vertex_labels directly.
        
        Returns:
            List of (triangle_idx, [var_point_idx1, var_point_idx2, var_point_idx3])
        """
        triple_points = []
        vertex_labels = self._vertex_labels
        
        for tri_idx, face in enumerate(self.mesh.faces):
            v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
            labels = {int(vertex_labels[v1]), int(vertex_labels[v2]), int(vertex_labels[v3])}
            
            if len(labels) == 3:
                edges = [
                    tuple(sorted([v1, v2])),
                    tuple(sorted([v2, v3])),
                    tuple(sorted([v3, v1]))
                ]
                var_points = []
                for edge in edges:
                    if edge in self.edge_to_varpoint:
                        var_points.append(self.edge_to_varpoint[edge])
                
                if len(var_points) == 3:
                    triple_points.append((tri_idx, var_points))
        
        self.logger.info(f"Identified {len(triple_points)} triple points")
        return triple_points
    
    def identify_triple_points_from_current_vps(self) -> List[Tuple[int, List[int]]]:
        """
        Identify triple points based on CURRENT VP positions (works after topology switches).
        
        This method is the correct way to identify triple points after topology switches,
        as it uses actual VP positions and their belongs_to_cells attribute rather than
        stale indicator_functions.
        
        A triangle is a triple point if:
        1. It has exactly 3 variable points on its edges
        2. Those 3 VPs collectively separate 3 different partition cells
        
        Example from Type 2 switch:
        - After switch, T2 has VP1, VP3, VP4 (3 VPs)
        - VP1 separates cells {A, B}, VP3 separates {B, C}, VP4 separates {A, C}
        - All cells: {A, B, C} → 3 different cells → T2 is triple point
        
        Returns:
            List of (triangle_idx, [var_point_idx1, var_point_idx2, var_point_idx3])
        """
        triple_points = []
        
        # Iterate through all triangle segments (after rebuild, includes 1-VP triangles)
        for tri_seg in self.triangle_segments:
            # Check if triangle has exactly 3 VPs
            if len(tri_seg.var_point_indices) == 3:
                # Collect all cells that these 3 VPs separate
                all_cells = set()
                for vp_idx in tri_seg.var_point_indices:
                    vp = self.variable_points[vp_idx]
                    all_cells.update(vp.belongs_to_cells)
                
                # True triple point: 3 VPs collectively separate 3 different cells
                if len(all_cells) == 3:
                    triple_points.append((tri_seg.triangle_idx, tri_seg.var_point_indices))
        
        self.logger.info(f"Identified {len(triple_points)} triple points from current VP state")
        return triple_points
    
    def get_boundary_variable_points(self, tol: float = 1e-3) -> List[int]:
        """
        Find variable points near edge endpoints (candidates for topology switch).
        
        Args:
            tol: Threshold for considering a point at boundary
            
        Returns:
            List of variable point indices with λ < tol or λ > 1-tol
        """
        boundary_points = []
        for vp in self.variable_points:
            if vp.on_boundary(tol):
                boundary_points.append(vp.global_idx)
        
        return boundary_points

