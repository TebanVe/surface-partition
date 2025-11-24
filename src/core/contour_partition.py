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
    
    Position: x = λ * v_start + (1-λ) * v_end
    
    As per paper Section 5: "each of these points belongs to at least two cells"
    because they are situated on mesh triangle edges that cross cell boundaries.
    
    Attributes:
        edge: Tuple of (vertex_idx_start, vertex_idx_end) on mesh triangle
        lambda_param: Parameter value in [0, 1]
        global_idx: Index in the global variable vector
        belongs_to_cells: Set of cell indices that this point belongs to (boundary between these cells)
    """
    edge: Tuple[int, int]
    lambda_param: float
    global_idx: int
    belongs_to_cells: Set[int]
    
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
        vertex_labels: Tuple of 3 cell labels for the vertices
        boundary_edges: List of edges that cross cell boundaries (normalized: smaller index first)
        var_point_indices: List of variable point indices corresponding to boundary_edges
    """
    triangle_idx: int
    vertex_indices: Tuple[int, int, int]
    vertex_labels: Tuple[int, int, int]
    boundary_edges: List[Tuple[int, int]]
    var_point_indices: List[int]
    
    def num_cells(self) -> int:
        """Return number of distinct cells in this triangle."""
        return len(set(self.vertex_labels))
    
    def is_triple_point(self) -> bool:
        """Check if this is a triple point (3 different cells meet)."""
        return self.num_cells() == 3
    
    def get_cell_indices(self) -> Set[int]:
        """Get set of unique cell indices in this triangle."""
        return set(self.vertex_labels)


class PartitionContour:
    """
    Global partition representation with variable points on mesh edges.
    
    This is the main data structure for Section 5 optimization. It manages:
    - All variable points across the partition
    - Triangle-based contour segment extraction
    - Topology information (which mesh triangle edges form which cell boundaries)
    - Conversion to/from indicator functions
    
    Attributes:
        mesh: The underlying TriMesh
        n_cells: Number of partition cells
        variable_points: List of all VariablePoint objects
        triangle_segments: List of TriangleSegment objects (triangle-based storage)
        indicator_functions: (N, n_cells) array of φ_i from equation (5.1)
        edge_to_varpoint: Map from mesh triangle edge tuple to variable point index
        triple_points: List of detected triple points (computed on demand)
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
        self.indicator_functions = indicator_functions
        self.n_cells = indicator_functions.shape[1]
        
        # Global data structures
        self.variable_points: List[VariablePoint] = []
        self.triangle_segments: List[TriangleSegment] = []  # Triangle-centric storage
        self.edge_to_varpoint: Dict[Tuple[int, int], int] = {}
        self.triple_points: Optional[List] = None  # Computed on demand
        
        # Choose initialization method based on available data
        if boundary_topology is not None:
            self.logger.info("Initializing from pre-computed boundary topology (efficient path)")
            self._initialize_from_boundary_topology(boundary_topology)
        else:
            self.logger.info("Initializing from indicator functions (scanning all triangles)")
            self._initialize_from_indicators()
        
        self.logger.info(f"Initialized PartitionContour: {len(self.variable_points)} variable points, "
                        f"{self.n_cells} partition cells")
    
    def _initialize_from_indicators(self):
        """
        Extract contours from indicator functions by finding mesh triangle edges
        that cross partition cell boundaries (where φ_i changes from 0 to 1).
        
        Uses triangle-based segment storage for geometrically valid contour extraction.
        """
        vertex_labels = np.argmax(self.indicator_functions, axis=1)
        
        # Iterate over all mesh triangles to find boundary edges
        for tri_idx, face in enumerate(self.mesh.faces):
            v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
            label1, label2, label3 = vertex_labels[v1], vertex_labels[v2], vertex_labels[v3]
            
            # Track boundary edges and variable points for this triangle
            boundary_edges_in_triangle = []
            var_points_in_triangle = []
            
            # Check each edge of the mesh triangle
            edges = [(v1, v2), (v2, v3), (v3, v1)]
            labels = [(label1, label2), (label2, label3), (label3, label1)]
            
            for edge, (lab_a, lab_b) in zip(edges, labels):
                if lab_a != lab_b:
                    # This mesh triangle edge crosses a cell boundary
                    # Normalize edge representation (smaller index first)
                    normalized_edge = tuple(sorted(edge))
                    
                    if normalized_edge not in self.edge_to_varpoint:
                        # Create new variable point at midpoint (λ = 0.5)
                        var_point = VariablePoint(
                            edge=normalized_edge,
                            lambda_param=0.5,
                            global_idx=len(self.variable_points),
                            belongs_to_cells={lab_a, lab_b}
                        )
                        self.variable_points.append(var_point)
                        self.edge_to_varpoint[normalized_edge] = var_point.global_idx
                        var_point_idx = var_point.global_idx
                    else:
                        # Update cell membership if this mesh triangle edge appears in multiple triangles
                        var_idx = self.edge_to_varpoint[normalized_edge]
                        self.variable_points[var_idx].belongs_to_cells.update([lab_a, lab_b])
                        var_point_idx = var_idx
                    
                    # Add to this triangle's boundary edges
                    boundary_edges_in_triangle.append(normalized_edge)
                    var_points_in_triangle.append(var_point_idx)
            
            # Create TriangleSegment if this triangle has boundary edges
            if boundary_edges_in_triangle:
                tri_seg = TriangleSegment(
                    triangle_idx=tri_idx,
                    vertex_indices=(v1, v2, v3),
                    vertex_labels=(label1, label2, label3),
                    boundary_edges=boundary_edges_in_triangle,
                    var_point_indices=var_points_in_triangle
                )
                self.triangle_segments.append(tri_seg)
        
        # Log triangle segment statistics
        num_two_cell = sum(1 for ts in self.triangle_segments if ts.num_cells() == 2)
        num_triple = sum(1 for ts in self.triangle_segments if ts.is_triple_point())
        self.logger.info(f"Created {len(self.triangle_segments)} triangle segments: "
                        f"{num_two_cell} two-cell, {num_triple} triple-point")
    
    def _initialize_from_boundary_topology(self, boundary_topology: Dict):
        """
        Initialize from pre-computed boundary topology (efficient path).
        
        This method avoids redundant triangle scanning by reusing topology information
        already computed by ContourAnalyzer.extract_contours_with_topology().
        
        Two-pass structure:
        - Pass 1: Create unique variable points on boundary edges
        - Pass 2: Create triangle segments linking triangles to their variable points
        
        Why two passes:
        - Each edge can be shared by multiple triangles (typically 2)
        - We need ONE variable point per unique edge (for optimization)
        - We need triangle_segments[] for segment extraction and area calculation
        
        Args:
            boundary_topology: Dict[region_idx] -> List[BoundaryTriangleInfo]
        """
        # PASS 1: Create unique variable points on all boundary edges
        # Scan all regions to find all unique edges and which cells they separate
        all_edges_info = {}  # edge -> set of cells on this edge
        
        for region_idx, tri_infos in boundary_topology.items():
            for tri_info in tri_infos:
                # For each crossed edge in this triangle
                for edge in tri_info.crossed_edges:
                    normalized_edge = tuple(sorted(edge))
                    
                    # Determine which cells this edge separates
                    # by examining the vertex labels at the edge endpoints
                    v_start_idx = tri_info.vertices.index(edge[0])
                    v_end_idx = tri_info.vertices.index(edge[1])
                    label_start = tri_info.vertex_labels[v_start_idx]
                    label_end = tri_info.vertex_labels[v_end_idx]
                    
                    if normalized_edge not in all_edges_info:
                        all_edges_info[normalized_edge] = set()
                    all_edges_info[normalized_edge].update([label_start, label_end])
        
        # Create variable points for all unique edges
        for normalized_edge, cells in all_edges_info.items():
            var_point = VariablePoint(
                edge=normalized_edge,
                lambda_param=0.5,  # Initial position at midpoint
                global_idx=len(self.variable_points),
                belongs_to_cells=cells
            )
            self.variable_points.append(var_point)
            self.edge_to_varpoint[normalized_edge] = var_point.global_idx
        
        self.logger.info(f"Pass 1 complete: Created {len(self.variable_points)} unique variable points")
        
        # PASS 2: Create triangle segments
        # This maps each triangle to its variable points
        # Essential for: segment extraction, area calculation, triple point detection
        for region_idx, tri_infos in boundary_topology.items():
            for tri_info in tri_infos:
                # Find variable point indices for this triangle's crossed edges
                var_point_indices = []
                boundary_edges_normalized = []
                
                for edge in tri_info.crossed_edges:
                    normalized_edge = tuple(sorted(edge))
                    var_point_indices.append(self.edge_to_varpoint[normalized_edge])
                    boundary_edges_normalized.append(normalized_edge)
                
                # Create triangle segment
                tri_seg = TriangleSegment(
                    triangle_idx=tri_info.triangle_idx,
                    vertex_indices=tri_info.vertices,
                    vertex_labels=tri_info.vertex_labels,
                    boundary_edges=boundary_edges_normalized,
                    var_point_indices=var_point_indices
                )
                self.triangle_segments.append(tri_seg)
        
        # Log statistics
        num_two_cell = sum(1 for ts in self.triangle_segments if ts.num_cells() == 2)
        num_triple = sum(1 for ts in self.triangle_segments if ts.is_triple_point())
        self.logger.info(f"Pass 2 complete: Created {len(self.triangle_segments)} triangle segments: "
                        f"{num_two_cell} two-cell, {num_triple} triple-point")
    
    def get_variable_vector(self) -> np.ndarray:
        """
        Return current λ parameters as optimization vector.
        
        Returns:
            Array of shape (n_variable_points,) with λ values
        """
        return np.array([vp.lambda_param for vp in self.variable_points])
    
    def set_variable_vector(self, lambda_vec: np.ndarray):
        """
        Update all λ parameters from optimization vector.
        
        Args:
            lambda_vec: Array of shape (n_variable_points,) with new λ values
        """
        if len(lambda_vec) != len(self.variable_points):
            raise ValueError(f"Lambda vector size {len(lambda_vec)} doesn't match "
                           f"number of variable points {len(self.variable_points)}")
        
        for i, lam in enumerate(lambda_vec):
            self.variable_points[i].lambda_param = float(np.clip(lam, 0.0, 1.0))
    
    def rebuild_triangle_segments_from_current_vps(self):
        """
        Rebuild triangle_segments list based on current variable point positions.
        
        CRITICAL for topology switching: After VPs move to new edges, the triangle_segments
        list becomes stale. This method re-scans the mesh and rebuilds the list based on
        current VP locations.
        
        This preserves existing variable_points and their lambda values - it only updates
        the triangle-to-VP mapping (which triangles contain which VPs).
        """
        self.logger.info("Rebuilding triangle_segments from current VPs...")
        
        # Clear existing triangle segments
        self.triangle_segments = []
        
        # Create map: edge -> VP index for quick lookup
        edge_to_vp = {}
        for vp_idx, vp in enumerate(self.variable_points):
            normalized_edge = tuple(sorted(vp.edge))
            edge_to_vp[normalized_edge] = vp_idx
        
        # Get vertex labels for checking boundary triangles
        vertex_labels = np.argmax(self.indicator_functions, axis=1)
        
        # Re-scan all mesh triangles
        for tri_idx, face in enumerate(self.mesh.faces):
            v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
            labels = [vertex_labels[v1], vertex_labels[v2], vertex_labels[v3]]
            
            # Check if this is a boundary triangle (mixed labels)
            if len(set(labels)) > 1:
                # This triangle has boundaries - find VPs on its edges
                tri_edges = [
                    tuple(sorted([v1, v2])),
                    tuple(sorted([v2, v3])),
                    tuple(sorted([v3, v1]))
                ]
                
                # Find which edges have VPs
                boundary_edges = []
                var_point_indices = []
                
                for edge in tri_edges:
                    if edge in edge_to_vp:
                        boundary_edges.append(edge)
                        var_point_indices.append(edge_to_vp[edge])
                
                # Only create TriangleSegment if triangle has VPs on boundary
                if len(var_point_indices) >= 2:
                    # Create TriangleSegment
                    tri_seg = TriangleSegment(
                        triangle_idx=tri_idx,
                        vertex_indices=(v1, v2, v3),
                        vertex_labels=tuple(labels),
                        boundary_edges=boundary_edges,
                        var_point_indices=var_point_indices
                    )
                    self.triangle_segments.append(tri_seg)
        
        # Log statistics
        num_two_cell = sum(1 for ts in self.triangle_segments if ts.num_cells() == 2)
        num_triple = sum(1 for ts in self.triangle_segments if ts.is_triple_point())
        self.logger.info(f"Rebuilt {len(self.triangle_segments)} triangle segments: "
                        f"{num_two_cell} two-cell, {num_triple} triple-point")
    
    def evaluate_variable_point(self, var_point_idx: int) -> np.ndarray:
        """Get 3D/2D coordinates of a variable point."""
        return self.variable_points[var_point_idx].evaluate(self.mesh.vertices)
    
    def get_triangle_based_segments(self) -> List[Tuple[int, int]]:
        """
        NEW - Phase 1: Extract all segments from triangle_segments.
        
        Returns list of (var_idx_i, var_idx_j) tuples for all contour segments.
        Each segment is returned once, even though it may belong to multiple cells.
        
        This method will be used in Phase 2 to refactor PerimeterCalculator.
        
        Returns:
            List of unique segment pairs as (var_point_idx1, var_point_idx2)
        """
        segments = []
        seen_segments = set()
        
        for tri_seg in self.triangle_segments:
            var_indices = tri_seg.var_point_indices
            
            if tri_seg.num_cells() == 2:
                # Two-cell triangle: one segment connecting the two variable points
                if len(var_indices) == 2:
                    seg = tuple(sorted(var_indices))
                    if seg not in seen_segments:
                        segments.append(seg)
                        seen_segments.add(seg)
            
            elif tri_seg.is_triple_point():
                # Triple-point triangle: three segments forming a small triangle
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
        NEW - Phase 1: Get segments for a specific cell from triangle_segments.
        
        This is a replacement for CellContour.get_segments() that will be used
        in Phase 2 to refactor PerimeterCalculator.
        
        Args:
            cell_idx: Index of the cell
            
        Returns:
            List of segment pairs (var_point_idx1, var_point_idx2) for this cell
        """
        segments = []
        seen_segments = set()
        
        for tri_seg in self.triangle_segments:
            # Only process triangles that involve this cell
            if cell_idx not in tri_seg.get_cell_indices():
                continue
            
            var_indices = tri_seg.var_point_indices
            
            if tri_seg.num_cells() == 2:
                # Two-cell triangle: add the segment if not already seen
                if len(var_indices) == 2:
                    seg = tuple(sorted(var_indices))
                    if seg not in seen_segments:
                        segments.append(seg)
                        seen_segments.add(seg)
            
            elif tri_seg.is_triple_point():
                # Triple-point triangle: segments handled entirely by SteinerHandler
                # Per paper Figure 7: each cell gets (2 Steiner edges - 1 original edge)
                # Do NOT add original triangle edges to regular perimeter calculation
                continue  # Skip this triangle
        
        return segments
    
    def to_visualization_format(self) -> Dict[int, List[np.ndarray]]:
        """
        Export refined contours in the same format as ContourAnalyzer.extract_contours().
        
        Phase 3: Uses triangle-based extraction to ensure geometrically valid segments.
        This method re-extracts contour segments from triangles using the current λ values,
        guaranteeing that segments are continuous and follow the surface correctly.
        
        Returns:
            Dict[region_idx] -> List[segment arrays (2, D)]
            where D is 2 or 3 depending on mesh dimension
        """
        contours_dict = {i: [] for i in range(self.n_cells)}
        
        # Phase 3: Triangle-based extraction
        for tri_seg in self.triangle_segments:
            if tri_seg.num_cells() == 2:
                # Two-cell triangle: compute level-set segment using current λ values
                if len(tri_seg.var_point_indices) == 2:
                    vp_idx1, vp_idx2 = tri_seg.var_point_indices
                    
                    p1 = self.evaluate_variable_point(vp_idx1)
                    p2 = self.evaluate_variable_point(vp_idx2)
                    
                    segment = np.vstack([p1, p2])
                    
                    # Add to both cells that share this boundary
                    cells_in_triangle = tri_seg.get_cell_indices()
                    for cell_idx in cells_in_triangle:
                        contours_dict[cell_idx].append(segment)
            
            elif tri_seg.is_triple_point():
                # Triple-point triangle: create small triangle connecting 3 variable points
                if len(tri_seg.var_point_indices) == 3:
                    vp_idx1, vp_idx2, vp_idx3 = tri_seg.var_point_indices
                    
                    p1 = self.evaluate_variable_point(vp_idx1)
                    p2 = self.evaluate_variable_point(vp_idx2)
                    p3 = self.evaluate_variable_point(vp_idx3)
                    
                    # Create three segments forming a small triangle
                    seg12 = np.vstack([p1, p2])
                    seg23 = np.vstack([p2, p3])
                    seg31 = np.vstack([p3, p1])
                    
                    # Add all three segments to all three cells
                    cells_in_triangle = tri_seg.get_cell_indices()
                    for cell_idx in cells_in_triangle:
                        contours_dict[cell_idx].append(seg12)
                        contours_dict[cell_idx].append(seg23)
                        contours_dict[cell_idx].append(seg31)
        
        total_segments = sum(len(segs) for segs in contours_dict.values())
        self.logger.info(f"Converted to visualization format (Phase 3 triangle-based): "
                        f"{total_segments} total segments")
        
        return contours_dict
    
    def save_refined_contours(self, output_path: str, 
                             perimeter: float,
                             areas: List[float],
                             optimization_info: Dict):
        """
        Save refined contours to HDF5 for visualization and analysis.
        
        Stores:
        - Optimized λ parameters
        - Evaluated contour segments (for visualization)
        - Triple point information
        - Optimization metadata (perimeter, areas, convergence info)
        
        Args:
            output_path: Path to HDF5 file
            perimeter: Final optimized total perimeter
            areas: List of cell areas
            optimization_info: Dict with optimization metadata
        """
        with h5py.File(output_path, 'w') as f:
            # Global metadata
            f.attrs['n_cells'] = self.n_cells
            f.attrs['n_variable_points'] = len(self.variable_points)
            f.attrs['final_perimeter'] = float(perimeter)
            f.attrs['target_area'] = float(optimization_info.get('target_area', 0.0))
            f.attrs['optimization_success'] = bool(optimization_info.get('success', False))
            f.attrs['n_iterations'] = int(optimization_info.get('n_iterations', 0))
            f.attrs['mesh_dimension'] = int(self.mesh.dim)
            
            # Save λ parameters
            lambda_vec = self.get_variable_vector()
            f.create_dataset('lambda_parameters', data=lambda_vec)
            
            # Save variable point metadata
            vp_grp = f.create_group('variable_points')
            for i, vp in enumerate(self.variable_points):
                vp_subgrp = vp_grp.create_group(f'vp_{i}')
                vp_subgrp.attrs['edge_start'] = vp.edge[0]
                vp_subgrp.attrs['edge_end'] = vp.edge[1]
                vp_subgrp.attrs['lambda'] = vp.lambda_param
                vp_subgrp.attrs['belongs_to_cells'] = list(vp.belongs_to_cells)
            
            # Save evaluated contours in visualization format
            viz_contours = self.to_visualization_format()
            for cell_idx, segments in viz_contours.items():
                grp = f.create_group(f'cell_{cell_idx}')
                grp.attrs['n_segments'] = len(segments)
                grp.attrs['area'] = float(areas[cell_idx]) if cell_idx < len(areas) else 0.0
                for seg_idx, seg in enumerate(segments):
                    grp.create_dataset(f'segment_{seg_idx}', data=seg)
            
            # Save triple points info if available
            if self.triple_points is not None and len(self.triple_points) > 0:
                tp_grp = f.create_group('triple_points')
                tp_grp.attrs['n_triple_points'] = len(self.triple_points)
                # Triple point details will be filled in by steiner_handler module
        
        self.logger.info(f"Saved refined contours to: {output_path}")
    
    @staticmethod
    def load_refined_contours(input_path: str) -> Dict[int, List[np.ndarray]]:
        """
        Load refined contours from HDF5 file in visualization format.
        
        Args:
            input_path: Path to HDF5 file created by save_refined_contours()
            
        Returns:
            Dict[cell_idx] -> List[segment arrays (2, D)]
        """
        with h5py.File(input_path, 'r') as f:
            n_cells = int(f.attrs['n_cells'])
            contours = {}
            
            for cell_idx in range(n_cells):
                grp = f[f'cell_{cell_idx}']
                n_segments = int(grp.attrs['n_segments'])
                segments = []
                for seg_idx in range(n_segments):
                    seg = grp[f'segment_{seg_idx}'][:]
                    segments.append(seg)
                contours[cell_idx] = segments
        
        return contours
    
    def identify_triple_points(self) -> List[Tuple[int, List[int]]]:
        """
        Identify mesh triangles where three different partition cells meet (triple points).
        
        These are mesh triangles with three variable points from three different cells,
        creating small void spaces that need Steiner tree treatment (Section 5).
        
        Returns:
            List of (triangle_idx, [var_point_idx1, var_point_idx2, var_point_idx3])
        """
        triple_points = []
        vertex_labels = np.argmax(self.indicator_functions, axis=1)
        
        for tri_idx, face in enumerate(self.mesh.faces):
            v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
            labels = {vertex_labels[v1], vertex_labels[v2], vertex_labels[v3]}
            
            # Triple point: all 3 vertices belong to different partition cells
            if len(labels) == 3:
                # Find the 3 variable points on this mesh triangle's edges
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

