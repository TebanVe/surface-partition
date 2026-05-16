"""
One-ring rebuild after a vertex flip (VP create/destroy/keep logic).

This is the core workhorse used by both Type 1 and Type 2 executors. Given a
vertex to flip and a new cell label, it enumerates all triangles in the 1-ring,
determines which VPs to destroy/create/keep, and updates the partition structures.
"""

import numpy as np
from typing import List, Tuple, Optional, Set

from ..logging_config import get_logger
from ..partition.contour_partition import PartitionContour, VariablePoint, TriangleSegment
from ..mesh.mesh_topology import MeshTopology
from .migration_types import RebuildResult

logger = get_logger(__name__)


def rebuild_one_ring(vertex: int,
                     new_cell: int,
                     partition: PartitionContour,
                     mesh_topology: MeshTopology,
                     is_type2: bool = False,
                     collapsed_edge: Optional[Tuple[int, int]] = None) -> RebuildResult:
    """
    Rebuild the 1-ring of a vertex after flipping its label to new_cell.
    
    Steps:
      1. Enumerate all triangles in the 1-ring
      2. Classify each edge from vertex as same-cell or different-cell under new label
      3. Determine which VPs to delete, keep, or create
      4. Update partition data structures
      5. Rebuild TriangleSegments for affected triangles
      6. Scan for new triple-point triangles
    
    Args:
        vertex: The vertex being flipped
        new_cell: The cell label to flip to
        partition: PartitionContour to modify in-place
        mesh_topology: Mesh connectivity
        is_type2: If True, allows new triple-point triangles and skips VP creation
                  on collapsed_edge (Type 2 executor handles it separately)
        collapsed_edge: Edge where Steiner collapsed (Type 2 only). Normalized tuple.
    """
    result = RebuildResult()
    vertex_labels = partition.vertex_labels
    old_cell = int(vertex_labels[vertex])
    
    ring_triangles = mesh_topology.get_triangles_at_vertex(vertex)
    result.affected_triangles = list(ring_triangles)
    
    edges_from_vertex = mesh_topology.get_edges_at_vertex(vertex)
    
    collapsed_normalized = tuple(sorted(collapsed_edge)) if collapsed_edge else None
    
    # Phase 1: Classify edges and determine VP fate
    to_destroy: List[int] = []
    to_keep: List[int] = []
    to_create_edges: List[Tuple[int, int]] = []
    
    for edge in edges_from_vertex:
        normalized = tuple(sorted(edge))
        other_v = edge[1] if edge[0] == vertex else edge[0]
        other_cell = int(vertex_labels[other_v])
        
        had_vp = normalized in partition.edge_to_varpoint
        needs_vp = (new_cell != other_cell)
        
        if is_type2 and collapsed_normalized and normalized == collapsed_normalized:
            if had_vp:
                vp_idx = partition.edge_to_varpoint[normalized]
                to_destroy.append(vp_idx)
            if needs_vp:
                result.collapsed_edge_needs_vp = True
            continue
        
        if had_vp and needs_vp:
            vp_idx = partition.edge_to_varpoint[normalized]
            to_keep.append(vp_idx)
        elif had_vp and not needs_vp:
            vp_idx = partition.edge_to_varpoint[normalized]
            to_destroy.append(vp_idx)
        elif not had_vp and needs_vp:
            to_create_edges.append(normalized)
        # not had_vp and not needs_vp: nothing to do
    
    logger.debug(f"1-ring rebuild at vertex {vertex} ({old_cell} -> {new_cell}): "
                f"destroy={len(to_destroy)}, keep={len(to_keep)}, create={len(to_create_edges)}")
    
    # Phase 2: Execute VP destruction
    for vp_idx in to_destroy:
        vp = partition.variable_points[vp_idx]
        vp.active = False
        normalized = tuple(sorted(vp.edge))
        partition.edge_to_varpoint.pop(normalized, None)
        result.destroyed_vps.append(vp_idx)
    
    # Phase 3: Update kept VPs' cell membership
    for vp_idx in to_keep:
        vp = partition.variable_points[vp_idx]
        other_v = vp.edge[1] if vp.edge[0] == vertex else vp.edge[0]
        other_cell = int(vertex_labels[other_v])
        vp.belongs_to_cells = {new_cell, other_cell}
        result.kept_vps.append(vp_idx)
    
    # Phase 4: Create new VPs
    for edge in to_create_edges:
        other_v = edge[1] if edge[0] == vertex else edge[0]
        other_cell = int(vertex_labels[other_v])
        
        new_vp = VariablePoint(
            edge=edge,
            lambda_param=0.5,
            global_idx=len(partition.variable_points),
            belongs_to_cells={new_cell, other_cell},
            active=True
        )
        partition.variable_points.append(new_vp)
        partition.edge_to_varpoint[edge] = new_vp.global_idx
        result.created_vps.append(new_vp.global_idx)
    
    # Phase 5: Flip the vertex label
    partition._vertex_labels[vertex] = new_cell
    
    # Phase 6: Rebuild TriangleSegments for affected triangles
    _rebuild_affected_triangle_segments(partition, result.affected_triangles)
    
    # Phase 7: Scan for new triple-point triangles
    vertex_labels = partition.vertex_labels
    for tri_idx in result.affected_triangles:
        face = partition.mesh.faces[tri_idx]
        v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
        labels = {int(vertex_labels[v1]), int(vertex_labels[v2]), int(vertex_labels[v3])}
        
        if len(labels) == 3:
            if is_type2:
                result.new_triple_triangle = tri_idx
                logger.debug(f"New triple-point triangle {tri_idx} created (expected for Type 2)")
            else:
                logger.error(f"Pure Type 1 flip created triple-point triangle {tri_idx}! "
                           f"Labels: v{v1}={vertex_labels[v1]}, v{v2}={vertex_labels[v2]}, "
                           f"v{v3}={vertex_labels[v3]}")
    
    # Phase 8: Rebuild VP adjacency for affected region
    partition._rebuild_vp_adjacency()
    
    logger.debug(f"1-ring rebuild complete: {len(result.destroyed_vps)} destroyed, "
               f"{len(result.created_vps)} created, {len(result.kept_vps)} kept")
    
    return result


def _rebuild_affected_triangle_segments(partition: PartitionContour,
                                         affected_triangles: List[int]):
    """Rebuild TriangleSegments for a set of affected triangles using current VP state."""
    edge_to_vp = {}
    for vp_idx, vp in enumerate(partition.variable_points):
        if not vp.active:
            continue
        normalized = tuple(sorted(vp.edge))
        edge_to_vp[normalized] = vp_idx
    
    for tri_idx in affected_triangles:
        partition._triangle_segments.pop(tri_idx, None)
        
        face = partition.mesh.faces[tri_idx]
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
        
        if var_point_indices:
            tri_seg = TriangleSegment(
                triangle_idx=tri_idx,
                vertex_indices=(v1, v2, v3),
                boundary_edges=boundary_edges,
                var_point_indices=var_point_indices
            )
            partition._triangle_segments[tri_idx] = tri_seg
