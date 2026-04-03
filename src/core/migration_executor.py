"""
Snapshot management and migration execution (Type 1 + Type 2 forward/rollback).

Three sections:
A) Snapshot capture/restore for local state rollback
B) Type 1 execution: flip + rebuild + validation
C) Type 2 execution: Steiner teardown, Type 1 rebuild, Steiner-to-VP conversion,
   new Steiner setup, and snapshot-based rollback
"""

import copy
import numpy as np
from typing import Dict, List, Optional, Set, Tuple, FrozenSet

from ..logging_config import get_logger
from .contour_partition import PartitionContour, VariablePoint, TriangleSegment
from .mesh_topology import MeshTopology
from .steiner_handler import SteinerHandler, TriplePoint
from .migration_types import (
    Type1Trigger, Type2Trigger, LocalStateSnapshot, SteinerSnapshot,
    TriplePointHistory, RebuildResult
)
from .one_ring_rebuilder import rebuild_one_ring

logger = get_logger(__name__)


# ============================================================================
# Section A: Snapshot Management
# ============================================================================

def capture_snapshot(vertex: int,
                     partition: PartitionContour,
                     mesh_topology: MeshTopology,
                     steiner_handler: Optional[SteinerHandler] = None,
                     source_triangle: int = -1,
                     target_triangle: int = -1) -> LocalStateSnapshot:
    """
    Capture a complete local state snapshot of the 1-ring for rollback.
    
    Reads vertex_labels, VP data, triangle_segments, vp_adjacency for all
    vertices and edges in the 1-ring of the given vertex.
    """
    ring_triangles = mesh_topology.get_triangles_at_vertex(vertex)
    
    ring_vertices: Set[int] = set()
    for tri_idx in ring_triangles:
        face = partition.mesh.faces[tri_idx]
        ring_vertices.update(int(face[i]) for i in range(3))
    
    vertex_labels_snap: Dict[int, int] = {}
    for v in ring_vertices:
        vertex_labels_snap[v] = int(partition.vertex_labels[v])
    
    affected_edges: Set[Tuple[int, int]] = set()
    for tri_idx in ring_triangles:
        face = partition.mesh.faces[tri_idx]
        vs = [int(face[i]) for i in range(3)]
        for i in range(3):
            for j in range(i + 1, 3):
                affected_edges.add(tuple(sorted([vs[i], vs[j]])))
    
    vp_data: Dict[Tuple[int, int], float] = {}
    vp_existence: Set[Tuple[int, int]] = set()
    vp_active_flags: Dict[int, bool] = {}
    vp_cells: Dict[Tuple[int, int], FrozenSet[int]] = {}
    
    for edge in affected_edges:
        vp_idx = partition.edge_to_varpoint.get(edge)
        if vp_idx is not None:
            vp = partition.variable_points[vp_idx]
            vp_data[edge] = vp.lambda_param
            vp_existence.add(edge)
            vp_active_flags[vp_idx] = vp.active
            vp_cells[edge] = frozenset(vp.belongs_to_cells)
    
    vp_adjacency_subset: Dict[int, Set[int]] = {}
    for edge in affected_edges:
        vp_idx = partition.edge_to_varpoint.get(edge)
        if vp_idx is not None:
            neighbors = partition.get_vp_neighbors(vp_idx)
            vp_adjacency_subset[vp_idx] = set(neighbors)
    
    triangle_segments_data: Dict[int, dict] = {}
    for tri_idx in ring_triangles:
        ts = partition.get_triangle_segment(tri_idx)
        if ts is not None:
            triangle_segments_data[tri_idx] = {
                'vertex_indices': ts.vertex_indices,
                'boundary_edges': list(ts.boundary_edges),
                'var_point_indices': list(ts.var_point_indices)
            }
    
    steiner_snap = None
    if steiner_handler:
        steiner_snap = _capture_steiner_snapshot(ring_triangles, steiner_handler, partition)
    
    return LocalStateSnapshot(
        flipped_vertex=vertex,
        source_triangle=source_triangle,
        target_triangle=target_triangle,
        vertex_labels=vertex_labels_snap,
        vp_data=vp_data,
        vp_existence=vp_existence,
        vp_active_flags=vp_active_flags,
        vp_cells=vp_cells,
        vp_adjacency_subset=vp_adjacency_subset,
        triangle_segments_data=triangle_segments_data,
        steiner_data=steiner_snap
    )


def _capture_steiner_snapshot(ring_triangles: List[int],
                              steiner_handler: SteinerHandler,
                              partition: PartitionContour) -> Optional[SteinerSnapshot]:
    """Capture Steiner infrastructure for any triple-point triangle in the ring."""
    ring_set = set(ring_triangles)
    for tp in steiner_handler.triple_points:
        if tp.triangle_idx in ring_set:
            vp_positions = [partition.evaluate_variable_point(vi) for vi in tp.var_point_indices]
            if tp.steiner_point is None:
                tp.compute_steiner_point(vp_positions=vp_positions)
            
            void_pairs = []
            vps = tp.var_point_indices
            for i in range(3):
                for j in range(i + 1, 3):
                    void_pairs.append((vps[i], vps[j]))
            
            arm_endpoints = [(np.copy(tp.steiner_point), vi) for vi in vps]
            
            return SteinerSnapshot(
                triple_triangle=tp.triangle_idx,
                steiner_position=np.copy(tp.steiner_point),
                vp_indices=tuple(vps),
                cell_indices=tuple(tp.cell_indices),
                void_side_pairs=void_pairs,
                arm_endpoints=arm_endpoints
            )
    return None


def restore_snapshot(snapshot: LocalStateSnapshot,
                     partition: PartitionContour,
                     mesh_topology: MeshTopology,
                     steiner_handler: Optional[SteinerHandler] = None):
    """
    Restore partition state from a LocalStateSnapshot.
    
    Writes back vertex labels, VP data, triangle segments, vp adjacency.
    If steiner_data exists and steiner_handler provided, reconstructs Steiner tree.
    """
    for v, label in snapshot.vertex_labels.items():
        partition._vertex_labels[v] = label
    
    for edge in snapshot.vp_existence:
        vp_idx = partition.edge_to_varpoint.get(edge)
        if vp_idx is not None:
            vp = partition.variable_points[vp_idx]
            vp.lambda_param = snapshot.vp_data[edge]
            vp.active = snapshot.vp_active_flags.get(vp_idx, True)
            if edge in snapshot.vp_cells:
                vp.belongs_to_cells = set(snapshot.vp_cells[edge])
        else:
            for vi, vp in enumerate(partition.variable_points):
                if tuple(sorted(vp.edge)) == edge:
                    vp.lambda_param = snapshot.vp_data[edge]
                    vp.active = snapshot.vp_active_flags.get(vi, True)
                    if vp.active:
                        partition.edge_to_varpoint[edge] = vi
                    if edge in snapshot.vp_cells:
                        vp.belongs_to_cells = set(snapshot.vp_cells[edge])
                    break
    
    for vp_idx, was_active in snapshot.vp_active_flags.items():
        if vp_idx < len(partition.variable_points):
            vp = partition.variable_points[vp_idx]
            vp.active = was_active
            edge = tuple(sorted(vp.edge))
            if was_active:
                partition.edge_to_varpoint[edge] = vp_idx
            else:
                partition.edge_to_varpoint.pop(edge, None)
    
    for tri_idx, ts_data in snapshot.triangle_segments_data.items():
        partition._triangle_segments[tri_idx] = TriangleSegment(
            triangle_idx=tri_idx,
            vertex_indices=ts_data['vertex_indices'],
            boundary_edges=ts_data['boundary_edges'],
            var_point_indices=ts_data['var_point_indices']
        )
    
    partition._rebuild_vp_adjacency()
    
    if steiner_handler and snapshot.steiner_data:
        _restore_steiner(snapshot.steiner_data, steiner_handler, partition)
    
    logger.info(f"Restored snapshot for vertex {snapshot.flipped_vertex}")


def _restore_steiner(steiner_snap: SteinerSnapshot,
                     steiner_handler: SteinerHandler,
                     partition: PartitionContour):
    """Reconstruct a Steiner tree from a SteinerSnapshot."""
    steiner_handler.triple_points = [
        tp for tp in steiner_handler.triple_points
        if tp.triangle_idx != steiner_snap.triple_triangle
    ]
    
    vertex_labels_at_tri = tuple(
        int(partition.vertex_labels[v])
        for v in partition.mesh.faces[steiner_snap.triple_triangle].astype(int)
    )
    face = partition.mesh.faces[steiner_snap.triple_triangle]
    vertex_indices = tuple(int(face[i]) for i in range(3))
    
    tp = TriplePoint(
        triangle_idx=steiner_snap.triple_triangle,
        var_point_indices=list(steiner_snap.vp_indices),
        cell_indices=list(steiner_snap.cell_indices),
        vertex_indices=vertex_indices,
        vertex_labels=vertex_labels_at_tri
    )
    tp._compute_cell_to_varpoint_mapping(partition)
    tp.steiner_point = np.copy(steiner_snap.steiner_position)
    tp.boundary_points = [
        partition.evaluate_variable_point(vi) for vi in steiner_snap.vp_indices
    ]
    
    steiner_handler.triple_points.append(tp)


# ============================================================================
# Section B: Type 1 Execution
# ============================================================================

def execute_type1(trigger: Type1Trigger,
                  partition: PartitionContour,
                  mesh_topology: MeshTopology) -> bool:
    """
    Execute a Type 1 migration: flip vertex label + rebuild 1-ring.
    
    Returns True on success, False if the flip created an invalid triple-point triangle.
    """
    logger.info(f"Executing Type 1: vertex {trigger.vertex} "
               f"({trigger.current_cell} -> {trigger.target_cell})")
    
    result = rebuild_one_ring(
        vertex=trigger.vertex,
        new_cell=trigger.target_cell,
        partition=partition,
        mesh_topology=mesh_topology,
        is_type2=False
    )
    
    if result.new_triple_triangle is not None:
        logger.error(f"Type 1 flip at vertex {trigger.vertex} created triple-point "
                    f"triangle {result.new_triple_triangle}. Migration invalid.")
        return False
    
    partition._build_segment_connectivity(force_rebuild=True)
    partition._build_segment_to_triangle_map()
    
    logger.info(f"Type 1 complete: {len(result.destroyed_vps)} destroyed, "
               f"{len(result.created_vps)} created, {len(result.kept_vps)} kept")
    return True


# ============================================================================
# Section C: Type 2 Execution (Forward + Rollback)
# ============================================================================

def execute_type2_forward(trigger: Type2Trigger,
                          partition: PartitionContour,
                          mesh_topology: MeshTopology,
                          steiner_handler: SteinerHandler,
                          history: TriplePointHistory) -> bool:
    """
    Execute a Type 2 forward migration.
    
    Steps:
      1. Capture snapshot
      2. Tear down old Steiner infrastructure in T
      3. Execute Type 1 rebuild (with is_type2=True, skipping collapsed edge)
      4. Convert Steiner point S to VP on collapsed edge (net +1 VP)
      5. Identify new triple-point triangle T'
      6. Set up new Steiner infrastructure in T'
      7. Record in history
    """
    logger.info(f"Executing Type 2 forward: triangle {trigger.triple_triangle}, "
               f"vertex {trigger.target_vertex} -> cell {trigger.target_cell}")
    
    collapsed_edge = tuple(sorted(trigger.collapsed_vp_edge))
    
    # Step 1: Capture snapshot
    snapshot = capture_snapshot(
        vertex=trigger.target_vertex,
        partition=partition,
        mesh_topology=mesh_topology,
        steiner_handler=steiner_handler,
        source_triangle=trigger.triple_triangle,
        target_triangle=-1
    )
    
    # Step 2: Tear down old Steiner infrastructure
    old_tp = _find_triple_point(steiner_handler, trigger.triple_triangle)
    if old_tp is not None:
        steiner_handler.triple_points.remove(old_tp)
        old_steiner_lambda = None
        collapsed_vp = partition.variable_points[trigger.collapsed_vp_idx]
        old_steiner_lambda = collapsed_vp.lambda_param
    else:
        logger.warning(f"No triple point found at triangle {trigger.triple_triangle}")
        old_steiner_lambda = partition.variable_points[trigger.collapsed_vp_idx].lambda_param
    
    # Step 3: Execute Type 1 rebuild with is_type2=True
    result = rebuild_one_ring(
        vertex=trigger.target_vertex,
        new_cell=trigger.target_cell,
        partition=partition,
        mesh_topology=mesh_topology,
        is_type2=True,
        collapsed_edge=collapsed_edge
    )
    
    # Step 3b: Convert Steiner point to VP on collapsed edge
    steiner_converted_vp_idx = None
    if result.collapsed_edge_needs_vp:
        v1, v2 = collapsed_edge
        cell_v1 = int(partition.vertex_labels[v1])
        cell_v2 = int(partition.vertex_labels[v2])
        
        new_vp = VariablePoint(
            edge=collapsed_edge,
            lambda_param=old_steiner_lambda if old_steiner_lambda is not None else 0.5,
            global_idx=len(partition.variable_points),
            belongs_to_cells={cell_v1, cell_v2},
            active=True
        )
        partition.variable_points.append(new_vp)
        partition.edge_to_varpoint[collapsed_edge] = new_vp.global_idx
        steiner_converted_vp_idx = new_vp.global_idx
        
        logger.info(f"Steiner-to-VP conversion: created VP {steiner_converted_vp_idx} "
                    f"on edge {collapsed_edge} (net +1 VP)")
    
    # Rebuild triangle segments for the collapsed-edge triangles
    collapsed_tris = mesh_topology.get_triangles_sharing_edge(collapsed_edge)
    for tri_idx in collapsed_tris:
        _rebuild_single_triangle_segment(partition, tri_idx)
    
    # Step 4: Identify new triple-point triangle T'
    new_triple_tri = result.new_triple_triangle
    if new_triple_tri is None:
        vertex_labels = partition.vertex_labels
        for tri_idx in result.affected_triangles:
            face = partition.mesh.faces[tri_idx]
            vs = [int(face[i]) for i in range(3)]
            labels = set(int(vertex_labels[v]) for v in vs)
            if len(labels) == 3:
                new_triple_tri = tri_idx
                break
    
    if new_triple_tri is None:
        for tri_idx in collapsed_tris:
            face = partition.mesh.faces[tri_idx]
            vs = [int(face[i]) for i in range(3)]
            labels = set(int(partition.vertex_labels[v]) for v in vs)
            if len(labels) == 3:
                new_triple_tri = tri_idx
                break
    
    # Step 5: Set up new Steiner infrastructure in T'
    if new_triple_tri is not None:
        _setup_new_steiner(new_triple_tri, partition, steiner_handler)
        snapshot.target_triangle = new_triple_tri
    else:
        logger.warning("No new triple-point triangle found after Type 2 forward migration")
    
    # Step 6: Record snapshot and history
    snapshot.steiner_converted_vp_idx = steiner_converted_vp_idx
    history.snapshots.append(snapshot)
    history.flipped_vertices.append(trigger.target_vertex)
    if new_triple_tri is not None:
        history.visited_triangles.append(new_triple_tri)
    
    partition._build_segment_connectivity(force_rebuild=True)
    partition._build_segment_to_triangle_map()
    partition._rebuild_vp_adjacency()
    
    logger.info(f"Type 2 forward complete: new triple triangle = {new_triple_tri}")
    return True


def execute_type2_rollback(target_triangle: int,
                           history: TriplePointHistory,
                           partition: PartitionContour,
                           mesh_topology: MeshTopology,
                           steiner_handler: SteinerHandler) -> bool:
    """
    Roll back a Type 2 migration to a target triangle using snapshot-based restore.
    
    For each step in reverse: tear down current Steiner, deactivate the
    Steiner-converted VP (-1 VP), restore the snapshot, reconstruct old Steiner.
    """
    if target_triangle not in history.visited_triangles:
        logger.error(f"Triangle {target_triangle} not in history")
        return False
    
    target_pos = history.visited_triangles.index(target_triangle)
    current_pos = len(history.visited_triangles) - 1
    
    if target_pos >= current_pos:
        logger.warning(f"Target position {target_pos} >= current {current_pos}, nothing to roll back")
        return False
    
    logger.info(f"Type 2 rollback: {current_pos - target_pos} steps back "
               f"(from triangle {history.visited_triangles[-1]} to {target_triangle})")
    
    for i in range(current_pos - 1, target_pos - 1, -1):
        snapshot = history.snapshots[i]
        
        current_tri = history.visited_triangles[i + 1] if i + 1 < len(history.visited_triangles) else None
        if current_tri is not None:
            _teardown_steiner(current_tri, steiner_handler)
        
        if snapshot.steiner_converted_vp_idx is not None:
            sc_idx = snapshot.steiner_converted_vp_idx
            if sc_idx < len(partition.variable_points):
                vp = partition.variable_points[sc_idx]
                vp.active = False
                edge = tuple(sorted(vp.edge))
                partition.edge_to_varpoint.pop(edge, None)
                logger.debug(f"Deactivated Steiner-converted VP {sc_idx} (-1 VP)")
        
        restore_snapshot(snapshot, partition, mesh_topology, steiner_handler)
    
    history.visited_triangles = history.visited_triangles[:target_pos + 1]
    history.snapshots = history.snapshots[:target_pos]
    history.flipped_vertices = history.flipped_vertices[:target_pos]
    
    partition.rebuild_active_vp_indices()
    partition._build_segment_connectivity(force_rebuild=True)
    partition._build_segment_to_triangle_map()
    
    logger.info(f"Type 2 rollback complete. Triple junction now at triangle {target_triangle}")
    return True


# ============================================================================
# Helpers
# ============================================================================

def _find_triple_point(steiner_handler: SteinerHandler,
                       triangle_idx: int) -> Optional[TriplePoint]:
    """Find a TriplePoint by triangle index."""
    for tp in steiner_handler.triple_points:
        if tp.triangle_idx == triangle_idx:
            return tp
    return None


def _teardown_steiner(triangle_idx: int, steiner_handler: SteinerHandler):
    """Remove Steiner infrastructure for a triangle."""
    steiner_handler.triple_points = [
        tp for tp in steiner_handler.triple_points
        if tp.triangle_idx != triangle_idx
    ]


def _setup_new_steiner(triangle_idx: int,
                       partition: PartitionContour,
                       steiner_handler: SteinerHandler):
    """Create Steiner infrastructure for a new triple-point triangle."""
    ts = partition.get_triangle_segment(triangle_idx)
    if ts is None or len(ts.var_point_indices) != 3:
        logger.warning(f"Cannot set up Steiner for triangle {triangle_idx}: "
                      f"needs 3 VPs, found {len(ts.var_point_indices) if ts else 0}")
        return
    
    tp = TriplePoint.from_partition(triangle_idx, ts.var_point_indices, partition)
    vp_positions = [partition.evaluate_variable_point(vi) for vi in ts.var_point_indices]
    tp.compute_steiner_point(vp_positions=vp_positions)
    steiner_handler.triple_points.append(tp)
    
    logger.info(f"Set up new Steiner tree in triangle {triangle_idx} "
               f"with VPs {ts.var_point_indices}")


def _rebuild_single_triangle_segment(partition: PartitionContour, tri_idx: int):
    """Rebuild the TriangleSegment for a single triangle."""
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
        vp_idx = partition.edge_to_varpoint.get(edge)
        if vp_idx is not None and partition.variable_points[vp_idx].active:
            boundary_edges.append(edge)
            var_point_indices.append(vp_idx)
    
    if var_point_indices:
        partition._triangle_segments[tri_idx] = TriangleSegment(
            triangle_idx=tri_idx,
            vertex_indices=(v1, v2, v3),
            boundary_edges=boundary_edges,
            var_point_indices=var_point_indices
        )
