"""
Trigger detection and conflict resolution for topology migrations.

Three sections:
A) Type 1 detection: three-VP validation at candidate vertices
B) Type 2 detection: Steiner collapse / void angle analysis
C) Conflict resolution: conflict graph, priority selection, batch/single modes
"""

from typing import List, Dict, Set, Tuple, Union, Optional
from collections import defaultdict

import numpy as np

from ..logging_config import get_logger
from .contour_partition import PartitionContour
from .mesh_topology import MeshTopology
from .steiner_handler import SteinerHandler, TriplePoint
from .migration_types import Type1Trigger, Type2Trigger
from . import migration_utils

logger = get_logger(__name__)


# ============================================================================
# Section A: Type 1 Detection
# ============================================================================

def detect_type1_triggers(partition: PartitionContour,
                          mesh_topology: MeshTopology,
                          delta: float = 0.05) -> List[Type1Trigger]:
    """
    Detect Type 1 migration triggers using the three-VP validation criterion.
    
    For each VP where min(λ, 1-λ) < delta, identify the target vertex. Then
    validate: the candidate must have >= 3 VPs on boundary edges approaching it.
    """
    vertex_candidates: Dict[int, List[int]] = defaultdict(list)
    
    for vp_idx, vp in enumerate(partition.variable_points):
        if not vp.active:
            continue
        dist = migration_utils.compute_boundary_distance(vp)
        if dist < delta:
            target_v = migration_utils.identify_target_vertex(vp)
            if target_v is not None:
                vertex_candidates[target_v].append(vp_idx)
    
    triggers: List[Type1Trigger] = []
    vertex_labels = partition.vertex_labels
    
    for vertex, approaching_vps in vertex_candidates.items():
        current_cell = int(vertex_labels[vertex])
        
        all_boundary_vps = _count_boundary_vps_at_vertex(
            vertex, partition, mesh_topology, vertex_labels)
        
        n_approaching = len(approaching_vps)
        if n_approaching < 3:
            logger.debug(f"Vertex {vertex}: only {n_approaching} approaching VPs, need >= 3. Skipped.")
            continue
        
        all_approaching = _all_boundary_vps_approaching(
            vertex, approaching_vps, all_boundary_vps, partition)
        if not all_approaching:
            logger.debug(f"Vertex {vertex}: not all boundary VPs approaching. Skipped.")
            continue
        
        target_cell = _determine_target_cell(vertex, approaching_vps, partition, vertex_labels)
        if target_cell is None or target_cell == current_cell:
            continue
        
        min_dist = min(migration_utils.compute_boundary_distance(partition.variable_points[vi])
                       for vi in approaching_vps)
        
        triggers.append(Type1Trigger(
            vertex=vertex,
            current_cell=current_cell,
            target_cell=target_cell,
            approaching_vps=approaching_vps,
            min_lambda_distance=min_dist,
            n_boundary_vps=len(all_boundary_vps)
        ))
        logger.info(f"Type 1 trigger: vertex {vertex} ({current_cell} -> {target_cell}), "
                    f"{n_approaching} approaching VPs, min_dist={min_dist:.4f}")
    
    return triggers


def _count_boundary_vps_at_vertex(vertex: int, partition: PartitionContour,
                                   mesh_topology: MeshTopology,
                                   vertex_labels: np.ndarray) -> List[int]:
    """Return all VP indices on edges from vertex to different-cell neighbors."""
    current_cell = int(vertex_labels[vertex])
    boundary_vps = []
    
    for edge in mesh_topology.get_edges_at_vertex(vertex):
        other = edge[1] if edge[0] == vertex else edge[0]
        if int(vertex_labels[other]) != current_cell:
            normalized = tuple(sorted(edge))
            vp_idx = partition.edge_to_varpoint.get(normalized)
            if vp_idx is not None and partition.variable_points[vp_idx].active:
                boundary_vps.append(vp_idx)
    
    return boundary_vps


def _all_boundary_vps_approaching(vertex: int, approaching: List[int],
                                   all_boundary: List[int],
                                   partition: PartitionContour) -> bool:
    """Check that every boundary VP at this vertex is in the approaching set."""
    approaching_set = set(approaching)
    for vp_idx in all_boundary:
        if vp_idx not in approaching_set:
            vp = partition.variable_points[vp_idx]
            target = migration_utils.identify_target_vertex(vp)
            if target != vertex:
                return False
    return True


def _determine_target_cell(vertex: int, approaching_vps: List[int],
                           partition: PartitionContour,
                           vertex_labels: np.ndarray) -> Optional[int]:
    """Determine the target cell for a Type 1 flip based on approaching VP neighbors."""
    current_cell = int(vertex_labels[vertex])
    cell_votes: Dict[int, int] = defaultdict(int)
    
    for vp_idx in approaching_vps:
        vp = partition.variable_points[vp_idx]
        other_v = vp.edge[1] if vp.edge[0] == vertex else vp.edge[0]
        neighbor_cell = int(vertex_labels[other_v])
        if neighbor_cell != current_cell:
            cell_votes[neighbor_cell] += 1
    
    if not cell_votes:
        return None
    return max(cell_votes, key=cell_votes.get)


# ============================================================================
# Section B: Type 2 Detection
# ============================================================================

def detect_type2_triggers(partition: PartitionContour,
                          steiner_handler: SteinerHandler,
                          mesh_topology: MeshTopology,
                          angle_threshold: float = 120.0) -> List[Type2Trigger]:
    """
    Detect Type 2 migration triggers (Steiner collapse / void angle >= 120°).
    """
    triggers: List[Type2Trigger] = []
    
    for tp in steiner_handler.triple_points:
        vp_positions = [partition.evaluate_variable_point(vi) for vi in tp.var_point_indices]
        angles = tp.compute_void_angles(vp_positions)
        
        max_angle_idx = int(np.argmax(angles))
        max_angle = angles[max_angle_idx]
        
        if max_angle < angle_threshold:
            continue
        
        collapsed_vp_idx = tp.var_point_indices[max_angle_idx]
        collapsed_vp = partition.variable_points[collapsed_vp_idx]
        collapsed_edge = collapsed_vp.edge
        
        target_vertex = migration_utils.identify_target_vertex(collapsed_vp)
        if target_vertex is None:
            continue
        
        target_cell = _determine_type2_target_cell(collapsed_edge, tp.cell_indices,
                                                    partition.vertex_labels)
        if target_cell is None:
            continue
        
        triggers.append(Type2Trigger(
            triple_triangle=tp.triangle_idx,
            collapsed_vp_edge=collapsed_edge,
            collapsed_vp_idx=collapsed_vp_idx,
            target_vertex=target_vertex,
            target_cell=target_cell,
            angle_at_collapsed=max_angle,
            void_angles=tuple(angles)
        ))
        logger.info(f"Type 2 trigger: triangle {tp.triangle_idx}, collapsed VP {collapsed_vp_idx} "
                    f"on edge {collapsed_edge}, target vertex {target_vertex} -> cell {target_cell}, "
                    f"angle={max_angle:.1f}°")
    
    return triggers


def _determine_type2_target_cell(collapsed_edge: Tuple[int, int],
                                  cell_indices: List[int],
                                  vertex_labels: np.ndarray) -> Optional[int]:
    """The third cell: the one NOT on the collapsed boundary."""
    v1, v2 = collapsed_edge
    cell_a = int(vertex_labels[v1])
    cell_b = int(vertex_labels[v2])
    boundary_cells = {cell_a, cell_b}
    
    for cell in cell_indices:
        if cell not in boundary_cells:
            return cell
    return None


# ============================================================================
# Section C: Conflict Resolution
# ============================================================================

def resolve_conflicts(type1_triggers: List[Type1Trigger],
                      type2_triggers: List[Type2Trigger],
                      mesh_topology: MeshTopology,
                      batch_mode: bool = True) -> List[Union[Type1Trigger, Type2Trigger]]:
    """
    Build conflict graph and select non-conflicting highest-priority triggers.
    
    Two triggers conflict if their candidate vertices are at mesh distance <= 1.
    From each connected component, selects the highest-priority trigger:
      1. Type 2 overrides Type 1
      2. Most extreme lambda distance (smallest min_lambda_distance)
      3. Most approaching VPs
    """
    if not batch_mode:
        return [select_highest_priority(type1_triggers, type2_triggers)]
    
    all_triggers: List[Union[Type1Trigger, Type2Trigger]] = []
    all_triggers.extend(type2_triggers)
    all_triggers.extend(type1_triggers)
    
    if len(all_triggers) <= 1:
        return all_triggers
    
    def get_vertex(t):
        if isinstance(t, Type2Trigger):
            return t.target_vertex
        return t.vertex
    
    n = len(all_triggers)
    adj: Dict[int, Set[int]] = defaultdict(set)
    
    for i in range(n):
        for j in range(i + 1, n):
            vi = get_vertex(all_triggers[i])
            vj = get_vertex(all_triggers[j])
            if vi == vj or mesh_topology.are_neighbors(vi, vj):
                adj[i].add(j)
                adj[j].add(i)
    
    visited = [False] * n
    components: List[List[int]] = []
    
    for start in range(n):
        if visited[start]:
            continue
        component = []
        stack = [start]
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            component.append(node)
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    stack.append(neighbor)
        components.append(component)
    
    selected: List[Union[Type1Trigger, Type2Trigger]] = []
    for component in components:
        best_idx = _select_best_from_component(component, all_triggers)
        selected.append(all_triggers[best_idx])
    
    logger.info(f"Conflict resolution: {n} triggers in {len(components)} components -> "
                f"{len(selected)} non-conflicting triggers selected")
    return selected


def _select_best_from_component(component: List[int],
                                 all_triggers: List) -> int:
    """Select the highest-priority trigger from a conflict component."""
    def priority_key(idx):
        t = all_triggers[idx]
        is_type2 = isinstance(t, Type2Trigger)
        if is_type2:
            return (1, -t.angle_at_collapsed, 0)
        else:
            return (0, 0, -t.min_lambda_distance)
    
    return max(component, key=priority_key)


def select_highest_priority(type1_triggers: List[Type1Trigger],
                           type2_triggers: List[Type2Trigger]) -> Optional[Union[Type1Trigger, Type2Trigger]]:
    """Select the single globally highest-priority trigger."""
    if type2_triggers:
        return max(type2_triggers, key=lambda t: t.angle_at_collapsed)
    if type1_triggers:
        return min(type1_triggers, key=lambda t: t.min_lambda_distance)
    return None
