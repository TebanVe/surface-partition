"""
Pure data structures for the topology migration system.

Contains all dataclasses used by migration_detector, one_ring_rebuilder,
migration_executor, and migration_orchestrator. No logic, no imports
from other src/core modules (except numpy for arrays).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, FrozenSet


@dataclass
class Type1Trigger:
    """Detection result for a Type 1 migration candidate."""
    vertex: int
    current_cell: int
    target_cell: int
    approaching_vps: List[int]
    min_lambda_distance: float
    n_boundary_vps: int


@dataclass
class Type2Trigger:
    """Detection result for a Type 2 migration candidate."""
    triple_triangle: int
    collapsed_vp_edge: Tuple[int, int]
    collapsed_vp_idx: int
    target_vertex: int
    target_cell: int
    angle_at_collapsed: float
    void_angles: Tuple[float, float, float]


@dataclass
class SteinerSnapshot:
    """Captures the full Steiner infrastructure of a triple-point triangle for rollback."""
    triple_triangle: int
    steiner_position: np.ndarray
    vp_indices: Tuple[int, int, int]
    cell_indices: Tuple[int, int, int]
    void_side_pairs: List[Tuple[int, int]]
    arm_endpoints: List[Tuple[np.ndarray, int]]


@dataclass
class LocalStateSnapshot:
    """Captures the local state of a 1-ring before a migration, enabling rollback."""
    flipped_vertex: int
    source_triangle: int
    target_triangle: int
    vertex_labels: Dict[int, int]
    vp_data: Dict[Tuple[int, int], float]
    vp_existence: Set[Tuple[int, int]]
    vp_active_flags: Dict[int, bool]
    vp_cells: Dict[Tuple[int, int], FrozenSet[int]]
    vp_adjacency_subset: Dict[int, Set[int]]
    triangle_segments_data: Dict[int, dict]
    steiner_data: Optional[SteinerSnapshot] = None
    steiner_converted_vp_idx: Optional[int] = None


@dataclass
class TriplePointHistory:
    """Tracks the migration history of a single triple junction across iterations."""
    triple_id: FrozenSet[int]
    visited_triangles: List[int] = field(default_factory=list)
    snapshots: List[LocalStateSnapshot] = field(default_factory=list)
    flipped_vertices: List[int] = field(default_factory=list)


@dataclass
class RebuildResult:
    """Result of a 1-ring rebuild after a vertex flip."""
    destroyed_vps: List[int] = field(default_factory=list)
    created_vps: List[int] = field(default_factory=list)
    kept_vps: List[int] = field(default_factory=list)
    new_triple_triangle: Optional[int] = None
    affected_triangles: List[int] = field(default_factory=list)
    collapsed_edge_needs_vp: bool = False


@dataclass
class DetectionResult:
    """Result of trigger detection phase."""
    type1_triggers: List[Type1Trigger] = field(default_factory=list)
    type2_triggers: List[Type2Trigger] = field(default_factory=list)
    conflicts_resolved: bool = False


@dataclass
class MigrationResult:
    """Result of executing a batch of migrations."""
    type1_applied: int = 0
    type2_forward_applied: int = 0
    type2_rollbacks_applied: int = 0
    failed: bool = False
    error_message: Optional[str] = None


@dataclass
class CycleResult:
    """Result of one optimize-detect-switch cycle."""
    detection: DetectionResult = field(default_factory=DetectionResult)
    migration: MigrationResult = field(default_factory=MigrationResult)
    converged: bool = False
