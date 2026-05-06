"""
Top-level migration orchestrator: optimize-detect-switch loop.

Provides the MigrationOrchestrator class that replaces the old TopologySwitcher
with a clean public API for test scripts and workflow scripts.
"""

from typing import Dict, List, Optional, Union, FrozenSet
from dataclasses import dataclass, field

from ..logging_config import get_logger
from ..partition.contour_partition import PartitionContour
from ..mesh.tri_mesh import TriMesh
from ..mesh.mesh_topology import MeshTopology
from ..partition.steiner_handler import SteinerHandler
from ..partition.area_calculator import AreaCalculator
from ..partition.perimeter_calculator import PerimeterCalculator
from .migration_types import (
    Type1Trigger, Type2Trigger, TriplePointHistory,
    DetectionResult, MigrationResult
)
from .migration_detector import (
    detect_type1_triggers, detect_type2_triggers, resolve_conflicts
)
from .migration_executor import (
    execute_type1, execute_type2_forward, execute_type2_rollback
)


@dataclass
class MigrationConfig:
    """Configuration for the migration orchestrator."""
    delta: float = 0.05
    angle_threshold: float = 120.0


class MigrationOrchestrator:
    """
    Top-level orchestrator for topology migrations.
    
    Provides a unified API for detection, conflict resolution, and execution of
    Type 1 (vertex collapse) and Type 2 (Steiner / triple-point) migrations.
    """
    
    def __init__(self, partition: PartitionContour, mesh: TriMesh,
                 mesh_topology: MeshTopology, config: Optional[MigrationConfig] = None):
        self.partition = partition
        self.mesh = mesh
        self.mesh_topology = mesh_topology
        self.config = config or MigrationConfig()
        self.logger = get_logger(__name__)
        
        self._steiner_handler: Optional[SteinerHandler] = None
        self._triple_point_histories: Dict[FrozenSet[int], TriplePointHistory] = {}
        self._type1_history: List[tuple] = []
        
        self._last_detection: Optional[DetectionResult] = None
    
    @property
    def steiner_handler(self) -> SteinerHandler:
        if self._steiner_handler is None:
            self._steiner_handler = SteinerHandler(self.mesh, self.partition)
        return self._steiner_handler
    
    def reinitialize_steiner_handler(self):
        """Force re-detection of triple points (call after migrations)."""
        self._steiner_handler = SteinerHandler(self.mesh, self.partition)
    
    @property
    def triple_point_histories(self) -> Dict[FrozenSet[int], TriplePointHistory]:
        return self._triple_point_histories
    
    def detect_all_triggers(self, delta: Optional[float] = None) -> DetectionResult:
        """Run full trigger detection (Type 1 + Type 2).

        Type 1 detection receives the SteinerHandler so that candidates whose
        1-ring touches an existing triple-point triangle are rejected (the site
        should be handled by Type 2, not Type 1).
        """
        d = delta if delta is not None else self.config.delta

        t1 = detect_type1_triggers(self.partition, self.mesh_topology, d,
                                    steiner_handler=self.steiner_handler)
        t2 = detect_type2_triggers(self.partition, self.steiner_handler,
                                    self.mesh_topology, self.config.angle_threshold)
        
        self._last_detection = DetectionResult(
            type1_triggers=t1,
            type2_triggers=t2,
            conflicts_resolved=False
        )
        
        self.logger.info(f"Detection: {len(t1)} Type 1 triggers, {len(t2)} Type 2 triggers")
        return self._last_detection
    
    def execute_migrations(self, mode: str = 'batch') -> MigrationResult:
        """
        Execute migrations from the last detection result.
        
        Args:
            mode: 'batch' for conflict-resolved batch, 'single' for highest-priority only
        """
        if self._last_detection is None:
            self.logger.warning("No detection results. Call detect_all_triggers() first.")
            return MigrationResult()
        
        det = self._last_detection
        result = MigrationResult()
        
        t2_reversals, t2_forward = self._check_reversals(det.type2_triggers)
        
        for reversal_trigger, history, target_tri in t2_reversals:
            success = execute_type2_rollback(
                target_tri, history, self.partition, self.mesh_topology, self.steiner_handler
            )
            if success:
                result.type2_rollbacks_applied += 1
            else:
                result.failed = True
                result.error_message = f"Type 2 rollback failed for triangle {target_tri}"
                return result
        
        remaining_t2 = [t for t in t2_forward]
        remaining_t1 = list(det.type1_triggers)
        
        batch = mode == 'batch'
        selected = resolve_conflicts(remaining_t1, remaining_t2, self.mesh_topology,
                                     batch_mode=batch)
        
        if selected is None:
            selected = []
        
        for trigger in selected:
            if isinstance(trigger, Type2Trigger):
                history = self._get_or_create_history(trigger)
                success = execute_type2_forward(
                    trigger, self.partition, self.mesh_topology,
                    self.steiner_handler, history
                )
                if success:
                    result.type2_forward_applied += 1
                else:
                    result.failed = True
                    result.error_message = f"Type 2 forward failed at triangle {trigger.triple_triangle}"
                    return result
            
            elif isinstance(trigger, Type1Trigger):
                if self._is_type1_oscillation(trigger):
                    self.logger.info(f"Skipping Type 1 at vertex {trigger.vertex}: oscillation detected")
                    continue
                
                success = execute_type1(trigger, self.partition, self.mesh_topology)
                if success:
                    result.type1_applied += 1
                    self._type1_history.append(
                        (trigger.vertex, trigger.current_cell, trigger.target_cell))
                else:
                    result.failed = True
                    result.error_message = f"Type 1 failed at vertex {trigger.vertex}"
                    return result
        
        self.partition.rebuild_active_vp_indices()
        self.reinitialize_steiner_handler()
        self._last_detection = None
        
        self.logger.info(f"Migration result: {result.type1_applied} Type 1, "
                        f"{result.type2_forward_applied} Type 2 forward, "
                        f"{result.type2_rollbacks_applied} Type 2 rollbacks")
        return result
    
    def execute_single_trigger(self, trigger: Union[Type1Trigger, Type2Trigger]) -> bool:
        """Execute a single trigger directly (for debug scripts)."""
        if isinstance(trigger, Type2Trigger):
            history = self._get_or_create_history(trigger)
            success = execute_type2_forward(
                trigger, self.partition, self.mesh_topology,
                self.steiner_handler, history
            )
        else:
            success = execute_type1(trigger, self.partition, self.mesh_topology)
        
        if success:
            self.partition.rebuild_active_vp_indices()
            self.reinitialize_steiner_handler()
        
        return success
    
    def _check_reversals(self, type2_triggers):
        """Check if any Type 2 triggers would cause a reversal (return to visited triangle)."""
        reversals = []
        forward = []
        
        for trigger in type2_triggers:
            history = self._find_history_for_trigger(trigger)
            if history is not None:
                target_tri = self._check_reversal_target(trigger, history)
                if target_tri is not None:
                    reversals.append((trigger, history, target_tri))
                    continue
            forward.append(trigger)
        
        return reversals, forward
    
    def _check_reversal_target(self, trigger: Type2Trigger,
                                history: TriplePointHistory) -> Optional[int]:
        """If the trigger would move the junction to a previously-visited triangle, return it."""
        collapsed_edge = tuple(sorted(trigger.collapsed_vp_edge))
        adjacent_tris = self.mesh_topology.get_triangles_sharing_edge(collapsed_edge)
        
        for tri in adjacent_tris:
            if tri != trigger.triple_triangle and tri in history.visited_triangles:
                return tri
        return None
    
    def _find_history_for_trigger(self, trigger: Type2Trigger) -> Optional[TriplePointHistory]:
        """Find the history for the triple junction at the trigger's triangle."""
        for tid, history in self._triple_point_histories.items():
            if (history.visited_triangles and 
                history.visited_triangles[-1] == trigger.triple_triangle):
                return history
        return None
    
    def _get_or_create_history(self, trigger: Type2Trigger) -> TriplePointHistory:
        """Get existing or create new TriplePointHistory for a Type 2 trigger."""
        existing = self._find_history_for_trigger(trigger)
        if existing is not None:
            return existing
        
        tp = None
        for t in self.steiner_handler.triple_points:
            if t.triangle_idx == trigger.triple_triangle:
                tp = t
                break
        
        if tp is not None:
            triple_id = frozenset(tp.cell_indices)
        else:
            triple_id = frozenset([trigger.target_cell])
        
        history = TriplePointHistory(
            triple_id=triple_id,
            visited_triangles=[trigger.triple_triangle]
        )
        self._triple_point_histories[triple_id] = history
        return history
    
    def _is_type1_oscillation(self, trigger: Type1Trigger) -> bool:
        """Check if this Type 1 flip would reverse a previous one."""
        reverse = (trigger.vertex, trigger.target_cell, trigger.current_cell)
        return reverse in self._type1_history
