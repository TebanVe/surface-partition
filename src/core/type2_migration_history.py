"""
Type 2 Migration History Tracking.

This module provides in-memory data structures for tracking triple point
migration history across iterations. Used for detecting and executing
reverse migrations when triple points oscillate between triangles.

Classes:
    Type2MigrationRecord: History for a single triple point
    Type2MigrationHistory: Manager for all triple points

Author: Type 2 reverse migration system
Date: February 2026
"""

from typing import Dict, List, Optional, Tuple
import logging


class Type2MigrationRecord:
    """
    Track migration history for a single triple point.
    
    Attributes:
        original_triangle: Triangle where triple point originated (persistent ID)
        triangle_sequence: List of triangles visited [origin, tri1, tri2, ...]
        iteration_sequence: List of iterations when migrations occurred [2, 3, ...]
        vp_records: List of VP state data for each migration (for reversal)
    """
    
    def __init__(self, original_triangle: int):
        """
        Initialize migration record.
        
        Args:
            original_triangle: Triangle index where triple point started
        """
        self.original_triangle = original_triangle
        self.triangle_sequence = [original_triangle]  # Start with original
        self.iteration_sequence = []  # Empty until first migration
        self.vp_records = []  # One entry per migration
    
    def add_forward_migration(
        self, 
        target_triangle: int,
        iteration: int,
        vp_record: Dict
    ):
        """
        Record a forward migration.
        
        Args:
            target_triangle: Triangle being migrated to
            iteration: Iteration number when migration occurred
            vp_record: Dict containing:
                - created_vp_idx: VP that was created (steiner VP)
                - common_vertex: Center vertex of the fan
                - moved_vps: Dict of {vp_idx: {old_edge, old_lambda, old_distance_to_common}}
        """
        self.triangle_sequence.append(target_triangle)
        self.iteration_sequence.append(iteration)
        self.vp_records.append(vp_record)
    
    def truncate_to_index(self, target_index: int):
        """
        Truncate history after reverse migration.
        
        This removes all migrations from target_index+1 onwards, effectively
        "undoing" those migrations from the recorded history.
        
        Args:
            target_index: Index to truncate to (keep everything up to and including this)
        """
        self.triangle_sequence = self.triangle_sequence[:target_index + 1]
        self.iteration_sequence = self.iteration_sequence[:target_index]
        self.vp_records = self.vp_records[:target_index]
    
    def get_current_triangle(self) -> int:
        """Get the current triangle (last in sequence)."""
        return self.triangle_sequence[-1]
    
    def get_num_migrations(self) -> int:
        """Get number of migrations from original position."""
        return len(self.triangle_sequence) - 1
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"Type2MigrationRecord(original={self.original_triangle}, "
                f"path={self.triangle_sequence}, iters={self.iteration_sequence})")


class Type2MigrationHistory:
    """
    Manager for all triple point migration histories.
    
    Tracks migration history for all triple points that have undergone
    Type 2 migrations. Uses original triangle index as persistent key.
    
    Attributes:
        records: Dict mapping original_triangle -> Type2MigrationRecord
        current_iteration: Current iteration number (for recording)
    """
    
    def __init__(self):
        """Initialize empty migration history."""
        self.records: Dict[int, Type2MigrationRecord] = {}
        self.current_iteration: Optional[int] = None
    
    def find_record_by_current_triangle(self, current_triangle: int) -> Optional[Type2MigrationRecord]:
        """
        Find the record whose current position matches the given triangle.
        
        Args:
            current_triangle: Triangle index to search for
            
        Returns:
            Type2MigrationRecord if found, None otherwise
        """
        for record in self.records.values():
            if record.get_current_triangle() == current_triangle:
                return record
        return None
    
    def check_for_reverse(
        self, 
        current_triangle: int, 
        target_triangle: int
    ) -> Optional[Tuple[int, int]]:
        """
        Check if this is a reverse migration.
        
        A migration is a reverse if:
        1. There's a record for the current triangle
        2. The target triangle appears earlier in that record's sequence
        
        Args:
            current_triangle: Triangle the triple point is currently in
            target_triangle: Triangle the triple point wants to migrate to
            
        Returns:
            None: Not reversible (forward migration)
            (original_triangle, target_index): Reversible, truncate to target_index
        """
        record = self.find_record_by_current_triangle(current_triangle)
        if record is None:
            return None
        
        # Check if target appears earlier in sequence
        # Exclude the last element (current position) from search
        try:
            target_index = record.triangle_sequence[:-1].index(target_triangle)
            return (record.original_triangle, target_index)
        except ValueError:
            return None  # Target not in history
    
    def record_forward_migration(
        self,
        current_triangle: int,
        target_triangle: int,
        iteration: int,
        vp_record: Dict
    ):
        """
        Record a forward migration.
        
        Creates a new record if this is the first migration from this triangle,
        otherwise appends to existing record.
        
        Args:
            current_triangle: Triangle migrating from
            target_triangle: Triangle migrating to
            iteration: Iteration number
            vp_record: VP state data for reversal
        """
        # Find or create record
        record = self.find_record_by_current_triangle(current_triangle)
        
        if record is None:
            # First migration from this triangle - create new record
            record = Type2MigrationRecord(current_triangle)
            self.records[current_triangle] = record
        
        # Add the migration
        record.add_forward_migration(target_triangle, iteration, vp_record)
    
    def get_summary(self) -> Dict:
        """
        Get summary statistics of migration history.
        
        Returns:
            Dict with summary information
        """
        return {
            'num_tracked_triple_points': len(self.records),
            'total_migrations': sum(r.get_num_migrations() for r in self.records.values()),
            'current_iteration': self.current_iteration
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"Type2MigrationHistory("
                f"{len(self.records)} triple points, "
                f"iteration {self.current_iteration})")
