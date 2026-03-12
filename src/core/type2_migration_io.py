"""
Type 2 Migration History HDF5 I/O.

Functions for saving and loading Type2MigrationHistory objects to/from HDF5 files.

Functions:
    save_type2_migration_history: Save history to HDF5 file
    load_type2_migration_history: Load history from HDF5 file

Author: Type 2 reverse migration system
Date: February 2026
"""

import h5py
import numpy as np
from typing import Dict
from .type2_migration_history import Type2MigrationHistory, Type2MigrationRecord


def save_type2_migration_history(h5_file: h5py.File, history: Type2MigrationHistory):
    """
    Save Type2MigrationHistory to HDF5 file.
    
    Creates a 'triple_point_migration_history' group containing one subgroup
    per tracked triple point, keyed by original triangle index.
    
    Structure:
        triple_point_migration_history/
            triangle_<original_tri>/
                triangle_sequence [array]
                iteration_sequence [array]
                vp_records/
                    migration_0/
                        created_vp_idx [int]
                        common_vertex [int]
                        moved_vp_data [compound dataset]
                    migration_1/
                        ...
    
    Args:
        h5_file: Open HDF5 file handle (in write mode)
        history: Type2MigrationHistory object to save
    """
    # Create or clear the migration history group
    if 'triple_point_migration_history' in h5_file:
        del h5_file['triple_point_migration_history']
    
    if len(history.records) == 0:
        # No history to save - create empty group as marker
        h5_file.create_group('triple_point_migration_history')
        return
    
    history_group = h5_file.create_group('triple_point_migration_history')
    
    # Save each triple point's record
    for original_tri, record in history.records.items():
        tp_group = history_group.create_group(f'triangle_{original_tri}')
        
        # Save triangle sequence
        tp_group.create_dataset(
            'triangle_sequence',
            data=np.array(record.triangle_sequence, dtype=np.int32)
        )
        
        # Save iteration sequence
        tp_group.create_dataset(
            'iteration_sequence',
            data=np.array(record.iteration_sequence, dtype=np.int32)
        )
        
        # Save VP records
        vp_records_group = tp_group.create_group('vp_records')
        
        for mig_idx, vp_rec in enumerate(record.vp_records):
            mig_group = vp_records_group.create_group(f'migration_{mig_idx}')
            
            # Save created VP index
            mig_group.create_dataset('created_vp_idx', data=vp_rec['created_vp_idx'])
            
            # Save common vertex
            mig_group.create_dataset('common_vertex', data=vp_rec['common_vertex'])
            
            # Save cell flip info (for reverse migrations)
            if 'old_cell' in vp_rec:
                mig_group.create_dataset('old_cell', data=vp_rec['old_cell'])
            if 'new_cell' in vp_rec:
                mig_group.create_dataset('new_cell', data=vp_rec['new_cell'])
            
            # Save moved VPs data as compound dataset
            moved_vps = vp_rec['moved_vps']
            if len(moved_vps) > 0:
                # Convert dict to arrays
                vp_indices = []
                old_edge_v1 = []
                old_edge_v2 = []
                old_lambdas = []
                old_distances = []
                
                for vp_idx, vp_data in moved_vps.items():
                    vp_indices.append(vp_idx)
                    old_edge_v1.append(vp_data['old_edge'][0])
                    old_edge_v2.append(vp_data['old_edge'][1])
                    old_lambdas.append(vp_data['old_lambda'])
                    old_distances.append(vp_data['old_distance_to_common'])
                
                # Create compound dataset
                moved_vp_group = mig_group.create_group('moved_vp_data')
                moved_vp_group.create_dataset('vp_idx', data=np.array(vp_indices, dtype=np.int32))
                moved_vp_group.create_dataset('old_edge_v1', data=np.array(old_edge_v1, dtype=np.int32))
                moved_vp_group.create_dataset('old_edge_v2', data=np.array(old_edge_v2, dtype=np.int32))
                moved_vp_group.create_dataset('old_lambda', data=np.array(old_lambdas, dtype=np.float64))
                moved_vp_group.create_dataset('old_distance_to_common', data=np.array(old_distances, dtype=np.float64))


def load_type2_migration_history(h5_file: h5py.File) -> Type2MigrationHistory:
    """
    Load Type2MigrationHistory from HDF5 file.
    
    Args:
        h5_file: Open HDF5 file handle (in read mode)
        
    Returns:
        Type2MigrationHistory object (may be empty if no history was saved)
    """
    history = Type2MigrationHistory()
    
    # Check if migration history exists
    if 'triple_point_migration_history' not in h5_file:
        return history  # Return empty history
    
    history_group = h5_file['triple_point_migration_history']
    
    # Load each triple point's record
    for tp_key in history_group.keys():
        if not tp_key.startswith('triangle_'):
            continue
        
        tp_group = history_group[tp_key]
        
        # Extract original triangle index from key
        original_tri = int(tp_key.split('_')[1])
        
        # Load sequences
        triangle_sequence = list(tp_group['triangle_sequence'][:])
        iteration_sequence = list(tp_group['iteration_sequence'][:])
        
        # Create record
        record = Type2MigrationRecord(original_tri)
        record.triangle_sequence = triangle_sequence
        record.iteration_sequence = iteration_sequence
        
        # Load VP records
        vp_records_group = tp_group['vp_records']
        
        for mig_key in sorted(vp_records_group.keys(), key=lambda x: int(x.split('_')[1])):
            mig_group = vp_records_group[mig_key]
            
            # Load basic data
            created_vp_idx = int(mig_group['created_vp_idx'][()])
            common_vertex = int(mig_group['common_vertex'][()])
            
            # Load moved VPs data
            moved_vps = {}
            if 'moved_vp_data' in mig_group:
                moved_vp_group = mig_group['moved_vp_data']
                
                vp_indices = moved_vp_group['vp_idx'][:]
                old_edge_v1 = moved_vp_group['old_edge_v1'][:]
                old_edge_v2 = moved_vp_group['old_edge_v2'][:]
                old_lambdas = moved_vp_group['old_lambda'][:]
                old_distances = moved_vp_group['old_distance_to_common'][:]
                
                for i, vp_idx in enumerate(vp_indices):
                    moved_vps[int(vp_idx)] = {
                        'old_edge': (int(old_edge_v1[i]), int(old_edge_v2[i])),
                        'old_lambda': float(old_lambdas[i]),
                        'old_distance_to_common': float(old_distances[i])
                    }
            
            # Create vp_record
            vp_record = {
                'created_vp_idx': created_vp_idx,
                'common_vertex': common_vertex,
                'moved_vps': moved_vps
            }
            
            # Load cell flip info if present (added 2026-03-12)
            if 'old_cell' in mig_group:
                vp_record['old_cell'] = int(mig_group['old_cell'][()])
            if 'new_cell' in mig_group:
                vp_record['new_cell'] = int(mig_group['new_cell'][()])
            
            record.vp_records.append(vp_record)
        
        # Add record to history
        history.records[original_tri] = record
    
    return history
