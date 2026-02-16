"""
Utility functions for loading partition data from refined contours files.

This module provides the load_partition_from_refined_file function used by
visualization scripts to load mesh and partition data.
"""

import os
import h5py
from src.find_contours import ContourAnalyzer
from src.core.tri_mesh import TriMesh
from src.core.contour_partition import PartitionContour


def load_partition_from_refined_file(refined_path, verbose=False):
    """
    Load partition from refined_contours.h5 file.
    
    Handles two cases:
    1. Original refined file (iteration 1): Uses matching base solution file
    2. Iteration files (iteration 2+): Falls back to original base solution if iteration-specific base doesn't exist
    
    The refined file contains:
    - lambda_parameters: optimized VP positions
    - indicator_functions: updated after migrations (for iteration files)
    - Metadata about optimization
    
    The base solution file contains:
    - vertices, faces: mesh geometry
    - x_opt (densities): for indicator functions (if not in refined file)
    
    Returns:
        tuple: (mesh, partition) ready for visualization
    """
    import re
    
    if verbose:
        print(f"Loading from refined contours file...")
        print(f"  Refined: {refined_path}")
    
    # Derive base solution path
    base_solution_path = refined_path.replace('_refined_contours.h5', '.h5')
    
    # If iteration file and base doesn't exist, fall back to original
    if not os.path.exists(base_solution_path):
        if 'iteration' in base_solution_path:
            # Try original base file (without _iterationN suffix)
            original_base = re.sub(r'_iteration\d+\.h5$', '.h5', base_solution_path)
            if os.path.exists(original_base):
                if verbose:
                    print(f"  Base solution (iteration-specific): {base_solution_path} (not found)")
                    print(f"  Using original base: {original_base}")
                base_solution_path = original_base
            else:
                raise FileNotFoundError(
                    f"Neither iteration nor original base solution file found:\n"
                    f"  Tried: {base_solution_path}\n"
                    f"  Tried: {original_base}"
                )
        else:
            raise FileNotFoundError(
                f"Base solution file not found: {base_solution_path}\n"
                f"The refined_contours.h5 file needs the corresponding base solution file."
            )
    else:
        if verbose:
            print(f"  Base solution: {base_solution_path}")
    
    # Check if refined file has indicator functions (post-migration iteration files)
    has_stored_indicators = False
    with h5py.File(refined_path, 'r') as f:
        if 'indicator_functions' in f:
            has_stored_indicators = True
            if verbose:
                print(f"  ✓ Refined file contains updated indicator functions (post-migration state)")
    
    if has_stored_indicators:
        # Load from refined file (iteration file with updated indicators)
        with h5py.File(refined_path, 'r') as f:
            indicator_functions = f['indicator_functions'][:]
        
        # Load mesh from base solution
        analyzer = ContourAnalyzer(base_solution_path)
        analyzer.load_results(use_initial_condition=False)
        mesh = TriMesh(analyzer.vertices, analyzer.faces)
        
        if verbose:
            print(f"  ✓ Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            print(f"  ✓ Loaded indicator functions from refined file: {indicator_functions.shape[1]} cells")
    else:
        # Standard loading: compute indicators from base solution
        analyzer = ContourAnalyzer(base_solution_path)
        analyzer.load_results(use_initial_condition=False)
        
        mesh = TriMesh(analyzer.vertices, analyzer.faces)
        if verbose:
            print(f"  ✓ Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # Compute indicator functions from base solution
        indicator_functions = analyzer.compute_indicator_functions()
        n_cells = indicator_functions.shape[1]
        if verbose:
            print(f"  ✓ Computed indicator functions: {n_cells} cells")
    
    # Extract boundary topology (efficient initialization)
    # NOTE: For iteration files with stored indicators, we need to extract topology from those updated indicators
    if has_stored_indicators:
        # Create a proper analyzer with updated indicators
        temp_analyzer = ContourAnalyzer.__new__(ContourAnalyzer)
        temp_analyzer.vertices = mesh.vertices
        temp_analyzer.faces = mesh.faces
        temp_analyzer.densities = indicator_functions
        temp_analyzer.h5_filename = refined_path  # For any error messages
        temp_analyzer.logger = analyzer.logger  # Reuse logger from base analyzer
        
        raw_contours, boundary_topology = temp_analyzer.extract_contours_with_topology()
    else:
        # Standard case: use analyzer that already has correct densities
        raw_contours, boundary_topology = analyzer.extract_contours_with_topology()
    
    n_boundary_triangles = sum(len(v) for v in boundary_topology.values())
    if verbose:
        print(f"  ✓ Extracted boundary topology: {n_boundary_triangles} boundary triangles")
    
    # Create partition (this initializes VPs with default λ=0.5)
    partition = PartitionContour(mesh, indicator_functions, boundary_topology=boundary_topology)
    if verbose:
        print(f"  ✓ Created partition: {len(partition.variable_points)} VPs")
    
    # Load optimized λ parameters from refined file
    with h5py.File(refined_path, 'r') as f:
        if 'lambda_parameters' in f:
            lambda_opt = f['lambda_parameters'][:]
            
            # Verify match
            if len(lambda_opt) != len(partition.variable_points):
                # Check if this is an iteration file without stored indicators
                is_iteration_file = 'iteration' in os.path.basename(refined_path)
                
                error_msg = (
                    f"Mismatch: refined file has {len(lambda_opt)} λ parameters, "
                    f"but partition has {len(partition.variable_points)} VPs."
                )
                
                if is_iteration_file and not has_stored_indicators:
                    error_msg += (
                        f"\n\nThis iteration file was created before indicator functions were "
                        f"added to the export format.\n"
                        f"Solution: Re-run test_migration_and_continue.py to generate a new "
                        f"iteration file with updated indicator functions."
                    )
                else:
                    error_msg += (
                        f"\n\nThis likely means the partition state changed (migrations) but "
                        f"indicator functions were not saved in the refined file."
                    )
                
                raise ValueError(error_msg)
            
            partition.set_variable_vector(lambda_opt)
            if verbose:
                print(f"  ✓ Applied optimized λ values: {len(lambda_opt)} parameters")
        else:
            raise ValueError("No lambda_parameters found in refined file")
    
    return mesh, partition
