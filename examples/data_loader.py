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
    Load partition from refined_contours.h5 file and base solution file.
    
    The refined file contains:
    - lambda_parameters: optimized VP positions
    - Metadata about optimization
    
    The base solution file contains:
    - vertices, faces: mesh geometry
    - x_opt (densities): for indicator functions
    
    Returns:
        tuple: (mesh, partition) ready for visualization
    """
    if verbose:
        print(f"Loading from refined contours file...")
        print(f"  Refined: {refined_path}")
    
    # Derive base solution path
    base_solution_path = refined_path.replace('_refined_contours.h5', '.h5')
    if verbose:
        print(f"  Base solution: {base_solution_path}")
    
    if not os.path.exists(base_solution_path):
        raise FileNotFoundError(
            f"Base solution file not found: {base_solution_path}\n"
            f"The refined_contours.h5 file needs the corresponding base solution file "
            f"(without _refined_contours) in the same directory."
        )
    
    # Load base solution (mesh + densities)
    analyzer = ContourAnalyzer(base_solution_path)
    analyzer.load_results(use_initial_condition=False)
    
    mesh = TriMesh(analyzer.vertices, analyzer.faces)
    if verbose:
        print(f"  ✓ Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Compute indicator functions
    indicator_functions = analyzer.compute_indicator_functions()
    n_cells = indicator_functions.shape[1]
    if verbose:
        print(f"  ✓ Computed indicator functions: {n_cells} cells")
    
    # Extract boundary topology (efficient initialization)
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
            partition.set_variable_vector(lambda_opt)
            if verbose:
                print(f"  ✓ Applied optimized λ values: {len(lambda_opt)} parameters")
            
            # Verify match
            if len(lambda_opt) != len(partition.variable_points):
                raise ValueError(
                    f"Mismatch: refined file has {len(lambda_opt)} λ parameters, "
                    f"but partition has {len(partition.variable_points)} VPs"
                )
        else:
            raise ValueError("No lambda_parameters found in refined file")
    
    return mesh, partition
