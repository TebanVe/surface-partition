"""
Utility functions for loading partition data from solution files.

This module provides loaders for both base solution files and refined contours
files, used by visualization scripts to construct mesh and partition objects.

It also provides layout-detection helpers that support both the structured
run directory layout (solution/, traces/, refinement/{campaign}/, ...) and
the legacy flat layout where all artifacts share a single directory.
"""

import os
import re
import h5py
from ..partition.find_contours import ContourAnalyzer
from ..mesh.tri_mesh import TriMesh
from ..partition.contour_partition import PartitionContour


# ---------------------------------------------------------------------------
# Layout detection
# ---------------------------------------------------------------------------

def detect_run_layout(file_path):
    """Detect the run directory and layout type for a solution or checkpoint path.

    Returns:
        (layout, run_dir) where layout is 'structured' or 'flat'.
        For structured layout, run_dir is the root of the run directory
        (parent of solution/, refinement/, etc.).
        For flat layout, run_dir is the immediate parent directory.
    """
    abs_path = os.path.abspath(file_path)
    parent = os.path.dirname(abs_path)
    parent_name = os.path.basename(parent)

    if parent_name == 'solution':
        return 'structured', os.path.dirname(parent)

    grandparent = os.path.dirname(parent)
    grandparent_name = os.path.basename(grandparent)
    if grandparent_name == 'refinement':
        return 'structured', os.path.dirname(grandparent)

    return 'flat', parent


def find_base_solution_path(refined_path, verbose=False):
    """Find the base solution HDF5 corresponding to a refined checkpoint.

    Search order:
        1. ``base_solution_path`` attribute stored inside the HDF5 file
           (resolved relative to the refined file's directory).
        2. Structured layout: look for a ``.h5`` file in ``{run_dir}/solution/``.
        3. Legacy filename-based derivation (strip suffixes from the refined
           filename).

    Returns:
        Absolute path to the base solution file.

    Raises:
        FileNotFoundError: if no base solution can be located.
    """
    abs_refined = os.path.abspath(refined_path)

    # --- 1. HDF5 attribute ---------------------------------------------------
    try:
        with h5py.File(abs_refined, 'r') as f:
            stored = f.attrs.get('base_solution_path')
            if stored is not None:
                stored = str(stored)
                if not os.path.isabs(stored):
                    stored = os.path.join(os.path.dirname(abs_refined), stored)
                stored = os.path.normpath(stored)
                if os.path.exists(stored):
                    if verbose:
                        print(f"  Base solution (from HDF5 attr): {stored}")
                    return stored
    except Exception:
        pass

    # --- 2. Structured layout -------------------------------------------------
    layout, run_dir = detect_run_layout(abs_refined)
    if layout == 'structured':
        solution_dir = os.path.join(run_dir, 'solution')
        if os.path.isdir(solution_dir):
            candidates = [
                f for f in os.listdir(solution_dir)
                if f.endswith('.h5') and not f.endswith('_refined_contours.h5')
            ]
            if len(candidates) == 1:
                base = os.path.join(solution_dir, candidates[0])
                if verbose:
                    print(f"  Base solution (structured layout): {base}")
                return base
            if len(candidates) > 1:
                candidates.sort()
                base = os.path.join(solution_dir, candidates[-1])
                if verbose:
                    print(f"  Base solution (structured, latest of {len(candidates)}): {base}")
                return base

    # --- 3. Legacy filename derivation ----------------------------------------
    base_solution_path = abs_refined.replace('_refined_contours.h5', '.h5')

    if not os.path.exists(base_solution_path):
        if 'iteration' in base_solution_path:
            original_base = re.sub(r'_iteration\d+\.h5$', '.h5', base_solution_path)
            original_base_no_btol = re.sub(r'_btol[\d.e+-]+(?=\.h5)', '', original_base)
            for candidate in (original_base_no_btol, original_base):
                if os.path.exists(candidate):
                    if verbose:
                        print(f"  Base solution (legacy fallback): {candidate}")
                    return candidate
        else:
            base_no_btol = re.sub(r'_btol[\d.e+-]+(?=\.h5)', '', base_solution_path)
            if os.path.exists(base_no_btol):
                if verbose:
                    print(f"  Base solution (legacy, no btol): {base_no_btol}")
                return base_no_btol

    if os.path.exists(base_solution_path):
        if verbose:
            print(f"  Base solution: {base_solution_path}")
        return base_solution_path

    raise FileNotFoundError(
        f"Could not locate base solution for: {refined_path}\n"
        f"  Tried HDF5 attribute, structured layout ({run_dir}/solution/), "
        f"and legacy filename derivation."
    )


def load_partition_from_base_file(base_path, use_initial=False, verbose=False):
    """
    Load partition from a base solution .h5 file.

    The base file contains mesh geometry (vertices, faces) and optimization
    densities (x_opt or x0).  Variable points are placed at the default
    lambda=0.5 (edge midpoints) since no refined lambda parameters exist.

    Args:
        base_path: Path to the base solution .h5 file
        use_initial: If True, use x0 (initial condition) instead of x_opt
        verbose: Print progress messages

    Returns:
        tuple: (mesh, partition) ready for visualization
    """
    if verbose:
        print(f"Loading from base solution file...")
        print(f"  Base: {base_path}")

    analyzer = ContourAnalyzer(base_path)
    analyzer.load_results(use_initial_condition=use_initial)

    mesh = TriMesh(analyzer.vertices, analyzer.faces)
    if verbose:
        print(f"  ✓ Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    indicator_functions = analyzer.compute_indicator_functions()
    if verbose:
        print(f"  ✓ Computed indicator functions: {indicator_functions.shape[1]} cells")

    _contours, boundary_topology = analyzer.extract_contours_with_topology()

    n_boundary_triangles = sum(len(v) for v in boundary_topology.values())
    if verbose:
        print(f"  ✓ Extracted boundary topology: {n_boundary_triangles} boundary triangles")

    partition = PartitionContour(mesh, indicator_functions, boundary_topology=boundary_topology)
    if verbose:
        print(f"  ✓ Created partition: {len(partition.variable_points)} VPs (default λ=0.5)")

    return mesh, partition


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
    if verbose:
        print(f"Loading from refined contours file...")
        print(f"  Refined: {refined_path}")
    
    base_solution_path = find_base_solution_path(refined_path, verbose=verbose)
    
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
        if 'lambda_parameters' not in f:
            raise ValueError("No lambda_parameters found in refined file")
        
        lambda_opt = f['lambda_parameters'][:]
        has_vp_edges = 'vp_edges' in f
        saved_edges = f['vp_edges'][:] if has_vp_edges else None
        
        # Verify VP count match
        if len(lambda_opt) != len(partition.variable_points):
            is_iteration_file = 'iteration' in os.path.basename(refined_path)
            
            error_msg = (
                f"Mismatch: refined file has {len(lambda_opt)} λ parameters, "
                f"but partition has {len(partition.variable_points)} VPs."
            )
            
            if is_iteration_file and not has_stored_indicators:
                error_msg += (
                    f"\n\nThis iteration file was created before indicator functions were "
                    f"added to the export format.\n"
                    f"Solution: Re-run scripts/refine_perimeter.py to generate a new "
                    f"iteration file with updated indicator functions."
                )
            else:
                error_msg += (
                    f"\n\nThis likely means the partition state changed (migrations) but "
                    f"indicator functions were not saved in the refined file."
                )
            
            raise ValueError(error_msg)
        
        if has_vp_edges:
            # Edge-keyed assignment: match lambdas to VPs by edge, not by index.
            # After migrations, the VP list order in memory diverges from the
            # sorted-edge order used by _initialize_from_boundary_topology.
            edge_to_lambda = {}
            for i in range(len(lambda_opt)):
                edge_key = (int(saved_edges[i, 0]), int(saved_edges[i, 1]))
                edge_to_lambda[edge_key] = float(lambda_opt[i])
            
            matched = 0
            unmatched = 0
            for vp in partition.variable_points:
                if vp.edge in edge_to_lambda:
                    vp.lambda_param = edge_to_lambda[vp.edge]
                    matched += 1
                else:
                    unmatched += 1
            
            if verbose:
                print(f"  ✓ Applied λ values by edge key: {matched} matched, {unmatched} unmatched")
            if unmatched > 0 and verbose:
                print(f"  ⚠ {unmatched} VPs had no matching edge in saved data (kept at λ=0.5)")
        else:
            # Fallback: apply by index (backward compatibility with old files)
            partition.set_variable_vector(lambda_opt)
            if verbose:
                print(f"  ✓ Applied optimized λ values by index: {len(lambda_opt)} parameters")
                print(f"    (No vp_edges in file — using legacy index-based assignment)")
    
    return mesh, partition
