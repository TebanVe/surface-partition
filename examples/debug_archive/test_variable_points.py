#!/usr/bin/env python3
"""
Test variable point extraction and positioning from partition contours.

This script tests whether variable points are correctly positioned on
the initial partition boundaries, before any optimization is applied.

Usage:
    python examples/test_variable_points.py --solution results/run_xyz/solution.h5
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.find_contours import ContourAnalyzer
from src.core.tri_mesh import TriMesh
from src.core.contour_partition import PartitionContour
from src.logging_config import get_logger, setup_logging


def extract_variable_point_data(partition):
    """
    Extract variable point data from PartitionContour object.
    
    Args:
        partition: PartitionContour instance
    
    Returns:
        dict: Variable point data and statistics
    """
    data = {
        'variable_points': partition.variable_points,
        'edge_to_varpoint': partition.edge_to_varpoint,
        'n_variable_points': len(partition.variable_points),
        'n_edges': len(partition.edge_to_varpoint),
        'n_cells': partition.n_cells
    }
    
    # Extract lambda values
    lambda_values = np.array([vp.lambda_param for vp in partition.variable_points])
    data['lambda_values'] = lambda_values
    data['lambda_stats'] = {
        'min': np.min(lambda_values),
        'max': np.max(lambda_values),
        'mean': np.mean(lambda_values),
        'std': np.std(lambda_values),
        'all_0_5': np.allclose(lambda_values, 0.5, atol=1e-10)
    }
    
    # Count variable points per cell using triangle-based method
    cell_counts = {}
    for cell_idx in range(partition.n_cells):
        segments = partition.get_cell_segments_from_triangles(cell_idx)
        # Count unique variable points in this cell
        unique_vps = set()
        for seg in segments:
            unique_vps.add(seg[0])
            unique_vps.add(seg[1])
        cell_counts[cell_idx] = len(unique_vps)
    data['cell_counts'] = cell_counts
    
    # Extract edge information
    edges = []
    for vp in partition.variable_points:
        edges.append(vp.edge)
    data['edges'] = edges
    
    return data


def analyze_variable_point_distribution(partition, analyzer):
    """
    Perform comprehensive analysis of variable point distribution.
    
    Args:
        partition: PartitionContour instance
        analyzer: ContourAnalyzer instance
    
    Returns:
        dict: Distribution analysis results
    """
    analysis = {
        'cell_distribution': {},
        'edge_analysis': {},
        'duplicate_check': {},
        'coverage_analysis': {},
        'statistics': {},
        'issues': []
    }
    
    # 1. Cell distribution analysis
    cell_counts = []
    for cell_idx in range(partition.n_cells):
        segments = partition.get_cell_segments_from_triangles(cell_idx)
        # Count unique variable points in this cell
        unique_vps = set()
        for seg in segments:
            unique_vps.add(seg[0])
            unique_vps.add(seg[1])
        count = len(unique_vps)
        cell_counts.append(count)
        analysis['cell_distribution'][cell_idx] = count
    
    # Statistical analysis of cell distribution
    cell_counts_array = np.array(cell_counts)
    analysis['statistics'] = {
        'total_variable_points': len(partition.variable_points),
        'cells_with_points': len([c for c in cell_counts if c > 0]),
        'mean_per_cell': np.mean(cell_counts_array),
        'std_per_cell': np.std(cell_counts_array),
        'min_per_cell': np.min(cell_counts_array),
        'max_per_cell': np.max(cell_counts_array),
        'cv_per_cell': np.std(cell_counts_array) / np.mean(cell_counts_array) if np.mean(cell_counts_array) > 0 else 0
    }
    
    # 2. Edge analysis
    edges = [vp.edge for vp in partition.variable_points]
    unique_edges = set(edges)
    analysis['edge_analysis'] = {
        'total_edges_with_points': len(unique_edges),
        'total_variable_points': len(edges),
        'edges_with_multiple_points': len(edges) - len(unique_edges)
    }
    
    # 3. Duplicate check
    edge_counts = {}
    for edge in edges:
        edge_counts[edge] = edge_counts.get(edge, 0) + 1
    
    duplicates = {edge: count for edge, count in edge_counts.items() if count > 1}
    analysis['duplicate_check'] = {
        'has_duplicates': len(duplicates) > 0,
        'duplicate_edges': duplicates,
        'num_duplicate_edges': len(duplicates)
    }
    
    # 4. Coverage analysis
    # Get all mesh edges
    mesh_edges = set()
    for face in analyzer.faces:
        v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
        mesh_edges.add(tuple(sorted([v1, v2])))
        mesh_edges.add(tuple(sorted([v2, v3])))
        mesh_edges.add(tuple(sorted([v3, v1])))
    
    # Get boundary edges (where vertex assignments differ)
    vertex_labels = np.argmax(analyzer.densities, axis=1)
    boundary_edges = set()
    
    for face in analyzer.faces:
        v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
        label1, label2, label3 = vertex_labels[v1], vertex_labels[v2], vertex_labels[v3]
        
        # Check each edge of the triangle
        if label1 != label2:
            boundary_edges.add(tuple(sorted([v1, v2])))
        if label2 != label3:
            boundary_edges.add(tuple(sorted([v2, v3])))
        if label3 != label1:
            boundary_edges.add(tuple(sorted([v3, v1])))
    
    # Check coverage
    edges_with_variable_points = set(partition.edge_to_varpoint.keys())
    missing_boundary_edges = boundary_edges - edges_with_variable_points
    extra_edges = edges_with_variable_points - boundary_edges
    
    analysis['coverage_analysis'] = {
        'total_mesh_edges': len(mesh_edges),
        'boundary_edges': len(boundary_edges),
        'edges_with_variable_points': len(edges_with_variable_points),
        'missing_boundary_edges': len(missing_boundary_edges),
        'extra_edges': len(extra_edges),
        'coverage_ratio': len(edges_with_variable_points) / len(boundary_edges) if len(boundary_edges) > 0 else 0,
        'missing_edges_list': list(missing_boundary_edges),
        'extra_edges_list': list(extra_edges)
    }
    
    # 5. Issue detection
    issues = []
    
    # Check for unbalanced distribution
    cv = analysis['statistics']['cv_per_cell']
    if cv > 0.5:  # Coefficient of variation > 50%
        issues.append(f"Unbalanced distribution across cells (CV={cv:.3f})")
    
    # Check for duplicates
    if analysis['duplicate_check']['has_duplicates']:
        issues.append(f"Duplicate variable points on {len(duplicates)} edges")
    
    # Check for missing coverage
    coverage_ratio = analysis['coverage_analysis']['coverage_ratio']
    if coverage_ratio < 0.95:  # Less than 95% coverage
        issues.append(f"Low boundary edge coverage ({coverage_ratio:.1%})")
    
    # Check for reasonable total count
    expected_range = (len(analyzer.vertices) * 0.05, len(analyzer.vertices) * 0.2)
    total_points = analysis['statistics']['total_variable_points']
    if not (expected_range[0] <= total_points <= expected_range[1]):
        issues.append(f"Variable point count ({total_points}) outside expected range ({expected_range[0]:.0f}-{expected_range[1]:.0f})")
    
    analysis['issues'] = issues
    
    return analysis


def visualize_variable_points(analyzer, partition, output_dir="diagnostics"):
    """
    Create visualization showing variable points on mesh surface.
    
    Args:
        analyzer: ContourAnalyzer instance
        partition: PartitionContour instance
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine if 2D or 3D
    is_3d = analyzer.vertices.shape[1] == 3
    
    fig = plt.figure(figsize=(15, 12))
    if is_3d:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
    
    # Plot mesh surface
    if is_3d:
        # For 3D, plot mesh triangles
        for face in analyzer.faces:
            v1, v2, v3 = analyzer.vertices[face.astype(int)]
            ax.plot_trisurf([v1[0], v2[0], v3[0]], 
                           [v1[1], v2[1], v3[1]], 
                           [v1[2], v2[2], v3[2]], 
                           alpha=0.1, color='lightgray')
    else:
        # For 2D, plot mesh edges
        for face in analyzer.faces:
            v1, v2, v3 = analyzer.vertices[face.astype(int)]
            ax.plot([v1[0], v2[0], v3[0], v1[0]], 
                   [v1[1], v2[1], v3[1], v1[1]], 
                   'k-', alpha=0.3, linewidth=0.5)
    
    # Color vertices by cell assignment
    vertex_labels = np.argmax(analyzer.densities, axis=1)
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown']
    
    # Plot vertices belonging to each region (with transparency)
    for region_idx in range(partition.n_cells):
        region_vertices = analyzer.vertices[vertex_labels == region_idx]
        if len(region_vertices) > 0:
            region_color = colors[region_idx % len(colors)]
            if is_3d:
                ax.scatter(region_vertices[:, 0], region_vertices[:, 1], region_vertices[:, 2],
                          c=region_color, alpha=0.2, s=0.5, label=f'Cell {region_idx}')
            else:
                ax.scatter(region_vertices[:, 0], region_vertices[:, 1],
                          c=region_color, alpha=0.2, s=0.5, label=f'Cell {region_idx}')
    
    # Plot variable points
    for vp_idx, vp in enumerate(partition.variable_points):
        # Get edge vertices
        v1_idx, v2_idx = vp.edge
        v1 = analyzer.vertices[v1_idx]
        v2 = analyzer.vertices[v2_idx]
        
        # Calculate variable point position
        lambda_val = vp.lambda_param
        var_point_pos = lambda_val * v1 + (1 - lambda_val) * v2
        
        # Color by cell membership
        if len(vp.belongs_to_cells) > 0:
            cell_idx = min(vp.belongs_to_cells)  # Use first cell for coloring
            point_color = colors[cell_idx % len(colors)]
        else:
            point_color = 'black'
        
        # Plot variable point
        if is_3d:
            ax.scatter(var_point_pos[0], var_point_pos[1], var_point_pos[2],
                      c=point_color, s=20, alpha=0.8, edgecolors='black', linewidth=0.5)
        else:
            ax.scatter(var_point_pos[0], var_point_pos[1],
                      c=point_color, s=20, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # Add title and labels
    ax.set_title('Variable Points on Partition Boundaries')
    if is_3d:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    else:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Save plot
    output_file = os.path.join(output_dir, 'variable_points_overview.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_file


def load_optimized_lambda_values(refined_file):
    """
    Load optimized lambda values from refined contours file.
    
    Args:
        refined_file: Path to refined contours .h5 file
    
    Returns:
        np.ndarray: Optimized lambda values
    """
    import h5py
    
    with h5py.File(refined_file, 'r') as f:
        if 'lambda_parameters' in f:
            return f['lambda_parameters'][:]
        else:
            raise ValueError("No lambda_parameters found in refined file")


def visualize_variable_points_by_cell(analyzer, partition, refined_file=None, output_dir="diagnostics"):
    """
    Create separate visualizations for each cell's variable points overlaid on initial boundaries.
    Optionally include optimized variable points for comparison.
    
    Args:
        analyzer: ContourAnalyzer instance
        partition: PartitionContour instance
        refined_file: Optional path to refined contours .h5 file
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine if 2D or 3D
    is_3d = analyzer.vertices.shape[1] == 3
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown']
    
    # Extract initial contours (same as test_initial_partitions.py)
    contours = analyzer.extract_contours()
    
    # Load optimized lambda values if refined file provided
    optimized_lambda = None
    if refined_file and os.path.exists(refined_file):
        try:
            optimized_lambda = load_optimized_lambda_values(refined_file)
            print(f"Loaded optimized lambda values: {len(optimized_lambda)} points")
        except Exception as e:
            print(f"Warning: Could not load optimized lambda values: {e}")
            optimized_lambda = None
    
    output_files = []
    
    for cell_idx in range(partition.n_cells):
        fig = plt.figure(figsize=(12, 10))
        if is_3d:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)
        
        # Plot mesh surface
        if is_3d:
            for face in analyzer.faces:
                v1, v2, v3 = analyzer.vertices[face.astype(int)]
                ax.plot_trisurf([v1[0], v2[0], v3[0]], 
                               [v1[1], v2[1], v3[1]], 
                               [v1[2], v2[2], v3[2]], 
                               alpha=0.1, color='lightgray')
        else:
            for face in analyzer.faces:
                v1, v2, v3 = analyzer.vertices[face.astype(int)]
                ax.plot([v1[0], v2[0], v3[0], v1[0]], 
                       [v1[1], v2[1], v3[1], v1[1]], 
                       'k-', alpha=0.3, linewidth=0.5)
        
        # Plot vertices belonging to this cell (reduced visibility)
        vertex_labels = np.argmax(analyzer.densities, axis=1)
        region_vertices = analyzer.vertices[vertex_labels == cell_idx]
        if len(region_vertices) > 0:
            cell_color = colors[cell_idx % len(colors)]
            if is_3d:
                ax.scatter(region_vertices[:, 0], region_vertices[:, 1], region_vertices[:, 2],
                          c=cell_color, alpha=0.1, s=0.5)
            else:
                ax.scatter(region_vertices[:, 0], region_vertices[:, 1],
                          c=cell_color, alpha=0.1, s=0.5)
        
        # Plot initial contour polylines (the true boundaries)
        if cell_idx in contours:
            segments = contours[cell_idx]
            polylines = analyzer.stitch_segments_to_polylines(segments)
            
            for i, poly in enumerate(polylines):
                if len(poly) < 2:
                    continue
                    
                if is_3d:
                    ax.plot(poly[:, 0], poly[:, 1], poly[:, 2], 
                           color='black', linewidth=1, alpha=0.6,
                           label=f'Initial Boundary {i+1}' if i == 0 else "")
                else:
                    ax.plot(poly[:, 0], poly[:, 1], 
                           color='black', linewidth=1, alpha=0.6,
                           label=f'Initial Boundary {i+1}' if i == 0 else "")
        
        # Plot variable points belonging to this cell
        # Get variable points from triangle-based segments
        segments = partition.get_cell_segments_from_triangles(cell_idx)
        unique_vps = set()
        for seg in segments:
            unique_vps.add(seg[0])
            unique_vps.add(seg[1])
        cell_vp_indices = sorted(list(unique_vps))
        cell_color = colors[cell_idx % len(colors)]
        
        # Calculate and plot initial variable point positions
        initial_positions = []
        for i, vp_idx in enumerate(cell_vp_indices):
            vp = partition.variable_points[vp_idx]
            
            # Get edge vertices
            v1_idx, v2_idx = vp.edge
            v1 = analyzer.vertices[v1_idx]
            v2 = analyzer.vertices[v2_idx]
            
            # Calculate initial variable point position
            lambda_val = vp.lambda_param
            var_point_pos = lambda_val * v1 + (1 - lambda_val) * v2
            initial_positions.append(var_point_pos)
            
            # Plot initial variable point
            if is_3d:
                ax.scatter(var_point_pos[0], var_point_pos[1], var_point_pos[2],
                          c='white', s=2, alpha=1.0, edgecolors=cell_color, linewidth=0.3,
                          marker='o', label=f'Initial Points' if i == 0 else "")
            else:
                ax.scatter(var_point_pos[0], var_point_pos[1],
                          c='white', s=2, alpha=1.0, edgecolors=cell_color, linewidth=0.3,
                          marker='o', label=f'Initial Points' if i == 0 else "")
        
        # Plot optimized variable points if available
        if optimized_lambda is not None:
            optimized_positions = []
            for i, vp_idx in enumerate(cell_vp_indices):
                vp = partition.variable_points[vp_idx]
                
                # Get edge vertices
                v1_idx, v2_idx = vp.edge
                v1 = analyzer.vertices[v1_idx]
                v2 = analyzer.vertices[v2_idx]
                
                # Calculate optimized variable point position
                optimized_lambda_val = optimized_lambda[vp_idx]
                optimized_pos = optimized_lambda_val * v1 + (1 - optimized_lambda_val) * v2
                optimized_positions.append(optimized_pos)
                
                # Plot optimized variable point
                if is_3d:
                    ax.scatter(optimized_pos[0], optimized_pos[1], optimized_pos[2],
                              c=cell_color, s=2, alpha=0.8, edgecolors='black', linewidth=0.3,
                              marker='s', label=f'Optimized Points' if i == 0 else "")
                else:
                    ax.scatter(optimized_pos[0], optimized_pos[1],
                              c=cell_color, s=2, alpha=0.8, edgecolors='black', linewidth=0.3,
                              marker='s', label=f'Optimized Points' if i == 0 else "")
            
            # Draw arrows showing movement (for significant changes)
            for i, (init_pos, opt_pos) in enumerate(zip(initial_positions, optimized_positions)):
                # Only draw arrows for significant changes
                change_magnitude = np.linalg.norm(opt_pos - init_pos)
                if change_magnitude > 0.01:  # Threshold for significant movement
                    if is_3d:
                        ax.quiver(init_pos[0], init_pos[1], init_pos[2],
                                 opt_pos[0] - init_pos[0], opt_pos[1] - init_pos[1], opt_pos[2] - init_pos[2],
                                 color='gray', alpha=0.6, arrow_length_ratio=0.1, linewidth=0.5)
                    else:
                        ax.annotate('', xy=opt_pos, xytext=init_pos,
                                   arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6, lw=0.5))
        
        # Add title and labels
        title = f'Cell {cell_idx} - Variable Points Comparison ({len(cell_vp_indices)} points)'
        if optimized_lambda is not None:
            title += ' - Initial vs Optimized'
        ax.set_title(title)
        
        if is_3d:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        else:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
        
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Save plot
        output_file = os.path.join(output_dir, f'cell_{cell_idx}_variable_points.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        output_files.append(output_file)
    
    return output_files


def main():
    parser = argparse.ArgumentParser(
        description='Test variable point extraction and positioning'
    )
    parser.add_argument('--solution', type=str, required=True,
                       help='Path to input solution .h5 file from PGD/SLSQP optimization')
    parser.add_argument('--refined', type=str, default=None,
                       help='Optional path to refined contours .h5 file for comparison')
    parser.add_argument('--output-dir', type=str, default='diagnostics',
                       help='Directory to save diagnostic plots (default: diagnostics)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    logger = get_logger(__name__)
    
    logger.info("="*80)
    logger.info("Testing Variable Point Extraction and Positioning")
    logger.info("="*80)
    logger.info(f"Input solution: {args.solution}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Check input file exists
    if not os.path.exists(args.solution):
        logger.error(f"Solution file not found: {args.solution}")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Step 1: Load solution and create partition contour
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Step 1: Loading solution and creating partition contour...")
    
    analyzer = ContourAnalyzer(args.solution)
    analyzer.load_results(use_initial_condition=False)
    
    n_vertices = analyzer.vertices.shape[0]
    n_partitions = analyzer.densities.shape[1]
    mesh_dim = analyzer.vertices.shape[1]
    
    logger.info(f"Mesh: {n_vertices} vertices, {n_partitions} partitions, dimension={mesh_dim}")
    
    # Compute indicator functions
    indicators = analyzer.compute_indicator_functions()
    
    # Create mesh and partition contour
    mesh = TriMesh(analyzer.vertices, analyzer.faces)
    partition = PartitionContour(mesh, indicators)
    
    logger.info(f"Created PartitionContour: {len(partition.variable_points)} variable points")
    
    # -------------------------------------------------------------------------
    # Step 2: Extract and analyze variable point data
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Step 2: Analyzing variable point data...")
    
    vp_data = extract_variable_point_data(partition)
    
    # Log statistics
    logger.info(f"Total variable points: {vp_data['n_variable_points']}")
    logger.info(f"Total edges with variable points: {vp_data['n_edges']}")
    logger.info(f"Number of cells: {vp_data['n_cells']}")
    
    # Lambda statistics
    lambda_stats = vp_data['lambda_stats']
    logger.info(f"Lambda parameter statistics:")
    logger.info(f"  - Min: {lambda_stats['min']:.6f}")
    logger.info(f"  - Max: {lambda_stats['max']:.6f}")
    logger.info(f"  - Mean: {lambda_stats['mean']:.6f}")
    logger.info(f"  - Std: {lambda_stats['std']:.6f}")
    logger.info(f"  - All 0.5: {lambda_stats['all_0_5']}")
    
    # Cell distribution
    logger.info(f"Variable points per cell:")
    for cell_idx, count in vp_data['cell_counts'].items():
        logger.info(f"  - Cell {cell_idx}: {count} points")
    
    # -------------------------------------------------------------------------
    # Step 2.5: Comprehensive distribution analysis
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Step 2.5: Comprehensive distribution analysis...")
    
    # Perform comprehensive analysis
    analysis = analyze_variable_point_distribution(partition, analyzer)
    
    # Report statistics
    stats = analysis['statistics']
    logger.info(f"Distribution Statistics:")
    logger.info(f"  Total variable points: {stats['total_variable_points']}")
    logger.info(f"  Cells with points: {stats['cells_with_points']}")
    logger.info(f"  Mean per cell: {stats['mean_per_cell']:.1f}")
    logger.info(f"  Std per cell: {stats['std_per_cell']:.1f}")
    logger.info(f"  Min per cell: {stats['min_per_cell']}")
    logger.info(f"  Max per cell: {stats['max_per_cell']}")
    logger.info(f"  Coefficient of variation: {stats['cv_per_cell']:.3f}")
    
    # Report edge analysis
    edge_analysis = analysis['edge_analysis']
    logger.info(f"Edge Analysis:")
    logger.info(f"  Total edges with variable points: {edge_analysis['total_edges_with_points']}")
    logger.info(f"  Edges with multiple points: {edge_analysis['edges_with_multiple_points']}")
    
    # Report duplicate check
    duplicate_check = analysis['duplicate_check']
    if duplicate_check['has_duplicates']:
        logger.warning(f"DUPLICATE VARIABLE POINTS DETECTED!")
        logger.warning(f"  Number of duplicate edges: {duplicate_check['num_duplicate_edges']}")
        for edge, count in duplicate_check['duplicate_edges'].items():
            logger.warning(f"    Edge {edge}: {count} variable points")
    else:
        logger.info("No duplicate variable points found")
    
    # Report coverage analysis
    coverage = analysis['coverage_analysis']
    logger.info(f"Coverage Analysis:")
    logger.info(f"  Total mesh edges: {coverage['total_mesh_edges']}")
    logger.info(f"  Boundary edges: {coverage['boundary_edges']}")
    logger.info(f"  Edges with variable points: {coverage['edges_with_variable_points']}")
    logger.info(f"  Missing boundary edges: {coverage['missing_boundary_edges']}")
    logger.info(f"  Extra edges: {coverage['extra_edges']}")
    logger.info(f"  Coverage ratio: {coverage['coverage_ratio']:.1%}")
    
    # Report issues
    if analysis['issues']:
        logger.warning("DISTRIBUTION ISSUES DETECTED:")
        for issue in analysis['issues']:
            logger.warning(f"  - {issue}")
    else:
        logger.info("No distribution issues detected")
    
    # -------------------------------------------------------------------------
    # Step 3: Create visualizations
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Step 3: Creating visualizations...")
    
    # Overview visualization
    overview_file = visualize_variable_points(analyzer, partition, args.output_dir)
    logger.info(f"Overview visualization saved: {overview_file}")
    
    # Per-cell visualizations
    cell_files = visualize_variable_points_by_cell(analyzer, partition, args.refined, args.output_dir)
    for i, cell_file in enumerate(cell_files):
        logger.info(f"Cell {i} visualization saved: {cell_file}")
    
    # -------------------------------------------------------------------------
    # Step 4: Summary and analysis
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("="*80)
    logger.info("VARIABLE POINT ANALYSIS RESULTS")
    logger.info("="*80)
    
    # Check for potential issues
    issues = []
    
    if not lambda_stats['all_0_5']:
        issues.append("Lambda values are not all 0.5 (unexpected for initial state)")
    
    # Check for balanced distribution
    cell_counts = list(vp_data['cell_counts'].values())
    if len(cell_counts) > 1:
        count_std = np.std(cell_counts)
        count_mean = np.mean(cell_counts)
        if count_std / count_mean > 0.5:  # More than 50% variation
            issues.append("Unbalanced variable point distribution across cells")
    
    # Check for reasonable total count
    expected_range = (n_vertices * 0.05, n_vertices * 0.2)  # 5-20% of vertices
    if not (expected_range[0] <= vp_data['n_variable_points'] <= expected_range[1]):
        issues.append(f"Variable point count ({vp_data['n_variable_points']}) outside expected range ({expected_range[0]:.0f}-{expected_range[1]:.0f})")
    
    if issues:
        logger.warning("⚠️  POTENTIAL ISSUES DETECTED:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        logger.warning("")
        logger.warning("Variable point extraction may have problems.")
        logger.warning("Check visualizations for misaligned or missing points.")
    else:
        logger.info("✅ NO OBVIOUS ISSUES DETECTED")
        logger.info("Variable point extraction appears correct.")
        logger.info("Check visualizations to verify alignment with boundaries.")
    
    logger.info("")
    logger.info(f"Diagnostic plots saved in: {args.output_dir}/")
    logger.info("="*80)
    
    return 0 if not issues else 1


if __name__ == '__main__':
    sys.exit(main())
