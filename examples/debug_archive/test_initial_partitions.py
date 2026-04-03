#!/usr/bin/env python3
"""
Test initial partition boundary connectivity from relaxation results.

This script tests whether the initial PGD/SLSQP optimization produces
proper closed boundary loops for each partition cell, before any
perimeter refinement is applied.

Usage:
    python examples/test_initial_partitions.py --solution results/run_xyz/solution.h5
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.partition.find_contours import ContourAnalyzer
from src.logging_config import get_logger, setup_logging


def test_boundary_connectivity(polylines, region_idx, tolerance=1e-6):
    """
    Test if polylines form closed loops for a given region.
    
    For torus topology, regions can have multiple boundary components:
    - Simple regions: 1 boundary loop
    - Complex regions: Multiple boundary loops (e.g., inner torus region)
    
    Args:
        polylines: List of polylines from stitch_segments_to_polylines
        region_idx: Index of the region being tested
        tolerance: Distance tolerance for considering endpoints as connected
    
    Returns:
        dict: Connectivity statistics
    """
    stats = {
        'region': region_idx,
        'num_polylines': len(polylines),
        'total_segments': sum(len(poly) - 1 for poly in polylines),
        'closed_loops': 0,
        'open_segments': 0,
        'total_length': 0.0,
        'is_properly_closed': False,
        'topology_type': 'unknown'
    }
    
    for poly in polylines:
        if len(poly) < 2:
            continue
            
        # Calculate polyline length
        length = 0.0
        for i in range(len(poly) - 1):
            length += np.linalg.norm(poly[i+1] - poly[i])
        stats['total_length'] += length
        
        # Check if polyline is closed (start and end points are close)
        start_point = poly[0]
        end_point = poly[-1]
        distance = np.linalg.norm(end_point - start_point)
        
        if distance < tolerance:
            stats['closed_loops'] += 1
        else:
            stats['open_segments'] += 1
    
    # Determine topology type and proper closure
    if stats['open_segments'] > 0:
        # Has open segments - definitely not properly closed
        stats['is_properly_closed'] = False
        stats['topology_type'] = 'fragmented'
    elif stats['closed_loops'] == 1:
        # Single closed loop - simple region
        stats['is_properly_closed'] = True
        stats['topology_type'] = 'simple'
    elif stats['closed_loops'] > 1:
        # Multiple closed loops - complex region (e.g., torus inner region)
        stats['is_properly_closed'] = True
        stats['topology_type'] = 'complex'
    else:
        # No loops at all - empty region
        stats['is_properly_closed'] = False
        stats['topology_type'] = 'empty'
    
    return stats


def visualize_cell_boundary(analyzer, polylines, region_idx, output_dir="diagnostics"):
    """
    Create visualization for a single partition cell's boundary.
    
    Args:
        analyzer: ContourAnalyzer instance
        polylines: List of polylines for this region
        region_idx: Index of the region
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine if 2D or 3D
    is_3d = analyzer.vertices.shape[1] == 3
    
    fig = plt.figure(figsize=(12, 10))
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
    region_color = colors[region_idx % len(colors)]
    
    # Plot vertices belonging to this region
    region_vertices = analyzer.vertices[vertex_labels == region_idx]
    if len(region_vertices) > 0:
        if is_3d:
            ax.scatter(region_vertices[:, 0], region_vertices[:, 1], region_vertices[:, 2],
                      c=region_color, alpha=0.3, s=1)
        else:
            ax.scatter(region_vertices[:, 0], region_vertices[:, 1],
                      c=region_color, alpha=0.3, s=1)
    
    # Plot boundary polylines
    for i, poly in enumerate(polylines):
        if len(poly) < 2:
            continue
            
        if is_3d:
            ax.plot(poly[:, 0], poly[:, 1], poly[:, 2], 
                   color=region_color, linewidth=3, alpha=0.8,
                   label=f'Boundary {i+1}' if i == 0 else "")
        else:
            ax.plot(poly[:, 0], poly[:, 1], 
                   color=region_color, linewidth=3, alpha=0.8,
                   label=f'Boundary {i+1}' if i == 0 else "")
    
    # Add title and labels
    ax.set_title(f'Partition Cell {region_idx} - Boundary Analysis')
    if is_3d:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    else:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save plot
    output_file = os.path.join(output_dir, f'cell_{region_idx}_boundary.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Test initial partition boundary connectivity'
    )
    parser.add_argument('--solution', type=str, required=True,
                       help='Path to input solution .h5 file from PGD/SLSQP optimization')
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
    logger.info("Testing Initial Partition Boundary Connectivity")
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
    # Step 1: Load solution and extract contours
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Step 1: Loading solution and extracting contours...")
    
    analyzer = ContourAnalyzer(args.solution)
    analyzer.load_results(use_initial_condition=False)
    
    n_vertices = analyzer.vertices.shape[0]
    n_partitions = analyzer.densities.shape[1]
    mesh_dim = analyzer.vertices.shape[1]
    
    logger.info(f"Mesh: {n_vertices} vertices, {n_partitions} partitions, dimension={mesh_dim}")
    
    # Extract contours using existing functionality
    contours = analyzer.extract_contours()
    
    # -------------------------------------------------------------------------
    # Step 2: Test connectivity for each partition cell
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Step 2: Testing boundary connectivity...")
    
    connectivity_results = []
    all_properly_closed = True
    
    for region_idx, segments in contours.items():
        logger.info(f"")
        logger.info(f"--- Testing Region {region_idx} ---")
        logger.info(f"Raw segments: {len(segments)}")
        
        # Use existing stitching functionality
        polylines = analyzer.stitch_segments_to_polylines(segments)
        logger.info(f"Stitched polylines: {len(polylines)}")
        
        # Test connectivity
        stats = test_boundary_connectivity(polylines, region_idx)
        connectivity_results.append(stats)
        
        # Log results
        logger.info(f"Closed loops: {stats['closed_loops']}")
        logger.info(f"Open segments: {stats['open_segments']}")
        logger.info(f"Total boundary length: {stats['total_length']:.6f}")
        logger.info(f"Topology type: {stats['topology_type']}")
        logger.info(f"Properly closed: {stats['is_properly_closed']}")
        
        if not stats['is_properly_closed']:
            all_properly_closed = False
            logger.warning(f"Region {region_idx} has connectivity issues!")
        
        # Create visualization
        output_file = visualize_cell_boundary(analyzer, polylines, region_idx, args.output_dir)
        logger.info(f"Visualization saved: {output_file}")
    
    # -------------------------------------------------------------------------
    # Step 3: Summary report
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("="*80)
    logger.info("CONNECTIVITY TEST RESULTS")
    logger.info("="*80)
    
    for stats in connectivity_results:
        status = "✅ PASS" if stats['is_properly_closed'] else "❌ FAIL"
        logger.info(f"Region {stats['region']}: {status}")
        logger.info(f"  - Polylines: {stats['num_polylines']}")
        logger.info(f"  - Closed loops: {stats['closed_loops']}")
        logger.info(f"  - Open segments: {stats['open_segments']}")
        logger.info(f"  - Topology type: {stats['topology_type']}")
        logger.info(f"  - Total length: {stats['total_length']:.6f}")
    
    logger.info("")
    if all_properly_closed:
        logger.info("🎉 ALL REGIONS HAVE PROPER CLOSED BOUNDARIES")
        logger.info("The initial relaxation produces correct partition boundaries.")
        logger.info("The problem likely occurs during perimeter refinement.")
    else:
        logger.info("⚠️  SOME REGIONS HAVE CONNECTIVITY ISSUES")
        logger.info("The initial relaxation has boundary connectivity problems.")
        logger.info("This needs to be fixed before perimeter refinement.")
    
    logger.info("")
    logger.info(f"Diagnostic plots saved in: {args.output_dir}/")
    logger.info("="*80)
    
    return 0 if all_properly_closed else 1


if __name__ == '__main__':
    sys.exit(main())
