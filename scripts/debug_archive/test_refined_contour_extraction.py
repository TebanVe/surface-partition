#!/usr/bin/env python3
"""
Phase 1 Visual Validation: Triangle-Based Contour Extraction

This script validates the NEW Phase 1 triangle-based contour extraction method
by creating interactive PyVista visualizations and automatically saving PNGs.

The script:
1. Loads mesh and creates PartitionContour with Phase 1 structures
2. Loads optimized λ values from refined file
3. Extracts contours using NEW triangle-based method
4. Visualizes with PyVista (interactive 3D)
5. Auto-saves PNGs when windows close

Usage:
    python scripts/test_refined_contour_extraction.py \
        --original results/run_xyz/solution.h5 \
        --refined results/run_xyz/refined_contours.h5 \
        --output-dir diagnostics/phase1_visual \
        [--no-interactive]
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pyvista as pv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.mesh.tri_mesh import TriMesh
from src.partition.contour_partition import PartitionContour
from src.logging_config import get_logger, setup_logging


def load_partition_contour(solution_file):
    """
    Load mesh and create PartitionContour with Phase 1 structures.
    
    Args:
        solution_file: Path to original solution .h5 file
    
    Returns:
        tuple: (TriMesh, PartitionContour)
    """
    logger = get_logger(__name__)
    
    with h5py.File(solution_file, 'r') as f:
        vertices = f['vertices'][:]
        faces = f['faces'][:]
        x_opt = f['x_opt'][:]
    
    # Reshape x_opt and create binary indicators
    n_vertices = vertices.shape[0]
    n_cells = len(x_opt) // n_vertices
    x_opt = x_opt.reshape((n_vertices, n_cells))
    
    indicator_functions = np.zeros_like(x_opt)
    vertex_labels = np.argmax(x_opt, axis=1)
    indicator_functions[np.arange(n_vertices), vertex_labels] = 1.0
    
    # Create mesh and partition (triggers Phase 1 initialization)
    mesh = TriMesh(vertices, faces)
    partition = PartitionContour(mesh, indicator_functions)
    
    logger.info(f"Created PartitionContour: {len(partition.variable_points)} variable points, "
               f"{len(partition.triangle_segments)} triangle segments")
    
    return mesh, partition


def load_optimized_lambda(refined_file, partition):
    """
    Load optimized λ values from refined HDF5 file and apply to partition.
    
    Args:
        refined_file: Path to refined contours .h5 file
        partition: PartitionContour object
    
    Returns:
        np.ndarray: Optimized λ vector
    """
    logger = get_logger(__name__)
    
    with h5py.File(refined_file, 'r') as f:
        if 'lambda_parameters' in f:
            lambda_opt = f['lambda_parameters'][:]
            logger.info(f"Loaded {len(lambda_opt)} optimized λ values from refined file")
        else:
            logger.warning("No lambda_parameters in refined file, using initial λ=0.5")
            lambda_opt = partition.get_variable_vector()
    
    # Apply to partition
    partition.set_variable_vector(lambda_opt)
    
    return lambda_opt


def extract_segments_triangle_based(partition):
    """
    NEW Phase 1: Extract contour segments using triangle-based method.
    
    This is the CORRECT approach that will replace to_visualization_format() in Phase 3.
    It re-extracts segments from triangles using optimized λ values.
    
    Args:
        partition: PartitionContour with triangle_segments and optimized λ values
    
    Returns:
        Dict[cell_idx] -> List[segment arrays (2, D)]
    """
    logger = get_logger(__name__)
    contours_dict = {i: [] for i in range(partition.n_cells)}
    
    for tri_seg in partition.triangle_segments:
        # Get triangle vertices
        v1_idx, v2_idx, v3_idx = tri_seg.vertex_indices
        v1 = partition.mesh.vertices[v1_idx]
        v2 = partition.mesh.vertices[v2_idx]
        v3 = partition.mesh.vertices[v3_idx]
        
        label1, label2, label3 = tri_seg.vertex_labels
        
        if tri_seg.num_cells() == 2:
            # Two-cell triangle: compute level-set segment
            # Find the two variable points and evaluate them
            if len(tri_seg.var_point_indices) == 2:
                vp_idx1, vp_idx2 = tri_seg.var_point_indices
                
                p1 = partition.evaluate_variable_point(vp_idx1)
                p2 = partition.evaluate_variable_point(vp_idx2)
                
                segment = np.vstack([p1, p2])
                
                # Add to both cells that share this boundary
                cells_in_triangle = tri_seg.get_cell_indices()
                for cell_idx in cells_in_triangle:
                    contours_dict[cell_idx].append(segment)
        
        elif tri_seg.is_triple_point():
            # Triple-point triangle: create small triangle connecting 3 variable points
            if len(tri_seg.var_point_indices) == 3:
                vp_idx1, vp_idx2, vp_idx3 = tri_seg.var_point_indices
                
                p1 = partition.evaluate_variable_point(vp_idx1)
                p2 = partition.evaluate_variable_point(vp_idx2)
                p3 = partition.evaluate_variable_point(vp_idx3)
                
                # Create three segments forming a small triangle
                seg12 = np.vstack([p1, p2])
                seg23 = np.vstack([p2, p3])
                seg31 = np.vstack([p3, p1])
                
                # Add all three segments to all three cells
                cells_in_triangle = tri_seg.get_cell_indices()
                for cell_idx in cells_in_triangle:
                    contours_dict[cell_idx].append(seg12)
                    contours_dict[cell_idx].append(seg23)
                    contours_dict[cell_idx].append(seg31)
    
    total_segments = sum(len(segs) for segs in contours_dict.values())
    logger.info(f"Extracted {total_segments} segments using triangle-based method")
    
    return contours_dict


def visualize_contours_pyvista(mesh, partition, contours, cell_idx, output_file, 
                                 interactive=True):
    """
    Visualize contours using PyVista for interactive 3D exploration.
    Auto-saves PNG when window is closed.
    
    Args:
        mesh: TriMesh object
        partition: PartitionContour object
        contours: Dict of cell_idx -> List of segments
        cell_idx: Index of cell to visualize (or None for all cells)
        output_file: Path to save PNG
        interactive: If True, show interactive window
    """
    logger = get_logger(__name__)
    plotter = pv.Plotter(off_screen=not interactive)
    
    # Create mesh surface
    points = mesh.vertices
    faces_pyvista = np.hstack([np.full((mesh.faces.shape[0], 1), 3), mesh.faces])
    mesh_pv = pv.PolyData(points, faces_pyvista)
    
    # Add mesh surface (semi-transparent)
    plotter.add_mesh(mesh_pv, color='lightgray', opacity=0.3, 
                     show_edges=False, label='Mesh Surface')
    
    # Define colors for cells
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'cyan', 'magenta', 'orange']
    
    # Determine which cells to plot
    if cell_idx is not None:
        cells_to_plot = [cell_idx]
        title = f'Phase 1 Contours - Cell {cell_idx}'
    else:
        cells_to_plot = list(contours.keys())
        title = 'Phase 1 Contours - All Cells'
    
    # Plot contours for selected cells
    for idx in cells_to_plot:
        if idx not in contours:
            continue
            
        segments = contours[idx]
        cell_color = colors[idx % len(colors)]
        
        # Create polylines from segments
        for seg in segments:
            if seg.shape[0] < 2:
                continue
            line = pv.Line(seg[0], seg[1])
            plotter.add_mesh(line, color=cell_color, line_width=3, 
                           label=f'Cell {idx}' if seg is segments[0] else '')
    
    # Add variable points colored by λ value
    if partition is not None:
        lambda_values = partition.get_variable_vector()
        vp_positions = np.array([partition.evaluate_variable_point(i) 
                                for i in range(len(partition.variable_points))])
        
        # Create point cloud
        vp_cloud = pv.PolyData(vp_positions)
        vp_cloud['lambda'] = lambda_values
        
        # Add with colormap: blue (λ=0) -> green (λ=0.5) -> red (λ=1)
        plotter.add_mesh(vp_cloud, scalars='lambda', cmap='RdYlBu_r',
                        point_size=8, render_points_as_spheres=True,
                        clim=[0, 1], show_scalar_bar=True,
                        scalar_bar_args={'title': 'Lambda Value'})
    
    plotter.add_title(title, font_size=14)
    plotter.add_axes()
    plotter.show_bounds(grid='front', location='outer', all_edges=True)
    
    # Show or save
    if interactive:
        logger.info(f"Opening interactive window for {title}")
        logger.info("Close window to auto-save PNG and continue...")
        plotter.show(screenshot=output_file)
        logger.info(f"Saved screenshot: {output_file}")
    else:
        plotter.screenshot(output_file)
        logger.info(f"Saved screenshot: {output_file}")
    
    plotter.close()


def analyze_contour_quality(contours):
    """
    Analyze quality of extracted contours.
    
    Args:
        contours: Dict of cell_idx -> List of segments
    
    Returns:
        dict: Quality metrics per cell
    """
    logger = get_logger(__name__)
    
    logger.info("\n" + "="*60)
    logger.info("CONTOUR QUALITY ANALYSIS")
    logger.info("="*60)
    
    quality_metrics = {}
    
    for cell_idx, segments in contours.items():
        if len(segments) == 0:
            logger.warning(f"Cell {cell_idx}: No segments!")
            continue
        
        # Calculate total perimeter
        total_length = sum(np.linalg.norm(seg[1] - seg[0]) for seg in segments)
        
        # Calculate segment lengths
        segment_lengths = [np.linalg.norm(seg[1] - seg[0]) for seg in segments]
        
        metrics = {
            'n_segments': len(segments),
            'total_perimeter': total_length,
            'mean_segment_length': np.mean(segment_lengths),
            'min_segment_length': np.min(segment_lengths),
            'max_segment_length': np.max(segment_lengths),
        }
        
        quality_metrics[cell_idx] = metrics
        
        logger.info(f"\nCell {cell_idx}:")
        logger.info(f"  Segments: {metrics['n_segments']}")
        logger.info(f"  Total Perimeter: {metrics['total_perimeter']:.6f}")
        logger.info(f"  Segment lengths: mean={metrics['mean_segment_length']:.6f}, "
                   f"min={metrics['min_segment_length']:.6f}, max={metrics['max_segment_length']:.6f}")
    
    logger.info("\n" + "="*60)
    
    return quality_metrics


def main():
    parser = argparse.ArgumentParser(
        description='Phase 1 Visual Validation: Triangle-Based Contour Extraction'
    )
    parser.add_argument('--original', type=str, required=True,
                       help='Path to original solution .h5 file')
    parser.add_argument('--refined', type=str, required=True,
                       help='Path to refined contours .h5 file (for optimized λ values)')
    parser.add_argument('--output-dir', type=str, default='diagnostics/phase1_visual',
                       help='Directory to save visualizations (default: diagnostics/phase1_visual)')
    parser.add_argument('--no-interactive', action='store_true',
                       help='Skip interactive PyVista windows, just save PNGs')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    logger = get_logger(__name__)
    
    logger.info("="*80)
    logger.info("PHASE 1 VISUAL VALIDATION: Triangle-Based Contour Extraction")
    logger.info("="*80)
    logger.info(f"Original solution: {args.original}")
    logger.info(f"Refined file: {args.refined}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Interactive mode: {not args.no_interactive}")
    
    # Check input files exist
    if not os.path.exists(args.original):
        logger.error(f"Original solution file not found: {args.original}")
        return 1
    if not os.path.exists(args.refined):
        logger.error(f"Refined file not found: {args.refined}")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Load mesh and create PartitionContour (Phase 1 initialization)
    logger.info("\n" + "="*60)
    logger.info("Step 1: Loading mesh and creating PartitionContour")
    logger.info("="*60)
    mesh, partition = load_partition_contour(args.original)
    
    # Step 2: Load optimized λ values
    logger.info("\n" + "="*60)
    logger.info("Step 2: Loading optimized λ values")
    logger.info("="*60)
    lambda_opt = load_optimized_lambda(args.refined, partition)
    
    # Step 3: Extract contours using NEW triangle-based method
    logger.info("\n" + "="*60)
    logger.info("Step 3: Extracting contours (Triangle-Based Method)")
    logger.info("="*60)
    contours = extract_segments_triangle_based(partition)
    
    for cell_idx, segments in contours.items():
        logger.info(f"  Cell {cell_idx}: {len(segments)} segments")
    
    # Step 4: Analyze contour quality
    logger.info("\n" + "="*60)
    logger.info("Step 4: Analyzing contour quality")
    logger.info("="*60)
    quality_metrics = analyze_contour_quality(contours)
    
    # Step 5: Create visualizations
    logger.info("\n" + "="*60)
    logger.info("Step 5: Creating visualizations")
    logger.info("="*60)
    
    interactive = not args.no_interactive
    
    # Visualize each cell individually
    for cell_idx in sorted(contours.keys()):
        output_file = os.path.join(args.output_dir, f'cell_{cell_idx}_contours.png')
        logger.info(f"\nVisualizing Cell {cell_idx}...")
        visualize_contours_pyvista(mesh, partition, contours, cell_idx, 
                                   output_file, interactive=interactive)
    
    # Create overview with all cells
    logger.info(f"\nCreating overview with all cells...")
    output_file = os.path.join(args.output_dir, 'all_cells_overview.png')
    visualize_contours_pyvista(mesh, partition, contours, None, 
                               output_file, interactive=interactive)
    
    # Save quality report
    report_file = os.path.join(args.output_dir, 'quality_report.txt')
    with open(report_file, 'w') as f:
        f.write("PHASE 1 CONTOUR QUALITY REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Original solution: {args.original}\n")
        f.write(f"Refined file: {args.refined}\n")
        f.write(f"Variable points: {len(partition.variable_points)}\n")
        f.write(f"Triangle segments: {len(partition.triangle_segments)}\n")
        f.write(f"Cells: {partition.n_cells}\n\n")
        
        for cell_idx, metrics in quality_metrics.items():
            f.write(f"\nCell {cell_idx}:\n")
            f.write(f"  Segments: {metrics['n_segments']}\n")
            f.write(f"  Total Perimeter: {metrics['total_perimeter']:.6f}\n")
            f.write(f"  Mean segment length: {metrics['mean_segment_length']:.6f}\n")
            f.write(f"  Min segment length: {metrics['min_segment_length']:.6f}\n")
            f.write(f"  Max segment length: {metrics['max_segment_length']:.6f}\n")
    
    logger.info(f"\n✅ Quality report saved: {report_file}")
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("✅ PHASE 1 VISUAL VALIDATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Visualizations saved in: {args.output_dir}/")
    logger.info(f"Quality report: {report_file}")
    logger.info("\nIf contours appear smooth and continuous, Phase 1 is working correctly!")
    logger.info("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

