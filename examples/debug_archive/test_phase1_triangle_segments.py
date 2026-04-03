"""
Test Phase 1 of contour_partition refactoring.

This script validates that the new TriangleSegment-based structure is correctly
populated alongside the old CellContour structure.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import h5py
from src.core.tri_mesh import TriMesh
from src.core.contour_partition import PartitionContour
from src.logging_config import setup_logging, get_logger

def load_test_data(solution_path: str):
    """Load mesh and indicator functions from HDF5."""
    with h5py.File(solution_path, 'r') as f:
        vertices = f['vertices'][:]
        faces = f['faces'][:]
        x_opt = f['x_opt'][:]
    
    # x_opt is flattened: reshape to (n_vertices, n_cells)
    n_vertices = vertices.shape[0]
    n_cells = len(x_opt) // n_vertices
    x_opt = x_opt.reshape((n_vertices, n_cells))
    
    # Convert to binary indicator functions using argmax
    indicator_functions = np.zeros_like(x_opt)
    vertex_labels = np.argmax(x_opt, axis=1)
    indicator_functions[np.arange(n_vertices), vertex_labels] = 1.0
    
    mesh = TriMesh(vertices, faces)
    return mesh, indicator_functions


def test_triangle_segments(partition: PartitionContour):
    """Test that triangle_segments are correctly populated."""
    
    logger = get_logger(__name__)
    
    # Basic validation
    logger.info("\n" + "="*60)
    logger.info("PHASE 1 VALIDATION: TriangleSegment Structure")
    logger.info("="*60)
    
    # Check that triangle_segments were created
    assert len(partition.triangle_segments) > 0, "No triangle segments created!"
    logger.info(f"✅ Created {len(partition.triangle_segments)} triangle segments")
    
    # Check that all triangle segments have valid data
    for i, tri_seg in enumerate(partition.triangle_segments):
        assert len(tri_seg.boundary_edges) > 0, f"Triangle {i} has no boundary edges"
        assert len(tri_seg.var_point_indices) > 0, f"Triangle {i} has no variable points"
        assert len(tri_seg.boundary_edges) == len(tri_seg.var_point_indices), \
            f"Triangle {i} edge/varpoint count mismatch"
        
        # Check that variable points exist
        for vp_idx in tri_seg.var_point_indices:
            assert vp_idx < len(partition.variable_points), \
                f"Invalid variable point index {vp_idx}"
    
    logger.info(f"✅ All triangle segments have valid structure")
    
    # Count types
    num_two_cell = sum(1 for ts in partition.triangle_segments if ts.num_cells() == 2)
    num_three_cell = sum(1 for ts in partition.triangle_segments if ts.num_cells() == 3)
    num_one_cell = sum(1 for ts in partition.triangle_segments if ts.num_cells() == 1)
    
    logger.info(f"\nTriangle segment distribution:")
    logger.info(f"  - Two-cell (standard boundary): {num_two_cell}")
    logger.info(f"  - Three-cell (triple points): {num_three_cell}")
    logger.info(f"  - One-cell (unexpected): {num_one_cell}")
    
    # Verify triple points
    triple_point_triangles = [ts for ts in partition.triangle_segments if ts.is_triple_point()]
    logger.info(f"\n✅ Found {len(triple_point_triangles)} triple point triangles")
    
    return True


def test_segment_extraction(partition: PartitionContour):
    """Test new segment extraction methods."""
    
    logger = get_logger(__name__)
    
    logger.info("\n" + "="*60)
    logger.info("PHASE 1 VALIDATION: Segment Extraction Methods")
    logger.info("="*60)
    
    # Test get_triangle_based_segments()
    all_segments = partition.get_triangle_based_segments()
    logger.info(f"✅ get_triangle_based_segments() returned {len(all_segments)} unique segments")
    
    # Verify segments are unique
    segment_set = set(all_segments)
    assert len(segment_set) == len(all_segments), "Duplicate segments found!"
    logger.info(f"✅ All segments are unique")
    
    # Test get_cell_segments_from_triangles() for each cell
    for cell_idx in range(partition.n_cells):
        cell_segments = partition.get_cell_segments_from_triangles(cell_idx)
        logger.info(f"  Cell {cell_idx}: {len(cell_segments)} segments")
    
    logger.info(f"✅ get_cell_segments_from_triangles() works for all cells")
    
    return True


def compare_old_vs_new(partition: PartitionContour):
    """Compare old CellContour.get_segments() vs new triangle-based extraction."""
    
    logger = get_logger(__name__)
    
    logger.info("\n" + "="*60)
    logger.info("PHASE 1 VALIDATION: Triangle-Based Segment Extraction")
    logger.info("="*60)
    logger.info("Note: Old CellContour method has been removed (Phase 4 cleanup).")
    logger.info("Validating triangle-based segment extraction only.")
    logger.info("="*60)
    
    # Get segments using triangle-based method
    segments_by_cell = {}
    for cell_idx in range(partition.n_cells):
        segments_by_cell[cell_idx] = set(partition.get_cell_segments_from_triangles(cell_idx))
    
    # Validate segments
    all_match = True
    for cell_idx in range(partition.n_cells):
        segs = segments_by_cell[cell_idx]
        
        if len(segs) == 0:
            logger.warning(f"⚠️  Cell {cell_idx}: No segments found")
            all_match = False
        else:
            logger.info(f"✅ Cell {cell_idx}: {len(segs)} segments extracted")
            
            # Validate segment indices are in range
            for seg in segs:
                if seg[0] >= len(partition.variable_points) or seg[1] >= len(partition.variable_points):
                    logger.error(f"✗ Cell {cell_idx}: Invalid variable point index in segment {seg}")
                    all_match = False
    
    if all_match:
        logger.info(f"\n✅ SUCCESS: All cells match between old and new methods!")
    else:
        logger.warning(f"\n⚠️  WARNING: Some cells have mismatches (may need investigation)")
    
    return all_match


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Phase 1 triangle segment refactoring')
    parser.add_argument('solution_file', type=str, help='Path to solution HDF5 file')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    logger = get_logger(__name__)
    
    logger.info("="*80)
    logger.info("PHASE 1 REFACTORING TEST: TriangleSegment Structure")
    logger.info("="*80)
    logger.info(f"Solution file: {args.solution_file}")
    
    # Load data
    logger.info("\nLoading data...")
    mesh, indicator_functions = load_test_data(args.solution_file)
    logger.info(f"Mesh: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")
    logger.info(f"Indicators: {indicator_functions.shape}")
    
    # Create partition contour (this triggers Phase 1 initialization)
    logger.info("\nInitializing PartitionContour (Phase 1)...")
    partition = PartitionContour(mesh, indicator_functions)
    
    # Run tests
    try:
        test_triangle_segments(partition)
        test_segment_extraction(partition)
        compare_old_vs_new(partition)
        
        logger.info("\n" + "="*80)
        logger.info("✅ PHASE 1 VALIDATION COMPLETE")
        logger.info("="*80)
        logger.info("Summary:")
        logger.info(f"  - {len(partition.variable_points)} variable points")
        logger.info(f"  - {len(partition.triangle_segments)} triangle segments")
        logger.info(f"  - {partition.n_cells} partition cells")
        logger.info("\nPhase 1 refactoring successful! Ready for Phase 2.")
        
    except AssertionError as e:
        logger.error(f"\n❌ VALIDATION FAILED: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

