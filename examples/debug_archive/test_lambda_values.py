#!/usr/bin/env python3
"""
Test lambda value changes during perimeter optimization.

This script compares lambda values before and after perimeter refinement
to verify that the optimization actually modifies the variable point positions.

Usage:
    python examples/test_lambda_values.py --solution results/run_xyz/solution.h5 --refined results/run_xyz/refined_contours.h5
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import h5py

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.partition.find_contours import ContourAnalyzer
from src.mesh.tri_mesh import TriMesh
from src.partition.contour_partition import PartitionContour
from src.logging_config import get_logger, setup_logging


def load_initial_lambda_values(solution_file):
    """
    Load initial lambda values from the original solution.
    
    Args:
        solution_file: Path to original solution .h5 file
    
    Returns:
        dict: Initial lambda values and partition data
    """
    analyzer = ContourAnalyzer(solution_file)
    analyzer.load_results(use_initial_condition=False)
    
    # Compute indicator functions
    indicators = analyzer.compute_indicator_functions()
    
    # Create mesh and partition contour
    mesh = TriMesh(analyzer.vertices, analyzer.faces)
    partition = PartitionContour(mesh, indicators)
    
    # Extract lambda values
    lambda_values = np.array([vp.lambda_param for vp in partition.variable_points])
    
    return {
        'analyzer': analyzer,
        'partition': partition,
        'lambda_values': lambda_values,
        'mesh': mesh,
        'indicators': indicators
    }


def load_refined_lambda_values(refined_file):
    """
    Load refined lambda values from the perimeter refinement output.
    
    Args:
        refined_file: Path to refined contours .h5 file
    
    Returns:
        dict: Refined lambda values and data
    """
    with h5py.File(refined_file, 'r') as f:
        # Debug: Print file structure
        def print_structure(name, obj):
            print(f"  {name}: {type(obj)}")
            if isinstance(obj, h5py.Dataset):
                print(f"    Shape: {obj.shape}, Dtype: {obj.dtype}")
        
        print("HDF5 file structure:")
        f.visititems(print_structure)
        
        # Load variable points data
        lambda_values = None
        
        # Try different possible locations for lambda values
        if 'variable_points' in f:
            var_points_data = f['variable_points']
            if isinstance(var_points_data, h5py.Dataset):
                data_array = var_points_data[:]
                if data_array.shape[1] >= 3:  # At least 3 columns
                    lambda_values = data_array[:, 2]  # Lambda is typically the 3rd column
                else:
                    lambda_values = data_array.flatten()  # If only 1 column
            else:
                # If it's a group, look for lambda data inside
                if 'lambda' in var_points_data:
                    lambda_values = var_points_data['lambda'][:]
                elif 'lambda_values' in var_points_data:
                    lambda_values = var_points_data['lambda_values'][:]
        
        elif 'lambda_values' in f:
            lambda_values = f['lambda_values'][:]
        
        elif 'lambda' in f:
            lambda_values = f['lambda'][:]
        
        # If still no lambda values found, look for any dataset that might contain them
        if lambda_values is None:
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    dataset = f[key]
                    if dataset.shape == (822,) or dataset.shape == (822, 1):
                        print(f"Found potential lambda dataset: {key} with shape {dataset.shape}")
                        lambda_values = dataset[:].flatten()
                        break
        
        if lambda_values is None:
            raise ValueError(f"No lambda values found in refined file. Available keys: {list(f.keys())}")
        
        # Load other data if available
        data = {
            'lambda_values': lambda_values,
            'file_structure': list(f.keys())
        }
        
        # Try to load additional data
        for key in ['edges', 'cells', 'segments']:
            if key in f:
                data[key] = f[key][:]
    
    return data


def analyze_lambda_changes(initial_data, refined_data):
    """
    Analyze changes in lambda values between initial and refined states.
    
    Args:
        initial_data: Initial lambda values and data
        refined_data: Refined lambda values and data
    
    Returns:
        dict: Analysis results
    """
    initial_lambda = initial_data['lambda_values']
    refined_lambda = refined_data['lambda_values']
    
    # Ensure same length
    if len(initial_lambda) != len(refined_lambda):
        raise ValueError(f"Lambda array length mismatch: {len(initial_lambda)} vs {len(refined_lambda)}")
    
    # Calculate changes
    lambda_changes = refined_lambda - initial_lambda
    
    analysis = {
        'initial_stats': {
            'min': np.min(initial_lambda),
            'max': np.max(initial_lambda),
            'mean': np.mean(initial_lambda),
            'std': np.std(initial_lambda),
            'all_0_5': np.allclose(initial_lambda, 0.5, atol=1e-10)
        },
        'refined_stats': {
            'min': np.min(refined_lambda),
            'max': np.max(refined_lambda),
            'mean': np.mean(refined_lambda),
            'std': np.std(refined_lambda),
            'all_0_5': np.allclose(refined_lambda, 0.5, atol=1e-10)
        },
        'change_stats': {
            'min_change': np.min(lambda_changes),
            'max_change': np.max(lambda_changes),
            'mean_change': np.mean(lambda_changes),
            'std_change': np.std(lambda_changes),
            'abs_mean_change': np.mean(np.abs(lambda_changes)),
            'max_abs_change': np.max(np.abs(lambda_changes))
        },
        'optimization_occurred': {
            'values_changed': not np.allclose(initial_lambda, refined_lambda, atol=1e-10),
            'significant_changes': np.max(np.abs(lambda_changes)) > 1e-6,
            'reasonable_range': np.all((refined_lambda >= 0) & (refined_lambda <= 1)),
            'not_all_extreme': not (np.allclose(refined_lambda, 0, atol=1e-10) or 
                                   np.allclose(refined_lambda, 1, atol=1e-10))
        }
    }
    
    return analysis


def visualize_lambda_changes(initial_data, refined_data, analysis, output_dir="diagnostics"):
    """
    Create visualizations showing lambda value changes.
    
    Args:
        initial_data: Initial lambda values and data
        refined_data: Refined lambda values and data
        analysis: Analysis results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    initial_lambda = initial_data['lambda_values']
    refined_lambda = refined_data['lambda_values']
    lambda_changes = refined_lambda - initial_lambda
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Lambda Value Analysis: Initial vs Refined', fontsize=16)
    
    # 1. Histogram comparison
    ax1 = axes[0, 0]
    ax1.hist(initial_lambda, bins=50, alpha=0.7, label='Initial', color='blue', density=True)
    ax1.hist(refined_lambda, bins=50, alpha=0.7, label='Refined', color='red', density=True)
    ax1.set_xlabel('Lambda Value')
    ax1.set_ylabel('Density')
    ax1.set_title('Lambda Value Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Scatter plot: Initial vs Refined
    ax2 = axes[0, 1]
    ax2.scatter(initial_lambda, refined_lambda, alpha=0.6, s=1)
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='No change')
    ax2.set_xlabel('Initial Lambda')
    ax2.set_ylabel('Refined Lambda')
    ax2.set_title('Initial vs Refined Lambda Values')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Change histogram
    ax3 = axes[1, 0]
    ax3.hist(lambda_changes, bins=50, alpha=0.7, color='green')
    ax3.axvline(0, color='black', linestyle='--', alpha=0.5, label='No change')
    ax3.set_xlabel('Lambda Change (Refined - Initial)')
    ax3.set_ylabel('Count')
    ax3.set_title('Lambda Value Changes')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Change magnitude
    ax4 = axes[1, 1]
    abs_changes = np.abs(lambda_changes)
    ax4.hist(abs_changes, bins=50, alpha=0.7, color='orange')
    ax4.set_xlabel('Absolute Lambda Change')
    ax4.set_ylabel('Count')
    ax4.set_title('Magnitude of Lambda Changes')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(output_dir, 'lambda_analysis.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Test lambda value changes during perimeter optimization'
    )
    parser.add_argument('--solution', type=str, required=True,
                       help='Path to input solution .h5 file from PGD/SLSQP optimization')
    parser.add_argument('--refined', type=str, required=True,
                       help='Path to refined contours .h5 file from perimeter optimization')
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
    logger.info("Testing Lambda Value Changes During Perimeter Optimization")
    logger.info("="*80)
    logger.info(f"Initial solution: {args.solution}")
    logger.info(f"Refined solution: {args.refined}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Check input files exist
    if not os.path.exists(args.solution):
        logger.error(f"Solution file not found: {args.solution}")
        return 1
    
    if not os.path.exists(args.refined):
        logger.error(f"Refined file not found: {args.refined}")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Step 1: Load initial lambda values
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Step 1: Loading initial lambda values...")
    
    try:
        initial_data = load_initial_lambda_values(args.solution)
        logger.info(f"Loaded initial lambda values: {len(initial_data['lambda_values'])} points")
        
        # Log initial statistics
        init_stats = {
            'min': np.min(initial_data['lambda_values']),
            'max': np.max(initial_data['lambda_values']),
            'mean': np.mean(initial_data['lambda_values']),
            'std': np.std(initial_data['lambda_values']),
            'all_0_5': np.allclose(initial_data['lambda_values'], 0.5, atol=1e-10)
        }
        
        logger.info(f"Initial lambda statistics:")
        logger.info(f"  Min: {init_stats['min']:.6f}")
        logger.info(f"  Max: {init_stats['max']:.6f}")
        logger.info(f"  Mean: {init_stats['mean']:.6f}")
        logger.info(f"  Std: {init_stats['std']:.6f}")
        logger.info(f"  All 0.5: {init_stats['all_0_5']}")
        
    except Exception as e:
        logger.error(f"Failed to load initial data: {e}")
        return 1
    
    # -------------------------------------------------------------------------
    # Step 2: Load refined lambda values
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Step 2: Loading refined lambda values...")
    
    try:
        refined_data = load_refined_lambda_values(args.refined)
        logger.info(f"Loaded refined lambda values: {len(refined_data['lambda_values'])} points")
        
        # Log refined statistics
        ref_stats = {
            'min': np.min(refined_data['lambda_values']),
            'max': np.max(refined_data['lambda_values']),
            'mean': np.mean(refined_data['lambda_values']),
            'std': np.std(refined_data['lambda_values']),
            'all_0_5': np.allclose(refined_data['lambda_values'], 0.5, atol=1e-10)
        }
        
        logger.info(f"Refined lambda statistics:")
        logger.info(f"  Min: {ref_stats['min']:.6f}")
        logger.info(f"  Max: {ref_stats['max']:.6f}")
        logger.info(f"  Mean: {ref_stats['mean']:.6f}")
        logger.info(f"  Std: {ref_stats['std']:.6f}")
        logger.info(f"  All 0.5: {ref_stats['all_0_5']}")
        
        # Log file structure
        logger.info(f"Refined file structure: {refined_data['file_structure']}")
        
    except Exception as e:
        logger.error(f"Failed to load refined data: {e}")
        return 1
    
    # -------------------------------------------------------------------------
    # Step 3: Analyze lambda changes
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Step 3: Analyzing lambda value changes...")
    
    try:
        analysis = analyze_lambda_changes(initial_data, refined_data)
        
        # Report change statistics
        change_stats = analysis['change_stats']
        logger.info(f"Lambda change statistics:")
        logger.info(f"  Min change: {change_stats['min_change']:.6f}")
        logger.info(f"  Max change: {change_stats['max_change']:.6f}")
        logger.info(f"  Mean change: {change_stats['mean_change']:.6f}")
        logger.info(f"  Std change: {change_stats['std_change']:.6f}")
        logger.info(f"  Mean absolute change: {change_stats['abs_mean_change']:.6f}")
        logger.info(f"  Max absolute change: {change_stats['max_abs_change']:.6f}")
        
        # Report optimization status
        opt_status = analysis['optimization_occurred']
        logger.info(f"Optimization analysis:")
        logger.info(f"  Values changed: {opt_status['values_changed']}")
        logger.info(f"  Significant changes: {opt_status['significant_changes']}")
        logger.info(f"  Reasonable range: {opt_status['reasonable_range']}")
        logger.info(f"  Not all extreme: {opt_status['not_all_extreme']}")
        
    except Exception as e:
        logger.error(f"Failed to analyze lambda changes: {e}")
        return 1
    
    # -------------------------------------------------------------------------
    # Step 4: Create visualizations
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Step 4: Creating visualizations...")
    
    try:
        viz_file = visualize_lambda_changes(initial_data, refined_data, analysis, args.output_dir)
        logger.info(f"Lambda analysis visualization saved: {viz_file}")
        
    except Exception as e:
        logger.error(f"Failed to create visualizations: {e}")
        return 1
    
    # -------------------------------------------------------------------------
    # Step 5: Summary report
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("="*80)
    logger.info("LAMBDA VALUE ANALYSIS RESULTS")
    logger.info("="*80)
    
    # Determine if optimization worked
    opt_status = analysis['optimization_occurred']
    if opt_status['values_changed'] and opt_status['significant_changes']:
        if opt_status['reasonable_range'] and opt_status['not_all_extreme']:
            logger.info("✅ OPTIMIZATION WORKED CORRECTLY")
            logger.info("Lambda values were successfully modified during optimization.")
            logger.info("Values are in reasonable range and show meaningful changes.")
        else:
            logger.warning("⚠️  OPTIMIZATION MODIFIED VALUES BUT WITH ISSUES")
            logger.warning("Lambda values changed but may have unreasonable values.")
    else:
        logger.warning("❌ OPTIMIZATION DID NOT WORK")
        logger.warning("Lambda values were not significantly modified.")
        logger.warning("This suggests the optimization failed or didn't run properly.")
    
    # Report key findings
    change_stats = analysis['change_stats']
    logger.info("")
    logger.info("Key Findings:")
    logger.info(f"  - Maximum lambda change: {change_stats['max_abs_change']:.6f}")
    logger.info(f"  - Average lambda change: {change_stats['abs_mean_change']:.6f}")
    logger.info(f"  - Refined lambda range: [{analysis['refined_stats']['min']:.3f}, {analysis['refined_stats']['max']:.3f}]")
    
    logger.info("")
    logger.info(f"Diagnostic plots saved in: {args.output_dir}/")
    logger.info("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
