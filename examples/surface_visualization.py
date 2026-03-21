#!/usr/bin/env python3
import os
import sys
import argparse
import h5py
import yaml

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from find_contours import ContourAnalyzer
from plot_utils import plot_partitions_with_contours, plot_contours_on_ring, build_plot_title
from core.contour_partition import PartitionContour
from core.tri_mesh import TriMesh
import numpy as np


def load_partition_contour_from_analyzer(analyzer):
	"""
	Create PartitionContour with Phase 1 structures from ContourAnalyzer data.
	
	Args:
		analyzer: ContourAnalyzer object with loaded results
		
	Returns:
		tuple: (TriMesh, PartitionContour)
	"""
	# Create mesh from analyzer data
	mesh = TriMesh(analyzer.vertices, analyzer.faces)
	
	# Compute indicator functions from densities (winner-takes-all)
	indicator_functions = analyzer.compute_indicator_functions()
	
	# Create partition (triggers Phase 1 initialization)
	partition = PartitionContour(mesh, indicator_functions)
	
	return mesh, partition


def load_optimized_lambda_from_file(refined_path, partition):
	"""
	Load optimized λ values from refined HDF5 file and apply to partition.
	
	Args:
		refined_path: Path to refined contours .h5 file
		partition: PartitionContour object
		
	Returns:
		np.ndarray: Optimized λ vector (or None if not found)
	"""
	try:
		with h5py.File(refined_path, 'r') as f:
			if 'lambda_parameters' in f:
				lambda_opt = f['lambda_parameters'][:]
				partition.set_variable_vector(lambda_opt)
				return lambda_opt
	except Exception as e:
		print(f"Warning: Could not load lambda parameters: {e}")
	
	return None


def main():
	parser = argparse.ArgumentParser(description='Visualize surface partitions (R2 via Matplotlib, R3 via PyVista) from solution file')
	parser.add_argument('--solution', type=str, required=True, help='Path to solution .h5 file')
	parser.add_argument('--use-initial', action='store_true', help='Use x0 instead of x_opt')
	parser.add_argument('--level', type=float, default=0.5, help='Level-set threshold for contours (default: 0.5)')
	parser.add_argument('--save', type=str, help='Path to save image (e.g., results/partition.png)')
	parser.add_argument('--refined', action='store_true', help='Load and visualize refined contours if available')
	parser.add_argument('--comparison', action='store_true', help='Show side-by-side comparison of raw vs refined contours')
	# 2D options
	parser.add_argument('--no-fill', action='store_true', help='Disable filled partition rendering (2D only)')
	parser.add_argument('--no-mesh', action='store_true', help='Disable mesh overlay when contours only (2D only)')
	# 3D options
	parser.add_argument('--show-normals', action='store_true', help='Show triangle normals (3D only)')
	parser.add_argument('--normal-scale', type=float, default=0.1, help='Scale factor for normal length (3D only)')
	parser.add_argument('--normal-color', type=str, default='yellow', help='Color for normal vectors (3D only)')
	parser.add_argument('--color-partition', action='store_true', help='Color triangles by partition (3D only)')
	parser.add_argument('--show-steiner', action='store_true', help='Show Steiner points (triple points) and void triangles (requires --refined)')
	parser.add_argument('--steiner-size', type=float, default=0.02, help='Size of Steiner point spheres (3D only)')
	args = parser.parse_args()

	analyzer = ContourAnalyzer(args.solution)
	analyzer.load_results(use_initial_condition=args.use_initial)

	# Compose title from H5 attrs or metadata fallback
	surface = None
	label1 = 'v1'
	label2 = 'v2'
	var1_val = None
	var2_val = None
	lam = None
	seed = None
	optimizer = 'PySLSQP'
	try:
		with h5py.File(args.solution, 'r') as f:
			surface = f.attrs.get('surface')
			labels = f.attrs.get('resolution_labels')
			if labels is not None and len(labels) >= 2:
				label1 = labels[0]
				label2 = labels[1]
			var1_val = f.attrs.get('var1')
			var2_val = f.attrs.get('var2')
			lam = f.attrs.get('lambda_penalty')
			seed = f.attrs.get('seed')
			opt_attr = f.attrs.get('optimizer')
			optimizer = opt_attr if opt_attr is not None else optimizer
	except Exception:
		pass
	# Fallback: try metadata.yaml in parent dir
	if var1_val is None or var2_val is None or surface is None:
		run_dir = os.path.dirname(args.solution)
		meta_path = os.path.join(run_dir, 'metadata.yaml')
		if os.path.exists(meta_path):
			with open(meta_path, 'r') as mf:
				meta = yaml.safe_load(mf)
			surface = surface or meta.get('input_parameters', {}).get('surface')
			labels = meta.get('input_parameters', {}).get('resolution_labels')
			if labels and len(labels) >= 2:
				label1, label2 = labels[0], labels[1]
			levels = meta.get('levels') or []
			if levels:
				last = levels[-1]
				var1_val = var1_val or last.get(label1)
				var2_val = var2_val or last.get(label2)
			lam = lam or meta.get('input_parameters', {}).get('lambda_penalty')
			seed = seed or meta.get('input_parameters', {}).get('seed')
			optimizer = meta.get('optimizer') or optimizer
	# Final fallback: parse from filename
	if var1_val is None or var2_val is None:
		name = os.path.basename(args.solution)
		import re
		m1 = re.search(r"_v1([a-zA-Z]+)?(\d+)", name)
		m2 = re.search(r"_v2([a-zA-Z]+)?(\d+)", name)
		if m1:
			if m1.group(1):
				label1 = m1.group(1)
			var1_val = int(m1.group(2))
		if m2:
			if m2.group(1):
				label2 = m2.group(1)
			var2_val = int(m2.group(2))
	var1_val = int(var1_val) if var1_val is not None else 0
	var2_val = int(var2_val) if var2_val is not None else 0
	title_str = build_plot_title(optimizer, surface, label1, var1_val, label2, var2_val, lam, seed, None, prefix='Partition')

	# Extract raw contours from indicator functions
	raw_contours = analyzer.extract_contours(level=args.level)
	
	# Try to load refined contours if requested
	refined_contours = None
	steiner_info = None  # Will store Steiner points and void triangles
	partition = None     # Keep reference for Steiner visualization
	mesh = None
	
	if args.refined or args.comparison or args.show_steiner:
		refined_path = args.solution.replace('.h5', '_refined_contours.h5')
		if os.path.exists(refined_path):
			try:
				# Phase 3: Use built-in triangle-based extraction
				print(f"Loading refined contours using Phase 3 triangle-based extraction...")
				
				# Create PartitionContour with Phase 1 structures
				mesh, partition = load_partition_contour_from_analyzer(analyzer)
				
				# Load optimized λ values from refined file
				lambda_opt = load_optimized_lambda_from_file(refined_path, partition)
				
				if lambda_opt is not None:
					# Use built-in to_visualization_format() (now uses Phase 3 extraction)
					refined_contours = partition.to_visualization_format()
					print(f"✅ Extracted refined contours from: {refined_path}")
					print(f"   Variable points: {len(partition.variable_points)}")
					print(f"   Triangle segments: {len(partition.triangle_segments)}")
					
					# Extract Steiner point information if requested
					if args.show_steiner:
						from core.steiner_handler import SteinerHandler
						steiner_handler = SteinerHandler(mesh, partition)
						steiner_info = {
							'steiner_points': [],
							'void_triangles': [],
							'triple_point_data': []
						}
						for tp in steiner_handler.triple_points:
							# Compute Steiner point
							steiner_pt = tp.compute_steiner_point(partition=partition)
							steiner_info['steiner_points'].append(steiner_pt)
							
							# Get void triangle vertices (the 3 variable points)
							void_vertices = []
							for vp_idx in tp.var_point_indices:
								vp = partition.variable_points[vp_idx]
								pos = vp.evaluate(mesh.vertices)
								void_vertices.append(pos)
							steiner_info['void_triangles'].append(np.array(void_vertices))
							
							# Store additional info for debugging
							steiner_info['triple_point_data'].append({
								'triangle_idx': tp.triangle_idx,
								'cell_indices': tp.cell_indices,
								'var_point_indices': tp.var_point_indices
							})
						
						print(f"   Triple points: {len(steiner_handler.triple_points)}")
				else:
					print(f"Warning: Could not load lambda parameters from refined file")
					print(f"         Falling back to raw contours")
					
			except Exception as e:
				print(f"Warning: Failed to extract refined contours: {e}")
				print(f"         Falling back to raw contours")
		else:
			print(f"Warning: Refined contours not found at {refined_path}")
			print(f"         Run perimeter refinement first: python examples/refine_perimeter.py --solution {args.solution}")
	
	# Choose which contours to visualize
	if refined_contours is not None and not args.comparison:
		contours = refined_contours
		title_str += " (Refined)"
	else:
		contours = raw_contours
		if not args.comparison:
			title_str += " (Raw)"
	
	verts = analyzer.vertices
	faces = analyzer.faces

	# Branch by dimension
	D = int(verts.shape[1])
	if D == 2:
		if args.no_fill:
			fig, ax = plot_contours_on_ring(verts, faces, contours, show_mesh=not args.no_mesh, show_vertices=False, title=title_str)
		else:
			triangle_labels = analyzer.label_triangles_from_indicator()
			fig, ax = plot_partitions_with_contours(verts, faces, triangle_labels, contours, title=title_str)
		if args.save:
			import matplotlib.pyplot as plt
			os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
			fig.savefig(args.save, dpi=300, bbox_inches='tight')
			print(f"Saved visualization to: {args.save}")
		else:
			import matplotlib.pyplot as plt
			plt.show()
	elif D == 3:
		from plot_utils_3d import plot_mesh_with_contours_pyvista
		# Derive default save path if not provided: save into solution's directory with informative name
		save_path = args.save
		if not save_path:
			run_dir = os.path.dirname(args.solution) or '.'
			# Build a concise filename using metadata-like tokens
			lam_str = f"{lam}" if lam is not None else "0.0"
			seed_str = f"{int(seed)}" if seed is not None else ""
			fname = f"partition_{surface or 'surface'}_v1{label1}{var1_val}_v2{label2}{var2_val}_lam{lam_str}"
			if seed_str:
				fname += f"_seed{seed_str}"
			# If visualizing initial condition, add suffix to avoid overwriting final solution image
			if args.use_initial:
				fname += "_initial"
			fname += ".png"
			save_path = os.path.join(run_dir, fname)
		# Compute per-face labels by majority vote (approximate region fill) only when requested
		triangle_labels = analyzer.label_triangles_from_indicator() if args.color_partition else None
		# Light pastel palette for partitions to keep contours visually strong
		light_palette = [
			'#c6dbef', '#c7e9c0', '#fdd0a2', '#e5d8bd', '#d9d9d9', '#f2f0f7',
			'#e7e1ef', '#fee0d2', '#ffffcc', '#d0e1f9', '#fde0ef', '#e0ecf4'
		]
		plot_mesh_with_contours_pyvista(
			verts, faces, contours,
			show_edges=True,
			show_normals=args.show_normals,
			normal_scale=args.normal_scale,
			normal_color=args.normal_color,
			face_labels=triangle_labels,
			palette=light_palette if args.color_partition else None,
			show_scalar_bar=False,
			save_path=save_path,
			steiner_info=steiner_info if args.show_steiner else None,
			steiner_size=args.steiner_size
		)
		print(f"Saved 3D visualization to: {save_path}")
	else:
		raise ValueError(f"Unsupported vertex dimension: {D}")


if __name__ == '__main__':
	main()
