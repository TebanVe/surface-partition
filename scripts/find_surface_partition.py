#!/usr/bin/env python3
import os
import sys
import time
import argparse
import yaml
import h5py
import datetime
import logging
import getpass
import platform
import socket
import numpy as np
import gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging_config import setup_logging, get_logger
from src.optimization.pyslsqp_optimizer import PySLSQPOptimizer, RefinementTriggered
from src.optimization.pgd_optimizer import ProjectedGradientOptimizer
from src.optimization.projection import (
	orthogonal_projection_iterative,
	create_initial_condition_with_projection,
	validate_projection_result,
)
from src.mesh.interpolation import nearest_neighbor_interpolate
from src.partition.find_contours import ContourAnalyzer
from src.partition.contour_partition import PartitionContour
from src.optimization.perimeter_optimizer import PerimeterOptimizer


def optimize_surface_partition(provider, config, solution_dir=None):
	logger = get_logger(__name__)
	initial_n_partitions = config.n_partitions
	surface = provider.surface_name()
	label1, label2 = provider.resolution_labels()
	n1_init, n2_init = provider.get_resolution()
	timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
	refinement_levels = getattr(config, 'refinement_levels', 1)
	v1_info, v2_info = provider.resolution_summary(refinement_levels)

	outdir = f"results/run_{timestamp}_surf{surface}_npart{initial_n_partitions}_v1{label1}{v1_info}_v2{label2}{v2_info}_lam{getattr(config, 'lambda_penalty', 0.0)}_seed{config.seed}"
	os.makedirs(outdir, exist_ok=True)
	logfile_path = os.path.join(outdir, 'run.log')

	# root logger to file
	root_logger = logging.getLogger()
	file_handler = logging.FileHandler(logfile_path)
	file_handler.setLevel(logging.DEBUG)
	file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
	root_logger.addHandler(file_handler)

	results = []
	levels_meta = []
	logger.info(f"Starting surface partition optimization with {refinement_levels} refinement levels")

	# Keep lightweight state for interpolation between levels
	prev_vertices = None
	prev_x_opt = None

	for level in range(refinement_levels):
		logger.info("=" * 80)
		logger.info(f"Refinement Level {level+1}/{refinement_levels}")
		logger.info("=" * 80)

		# Set resolution for this level (based on provider's stored increments)
		n1 = getattr(provider, 'init_n_radial', n1_init) + level * getattr(provider, 'incr_n_radial', 0)
		n2 = getattr(provider, 'init_n_angular', n2_init) + level * getattr(provider, 'incr_n_angular', 0)
		provider.set_resolution(n1, n2)

		mesh = provider.build()
		from src.mesh.tri_mesh import TriMesh
		assert isinstance(mesh, TriMesh)
		mesh.compute_matrices()
		stats = mesh.get_mesh_statistics()
		epsilon = np.sqrt(stats['mean_triangle_area']) if stats['mean_triangle_area'] > 0 else 1e-2
		logger.info(f"epsilon set to sqrt(mean_triangle_area) = {epsilon:.3e}")

		# Choose total_area for constraints
		if getattr(config, 'use_discrete_area_for_constraints', True):
			total_area = float(np.sum(mesh.v))
		else:
			theoretical = getattr(provider, 'theoretical_total_area', None)
			total_area = provider.theoretical_total_area() if callable(theoretical) else float(np.sum(mesh.v))

		optimizer_name = 'PySLSQP' if getattr(config, 'optimizer_type', 'pyslsqp') == 'pyslsqp' else 'PGD'
		if optimizer_name == 'PySLSQP':
			optimizer = PySLSQPOptimizer(K=mesh.K, M=mesh.M, v=mesh.v, n_partitions=config.n_partitions,
								epsilon=epsilon, total_area=total_area,
								lambda_penalty=getattr(config, 'lambda_penalty', 0.0),
								refine_patience=int(getattr(config, 'refine_patience', 30)),
								refine_delta_energy=float(getattr(config, 'refine_delta_energy', 1e-4)),
								refine_grad_tol=float(getattr(config, 'refine_grad_tol', 1e-2)),
								refine_constraint_tol=float(getattr(config, 'refine_constraint_tol', 1e-2)),
								logger=logger)
		else:
			optimizer = ProjectedGradientOptimizer(K=mesh.K, M=mesh.M, v=mesh.v, n_partitions=config.n_partitions,
										epsilon=epsilon, total_area=total_area,
										lambda_penalty=getattr(config, 'lambda_penalty', 0.0),
										refine_patience=int(getattr(config, 'refine_patience', 30)),
										refine_delta_energy=float(getattr(config, 'refine_delta_energy', 1e-4)),
										refine_grad_tol=float(getattr(config, 'refine_grad_tol', 1e-2)),
										refine_constraint_tol=float(getattr(config, 'refine_constraint_tol', 1e-2)),
										logger=logger)
			# Configure PGD constant-phase penalty options
			if hasattr(optimizer, 'penalty_target_mode'):
				optimizer.penalty_target_mode = str(getattr(config, 'penalty_target_mode', 'fixed'))
			if hasattr(optimizer, 'penalty_eps'):
				optimizer.penalty_eps = float(getattr(config, 'penalty_eps', 1e-8))

		N = len(mesh.v)
		if level == 0:
			x0 = create_initial_condition_with_projection(N, config.n_partitions, mesh.v, seed=config.seed, method="iterative")
		else:
			# Interpolate from previous level using cached lightweight state
			x0 = nearest_neighbor_interpolate(prev_vertices, mesh.vertices, prev_x_opt, config.n_partitions)
			A = x0.reshape(N, config.n_partitions)
			A = orthogonal_projection_iterative(A, np.ones(config.n_partitions), np.sum(mesh.v) / config.n_partitions * np.ones(config.n_partitions), mesh.v, max_iter=100, tol=1e-8)
			x0 = A.flatten()

		start = time.time()
		try:
			if optimizer_name == 'PySLSQP':
				x_opt, success = optimizer.optimize(
					x0=x0,
					maxiter=getattr(config, 'max_iter', 1000),
					ftol=float(getattr(config, 'tol', 1e-6)),
					use_analytic=getattr(config, 'use_analytic', True),
					results_dir=outdir,
					run_name=f"pyslsqp_part{config.n_partitions}_v1{label1}{provider.get_resolution()[0]}_v2{label2}{provider.get_resolution()[1]}_level{level}",
					is_mesh_refinement=(level > 0),
					save_itr=getattr(config, 'pyslsqp_save_itr', 'major')
				)
			else:
				x_opt, success = optimizer.optimize(
					x0=x0,
					maxiter=getattr(config, 'max_iter', 1000),
					step0=float(getattr(config, 'pgd_step0', 1.0)),
					armijo_c=float(getattr(config, 'pgd_armijo_c', 1e-4)),
					backtrack_rho=float(getattr(config, 'pgd_backtrack_rho', 0.5)),
					projection_max_iter=int(getattr(config, 'pgd_projection_max_iter', 100)),
					projection_tol=float(getattr(config, 'pgd_projection_tol', 1e-8)),
					results_dir=outdir,
					run_name=f"pgd_part{config.n_partitions}_v1{label1}{provider.get_resolution()[0]}_v2{label2}{provider.get_resolution()[1]}_level{level}",
					is_mesh_refinement=(level > 0),
					data_save_stride=int(getattr(config, 'h5_save_stride', 1)),
					data_save_vars=getattr(config, 'h5_save_vars', ['x']),
					save_first_last=bool(getattr(config, 'h5_always_save_first_last', True)),
					log_frequency=int(getattr(config, 'run_log_frequency', getattr(config, 'log_frequency', 50))),
					refine_trigger_mode=str(getattr(config, 'refine_trigger_mode', 'full')),
					refine_gnorm_patience=int(getattr(config, 'refine_gnorm_patience', 30)),
					refine_gnorm_delta=float(getattr(config, 'refine_gnorm_delta', 1e-4)),
					refine_feas_patience=int(getattr(config, 'refine_feas_patience', 30)),
					refine_feas_delta=float(getattr(config, 'refine_feas_delta', 1e-6)),
					enable_refinement_triggers=bool(getattr(config, 'enable_refinement_triggers', True)),
				)
		except RefinementTriggered:
			logger.info(f"Refinement triggered early at level {level+1}")
			x_opt = getattr(optimizer, 'prev_x', x0)
			success = False
		elapsed = time.time() - start

		energy_val = optimizer.compute_energy(x_opt)
		results.append({
			'level': level,
			'epsilon': epsilon,
			'x_opt': x_opt,
			'energy': energy_val,
			'iterations': len(optimizer.log.get('iterations', [])),
			'time': elapsed,
			'success': success,
		})
		levels_meta.append({
			'level': level,
			label1: int(provider.get_resolution()[0]),
			label2: int(provider.get_resolution()[1]),
			'N': int(N),
			'v_sum': float(np.sum(mesh.v)),
			'epsilon': float(epsilon),
			'files': {
				'internal_data': os.path.abspath(getattr(optimizer, 'internal_data_file', '')),
				'summary': os.path.abspath(getattr(optimizer, 'summary_file', '')),
			},
			'iters': {
				'num_iterations': int(results[-1]['iterations'])
			}
		})

		# Per-level result summary
		logger.info(f"Results for level {level+1}:")
		logger.info(f"  Energy: {energy_val:.6e}")
		logger.info(f"  Iterations: {results[-1]['iterations']}")
		logger.info(f"  Time: {elapsed:.2f}s")
		logger.info(f"  Success: {success}")

		# Prepare for next level: cache minimal data for interpolation
		prev_vertices = mesh.vertices.copy()
		prev_x_opt = x_opt.copy()

		# Free heavy objects before moving to next level
		if level < refinement_levels - 1:
			try:
				mesh.K = None
				mesh.M = None
			except Exception:
				pass
			del optimizer
			del mesh
			gc.collect()

	# Save final solution (use last built mesh still in scope)
	final = results[-1]
	x_opt = final['x_opt']
	solution_path = os.path.join(solution_dir or outdir, f"surface_part{config.n_partitions}_surf{surface}_v1{label1}{v1_info}_v2{label2}{v2_info}_lam{getattr(config, 'lambda_penalty', 0.0)}_seed{config.seed}_{timestamp}.h5")
	with h5py.File(solution_path, 'w') as f:
		f.create_dataset('x_opt', data=x_opt)
		f.create_dataset('x0', data=x0)
		f.create_dataset('vertices', data=mesh.vertices)
		f.create_dataset('faces', data=mesh.faces, dtype='i4')
		f.attrs['n_partitions'] = config.n_partitions
		# Title/metadata attributes for surface-agnostic visualization
		f.attrs['surface'] = surface
		f.attrs['resolution_labels'] = [label1, label2]
		# Last-level resolution values
		if levels_meta:
			last_level = levels_meta[-1]
			f.attrs['var1'] = int(last_level.get(label1, provider.get_resolution()[0]))
			f.attrs['var2'] = int(last_level.get(label2, provider.get_resolution()[1]))
		else:
			f.attrs['var1'] = int(provider.get_resolution()[0])
			f.attrs['var2'] = int(provider.get_resolution()[1])
		f.attrs['lambda_penalty'] = float(getattr(config, 'lambda_penalty', 0.0))
		f.attrs['seed'] = int(config.seed)
		f.attrs['optimizer'] = 'PySLSQP' if optimizer_name == 'PySLSQP' else 'PGD'
		f.attrs['use_analytic'] = bool(getattr(config, 'use_analytic', True))

	# Compute cumulative iteration offsets per level
	cum = 0
	for lm in levels_meta:
		ni = int(lm.get('iters', {}).get('num_iterations', 0))
		lm['iters']['start_index_global'] = int(cum)
		lm['iters']['end_index_global'] = int(cum + max(ni - 1, 0))
		cum += ni

	# Compute initial perimeter from contour extraction (no IPOPT optimization).
	# This provides a lambda-independent metric for comparing across runs.
	initial_perimeter = None
	try:
		analyzer = ContourAnalyzer(solution_path)
		analyzer.load_results(use_initial_condition=False)

		indicators = analyzer.compute_indicator_functions()
		_, boundary_topology = analyzer.extract_contours_with_topology()

		partition = PartitionContour(mesh, indicators, boundary_topology=boundary_topology)

		target_area_final = total_area / config.n_partitions
		perim_optimizer = PerimeterOptimizer(partition, mesh, target_area_final)
		x0_perim = partition.get_variable_vector()
		initial_perimeter = perim_optimizer.objective(x0_perim)

		logger.info(f"Initial perimeter (contour extraction): {initial_perimeter:.10f}")
	except Exception as e:
		logger.warning(f"Could not compute initial perimeter: {e}")

	# Theoretical total area (if provider exposes it)
	theoretical_total_area = None
	if hasattr(provider, 'theoretical_total_area') and callable(provider.theoretical_total_area):
		theoretical_total_area = float(provider.theoretical_total_area())

	# Save metadata
	meta = {
		'input_parameters': {
			'refinement_levels': int(refinement_levels),
			'use_analytic': bool(getattr(config, 'use_analytic', True)),
			'lambda_penalty': float(getattr(config, 'lambda_penalty', 0.0)),
			'seed': int(config.seed),
			'surface': surface,
			'resolution_labels': [label1, label2],
			'resolution_summary': [v1_info, v2_info],
			'use_discrete_area_for_constraints': bool(getattr(config, 'use_discrete_area_for_constraints', True)),
			'n_partitions': int(config.n_partitions),
			'optimizer_type': 'pyslsqp' if optimizer_name == 'PySLSQP' else 'pgd',
			# Surface-specific parameters for reconstruction by analyzer
			'surface_params': (
				{'r_inner': float(getattr(provider, 'r_inner', 0.5)), 'r_outer': float(getattr(provider, 'r_outer', 1.0))}
				if surface == 'ring' else
				{'R': float(getattr(provider, 'R', 1.0)), 'r': float(getattr(provider, 'r', 0.3))}
			),
			# PGD-only parameters persisted for analysis/reproducibility
			'run_log_frequency': int(getattr(config, 'run_log_frequency', getattr(config, 'log_frequency', 50))),
			'h5_save_stride': int(getattr(config, 'h5_save_stride', 1)),
			'h5_save_vars': list(getattr(config, 'h5_save_vars', ['x'])),
			'h5_always_save_first_last': bool(getattr(config, 'h5_always_save_first_last', True)),
			'refine_trigger_mode': str(getattr(config, 'refine_trigger_mode', 'full')),
			'refine_gnorm_patience': int(getattr(config, 'refine_gnorm_patience', 30)),
			'refine_gnorm_delta': float(getattr(config, 'refine_gnorm_delta', 1e-4)),
			'refine_feas_patience': int(getattr(config, 'refine_feas_patience', 30)),
			'refine_feas_delta': float(getattr(config, 'refine_feas_delta', 1e-6)),
			'enable_refinement_triggers': bool(getattr(config, 'enable_refinement_triggers', True)),
		},
		'levels': levels_meta,
		'final_mesh_stats': mesh.get_mesh_statistics(),
		'final_epsilon': float(results[-1]['epsilon']),
		'final_energy': float(results[-1]['energy']),
		'final_iterations': int(results[-1]['iterations']),
		'run_time_seconds': float(results[-1]['time']),
		'success': bool(results[-1]['success']),
		'datetime': timestamp,
		'user': getpass.getuser(),
		'hostname': socket.gethostname(),
		'platform': platform.platform(),
		'python_version': platform.python_version(),
		'solution_path': os.path.abspath(solution_path),
		'optimizer': 'PySLSQP' if optimizer_name == 'PySLSQP' else 'PGD',
		'theoretical_total_area': theoretical_total_area,
		'initial_perimeter': float(initial_perimeter) if initial_perimeter is not None else None,
		'files': {
			'run_log': os.path.abspath(logfile_path),
			'solution_path': os.path.abspath(solution_path),
		}
	}
	with open(os.path.join(outdir, 'metadata.yaml'), 'w') as f:
		yaml.dump(meta, f)

	# Refinement summary table (independent of optimizer)
	logger.info("Refinement Summary:")
	logger.info("=" * 80)
	logger.info(" Level    Mesh Size       Energy Iterations   Time (s)")
	logger.info("-" * 80)
	for i, res in enumerate(results, start=1):
		lv = levels_meta[i-1]
		mesh_size = f"{int(lv.get(label1, 0))}x{int(lv.get(label2, 0))}"
		logger.info(f"{i:6d} {mesh_size:>11s} {res['energy']:12.6e} {res['iterations']:10d} {res['time']:10.2f}")
	if initial_perimeter is not None:
		logger.info(f"Initial perimeter (contour extraction): {initial_perimeter:.10f}")
	else:
		logger.info("Initial perimeter: could not be computed")
	logger.info(f"✅ Solution file saved: {solution_path}")
	print(f"Surface partition optimization complete. See {logfile_path} for details.\n")
	print(f"Results saved in: {outdir}")
	return results


def main():
	from src.config import Config
	parser = argparse.ArgumentParser(description='Generic surface partition optimization')
	parser.add_argument('--input', type=str, help='Path to input YAML')
	parser.add_argument('--solution-dir', type=str, help='Directory to save solutions')
	parser.add_argument('--surface', type=str, default='torus', help='Surface type (torus)')
	args = parser.parse_args()

	setup_logging(log_level='INFO', log_to_console=True, log_to_file=False)
	logger = get_logger(__name__)

	if args.input:
		with open(args.input, 'r') as f:
			params = yaml.safe_load(f)
		config = Config(params)
	else:
		config = Config()

	if args.surface == 'torus':
		from src.surfaces.torus import TorusMeshProvider
		n_theta = int(getattr(config, 'n_theta', 32))
		n_phi = int(getattr(config, 'n_phi', 24))
		R = float(getattr(config, 'R', 1.0))
		r = float(getattr(config, 'r', 0.3))
		n_theta_increment = int(getattr(config, 'n_theta_increment', 0))
		n_phi_increment = int(getattr(config, 'n_phi_increment', 0))
		provider = TorusMeshProvider(n_theta, n_phi, R, r,
									n_theta_increment=n_theta_increment,
									n_phi_increment=n_phi_increment)
	else:
		raise ValueError(f"Unsupported surface type: {args.surface}")

	optimize_surface_partition(provider, config, solution_dir=args.solution_dir)


if __name__ == '__main__':
	main() 