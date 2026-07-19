import os
import datetime
import time
import logging
from typing import Optional, Tuple, List, Dict

import h5py
import numpy as np

from ..logging_config import get_logger
from ..profiling import RelaxationProfilingState
from .exceptions import RefinementTriggered
from .projection import orthogonal_projection_iterative, orthogonal_projection_newton


class ProjectedGradientOptimizer:
	"""
	Projected Gradient Descent optimizer with per-step projection onto
	partition unity and equal-area constraints. Produces analyzer-compatible
	summary and internal-data artifacts.
	"""

	def __init__(
		self,
		K: np.ndarray,
		M: np.ndarray,
		v: np.ndarray,
		n_partitions: int,
		epsilon: float,
		total_area: Optional[float] = None,
		lambda_penalty: float = 0.0,
		refine_patience: int = 30,
		refine_delta_energy: float = 1e-4,
		refine_grad_tol: float = 1e-2,
		refine_constraint_tol: float = 1e-2,
		projection_method: str = 'iterative',
		logger=None,
	):
		self.logger = logger or get_logger(__name__)
		self.K = K
		self.M = M
		self.v = v
		self.n_partitions = int(n_partitions)
		self.epsilon = float(epsilon)
		self.lambda_penalty = float(lambda_penalty)
		# Prefer geometric total_area from v; fall back to provided
		self.total_area = float(total_area) if total_area is not None else float(np.sum(v))
		self.target_area = self.total_area / self.n_partitions
		# Precompute total weight and penalty defaults
		self.W = float(np.sum(self.v))
		self.mu_target = 1.0 / self.n_partitions
		self.penalty_target_mode = 'fixed'  # or 'adaptive'
		self.penalty_eps = 1e-8
		
		# Refinement criteria
		self.refine_patience = int(refine_patience)
		self.refine_delta_energy = float(refine_delta_energy)
		self.refine_grad_tol = float(refine_grad_tol)
		self.refine_constraint_tol = float(refine_constraint_tol)
		
		# Logging cache
		self.log = {
			'iterations': [],
			'energy_changes': [],
			'area_evolution': [],
			'gnorm': [],
			'feas': [],
		}
		self.prev_x = None
		self.curr_x = None
		# Constraint-projection backend: 'iterative' (default, byte-identical) or
		# 'newton' (exact dual projection). _warm_beta threads the area duals
		# across PGD steps for the newton path.
		self.projection_method = str(projection_method)
		self._warm_beta = None

	def _project(self, A, c, d, projection_max_iter, projection_tol,
		logger, profile, warm_beta, pre_clip):
		"""Dispatch the per-step constraint projection; returns (A_proj, beta).

		'newton' uses the exact dual projection (docs/math/08-dual-newton-...),
		warm-started on beta and WITHOUT the pre-clip (it returns feasible
		entries directly). 'iterative' is byte-identical to the legacy path:
		optional pre-clip, then the iterative projection (returns beta=None).
		"""
		if self.projection_method == 'newton':
			return orthogonal_projection_newton(
				A, c, d, self.v, max_iter=projection_max_iter,
				tol=min(projection_tol, 1e-10), logger=logger, _prof=profile,
				beta0=warm_beta, return_beta=True,
			)
		A_in = np.clip(A, 1e-8, 1 - 1e-8) if pre_clip else A
		A_proj = orthogonal_projection_iterative(
			A_in, c, d, self.v, max_iter=projection_max_iter,
			tol=projection_tol, logger=logger, _prof=profile,
		)
		return A_proj, None

	def compute_energy(self, x: np.ndarray, return_components: bool = False):
		"""
		Compute total energy and optionally return individual components.
		
		Args:
			x: Solution vector
			return_components: If True, return dict with components; if False, return float
		
		Returns:
			float: Total energy (if return_components=False)
			dict: {'total', 'grad', 'interface', 'penalty'} (if return_components=True)
		"""
		N = len(self.v)
		n = self.n_partitions
		phi = x.reshape(N, n)
		
		# Accumulate components separately
		total_grad = 0.0
		total_interface = 0.0
		total_penalty = 0.0
		
		for i in range(n):
			u = phi[:, i]
			grad_term = self.epsilon * float(u.T @ (self.K @ u))
			interface_vec = u * (1 - u)  # q=u(1-u) double-well; not u**2*(1-u)**2
			interface_term = (1 / self.epsilon) * float(interface_vec.T @ (self.M @ interface_vec))
			total_grad += grad_term
			total_interface += interface_term
		
		if self.lambda_penalty > 0:
			for i in range(n):
				u = phi[:, i]
				# Weighted mean and variance
				mu = float((self.v @ u) / self.W)
				center = u - mu
				var_w = float(((center * self.v) @ center) / self.W)
				# Target variance (fixed or adaptive)
				if self.penalty_target_mode == 'adaptive':
					T = mu * (1.0 - mu)
				else:
					mu_t = self.mu_target
					T = mu_t * (1.0 - mu_t)
				T_eff = T + self.penalty_eps
				penalty_term = self.lambda_penalty * (1.0 - var_w / T_eff)
				total_penalty += penalty_term
		
		total_energy = total_grad + total_interface + total_penalty
		
		if return_components:
			return {
				'total': total_energy,
				'grad': total_grad,
				'interface': total_interface,
				'penalty': total_penalty
			}
		else:
			return total_energy

	def compute_gradient(self, x: np.ndarray) -> np.ndarray:
		N = len(self.v)
		n = self.n_partitions
		phi = x.reshape(N, n)
		g = np.zeros_like(x)
		G = g.reshape(N, n)
		for i in range(n):
			u = phi[:, i]
			grad_grad = 2 * self.epsilon * (self.K @ u)
			interface_vec = u * (1 - u)  # q=u(1-u) double-well; not u**2*(1-u)**2
			grad_interface = (2 / self.epsilon) * (self.M @ interface_vec) * (1 - 2 * u)
			G[:, i] = grad_grad + grad_interface
		if self.lambda_penalty > 0:
			for i in range(n):
				u = phi[:, i]
				# Weighted statistics
				mu = float((self.v @ u) / self.W)
				center = u - mu
				var_w = float(((center * self.v) @ center) / self.W)
				# Target variance (fixed or adaptive)
				if self.penalty_target_mode == 'adaptive':
					T = mu * (1.0 - mu)
				else:
					mu_t = self.mu_target
					T = mu_t * (1.0 - mu_t)
				T_eff = T + self.penalty_eps
				# Gradient of weighted variance: (2/W) diag(v) (u - mu*1)
				grad_var = (2.0 / self.W) * (self.v * center)
				if self.penalty_target_mode == 'adaptive':
					# Full adaptive gradient: -lambda [ (1/T) grad_var - (Var/T^2) (1-2mu) (v/W) ]
					term1 = grad_var / T_eff
					term2 = (var_w / (T_eff * T_eff)) * (1.0 - 2.0 * mu) * (self.v / self.W)
					G[:, i] += -self.lambda_penalty * (term1 - term2)
				else:
					# Fixed target gradient: -lambda * (1/T) * grad_var
					G[:, i] += -self.lambda_penalty * (grad_var / T_eff)
		return g

	def constraint_fun(self, x: np.ndarray) -> np.ndarray:
		N = len(self.v)
		n = self.n_partitions
		phi = x.reshape(N, n)
		row_sums = np.sum(phi, axis=1)[:-1] - 1.0
		area_sums = self.v @ phi
		area_constraints = area_sums[:-1] - self.target_area
		return np.concatenate([row_sums, area_constraints])

	def _save_iteration_h5(self, h5, k: int, x: np.ndarray, g: np.ndarray, f: float, cvec: np.ndarray, save_vars: List[str], energy_components: Optional[Dict[str, float]] = None):
		grp = h5.create_group(f'iter_{k}')
		if 'x' in save_vars:
			grp.create_dataset('x', data=x)
		if 'gradient' in save_vars:
			grp.create_dataset('gradient', data=g)
		if 'objective' in save_vars:
			grp.create_dataset('objective', data=f)
		if 'constraints' in save_vars:
			grp.create_dataset('constraints', data=cvec)
		grp.create_dataset('ismajor', data=True)
		# Save energy components if provided
		if energy_components is not None:
			grp.create_dataset('energy_total', data=energy_components['total'])
			grp.create_dataset('energy_grad', data=energy_components['grad'])
			grp.create_dataset('energy_interface', data=energy_components['interface'])
			grp.create_dataset('energy_penalty', data=energy_components['penalty'])

	def _append_summary_line(self, fh, k: int, f: float, gnorm: float, cnorm: float, feas: float, step: float):
		# Columns (9 tokens): MAJOR-idx, NFEV, NGEV, OBJFUN, GNORM, CNORM, FEAS, OPT, STEP (OPT dummy 0)
		line = f"{k} 0 0 {f:.16e} {gnorm:.16e} {cnorm:.16e} {feas:.16e} 0 {step:.16e}\n"
		fh.write(line)

	def optimize(
		self,
		x0: Optional[np.ndarray] = None,
		maxiter: int = 1000,
		step0: float = 1.0,
		armijo_c: float = 1e-4,
		backtrack_rho: float = 0.5,
		projection_max_iter: int = 100,
		projection_tol: float = 1e-8,
		log_frequency: int = 50,
		results_dir: Optional[str] = None,
		run_name: Optional[str] = None,
		is_mesh_refinement: bool = False,
		data_save_stride: int = 1,
		data_save_vars: Optional[List[str]] = None,
		save_first_last: bool = True,
		refine_trigger_mode: str = 'full',
		refine_gnorm_patience: int = 30,
		refine_gnorm_delta: float = 1e-4,
		refine_feas_patience: int = 30,
		refine_feas_delta: float = 1e-6,
		enable_refinement_triggers: bool = True,
		profile: Optional[RelaxationProfilingState] = None,
	) -> Tuple[np.ndarray, bool]:
		"""
		Run PGD with per-step projection and Armijo backtracking.
		"""
		N = len(self.v)
		n = self.n_partitions
		self._warm_beta = None  # warm-start duals for the newton projection path
		if x0 is None:
			# Random simplex init then project
			x0 = np.random.rand(N * n)
			A0 = x0.reshape(N, n)
			c = np.ones(n)
			d = (np.sum(self.v) / n) * np.ones(n)
			A0, _b = self._project(
				A0, c, d, projection_max_iter, projection_tol,
				self.logger, profile, self._warm_beta, pre_clip=False
			)
			if _b is not None:
				self._warm_beta = _b
			x0 = A0.flatten()

		if results_dir is None:
			results_dir = "results"
		if run_name is None:
			run_name = f"pgd_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
		os.makedirs(results_dir, exist_ok=True)
		summary_filename = os.path.join(results_dir, f"{run_name}_summary.out")
		internal_data_filename = os.path.join(results_dir, f"{run_name}_internal_data.hdf5")
		self.optimization_run_name = run_name
		self.optimization_results_dir = results_dir
		self.summary_file = summary_filename
		self.internal_data_file = internal_data_filename

		# Build a quiet logger for projection calls to avoid per-step spam
		proj_logger = get_logger(__name__ + ".projection")
		proj_logger.setLevel(logging.WARNING)

		A = x0.reshape(N, n).copy()
		c = np.ones(n)
		d = (np.sum(self.v) / n) * np.ones(n)
		A, _b = self._project(
			A, c, d, projection_max_iter, projection_tol,
			proj_logger, profile, self._warm_beta, pre_clip=True
		)
		if _b is not None:
			self._warm_beta = _b
		x = A.flatten()

		E = self.compute_energy(x)
		best_x = x.copy()
		best_E = E
		self.prev_x = None
		self.curr_x = x.copy()

		self.logger.info("Starting PGD optimization")
		start_time = time.time()

		# Determine what to save in HDF5
		save_vars_h5 = ['x'] if data_save_vars is None else list(data_save_vars)
		stride = max(1, int(data_save_stride))

		# Open files
		with open(summary_filename, 'w') as summary_fh, h5py.File(internal_data_filename, 'w') as h5f:
			# Add HDF5 metadata for energy components
			h5f.attrs['energy_schema_version'] = 1
			h5f.attrs['optimizer'] = 'pgd'
			h5f.attrs['epsilon'] = self.epsilon
			h5f.attrs['lambda_penalty'] = self.lambda_penalty
			h5f.attrs['penalty_target_mode'] = self.penalty_target_mode
			h5f.attrs['n_partitions'] = self.n_partitions
			
			# Optional header line (analyzer ignores lines starting with 'MAJOR')
			summary_fh.write("MAJOR NFEV NGEV OBJFUN GNORM CNORM FEAS OPT STEP\n")

			# Gradient at the initial x; reused across iterations (see Change C).
			if profile is not None:
				_t_g = time.perf_counter()
			g = self.compute_gradient(x)
			if profile is not None:
				profile.record('gradient', time.perf_counter() - _t_g)

			# Change A: warm-start the line search from the last accepted step.
			# Seeded at step0 so iteration 0 is identical to the hard-reset version.
			prev_step = float(step0)

			for k in range(maxiter):
				# `g` holds the gradient at the current x on loop entry
				# (the initial gradient above, or g_post carried forward below).
				# Backtracking line search.
				# Warm-start one notch above the last accepted step (capped at
				# step0) so the search converges in ~1-2 trials instead of
				# re-walking from step0 each iteration. backtrack_rho in (0,1) so
				# prev_step/backtrack_rho > prev_step; on iter 0 prev_step==step0
				# and the min selects step0 (identical to the old behaviour).
				step = min(float(step0), prev_step / backtrack_rho)
				accepted = False
				n_backtracks = 0
				# ||g||^2 is invariant across the line search (audit #8): hoist it.
				gg = float(np.dot(g, g))
				if profile is not None:
					_t_bt = time.perf_counter()
				while True:
					if profile is not None:
						n_backtracks += 1
					A_trial = x.reshape(N, n) - step * g.reshape(N, n)
					A_trial, _b_trial = self._project(
						A_trial, c, d, projection_max_iter, projection_tol,
						proj_logger, profile, self._warm_beta, pre_clip=True
					)
					x_trial = A_trial.flatten()
					if profile is not None:
						_t_e = time.perf_counter()
					E_trial = self.compute_energy(x_trial)
					if profile is not None:
						profile.record('energy', time.perf_counter() - _t_e)
					# Armijo condition with ||g||^2 surrogate
					if E_trial <= E - armijo_c * step * gg:
						accepted = True
						x = x_trial
						E = E_trial
						prev_step = step  # carry the accepted step forward
						if _b_trial is not None:
							self._warm_beta = _b_trial
						break
					step *= backtrack_rho
					if step < 1e-12:
						# Unable to make progress; recover from step0 next iteration.
						prev_step = float(step0)
						break
				if profile is not None:
					profile.record('backtrack', time.perf_counter() - _t_bt)
					profile.add_counter('backtracks_per_iter_total', n_backtracks)
					profile.add_counter('major_iterations', 1)

				# Recompute gradient and constraints at the accepted iterate (or current if not accepted)
				if profile is not None:
					_t_g = time.perf_counter()
				g_post = self.compute_gradient(x)
				if profile is not None:
					profile.record('gradient', time.perf_counter() - _t_g)
				if profile is not None:
					_t_c = time.perf_counter()
				cvec_post = self.constraint_fun(x)
				if profile is not None:
					profile.record('constraints', time.perf_counter() - _t_c)
				gnorm_post = float(np.linalg.norm(g_post))
				cnorm_post = float(np.linalg.norm(cvec_post))
				feas_post = float(np.max(np.abs(cvec_post))) if cvec_post.size > 0 else 0.0

				# Save iteration (post-accept values) according to stride/vars
				should_save_iter = (k % stride == 0) or (save_first_last and (k == 0 or k == maxiter - 1))
				if should_save_iter:
					if profile is not None:
						_t_s = time.perf_counter()
					# Compute energy components for saving
					energy_components = self.compute_energy(x, return_components=True)
					self._save_iteration_h5(h5f, k, x, g_post, E, cvec_post, save_vars_h5, energy_components=energy_components)
					if profile is not None:
						profile.record('h5_save', time.perf_counter() - _t_s)
				self._append_summary_line(summary_fh, k, E, gnorm_post, cnorm_post, feas_post, step)
				if profile is not None:
					_t_f = time.perf_counter()
				summary_fh.flush()
				h5f.flush()
				if profile is not None:
					profile.record('h5_flush', time.perf_counter() - _t_f)

				# Track logs
				self.log['iterations'].append(k)
				areas = self.v @ x.reshape(N, n)
				self.log['area_evolution'].append(areas.copy())
				self.log['energy_changes'].append(0.0 if k == 0 else (E - best_E))
				self.log['gnorm'].append(gnorm_post)
				self.log['feas'].append(feas_post)
				self.prev_x = self.curr_x
				self.curr_x = x.copy()
				# Change C: x is not mutated between g_post (above) and the next
				# iteration's line search, so g_post is exactly that iteration's
				# pre-step gradient. Carry it forward to halve gradient evals.
				# (Invariant: nothing below/after must mutate x before loop re-entry.)
				g = g_post

				# Best-so-far
				if E < best_E:
					best_E = E
					best_x = x.copy()

				# Progress log
				if k % max(1, log_frequency) == 0:
					self.logger.info(f"  Iteration {k}: Energy={E:.12e}")
					self.logger.info(f"    GNORM={gnorm_post:.6e}, FEAS={feas_post:.6e}, STEP={step:.3e}")
					areas_log = self.v @ x.reshape(N, n)
					self.logger.info(f"    Target area per partition: {self.target_area:.6e}")
					self.logger.info(f"    Current partition areas: {areas_log}")

				# Refinement trigger check
				if profile is not None:
					_t_tc = time.perf_counter()
				if enable_refinement_triggers and (k + 1 >= self.refine_patience):
					recent = self.log['energy_changes'][-self.refine_patience:]
					stable = all(abs(de) < self.refine_delta_energy for de in recent)
					if stable:
						if refine_trigger_mode == 'energy':
							self.logger.info(f"Refinement triggered at iteration {k} (energy criterion)")
							raise RefinementTriggered()
						else:
							# plateau checks for gnorm and feas
							gn_ok = (gnorm_post < self.refine_grad_tol)
							fe_ok = (feas_post < self.refine_constraint_tol)
							if not gn_ok and len(self.log['gnorm']) >= refine_gnorm_patience:
								recent_g = self.log['gnorm'][-refine_gnorm_patience:]
								gn_plateau = all(abs(recent_g[i] - recent_g[i-1]) < refine_gnorm_delta for i in range(1, len(recent_g)))
								gn_ok = gn_ok or gn_plateau
							if not fe_ok and len(self.log['feas']) >= refine_feas_patience:
								recent_f = self.log['feas'][-refine_feas_patience:]
								fe_plateau = all(abs(recent_f[i] - recent_f[i-1]) < refine_feas_delta for i in range(1, len(recent_f)))
								fe_ok = fe_ok or fe_plateau
							if gn_ok and fe_ok:
								self.logger.info(f"Refinement triggered at iteration {k}")
								raise RefinementTriggered()
				if profile is not None:
					profile.record('trigger_check', time.perf_counter() - _t_tc)

		# Final summary log
		elapsed = time.time() - start_time
		self.logger.info(f"PGD optimization completed: Success=True")
		self.logger.info(f"  Summary saved to: {summary_filename}")
		self.logger.info(f"  Internal data saved to: {internal_data_filename}")
		self.logger.info(f"  optimization completed: {elapsed:.3f}s")

		# Return best found
		return best_x.copy(), True 