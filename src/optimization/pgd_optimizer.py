import os
import datetime
import time
import logging
from collections import deque
from typing import Optional, Tuple, List, Dict

import h5py
import numpy as np

from ..logging_config import get_logger
from ..profiling import RelaxationProfilingState
from .exceptions import RefinementTriggered
from .projection import (
	orthogonal_projection_iterative,
	project_rows_onto_simplex,
)

# Cap for the bounded trailing-window run logs (energy_changes/gnorm/feas).
# Must exceed every refine_*_patience so the trigger windows are unaffected;
# patiences are O(10), so this is ~250x margin while keeping the logs O(1).
_LOG_WINDOW_CAP = 8192


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
		soft_area_constraint: bool = False,
		soft_area_mu: float = 0.0,
		struct_trigger_enabled: bool = False,
		struct_window: int = 2000,
		struct_rate_tol: float = 1e-6,
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

		# Soft continuous equal-area constraint (A2). Default OFF; when off,
		# every code path below is byte-for-byte the exact-projection one.
		#
		# The exact constraint v^T u_k = Abar is enforced by an ITERATIVE
		# alternating projection that costs 93.3% of Phase 1 wall time (mean
		# 46.3 inner iterations per call at N=300). Enabling this flag moves
		# equal area into the objective as
		#     P_area = (mu/2) * sum_k ((v @ u_k - Abar)/Abar)^2,
		#     dP_area/du_ik = (mu/Abar^2) * v_i * (v @ u_k - Abar),
		# leaving only sum-to-one + box on the feasible set -- a CLOSED-FORM,
		# non-iterative per-vertex simplex projection
		# (projection.project_rows_onto_simplex).
		#
		# Rationale: the constraint's delivered meaning -- an equal-area
		# WINNER-TAKE-ALL partition -- is now guaranteed independently, exactly
		# and at any N, by the balanced readout at extraction time
		# (src/partition/balanced_readout.py). What Phase 1 must still do is
		# keep every cell ALIVE and grow a sensible interface structure; the
		# open question this flag exists to answer is whether the hard mass
		# constraint was also the thing keeping cells alive early in the flow.
		self.soft_area_constraint = bool(soft_area_constraint)
		self.soft_area_mu = float(soft_area_mu)

		# Structure-based refinement trigger: a STUCK DETECTOR, not a second
		# convergence test. It watches the only thing the level actually
		# delivers -- the winner-take-all label field argmax_k u_ik -- and fires
		# when that field has been frozen for a long time.
		#
		# Why it exists: the energy-plateau trigger tests |dE| against an
		# ABSOLUTE refine_delta_energy (1e-8), so a level whose partition has
		# stopped changing still runs to the iteration cap as long as the energy
		# keeps creeping. Measured on the N=100 deliverable run_20260709_081548:
		# level 0's labels are frozen from ~iteration 6,000 (<=2 of 9,600
		# vertices flip per 500 iterations) and its discrete areas are unchanged
		# from 10,000 onward, yet it runs all 30,000 iterations -- 13,055 s,
		# 27.1% of that run's entire Phase 1 ladder, after the answer stopped
		# moving.
		#
		# Why the confirmation window is LONG (default 2,000 iterations) rather
		# than the rate threshold being tight: the frozen churn rate at N=100
		# level 0 (4.2e-7 flips per iteration per vertex) is indistinguishable
		# from the rate at N=300 level 4 shortly BEFORE its energy trigger
		# legitimately fires (4.6e-7 at iteration 2,000, fired at 2,246). Rate
		# alone cannot separate "frozen for good" from "nearly done"; duration
		# can. With a 2,000-iteration window this rule provably cannot fire on
		# either N=300 level (they end at 3,838 and 2,247, so a sustained window
		# never completes), leaving those runs governed by the existing rule,
		# while the N=100 pathology fires at ~7,500.
		#
		# Cost: one O(V*n) argmax per iteration, ~0.1% of an iteration.
		# Default OFF -- when off, no argmax is taken and the trigger logic is
		# byte-for-byte the existing one.
		self.struct_trigger_enabled = bool(struct_trigger_enabled)
		self.struct_window = int(struct_window)
		self.struct_rate_tol = float(struct_rate_tol)

		# Refinement criteria
		self.refine_patience = int(refine_patience)
		self.refine_delta_energy = float(refine_delta_energy)
		self.refine_grad_tol = float(refine_grad_tol)
		self.refine_constraint_tol = float(refine_constraint_tol)
		
		# Logging cache. Bounded in memory: 'iterations' is a running count
		# (the only external reader wants len == total iters), and the trigger
		# reads only the last refine_*_patience entries of the scalar traces, so
		# capped deques are behaviorally identical to unbounded lists. The old
		# 'area_evolution' trace was write-only (never read) and is dropped.
		self.log = {
			'iterations': 0,
			'energy_changes': deque(maxlen=_LOG_WINDOW_CAP),
			'gnorm': deque(maxlen=_LOG_WINDOW_CAP),
			'feas': deque(maxlen=_LOG_WINDOW_CAP),
		}
		self.prev_x = None
		self.curr_x = None

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
		
		total_area_penalty = 0.0
		if self.soft_area_constraint:
			r = (self.v @ phi - self.target_area) / self.target_area
			total_area_penalty = 0.5 * self.soft_area_mu * float(r @ r)

		total_energy = (total_grad + total_interface + total_penalty
		                + total_area_penalty)

		if return_components:
			components = {
				'total': total_energy,
				'grad': total_grad,
				'interface': total_interface,
				'penalty': total_penalty
			}
			if self.soft_area_constraint:
				components['area_penalty'] = total_area_penalty
			return components
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
		if self.soft_area_constraint:
			# dP_area/du_ik = (mu/Abar^2) * v_i * (v @ u_k - Abar); rank-one in
			# (v, area deviation), so O(V*n) with no extra passes over K or M.
			dev = self.v @ phi - self.target_area
			G += (self.soft_area_mu / (self.target_area ** 2)) * np.outer(self.v, dev)
		return g

	def constraint_fun(self, x: np.ndarray) -> np.ndarray:
		"""Residuals of the constraints actually ENFORCED on the feasible set.

		FEAS (the max-abs of this vector) drives the refinement trigger's
		feasibility test, so it must describe the hard constraints only. Under
		``soft_area_constraint`` equal area is no longer a constraint -- it is a
		term in the objective, which the trigger already sees through the energy
		plateau -- so including its residual here would permanently pin FEAS
		above ``refine_constraint_tol`` and silently change the trigger's
		meaning between the two arms. The area deviation is instead reported in
		the progress log and via :meth:`area_deviation`.
		"""
		N = len(self.v)
		n = self.n_partitions
		phi = x.reshape(N, n)
		row_sums = np.sum(phi, axis=1)[:-1] - 1.0
		if self.soft_area_constraint:
			return row_sums
		area_sums = self.v @ phi
		area_constraints = area_sums[:-1] - self.target_area
		return np.concatenate([row_sums, area_constraints])

	def area_deviation(self, x: np.ndarray) -> Tuple[float, float]:
		"""(max abs, max relative) continuous cell-area deviation from target.

		Diagnostic for the soft-area arm: this is the quantity the exact
		projection drove to ~1e-10 and the penalty only discourages. Note it is
		the CONTINUOUS mass deviation, not the winner-take-all territory
		deviation that `detect_area_imbalance` gates on.
		"""
		phi = x.reshape(len(self.v), self.n_partitions)
		dev = self.v @ phi - self.target_area
		m = float(np.max(np.abs(dev)))
		return m, m / self.target_area

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
			if 'area_penalty' in energy_components:
				grp.create_dataset('energy_area_penalty', data=energy_components['area_penalty'])

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
		if x0 is None:
			# Random simplex init then project
			x0 = np.random.rand(N * n)
			A0 = x0.reshape(N, n)
			c = np.ones(n)
			d = (np.sum(self.v) / n) * np.ones(n)
			A0 = orthogonal_projection_iterative(A0, c, d, self.v, max_iter=projection_max_iter, tol=projection_tol, logger=self.logger, _prof=profile)
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
		A = np.clip(A, 1e-8, 1 - 1e-8)
		c = np.ones(n)
		d = (np.sum(self.v) / n) * np.ones(n)
		# The ENTRY projection stays exact even under soft_area_constraint: it is
		# one call per level (not per line-search trial), so it costs nothing
		# measurable, and it makes both arms of the A/B start each level from the
		# identical feasible iterate. Only the per-trial projection inside the
		# line search below is swapped for the closed-form simplex projection.
		A = orthogonal_projection_iterative(A, c, d, self.v, max_iter=projection_max_iter, tol=projection_tol, logger=proj_logger, _prof=profile)
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
			h5f.attrs['soft_area_constraint'] = self.soft_area_constraint
			if self.soft_area_constraint:
				h5f.attrs['soft_area_mu'] = self.soft_area_mu
			
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

			# Structure-trigger state: previous winner-take-all label field and
			# a rolling per-iteration flip count over the confirmation window.
			prev_labels = None
			label_changes = None
			if self.struct_trigger_enabled:
				label_changes = deque(maxlen=max(1, self.struct_window))
				prev_labels = np.argmax(x.reshape(N, n), axis=1)

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
					if self.soft_area_constraint:
						# Closed-form simplex projection: enforces sum-to-one +
						# box exactly in one pass. It subsumes the clip above
						# (the box floor is an argument), so no separate clip.
						A_trial = project_rows_onto_simplex(
							A_trial, lo=1e-8, _prof=profile
						)
					else:
						A_trial = np.clip(A_trial, 1e-8, 1 - 1e-8)
						A_trial = orthogonal_projection_iterative(
							A_trial, c, d, self.v, max_iter=projection_max_iter, tol=projection_tol, logger=proj_logger, _prof=profile
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

				# Track logs (bounded — see self.log init)
				self.log['iterations'] += 1
				self.log['energy_changes'].append(0.0 if k == 0 else (E - best_E))
				self.log['gnorm'].append(gnorm_post)
				self.log['feas'].append(feas_post)
				if self.struct_trigger_enabled:
					labels = np.argmax(x.reshape(N, n), axis=1)
					label_changes.append(
						int(np.count_nonzero(labels != prev_labels))
					)
					prev_labels = labels
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
					if self.soft_area_constraint:
						# FEAS above covers sum-to-one only; area is soft now, so
						# report its drift explicitly or it goes unobserved.
						a_abs, a_rel = self.area_deviation(x)
						self.logger.info(
							f"    Soft area: worst |dev|={a_abs:.6e} "
							f"({a_rel * 100:.3f}% of target), mu={self.soft_area_mu:g}"
						)
					self.logger.info(f"    Current partition areas: {areas_log}")

				# Refinement trigger check
				if profile is not None:
					_t_tc = time.perf_counter()
				if (enable_refinement_triggers and self.struct_trigger_enabled
						and len(label_changes) >= self.struct_window):
					# Budget of permitted flips across the whole window:
					# rate_tol (flips per iteration per vertex) * V * window.
					budget = self.struct_rate_tol * N * self.struct_window
					total_flips = sum(label_changes)
					if total_flips <= budget:
						self.logger.info(
							f"Refinement triggered at iteration {k} (structure "
							f"frozen: {total_flips} label flips over the last "
							f"{self.struct_window} iterations, budget "
							f"{budget:.1f}; the winner-take-all partition has "
							f"stopped changing)"
						)
						if profile is not None:
							profile.record(
								'trigger_check', time.perf_counter() - _t_tc
							)
						raise RefinementTriggered()
				if enable_refinement_triggers and (k + 1 >= self.refine_patience):
					recent = list(self.log['energy_changes'])[-self.refine_patience:]
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
								recent_g = list(self.log['gnorm'])[-refine_gnorm_patience:]
								gn_plateau = all(abs(recent_g[i] - recent_g[i-1]) < refine_gnorm_delta for i in range(1, len(recent_g)))
								gn_ok = gn_ok or gn_plateau
							if not fe_ok and len(self.log['feas']) >= refine_feas_patience:
								recent_f = list(self.log['feas'])[-refine_feas_patience:]
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