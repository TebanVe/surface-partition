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
from .projection import orthogonal_projection_iterative


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
		wta_balance_enabled: bool = False,
		wta_balance_gamma: float = 0.0,
		wta_balance_power: float = 2.0,
		wta_trim_enabled: bool = False,
		wta_trim_period: int = 200,
		wta_trim_damping: float = 0.5,
		wta_trim_clamp: float = 0.20,
		pgd_reduced_gradient: bool = False,
		pgd_dual_sweeps: int = 8,
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

		# WTA balance term (docs/math/07-phase1-wta-balance): soft-territory
		# penalty P_bal = (gamma/2) sum_k r_k^2, r_k = (T_k - Abar)/Abar with
		# T_k = sum_i v_i u_ik^p / S_i. Default off; zero overhead when off.
		self.wta_balance_enabled = bool(wta_balance_enabled)
		self.wta_balance_gamma = float(wta_balance_gamma)
		self.wta_balance_power = float(wta_balance_power)

		# Discrete-area trim (docs/math/07-phase1-wta-balance sec. 8): every
		# `period` ACCEPTED iterations, retarget the projection's per-cell
		# area targets d toward exact discrete (argmax) equality:
		#   d <- clip(d + damping*(Abar - T_wta), Abar*(1-clamp), Abar*(1+clamp))
		# then renormalize sum(d) = total_area. d is per-optimize() state, so
		# the per-level reset (d = Abar*1) is structural: each refinement
		# level constructs a fresh optimize() call. Default off.
		self.wta_trim_enabled = bool(wta_trim_enabled)
		self.wta_trim_period = int(wta_trim_period)
		self.wta_trim_damping = float(wta_trim_damping)
		self.wta_trim_clamp = float(wta_trim_clamp)

		# P2 reduced-gradient step + acceptance + trigger fix
		# (docs/plans/PHASE1_N1000_VALIDITY_PLAN.md sec. 2.3). The plain PGD
		# step P(clip(x - s g)) freezes on a non-stationary iterate because g
		# is mostly bound-infeasible; stepping along the reduced (projected)
		# gradient g_t restores feasible descent. Default off; when off the
		# line search and triggers are byte-for-byte the legacy path.
		self.pgd_reduced_gradient = bool(pgd_reduced_gradient)
		self.pgd_dual_sweeps = int(pgd_dual_sweeps)
		
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

	def _wta_soft_territory(self, phi: np.ndarray):
		"""Soft-territory quantities for the WTA balance term.

		Args:
			phi: (V, n) density matrix (vertices x cells).

		Returns:
			(S, W_soft, r): the per-vertex normalizer S_i = sum_l u_il^p
			(guarded away from 0), the (V, n) soft weights W_soft = u^p / S,
			and the (n,) relative territory deviations
			r_k = (T_k - target_area)/target_area with T_k = v @ W_soft[:, k].

		On the feasible set S_i in [n^(1-p), 1] (rows on the simplex), so the
		guard is defensive only: the gradient may be evaluated on
		pre-projection iterates. Fully vectorized, O(V*n).
		"""
		p = self.wta_balance_power
		Up = phi ** p
		S = np.maximum(Up.sum(axis=1), np.finfo(np.float64).tiny)
		W_soft = Up / S[:, None]
		T = W_soft.T @ self.v
		r = (T - self.target_area) / self.target_area
		return S, W_soft, r

	def _apply_wta_trim(self, A: np.ndarray, d: np.ndarray) -> np.ndarray:
		"""One damped trim update of the projection area targets d.

		Measures the hard winner-take-all territories via the existing argmax
		machinery (detect_area_imbalance) and moves each target toward the
		deficit: the update's fixed point (clamp inactive) is exact discrete
		equality T_wta = Abar, independent of the soft-territory surrogate.
		Returns the new d; does not mutate A. O(V) + O(n) per call.
		"""
		from ..partition.find_contours import detect_area_imbalance

		result = detect_area_imbalance(A, self.v, self.n_partitions)
		T_wta = np.asarray(result['discrete_areas'], dtype=np.float64)
		Abar = self.target_area
		d_new = np.clip(
			d + self.wta_trim_damping * (Abar - T_wta),
			Abar * (1.0 - self.wta_trim_clamp),
			Abar * (1.0 + self.wta_trim_clamp),
		)
		# The projection requires a feasible target sum; sum(T_wta) == total
		# area identically, so this is the identity unless the clamp bites.
		d_new *= self.total_area / float(np.sum(d_new))
		self.logger.info(
			f"WTA trim: worst discrete dev "
			f"{result['worst_rel_dev'] * 100:.2f}% "
			f"(cell {result['worst_cell']}, "
			f"{result['n_imbalanced']} over {result['rel_threshold'] * 100:.0f}%); "
			f"d range [{d_new.min() / Abar:.3f}, {d_new.max() / Abar:.3f}] x Abar"
		)
		return d_new

	def _reduced_gradient(self, x: np.ndarray, g: np.ndarray, sweeps: int,
	                      alpha0: Optional[np.ndarray] = None,
	                      beta0: Optional[np.ndarray] = None):
		"""Project g onto the tangent of the active constraints (free set).

		The reduced gradient is g_t = g - alpha (x) 1^T - v (x) beta^T on the
		free set (entries NOT pinned at a box bound with outward gradient),
		zero on the pinned set. The duals alpha in R^V (sum-to-one, one per
		vertex/row) and beta in R^N (equal-area, one per cell/column) solve
		the two coupled normal equations
		  sum_{k free} (g_ik - alpha_i - v_i beta_k)       = 0  (row i)
		  sum_{i free} v_i (g_ik - alpha_i - v_i beta_k)    = 0  (col k)
		by `sweeps` Gauss-Seidel block passes (each = two O(V*N) reductions),
		warm-started from (alpha0, beta0). Returns (g_t, alpha, beta).

		Rationale + measured KKT residual in
		docs/plans/PHASE1_N1000_VALIDITY_PLAN.md sec. 2.3.
		"""
		N = len(self.v)
		n = self.n_partitions
		U = x.reshape(N, n)
		G = g.reshape(N, n)
		v = self.v
		# Active-set threshold slightly above the projection's 1e-8 floor:
		# entries this close to a bound with outward gradient stay pinned.
		lb, ub = 1e-6, 1.0 - 1e-6
		pinned = ((U <= lb) & (G > 0.0)) | ((U >= ub) & (G < 0.0))
		Fm = (~pinned).astype(np.float64)
		alpha = np.zeros(N) if alpha0 is None else alpha0.copy()
		beta = np.zeros(n) if beta0 is None else beta0.copy()
		row_cnt = Fm.sum(axis=1)                       # |F_i|
		col_vv = (v[:, None] ** 2 * Fm).sum(axis=0)    # sum_i v_i^2 over free
		row_safe = np.maximum(row_cnt, 1.0)
		col_safe = np.maximum(col_vv, np.finfo(np.float64).tiny)
		for _ in range(max(1, sweeps)):
			num_a = (Fm * (G - v[:, None] * beta[None, :])).sum(axis=1)
			alpha = np.where(row_cnt > 0, num_a / row_safe, 0.0)
			num_b = (Fm * v[:, None] * (G - alpha[:, None])).sum(axis=0)
			beta = np.where(col_vv > 0, num_b / col_safe, 0.0)
		Gt = (G - alpha[:, None] - v[:, None] * beta[None, :]) * Fm
		return Gt.reshape(-1), alpha, beta

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
		
		total_wta_balance = 0.0
		if self.wta_balance_enabled:
			_, _, r = self._wta_soft_territory(phi)
			total_wta_balance = 0.5 * self.wta_balance_gamma * float(r @ r)

		total_energy = total_grad + total_interface + total_penalty + total_wta_balance

		if return_components:
			components = {
				'total': total_energy,
				'grad': total_grad,
				'interface': total_interface,
				'penalty': total_penalty
			}
			if self.wta_balance_enabled:
				components['wta_balance'] = total_wta_balance
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
		if self.wta_balance_enabled:
			# dP_bal/du_ik = (gamma*p/Abar) v_i (u_ik^(p-1)/S_i) (r_k - m_i),
			# m_i = sum_l r_l w_il (docs/math/07-phase1-wta-balance eq. 2.7).
			p = self.wta_balance_power
			S, W_soft, r = self._wta_soft_territory(phi)
			m = W_soft @ r
			G += (
				(self.wta_balance_gamma * p / self.target_area)
				* self.v[:, None]
				* (phi ** (p - 1) / S[:, None])
				* (r[None, :] - m[:, None])
			)
		return g

	def constraint_fun(self, x: np.ndarray, area_targets: Optional[np.ndarray] = None) -> np.ndarray:
		N = len(self.v)
		n = self.n_partitions
		phi = x.reshape(N, n)
		row_sums = np.sum(phi, axis=1)[:-1] - 1.0
		area_sums = self.v @ phi
		if area_targets is None:
			area_constraints = area_sums[:-1] - self.target_area
		else:
			# Trim-retargeted runs measure feasibility against the active
			# per-cell targets d, not the nominal Abar.
			area_constraints = area_sums[:-1] - area_targets[:-1]
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
			if 'wta_balance' in energy_components:
				grp.create_dataset('energy_wta_balance', data=energy_components['wta_balance'])

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

			# Accepted-step counter for the discrete-area trim cadence.
			n_accepted = 0

			# P2 reduced-gradient state: warm-started duals + level-init energy
			# + one-shot stall warning. Only used when pgd_reduced_gradient.
			dual_alpha = None
			dual_beta = None
			E_level_init = E
			stall_warned = False
			if self.pgd_reduced_gradient:
				self.log.setdefault('accepted', [])
				self.log.setdefault('gtnorm', [])

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
				# Step direction and its squared norm (the Armijo surrogate).
				# P2: step along the reduced gradient g_t = P_tangent(g); the
				# plain gradient is mostly bound-infeasible so P(clip(x - s g))
				# freezes. Off: g_search is g and dir_sq == ||g||^2, so the
				# backtracking below is byte-for-byte the legacy path.
				if self.pgd_reduced_gradient:
					g_search, dual_alpha, dual_beta = self._reduced_gradient(
						x, g, self.pgd_dual_sweeps, dual_alpha, dual_beta
					)
					dir_sq = float(np.dot(g_search, g_search))
					gt_norm = float(np.sqrt(dir_sq))
				else:
					g_search = g
					dir_sq = float(np.dot(g, g))
					gt_norm = None
				if profile is not None:
					_t_bt = time.perf_counter()
				while True:
					if profile is not None:
						n_backtracks += 1
					A_trial = x.reshape(N, n) - step * g_search.reshape(N, n)
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
					# Armijo sufficient decrease against the step direction's
					# squared norm (prox form E <= E - (c/s)||x+ - x||^2 in the
					# P2 reduced-gradient regime; ||g||^2 surrogate off).
					if E_trial <= E - armijo_c * step * dir_sq:
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

				if accepted:
					n_accepted += 1
					if (self.wta_trim_enabled
							and n_accepted % self.wta_trim_period == 0):
						# Retarget the projection toward exact discrete
						# equality, re-project the iterate onto the new
						# targets and re-baseline E so the next Armijo test
						# compares like with like. g_post below is computed
						# at the re-projected x, so the gradient-reuse
						# invariant (Change C) is preserved.
						d = self._apply_wta_trim(x.reshape(N, n), d)
						A_ret = orthogonal_projection_iterative(
							x.reshape(N, n), c, d, self.v,
							max_iter=projection_max_iter, tol=projection_tol,
							logger=proj_logger, _prof=profile
						)
						x = A_ret.flatten()
						E = self.compute_energy(x)

				# Recompute gradient and constraints at the accepted iterate (or current if not accepted)
				if profile is not None:
					_t_g = time.perf_counter()
				g_post = self.compute_gradient(x)
				if profile is not None:
					profile.record('gradient', time.perf_counter() - _t_g)
				if profile is not None:
					_t_c = time.perf_counter()
				cvec_post = self.constraint_fun(
					x, area_targets=d if self.wta_trim_enabled else None
				)
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
				if self.pgd_reduced_gradient:
					# gt_norm is the projected stationarity at the pre-step x
					# (from g, which equals last iter's g_post); a valid, cheap
					# stationarity proxy for the trigger below.
					self.log['accepted'].append(bool(accepted))
					self.log['gtnorm'].append(
						gt_norm if gt_norm is not None else gnorm_post
					)
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
					if self.pgd_reduced_gradient:
						# P2 trigger fix (validity plan sec. 2.3): (i) stationarity
						# on the PROJECTED gradient ||g_t||, not the mostly
						# bound-infeasible raw ||g||; (ii) energy-plateau tolerance
						# relative to the level's cumulative decrease; (iii) "no
						# accepted step within the patience window" = STALLED, not
						# converged -- warn and do NOT fire the trigger.
						recent_acc = self.log['accepted'][-self.refine_patience:]
						stalled = (len(recent_acc) >= self.refine_patience
						           and not any(recent_acc))
						if stalled:
							if not stall_warned:
								self.logger.warning(
									f"PGD STALLED at iteration {k}: no accepted "
									f"step in the last {self.refine_patience} "
									f"iterations (||g_t||={gt_norm:.3e}, "
									f"FEAS={feas_post:.3e}). NOT firing the "
									f"refinement trigger (this is a stall, not "
									f"convergence)."
								)
								stall_warned = True
						else:
							stall_warned = False
							if self.wta_trim_enabled:
								# The discrete-area trim retargets `d` every trim_period
								# iters, which re-baselines the energy (a sawtooth), so
								# raw-energy stability is not a valid convergence signal
								# (docs/experiments/04-territory-aware-highn-validation).
								# Trigger on projected stationarity ||g_t|| + feasibility
								# directly -- the balance-plateau trigger.
								stable = True
							else:
								cum_dec = max(E_level_init - E, self.penalty_eps)
								recent = self.log['energy_changes'][-self.refine_patience:]
								stable = all(
									abs(de) < self.refine_delta_energy * cum_dec
									for de in recent
								)
							if stable:
								if refine_trigger_mode == 'energy':
									self.logger.info(f"Refinement triggered at iteration {k} (energy criterion)")
									raise RefinementTriggered()
								gt_ok = (gt_norm < self.refine_grad_tol)
								fe_ok = (feas_post < self.refine_constraint_tol)
								if not gt_ok and len(self.log['gtnorm']) >= refine_gnorm_patience:
									recent_gt = self.log['gtnorm'][-refine_gnorm_patience:]
									gt_plateau = all(abs(recent_gt[i] - recent_gt[i-1]) < refine_gnorm_delta for i in range(1, len(recent_gt)))
									gt_ok = gt_ok or gt_plateau
								if not fe_ok and len(self.log['feas']) >= refine_feas_patience:
									recent_f = self.log['feas'][-refine_feas_patience:]
									fe_plateau = all(abs(recent_f[i] - recent_f[i-1]) < refine_feas_delta for i in range(1, len(recent_f)))
									fe_ok = fe_ok or fe_plateau
								if gt_ok and fe_ok:
									self.logger.info(f"Refinement triggered at iteration {k} (projected stationarity)")
									raise RefinementTriggered()
					else:
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

		# Return best found. With the trim active, energies before and after a
		# retarget are not comparable (the constraint set changed) and only
		# the final iterate satisfies the final targets — return it instead.
		if self.wta_trim_enabled:
			return x.copy(), True
		return best_x.copy(), True 