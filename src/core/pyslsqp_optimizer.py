import numpy as np
from typing import Tuple, Optional
import os
import datetime

try:
	from ..logging_config import get_logger, log_performance
except Exception:
	import sys
	import os
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
	from logging_config import get_logger, log_performance

# PySLSQP
try:
	import pyslsqp
	PYSLSQP_AVAILABLE = True
except Exception:
	PYSLSQP_AVAILABLE = False


class RefinementTriggered(Exception):
	"""Raised when refinement criteria are satisfied during optimization."""
	pass


class PySLSQPOptimizer:
	"""
	Surface-agnostic PySLSQP optimizer for partition problems on triangle meshes.
	- Consumes K, M, v, epsilon, n_partitions, and total_area (for equal-area constraints)
	- Provides full logging, callback, hot-start support, and summary/internal-data saving
	"""
	def __init__(self, K: np.ndarray, M: np.ndarray, v: np.ndarray, n_partitions: int,
				epsilon: float, total_area: Optional[float] = None, lambda_penalty: float = 0.0,
				refine_patience: int = 30, refine_delta_energy: float = 1e-4,
				refine_grad_tol: float = 1e-2, refine_constraint_tol: float = 1e-2,
				logger=None):
		if not PYSLSQP_AVAILABLE:
			raise ImportError("PySLSQP is not available. Please install it first.")
		self.logger = logger or get_logger(__name__)
		self.K = K
		self.M = M
		self.v = v
		self.n_partitions = n_partitions
		self.epsilon = epsilon
		self.lambda_penalty = lambda_penalty
		self.total_area = float(total_area) if total_area is not None else float(np.sum(v))
		self.target_area = self.total_area / n_partitions
		self.refine_patience = refine_patience
		self.refine_delta_energy = refine_delta_energy
		self.refine_grad_tol = refine_grad_tol
		self.refine_constraint_tol = refine_constraint_tol
		self.log = {
			'iterations': [],
			'energy_changes': [],
			'warnings': [],
			'area_evolution': []
		}
		self.prev_x = None
		self.curr_x = None
		self.log_frequency = 50
		self.use_last_valid_iterate = True
		self.use_analytic = True

	# Initial condition helpers
	def validate_initial_condition(self, x0: np.ndarray) -> bool:
		N = len(self.v)
		n = self.n_partitions
		if len(x0) != N * n:
			self.logger.error(f"Initial condition dimension mismatch: got {len(x0)}, expected {N * n}")
			return False
		phi = x0.reshape(N, n)
		if np.any(phi < 0) or np.any(phi > 1):
			self.logger.error("Initial condition violates bounds: values must be in [0, 1]")
			return False
		row_sums = np.sum(phi, axis=1)
		violation = np.max(np.abs(row_sums - 1.0))
		if violation > 1e-6:
			self.logger.warning(f"Initial condition violates partition constraint: max violation = {violation:.2e}")
			return False
		self.logger.info("✓ Initial condition validation passed")
		return True

	def generate_initial_condition(self, method: str = "random") -> np.ndarray:
		N = len(self.v)
		n = self.n_partitions
		if method == "random":
			x0 = np.random.rand(N * n)
			for i in range(N):
				row = x0[i::N]
				s = np.sum(row)
				x0[i::N] = row / s if s > 0 else (1.0 / n)
			return x0
		elif method == "uniform":
			return np.ones(N * n) / n
		else:
			raise ValueError(f"Unknown method: {method}")

	def process_initial_condition(self, x0: np.ndarray, normalize: bool = True) -> np.ndarray:
		N = len(self.v)
		n = self.n_partitions
		if len(x0) != N * n:
			raise ValueError(f"Initial condition dimension mismatch: got {len(x0)}, expected {N * n}")
		phi = x0.reshape(N, n).copy()
		phi = np.clip(phi, 0, 1)
		if normalize:
			for i in range(N):
				s = np.sum(phi[i, :])
				phi[i, :] = phi[i, :] / s if s > 0 else (1.0 / n)
		return phi.flatten()

	# Energy/grad/constraints
	def compute_energy(self, x: np.ndarray) -> float:
		N = len(self.v)
		n = self.n_partitions
		phi = x.reshape(N, n)
		total_energy = 0.0
		for i in range(n):
			u = phi[:, i]
			grad_term = self.epsilon * float(u.T @ (self.K @ u))
			interface_vec = u ** 2 * (1 - u) ** 2
			interface_term = (1 / self.epsilon) * float(interface_vec.T @ (self.M @ interface_vec))
			total_energy += grad_term + interface_term
		if self.lambda_penalty > 0:
			for i in range(n):
				u = phi[:, i]
				mu = float(np.mean(u))
				var = float(np.var(u))
				total_energy += self.lambda_penalty * (1.0 - var / (mu * (1 - mu) + 1e-8))
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
			interface_vec = u ** 2 * (1 - u) ** 2
			grad_interface = (2 / self.epsilon) * (self.M @ interface_vec) * (1 - 2 * u)
			G[:, i] = grad_grad + grad_interface
		if self.lambda_penalty > 0:
			for i in range(n):
				u = phi[:, i]
				mu = float(np.mean(u))
				G[:, i] += self.lambda_penalty * (-2 * (u - mu) / N) / (mu * (1 - mu) + 1e-8)
		return g

	def constraint_fun(self, x: np.ndarray) -> np.ndarray:
		N = len(self.v)
		n = self.n_partitions
		phi = x.reshape(N, n)
		row_sums = np.sum(phi, axis=1)[:-1] - 1.0
		area_sums = self.v @ phi
		area_constraints = area_sums[:-1] - self.target_area
		return np.concatenate([row_sums, area_constraints])

	def constraint_jac(self, x: np.ndarray) -> np.ndarray:
		N = len(self.v)
		n = self.n_partitions
		row_sum_jac = np.zeros((N - 1, N * n))
		for i in range(N - 1):
			row_sum_jac[i, i::N] = 1.0
		area_jac = np.zeros((n - 1, N * n))
		for i in range(n - 1):
			area_jac[i, i * N:(i + 1) * N] = self.v
		return np.vstack([row_sum_jac, area_jac])

	def compute_area_evolution(self, x: np.ndarray) -> np.ndarray:
		N = len(self.v)
		n = self.n_partitions
		phi = x.reshape(N, n)
		return self.v @ phi

	@log_performance("optimization")
	def optimize(self, x0: Optional[np.ndarray] = None, maxiter: int = 1000, ftol: float = 1e-6,
				eps: float = 1e-8, disp: bool = False, use_analytic: bool = True,
				log_frequency: int = 50, use_last_valid_iterate: bool = True,
				is_mesh_refinement: bool = False, results_dir: Optional[str] = None,
				run_name: Optional[str] = None, hot_start_file: Optional[str] = None,
				save_itr: Optional[str] = None, initial_condition_method: str = "random",
				validate_initial: bool = True, process_initial: bool = True) -> Tuple[np.ndarray, bool]:
		# Initial condition
		if x0 is None:
			self.logger.info(f"Generating initial condition using method: {initial_condition_method}")
			x0 = self.generate_initial_condition(method=initial_condition_method)
		if process_initial:
			self.logger.info("Processing initial condition...")
			x0 = self.process_initial_condition(x0, normalize=True)
		if validate_initial and not self.validate_initial_condition(x0):
			raise ValueError("Initial condition validation failed")

		# Files
		if results_dir is None:
			results_dir = "results"
		if run_name is None:
			run_name = f"opt_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
		os.makedirs(results_dir, exist_ok=True)
		summary_filename = os.path.join(results_dir, f"{run_name}_summary.out")
		internal_data_filename = os.path.join(results_dir, f"{run_name}_internal_data.hdf5")
		self.optimization_run_name = run_name
		self.optimization_results_dir = results_dir
		self.summary_file = summary_filename
		self.internal_data_file = internal_data_filename

		# Logs
		self.log_frequency = log_frequency
		self.use_last_valid_iterate = use_last_valid_iterate
		self.use_analytic = use_analytic
		using_hot_start = hot_start_file and os.path.exists(hot_start_file)
		if not is_mesh_refinement:
			self.log = {'iterations': [0], 'energy_changes': [0.0], 'warnings': [],
						'area_evolution': [self.compute_area_evolution(x0)] if not using_hot_start else []}
		else:
			self.log = {'iterations': [], 'energy_changes': [], 'warnings': [], 'area_evolution': []}

		# Configure PySLSQP
		problem_size = len(x0)
		xl = np.zeros(problem_size)
		xu = np.ones(problem_size)
		meq = len(self.v) - 1 + self.n_partitions - 1
		params = {
			'obj': self.compute_energy,
			'grad': self.compute_gradient if use_analytic else None,
			'con': self.constraint_fun,
			'jac': self.constraint_jac,
			'meq': meq,
			'xl': xl,
			'xu': xu,
			'maxiter': maxiter,
			'acc': ftol,
			'iprint': 0 if not disp else 2,
			'callback': None if using_hot_start else self.callback,
			'summary_filename': summary_filename,
			'save_vars': ['x', 'objective', 'constraints', 'gradient', 'jacobian'],
			'save_itr': save_itr or 'major',
			'save_filename': internal_data_filename
		}
		if using_hot_start:
			params['hot_start'] = True
			params['load_filename'] = hot_start_file
			self.logger.info(f"🔥 Hot-start enabled with data from: {hot_start_file}")
		self.logger.info(f"Starting PySLSQP optimization with {'analytic' if use_analytic else 'finite-difference'} gradients...")
		try:
			res = pyslsqp.optimize(x0, **params)
			x_opt = res['x']
			success = bool(res['success'])
			self.logger.info(f"PySLSQP optimization completed: Success={success}")
			self.logger.info(f"  Summary saved to: {summary_filename}")
			self.logger.info(f"  Internal data saved to: {internal_data_filename}")
		except Exception as e:
			self.logger.error(f"PySLSQP optimization failed: {e}")
			return x0, False
		if not success and self.prev_x is not None and self.use_last_valid_iterate:
			self.logger.warning("Returning last valid iterate before unsuccessful termination.")
			for key in ['iterations', 'energy_changes', 'area_evolution']:
				if self.log[key]:
					self.log[key].pop()
			return self.prev_x.copy(), success
		return x_opt, success

	def callback(self, xk: np.ndarray):
		self.prev_x = getattr(self, 'curr_x', None)
		iter_num = len(self.log['iterations'])
		N = len(self.v)
		n = self.n_partitions
		phi = xk.reshape(N, n)
		self.curr_x = xk.copy()
		current_energy = self.compute_energy(xk)
		self.log['iterations'].append(iter_num)
		if iter_num > 0:
			self.log['energy_changes'].append(current_energy - self.log.get('last_energy', current_energy))
		else:
			self.log['energy_changes'].append(0.0)
		self.log['last_energy'] = current_energy
		# Refinement trigger
		if len(self.log['energy_changes']) >= self.refine_patience:
			recent = self.log['energy_changes'][-self.refine_patience:]
			stable = all(abs(de) < self.refine_delta_energy for de in recent)
			if stable:
				grad_norm = np.linalg.norm(self.compute_gradient(xk))
				constraint_violation = np.max(np.abs(self.constraint_fun(xk)))
				if grad_norm < self.refine_grad_tol and constraint_violation < self.refine_constraint_tol:
					self.logger.info(f"Refinement triggered at iteration {iter_num}")
					raise RefinementTriggered()
		# Progress log
		if iter_num % self.log_frequency == 0:
			self.logger.info(f"  Iteration {iter_num}: Energy={current_energy:.6e}")
		# Area evolution
		areas = self.v @ phi
		self.log['area_evolution'].append(areas.copy())
		if iter_num % self.log_frequency == 0:
			self.logger.info(f"    Target area per partition: {self.target_area:.6e}")
			self.logger.info(f"    Current partition areas: {areas}")