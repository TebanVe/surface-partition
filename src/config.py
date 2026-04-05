import numpy as np

class Config:
	"""Configuration parameters for surface partition optimization."""
	def __init__(self, params=None):
		# Optimization parameters
		self.n_partitions = 3
		self.lambda_penalty = 1.0
		self.max_iter = 1000
		
		# SLSQP parameters
		self.tol = 1e-6
		self.slsqp_eps = 1e-8
		self.slsqp_disp = False
		
		# Choose constraint area source
		# True: use discrete area (Σv) for constraints; False: use theoretical area
		self.use_discrete_area_for_constraints = True
		
		# Optimization parameters
		self.starget = 1.0
		self.c = 0.5  # Armijo condition parameter
		self.rho = 0.5  # Step size reduction factor
		self.m = 10  # Number of corrections to store (for LBFGS)
		self.seed = 42  # Default seed for random initialization
		
		# PGD parameters
		self.pgd_step0 = 1.0
		self.pgd_armijo_c = 1e-4
		self.pgd_backtrack_rho = 0.5
		self.pgd_projection_max_iter = 100
		self.pgd_projection_tol = 1e-8
		
		# Refinement parameters
		self.refinement_levels = 1
		self.use_analytic = True
		
		# Mesh refinement convergence criteria
		self.refine_patience = 30
		self.refine_delta_energy = 1e-4
		self.refine_grad_tol = 1e-2
		self.refine_constraint_tol = 1e-2
		# Plateau criteria for gradient and feasibility
		self.refine_gnorm_patience = 30
		self.refine_gnorm_delta = 1e-4
		self.refine_feas_patience = 30
		self.refine_feas_delta = 1e-6
		# Enable/disable early refinement triggers
		self.enable_refinement_triggers = True
		
		# Initial condition parameters
		self.use_custom_initial_condition = False
		self.initial_condition_path = None
		self.allow_random_fallback = True
		
		# Projection parameters for initial condition creation
		self.projection_max_iter = 100
		
		# Matrix testing parameters
		self.test_barycentric = True
		self.test_stable = True
		self.test_stable_fem = True
		self.matrix_test_output_dir = "matrix_test_results"
		
		# Logging parameters
		self.log_frequency = 50
		self.use_last_valid_iterate = True
		
		# PGD-only: artifact size and logging controls
		self.run_log_frequency = 100  # console/file logging cadence for PGD
		self.h5_save_stride = 10      # save every k-th iterate to HDF5
		self.h5_save_vars = ['x']     # datasets to store in HDF5 among ['x','constraints']
		self.h5_always_save_first_last = True  # ensure first/last are saved regardless of stride
		self.refine_trigger_mode = 'full'  # 'full' uses energy+gnorm+feas, 'energy' uses energy only
		# PGD-only: constant-phase penalty controls
		self.penalty_target_mode = 'fixed'  # 'fixed' (paper) or 'adaptive'
		self.penalty_eps = 1e-8            # small stabilizer in denominators
		
		# Torus parameters (for torus surface)
		self.n_theta = 32       # Samples along major circle
		self.n_phi = 24         # Samples along minor (tube) circle
		self.R = 1.0            # Major radius
		self.r = 0.3            # Minor radius
		self.n_theta_increment = 0
		self.n_phi_increment = 0
		
		# Override with params if provided
		if params:
			print("\nOverriding default parameters with:")
			for k, v in params.items():
				if hasattr(self, k):
					old_value = getattr(self, k)
					# Ensure numeric parameters are properly typed
					default_value = getattr(self, k)
					if isinstance(default_value, float) and v is not None:
						v = float(v)
					elif isinstance(default_value, int) and v is not None:
						v = int(v)
					elif isinstance(default_value, bool) and v is not None:
						v = bool(v)
					setattr(self, k, v)
					print(f"  {k}: {old_value} -> {v}")
				else:
					print(f"  Warning: Unknown parameter '{k}' with value {v}")
			print("\n")