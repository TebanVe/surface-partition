"""
Surface-agnostic partition project core exports.
"""

__version__ = "0.2.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .config import Config
from .plot_utils import plot_ring_mesh, plot_matrices, plot_mesh_statistics, save_mesh_plots
from .logging_config import setup_logging, get_logger, log_performance
# Expose core optimizer
from .core.pyslsqp_optimizer import PySLSQPOptimizer, RefinementTriggered
from .core.pgd_optimizer import ProjectedGradientOptimizer
from .projection_iterative import (
	orthogonal_projection_iterative,
	orthogonal_projection_direct,
	validate_projection_result,
	create_initial_condition_with_projection
)

__all__ = [
	"Config",
	"plot_ring_mesh",
	"plot_matrices",
	"plot_mesh_statistics",
	"save_mesh_plots",
	"setup_logging",
	"get_logger",
	"log_performance",
	"PySLSQPOptimizer",
	"ProjectedGradientOptimizer",
	"RefinementTriggered",
	"orthogonal_projection_iterative",
	"orthogonal_projection_direct",
	"validate_projection_result",
	"create_initial_condition_with_projection"
] 