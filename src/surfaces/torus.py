import numpy as np
from typing import Tuple

from ..logging_config import get_logger
from ..core.tri_mesh import TriMesh


class TorusMeshProvider:
	"""
	Surface provider for a torus of revolution embedded in R3.
	Builds a TriMesh from torus parameters and provides naming metadata
	for orchestrators.
	"""

	def __init__(self, n_theta: int, n_phi: int, R: float, r: float,
				 n_theta_increment: int = 0, n_phi_increment: int = 0):
		self.logger = get_logger(__name__)
		# Resolution parameters (primary)
		self.n_theta = int(n_theta)
		self.n_phi = int(n_phi)
		# Geometry parameters
		self.R = float(R)
		self.r = float(r)
		# Store initial and increment values for refinement summary
		self.init_n_theta = int(n_theta)
		self.init_n_phi = int(n_phi)
		self.incr_n_theta = int(n_theta_increment)
		self.incr_n_phi = int(n_phi_increment)
		# Aliases to match orchestrator's current attribute names (compat layer)
		self.init_n_radial = self.init_n_theta
		self.init_n_angular = self.init_n_phi
		self.incr_n_radial = self.incr_n_theta
		self.incr_n_angular = self.incr_n_phi

	# Surface identity and resolution labels
	def surface_name(self) -> str:
		return "torus"

	def resolution_labels(self) -> Tuple[str, str]:
		# nt: number of samples along major circle, np: along tube circle
		return ("nt", "np")

	def get_resolution(self) -> Tuple[int, int]:
		return (int(self.n_theta), int(self.n_phi))

	def set_resolution(self, n1: int, n2: int) -> None:
		self.n_theta = int(n1)
		self.n_phi = int(n2)

	def resolution_summary(self, refinement_levels: int) -> Tuple[str, str]:
		if refinement_levels > 1:
			final_nt = self.init_n_theta + (refinement_levels - 1) * self.incr_n_theta
			final_np = self.init_n_phi + (refinement_levels - 1) * self.incr_n_phi
			v1 = f"{self.init_n_theta}-{final_nt}_incr{self.incr_n_theta}"
			v2 = f"{self.init_n_phi}-{final_np}_incr{self.incr_n_phi}"
			return v1, v2
		else:
			return f"{self.init_n_theta}", f"{self.init_n_phi}"

	def _generate_vertices(self) -> np.ndarray:
		"""Generate torus vertices in R3 using parametric equations."""
		u_vals = np.linspace(0.0, 2.0 * np.pi, self.n_theta, endpoint=False)
		v_vals = np.linspace(0.0, 2.0 * np.pi, self.n_phi, endpoint=False)
		verts = []
		for u in u_vals:
			cu, su = np.cos(u), np.sin(u)
			for v in v_vals:
				cv, sv = np.cos(v), np.sin(v)
				x = (self.R + self.r * cv) * cu
				y = (self.R + self.r * cv) * su
				z = self.r * sv
				verts.append([x, y, z])
		return np.array(verts, dtype=float)

	def _generate_triangles(self) -> np.ndarray:
		"""Generate counterclockwise triangles for structured torus grid with wrap-around."""
		tris = []
		for i in range(self.n_theta):
			for j in range(self.n_phi):
				current = i * self.n_phi + j
				next_theta = ((i + 1) % self.n_theta) * self.n_phi + j
				next_phi = i * self.n_phi + ((j + 1) % self.n_phi)
				next_both = ((i + 1) % self.n_theta) * self.n_phi + ((j + 1) % self.n_phi)
				# Triangle 1 and 2 per quad
				tris.append([current, next_theta, next_phi])
				tris.append([next_theta, next_both, next_phi])
		return np.array(tris, dtype=int)

	def build(self) -> TriMesh:
		self.logger.info(
			f"Building torus TriMesh: nt={self.n_theta}, np={self.n_phi}, R={self.R}, r={self.r}"
		)
		vertices = self._generate_vertices()
		faces = self._generate_triangles()
		return TriMesh(vertices, faces)

	def theoretical_total_area(self) -> float:
		# Surface area of a torus of revolution
		return float(4.0 * (np.pi ** 2) * self.R * self.r)
