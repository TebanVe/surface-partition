import numpy as np
import scipy.sparse as sparse
from typing import Tuple, Dict, List

from ..logging_config import get_logger, log_performance


class TriMesh:
	"""
	Surface-agnostic triangle mesh container with P1 FEM assembly.
	Works for planar meshes in R2 or embedded surfaces in R3.
	"""
	def __init__(self, vertices: np.ndarray, faces: np.ndarray):
		self.logger = get_logger(__name__)
		self.vertices = np.asarray(vertices)
		self.faces = np.asarray(faces, dtype=int)
		if self.vertices.ndim != 2 or self.vertices.shape[1] not in (2, 3):
			raise ValueError("vertices must be (N,2) or (N,3)")
		if self.faces.ndim != 2 or self.faces.shape[1] != 3:
			raise ValueError("faces must be (T,3)")
		self.mass_matrix = None
		self.stiffness_matrix = None
		self._triangle_areas = None
		self._mean_triangle_area = None

	@property
	def dim(self) -> int:
		return int(self.vertices.shape[1])

	@property
	def triangle_areas(self) -> np.ndarray:
		if self._triangle_areas is None:
			self._compute_triangle_areas()
		return self._triangle_areas

	def _compute_triangle_areas(self):
		v = self.vertices
		areas = []
		if self.dim == 2:
			for f in self.faces:
				p1, p2, p3 = v[f[0]], v[f[1]], v[f[2]]
				area = 0.5 * abs(np.cross(p2 - p1, p3 - p1))
				areas.append(area)
		else:  # dim == 3
			for f in self.faces:
				p1, p2, p3 = v[f[0]], v[f[1]], v[f[2]]
				n = np.cross(p2 - p1, p3 - p1)
				area = 0.5 * np.linalg.norm(n)
				areas.append(area)
		self._triangle_areas = np.asarray(areas)
		self._mean_triangle_area = float(np.mean(self._triangle_areas)) if len(areas) else 0.0

	@log_performance("matrix computation")
	def compute_matrices(self) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
		"""
		Assemble mass (M) and stiffness (K) matrices for P1 elements on triangles.
		Formulas applied in the triangle's plane; valid for R2 or R3 surfaces.
		"""
		v = self.vertices
		f = self.faces
		T = f.shape[0]
		N = v.shape[0]
		M = sparse.lil_matrix((N, N))
		K = sparse.lil_matrix((N, N))

		# Precompute triangle areas and normals (for 3D)
		if self._triangle_areas is None:
			self._compute_triangle_areas()

		for t in range(T):
			i, j, k = f[t]
			p1, p2, p3 = v[i], v[j], v[k]
			area = self._triangle_areas[t]
			if area == 0:
				continue

			# Local mass matrix
			local_mass = (area / 12.0) * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])

			# Local stiffness matrix using in-plane gradients for P1
			# Gradients of barycentric basis on triangle:
			# For vertices a=i, b=j, c=k, define opposite edges:
			# e_a = p_c - p_b, e_b = p_a - p_c, e_c = p_b - p_a
			# For embedded surfaces in R3, grad phi_a = (n x e_a) / (2A), etc.
			if self.dim == 2:
				e_i = p3 - p2
				e_j = p1 - p3
				e_k = p2 - p1
				# Pseudo "normals" in 2D use rotate by 90°: n = (0,0,1) effectively
				rot = np.array([[0, -1], [1, 0]])
				g_i = rot @ e_i / (2 * area)
				g_j = rot @ e_j / (2 * area)
				g_k = rot @ e_k / (2 * area)
			else:
				n_vec = np.cross(p2 - p1, p3 - p1)
				norm_n = np.linalg.norm(n_vec)
				if norm_n == 0:
					continue
				n = n_vec / norm_n
				e_i = p3 - p2
				e_j = p1 - p3
				e_k = p2 - p1
				g_i = np.cross(n, e_i) / (2 * area)
				g_j = np.cross(n, e_j) / (2 * area)
				g_k = np.cross(n, e_k) / (2 * area)

			local_stiffness = np.array([
				[float(np.dot(g_i, g_i)), float(np.dot(g_i, g_j)), float(np.dot(g_i, g_k))],
				[float(np.dot(g_j, g_i)), float(np.dot(g_j, g_j)), float(np.dot(g_j, g_k))],
				[float(np.dot(g_k, g_i)), float(np.dot(g_k, g_j)), float(np.dot(g_k, g_k))],
			]) * area

			# Assemble
			idx = [i, j, k]
			for a in range(3):
				for b in range(3):
					M[idx[a], idx[b]] += local_mass[a, b]
					K[idx[a], idx[b]] += local_stiffness[a, b]

		self.mass_matrix = M.tocsr()
		self.stiffness_matrix = K.tocsr()
		self.logger.info(f"Matrix computation completed: M {self.mass_matrix.shape}, K {self.stiffness_matrix.shape}")
		return self.mass_matrix, self.stiffness_matrix

	@property
	def M(self) -> sparse.csr_matrix:
		if self.mass_matrix is None:
			self.compute_matrices()
		return self.mass_matrix

	@property
	def K(self) -> sparse.csr_matrix:
		if self.stiffness_matrix is None:
			self.compute_matrices()
		return self.stiffness_matrix

	@property
	def v(self) -> np.ndarray:
		# Column sum of M without densifying: returns shape (N,)
		return np.asarray(self.M.sum(axis=0)).ravel()

	def get_mesh_statistics(self) -> Dict[str, float]:
		areas = self.triangle_areas
		return {
			'n_vertices': int(self.vertices.shape[0]),
			'n_triangles': int(self.faces.shape[0]),
			'total_area': float(self.M.sum()),
			'mean_triangle_area': float(np.mean(areas)) if areas.size else 0.0,
			'min_triangle_area': float(np.min(areas)) if areas.size else 0.0,
			'max_triangle_area': float(np.max(areas)) if areas.size else 0.0,
			'dim': int(self.dim),
		} 