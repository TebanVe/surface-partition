import numpy as np
from typing import Optional


def nearest_neighbor_interpolate(old_vertices: np.ndarray, new_vertices: np.ndarray,
                                 old_x: np.ndarray, n_partitions: int) -> np.ndarray:
	"""
	Map a solution from old mesh to new mesh by nearest neighbor in Euclidean space (R2 or R3).
	old_x is flattened (N_old * n_partitions,).
	"""
	old_vertices = np.asarray(old_vertices)
	new_vertices = np.asarray(new_vertices)
	N_old = old_vertices.shape[0]
	N_new = new_vertices.shape[0]
	if old_x.shape[0] != N_old * n_partitions:
		raise ValueError(f"old_x length {old_x.shape[0]} != N_old * n_partitions {N_old * n_partitions}")
	old_phi = old_x.reshape(N_old, n_partitions)
	new_phi = np.zeros((N_new, n_partitions))
	for i in range(N_new):
		p = new_vertices[i]
		d = np.linalg.norm(old_vertices - p, axis=1)
		j = int(np.argmin(d))
		new_phi[i, :] = old_phi[j, :]
	return new_phi.flatten() 