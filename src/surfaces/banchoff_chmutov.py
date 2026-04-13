import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph

from .implicit import ImplicitSurfaceProvider
from ..mesh.tri_mesh import TriMesh


class BanchoffChmutovMeshProvider(ImplicitSurfaceProvider):
    """Banchoff-Chmutov surface of order 4, from Bogosel & Oudet (2017), Figure 4."""

    def __init__(
        self,
        n_grid_x=100,
        n_grid_y=100,
        n_grid_z=100,
        n_grid_x_increment=0,
        n_grid_y_increment=0,
    ):
        super().__init__(
            n_grid_x, n_grid_y, n_grid_z, n_grid_x_increment, n_grid_y_increment
        )

    def surface_name(self) -> str:
        return "banchoff_chmutov"

    def implicit_function(self, x, y, z):
        def T4(X):
            return 8 * X**4 - 8 * X**2 + 1

        return T4(x) + T4(y) + T4(z)

    def bounding_box(self):
        return ((-1.1, 1.1), (-1.1, 1.1), (-1.1, 1.1))

    def build(self) -> TriMesh:
        """Build mesh and keep only the largest connected component."""
        mesh = super().build()
        verts, faces = mesh.vertices, mesh.faces
        n_verts = len(verts)

        rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
        cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
        data = np.ones(len(rows), dtype=np.int8)
        adj = sp.csr_matrix((data, (rows, cols)), shape=(n_verts, n_verts))
        adj = adj + adj.T

        n_components, labels = csgraph.connected_components(
            adj, directed=False, return_labels=True
        )

        if n_components > 1:
            sizes = np.bincount(labels)
            largest = int(np.argmax(sizes))
            self.logger.info(
                f"Banchoff-Chmutov: {n_components} components found, "
                f"keeping largest ({sizes[largest]} vertices)"
            )
            keep_verts = np.where(labels == largest)[0]
            vert_map = np.full(n_verts, -1, dtype=int)
            vert_map[keep_verts] = np.arange(len(keep_verts))
            face_mask = np.all(vert_map[faces] >= 0, axis=1)
            verts = verts[keep_verts]
            faces = vert_map[faces[face_mask]]
        else:
            self.logger.info("Banchoff-Chmutov: single connected component")

        return TriMesh(verts, faces)
