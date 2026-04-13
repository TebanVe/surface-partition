import numpy as np
from abc import abstractmethod
from typing import Tuple, Optional

from ..logging_config import get_logger
from ..mesh.tri_mesh import TriMesh
from .base import SurfaceProvider


class ImplicitSurfaceProvider(SurfaceProvider):
    """Base for surfaces defined as zero level sets of f(x,y,z) = 0.

    Resolution is controlled by (n_grid_x, n_grid_y) for the SurfaceProvider
    two-dimensional interface.  The z-grid resolution scales proportionally
    with n_grid_x so that voxel aspect ratio is preserved across refinement
    levels.  There is no separate n_grid_z_increment parameter.
    """

    def __init__(
        self,
        n_grid_x: int,
        n_grid_y: int,
        n_grid_z: int,
        n_grid_x_increment: int = 0,
        n_grid_y_increment: int = 0,
    ):
        self.logger = get_logger(__name__)
        self.n_grid_x = int(n_grid_x)
        self.n_grid_y = int(n_grid_y)
        self.n_grid_z = int(n_grid_z)
        self.init_n_grid_x = int(n_grid_x)
        self.init_n_grid_y = int(n_grid_y)
        self.init_n_grid_z = int(n_grid_z)
        self.incr_n_grid_x = int(n_grid_x_increment)
        self.incr_n_grid_y = int(n_grid_y_increment)

    @abstractmethod
    def implicit_function(self, x, y, z):
        """Evaluate f(x,y,z). Surface is where f = 0. Must work with numpy arrays."""
        ...

    @abstractmethod
    def bounding_box(self):
        """Return ((xmin,xmax), (ymin,ymax), (zmin,zmax)) enclosing the surface."""
        ...

    def get_resolution(self) -> Tuple[int, int]:
        return (self.n_grid_x, self.n_grid_y)

    def set_resolution(self, n1: int, n2: int) -> None:
        ratio = n1 / self.init_n_grid_x if self.init_n_grid_x > 0 else 1.0
        self.n_grid_x = int(n1)
        self.n_grid_y = int(n2)
        self.n_grid_z = max(int(self.init_n_grid_z * ratio), 4)

    def get_initial_resolution(self) -> Tuple[int, int]:
        return (self.init_n_grid_x, self.init_n_grid_y)

    def get_resolution_increment(self) -> Tuple[int, int]:
        return (self.incr_n_grid_x, self.incr_n_grid_y)

    def resolution_labels(self) -> Tuple[str, str]:
        return ("ngx", "ngy")

    def resolution_summary(self, refinement_levels: int) -> Tuple[str, str]:
        if refinement_levels > 1:
            final_x = (
                self.init_n_grid_x
                + (refinement_levels - 1) * self.incr_n_grid_x
            )
            final_y = (
                self.init_n_grid_y
                + (refinement_levels - 1) * self.incr_n_grid_y
            )
            return (
                f"{self.init_n_grid_x}-{final_x}_incr{self.incr_n_grid_x}",
                f"{self.init_n_grid_y}-{final_y}_incr{self.incr_n_grid_y}",
            )
        return str(self.init_n_grid_x), str(self.init_n_grid_y)

    def build(self) -> TriMesh:
        try:
            from skimage.measure import marching_cubes
        except ImportError:
            raise ImportError(
                "scikit-image is required for implicit surface providers. "
                "Install with: pip install -e '.[implicit]'"
            )
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = self.bounding_box()
        x = np.linspace(xmin, xmax, self.n_grid_x)
        y = np.linspace(ymin, ymax, self.n_grid_y)
        z = np.linspace(zmin, zmax, self.n_grid_z)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        volume = self.implicit_function(X, Y, Z)
        spacing = (
            (xmax - xmin) / (self.n_grid_x - 1),
            (ymax - ymin) / (self.n_grid_y - 1),
            (zmax - zmin) / (self.n_grid_z - 1),
        )
        verts, faces, normals, values = marching_cubes(
            volume, level=0.0, spacing=spacing
        )
        verts[:, 0] += xmin
        verts[:, 1] += ymin
        verts[:, 2] += zmin
        self.logger.info(
            f"Built implicit surface mesh: {len(verts)} vertices, "
            f"{len(faces)} triangles "
            f"(grid {self.n_grid_x}x{self.n_grid_y}x{self.n_grid_z})"
        )
        return TriMesh(verts, faces)

    def theoretical_total_area(self) -> Optional[float]:
        return None
