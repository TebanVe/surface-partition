import numpy as np

from .implicit import ImplicitSurfaceProvider


class DoubleTorusMeshProvider(ImplicitSurfaceProvider):
    """Double torus from Bogosel & Oudet (2017), Figure 3."""

    def __init__(
        self,
        n_grid_x=100,
        n_grid_y=100,
        n_grid_z=100,
        c=0.03,
        n_grid_x_increment=0,
        n_grid_y_increment=0,
    ):
        super().__init__(
            n_grid_x, n_grid_y, n_grid_z, n_grid_x_increment, n_grid_y_increment
        )
        self.c = float(c)

    def surface_name(self) -> str:
        return "double_torus"

    def implicit_function(self, x, y, z):
        return (x * (x - 1) ** 2 * (x - 2) + y**2) ** 2 + z**2 - self.c

    def bounding_box(self):
        return ((-0.5, 2.5), (-1.0, 1.0), (-0.3, 0.3))
