import numpy as np
from typing import Tuple, Optional

from ..logging_config import get_logger
from ..mesh.tri_mesh import TriMesh
from .base import SurfaceProvider


class EllipsoidMeshProvider(SurfaceProvider):
    """
    Surface provider for a triaxial ellipsoid embedded in R3.
    Builds a TriMesh using spherical-coordinate parametrization with
    polar cap triangles at the poles and quad-split strips elsewhere.
    """

    def __init__(
        self,
        n_theta: int,
        n_phi: int,
        a: float = 1.0,
        b: float = 1.0,
        c: float = 1.0,
        n_theta_increment: int = 0,
        n_phi_increment: int = 0,
    ):
        self.logger = get_logger(__name__)
        self.n_theta = int(n_theta)
        self.n_phi = int(n_phi)
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.init_n_theta = int(n_theta)
        self.init_n_phi = int(n_phi)
        self.incr_n_theta = int(n_theta_increment)
        self.incr_n_phi = int(n_phi_increment)

    def surface_name(self) -> str:
        return "ellipsoid"

    def resolution_labels(self) -> Tuple[str, str]:
        return ("nt", "np")

    def get_resolution(self) -> Tuple[int, int]:
        return (self.n_theta, self.n_phi)

    def set_resolution(self, n1: int, n2: int) -> None:
        self.n_theta = int(n1)
        self.n_phi = int(n2)

    def get_initial_resolution(self) -> Tuple[int, int]:
        return (self.init_n_theta, self.init_n_phi)

    def get_resolution_increment(self) -> Tuple[int, int]:
        return (self.incr_n_theta, self.incr_n_phi)

    def resolution_summary(self, refinement_levels: int) -> Tuple[str, str]:
        if refinement_levels > 1:
            final_nt = (
                self.init_n_theta + (refinement_levels - 1) * self.incr_n_theta
            )
            final_np = (
                self.init_n_phi + (refinement_levels - 1) * self.incr_n_phi
            )
            return (
                f"{self.init_n_theta}-{final_nt}_incr{self.incr_n_theta}",
                f"{self.init_n_phi}-{final_np}_incr{self.incr_n_phi}",
            )
        return str(self.init_n_theta), str(self.init_n_phi)

    def _generate_vertices(self) -> np.ndarray:
        """Generate ellipsoid vertices with pole vertices + latitude rings."""
        phi_vals = np.linspace(0, 2 * np.pi, self.n_phi, endpoint=False)
        # n_theta - 2 interior rings (exclude the two poles)
        theta_vals = np.linspace(0, np.pi, self.n_theta)[1:-1]

        verts = []
        # North pole
        verts.append([0.0, 0.0, self.c])
        # Interior rings
        for theta in theta_vals:
            st, ct = np.sin(theta), np.cos(theta)
            for phi in phi_vals:
                cp, sp = np.cos(phi), np.sin(phi)
                verts.append([self.a * st * cp, self.b * st * sp, self.c * ct])
        # South pole
        verts.append([0.0, 0.0, -self.c])
        return np.array(verts, dtype=float)

    def _generate_triangles(self) -> np.ndarray:
        """Generate triangles: polar caps + quad-split strips."""
        n_rings = self.n_theta - 2
        tris = []

        # North cap: vertex 0 connects to the first ring (vertices 1..n_phi)
        for j in range(self.n_phi):
            j_next = (j + 1) % self.n_phi
            tris.append([0, 1 + j, 1 + j_next])

        # Interior strips between ring i and ring i+1
        for i in range(n_rings - 1):
            ring_start = 1 + i * self.n_phi
            next_ring_start = 1 + (i + 1) * self.n_phi
            for j in range(self.n_phi):
                j_next = (j + 1) % self.n_phi
                c0 = ring_start + j
                c1 = ring_start + j_next
                n0 = next_ring_start + j
                n1 = next_ring_start + j_next
                tris.append([c0, n0, c1])
                tris.append([n0, n1, c1])

        # South cap: last vertex connects to the last ring
        south_idx = 1 + n_rings * self.n_phi
        last_ring_start = 1 + (n_rings - 1) * self.n_phi
        for j in range(self.n_phi):
            j_next = (j + 1) % self.n_phi
            tris.append([south_idx, last_ring_start + j_next, last_ring_start + j])

        return np.array(tris, dtype=int)

    def build(self) -> TriMesh:
        self.logger.info(
            f"Building ellipsoid TriMesh: nt={self.n_theta}, np={self.n_phi}, "
            f"a={self.a}, b={self.b}, c={self.c}"
        )
        vertices = self._generate_vertices()
        faces = self._generate_triangles()
        return TriMesh(vertices, faces)

    def theoretical_total_area(self) -> Optional[float]:
        if abs(self.a - self.b) < 1e-12:
            a, c = self.a, self.c
            if abs(a - c) < 1e-12:
                # Sphere
                return 4.0 * np.pi * a**2
            elif c < a:
                # Oblate spheroid
                e = np.sqrt(1.0 - (c / a) ** 2)
                return float(
                    2.0
                    * np.pi
                    * a**2
                    * (1.0 + (c**2 / (a**2 * e)) * np.arctanh(e))
                )
            else:
                # Prolate spheroid
                e = np.sqrt(1.0 - (a / c) ** 2)
                return float(
                    2.0
                    * np.pi
                    * a**2
                    * (1.0 + (c / (a * e)) * np.arcsin(e))
                )
        return None
