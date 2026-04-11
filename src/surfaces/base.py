from abc import ABC, abstractmethod
from typing import Tuple, Optional

from ..mesh.tri_mesh import TriMesh


class SurfaceProvider(ABC):
    """Abstract base class for surface mesh providers.

    Each provider builds a TriMesh for a specific closed surface in R3.
    The pipeline uses resolution methods for multi-level mesh refinement:
    at level L the resolution is get_initial_resolution() + L * get_resolution_increment().
    """

    @abstractmethod
    def surface_name(self) -> str:
        """Short identifier used in file/directory naming (e.g. 'torus')."""
        ...

    @abstractmethod
    def resolution_labels(self) -> Tuple[str, str]:
        """Short labels for the two resolution dimensions (e.g. ('nt', 'np'))."""
        ...

    @abstractmethod
    def get_resolution(self) -> Tuple[int, int]:
        """Current resolution as (n1, n2)."""
        ...

    @abstractmethod
    def set_resolution(self, n1: int, n2: int) -> None:
        """Set current resolution to (n1, n2)."""
        ...

    @abstractmethod
    def get_initial_resolution(self) -> Tuple[int, int]:
        """Base resolution at refinement level 0."""
        ...

    @abstractmethod
    def get_resolution_increment(self) -> Tuple[int, int]:
        """Per-level resolution increment (dn1, dn2)."""
        ...

    @abstractmethod
    def resolution_summary(self, refinement_levels: int) -> Tuple[str, str]:
        """Human-readable resolution range strings for output naming."""
        ...

    @abstractmethod
    def build(self) -> TriMesh:
        """Construct and return a TriMesh at the current resolution."""
        ...

    def theoretical_total_area(self) -> Optional[float]:
        """Exact surface area if known analytically, else None."""
        return None
