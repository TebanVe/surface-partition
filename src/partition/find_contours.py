import os
import h5py
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass

from ..logging_config import get_logger


# A cell whose peak density never reaches the 0.5 argmax/contour level is
# "weak": it barely wins (or loses) every vertex. 0.5 is the partition's
# defining level (winner-take-all / 0.5 level-set), not a tunable parameter.
WEAK_CELL_DENSITY_THRESHOLD = 0.5


def detect_dormant_cells(densities: np.ndarray,
                         weak_threshold: float = WEAK_CELL_DENSITY_THRESHOLD) -> dict:
    """Identify cells that vanish or nearly vanish under winner-take-all.

    A relaxation solution can satisfy the equal-area and sum-to-one constraints
    exactly while some density column never (or barely) wins the per-vertex
    argmax: the cell carries its target area diffusely and is everywhere
    second place. Such a solution is a consistent continuous minimizer but not
    a usable N-region partition (the discrete partition is missing regions).

    Args:
        densities: (V, n) density matrix (rows approximately sum to 1).
        weak_threshold: a cell is "weak" if its peak density never reaches this
            level (the argmax/contour level that defines region interiors).

    Returns:
        dict with keys: wins_per_cell, max_density_per_cell, dead (cells with
        zero argmax wins), weak (cells with peak density < weak_threshold),
        n_cells, n_effective (n_cells - len(dead)), weak_threshold.
    """
    phi = np.asarray(densities)
    if phi.ndim != 2:
        raise ValueError(f"densities must be 2-D (V, n); got shape {phi.shape}")
    n_cells = int(phi.shape[1])
    winners = np.argmax(phi, axis=1)
    wins = np.bincount(winners, minlength=n_cells)
    max_density = phi.max(axis=0)
    dead = [k for k in range(n_cells) if wins[k] == 0]
    weak = [k for k in range(n_cells) if max_density[k] < weak_threshold]
    return {
        'wins_per_cell': [int(w) for w in wins],
        'max_density_per_cell': [float(m) for m in max_density],
        'dead': dead,
        'weak': weak,
        'n_cells': n_cells,
        'n_effective': n_cells - len(dead),
        'weak_threshold': float(weak_threshold),
    }


@dataclass
class BoundaryTriangleInfo:
    """
    Container for boundary triangle topology information.
    
    Stores information about a triangle that crosses partition boundaries,
    including which edges are crossed and which cells meet at this triangle.
    
    Attributes:
        triangle_idx: Index of the triangle in the mesh
        vertices: Tuple of 3 vertex indices (v1, v2, v3)
        vertex_labels: Tuple of 3 cell labels for the vertices
        crossed_edges: List of edges that cross cell boundaries
        segments: List of 3D segment arrays extracted from this triangle
    """
    triangle_idx: int
    vertices: Tuple[int, int, int]
    vertex_labels: Tuple[int, int, int]
    crossed_edges: List[Tuple[int, int]]
    segments: List[np.ndarray]


class ContourAnalyzer:
    """
    Analyze and visualize contours for partition results on triangulated surfaces
    in R^2 (planar) or embedded in R^3.

    This follows the paper's approach (see manifold_partition.md, eq. (5.1)):
    - Compute indicator functions via winner-takes-all on densities
    - Extract 0.5 level-set segments per region across mesh triangles
    """

    def __init__(self, result_path: str, logger=None):
        self.result_path = Path(result_path)
        self.logger = logger or get_logger(__name__)

        self.x: Optional[np.ndarray] = None
        self.vertices: Optional[np.ndarray] = None  # shape (N, 2)
        self.faces: Optional[np.ndarray] = None     # shape (T, 3)
        self.densities: Optional[np.ndarray] = None  # shape (N, n_partitions)
        self.level: float = 0.5

    def load_results(self, use_initial_condition: bool = False) -> None:
        """
        Load solution and mesh from .h5 file.

        Args:
            use_initial_condition: If True, load x0 instead of x_opt
        """
        if not self.result_path.exists() or not self.result_path.is_file():
            raise FileNotFoundError(f"Solution file not found: {self.result_path}")
        if self.result_path.suffix.lower() != ".h5":
            raise ValueError(f"Expected .h5 solution file, got: {self.result_path}")

        dataset = 'x0' if use_initial_condition else 'x_opt'
        with h5py.File(self.result_path, 'r') as f:
            if dataset not in f:
                raise ValueError(f"Dataset '{dataset}' not found in {self.result_path}")
            if 'vertices' not in f or 'faces' not in f:
                raise ValueError("Solution file must contain 'vertices' and 'faces' datasets")

            self.x = f[dataset][:]
            self.vertices = f['vertices'][:]
            self.faces = f['faces'][:]

        if self.vertices.ndim != 2 or self.vertices.shape[1] not in (2, 3):
            raise ValueError(f"Vertices must be (N,2) or (N,3); got {self.vertices.shape}")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError(f"Faces must be (T,3); got {self.faces.shape}")

        n_vertices = self.vertices.shape[0]
        if self.x.shape[0] % n_vertices != 0:
            raise ValueError(
                f"Solution length {self.x.shape[0]} not divisible by n_vertices {n_vertices}"
            )
        n_partitions = self.x.shape[0] // n_vertices
        self.densities = self.x.reshape(n_vertices, n_partitions)

        self.logger.info(
            f"Loaded {'x0' if use_initial_condition else 'x_opt'}: "
            f"{n_vertices} vertices, {n_partitions} partitions"
        )

    def compute_indicator_functions(self) -> np.ndarray:
        """
        Compute indicator functions chi via winner-takes-all on densities.

        Returns:
            chi: (N, n_partitions) binary matrix
        """
        if self.densities is None:
            raise ValueError("Call load_results() before compute_indicator_functions()")

        n_vertices, n_partitions = self.densities.shape
        chi = np.zeros_like(self.densities)
        max_indices = np.argmax(self.densities, axis=1)
        chi[np.arange(n_vertices), max_indices] = 1.0
        return chi

    def _find_triangle_level_segments(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
                                      d1: float, d2: float, d3: float, level: float) -> List[np.ndarray]:
        """
        Find up to one segment of the level-set within a triangle for a scalar field.

        Returns a list with either 0 or 1 segment; each segment is (2, D) with D in {2,3}.
        """
        points = []
        # Edge (p1, p2)
        if (d1 > level) != (d2 > level):
            t = (level - d1) / (d2 - d1)
            points.append(p1 + t * (p2 - p1))
        # Edge (p2, p3)
        if (d2 > level) != (d3 > level):
            t = (level - d2) / (d3 - d2)
            points.append(p2 + t * (p3 - p2))
        # Edge (p3, p1)
        if (d3 > level) != (d1 > level):
            t = (level - d3) / (d1 - d3)
            points.append(p3 + t * (p1 - p3))

        if len(points) == 2:
            return [np.vstack(points)]  # shape (2, 2)
        return []

    def extract_contours_with_topology(self, level: float = 0.5) -> Tuple[Dict[int, List[np.ndarray]], Dict[int, List[BoundaryTriangleInfo]]]:
        """
        Extract contour segments AND boundary topology information per region.
        
        This method performs the same contour extraction as extract_contours(),
        but also collects topology information about which triangles cross boundaries
        and which edges are involved. This information can be reused by PartitionContour
        to avoid redundant triangle scanning.

        Args:
            level: level-set threshold (default 0.5)
            
        Returns:
            contours: Dict[region_idx] -> List[segment arrays (2, D)]
            boundary_topology: Dict[region_idx] -> List[BoundaryTriangleInfo]
        """
        if self.densities is None:
            raise ValueError("Call load_results() before extract_contours_with_topology()")

        self.level = level
        chi = self.compute_indicator_functions()
        vertex_labels = np.argmax(chi, axis=1)  # Global vertex labels
        n_regions = chi.shape[1]

        contours: Dict[int, List[np.ndarray]] = {i: [] for i in range(n_regions)}
        boundary_topology: Dict[int, List[BoundaryTriangleInfo]] = {i: [] for i in range(n_regions)}

        for region_idx in range(n_regions):
            chi_region = chi[:, region_idx]

            for tri_idx, face in enumerate(self.faces):
                v1, v2, v3 = map(int, face)
                d1, d2, d3 = chi_region[v1], chi_region[v2], chi_region[v3]
                label1, label2, label3 = vertex_labels[v1], vertex_labels[v2], vertex_labels[v3]

                # Only if triangle is cut by the level set
                if (d1 > level) != (d2 > level) or (d2 > level) != (d3 > level) or (d3 > level) != (d1 > level):
                    p1 = self.vertices[v1]
                    p2 = self.vertices[v2]
                    p3 = self.vertices[v3]
                    
                    # Extract segment coordinates (existing logic)
                    segments = self._find_triangle_level_segments(p1, p2, p3, d1, d2, d3, level)
                    contours[region_idx].extend(segments)
                    
                    # NEW: Identify which edges are crossed
                    crossed_edges = []
                    if (d1 > level) != (d2 > level):
                        crossed_edges.append((v1, v2))
                    if (d2 > level) != (d3 > level):
                        crossed_edges.append((v2, v3))
                    if (d3 > level) != (d1 > level):
                        crossed_edges.append((v3, v1))
                    
                    # Store topology information
                    tri_info = BoundaryTriangleInfo(
                        triangle_idx=tri_idx,
                        vertices=(v1, v2, v3),
                        vertex_labels=(label1, label2, label3),
                        crossed_edges=crossed_edges,
                        segments=segments
                    )
                    boundary_topology[region_idx].append(tri_info)

            self.logger.info(f"Region {region_idx}: extracted {len(contours[region_idx])} contour segments "
                           f"from {len(boundary_topology[region_idx])} boundary triangles at level {level}")

        return contours, boundary_topology

    def extract_contours(self, level: float = 0.5) -> Dict[int, List[np.ndarray]]:
        """
        Extract contour segments per region using indicator functions at a given level.
        
        This method maintains backward compatibility by calling extract_contours_with_topology()
        and returning only the contours (not the topology information).

        Args:
            level: level-set threshold (default 0.5)
            
        Returns:
            Dict region_index -> list of segments (each segment shape (2, D))
        """
        contours, _ = self.extract_contours_with_topology(level)
        return contours

    def stitch_segments_to_polylines(self, segments: List[np.ndarray], tol: float = 1e-8) -> List[np.ndarray]:
        """
        Greedy stitching of small line segments into ordered polylines by connecting
        endpoints within a tolerance. Returns list of polylines (M_i, 2).
        """
        if not segments:
            return []

        remaining = [seg.copy() for seg in segments]
        polylines: List[np.ndarray] = []

        while remaining:
            # Start a new polyline with one segment
            poly = remaining.pop()
            start, end = poly[0], poly[1]

            extended = True
            while extended:
                extended = False
                for i in range(len(remaining)):
                    s = remaining[i]
                    s0, s1 = s[0], s[1]
                    if np.linalg.norm(end - s0) < tol:
                        # append forward
                        poly = np.vstack([poly, s1])
                        end = s1
                        remaining.pop(i)
                        extended = True
                        break
                    if np.linalg.norm(end - s1) < tol:
                        # append reversed
                        poly = np.vstack([poly, s0])
                        end = s0
                        remaining.pop(i)
                        extended = True
                        break
                    if np.linalg.norm(start - s1) < tol:
                        # prepend forward
                        poly = np.vstack([s0, poly])
                        start = s0
                        remaining.pop(i)
                        extended = True
                        break
                    if np.linalg.norm(start - s0) < tol:
                        # prepend reversed
                        poly = np.vstack([s1, poly])
                        start = s1
                        remaining.pop(i)
                        extended = True
                        break

            polylines.append(poly)

        return polylines 