"""Neighbour-distinct cell coloring for partition visualization.

Assigns a color to every partition cell such that no two topologically
adjacent cells (cells that share a boundary) ever receive the same color, so
neighbouring regions are always visually distinguishable. Scales to large
cell counts.

Color source: ``colorcet`` glasbey (perceptually-maximised categorical
colors) when installed, otherwise a dependency-free golden-angle HSV palette.
Both produce mutually-distinct, fully-saturated colors (not pastels).
"""

import colorsys
from typing import List, Optional

import numpy as np

# Conjugate golden ratio: stepping the hue by this fraction of the circle each
# time keeps successive colors far apart (≈137.5°), so even nearby palette
# indices are easy to tell apart.
_GOLDEN_RATIO_CONJUGATE = 0.6180339887498949


def _hsv_palette(n: int) -> List[str]:
    """Generate ``n`` distinct hex colors via golden-angle hue rotation.

    Saturation and value are cycled through a few tiers so that colors which
    happen to land on similar hues still separate by brightness/intensity.
    """
    sat_tiers = (0.90, 0.65, 1.00)
    val_tiers = (0.95, 0.78, 0.60)
    colors: List[str] = []
    for i in range(n):
        h = (i * _GOLDEN_RATIO_CONJUGATE) % 1.0
        s = sat_tiers[i % len(sat_tiers)]
        v = val_tiers[(i // len(sat_tiers)) % len(val_tiers)]
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append(f"#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}")
    return colors


def distinct_palette(n: int) -> List[str]:
    """Return ``n`` mutually-distinct hex colors.

    Uses ``colorcet`` glasbey (up to 256 perceptually-maximised colors) when
    available; otherwise falls back to the golden-angle HSV palette, which can
    produce any number of colors.
    """
    if n <= 0:
        return []
    try:
        import colorcet
        glasbey = colorcet.glasbey
        if n <= len(glasbey):
            return list(glasbey[:n])
    except Exception:
        pass
    return _hsv_palette(n)


def build_cell_adjacency(vertex_labels: np.ndarray, faces: np.ndarray,
                         n_cells: int) -> List[set]:
    """Build the cell-adjacency graph from per-vertex winner labels.

    Two cells are adjacent when they meet on a mesh face (their argmax labels
    co-occur on a triangle). Vectorized: O(n_faces).

    Returns a list of ``n_cells`` sets; entry ``k`` holds the cells adjacent
    to cell ``k``.
    """
    labels = np.asarray(vertex_labels)
    face_labels = labels[np.asarray(faces)]  # (T, 3)

    edges = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        e = np.stack([face_labels[:, a], face_labels[:, b]], axis=1)
        e = e[e[:, 0] != e[:, 1]]
        if e.size:
            edges.append(e)

    adjacency: List[set] = [set() for _ in range(n_cells)]
    if not edges:
        return adjacency

    pairs = np.vstack(edges)
    pairs.sort(axis=1)                  # canonical (min, max) ordering
    pairs = np.unique(pairs, axis=0)    # collapse to distinct neighbour pairs
    for x, y in pairs:
        xi, yi = int(x), int(y)
        adjacency[xi].add(yi)
        adjacency[yi].add(xi)
    return adjacency


def assign_cell_colors(adjacency: List[set], n_cells: int,
                       palette: Optional[List[str]] = None) -> List[str]:
    """Color cells so no two adjacent cells share a color.

    Greedy graph coloring in Welsh-Powell order (highest-degree first),
    choosing the least-used permissible color at each step. With a palette of
    at least ``n_cells`` colors this gives every cell a distinct color (maximum
    variety); when the palette is smaller than ``n_cells`` it reuses colors but
    never on adjacent cells, as long as the palette has more colors than the
    largest cell degree.

    Returns a list of ``n_cells`` hex color strings.
    """
    if palette is None:
        palette = distinct_palette(n_cells)
    n_colors = len(palette)
    if n_colors == 0:
        return ['#808080'] * n_cells

    color_idx = [-1] * n_cells
    usage = [0] * n_colors
    order = sorted(range(n_cells), key=lambda c: (-len(adjacency[c]), c))

    for c in order:
        forbidden = {color_idx[nb] for nb in adjacency[c] if color_idx[nb] >= 0}
        best = None
        for k in range(n_colors):
            if k in forbidden:
                continue
            if best is None or usage[k] < usage[best]:
                best = k
        if best is None:
            # Palette smaller than (max degree + 1): cannot avoid a clash.
            # Pick the globally least-used color so reuse stays balanced.
            best = min(range(n_colors), key=lambda k: usage[k])
        color_idx[c] = best
        usage[best] += 1

    return [palette[color_idx[c]] for c in range(n_cells)]


def cell_colors_for_partition(vertex_labels: np.ndarray, faces: np.ndarray,
                              n_cells: int) -> List[str]:
    """Convenience: build adjacency and assign neighbour-distinct colors."""
    adjacency = build_cell_adjacency(vertex_labels, faces, n_cells)
    return assign_cell_colors(adjacency, n_cells)
