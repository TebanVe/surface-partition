"""
Shared PyVista helper functions for partition visualization.

Extracted from examples/visualize_type2_triple_point.py so that
visualization scripts do not cross-import from each other.
"""

import numpy as np
from typing import Optional, List, Dict

import pyvista as pv

from ..mesh.tri_mesh import TriMesh
from ..partition.contour_partition import PartitionContour
from ..partition.area_calculator import AreaCalculator
from ..partition.steiner_handler import SteinerHandler


def _order_polygon_vertices(points: np.ndarray, mesh: TriMesh, tri_idx: int) -> np.ndarray:
    """
    Order polygon vertices counter-clockwise around triangle centroid.

    Args:
        points: (N, 3) array of polygon vertices
        mesh: TriMesh (for normal computation)
        tri_idx: Triangle index (for normal computation)

    Returns:
        Ordered vertices (N, 3)
    """
    if len(points) < 3:
        return points

    centroid = np.mean(points, axis=0)

    face = mesh.faces[tri_idx]
    v1, v2, v3 = [mesh.vertices[int(i)] for i in face]
    normal = np.cross(v2 - v1, v3 - v1)
    normal = normal / (np.linalg.norm(normal) + 1e-12)

    v0 = points[0] - centroid
    v0 = v0 / (np.linalg.norm(v0) + 1e-12)

    angles = []
    for p in points:
        v = p - centroid
        v = v / (np.linalg.norm(v) + 1e-12)
        cos_angle = np.dot(v0, v)
        sin_angle = np.dot(np.cross(v0, v), normal)
        angle = np.arctan2(sin_angle, cos_angle)
        angles.append(angle)

    sorted_indices = np.argsort(angles)
    return points[sorted_indices]


def _vertex_on_cell_side_for_viz(
    vertex_pos: np.ndarray,
    boundary_p1: np.ndarray,
    boundary_p2: np.ndarray,
    cell_idx: int,
    mesh: TriMesh,
    tri_idx: int,
    partition: PartitionContour
) -> bool:
    """
    Determine if a mesh vertex belongs to the given cell side of a boundary.

    Uses indicator_functions to check vertex cell assignment.
    """
    distances = np.linalg.norm(mesh.vertices - vertex_pos, axis=1)
    vertex_idx = np.argmin(distances)

    vertex_labels = partition.vertex_labels
    vertex_cell = int(vertex_labels[vertex_idx])

    return vertex_cell == cell_idx


def compute_cell_portion_in_triangle_simple(
    mesh: TriMesh,
    partition: PartitionContour,
    tri_idx: int,
    cell_idx: int,
    tri_idx_to_segment: Optional[Dict] = None,
    debug: bool = False
) -> Optional[np.ndarray]:
    """
    Compute polygon for cell portion of boundary triangle (simplified - no crossing cache).

    Returns:
        (N, 3) array of polygon vertices, or None if triangle doesn't contribute
    """
    face = mesh.faces[tri_idx]
    v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
    vertex_labels = partition.vertex_labels
    labels = [vertex_labels[v1], vertex_labels[v2], vertex_labels[v3]]

    if debug and tri_idx in [18150, 18152]:
        print(f"\n{'='*60}")
        print(f"DEBUG: compute_cell_portion for Triangle {tri_idx}, Cell {cell_idx}")
        print(f"{'='*60}")
        print(f"Triangle vertices: {v1}, {v2}, {v3}")
        print(f"Vertex labels: {labels}")

    is_boundary = False
    vp_positions = []
    vp_indices_in_tri = []

    if tri_idx_to_segment is not None:
        tri_seg = tri_idx_to_segment.get(tri_idx)
        if tri_seg:
            if debug and tri_idx in [18150, 18152]:
                print(f"Triangle segment found with VPs: {tri_seg.var_point_indices}")

            for vp_idx in tri_seg.var_point_indices:
                vp = partition.variable_points[vp_idx]
                if not getattr(vp, 'active', True):
                    continue

                if debug and tri_idx in [18150, 18152]:
                    print(f"  VP {vp_idx}:")
                    print(f"    edge: {vp.edge}")
                    print(f"    belongs_to_cells: {vp.belongs_to_cells}")
                    print(f"    cell {cell_idx} in belongs_to_cells? {cell_idx in vp.belongs_to_cells}")

                if cell_idx in vp.belongs_to_cells:
                    is_boundary = True
                    vp_pos = vp.evaluate(mesh.vertices)
                    vp_positions.append(vp_pos)
                    vp_indices_in_tri.append(vp_idx)

                    if debug and tri_idx in [18150, 18152]:
                        print(f"    -> INCLUDED (position: {vp_pos})")
    else:
        for tri_seg in partition.triangle_segments:
            if tri_seg.triangle_idx == tri_idx:
                for vp_idx in tri_seg.var_point_indices:
                    vp = partition.variable_points[vp_idx]
                    if not getattr(vp, 'active', True):
                        continue
                    if cell_idx in vp.belongs_to_cells:
                        is_boundary = True
                        vp_positions.append(vp.evaluate(mesh.vertices))
                        vp_indices_in_tri.append(vp_idx)
                break

    if debug and tri_idx in [18150, 18152]:
        print(f"Is boundary for cell {cell_idx}? {is_boundary}")
        print(f"VP positions collected: {len(vp_positions)} (VPs: {vp_indices_in_tri})")

    if not is_boundary:
        return None

    vertices_inside = []

    if len(vp_positions) >= 2:
        vp_pos1 = vp_positions[0]
        vp_pos2 = vp_positions[1]

        if debug and tri_idx in [18150, 18152]:
            print(f"Using VP-based vertex selection (2+ VPs)")
            print(f"Checking which triangle vertices are on cell {cell_idx} side:")

        for v in [v1, v2, v3]:
            v_pos = mesh.vertices[v]
            is_inside = _vertex_on_cell_side_for_viz(v_pos, vp_pos1, vp_pos2, cell_idx,
                                           mesh, tri_idx, partition)

            if debug and tri_idx in [18150, 18152]:
                print(f"  Vertex {v} (cell {vertex_labels[v]}): {'INSIDE' if is_inside else 'OUTSIDE'}")

            if is_inside:
                vertices_inside.append(v_pos)
    else:
        if debug and tri_idx in [18150, 18152]:
            print(f"Using indicator_functions fallback (< 2 VPs)")

        for v, lab in zip([v1, v2, v3], labels):
            if lab == cell_idx:
                vertices_inside.append(mesh.vertices[v])

                if debug and tri_idx in [18150, 18152]:
                    print(f"  Vertex {v} included (label matches cell {cell_idx})")

    all_points = vertices_inside + vp_positions

    if debug and tri_idx in [18150, 18152]:
        print(f"Final polygon construction:")
        print(f"  Vertices inside: {len(vertices_inside)}")
        print(f"  VP positions: {len(vp_positions)}")
        print(f"  Total points: {len(all_points)}")

        if len(all_points) < 3:
            print(f"  RETURNING None - not enough points for polygon!")
        else:
            print(f"  Returning polygon with {len(all_points)} vertices")

    if len(all_points) < 3:
        return None

    return _order_polygon_vertices(np.array(all_points), mesh, tri_idx)


def compute_triple_point_cell_portion(
    mesh: TriMesh,
    partition: PartitionContour,
    steiner_handler: SteinerHandler,
    tri_idx: int,
    cell_idx: int
) -> Optional[List[np.ndarray]]:
    """
    Compute the portion of a triple point triangle belonging to cell_idx.

    Returns BOTH void interior AND corner region.

    Returns:
        List of polygons (numpy arrays) or None if not found:
        - [0]: Void interior wedge (VP1, VP2, steiner)
        - [1]: Corner triangle (mesh_vertex, VP1, VP2) [if exists]
    """
    triple_point = None
    for tp in steiner_handler.triple_points:
        if tp.triangle_idx == tri_idx:
            triple_point = tp
            break

    if not triple_point or cell_idx not in triple_point.cell_indices:
        return None

    vp_positions_for_steiner = [partition.evaluate_variable_point(vi) for vi in triple_point.var_point_indices]
    steiner_pos = triple_point.compute_steiner_point(vp_positions=vp_positions_for_steiner)

    if cell_idx not in triple_point.cell_to_varpoint_pair:
        return None

    vp_idx1, vp_idx2 = triple_point.cell_to_varpoint_pair[cell_idx]
    vp1 = partition.variable_points[vp_idx1]
    vp2 = partition.variable_points[vp_idx2]
    vp1_pos = vp1.evaluate(mesh.vertices)
    vp2_pos = vp2.evaluate(mesh.vertices)

    polygons = []

    void_wedge = np.array([steiner_pos, vp2_pos, vp1_pos])
    polygons.append(void_wedge)

    if cell_idx in triple_point.cell_to_mesh_vertex:
        mesh_vertex_idx = triple_point.cell_to_mesh_vertex[cell_idx]
        mesh_vertex_pos = mesh.vertices[mesh_vertex_idx]
        corner_triangle = np.array([mesh_vertex_pos, vp2_pos, vp1_pos])
        polygons.append(corner_triangle)

    return polygons


def render_single_region_simple(
    plotter: pv.Plotter,
    mesh: TriMesh,
    partition: PartitionContour,
    area_calc: AreaCalculator,
    steiner_handler: SteinerHandler,
    cell_idx: int,
    color: str,
    opacity: float = 1.0,
    backface_culling: bool = False,
):
    """
    Render ONE region with precise boundaries (simplified for vertex-collapse).

    Args:
        backface_culling: If True, hide back-facing triangles (default: False to match Type 1)
    """
    tri_idx_to_segment = {}
    for tri_seg in partition.triangle_segments:
        tri_idx_to_segment[tri_seg.triangle_idx] = tri_seg

    all_vertices = []
    all_faces = []
    vertex_offset = 0

    interior_tris = area_calc.cell_interior_triangles.get(cell_idx, [])
    for tri_idx in interior_tris:
        face = mesh.faces[tri_idx]
        vertices = mesh.vertices[[int(face[0]), int(face[1]), int(face[2])]]
        all_vertices.append(vertices)
        all_faces.extend([3, vertex_offset, vertex_offset+1, vertex_offset+2])
        vertex_offset += 3

    boundary_tris = area_calc.cell_boundary_triangles.get(cell_idx, [])
    for tri_idx in boundary_tris:
        debug_enabled = tri_idx in [18150, 18152]

        poly_vertices = compute_cell_portion_in_triangle_simple(
            mesh, partition, tri_idx, cell_idx, tri_idx_to_segment, debug=debug_enabled
        )
        if poly_vertices is not None:
            n_verts = len(poly_vertices)
            all_vertices.append(poly_vertices)
            face_indices = [n_verts] + list(range(vertex_offset, vertex_offset + n_verts))
            all_faces.extend(face_indices)
            vertex_offset += n_verts

    for tp in steiner_handler.triple_points:
        if cell_idx in tp.cell_indices:
            polygons = compute_triple_point_cell_portion(
                mesh, partition, steiner_handler, tp.triangle_idx, cell_idx
            )
            if polygons is not None:
                for poly_vertices in polygons:
                    n_verts = len(poly_vertices)
                    all_vertices.append(poly_vertices)
                    face_indices = [n_verts] + list(range(vertex_offset, vertex_offset + n_verts))
                    all_faces.extend(face_indices)
                    vertex_offset += n_verts

    if all_vertices:
        all_vertices = np.vstack(all_vertices)
        region_mesh = pv.PolyData(all_vertices, faces=all_faces)
        plotter.add_mesh(region_mesh, color=color, opacity=opacity,
                        show_edges=True, edge_color='lightgray', line_width=0.5,
                        backface_culling=backface_culling)


def add_steiner_visualization(
    plotter: pv.Plotter,
    steiner_handler: SteinerHandler,
    partition: PartitionContour,
    mesh: TriMesh,
    steiner_size: float = 0.000005
):
    """Add Steiner points and void triangles to visualization."""
    triple_points = steiner_handler.triple_points

    if not triple_points:
        return

    print(f"  Adding {len(triple_points)} Steiner points and void triangles...")

    for tp in triple_points:
        vp_positions_for_steiner = [partition.evaluate_variable_point(vi) for vi in tp.var_point_indices]
        steiner_pt = tp.compute_steiner_point(vp_positions=vp_positions_for_steiner)

        sphere = pv.Sphere(radius=steiner_size, center=steiner_pt)
        plotter.add_mesh(sphere, color='orange', opacity=1.0)

        vp_positions = []
        for vp_idx in tp.var_point_indices:
            vp = partition.variable_points[vp_idx]
            if not getattr(vp, 'active', True):
                continue
            vp_pos = vp.evaluate(mesh.vertices)
            vp_positions.append(vp_pos)

        if len(vp_positions) == 3:
            for i in range(3):
                p1 = vp_positions[i]
                p2 = vp_positions[(i + 1) % 3]
                line = pv.Line(p1, p2)
                plotter.add_mesh(line, color='cyan', line_width=2, opacity=0.7)


def add_vp_visualization(
    plotter: pv.Plotter,
    partition: PartitionContour,
    mesh: TriMesh,
    vp_indices: List[int],
    vp_colors: List[str],
    vp_labels: List[str],
    vp_size: float = 0.0005
):
    """Add VP spheres with labels to visualization."""
    for vp_idx, color, label in zip(vp_indices, vp_colors, vp_labels):
        vp = partition.variable_points[vp_idx]
        vp_pos = vp.evaluate(mesh.vertices)

        sphere = pv.Sphere(radius=vp_size, center=vp_pos)
        plotter.add_mesh(sphere, color=color, opacity=1.0)

        if label:
            label_pos = vp_pos + np.array([0, 0, vp_size * 3])
            plotter.add_point_labels(
                [label_pos], [label],
                font_size=9, text_color='black',
                shape_color='white', shape_opacity=0.7,
                always_visible=True, point_size=8
            )
