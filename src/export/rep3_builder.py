"""
Build Representation 3: subdivided mesh with explicit per-face cell labels.

Given the original mesh, the active variable points, and the triple-point /
Steiner geometry, this produces three flat arrays:

- ``sub_vertices`` (V + n_vp + n_tp, 3) — original mesh vertices followed by
  active-VP positions followed by Steiner points.
- ``sub_faces``    (F_sub, 3)            — connectivity into ``sub_vertices``.
- ``face_labels``  (F_sub,)              — cell label per sub-triangle.

The vertex-index layout in ``sub_vertices`` is:

    indices  0 .. V-1                          → original mesh vertices
    indices  V .. V+n_vp-1                     → active VP positions (order of
                                                 ``active_vps``)
    indices  V+n_vp .. V+n_vp+n_tp-1           → Steiner points (order of
                                                 ``steiner_handler.triple_points``)
"""

from typing import List, Tuple

import numpy as np


def build_representation_3(
    partition,
    mesh,
    active_vps: List,
    steiner_handler,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = mesh.vertices
    V = vertices.shape[0]
    n_vp = len(active_vps)
    triple_points = steiner_handler.triple_points
    n_tp = len(triple_points)

    vertex_labels = partition.indicator_functions.argmax(axis=1).astype(np.int32)
    n_cells = partition.n_cells

    sub_vertices = np.empty((V + n_vp + n_tp, 3), dtype=np.float64)
    sub_vertices[:V] = vertices
    for k, vp in enumerate(active_vps):
        sub_vertices[V + k] = vp.evaluate(vertices)
    for k, tp in enumerate(triple_points):
        sub_vertices[V + n_vp + k] = tp.steiner_point

    vp_global_idx = {id(vp): V + k for k, vp in enumerate(active_vps)}
    steiner_global_idx = {
        tp.triangle_idx: V + n_vp + k for k, tp in enumerate(triple_points)
    }
    edge_to_sub_idx = {frozenset(vp.edge): V + k for k, vp in enumerate(active_vps)}
    triple_lookup = {tp.triangle_idx: tp for tp in triple_points}

    sub_faces: List[Tuple[int, int, int]] = []
    face_labels: List[int] = []

    for tri_idx, face in enumerate(mesh.faces):
        a, b, c = int(face[0]), int(face[1]), int(face[2])
        la, lb, lc = (
            int(vertex_labels[a]),
            int(vertex_labels[b]),
            int(vertex_labels[c]),
        )
        label_set = {la, lb, lc}

        if len(label_set) == 1:
            sub_faces.append((a, b, c))
            face_labels.append(la)
            continue

        if len(label_set) == 2:
            verts = [a, b, c]
            labs = [la, lb, lc]
            for i in range(3):
                if labs.count(labs[i]) == 1:
                    lone_i = i
                    break
            v_lone = verts[lone_i]
            pair_indices = [i for i in range(3) if i != lone_i]
            v_p0 = verts[pair_indices[0]]
            v_p1 = verts[pair_indices[1]]

            edge0 = frozenset((v_lone, v_p0))
            edge1 = frozenset((v_lone, v_p1))
            vp0_idx = edge_to_sub_idx[edge0]
            vp1_idx = edge_to_sub_idx[edge1]

            label_lone = labs[lone_i]
            label_pair = labs[pair_indices[0]]

            sub_faces.append((v_lone, vp0_idx, vp1_idx))
            face_labels.append(label_lone)
            sub_faces.append((v_p0, v_p1, vp1_idx))
            face_labels.append(label_pair)
            sub_faces.append((v_p0, vp1_idx, vp0_idx))
            face_labels.append(label_pair)
            continue

        # triple-point triangle
        tp = triple_lookup[tri_idx]
        steiner_idx = steiner_global_idx[tri_idx]
        for c_k in tp.cell_indices:
            pair = [
                vp_global_idx[id(partition.variable_points[vi])]
                for vi in tp.var_point_indices
                if c_k in partition.variable_points[vi].belongs_to_cells
            ]
            if len(pair) != 2:
                raise ValueError(
                    f"Triple point at triangle {tri_idx}: cell {c_k} bounded by "
                    f"{len(pair)} VPs, expected 2"
                )
            sub_faces.append((pair[0], pair[1], steiner_idx))
            face_labels.append(c_k)

    sub_faces_arr = np.array(sub_faces, dtype=np.int32)
    face_labels_arr = np.array(face_labels, dtype=np.int32)

    assert face_labels_arr.min() >= 0 and face_labels_arr.max() < n_cells, (
        f"face_labels out of range [0, {n_cells - 1}]: "
        f"min={face_labels_arr.min()}, max={face_labels_arr.max()}"
    )

    return sub_vertices, sub_faces_arr, face_labels_arr
