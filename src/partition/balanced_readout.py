"""Balanced readout: equal-area, connected extraction from a relaxation solution.

Phase 1 enforces equal *continuous* area (``integral u_k dA = A/N``) but the
partition handed downstream is the *discrete* winner-take-all argmax territory,
which nothing in the energy or constraints references. At high N the two diverge:
a cell can hold its full target mass diffusely while its argmax territory is far
below target (a "runt"), or is broken into disconnected islands (a "split"). See
``docs/reference/winner_take_all_partition_gap.md``.

This module closes that gap at extraction time, in two stages:

1. **Dual shifts** -- replace ``argmax_k u_ik`` with ``argmax_k [log u_ik + psi_k]``,
   where the N per-cell offsets ``psi`` solve the semi-discrete optimal-transport
   dual so every cell's discrete territory hits the equal-area target. Raising
   ``psi_k`` grows cell k's region monotonically outward from its existing core,
   so area is corrected *locally* -- unlike the discrete-area trim, which pushed
   the correction through the (nonlocal) projection and manufactured islands.
   Balanced up to one-vertex granularity at any N.

2. **Connectivity repair** -- the dual shifts fix area but not connectivity. Stray
   components are absorbed by the neighbour sharing the longest boundary, then
   equal areas are restored by moving single boundary vertices from richer cells
   to poorer ones, rejecting any move that would disconnect the donor. Both
   invariants (every cell connected, areas within the gate) hold by construction
   on exit.

The repair reports its own strain (territory moved, perimeter cost, moves blocked
by the connectivity check) so that a run which is beyond what repair can fix says
so rather than silently returning a degraded partition.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..logging_config import get_logger
from .find_contours import (
    DISCONNECTED_FRAGMENT_REL_THRESHOLD,
    detect_area_imbalance,
    detect_disconnected_cells,
)

logger = get_logger(__name__)

# Densities are clipped to this floor before the log so that exactly-zero entries
# (the box constraint u >= 0 is active on ~6% of entries) give a finite, very
# negative score rather than -inf, which would poison the argmax arithmetic.
LOG_DENSITY_FLOOR = 1e-300


@dataclass
class BalancedReadoutConfig:
    """Parameters of the balanced-readout stage."""

    gate_threshold: float = 0.05
    dual_iters: int = 400
    dual_eta0: float = 0.5
    dual_decay: float = 0.02
    repair_enabled: bool = True
    max_repair_sweeps: int = 50
    fragment_rel_threshold: float = DISCONNECTED_FRAGMENT_REL_THRESHOLD

    def to_dict(self) -> dict:
        return {
            "gate_threshold": float(self.gate_threshold),
            "dual_iters": int(self.dual_iters),
            "dual_eta0": float(self.dual_eta0),
            "dual_decay": float(self.dual_decay),
            "repair_enabled": bool(self.repair_enabled),
            "max_repair_sweeps": int(self.max_repair_sweeps),
            "fragment_rel_threshold": float(self.fragment_rel_threshold),
        }


def build_vertex_adjacency(faces: np.ndarray, n_vertices: int) -> List[np.ndarray]:
    """Vertex-to-vertex adjacency from triangle sides.

    Uses the mesh's own edge graph, so it respects the true surface topology
    (e.g. torus periodic wrap encoded in ``faces``), not a flat parametrization.
    """
    F = np.asarray(faces)
    edges = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    edges = np.vstack([edges, edges[:, ::-1]])
    order = np.argsort(edges[:, 0], kind="stable")
    edges = edges[order]
    starts = np.searchsorted(edges[:, 0], np.arange(n_vertices))
    ends = np.searchsorted(edges[:, 0], np.arange(n_vertices), side="right")
    return [np.unique(edges[s:e, 1]) for s, e in zip(starts, ends)]


def label_boundary_length(
    labels: np.ndarray, vertices: np.ndarray, faces: np.ndarray
) -> float:
    """Length of the label boundary, cutting each mixed triangle at edge midpoints.

    Depends only on the labeling, so it is directly comparable across readouts.
    A triangle carrying two distinct labels contributes the segment joining the
    midpoints of its two bi-labeled edges; one carrying three contributes three
    segments meeting at the centroid. This is a label-space proxy for perimeter,
    not the Phase 2 contour perimeter.
    """
    L = labels[faces]
    mixed = ~((L[:, 0] == L[:, 1]) & (L[:, 1] == L[:, 2]))
    F = faces[mixed]
    if F.shape[0] == 0:
        return 0.0
    P = vertices[F]
    Lm = labels[F]
    mid = np.stack(
        [
            (P[:, 0] + P[:, 1]) / 2,
            (P[:, 1] + P[:, 2]) / 2,
            (P[:, 2] + P[:, 0]) / 2,
        ],
        axis=1,
    )
    cut = np.stack(
        [Lm[:, 0] != Lm[:, 1], Lm[:, 1] != Lm[:, 2], Lm[:, 2] != Lm[:, 0]], axis=1
    )
    ncut = cut.sum(axis=1)
    total = 0.0

    two = ncut == 2
    if two.any():
        m, ct = mid[two], cut[two]
        idx = np.argsort(~ct, axis=1)[:, :2]
        p0 = np.take_along_axis(m, idx[:, 0:1, None], axis=1)[:, 0]
        p1 = np.take_along_axis(m, idx[:, 1:2, None], axis=1)[:, 0]
        total += float(np.linalg.norm(p0 - p1, axis=1).sum())

    three = ncut == 3
    if three.any():
        m = mid[three]
        total += float(np.linalg.norm(m - m.mean(axis=1, keepdims=True), axis=2).sum())
    return total


def solve_dual_offsets(
    log_densities: np.ndarray,
    lumped_mass: np.ndarray,
    target: float,
    config: BalancedReadoutConfig,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Per-cell offsets psi enforcing equal discrete area under a shifted argmax.

    Subgradient ascent on the concave dual of the assignment problem:
    ``psi_k -= eta * (T_k - target)/target``, with a decaying step. Vertices are
    indivisible, so exact equality is unattainable; the granularity floor is one
    vertex mass (max_i v_i / target). Returns the best iterate visited.
    """
    N = log_densities.shape[1]
    psi = np.zeros(N)
    best_worst = np.inf
    best_psi = psi.copy()
    best_labels = np.argmax(log_densities, axis=1)

    for it in range(config.dual_iters):
        labels = np.argmax(log_densities + psi, axis=1)
        areas = np.zeros(N)
        np.add.at(areas, labels, lumped_mass)
        rel = (areas - target) / target
        worst = float(np.abs(rel).max())
        if worst < best_worst:
            best_worst, best_psi, best_labels = worst, psi.copy(), labels.copy()
        if it % 50 == 0:
            logger.debug(
                "dual iter %4d: worst |dev| = %.3f%%, n over gate = %d",
                it,
                worst * 100,
                int((np.abs(rel) > config.gate_threshold).sum()),
            )
        psi -= (config.dual_eta0 / (1.0 + config.dual_decay * it)) * rel

    return best_psi, best_labels, best_worst


def _components_of_cell(
    cell_vertices: Set[int], adjacency: Sequence[np.ndarray]
) -> List[Set[int]]:
    """Connected components of one cell's territory, on the mesh edge graph."""
    remaining = set(cell_vertices)
    components: List[Set[int]] = []
    while remaining:
        start = next(iter(remaining))
        comp = {start}
        stack = [start]
        remaining.discard(start)
        while stack:
            x = stack.pop()
            for y in adjacency[x]:
                if y in remaining:
                    remaining.discard(y)
                    comp.add(int(y))
                    stack.append(int(y))
        components.append(comp)
    return components


def _stays_connected(
    cell_vertices: Set[int], adjacency: Sequence[np.ndarray], removed: int
) -> bool:
    """True if the cell's territory is still connected after dropping ``removed``.

    Exact articulation test, cheap because it is local to one cell (a few hundred
    vertices at the region counts of interest).
    """
    remaining = cell_vertices - {removed}
    if not remaining:
        return False
    start = next(iter(remaining))
    seen = {start}
    stack = [start]
    while stack:
        x = stack.pop()
        for y in adjacency[x]:
            yi = int(y)
            if yi in remaining and yi not in seen:
                seen.add(yi)
                stack.append(yi)
    return len(seen) == len(remaining)


def reassign_islands(
    labels: np.ndarray,
    adjacency: Sequence[np.ndarray],
    lumped_mass: np.ndarray,
    vertices: np.ndarray,
    n_partitions: int,
) -> Tuple[np.ndarray, dict]:
    """Absorb every stray component into the neighbour sharing the most boundary.

    A component adjacent to cell c can always be merged into c without breaking
    c's connectivity, so this stage is unconditionally safe and terminates with
    every cell connected (or empty, which is reported).
    """
    labels = labels.copy()
    cell_verts: List[Set[int]] = [set() for _ in range(n_partitions)]
    for i, k in enumerate(labels):
        cell_verts[int(k)].add(i)

    moved_area = 0.0
    n_components_absorbed = 0
    per_cell: List[dict] = []

    for k in range(n_partitions):
        comps = _components_of_cell(cell_verts[k], adjacency)
        if len(comps) <= 1:
            continue
        comps.sort(key=lambda c: -sum(lumped_mass[i] for i in c))
        main = comps[0]
        for stray in comps[1:]:
            # Longest shared boundary, measured by summed edge length.
            share: Dict[int, float] = {}
            for i in stray:
                for j in adjacency[i]:
                    jj = int(j)
                    other = int(labels[jj])
                    if other == k:
                        continue
                    length = float(np.linalg.norm(vertices[i] - vertices[jj]))
                    share[other] = share.get(other, 0.0) + length
            if not share:
                continue  # fully enclosed by its own cell: not reachable in practice
            receiver = max(share.items(), key=lambda kv: kv[1])[0]
            area = float(sum(lumped_mass[i] for i in stray))
            for i in stray:
                labels[i] = receiver
                cell_verts[k].discard(i)
                cell_verts[receiver].add(i)
            moved_area += area
            n_components_absorbed += 1
            per_cell.append(
                {
                    "cell": int(k),
                    "receiver": int(receiver),
                    "area": area,
                    "n_vertices": int(len(stray)),
                }
            )
        cell_verts[k] = main

    report = {
        "n_components_absorbed": int(n_components_absorbed),
        "area_absorbed": float(moved_area),
        "transfers": per_cell,
        "empty_cells": [k for k in range(n_partitions) if not cell_verts[k]],
    }
    return labels, report


def rebalance_boundaries(
    labels: np.ndarray,
    log_densities: np.ndarray,
    adjacency: Sequence[np.ndarray],
    lumped_mass: np.ndarray,
    target: float,
    n_partitions: int,
    config: BalancedReadoutConfig,
) -> Tuple[np.ndarray, dict]:
    """Restore equal areas by connectivity-preserving single-vertex transfers.

    Moving vertex i (mass v_i) from donor d to receiver r changes the imbalance
    objective ``sum_k (T_k - target)^2`` by ``2 v_i [(T_r - T_d) + v_i]``, so a
    move is improving exactly when ``T_d - T_r > v_i``. Candidates are ranked by
    ``log u_ir - log u_id`` -- how strongly the vertex already prefers the
    receiver -- so the cheapest boundary vertices move first. Receiver
    connectivity is automatic (i is adjacent to r); donor connectivity is checked
    exactly and the move is skipped if it would disconnect the donor.
    """
    labels = labels.copy()
    cell_verts: List[Set[int]] = [set() for _ in range(n_partitions)]
    for i, k in enumerate(labels):
        cell_verts[int(k)].add(i)
    areas = np.zeros(n_partitions)
    np.add.at(areas, labels, lumped_mass)

    n_moves = 0
    n_blocked = 0
    area_moved = 0.0
    sweeps = 0

    for sweep in range(config.max_repair_sweeps):
        sweeps = sweep + 1
        candidates = []
        for i in range(len(labels)):
            d = int(labels[i])
            neigh = {int(labels[j]) for j in adjacency[i]}
            neigh.discard(d)
            for r in neigh:
                if areas[d] - areas[r] > lumped_mass[i]:
                    gain = log_densities[i, r] - log_densities[i, d]
                    candidates.append((gain, i, d, r))
        if not candidates:
            break

        candidates.sort(key=lambda t: -t[0])
        applied = 0
        for _, i, d, r in candidates:
            if int(labels[i]) != d:
                continue  # stale after an earlier move this sweep
            if areas[d] - areas[r] <= lumped_mass[i]:
                continue
            if not any(int(labels[j]) == r for j in adjacency[i]):
                continue
            if not _stays_connected(cell_verts[d], adjacency, i):
                n_blocked += 1
                continue
            labels[i] = r
            cell_verts[d].discard(i)
            cell_verts[r].add(i)
            areas[d] -= lumped_mass[i]
            areas[r] += lumped_mass[i]
            area_moved += float(lumped_mass[i])
            applied += 1
            n_moves += 1
        if applied == 0:
            break

    rel = np.abs(areas - target) / target
    report = {
        "n_moves": int(n_moves),
        "n_blocked_by_connectivity": int(n_blocked),
        "area_moved": float(area_moved),
        "area_moved_rel_target": float(area_moved / target) if target > 0 else 0.0,
        "sweeps_used": int(sweeps),
        "hit_sweep_cap": bool(sweeps >= config.max_repair_sweeps),
        "worst_rel_dev_after": float(rel.max()),
        "n_over_gate_after": int((rel > config.gate_threshold).sum()),
    }
    return labels, report


def densities_matching_labels(densities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return densities whose per-vertex argmax equals ``labels``.

    At every vertex whose argmax disagrees with the target label, the winner's
    value and the target cell's value are *swapped*. The swap is exact: the row
    sum (and hence the partition-of-unity constraint) is untouched, the target
    cell inherits the strict row maximum, and the operation is invertible given
    the original labels. Downstream consumers read the densities only through
    ``argmax`` (``ContourAnalyzer.compute_indicator_functions`` builds a hard 0/1
    indicator), so this is a faithful carrier of the readout.
    """
    out = np.array(densities, dtype=float, copy=True)
    winners = np.argmax(out, axis=1)
    mismatch = np.where(winners != labels)[0]
    if mismatch.size:
        w = winners[mismatch]
        t = labels[mismatch]
        wv = out[mismatch, w].copy()
        tv = out[mismatch, t].copy()
        out[mismatch, w] = tv
        out[mismatch, t] = wv
    return out


def apply_balanced_readout(
    densities: np.ndarray,
    faces: np.ndarray,
    vertices: np.ndarray,
    lumped_mass: np.ndarray,
    n_partitions: int,
    config: Optional[BalancedReadoutConfig] = None,
) -> dict:
    """Run the full balanced readout: dual shifts, then connectivity repair.

    Returns a dict with the final labels, the offsets psi, a density field whose
    argmax reproduces those labels, and the three validity gates plus the
    label-boundary length at each stage (source / shifted / repaired).
    """
    config = config or BalancedReadoutConfig()
    n_vertices = densities.shape[0]
    target = float(lumped_mass.sum()) / n_partitions
    adjacency = build_vertex_adjacency(faces, n_vertices)

    log_densities = np.log(np.maximum(densities, LOG_DENSITY_FLOOR))
    source_labels = np.argmax(densities, axis=1)

    stages: Dict[str, dict] = {}

    def gate(name: str, dens: np.ndarray, labels: np.ndarray) -> None:
        stages[name] = {
            "area_imbalance": detect_area_imbalance(dens, lumped_mass, n_partitions),
            "disconnected_cells": detect_disconnected_cells(
                dens,
                faces,
                lumped_mass,
                fragment_rel_threshold=config.fragment_rel_threshold,
            ),
            "label_boundary_length": label_boundary_length(labels, vertices, faces),
        }

    gate("source", densities, source_labels)
    logger.info(
        "source: %d imbalanced (worst %.2f%%), %d fragmented",
        stages["source"]["area_imbalance"]["n_imbalanced"],
        stages["source"]["area_imbalance"]["worst_rel_dev"] * 100,
        stages["source"]["disconnected_cells"]["n_fragmented"],
    )

    psi, shifted_labels, dual_worst = solve_dual_offsets(
        log_densities, lumped_mass, target, config
    )
    shifted_densities = densities_matching_labels(densities, shifted_labels)
    gate("shifted", shifted_densities, shifted_labels)
    logger.info(
        "after dual shifts: %d imbalanced (worst %.2f%%), %d fragmented",
        stages["shifted"]["area_imbalance"]["n_imbalanced"],
        stages["shifted"]["area_imbalance"]["worst_rel_dev"] * 100,
        stages["shifted"]["disconnected_cells"]["n_fragmented"],
    )

    island_report: dict = {}
    rebalance_report: dict = {}
    final_labels = shifted_labels
    if config.repair_enabled:
        final_labels, island_report = reassign_islands(
            shifted_labels, adjacency, lumped_mass, vertices, n_partitions
        )
        logger.info(
            "islands absorbed: %d (area %.4f = %.1f%% of target)",
            island_report["n_components_absorbed"],
            island_report["area_absorbed"],
            island_report["area_absorbed"] / target * 100,
        )
        final_labels, rebalance_report = rebalance_boundaries(
            final_labels,
            log_densities,
            adjacency,
            lumped_mass,
            target,
            n_partitions,
            config,
        )
        logger.info(
            "rebalance: %d moves, %d blocked, worst |dev| %.2f%%",
            rebalance_report["n_moves"],
            rebalance_report["n_blocked_by_connectivity"],
            rebalance_report["worst_rel_dev_after"] * 100,
        )

    final_densities = densities_matching_labels(densities, final_labels)
    gate("repaired", final_densities, final_labels)
    logger.info(
        "final: %d imbalanced (worst %.2f%%), %d fragmented",
        stages["repaired"]["area_imbalance"]["n_imbalanced"],
        stages["repaired"]["area_imbalance"]["worst_rel_dev"] * 100,
        stages["repaired"]["disconnected_cells"]["n_fragmented"],
    )

    return {
        "labels": final_labels,
        "source_labels": source_labels,
        "psi": psi,
        "densities": final_densities,
        "target_area": target,
        "dual_worst_rel_dev": float(dual_worst),
        "vertex_granularity_rel": float(lumped_mass.max() / target),
        "n_relabeled_vs_source": int((final_labels != source_labels).sum()),
        "stages": stages,
        "island_report": island_report,
        "rebalance_report": rebalance_report,
        "config": config.to_dict(),
    }
