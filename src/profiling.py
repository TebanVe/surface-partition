"""
Opt-in profiling state for Phase 2 IPOPT refinement.

Usage: construct a ProfilingState, pass it through the call chain via
the `profile=` keyword argument on PerimeterOptimizer.optimize().
When profile is None everywhere, there is zero overhead.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ProfilingState:
    """Accumulates timing and Steiner recomputation counts across all IPOPT callbacks."""

    n_cells: int
    n_active_vps: int
    n_triple_points: int

    _wall_s: Dict[str, float] = field(default_factory=dict, repr=False)
    _count: Dict[str, int] = field(default_factory=dict, repr=False)
    _steiner: Dict[str, int] = field(default_factory=dict, repr=False)

    ipopt_iter_count: int = 0
    total_wall_s: float = 0.0
    topology_iterations: int = 0

    # Derived fields populated by finalize()
    _finalized: bool = field(default=False, repr=False)
    _summary: Dict = field(default_factory=dict, repr=False)

    def record(self, cb: str, elapsed: float) -> None:
        """Accumulate wall-clock time and invocation count for a callback."""
        self._wall_s[cb] = self._wall_s.get(cb, 0.0) + elapsed
        self._count[cb] = self._count.get(cb, 0) + 1

    def record_steiner(self, key: str, n: int = 1) -> None:
        """Accumulate Steiner recomputation count under a callback key."""
        self._steiner[key] = self._steiner.get(key, 0) + n

    def finalize(self) -> None:
        """Compute derived metrics (means, % breakdown). Call once after optimization ends."""
        if self._finalized:
            return
        cb_total = sum(self._wall_s.values())
        summary: Dict = {"total_wall_s": self.total_wall_s}
        for cb in ("objective", "gradient", "constraints", "jacobian", "hessian"):
            w = self._wall_s.get(cb, 0.0)
            pct = (w / self.total_wall_s * 100.0) if self.total_wall_s > 0 else 0.0
            summary[f"{cb}_pct_wall"] = round(pct, 2)
        overhead = self.total_wall_s - cb_total
        summary["overhead_pct_wall"] = round(
            (overhead / self.total_wall_s * 100.0) if self.total_wall_s > 0 else 0.0, 2
        )
        summary["steiner_total_recomps"] = sum(self._steiner.values())
        self._summary = summary
        self._finalized = True

    def to_yaml_dict(self, campaign_name: str = "", method: str = "ipopt",
                     exact_hessian: bool = True, mesh_vertices: int = 0) -> dict:
        """Return a YAML-serialisable dict matching the timing_profile.yaml schema."""
        if not self._finalized:
            self.finalize()

        callbacks: Dict = {}
        for cb in ("objective", "gradient", "constraints", "jacobian", "hessian"):
            w = self._wall_s.get(cb, 0.0)
            c = self._count.get(cb, 0)
            entry: Dict = {
                "invocation_count": c,
                "total_wall_s": round(w, 6),
                "mean_wall_s": round(w / c, 8) if c > 0 else 0.0,
            }
            s = self._steiner.get(cb, 0)
            if s > 0:
                entry["steiner_recomps_total"] = s
                entry["steiner_recomps_per_call_mean"] = round(s / c, 2) if c > 0 else 0.0
            callbacks[cb] = entry

        return {
            "schema_version": "1.0",
            "campaign_name": campaign_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_metadata": {
                "n_cells": self.n_cells,
                "n_active_vps": self.n_active_vps,
                "n_triple_points": self.n_triple_points,
                "mesh_vertices": mesh_vertices,
                "method": method,
                "exact_hessian": exact_hessian,
            },
            "ipopt_summary": {
                "total_wall_s": round(self.total_wall_s, 4),
                "ipopt_iter_count": self.ipopt_iter_count,
                "topology_iterations": self.topology_iterations,
            },
            "callbacks": callbacks,
            "summary": self._summary,
        }

    def to_index_dict(self) -> dict:
        """Return a flat dict of key scalars for experiment_index.yaml embedding."""
        if not self._finalized:
            self.finalize()
        d: Dict = {
            "total_wall_s": round(self.total_wall_s, 4),
            "ipopt_iter_count": self.ipopt_iter_count,
        }
        for cb in ("objective", "gradient", "constraints", "jacobian", "hessian"):
            w = self._wall_s.get(cb, 0.0)
            c = self._count.get(cb, 0)
            d[f"{cb}_total_wall_s"] = round(w, 6)
            d[f"{cb}_mean_wall_s"] = round(w / c, 8) if c > 0 else 0.0
            pct = self._summary.get(f"{cb}_pct_wall", 0.0)
            d[f"{cb}_pct_wall"] = pct
            s = self._steiner.get(cb, 0)
            if s > 0:
                d[f"{cb}_steiner_recomps_total"] = s
                d[f"{cb}_steiner_recomps_per_call"] = round(s / c, 2) if c > 0 else 0.0
        return d


@dataclass
class RelaxationProfilingState:
    """Per-level + aggregate timing accumulator for Phase 1 PGD.

    Same zero-overhead pattern as ProfilingState: every call site is guarded
    by ``if prof is not None:``. Per-level data is held in a list of dicts so
    a single timing_profile.yaml can be written at the end of the run.

    The per-level lifecycle is split across two calls because the level wall
    clock must start before the mesh is built (so level_wall_s spans mesh
    build + matrix assembly + PGD), while the mesh-derived metadata (vertex/
    triangle counts, matrix nnz) is only known after the build:
        start_level(level)         -> reset accumulators, start the clock
        set_level_mesh_stats(...)  -> record post-build metadata
        finalize_level()           -> snapshot into _levels, fold into aggregate
    """

    n_partitions: int

    # Mutable per-level state — snapshotted by finalize_level().
    _current_level: Optional[int] = None
    _current_n_vertices: int = 0
    _current_n_triangles: int = 0
    _current_nnz_K: int = 0
    _current_nnz_M: int = 0
    _current_epsilon: float = 0.0

    _wall_s: Dict[str, float] = field(default_factory=dict, repr=False)
    _count: Dict[str, int] = field(default_factory=dict, repr=False)
    _counters: Dict[str, int] = field(default_factory=dict, repr=False)

    _level_start_t: float = 0.0

    _levels: List[Dict] = field(default_factory=list, repr=False)

    # Aggregate across levels — accumulated in finalize_level().
    _agg_wall_s: Dict[str, float] = field(default_factory=dict, repr=False)
    _agg_count: Dict[str, int] = field(default_factory=dict, repr=False)
    _agg_counters: Dict[str, int] = field(default_factory=dict, repr=False)

    # Set by run_relaxation after the level loop ends.
    total_wall_s: float = 0.0

    _finalized: bool = field(default=False, repr=False)
    _summary: Dict = field(default_factory=dict, repr=False)

    def start_level(self, level: int) -> None:
        """Reset per-level accumulators and start the level wall clock.

        Called before the mesh build so level_wall_s spans the whole level.
        """
        self._current_level = level
        self._current_n_vertices = 0
        self._current_n_triangles = 0
        self._current_nnz_K = 0
        self._current_nnz_M = 0
        self._current_epsilon = 0.0
        self._wall_s = {}
        self._count = {}
        self._counters = {}
        self._level_start_t = time.perf_counter()

    def set_level_mesh_stats(self, n_vertices: int, n_triangles: int,
                             nnz_K: int, nnz_M: int, epsilon: float) -> None:
        """Record post-build mesh metadata for the current level."""
        self._current_n_vertices = n_vertices
        self._current_n_triangles = n_triangles
        self._current_nnz_K = nnz_K
        self._current_nnz_M = nnz_M
        self._current_epsilon = epsilon

    def record(self, cb: str, elapsed: float) -> None:
        """Accumulate wall-clock time and invocation count for a callback."""
        self._wall_s[cb] = self._wall_s.get(cb, 0.0) + elapsed
        self._count[cb] = self._count.get(cb, 0) + 1

    def add_counter(self, name: str, n: int = 1) -> None:
        """Bump a named integer counter for the current level."""
        self._counters[name] = self._counters.get(name, 0) + n

    def finalize_level(self) -> None:
        """Snapshot the current level into _levels and fold it into the aggregate."""
        level_wall_s = time.perf_counter() - self._level_start_t

        callbacks: Dict = {}
        for cb, w in self._wall_s.items():
            c = self._count.get(cb, 0)
            callbacks[cb] = {
                "invocation_count": c,
                "total_wall_s": round(w, 6),
                "mean_wall_s": round(w / c, 8) if c > 0 else 0.0,
            }

        major = self._counters.get("major_iterations", 0)
        bt_total = self._counters.get("backtracks_per_iter_total", 0)
        proj_inner = self._counters.get("projection_inner_iters_total", 0)
        proj_inv = self._counters.get("projection_invocations", 0)

        self._levels.append({
            "level": self._current_level,
            "n_vertices": self._current_n_vertices,
            "n_triangles": self._current_n_triangles,
            "nnz_K": self._current_nnz_K,
            "nnz_M": self._current_nnz_M,
            "epsilon": float(self._current_epsilon),
            "level_wall_s": round(level_wall_s, 4),
            "major_iterations": major,
            "mean_backtracks_per_iter": round(bt_total / major, 4) if major > 0 else 0.0,
            "mean_projection_inner_iters": (
                round(proj_inner / proj_inv, 4) if proj_inv > 0 else 0.0
            ),
            "callbacks": callbacks,
        })

        for cb, w in self._wall_s.items():
            self._agg_wall_s[cb] = self._agg_wall_s.get(cb, 0.0) + w
            self._agg_count[cb] = self._agg_count.get(cb, 0) + self._count.get(cb, 0)
        for name, n in self._counters.items():
            self._agg_counters[name] = self._agg_counters.get(name, 0) + n

    def finalize(self) -> None:
        """Compute the aggregate summary across levels. Call once at run end."""
        if self._finalized:
            return
        total = self.total_wall_s
        w = self._agg_wall_s

        def pct(x: float) -> float:
            return round((x / total * 100.0), 2) if total > 0 else 0.0

        # energy and projection wall are nested inside backtrack; report
        # backtrack net of them so the summary percentages partition total.
        energy = w.get("energy", 0.0)
        projection = w.get("projection", 0.0)
        backtrack = w.get("backtrack", 0.0)
        backtrack_net = max(0.0, backtrack - energy - projection)

        summary: Dict = {"total_wall_s": round(total, 4)}
        summary["matrix_assembly_pct_wall"] = pct(w.get("matrix_assembly", 0.0))
        summary["energy_pct_wall"] = pct(energy)
        summary["gradient_pct_wall"] = pct(w.get("gradient", 0.0))
        summary["projection_pct_wall"] = pct(projection)
        summary["backtrack_pct_wall"] = pct(backtrack_net)
        summary["h5_save_pct_wall"] = pct(w.get("h5_save", 0.0))
        summary["h5_flush_pct_wall"] = pct(w.get("h5_flush", 0.0))
        summary["trigger_check_pct_wall"] = pct(w.get("trigger_check", 0.0))

        accounted = (
            w.get("matrix_assembly", 0.0)
            + w.get("gradient", 0.0)
            + backtrack
            + w.get("h5_save", 0.0)
            + w.get("h5_flush", 0.0)
            + w.get("trigger_check", 0.0)
        )
        summary["overhead_pct_wall"] = pct(max(0.0, total - accounted))

        major = self._agg_counters.get("major_iterations", 0)
        bt_total = self._agg_counters.get("backtracks_per_iter_total", 0)
        proj_inner = self._agg_counters.get("projection_inner_iters_total", 0)
        proj_inv = self._agg_counters.get("projection_invocations", 0)
        summary["mean_backtracks_per_iter"] = (
            round(bt_total / major, 4) if major > 0 else 0.0
        )
        summary["mean_projection_inner_iters"] = (
            round(proj_inner / proj_inv, 4) if proj_inv > 0 else 0.0
        )
        summary["major_iterations_total"] = major

        self._summary = summary
        self._finalized = True

    def to_yaml_dict(self, *, surface: str, n_partitions: int,
                     refinement_levels: int) -> dict:
        """Return a YAML-serialisable dict matching the timing_profile.yaml schema."""
        if not self._finalized:
            self.finalize()
        return {
            "schema_version": "1.0",
            "phase": "relaxation",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_metadata": {
                "surface": surface,
                "n_partitions": n_partitions,
                "refinement_levels": refinement_levels,
            },
            "summary": self._summary,
            "levels": self._levels,
            "notes": (
                "backtrack wall includes nested energy & projection wall; "
                "init_condition wall includes nested projection wall. "
                "summary.backtrack_pct_wall is reported net of the nested "
                "energy & projection time so the summary percentages "
                "partition total_wall_s."
            ),
        }

    def to_index_dict(self) -> dict:
        """Return a flat dict of key scalars for experiment_index.yaml embedding."""
        if not self._finalized:
            self.finalize()
        s = self._summary
        d: Dict = {"relax_timing_total_wall_s": s.get("total_wall_s")}
        for cb in ("matrix_assembly", "energy", "gradient", "projection",
                   "backtrack", "h5_save", "h5_flush", "trigger_check"):
            d[f"relax_timing_{cb}_pct_wall"] = s.get(f"{cb}_pct_wall")
        d["relax_timing_mean_backtracks"] = s.get("mean_backtracks_per_iter")
        d["relax_timing_mean_projection_inner_iters"] = s.get(
            "mean_projection_inner_iters")
        d["relax_timing_major_iterations_total"] = s.get("major_iterations_total")
        return d
