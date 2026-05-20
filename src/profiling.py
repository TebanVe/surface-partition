"""
Opt-in profiling state for Phase 2 IPOPT refinement.

Usage: construct a ProfilingState, pass it through the call chain via
the `profile=` keyword argument on PerimeterOptimizer.optimize().
When profile is None everywhere, there is zero overhead.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional


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
