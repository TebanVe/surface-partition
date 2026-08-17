"""Phase 0b — the single evaluation harness for Phase 1 replacement arms.

Every arm (B: auction-dynamics MBO, C: capacity-constrained geodesic Lloyd) and
the PGD control are scored through *this* code and nothing else. That is the
whole point: two arms measured by two harnesses produce two numbers that cannot
be compared, and this project's recurring failure mode is a measurement path
quietly shaping the conclusion drawn from it.

See ``docs/plans/PHASE1_BC_REPLACEMENT_PLAN.md`` §0b for the pinned decisions
this module implements. The ones that live in code rather than prose:

* **Phase 2 settings are pinned by reusing a campaign's entire ``refinement.yaml``**,
  never by naming a few fields. The N=100 control ran with ``exact_hessian: true``
  and ``lbfgs_memory: 30``; naming only method/tolerance/iterations would silently
  substitute L-BFGS — a *different optimizer* — and the shakedown could not then
  reproduce the control's own perimeter.
* **An arm's labels enter as one-hot densities.** Downstream code reads densities
  only through ``argmax`` (``find_contours.compute_indicator_functions``), so a
  hard labelling is the honest representation and costs the arm nothing: the PGD
  control is hardened the same way, and every variable point starts at λ=0.5
  regardless. Neither side gets a sub-vertex head start.
* **Censoring is reported, not scored.** A run whose best perimeter is its final
  iterate was still improving when the iteration cap stopped it. For an arm that
  is a censored result; for a baseline it makes the number a bound, not a value.
"""
from __future__ import annotations

import glob
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

from ..mesh.tri_mesh import TriMesh
from .find_contours import (
    detect_area_imbalance,
    detect_disconnected_cells,
    detect_dormant_cells,
)

__all__ = [
    "GateReport",
    "Phase2Report",
    "labels_to_one_hot",
    "write_arm_solution",
    "run_gates",
    "vertex_granularity",
    "run_phase2",
    "scan_campaign",
]


@dataclass
class GateReport:
    """All three Phase 1 validity gates on one labelling."""

    n_dead: int
    n_weak: int
    min_peak_density: float
    n_imbalanced: int
    worst_rel_dev: float
    n_fragmented: int
    fragmented: List[int]
    worst_stray_rel: float
    vertex_granularity_rel: float

    def to_dict(self) -> dict:
        return {
            "dormant": {
                "n_dead": int(self.n_dead),
                "n_weak": int(self.n_weak),
                "min_peak_density": float(self.min_peak_density),
                # Vacuous for any balanced-assignment arm: one-hot labels give
                # peak density 1.0 by construction. Reported, never scored.
                "vacuous_for_arms": True,
            },
            "area_imbalance": {
                "n_imbalanced": int(self.n_imbalanced),
                "worst_rel_dev": float(self.worst_rel_dev),
                "vertex_granularity_rel": float(self.vertex_granularity_rel),
                # NOT vacuous: it is the assignment-solver failure detector.
                # worst_rel_dev >> granularity means the arm's balanced
                # assignment did not converge -- a harness fault to fix before
                # the arm is scored, not evidence against the arm.
                "harness_fault_threshold": float(2.0 * self.vertex_granularity_rel),
                "suspect_solver_failure": bool(
                    self.worst_rel_dev > 2.0 * self.vertex_granularity_rel
                ),
            },
            "connectivity": {
                "n_fragmented": int(self.n_fragmented),
                "fragmented": [int(c) for c in self.fragmented],
                "worst_stray_rel": float(self.worst_stray_rel),
            },
        }


@dataclass
class Phase2Report:
    """Outcome of one refinement campaign."""

    campaign_dir: str
    n_iterations: int
    initial_perimeter: Optional[float]
    best_perimeter: float
    best_iteration: int
    censored: bool
    trajectory: List[Tuple[int, float]] = field(default_factory=list)
    wall_seconds: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "campaign_dir": self.campaign_dir,
            "n_iterations": int(self.n_iterations),
            "initial_perimeter": (
                None if self.initial_perimeter is None
                else float(self.initial_perimeter)
            ),
            "best_perimeter": float(self.best_perimeter),
            "best_iteration": int(self.best_iteration),
            # True => the best iterate IS the last one, so the run was still
            # improving when the cap stopped it. Report, do not score.
            "censored": bool(self.censored),
            "trajectory": [[int(i), float(p)] for i, p in self.trajectory],
            "wall_seconds": (
                None if self.wall_seconds is None else float(self.wall_seconds)
            ),
        }


def labels_to_one_hot(labels: np.ndarray, n_partitions: int) -> np.ndarray:
    """(V,) integer labels -> (V, N) one-hot densities."""
    labels = np.asarray(labels).astype(int).ravel()
    if labels.size and (labels.min() < 0 or labels.max() >= n_partitions):
        raise ValueError(
            f"labels out of range [0, {n_partitions}): "
            f"min={labels.min()}, max={labels.max()}"
        )
    dens = np.zeros((labels.size, n_partitions), dtype=np.float64)
    dens[np.arange(labels.size), labels] = 1.0
    return dens


def write_arm_solution(
    path: str,
    labels: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    n_partitions: int,
    extra_attrs: Optional[Dict] = None,
) -> str:
    """Write an arm's labelling in the Phase 1 solution schema.

    Emitting the Phase 1 schema is what lets every downstream consumer --
    ``refine_perimeter.py``, ``visualize_partition_fast.py``,
    ``export_partition.py``, ``testing/check_fragmentation.py`` -- read an arm's
    output with no new flags and no special-casing.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    dens = labels_to_one_hot(labels, n_partitions)
    tmp = path + ".tmp"
    with h5py.File(tmp, "w") as f:
        f.create_dataset("x_opt", data=dens.ravel())
        f.create_dataset("x0", data=dens.ravel())
        f.create_dataset("vertices", data=vertices)
        f.create_dataset("faces", data=faces, dtype="i4")
        f.create_dataset("labels_final", data=labels.astype(np.int32))
        f.attrs["n_partitions"] = int(n_partitions)
        f.attrs["surface"] = "torus"
        f.attrs["completed_levels"] = 0
        f.attrs["optimizer"] = "arm"
        for k, v in (extra_attrs or {}).items():
            f.attrs[k] = v
    os.replace(tmp, path)
    return path


def vertex_granularity(lumped_mass: np.ndarray, n_partitions: int) -> float:
    """One-vertex mass as a fraction of the equal-area target.

    Vertices are indivisible, so no assignment method can drive the worst cell's
    relative deviation below this. Any acceptance bar set under it is impossible
    by construction rather than merely demanding.
    """
    target = float(np.sum(lumped_mass)) / n_partitions
    return float(np.max(lumped_mass)) / target


def run_gates(densities: np.ndarray, mesh: TriMesh, n_partitions: int) -> GateReport:
    """All three validity gates on one labelling."""
    dorm = detect_dormant_cells(densities)
    area = detect_area_imbalance(densities, mesh.v, n_partitions)
    disc = detect_disconnected_cells(densities, mesh.faces, mesh.v)
    return GateReport(
        n_dead=len(dorm.get("dead", [])),
        n_weak=len(dorm.get("weak", [])),
        min_peak_density=float(np.min(dorm.get("max_density_per_cell", [1.0]))),
        n_imbalanced=int(area.get("n_imbalanced", 0)),
        worst_rel_dev=float(area.get("worst_rel_dev", 0.0)),
        n_fragmented=int(disc.get("n_fragmented", 0)),
        fragmented=list(disc.get("fragmented", [])),
        worst_stray_rel=float(disc.get("worst_stray_rel", 0.0) or 0.0),
        vertex_granularity_rel=vertex_granularity(mesh.v, n_partitions),
    )


def scan_campaign(campaign_dir: str) -> Phase2Report:
    """Best perimeter, trajectory and censoring status of a refinement campaign.

    The best iterate is taken over the whole campaign rather than the last one,
    which is the project's standard workflow at the migration-cycling plateau
    (see CLAUDE.md): topology oscillates, so the final iterate is often not the
    best. Censoring is the complementary case -- best IS last, i.e. still
    falling.
    """
    files = sorted(glob.glob(os.path.join(campaign_dir, "iteration_*.h5")))
    if not files:
        raise FileNotFoundError(f"no iteration_*.h5 under {campaign_dir}")
    traj: List[Tuple[int, float]] = []
    initial: Optional[float] = None
    for fp in files:
        with h5py.File(fp, "r") as f:
            it = int(f.attrs["iteration_number"])
            traj.append((it, float(f.attrs["final_perimeter"])))
            if initial is None and "optimization_info" in f:
                grp = f["optimization_info"]
                if "initial_perimeter" in grp:
                    initial = float(np.array(grp["initial_perimeter"]).ravel()[0])
    traj.sort()
    best_it, best_p = min(traj, key=lambda r: r[1])
    return Phase2Report(
        campaign_dir=campaign_dir,
        n_iterations=len(traj),
        initial_perimeter=initial,
        best_perimeter=best_p,
        best_iteration=best_it,
        censored=(best_it == traj[-1][0]),
        trajectory=traj,
    )


def run_phase2(
    solution_path: str,
    campaign_config: str,
    repo_root: str,
    python: str = "python",
    extra_args: Optional[List[str]] = None,
) -> Tuple[Phase2Report, str]:
    """Refine one solution with a PINNED campaign config, and time it.

    ``campaign_config`` must be a campaign's own ``refinement.yaml`` (flat keys),
    which ``RefinementConfig.from_yaml_dict`` accepts directly. Passing the whole
    file is the point -- see the module docstring.

    Phase 2 is run as a subprocess so the arm travels exactly the code path the
    control travelled, rather than a reimplementation of it.
    """
    cmd = [
        python,
        os.path.join(repo_root, "scripts", "refine_perimeter.py"),
        "--solution", solution_path,
        "--config", campaign_config,
    ] + list(extra_args or [])
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"refine_perimeter.py failed ({proc.returncode})\n"
            f"--- stdout tail ---\n{proc.stdout[-2000:]}\n"
            f"--- stderr tail ---\n{proc.stderr[-2000:]}"
        )
    run_root = os.path.dirname(os.path.dirname(os.path.abspath(solution_path)))
    campaigns = sorted(
        glob.glob(os.path.join(run_root, "refinement", "*")),
        key=os.path.getmtime,
    )
    if not campaigns:
        raise FileNotFoundError(f"no refinement campaign created under {run_root}")
    report = scan_campaign(campaigns[-1])
    report.wall_seconds = wall
    return report, proc.stdout
