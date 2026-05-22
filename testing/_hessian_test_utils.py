"""Shared helpers for the Phase 2 exact-Hessian / Steiner-derivative tests.

Every harness script needs a compiled ``PerimeterOptimizer`` (and therefore a
compiled ``PartitionArrays``) loaded from a base solution or a refined
checkpoint.  ``build_optimizer`` is that single entry point.

Not a test itself — imported by the ``test_*`` and ``compare_*`` scripts in
this directory.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.io import (  # noqa: E402
    load_partition_from_base_file,
    load_partition_from_refined_file,
)
from src.pipeline.pipeline_orchestrator import detect_file_type  # noqa: E402
from src.partition.steiner_handler import SteinerHandler  # noqa: E402
from src.optimization.perimeter_optimizer import PerimeterOptimizer  # noqa: E402


def build_optimizer(solution_path: str) -> PerimeterOptimizer:
    """Load a base solution or refined checkpoint and return a compiled optimizer.

    The returned optimizer has ``_arrays`` populated (``compile()`` already
    called), so callers can read ``optimizer._arrays`` directly.

    Args:
        solution_path: Path to a Phase 1 base ``.h5`` or a Phase 2
            ``iteration_*.h5`` checkpoint.  A checkpoint exercises the Steiner
            code paths (it has triple points); a base file usually does not.

    Returns:
        A compiled :class:`PerimeterOptimizer`.
    """
    file_type = detect_file_type(solution_path)
    if file_type == 'base':
        mesh, partition = load_partition_from_base_file(solution_path, verbose=False)
    else:
        mesh, partition = load_partition_from_refined_file(solution_path, verbose=False)

    total_area = float(mesh.v.sum())
    target_area = total_area / partition.n_cells

    steiner = SteinerHandler(mesh, partition)
    optimizer = PerimeterOptimizer(
        partition, mesh, target_area,
        steiner_handler=steiner,
        use_vectorized=True,
    )
    optimizer.compile()
    return optimizer
