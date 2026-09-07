"""Build a ``SurfaceProvider`` from an experiment config, for any surface.

Why this exists
---------------
``scripts/find_surface_partition.py`` (Phase 1 / PGD) has always dispatched on
``experiment.surface`` and built any of the four providers. ``run_mbo_arm.py``
(approach B) did not: it imported ``TorusMeshProvider`` directly and read only
``cfg["surface"]["torus"]``, so approach B was torus-only even though
``src/partition/mbo_auction.py`` is entirely mesh-generic -- it derives ``tau``
from ``edge_lengths(mesh)`` and ``mesh.M.sum()`` and consumes ``faces``
directly, with no structured-grid assumption anywhere.

This module supplies the dispatch that was missing, as a *new* module rather
than a refactor of the Phase 1 CLI, so the stable production path is untouched.
``find_surface_partition.py`` deliberately still carries its own equivalent
branch; consolidating the two is a separate change with its own regression
burden and is not worth coupling to this one.

Structured vs unstructured
--------------------------
``is_structured`` answers one narrow question: does ``V == res1 * res2`` hold?
It is true for the torus, whose mesh is a product grid, and false for the
marching-cubes surfaces, where the resolution pair is a *sampling grid* and the
vertex count is whatever the level set intersects (an archived double-torus
solution carries ``var1*var2 = 200*150 = 30,000`` against ``V = 56,700``).

Callers use it to gate bookkeeping assertions that are only meaningful on a
product grid. It is deliberately a whitelist of one rather than a guess: the
ellipsoid builds its own vertex array with polar caps and has never been
checked against this identity, so it is not claimed here.
"""

from typing import Tuple

from .base import SurfaceProvider

#: Surfaces whose vertex count is exactly the product of the resolution pair.
#: See the module docstring -- a whitelist, not an inference.
STRUCTURED_SURFACES = frozenset({"torus"})


def is_structured(surface_name: str) -> bool:
    """True when ``V == res1 * res2`` holds for this surface's meshes."""
    return surface_name in STRUCTURED_SURFACES


def resolve_surface_params(cfg: dict, surface_name: str) -> dict:
    """Surface parameters from a sectioned or legacy-flat config dict.

    Same semantics as ``_resolve_surface_params`` in
    ``scripts/find_surface_partition.py``, so both entry points read the same
    configs identically.
    """
    surface_section = cfg.get("surface", {})
    if isinstance(surface_section, dict) and surface_name in surface_section:
        return surface_section[surface_name]
    return cfg


def surface_name_from_config(cfg: dict) -> str:
    """The surface a config selects.

    Prefers ``experiment.surface``; falls back to the single key under
    ``surface:``; defaults to ``torus`` so pre-existing torus configs that
    state neither keep working unchanged.
    """
    experiment = cfg.get("experiment", {})
    if isinstance(experiment, dict) and experiment.get("surface"):
        return str(experiment["surface"])
    if cfg.get("surface_type"):
        return str(cfg["surface_type"])
    surface_section = cfg.get("surface", {})
    if isinstance(surface_section, dict) and len(surface_section) == 1:
        return str(next(iter(surface_section)))
    return "torus"


def build_provider(cfg: dict, surface_name: str = None) -> SurfaceProvider:
    """Construct the provider a config selects, at its BASE resolution.

    The returned provider carries its refinement increments, so a ladder is
    driven through the ``SurfaceProvider`` interface --
    ``get_initial_resolution`` / ``get_resolution_increment`` /
    ``set_resolution`` -- with no per-surface knowledge in the caller.
    """
    name = surface_name or surface_name_from_config(cfg)
    sp = resolve_surface_params(cfg, name)

    if name == "torus":
        from .torus import TorusMeshProvider

        return TorusMeshProvider(
            int(sp.get("n_theta", 32)),
            int(sp.get("n_phi", 24)),
            float(sp.get("R", 1.0)),
            float(sp.get("r", 0.3)),
            n_theta_increment=int(sp.get("n_theta_increment", 0)),
            n_phi_increment=int(sp.get("n_phi_increment", 0)),
        )

    if name == "ellipsoid":
        from .ellipsoid import EllipsoidMeshProvider

        return EllipsoidMeshProvider(
            int(sp.get("n_theta", 32)),
            int(sp.get("n_phi", 64)),
            float(sp.get("a", 1.0)),
            float(sp.get("b", 1.0)),
            float(sp.get("c", 0.7)),
            n_theta_increment=int(sp.get("n_theta_increment", 0)),
            n_phi_increment=int(sp.get("n_phi_increment", 0)),
        )

    if name == "double_torus":
        from .double_torus import DoubleTorusMeshProvider

        return DoubleTorusMeshProvider(
            int(sp.get("n_grid_x", 100)),
            int(sp.get("n_grid_y", 100)),
            int(sp.get("n_grid_z", 100)),
            c=float(sp.get("c", 0.03)),
            n_grid_x_increment=int(sp.get("n_grid_x_increment", 0)),
            n_grid_y_increment=int(sp.get("n_grid_y_increment", 0)),
        )

    if name == "banchoff_chmutov":
        from .banchoff_chmutov import BanchoffChmutovMeshProvider

        return BanchoffChmutovMeshProvider(
            int(sp.get("n_grid_x", 100)),
            int(sp.get("n_grid_y", 100)),
            int(sp.get("n_grid_z", 100)),
            n_grid_x_increment=int(sp.get("n_grid_x_increment", 0)),
            n_grid_y_increment=int(sp.get("n_grid_y_increment", 0)),
        )

    raise ValueError(f"Unsupported surface type: {name}")


def ladder_resolutions(
    provider: SurfaceProvider, levels: int
) -> list:
    """The ``(res1, res2)`` pair at each level of the refinement ladder."""
    r1, r2 = provider.get_initial_resolution()
    d1, d2 = provider.get_resolution_increment()
    return [(r1 + L * d1, r2 + L * d2) for L in range(int(levels))]


def resolution_label(provider: SurfaceProvider, res: Tuple[int, int]) -> str:
    """``"100x96"`` -- for banners, using the provider's own axis labels."""
    return f"{res[0]}x{res[1]}"
