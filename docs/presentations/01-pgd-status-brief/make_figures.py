#!/usr/bin/env python3
"""Figures for docs/presentations/01-pgd-status-brief.

Every figure is built from files under results/ (listed in SOURCES below and in
README.md). Renders go through the project's own exact-geometry region builder
(scripts/visualize_partition_fast.py) so cell boundaries are the Phase 2 contours,
not a per-vertex colouring. Run from anywhere:

    python docs/presentations/01-pgd-status-brief/make_figures.py

Requires the project environment with pyvista (offscreen) and matplotlib.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.mesh.tri_mesh import TriMesh  # noqa: E402
from src.partition.area_calculator import AreaCalculator  # noqa: E402
from src.partition.find_contours import (  # noqa: E402
    AREA_IMBALANCE_REL_THRESHOLD,
    detect_area_imbalance,
    detect_disconnected_cells,
)
from src.partition.steiner_handler import SteinerHandler  # noqa: E402
from src.visualization.cell_coloring import assign_cell_colors, build_cell_adjacency  # noqa: E402

RESULTS = REPO / "results"

RUN_N100 = RESULTS / "run_20260709_081548_surftorus_npart100_v1nt100-348_incr62_v2np96-328_incr58_lam5.1_seed84172851"
RUN_N150_A = RESULTS / "run_20260710_215525_surftorus_npart150_v1nt100-224_incr62_v2np96-212_incr58_lam6.0_seed84172851"
RUN_N150_B = RESULTS / "run_20260711_165615_surftorus_npart150_v1nt100-348_incr62_v2np96-328_incr58_lam6.0_seed84172851"
RUN_N200_A = RESULTS / "run_20260713_211827_surftorus_npart200_v1nt100-224_incr62_v2np96-212_incr58_lam11.0_seed61803399"
RUN_N200_B = RESULTS / "run_20260722_175451_surftorus_npart200_v1nt100-348_incr62_v2np96-328_incr58_lam11.0_seed61803399"
RUN_N300_A = RESULTS / "run_20260806_123326_surftorus_npart300_v1nt100-224_incr62_v2np96-212_incr58_lam11.5_seed61803399"
RUN_N300_B = RESULTS / "run_20260808_191030_surftorus_npart300_v1nt100-348_incr62_v2np96-328_incr58_lam11.5_seed61803399"

CAMPAIGN = "ipopt_btol0.001_lbfgs30_hess_bestiter_partial"
READOUT = "readout/dualshift_gate0.05_repair/solution_balanced.h5"

SOURCES = {
    "fig_partition_n100.png": RUN_N100 / "refinement" / CAMPAIGN / "iteration_020_20260709_222553.h5",
    "fig_partition_n200.png": RUN_N200_B / "refinement" / CAMPAIGN / "iteration_016_20260723_152237.h5",
    "fig_n300_raw.png": RUN_N300_A / "solution" / "surface_part300_surftorus_v1nt100-224_incr62_v2np96-212_incr58_lam11.5_seed61803399_20260806_123326.h5",
    "fig_n300_readout.png": RUN_N300_A / READOUT,
}

# Phase 1 ladders: (label, [run dirs whose timing_profile.yaml levels are summed])
LADDERS = [
    ("n = 100", [RUN_N100]),
    ("n = 150", [RUN_N150_A, RUN_N150_B]),
    ("n = 200", [RUN_N200_A, RUN_N200_B]),
    ("n = 300", [RUN_N300_A, RUN_N300_B]),
]

# Deck-wide colours. Cell fills: a muted categorical set, assigned neighbour-
# distinct (never cycled by index). Plot colours: text tokens for labels, one
# hue per series, status red only for "outside the gate".
CELL_FILLS = ["#a9c6e8", "#f3c9a3", "#b8dfc4", "#f5e3a0", "#dcc4e3",
              "#c8d4a2", "#f4bcbc", "#bcd3de", "#e7d2b9", "#cfd0e6"]
GREY_FILL = "#e4e7ec"
HILITE_RUNT = "#e34948"
HILITE_SPLIT = "#eb6834"
BOUNDARY = "#3a3a3a"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8985"
SERIES_1, SERIES_2 = "#2a78d6", "#eb6834"
RENDER_SIZE = (2400, 1500)


# ---------------------------------------------------------------------------
# Renders
# ---------------------------------------------------------------------------

def _boundary_lines(mesh, partition, steiner_handler):
    """Polyline segments of the cell boundaries, from variable-point positions."""
    import pyvista as pv

    verts = mesh.vertices
    tp_by_tri = {tp.triangle_idx: tp for tp in steiner_handler.triple_points}
    points, lines = [], []

    def add(a, b):
        i = len(points)
        points.extend([a, b])
        lines.extend([2, i, i + 1])

    for seg in partition.triangle_segments:
        vps = [partition.variable_points[i] for i in seg.var_point_indices]
        vps = [vp for vp in vps if vp.active]
        pos = [vp.evaluate(verts) for vp in vps]
        if len(pos) == 2:
            add(pos[0], pos[1])
        elif len(pos) == 3:
            tp = tp_by_tri.get(seg.triangle_idx)
            centre = None
            if tp is not None:
                try:
                    centre = tp.steiner_point if tp.steiner_point is not None \
                        else tp.compute_steiner_point(partition=partition)
                except Exception:
                    centre = None
            if centre is None:
                centre = np.mean(pos, axis=0)
            for p in pos:
                add(p, centre)
    if not points:
        return None
    return pv.PolyData(np.asarray(points), lines=np.asarray(lines))


def render_partition(solution_path, out_png, view_dir, highlight=None,
                     zoom=1.35, lines=True):
    """Render one partition to PNG.

    highlight: None for neighbour-distinct colouring, or {cell: colour} to paint
    those cells on a grey partition. view_dir is the camera direction from the
    torus centre (z up); it is fixed per figure so before/after pairs share it.
    """
    import pyvista as pv
    from visualize_partition_fast import _render_region_fast, load_partition_smart

    print(f"[render] {out_png.name} <- {solution_path.relative_to(REPO)}")
    mesh, partition, _ = load_partition_smart(str(solution_path))
    area_calc = AreaCalculator(mesh, partition)
    steiner = SteinerHandler(mesh, partition)
    n_cells = partition.n_cells
    seg_lookup = {ts.triangle_idx: ts for ts in partition.triangle_segments}

    if highlight is None:
        adjacency = build_cell_adjacency(partition.vertex_labels, mesh.faces, n_cells)
        colors = assign_cell_colors(adjacency, n_cells, palette=CELL_FILLS)
    else:
        colors = [highlight.get(k, GREY_FILL) for k in range(n_cells)]

    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True, window_size=RENDER_SIZE, lighting="three lights")
    plotter.set_background("white")
    for k in range(n_cells):
        _render_region_fast(plotter, mesh, partition, area_calc, steiner, k,
                            colors[k], seg_lookup, opacity=1.0, show_edges=False)
    if lines:
        pl = _boundary_lines(mesh, partition, steiner)
        if pl is not None:
            plotter.add_mesh(pl, color=BOUNDARY, line_width=2.0,
                             render_lines_as_tubes=False)

    centre = (mesh.vertices.min(0) + mesh.vertices.max(0)) / 2
    extent = np.linalg.norm(mesh.vertices.max(0) - mesh.vertices.min(0))
    d = np.asarray(view_dir, float)
    d /= np.linalg.norm(d)
    plotter.camera_position = [(centre + d * extent * 1.8).tolist(),
                               centre.tolist(), [0, 0, 1]]
    plotter.camera.zoom(zoom)
    try:
        plotter.enable_anti_aliasing("ssaa")
    except Exception:
        pass
    plotter.screenshot(str(out_png), transparent_background=True)
    plotter.close()
    _autocrop(out_png)


def _autocrop(png_path, margin=24):
    """Trim the transparent canvas around the render so the slide gets the surface, not the air."""
    from PIL import Image
    im = Image.open(png_path)
    bbox = im.getchannel("A").getbbox()
    if bbox is None:
        return
    l, t, r, b = bbox
    l, t = max(l - margin, 0), max(t - margin, 0)
    r, b = min(r + margin, im.width), min(b + margin, im.height)
    im.crop((l, t, r, b)).save(png_path)


# ---------------------------------------------------------------------------
# Data for the plots
# ---------------------------------------------------------------------------

def load_densities(h5_path):
    import h5py
    with h5py.File(h5_path, "r") as f:
        V = f["vertices"].shape[0]
        N = int(f.attrs["n_partitions"])
        x = f["x_opt"][...]
        mesh = TriMesh(f["vertices"][...], f["faces"][...])
    return mesh, x.reshape(V, N), N


def territory_stats(h5_path):
    mesh, u, N = load_densities(h5_path)
    v = mesh.v
    imb = detect_area_imbalance(u, v, N)
    frag = detect_disconnected_cells(u, mesh.faces, v)
    target = imb["target_area"]
    mass = (v[:, None] * u).sum(0) / target
    return {
        "territory": np.asarray(imb["discrete_areas"]) / target,
        "mass": mass,
        "n_imbalanced": imb["n_imbalanced"],
        "imbalanced": list(imb["imbalanced"]),
        "worst_rel": imb["worst_rel_dev"],
        "n_fragmented": frag["n_fragmented"],
        "fragmented": [int(c) for c in frag.get("fragmented", [])],
    }


def ladder_hours():
    import yaml
    rows = []
    for label, runs in LADDERS:
        levels = []
        for r in runs:
            y = yaml.safe_load((r / "solution" / "timing_profile.yaml").read_text())
            levels += [float(l["level_wall_s"]) for l in y["levels"]]
        rows.append((label, sum(levels) / 3600.0, levels))
    return rows


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c4")
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=13, length=3, color="#c9c8c4")
    ax.yaxis.grid(True, color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_phase1_wall(out_svg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = ladder_hours()
    labels = [r[0] for r in rows]
    hours = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, hours, height=0.42, color=SERIES_1)
    for yi, h in zip(y, hours):
        lab = f"{h:.1f} h" + (f"  ({h/24:.1f} days)" if h >= 24 else "")
        ax.text(h + 1.0, yi, lab, va="center", ha="left", fontsize=14, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=14, color=INK)
    ax.set_xlim(0, max(hours) * 1.18)
    ax.set_xlabel("relaxation wall time, full mesh ladder to 114,144 vertices  [hours]",
                  fontsize=13, color=INK2)
    ax.xaxis.grid(True, color="#e6e5e1", linewidth=0.8)
    ax.yaxis.grid(False)
    _style(ax)
    ax.yaxis.grid(False)
    fig.tight_layout()
    fig.savefig(out_svg, format="svg")
    plt.close(fig)
    print(f"[plot]   {out_svg.name}: " + ", ".join(f"{l} {h:.1f} h" for l, h in zip(labels, hours)))
    return rows


def plot_territory(out_svg, raw, readout):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = len(raw["territory"])
    order = np.argsort(raw["territory"])
    x = np.arange(1, N + 1)
    gate = AREA_IMBALANCE_REL_THRESHOLD

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.axhspan(1 - gate, 1 + gate, color="#f0f0ed", zorder=0)
    ax.axhline(1.0, color="#c9c8c4", linewidth=0.8, zorder=1)
    ax.plot(x, raw["territory"][order], color=SERIES_1, linewidth=2.0,
            label="winner-take-all territory / target  (raw relaxation)", zorder=3)
    ax.plot(x, readout["territory"][order], color=SERIES_2, linewidth=2.0,
            label="territory / target  (after balanced readout)", zorder=4)
    ax.plot(x, raw["mass"][order], color=INK, linewidth=1.3, linestyle=(0, (3, 3)),
            label="continuous mass  ∫u dσ / target  (raw) — exactly 1", zorder=6)

    bad = np.where(np.abs(raw["territory"][order] - 1) > gate)[0]
    ax.scatter(x[bad], raw["territory"][order][bad], s=26, color=HILITE_RUNT,
               zorder=5, edgecolor="white", linewidth=0.8)
    worst = raw["territory"][order][0]
    ax.annotate(f"worst cell: {100*(worst-1):+.1f}%", xy=(x[0], worst),
                xytext=(x[0] + 22, worst - 0.02), fontsize=13, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK3, linewidth=0.8))
    ax.text(N * 0.35, 1 + gate + 0.012, f"±{int(gate*100)}% equal-area gate", ha="left",
            va="bottom", fontsize=12, color=INK2)

    ax.set_xlim(0, N + 1)
    ax.set_ylim(min(0.58, worst - 0.04), 1.16)
    ax.set_xlabel(f"cells, sorted by raw territory  (n = {N})", fontsize=13, color=INK2)
    ax.set_ylabel("area / equal-area target", fontsize=13, color=INK2)
    _style(ax)
    leg = ax.legend(loc="center right", fontsize=11.5, frameon=False)
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    fig.savefig(out_svg, format="svg")
    plt.close(fig)
    print(f"[plot]   {out_svg.name}: raw {raw['n_imbalanced']} imbalanced / worst "
          f"{100*raw['worst_rel']:.2f}% / {raw['n_fragmented']} fragmented; readout "
          f"{readout['n_imbalanced']} / {100*readout['worst_rel']:.2f}% / {readout['n_fragmented']}")


# ---------------------------------------------------------------------------

def main(argv):
    only = set(argv[1:])

    def want(name):
        return not only or name in only

    if want("plots"):
        plot_phase1_wall(HERE / "fig_phase1_wall.svg")
        raw = territory_stats(SOURCES["fig_n300_raw.png"])
        rd = territory_stats(SOURCES["fig_n300_readout.png"])
        plot_territory(HERE / "fig_territory_n300.svg", raw, rd)
    else:
        raw = rd = None

    view = (1.0, -1.15, 0.85)
    # N=300: face the two fragmented cells (274, 290; azimuth -48..76 deg) and the
    # worst runts (299 at -28 deg, 290 at -40 deg, 275/195 near -85 deg).
    view_n300 = (np.cos(np.radians(-50)), np.sin(np.radians(-50)), 0.9)
    if want("n100"):
        render_partition(SOURCES["fig_partition_n100.png"], HERE / "fig_partition_n100.png", view)
    if want("n200"):
        render_partition(SOURCES["fig_partition_n200.png"], HERE / "fig_partition_n200.png", view)
    if want("n300"):
        raw = raw or territory_stats(SOURCES["fig_n300_raw.png"])
        hl = {c: HILITE_RUNT for c in raw["imbalanced"]}
        hl.update({c: HILITE_SPLIT for c in raw["fragmented"]})
        print(f"[n300] highlighting {len(raw['imbalanced'])} imbalanced + {len(raw['fragmented'])} fragmented cells")
        render_partition(SOURCES["fig_n300_raw.png"], HERE / "fig_n300_raw.png", view_n300, highlight=hl)
        render_partition(SOURCES["fig_n300_readout.png"], HERE / "fig_n300_readout.png", view_n300, highlight=hl)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
