import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from typing import Optional

def build_plot_title(optimizer: str,
                     surface: Optional[str],
                     label1: str,
                     var1: int,
                     label2: str,
                     var2: int,
                     lambda_penalty: Optional[float],
                     seed: Optional[int],
                     use_analytic: Optional[bool] = None,
                     prefix: Optional[str] = None) -> str:
    parts = []
    if prefix:
        parts.append(prefix)
    if optimizer:
        parts.append(f"{optimizer}")
    if surface:
        parts.append(f"surf={surface}")
    parts.append(f"v_{label1}={var1}")
    parts.append(f"n_{label2}={var2}")
    if lambda_penalty is not None:
        parts.append(f"λ={lambda_penalty}")
    if seed is not None:
        parts.append(f"seed={seed}")
    if use_analytic is not None:
        parts.append(f"analytic={'yes' if use_analytic else 'no'}")
    return ", ".join(parts)


def plot_matrices(mesh, figsize=(15, 6)):
    """
    Plot the mass and stiffness matrices.

    Parameters
    ----------
    mesh : TriMesh
        The mesh with computed matrices.
    figsize : tuple
        Figure size.
    """
    if mesh.mass_matrix is None or mesh.stiffness_matrix is None:
        print("Matrices not computed yet. Call compute_matrices() first.")
        return None, None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    im1 = ax1.spy(mesh.mass_matrix, markersize=1)
    ax1.set_title(f'Mass Matrix M\nShape: {mesh.mass_matrix.shape}')
    ax1.set_xlabel('Column index')
    ax1.set_ylabel('Row index')

    im2 = ax2.spy(mesh.stiffness_matrix, markersize=1)
    ax2.set_title(f'Stiffness Matrix K\nShape: {mesh.stiffness_matrix.shape}')
    ax2.set_xlabel('Column index')
    ax2.set_ylabel('Row index')

    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_mesh_statistics(mesh, figsize=(12, 8)):
    """
    Plot mesh statistics and quality metrics.

    Parameters
    ----------
    mesh : TriMesh
        The mesh.
    figsize : tuple
        Figure size.
    """
    stats = mesh.get_mesh_statistics()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

    ax1.hist(mesh.triangle_areas, bins=20, alpha=0.7, edgecolor='black')
    ax1.set_title('Triangle Areas Distribution')
    ax1.set_xlabel('Area')
    ax1.set_ylabel('Count')
    ax1.axvline(stats['mean_triangle_area'], color='red', linestyle='--',
                label=f'Mean: {stats["mean_triangle_area"]:.4f}')
    ax1.legend()

    metrics = ['n_vertices', 'n_triangles', 'total_area']
    values = [stats[m] for m in metrics]
    labels = ['Vertices', 'Triangles', 'Computed Area']

    bars = ax2.bar(labels, values, alpha=0.7)
    ax2.set_title('Mesh Statistics')
    ax2.set_ylabel('Count/Area')

    for bar, value in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                f'{value:.2f}', ha='center', va='bottom')

    ax3.set_visible(False)
    ax4.set_visible(False)

    plt.tight_layout()
    return fig, (ax1, ax2, ax3, ax4)
