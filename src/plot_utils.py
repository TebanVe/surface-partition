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


def plot_ring_mesh(ring_mesh, show_triangles=True, show_vertices=True, 
                   figsize=(10, 8), title="Ring Mesh"):
    """
    Plot the ring mesh with triangles and vertices.
    
    Parameters:
    -----------
    ring_mesh : RingMesh
        The ring mesh to visualize
    show_triangles : bool
        Whether to show triangle edges
    show_vertices : bool
        Whether to show vertices
    figsize : tuple
        Figure size
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract vertices and triangles
    vertices = ring_mesh.vertices
    triangles = ring_mesh.triangles
    
    # Create triangulation object
    triangulation = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)
    
    # Plot triangles
    if show_triangles:
        ax.triplot(triangulation, 'b-', linewidth=0.5, alpha=0.7)
    
    # Plot vertices
    if show_vertices:
        ax.plot(vertices[:, 0], vertices[:, 1], 'ro', markersize=3)
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Add title and labels
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add circle boundaries for reference
    theta = np.linspace(0, 2*np.pi, 100)
    r_inner = ring_mesh.r_inner
    r_outer = ring_mesh.r_outer
    
    # Inner circle
    x_inner = r_inner * np.cos(theta)
    y_inner = r_inner * np.sin(theta)
    ax.plot(x_inner, y_inner, 'k--', linewidth=2, alpha=0.7, label=f'Inner radius ({r_inner})')
    
    # Outer circle
    x_outer = r_outer * np.cos(theta)
    y_outer = r_outer * np.sin(theta)
    ax.plot(x_outer, y_outer, 'k--', linewidth=2, alpha=0.7, label=f'Outer radius ({r_outer})')
    
    ax.legend()
    
    plt.tight_layout()
    return fig, ax

def plot_matrices(ring_mesh, figsize=(15, 6)):
    """
    Plot the mass and stiffness matrices.
    
    Parameters:
    -----------
    ring_mesh : RingMesh
        The ring mesh with computed matrices
    figsize : tuple
        Figure size
    """
    if ring_mesh.mass_matrix is None or ring_mesh.stiffness_matrix is None:
        print("Matrices not computed yet. Call compute_matrices() first.")
        return None, None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot mass matrix
    im1 = ax1.spy(ring_mesh.mass_matrix, markersize=1)
    ax1.set_title(f'Mass Matrix M\nShape: {ring_mesh.mass_matrix.shape}')
    ax1.set_xlabel('Column index')
    ax1.set_ylabel('Row index')
    
    # Plot stiffness matrix
    im2 = ax2.spy(ring_mesh.stiffness_matrix, markersize=1)
    ax2.set_title(f'Stiffness Matrix K\nShape: {ring_mesh.stiffness_matrix.shape}')
    ax2.set_xlabel('Column index')
    ax2.set_ylabel('Row index')
    
    plt.tight_layout()
    return fig, (ax1, ax2)

def plot_mesh_statistics(ring_mesh, figsize=(12, 8)):
    """
    Plot mesh statistics and quality metrics.
    
    Parameters:
    -----------
    ring_mesh : RingMesh
        The ring mesh
    figsize : tuple
        Figure size
    """
    stats = ring_mesh.get_mesh_statistics()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Triangle areas histogram
    ax1.hist(ring_mesh.triangle_areas, bins=20, alpha=0.7, edgecolor='black')
    ax1.set_title('Triangle Areas Distribution')
    ax1.set_xlabel('Area')
    ax1.set_ylabel('Count')
    ax1.axvline(stats['mean_triangle_area'], color='red', linestyle='--', 
                label=f'Mean: {stats["mean_triangle_area"]:.4f}')
    ax1.legend()
    
    # Plot 2: Mesh quality metrics
    metrics = ['n_vertices', 'n_triangles', 'total_area', 'theoretical_area']
    values = [stats[m] for m in metrics]
    labels = ['Vertices', 'Triangles', 'Computed Area', 'Theoretical Area']
    
    bars = ax2.bar(labels, values, alpha=0.7)
    ax2.set_title('Mesh Statistics')
    ax2.set_ylabel('Count/Area')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                f'{value:.2f}', ha='center', va='bottom')
    
    # Plot 3: Area comparison
    areas = [stats['total_area'], stats['theoretical_area']]
    labels = ['Computed', 'Theoretical']
    colors = ['blue', 'orange']
    
    bars = ax3.bar(labels, areas, color=colors, alpha=0.7)
    ax3.set_title('Area Comparison')
    ax3.set_ylabel('Area')
    
    # Add percentage difference
    diff_percent = abs(areas[0] - areas[1]) / areas[1] * 100
    ax3.text(0.5, max(areas) * 0.8, f'Difference: {diff_percent:.2f}%', 
             ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
    
    # Plot 4: Mesh parameters
    params = ['r_inner', 'r_outer', 'n_radial', 'n_angular']
    param_values = [stats[p] for p in params]
    param_labels = ['Inner Radius', 'Outer Radius', 'Radial Points', 'Angular Points']
    
    bars = ax4.bar(param_labels, param_values, alpha=0.7)
    ax4.set_title('Mesh Parameters')
    ax4.set_ylabel('Value')
    
    # Rotate x-axis labels for better readability
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig, (ax1, ax2, ax3, ax4)

def save_mesh_plots(ring_mesh, output_dir="visualizations", prefix="ring_mesh"):
    """
    Save all mesh plots to files.
    
    Parameters:
    -----------
    ring_mesh : RingMesh
        The ring mesh
    output_dir : str
        Output directory
    prefix : str
        File prefix
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save mesh plot
    fig, ax = plot_ring_mesh(ring_mesh)
    fig.savefig(os.path.join(output_dir, f"{prefix}_mesh.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Save matrix plots
    fig, (ax1, ax2) = plot_matrices(ring_mesh)
    if fig is not None:
        fig.savefig(os.path.join(output_dir, f"{prefix}_matrices.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # Save statistics plots
    fig, axes = plot_mesh_statistics(ring_mesh)
    fig.savefig(os.path.join(output_dir, f"{prefix}_statistics.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Plots saved to {output_dir}/")