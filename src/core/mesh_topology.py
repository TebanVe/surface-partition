"""
Mesh topology and connectivity for topology switching operations.

This module provides precomputed mesh connectivity structures that enable
efficient topology switching operations during partition optimization.

Key functionality:
- Vertex-to-edges mapping: Find all edges incident to a vertex
- Edge-to-triangles mapping: Find triangles sharing an edge
- Free edge detection: Find edges without variable points
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional

from ..logging_config import get_logger
from .tri_mesh import TriMesh


class MeshTopology:
    """
    Precomputed mesh connectivity for efficient topology switching.
    
    Built once at initialization, provides O(1) lookups for:
    - Which edges are incident to a vertex
    - Which triangles share an edge
    - Adjacency information for topology operations
    
    Attributes:
        mesh: The underlying TriMesh
        vertex_to_edges: Dict[vertex_idx] -> List[edge tuples]
        vertex_to_triangles: Dict[vertex_idx] -> List[triangle indices]
        edge_to_triangles: Dict[edge tuple] -> List[triangle indices]
    """
    
    def __init__(self, mesh: TriMesh):
        """
        Build mesh connectivity structures.
        
        Args:
            mesh: TriMesh object
        """
        self.mesh = mesh
        self.logger = get_logger(__name__)
        
        # Initialize connectivity maps
        self.vertex_to_edges: Dict[int, Set[Tuple[int, int]]] = {}
        self.vertex_to_triangles: Dict[int, List[int]] = {}
        self.edge_to_triangles: Dict[Tuple[int, int], List[int]] = {}
        self._vertex_neighbors: Dict[int, Set[int]] = {}
        
        # Build connectivity
        self._build_connectivity()
        self._build_vertex_neighbors()
        
        self.logger.info(f"Built MeshTopology: {len(self.vertex_to_edges)} vertices, "
                        f"{len(self.edge_to_triangles)} unique edges")
    
    def _build_connectivity(self):
        """
        Build all connectivity maps by scanning mesh triangles once.
        
        Time complexity: O(T) where T is number of triangles
        Space complexity: O(V + E) where V is vertices, E is edges
        """
        # Initialize for all vertices
        n_vertices = self.mesh.vertices.shape[0]
        for v_idx in range(n_vertices):
            self.vertex_to_edges[v_idx] = set()
            self.vertex_to_triangles[v_idx] = []
        
        # Scan all triangles
        for tri_idx, face in enumerate(self.mesh.faces):
            v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
            
            # Register triangle for each vertex
            self.vertex_to_triangles[v1].append(tri_idx)
            self.vertex_to_triangles[v2].append(tri_idx)
            self.vertex_to_triangles[v3].append(tri_idx)
            
            # Extract and normalize edges
            edges = [
                tuple(sorted([v1, v2])),
                tuple(sorted([v2, v3])),
                tuple(sorted([v3, v1]))
            ]
            
            # Register edges for vertices and triangles
            for edge in edges:
                v_start, v_end = edge
                
                # Add edge to both vertices
                self.vertex_to_edges[v_start].add(edge)
                self.vertex_to_edges[v_end].add(edge)
                
                # Add triangle to edge
                if edge not in self.edge_to_triangles:
                    self.edge_to_triangles[edge] = []
                self.edge_to_triangles[edge].append(tri_idx)
    
    def get_edges_at_vertex(self, vertex_idx: int) -> List[Tuple[int, int]]:
        """
        Get all edges incident to a vertex.
        
        Args:
            vertex_idx: Index of the vertex
            
        Returns:
            List of edge tuples (normalized: smaller index first)
        """
        return list(self.vertex_to_edges.get(vertex_idx, set()))
    
    def get_triangles_at_vertex(self, vertex_idx: int) -> List[int]:
        """
        Get all triangles containing a vertex.
        
        Args:
            vertex_idx: Index of the vertex
            
        Returns:
            List of triangle indices
        """
        return self.vertex_to_triangles.get(vertex_idx, [])
    
    def get_triangles_sharing_edge(self, edge: Tuple[int, int]) -> List[int]:
        """
        Get triangles sharing an edge.
        
        Args:
            edge: Edge tuple (normalized or not - will be normalized internally)
            
        Returns:
            List of triangle indices (typically 1 or 2)
        """
        normalized_edge = tuple(sorted(edge))
        return self.edge_to_triangles.get(normalized_edge, [])
    
    def _build_vertex_neighbors(self):
        """Build cached vertex neighbor lookup from edge data."""
        self._vertex_neighbors = {}
        for edge in self.edge_to_triangles:
            v1, v2 = edge
            self._vertex_neighbors.setdefault(v1, set()).add(v2)
            self._vertex_neighbors.setdefault(v2, set()).add(v1)
    
    def are_neighbors(self, v1: int, v2: int) -> bool:
        """Check if two vertices are at mesh distance 1."""
        return v2 in self._vertex_neighbors.get(v1, set())

