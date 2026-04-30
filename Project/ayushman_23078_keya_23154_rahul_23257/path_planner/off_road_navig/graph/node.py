"""
graph/node.py
-------------
Node3D: represents a traversable region (voxel cluster centroid) in the
        3D environment graph.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Node3D:
    """
    Attributes
    ----------
    node_id      : unique integer identifier
    x, y, z      : world-space centre of the cluster (metres)
    risk         : scalar traversal risk in [0, 1]
    label        : terrain class (0=ground,1=obstacle,2=water,3=veg,4=unknown)
    slope_deg    : average slope in degrees
    roughness    : surface roughness (m)
    wetness      : wetness proxy [0,1]
    point_count  : number of LiDAR points in this cluster
    neighbours   : list of (node_id, edge_cost) tuples — filled by builder
    """
    node_id    : int
    x          : float
    y          : float
    z          : float
    risk       : float = 0.0
    label      : int   = 0
    slope_deg  : float = 0.0
    roughness  : float = 0.0
    wetness    : float = 0.0
    point_count: int   = 0
    neighbours : list  = field(default_factory=list)   # [(node_id, cost), ...]

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    def euclidean_distance(self, other: "Node3D") -> float:
        return float(np.linalg.norm(self.position - other.position))

    def is_traversable(self, max_risk: float = 0.85) -> bool:
        """A node is traversable if its risk is below the threshold."""
        return self.risk < max_risk

    def __repr__(self) -> str:
        return (
            f"Node3D(id={self.node_id}, pos=({self.x:.1f},{self.y:.1f},{self.z:.1f}), "
            f"risk={self.risk:.2f}, label={self.label})"
        )

    def __hash__(self):
        return hash(self.node_id)

    def __eq__(self, other):
        return isinstance(other, Node3D) and self.node_id == other.node_id