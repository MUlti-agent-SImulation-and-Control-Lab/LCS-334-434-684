"""
graph/edge.py
-------------
Edge: directed connection between two Node3D objects.
"""

from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass
class Edge:
    """
    Attributes
    ----------
    src_id       : source node id
    dst_id       : destination node id
    distance     : Euclidean 3D distance (m)
    slope_deg    : inclination angle between src and dst (degrees)
    risk_blend   : average of src and dst risk scores
    traversal_cost: final edge weight used by pathfinding
    """
    src_id        : int
    dst_id        : int
    distance      : float
    slope_deg     : float = 0.0
    risk_blend    : float = 0.0
    traversal_cost: float = 0.0

    # ── Cost computation ────────────────────────────────────────────────

    @classmethod
    def from_nodes(cls, src, dst, risk_weight: float = 0.7) -> "Edge":
        """
        Build an Edge between two Node3D objects.

        traversal_cost = distance * (1 + risk_weight * risk_blend)
                         + slope_penalty

        Slope penalty adds cost for steep ascent/descent.
        """
        dx = dst.x - src.x
        dy = dst.y - src.y
        dz = dst.z - src.z
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)

        horiz = math.sqrt(dx*dx + dy*dy)
        if horiz > 1e-6:
            slope_deg = math.degrees(math.atan2(abs(dz), horiz))
        else:
            slope_deg = 90.0

        risk_blend = (src.risk + dst.risk) / 2.0

        slope_penalty = (slope_deg / 45.0) ** 2   # 0 → 0, 45° → 1
        traversal_cost = dist * (1.0 + 0.3 * slope_penalty)

        return cls(
            src_id=src.node_id,
            dst_id=dst.node_id,
            distance=dist,
            slope_deg=slope_deg,
            risk_blend=risk_blend,
            traversal_cost=traversal_cost,
        )

    def is_passable(self, max_slope_deg: float = 35.0, max_risk: float = 0.85) -> bool:
        return self.slope_deg <= max_slope_deg and self.risk_blend < max_risk

    def __repr__(self) -> str:
        return (
            f"Edge({self.src_id}→{self.dst_id}, "
            f"dist={self.distance:.2f}m, slope={self.slope_deg:.1f}°, "
            f"cost={self.traversal_cost:.3f})"
        )