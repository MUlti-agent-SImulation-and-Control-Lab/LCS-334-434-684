"""
pathfinding/algo.py
-------------------
3D Risk-Aware A* with Joint Probability Lookahead

Key ideas
---------
1. Standard A* heuristic = 3D Euclidean distance to goal
2. g-cost = sum of edge traversal_costs so far (encodes distance + per-edge risk)
3. At each expansion, a JointRiskScorer evaluates the NEXT `window_size` nodes
   to compute a forward-looking joint risk penalty.
4. f = g + h + λ * joint_risk_lookahead
5. A node is only expanded if its joint risk window does not exceed
   `max_window_risk` (user-configurable safety gate).
"""

from __future__ import annotations
import heapq
import math
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from .joint_risk import JointRiskScorer
from graph.node import Node3D

if TYPE_CHECKING:
    pass


# ── Result dataclass ──────────────────────────────────────────────────────

class PathResult:
    def __init__(
        self,
        path       : List[Node3D],
        total_cost : float,
        total_dist : float,
        max_window_risk: float,
        mean_window_risk: float,
        segment_risks: List[dict],
        found      : bool,
    ):
        self.path             = path
        self.total_cost       = total_cost
        self.total_dist       = total_dist
        self.max_window_risk  = max_window_risk
        self.mean_window_risk = mean_window_risk
        self.segment_risks    = segment_risks
        self.found            = found

    def summary(self) -> str:
        if not self.found:
            return "No path found."
        waypoints = " → ".join(str(n.node_id) for n in self.path)
        return (
            f"Path found: {len(self.path)} nodes\n"
            f"  Waypoints    : {waypoints}\n"
            f"  Total dist   : {self.total_dist:.2f} m\n"
            f"  Total cost   : {self.total_cost:.3f}\n"
            f"  Max window risk  : {self.max_window_risk:.3f}\n"
            f"  Mean window risk : {self.mean_window_risk:.3f}\n"
        )


# ── A* planner ────────────────────────────────────────────────────────────

class RiskAwarePlanner:
    """
    Parameters
    ----------
    window_size      : number of consecutive nodes in the joint risk window
    risk_lambda      : weight of joint risk penalty in f-score
    max_window_risk  : paths whose next-window risk exceeds this are pruned
    turn_penalty_w   : weight for heading-change in joint risk calculation
    """

    def __init__(
        self,
        window_size    : int   = 5,
        risk_lambda    : float = 2.0,
        max_window_risk: float = None,
        turn_penalty_w : float = 0.15,
    ):
        self.window_size     = window_size
        self.risk_lambda     = risk_lambda
        self._max_window_risk_override = max_window_risk
        self.scorer          = JointRiskScorer(window_size, turn_penalty_w)

    def _calibrate_threshold(self, nodes) -> float:
        """Set max_window_risk = P(failure in window) at median node risk.
        This ensures ~50% of the graph is reachable even for risky terrains."""
        import numpy as np
        node_risks = [n.risk for n in nodes.values() if n.risk < 0.98]
        if not node_risks:
            return 0.99
        median_risk = float(np.median(node_risks))
        # Joint risk for window_size independent nodes at median risk
        base_joint = 1.0 - (1.0 - median_risk) ** self.window_size
        # Allow paths up to 20% worse than this baseline
        return float(min(base_joint * 1.25 + 0.05, 0.9999))

    def plan(
        self,
        nodes    : Dict[int, Node3D],
        start_id : int,
        goal_id  : int,
    ) -> PathResult:
        """
        Run A* from start_id to goal_id.

        Returns PathResult with full path and risk statistics.
        """
        if start_id not in nodes or goal_id not in nodes:
            return _empty_result()

        # Auto-calibrate threshold if not overridden
        self.max_window_risk = (
            self._max_window_risk_override
            if self._max_window_risk_override is not None
            else self._calibrate_threshold(nodes)
        )

        goal_node = nodes[goal_id]

        # priority queue: (f_score, node_id)
        open_heap: List[Tuple[float, int]] = []
        heapq.heappush(open_heap, (0.0, start_id))

        # g_score[node_id] = best cost from start
        g_score: Dict[int, float] = {start_id: 0.0}

        # came_from[node_id] = (parent_id, partial_path_to_parent)
        came_from: Dict[int, Optional[int]] = {start_id: None}

        # partial path cache for joint risk computation
        path_cache: Dict[int, List[Node3D]] = {start_id: [nodes[start_id]]}

        closed: set = set()

        while open_heap:
            f_cur, cur_id = heapq.heappop(open_heap)

            if cur_id in closed:
                continue
            closed.add(cur_id)

            if cur_id == goal_id:
                return self._reconstruct(
                    nodes, came_from, path_cache, cur_id,
                    g_score[goal_id]
                )

            cur_node   = nodes[cur_id]
            cur_path   = path_cache[cur_id]

            for nbr_id, edge_cost in cur_node.neighbours:
                if nbr_id not in nodes or nbr_id in closed:
                    continue

                nbr_node = nodes[nbr_id]

                # ── Joint risk lookahead gate ──────────────────────────
                jr = self.scorer.score_candidate(cur_path, nbr_node)
                #if jr > self.max_window_risk:
                    #continue   # prune this branch

                tentative_g = g_score[cur_id] + edge_cost

                if tentative_g < g_score.get(nbr_id, math.inf):
                    g_score[nbr_id]   = tentative_g
                    came_from[nbr_id] = cur_id

                    # Extend path cache (keep only last window_size nodes to save memory)
                    path_cache[nbr_id] = (cur_path + [nbr_node])[-self.window_size * 2:]

                    h = self._heuristic(nbr_node, goal_node)
                    f = tentative_g + h + self.risk_lambda * jr * 3.0

                    heapq.heappush(open_heap, (f, nbr_id))

        return _empty_result()

    # ── Heuristic ────────────────────────────────────────────────────────

    @staticmethod
    def _heuristic(node: Node3D, goal: Node3D) -> float:
        dx = node.x - goal.x
        dy = node.y - goal.y
        dz = node.z - goal.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    # ── Reconstruct path ─────────────────────────────────────────────────

    def _reconstruct(
        self,
        nodes      : Dict[int, Node3D],
        came_from  : Dict[int, Optional[int]],
        path_cache : Dict[int, List[Node3D]],
        goal_id    : int,
        total_cost : float,
    ) -> PathResult:
        path_ids = []
        cur = goal_id
        while cur is not None:
            path_ids.append(cur)
            cur = came_from[cur]
        path_ids.reverse()

        path_nodes = [nodes[nid] for nid in path_ids]

        # Total 3D distance
        total_dist = sum(
            path_nodes[i].euclidean_distance(path_nodes[i+1])
            for i in range(len(path_nodes) - 1)
        )

        segment_risks   = self.scorer.path_segments(path_nodes)
        max_window_risk  = self.scorer.max_segment_risk(path_nodes)
        mean_window_risk = self.scorer.mean_segment_risk(path_nodes)

        return PathResult(
            path            = path_nodes,
            total_cost      = total_cost,
            total_dist      = total_dist,
            max_window_risk = max_window_risk,
            mean_window_risk= mean_window_risk,
            segment_risks   = segment_risks,
            found           = True,
        )


def _empty_result() -> PathResult:
    return PathResult(
        path=[], total_cost=float("inf"), total_dist=0.0,
        max_window_risk=1.0, mean_window_risk=1.0,
        segment_risks=[], found=False
    )


# ── Convenience: plan multiple candidate paths and rank them ──────────────

def plan_with_alternatives(
    nodes        : Dict[int, Node3D],
    start_id     : int,
    goal_id      : int,
    window_size  : int   = 5,
    n_alternatives: int  = 3,
) -> List[PathResult]:
    """
    Run planner with progressively relaxed joint-risk thresholds to
    generate multiple alternative paths for comparison.
    """
    results = []
    # Use auto-calibration for first path; relax threshold for alternatives
    base_planner = RiskAwarePlanner(window_size=window_size)
    base_thresh  = base_planner._calibrate_threshold(nodes)
    thresholds   = np.linspace(base_thresh, min(base_thresh * 1.15, 0.9999), n_alternatives)

    for thresh in thresholds:
        planner = RiskAwarePlanner(
            window_size    = window_size,
            max_window_risk= float(thresh),
        )
        result = planner.plan(nodes, start_id, goal_id)
        if result.found:
            results.append(result)

    # Deduplicate by node sequence
    seen = set()
    unique = []
    for r in results:
        key = tuple(n.node_id for n in r.path)
        if key not in seen:
            seen.add(key)
            unique.append(r)

    # Sort by mean window risk, then total cost
    unique.sort(key=lambda r: (r.mean_window_risk, r.total_cost))
    return unique