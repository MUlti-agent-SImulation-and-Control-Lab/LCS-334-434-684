"""
pathfinding/joint_risk.py
--------------------------
Joint probability distribution of risk over a sliding window of N nodes.

Motivation
----------
Node A may have low individual risk, but every forward path from A
involves sharp turns or water crossings — making the *joint* risk much
higher than its isolated score suggests.

This module computes:

  JointRisk(path[i : i+window]) = 1 - ∏ (1 - risk_j)   for j in window

This is the probability that AT LEAST ONE of the next `window` nodes on a
candidate path is a failure event (assuming independence as a first-order
approximation).

We also compute a turn-penalty that increases joint risk when the path
requires large heading changes between consecutive nodes.

Usage
-----
The pathfinding algo calls `joint_risk_lookahead()` during node expansion
to score candidate partial paths of length `window_size`.
"""

from __future__ import annotations
import math
import numpy as np
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from graph.node import Node3D


# ── Per-path joint risk ────────────────────────────────────────────────────

def joint_risk(
    nodes          : List["Node3D"],
    turn_penalty_w : float = 0.15,
) -> float:
    """
    Compute the joint risk for a sequence of nodes.

    P(failure in sequence) = 1 - ∏ (1 - risk_i)
    Plus a turn-angle penalty term.

    Parameters
    ----------
    nodes          : ordered list of Node3D objects (the window)
    turn_penalty_w : weight for heading-change penalty

    Returns
    -------
    Scalar in [0, 1]
    """
    if not nodes:
        return 0.0
    if len(nodes) == 1:
        return nodes[0].risk

    # ── Failure probability product ────────────────────────────────────
    survival = 1.0
    for node in nodes:
        survival *= max(0.0, 1.0 - node.risk)

    base_joint = 1.0 - survival

    # ── Turn penalty ────────────────────────────────────────────────────
    turn_penalty = _compute_turn_penalty(nodes)

    joint = min(base_joint + turn_penalty_w * turn_penalty, 1.0)
    return float(joint)


def _compute_turn_penalty(nodes: List["Node3D"]) -> float:
    """
    Compute normalised cumulative heading-change cost.

    Sharp turns on a slope are more dangerous, so we weight
    each turn by the slope of the incoming segment.

    Returns a value in [0, 1].
    """
    if len(nodes) < 3:
        return 0.0

    total_penalty = 0.0
    n_turns = len(nodes) - 2

    for i in range(1, len(nodes) - 1):
        prev, curr, nxt = nodes[i-1], nodes[i], nodes[i+1]

        # Incoming and outgoing 2D heading vectors
        in_vec  = np.array([curr.x - prev.x, curr.y - prev.y])
        out_vec = np.array([nxt.x - curr.x, nxt.y - curr.y])

        len_in  = np.linalg.norm(in_vec)
        len_out = np.linalg.norm(out_vec)

        if len_in < 1e-6 or len_out < 1e-6:
            continue

        cos_angle = np.dot(in_vec, out_vec) / (len_in * len_out)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        turn_rad  = math.acos(cos_angle)          # 0 = straight, π = U-turn

        # Slope weight: sharper slope → heavier penalty for the turn
        dz_in  = curr.z - prev.z
        horiz_in = len_in
        slope_weight = 1.0 + min(abs(dz_in) / (horiz_in + 1e-6), 1.0)

        total_penalty += (turn_rad / math.pi) * slope_weight

    # Normalise to [0, 1]
    return min(total_penalty / max(n_turns, 1), 1.0)


# ── Lookahead scorer for path expansion ────────────────────────────────────

class JointRiskScorer:
    """
    Maintains a sliding window of `window_size` nodes and scores
    candidate next-node expansions.

    Used by the pathfinding algorithm during node expansion.
    """

    def __init__(self, window_size: int = 5, turn_penalty_w: float = 0.15):
        self.window_size    = window_size
        self.turn_penalty_w = turn_penalty_w

    def score_candidate(
        self,
        path_so_far : List["Node3D"],
        candidate   : "Node3D",
    ) -> float:
        """
        Score of adding `candidate` to `path_so_far`.

        Takes the last (window_size - 1) nodes of path_so_far
        plus the candidate as the evaluation window.

        Returns joint risk in [0, 1].
        """
        window_prefix = path_so_far[-(self.window_size - 1):]
        window = window_prefix + [candidate]
        return joint_risk(window, self.turn_penalty_w)

    def path_segments(self, full_path: List["Node3D"]) -> List[dict]:
        """
        Evaluate joint risk at every position along a completed path.
        Useful for post-hoc analysis and visualisation.

        Returns list of dicts with keys: start_idx, end_idx, nodes, joint_risk
        """
        if len(full_path) < 2:
            return []

        segments = []
        for i in range(len(full_path) - self.window_size + 1):
            window = full_path[i : i + self.window_size]
            segments.append({
                "start_idx"  : i,
                "end_idx"    : i + self.window_size - 1,
                "node_ids"   : [n.node_id for n in window],
                "joint_risk" : joint_risk(window, self.turn_penalty_w),
            })

        return segments

    def max_segment_risk(self, full_path: List["Node3D"]) -> float:
        """Return the worst-case window risk along the full path."""
        segs = self.path_segments(full_path)
        if not segs:
            return joint_risk(full_path, self.turn_penalty_w)
        return max(s["joint_risk"] for s in segs)

    def mean_segment_risk(self, full_path: List["Node3D"]) -> float:
        """Return the mean window risk along the full path."""
        segs = self.path_segments(full_path)
        if not segs:
            return joint_risk(full_path, self.turn_penalty_w)
        return float(np.mean([s["joint_risk"] for s in segs]))