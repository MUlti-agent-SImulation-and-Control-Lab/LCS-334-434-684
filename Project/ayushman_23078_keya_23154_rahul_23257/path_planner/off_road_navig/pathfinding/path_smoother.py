"""
pathfinding/path_smoother.py
-----------------------------
Convert a list of Node3D waypoints into a smooth 3D trajectory using
cubic spline interpolation.

Also computes per-waypoint curvature, which can be used downstream for
speed planning (tighter curves → lower speed).
"""

from __future__ import annotations
import numpy as np
from scipy.interpolate import CubicSpline
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from graph.node import Node3D


def smooth_path(
    path_nodes     : List["Node3D"],
    points_per_m   : float = 2.0,
    min_points     : int   = 50,
) -> dict:
    """
    Fit a cubic spline through the path waypoints and resample
    at uniform arc-length intervals.

    Parameters
    ----------
    path_nodes   : ordered list of Node3D
    points_per_m : resampling density along the path
    min_points   : minimum output waypoints

    Returns
    -------
    dict with keys:
        'x', 'y', 'z'         : (M,) smoothed waypoint coordinates
        'curvature'            : (M,) absolute curvature κ
        'arc_length'           : (M,) cumulative arc length (m)
        'raw_waypoints'        : original (N, 3) array
    """
    if len(path_nodes) < 2:
        pos = np.array([[n.x, n.y, n.z] for n in path_nodes]) if path_nodes else np.zeros((1, 3))
        return {
            "x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2],
            "curvature": np.zeros(len(pos)),
            "arc_length": np.zeros(len(pos)),
            "raw_waypoints": pos,
        }

    raw = np.array([[n.x, n.y, n.z] for n in path_nodes], dtype=np.float64)

    # ── Parameterise by cumulative arc length ──────────────────────────
    diffs = np.diff(raw, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    t = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = t[-1]

    if total_length < 1e-6:
        return {
            "x": raw[:, 0], "y": raw[:, 1], "z": raw[:, 2],
            "curvature": np.zeros(len(raw)),
            "arc_length": t,
            "raw_waypoints": raw,
        }

    # ── Fit cubic splines ──────────────────────────────────────────────
    cs_x = CubicSpline(t, raw[:, 0])
    cs_y = CubicSpline(t, raw[:, 1])
    cs_z = CubicSpline(t, raw[:, 2])

    # ── Resample uniformly ─────────────────────────────────────────────
    n_out = max(min_points, int(total_length * points_per_m))
    t_new = np.linspace(0, total_length, n_out)

    x_new = cs_x(t_new)
    y_new = cs_y(t_new)
    z_new = cs_z(t_new)

    # ── Curvature κ = |r' × r''| / |r'|³ ────────────────────────────
    dx  = cs_x(t_new, 1);  dy  = cs_y(t_new, 1);  dz  = cs_z(t_new, 1)
    ddx = cs_x(t_new, 2);  ddy = cs_y(t_new, 2);  ddz = cs_z(t_new, 2)

    cross = np.column_stack([
        dy * ddz - dz * ddy,
        dz * ddx - dx * ddz,
        dx * ddy - dy * ddx,
    ])
    cross_mag = np.linalg.norm(cross, axis=1)
    speed_mag = np.linalg.norm(np.column_stack([dx, dy, dz]), axis=1)
    curvature  = cross_mag / (speed_mag ** 3 + 1e-12)

    return {
        "x"             : x_new.astype(np.float32),
        "y"             : y_new.astype(np.float32),
        "z"             : z_new.astype(np.float32),
        "curvature"     : curvature.astype(np.float32),
        "arc_length"    : t_new.astype(np.float32),
        "raw_waypoints" : raw,
    }