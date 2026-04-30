"""
lidar/visualizer.py
--------------------
Visualisation utilities for:
  - Raw point clouds (coloured by z or intensity)
  - Terrain-labelled clouds
  - Risk heatmap overlaid on cloud
  - 3D graph (nodes + edges)
  - Planned path overlaid on terrain
  - Per-segment joint risk heatmap

Uses matplotlib for maximum portability; Open3D for interactive mode.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D   # noqa
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from graph.node import Node3D
    from pathfinding.algo import PathResult

# ── Label colour map ──────────────────────────────────────────────────────
_LABEL_RGB = {
    0: (0.40, 0.75, 0.35),   # ground  - green
    1: (0.85, 0.20, 0.10),   # obstacle - red
    2: (0.10, 0.45, 0.90),   # water   - blue
    3: (0.10, 0.55, 0.20),   # veg     - dark green
    4: (0.70, 0.70, 0.70),   # unknown - grey
}
_LABEL_NAME = {0:"ground",1:"obstacle",2:"water",3:"vegetation",4:"unknown"}


def plot_cloud_risk(
    cloud  : np.ndarray,
    risks  : np.ndarray,
    labels : Optional[np.ndarray] = None,
    title  : str = "LiDAR Point Cloud — Risk Heatmap",
    max_pts: int = 20_000,
    ax     = None,
) -> plt.Figure:
    """
    Scatter plot coloured by risk (0=green → 1=red).
    If labels provided, obstacle points are shown separately.
    """
    fig = None
    if ax is None:
        fig = plt.figure(figsize=(12, 8))
        ax  = fig.add_subplot(111, projection="3d")

    # Subsample for speed
    idx = _subsample_indices(len(cloud), max_pts)
    x, y, z = cloud[idx, 0], cloud[idx, 1], cloud[idx, 2]
    r = risks[idx]

    cmap = cm.RdYlGn_r   # low risk=green, high risk=red
    sc = ax.scatter(x, y, z, c=r, cmap=cmap, vmin=0, vmax=1,
                    s=1.5, alpha=0.7, linewidths=0)

    cb = plt.colorbar(sc, ax=ax, shrink=0.5, pad=0.02)
    cb.set_label("Risk score")

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title(title)

    if fig:
        plt.tight_layout()
    return fig or ax.get_figure()


def plot_terrain_labels(
    cloud  : np.ndarray,
    labels : np.ndarray,
    title  : str = "Terrain Segmentation",
    max_pts: int = 20_000,
) -> plt.Figure:
    """Colour-coded scatter by terrain class label."""
    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_subplot(111, projection="3d")

    for label_id, (r, g, b) in _LABEL_RGB.items():
        mask = labels == label_id
        if not mask.any():
            continue
        idx  = _subsample_indices(mask.sum(), max_pts // 5)
        pts  = cloud[mask][idx]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   color=(r, g, b), s=1.5, alpha=0.7,
                   label=_LABEL_NAME[label_id], linewidths=0)

    ax.legend(markerscale=6, loc="upper left")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_graph(
    nodes  : Dict[int, "Node3D"],
    edges,
    path_result: Optional["PathResult"] = None,
    title  : str = "Traversability Graph",
    max_edges: int = 3000,
) -> plt.Figure:
    """
    Draw the 3D graph.  Nodes coloured by risk.
    Path highlighted in blue if provided.
    """
    fig = plt.figure(figsize=(13, 9))
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_title(title)

    node_list = list(nodes.values())
    xs = np.array([n.x for n in node_list])
    ys = np.array([n.y for n in node_list])
    zs = np.array([n.z for n in node_list])
    rs = np.array([n.risk for n in node_list])

    sc = ax.scatter(xs, ys, zs, c=rs, cmap=cm.RdYlGn_r,
                    vmin=0, vmax=1, s=15, alpha=0.85, zorder=3)
    plt.colorbar(sc, ax=ax, shrink=0.45, pad=0.02).set_label("Node risk")

    # Draw edges (subsample)
    node_map = {n.node_id: n for n in node_list}
    unique_pairs = set()
    drawn = 0
    for e in edges:
        key = (min(e.src_id, e.dst_id), max(e.src_id, e.dst_id))
        if key in unique_pairs or drawn >= max_edges:
            continue
        unique_pairs.add(key)
        src = node_map.get(e.src_id)
        dst = node_map.get(e.dst_id)
        if src and dst:
            ax.plot(
                [src.x, dst.x], [src.y, dst.y], [src.z, dst.z],
                color="grey", alpha=0.15, linewidth=0.5,
            )
            drawn += 1

    # Overlay planned path
    if path_result and path_result.found:
        px = [n.x for n in path_result.path]
        py = [n.y for n in path_result.path]
        pz = [n.z for n in path_result.path]
        ax.plot(px, py, pz, color="royalblue", linewidth=2.5,
                zorder=5, label="Planned path")
        ax.scatter([px[0]], [py[0]], [pz[0]], color="lime",
                   s=100, zorder=6, label="Start")
        ax.scatter([px[-1]], [py[-1]], [pz[-1]], color="red",
                   s=100, zorder=6, label="Goal")
        ax.legend()

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    plt.tight_layout()
    return fig


def plot_joint_risk_profile(
    path_result: "PathResult",
    title      : str = "Joint Risk Profile Along Path",
) -> plt.Figure:
    """
    Line chart of joint risk at each sliding window position along the path.
    Highlights dangerous segments in red.
    """
    segs = path_result.segment_risks
    if not segs:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No segment data", ha="center", transform=ax.transAxes)
        return fig

    positions  = [s["start_idx"] for s in segs]
    risk_vals  = [s["joint_risk"] for s in segs]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(positions, risk_vals, alpha=0.25, color="tomato")
    ax.plot(positions, risk_vals, color="tomato", linewidth=1.8, label="Joint risk")
    ax.axhline(0.5, color="orange", linestyle="--", linewidth=1, label="Moderate (0.5)")
    ax.axhline(0.75, color="red",   linestyle="--", linewidth=1, label="High (0.75)")
    ax.set_xlabel("Window start node index")
    ax.set_ylabel("Joint risk P(≥1 failure)")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig


def show_all(
    cloud      : np.ndarray,
    risks      : np.ndarray,
    labels     : np.ndarray,
    nodes      : Dict[int, "Node3D"],
    edges      ,
    path_result: Optional["PathResult"] = None,
):
    """Convenience: render all standard plots and show."""
    plot_terrain_labels(cloud, labels)
    plot_cloud_risk(cloud, risks)
    plot_graph(nodes, edges, path_result)
    if path_result and path_result.found:
        plot_joint_risk_profile(path_result)
    plt.show()


# ── Helpers ───────────────────────────────────────────────────────────────

def _subsample_indices(n: int, max_n: int) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    return np.random.choice(n, max_n, replace=False)