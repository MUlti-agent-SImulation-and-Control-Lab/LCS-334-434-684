"""
ui/dashboard.py
---------------
Visualisation Dashboard for the LiDAR Off-Road Pathfinder.

Panels
------
1. TOP-LEFT    : Bird's-eye 2D terrain heatmap (risk overlay)
2. TOP-CENTER  : 3D point cloud with terrain labels
3. TOP-RIGHT   : Traversability graph + planned path overlay
4. BOTTOM-LEFT : Joint risk profile along path (line chart)
5. BOTTOM-CENTER: Per-class terrain statistics (bar chart)
6. BOTTOM-RIGHT : Vehicle simulation (animated top-down traversal)

Usage
-----
  from ui.dashboard import Dashboard
  dash = Dashboard()
  fig  = dash.render(cloud, features, labels, risks, nodes, edges, path_result)
  dash.animate_vehicle(path_result, save_gif="path.gif")   # optional
  plt.show()
"""

from __future__ import annotations
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from graph.node import Node3D
    from graph.edge import Edge
    from pathfinding.algo import PathResult

matplotlib.rcParams.update({
    "font.family"     : "monospace",
    "axes.facecolor"  : "#0d1117",
    "figure.facecolor": "#0d1117",
    "text.color"      : "#e6edf3",
    "axes.labelcolor" : "#e6edf3",
    "xtick.color"     : "#8b949e",
    "ytick.color"     : "#8b949e",
    "axes.edgecolor"  : "#30363d",
    "grid.color"      : "#21262d",
    "axes.titlecolor" : "#e6edf3",
})

# Label palette
_LABEL_RGB = {
    0: "#3fb950",   # ground  — green
    1: "#f85149",   # obstacle — red
    2: "#58a6ff",   # water   — blue
    3: "#56d364",   # veg     — light green
    4: "#8b949e",   # unknown — grey
}
_LABEL_NAME = {0:"Ground", 1:"Obstacle", 2:"Water", 3:"Vegetation", 4:"Unknown"}

RISK_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "risk", ["#238636","#d29922","#f85149"]
)


class Dashboard:
    def __init__(self, figsize=(22, 12), dpi=110, max_pts=15_000):
        self.figsize  = figsize
        self.dpi      = dpi
        self.max_pts  = max_pts

    # ── Main render ──────────────────────────────────────────────────────

    def render(
        self,
        cloud       : np.ndarray,
        features    : np.ndarray,
        labels      : np.ndarray,
        risks       : np.ndarray,
        nodes       : Dict[int, "Node3D"],
        edges       : List["Edge"],
        path_result : Optional["PathResult"] = None,
        uncertainty : Optional[np.ndarray]   = None,   # (N,) std values
        title       : str = "LiDAR Off-Road Pathfinder — Analysis Dashboard",
    ) -> plt.Figure:

        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        fig.patch.set_facecolor("#0d1117")
        fig.suptitle(title, fontsize=14, fontweight="bold",
                     color="#e6edf3", y=0.98)

        gs = GridSpec(
            2, 3,
            figure=fig,
            hspace=0.38, wspace=0.28,
            left=0.05, right=0.97, top=0.93, bottom=0.06,
        )

        ax_bev   = fig.add_subplot(gs[0, 0])          # bird's eye
        ax_3d    = fig.add_subplot(gs[0, 1], projection="3d")  # 3D labels
        ax_graph = fig.add_subplot(gs[0, 2])          # graph + path
        ax_risk  = fig.add_subplot(gs[1, 0])          # joint risk profile
        ax_stats = fig.add_subplot(gs[1, 1])          # class stats
        ax_sim   = fig.add_subplot(gs[1, 2])          # vehicle sim

        idx = self._subsample(len(cloud), self.max_pts)

        self._panel_bird_eye(ax_bev,   cloud, risks, labels, idx, path_result)
        self._panel_3d_labels(ax_3d,   cloud, labels, idx, path_result)
        self._panel_graph(ax_graph,    nodes, edges, path_result)
        self._panel_risk_profile(ax_risk, path_result, uncertainty)
        self._panel_class_stats(ax_stats, labels, risks)
        self._panel_vehicle_sim(ax_sim,   nodes, path_result)

        return fig

    # ── Panel 1: Bird's-eye risk heatmap ─────────────────────────────────

    def _panel_bird_eye(self, ax, cloud, risks, labels, idx, path_result):
        ax.set_facecolor("#0d1117")
        ax.set_title("Bird's-Eye Risk Map", fontsize=10, pad=6)

        sc = ax.scatter(
            cloud[idx, 0], cloud[idx, 1],
            c=risks[idx], cmap=RISK_CMAP, vmin=0, vmax=1,
            s=1.2, alpha=0.75, linewidths=0, rasterized=True,
        )
        plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02,
                     label="Risk score").ax.yaxis.set_tick_params(color="#8b949e")

        # Obstacle overlay
        obs_mask = (labels[idx] == 1)
        if obs_mask.any():
            ax.scatter(cloud[idx][obs_mask, 0], cloud[idx][obs_mask, 1],
                       c="#f85149", s=2.5, alpha=0.5, marker="x", linewidths=0.8,
                       label="Obstacle", rasterized=True)

        # Path overlay
        if path_result and path_result.found:
            px = [n.x for n in path_result.path]
            py = [n.y for n in path_result.path]
            ax.plot(px, py, color="#58a6ff", linewidth=2.2, zorder=5, label="Path")
            ax.scatter([px[0]], [py[0]], color="#3fb950", s=80, zorder=6, marker="^")
            ax.scatter([px[-1]], [py[-1]], color="#f85149", s=80, zorder=6, marker="s")

        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.set_aspect("equal")
        _style_axes(ax)

    # ── Panel 2: 3D terrain labels ────────────────────────────────────────

    def _panel_3d_labels(self, ax, cloud, labels, idx, path_result):
        ax.set_facecolor("#0d1117")
        ax.set_title("3D Terrain Segmentation", fontsize=10, pad=6)

        for lbl_id, color in _LABEL_RGB.items():
            mask = labels[idx] == lbl_id
            if not mask.any():
                continue
            pts = cloud[idx][mask]
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       c=color, s=1.0, alpha=0.7, label=_LABEL_NAME[lbl_id],
                       linewidths=0, rasterized=True)

        if path_result and path_result.found:
            px = [n.x for n in path_result.path]
            py = [n.y for n in path_result.path]
            pz = [n.z for n in path_result.path]
            ax.plot(px, py, pz, color="#58a6ff", linewidth=2.5, zorder=5)

        legend = ax.legend(loc="upper left", markerscale=5, fontsize=7,
                           facecolor="#161b22", edgecolor="#30363d",
                           labelcolor="#e6edf3")
        ax.set_xlabel("X", fontsize=7); ax.set_ylabel("Y", fontsize=7)
        ax.set_zlabel("Z", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False

    # ── Panel 3: Graph + path ─────────────────────────────────────────────

    def _panel_graph(self, ax, nodes, edges, path_result):
        ax.set_facecolor("#0d1117")
        ax.set_title("Traversability Graph + Path", fontsize=10, pad=6)

        node_list = list(nodes.values())
        xs = [n.x for n in node_list]
        ys = [n.y for n in node_list]
        rs = [n.risk for n in node_list]

        # Draw edges (subsample)
        node_map   = {n.node_id: n for n in node_list}
        drawn_pairs = set()
        for e in edges[:3000]:
            key = (min(e.src_id, e.dst_id), max(e.src_id, e.dst_id))
            if key in drawn_pairs:
                continue
            drawn_pairs.add(key)
            src = node_map.get(e.src_id); dst = node_map.get(e.dst_id)
            if src and dst:
                ax.plot([src.x, dst.x], [src.y, dst.y],
                        color="#21262d", linewidth=0.4, alpha=0.8, zorder=1)

        sc = ax.scatter(xs, ys, c=rs, cmap=RISK_CMAP, vmin=0, vmax=1,
                        s=12, alpha=0.9, zorder=3, linewidths=0)
        plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, label="Node risk")

        if path_result and path_result.found:
            px = [n.x for n in path_result.path]
            py = [n.y for n in path_result.path]
            ax.plot(px, py, color="#58a6ff", linewidth=2.5, zorder=5)
            ax.scatter(px, py, c=[n.risk for n in path_result.path],
                       cmap=RISK_CMAP, vmin=0, vmax=1, s=35, zorder=6,
                       edgecolors="#e6edf3", linewidths=0.5)
            ax.scatter([px[0]], [py[0]], color="#3fb950", s=120, zorder=7,
                       marker="^", label="Start")
            ax.scatter([px[-1]], [py[-1]], color="#f85149", s=120, zorder=7,
                       marker="s", label="Goal")
            ax.legend(fontsize=8, facecolor="#161b22", edgecolor="#30363d",
                      labelcolor="#e6edf3")

        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.set_aspect("equal")
        _style_axes(ax)

    # ── Panel 4: Joint risk profile ───────────────────────────────────────

    def _panel_risk_profile(self, ax, path_result, uncertainty=None):
        ax.set_facecolor("#0d1117")
        ax.set_title("Joint Risk Profile (Sliding Window)", fontsize=10, pad=6)

        if not path_result or not path_result.found or not path_result.segment_risks:
            ax.text(0.5, 0.5, "No path", ha="center", va="center",
                    transform=ax.transAxes, color="#8b949e", fontsize=12)
            _style_axes(ax); return

        segs  = path_result.segment_risks
        pos   = [s["start_idx"] for s in segs]
        jrisk = [s["joint_risk"] for s in segs]

        ax.fill_between(pos, jrisk, alpha=0.25, color="#f85149")
        ax.plot(pos, jrisk, color="#f85149", linewidth=1.8, label="Joint risk")

        # Per-node individual risk
        node_risks = [n.risk for n in path_result.path]
        ax.step(range(len(node_risks)), node_risks,
                color="#d29922", linewidth=1.2, alpha=0.7, where="mid",
                label="Node risk")

        # Threshold lines
        ax.axhline(0.5,  color="#d29922", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axhline(0.75, color="#f85149", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.text(pos[-1] * 0.98, 0.51,  "Moderate",  color="#d29922", fontsize=7, ha="right")
        ax.text(pos[-1] * 0.98, 0.76,  "High",      color="#f85149", fontsize=7, ha="right")

        # Stats box
        stats_text = (
            f"Max: {path_result.max_window_risk:.3f}\n"
            f"Mean: {path_result.mean_window_risk:.3f}\n"
            f"Dist: {path_result.total_dist:.1f}m"
        )
        ax.text(0.02, 0.97, stats_text, transform=ax.transAxes,
                fontsize=8, va="top", color="#e6edf3",
                bbox=dict(facecolor="#161b22", edgecolor="#30363d",
                          boxstyle="round,pad=0.4"))

        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Node index"); ax.set_ylabel("Risk P(≥1 failure)")
        ax.legend(fontsize=8, facecolor="#161b22", edgecolor="#30363d",
                  labelcolor="#e6edf3", loc="lower right")
        _style_axes(ax)

    # ── Panel 5: Class statistics ─────────────────────────────────────────

    def _panel_class_stats(self, ax, labels, risks):
        ax.set_facecolor("#0d1117")
        ax.set_title("Terrain Class Statistics", fontsize=10, pad=6)

        class_ids = list(range(5))
        counts     = [int((labels == c).sum()) for c in class_ids]
        mean_risks = [
            float(risks[labels == c].mean()) if (labels == c).sum() > 0 else 0.0
            for c in class_ids
        ]
        names  = [_LABEL_NAME[c] for c in class_ids]
        colors = [_LABEL_RGB[c]  for c in class_ids]

        x = np.arange(len(class_ids))
        w = 0.4

        bars1 = ax.bar(x - w/2, counts, w, label="Point count",
                       color=colors, alpha=0.8, edgecolor="#30363d")

        ax2 = ax.twinx()
        ax2.set_facecolor("#0d1117")
        bars2 = ax2.bar(x + w/2, mean_risks, w, label="Mean risk",
                        color=colors, alpha=0.4, edgecolor="#30363d",
                        hatch="///")
        ax2.set_ylim(0, 1.2)
        ax2.set_ylabel("Mean risk score", color="#8b949e", fontsize=8)
        ax2.tick_params(colors="#8b949e", labelsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8, rotation=15)
        ax.set_ylabel("Point count", fontsize=8)

        # Count labels on bars
        for bar, cnt in zip(bars1, counts):
            if cnt > 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() * 1.01,
                        f"{cnt:,}", ha="center", fontsize=6.5, color="#e6edf3")

        legend_patches = [
            mpatches.Patch(color=colors[i], label=names[i]) for i in range(5)
        ]
        ax.legend(handles=legend_patches, fontsize=7,
                  facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3",
                  loc="upper right")
        _style_axes(ax)
        _style_axes(ax2)

    # ── Panel 6: Vehicle simulation ───────────────────────────────────────

    def _panel_vehicle_sim(self, ax, nodes, path_result):
        ax.set_facecolor("#0d1117")
        ax.set_title("Planned Route (Static Preview)", fontsize=10, pad=6)

        if not nodes:
            ax.text(0.5, 0.5, "No graph", ha="center", va="center",
                    transform=ax.transAxes, color="#8b949e"); return

        # Background: node risk field
        xs = np.array([n.x for n in nodes.values()])
        ys = np.array([n.y for n in nodes.values()])
        rs = np.array([n.risk for n in nodes.values()])
        ax.scatter(xs, ys, c=rs, cmap=RISK_CMAP, vmin=0, vmax=1,
                   s=8, alpha=0.5, linewidths=0, rasterized=True)

        if path_result and path_result.found:
            from pathfinding.path_smoother import smooth_path
            smoothed = smooth_path(path_result.path, points_per_m=3.0)
            sx, sy   = smoothed["x"], smoothed["y"]
            curv     = smoothed["curvature"]

            # Path coloured by curvature (sharp turns = yellow)
            for i in range(len(sx) - 1):
                c_val = float(np.clip(curv[i] * 20, 0, 1))
                col   = plt.cm.YlOrRd(c_val)
                ax.plot([sx[i], sx[i+1]], [sy[i], sy[i+1]],
                        color=col, linewidth=2.0, zorder=5)

            # Vehicle icon at start
            self._draw_vehicle(ax, sx[0], sy[0],
                                heading=np.arctan2(sy[1]-sy[0], sx[1]-sx[0]))

            # Mark high-curvature waypoints
            hi_curv_mask = curv > np.percentile(curv, 85)
            if hi_curv_mask.any():
                ax.scatter(sx[hi_curv_mask], sy[hi_curv_mask],
                           color="#d29922", s=18, zorder=6, marker="D",
                           label="Sharp turn", linewidths=0)

            ax.scatter([sx[0]], [sy[0]], color="#3fb950", s=120, zorder=7, marker="^")
            ax.scatter([sx[-1]], [sy[-1]], color="#f85149", s=120, zorder=7, marker="s")

            # Curvature colorbar legend
            sm = plt.cm.ScalarMappable(cmap="YlOrRd",
                                        norm=Normalize(vmin=0, vmax=1))
            sm.set_array([])
            cb = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
            cb.set_label("Turn sharpness", fontsize=7)
            cb.ax.yaxis.set_tick_params(color="#8b949e")

            stats = (
                f"Waypoints: {len(sx)}\n"
                f"Length: {smoothed['arc_length'][-1]:.1f} m\n"
                f"Max curv: {curv.max():.3f}"
            )
            ax.text(0.02, 0.97, stats, transform=ax.transAxes, fontsize=8,
                    va="top", color="#e6edf3",
                    bbox=dict(facecolor="#161b22", edgecolor="#30363d",
                              boxstyle="round,pad=0.4"))

        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.set_aspect("equal")
        _style_axes(ax)

    @staticmethod
    def _draw_vehicle(ax, x, y, heading=0.0, size=1.5):
        """Draw a simple triangle representing the vehicle."""
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        # Front, rear-left, rear-right in vehicle frame
        pts = np.array([
            [size,        0],
            [-size * 0.6,  size * 0.5],
            [-size * 0.6, -size * 0.5],
        ])
        rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        world_pts = (rot @ pts.T).T + np.array([x, y])
        tri = plt.Polygon(world_pts, color="#58a6ff", alpha=0.9, zorder=8)
        ax.add_patch(tri)

    # ── Animation ────────────────────────────────────────────────────────

    def animate_vehicle(
        self,
        nodes       : Dict[int, "Node3D"],
        path_result : "PathResult",
        save_path   : str | None = None,
        interval_ms : int = 80,
        trail_len   : int = 20,
    ) -> animation.FuncAnimation:
        """
        Animate the vehicle travelling along the smoothed path.
        Save as GIF if save_path is given.
        """
        from pathfinding.path_smoother import smooth_path

        smoothed = smooth_path(path_result.path, points_per_m=3.0)
        sx, sy, curv = smoothed["x"], smoothed["y"], smoothed["curvature"]
        n_frames = len(sx)

        fig, ax = plt.subplots(figsize=(9, 8), facecolor="#0d1117")
        ax.set_facecolor("#0d1117")

        # Background nodes
        xs = np.array([n.x for n in nodes.values()])
        ys = np.array([n.y for n in nodes.values()])
        rs = np.array([n.risk for n in nodes.values()])
        ax.scatter(xs, ys, c=rs, cmap=RISK_CMAP, vmin=0, vmax=1,
                   s=6, alpha=0.4, linewidths=0)
        ax.plot(sx, sy, color="#30363d", linewidth=1.5, zorder=1)
        ax.scatter([sx[0]], [sy[0]],  color="#3fb950", s=100, zorder=5, marker="^")
        ax.scatter([sx[-1]], [sy[-1]], color="#f85149", s=100, zorder=5, marker="s")
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.set_title("Vehicle Simulation", color="#e6edf3")
        _style_axes(ax)

        trail_line, = ax.plot([], [], color="#58a6ff", linewidth=2.0, alpha=0.8, zorder=3)
        vehicle_tri = plt.Polygon([[0,0],[0,0],[0,0]], color="#58a6ff", zorder=6)
        ax.add_patch(vehicle_tri)
        risk_text = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                             fontsize=9, va="top", color="#e6edf3",
                             bbox=dict(facecolor="#161b22", edgecolor="#30363d",
                                       boxstyle="round,pad=0.4"))

        def _update(frame):
            i = frame
            # Trail
            t_start = max(0, i - trail_len)
            trail_line.set_data(sx[t_start:i+1], sy[t_start:i+1])

            # Vehicle
            heading = np.arctan2(sy[min(i+1, n_frames-1)] - sy[i],
                                  sx[min(i+1, n_frames-1)] - sx[i])
            cos_h, sin_h = np.cos(heading), np.sin(heading)
            size = 1.2
            pts  = np.array([[size, 0], [-size*0.6, size*0.5], [-size*0.6, -size*0.5]])
            rot  = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
            wp   = (rot @ pts.T).T + np.array([sx[i], sy[i]])
            vehicle_tri.set_xy(wp)

            # Risk text
            cur_risk = float(np.clip(curv[i] * 20, 0, 1))
            risk_text.set_text(
                f"Progress: {100*i//n_frames:>3d}%\n"
                f"Pos: ({sx[i]:.1f}, {sy[i]:.1f})\n"
                f"Turn sharpness: {curv[i]:.3f}"
            )
            return trail_line, vehicle_tri, risk_text

        anim = animation.FuncAnimation(
            fig, _update, frames=n_frames,
            interval=interval_ms, blit=True,
        )

        if save_path:
            writer = animation.PillowWriter(fps=1000//interval_ms)
            anim.save(save_path, writer=writer, dpi=90)
            print(f"Animation saved → {save_path}")

        return anim


# ── Helper ────────────────────────────────────────────────────────────────

def _style_axes(ax):
    ax.tick_params(labelsize=7, colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.grid(True, color="#21262d", linewidth=0.5, alpha=0.6)

Dashboard._subsample = staticmethod(
    lambda n, max_n: (np.arange(n) if n <= max_n
                      else np.random.choice(n, max_n, replace=False))
)