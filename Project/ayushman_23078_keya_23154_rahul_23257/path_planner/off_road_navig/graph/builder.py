from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from typing import Dict, List, Tuple

from .node import Node3D
from .edge import Edge
from terrain.segmenter import LABEL_OBSTACLE


class GraphBuilder:
    """
    Parameters
    ----------
    voxel_size       : cluster cell size (metres) — controls graph resolution
    connect_radius   : max 3D distance to attempt edge creation (metres)
    max_slope_deg    : slope threshold above which edge is impassable
    max_risk         : risk threshold above which node is treated as obstacle
    risk_weight      : how much risk scales traversal cost vs pure distance
    """

    def __init__(
        self,
        voxel_size    : float = 0.5,
        connect_radius: float = 4.0,
        max_slope_deg : float = 35.0,
        max_risk      : float = 0.85,
        risk_weight   : float = 0.7,
    ):
        self.voxel_size     = voxel_size
        self.connect_radius = connect_radius
        self.max_slope_deg  = max_slope_deg
        self.max_risk       = max_risk
        self.risk_weight    = risk_weight

    def build(
        self,
        cloud   : np.ndarray,   # (N, 3+) preprocessed
        features: np.ndarray,   # (N, 8)  from feature_extractor
        labels  : np.ndarray,   # (N,)    from segmenter
        risks   : np.ndarray,   # (N,)    from risk_estimator
    ) -> Tuple[Dict[int, Node3D], List[Edge]]:
    
        nodes = self._cluster_to_nodes(cloud, features, labels, risks)
        self.verify_risk_variance(nodes)   # ← diagnostic, warns if flat
        edges = self._connect_nodes(nodes)
        return nodes, edges

    # ── Step 1: Voxel clustering ─────────────────────────────────────────

    def _cluster_to_nodes(
        self,
        cloud   : np.ndarray,
        features: np.ndarray,
        labels  : np.ndarray,
        risks   : np.ndarray,
    ) -> Dict[int, Node3D]:
        xyz  = cloud[:, :3]
        mins = xyz.min(axis=0)

        # Voxel index per point (use only x,y for 2.5D clustering)
        ix = np.floor((xyz[:, 0] - mins[0]) / self.voxel_size).astype(np.int32)
        iy = np.floor((xyz[:, 1] - mins[1]) / self.voxel_size).astype(np.int32)

        max_iy = iy.max() + 1
        voxel_key = ix.astype(np.int64) * max_iy + iy.astype(np.int64)

        unique_keys = np.unique(voxel_key)
        nodes: Dict[int, Node3D] = {}

        for nid, key in enumerate(unique_keys):
            mask = voxel_key == key
            pts  = xyz[mask]
            feat = features[mask]
            lbl  = labels[mask]
            rsk  = risks[mask]

            # Dominant label in the cluster
            dominant_label = int(np.bincount(lbl).argmax())
            obstacle_fraction = (lbl == LABEL_OBSTACLE).mean()

            agg_risk = float(
                np.median(rsk) * 0.75
                + obstacle_fraction * 0.25   # only obstacle density matters
            )
            agg_risk = min(agg_risk, 1.0)

            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            cz = float(np.median(pts[:, 2]))

            node = Node3D(
                node_id     = nid,
                x           = float(cx),
                y           = float(cy),
                z           = cz,
                risk        = agg_risk,
                label       = dominant_label,
                slope_deg   = float(feat[:, 0].mean()),
                roughness   = float(feat[:, 1].mean()),
                wetness     = float(feat[:, 6].mean()),
                point_count = int(mask.sum()),
            )
            nodes[nid] = node

        return nodes

    # ── Step 2: Edge creation ────────────────────────────────────────────

    def _connect_nodes(self, nodes: Dict[int, Node3D]) -> List[Edge]:
        node_list = list(nodes.values())
        positions = np.array([[n.x, n.y, n.z] for n in node_list], dtype=np.float32)
        ids       = [n.node_id for n in node_list]

        tree = cKDTree(positions)
        pairs = tree.query_pairs(r=self.connect_radius)

        edges: List[Edge] = []

        for i, j in pairs:
            src = nodes[ids[i]]
            dst = nodes[ids[j]]

            if src.risk >= self.max_risk or dst.risk >= self.max_risk:
                continue

            edge = Edge.from_nodes(src, dst, risk_weight=self.risk_weight)

            if edge.is_passable(self.max_slope_deg, self.max_risk):
                rev_edge = Edge.from_nodes(dst, src, risk_weight=self.risk_weight)
                edges.append(edge)
                edges.append(rev_edge)
                src.neighbours.append((dst.node_id, edge.traversal_cost))
                dst.neighbours.append((src.node_id, rev_edge.traversal_cost))

        return edges

    # ── Risk variance diagnostic ─────────────────────────────────────────

    @staticmethod
    def verify_risk_variance(nodes: Dict[int, Node3D]) -> bool:
        """
        Warn if risk distribution is too compressed for the cost function
        to differentiate paths. Call this after _cluster_to_nodes.
        """
        risks     = np.array([n.risk for n in nodes.values()])
        low_frac  = (risks < 0.2).mean()
        high_frac = (risks > 0.7).mean()
        std       = risks.std()

        print(f"\n[GraphBuilder] Risk distribution — "
              f"mean:{risks.mean():.3f}  std:{std:.3f}  "
              f"min:{risks.min():.3f}  max:{risks.max():.3f}")
        print(f"               Low(<0.2): {low_frac*100:.1f}%  "
              f"High(>0.7): {high_frac*100:.1f}%")

        ok = True
        if std < 0.15:
            print("  ⚠ WARNING: std < 0.15 — risk landscape too flat, "
                  "algorithms will produce identical paths")
            ok = False
        if low_frac < 0.10:
            print("  ⚠ WARNING: <10% low-risk nodes — no safe corridors, "
                  "check risk_estimator absolute thresholds")
            ok = False
        if ok:
            print("  ✓ Risk variance looks healthy")
        return ok

    # ── Utility ──────────────────────────────────────────────────────────

    @staticmethod
    def find_nearest_node(
        nodes: Dict[int, Node3D],
        query_xyz: Tuple[float, float, float],
    ) -> Node3D:
        """Return the traversable node closest to query_xyz."""
        q = np.array(query_xyz, dtype=np.float32)
        best_node = None
        best_dist = np.inf

        for node in nodes.values():
            if node.risk >= 0.85:
                continue
            d = float(np.linalg.norm(node.position - q))
            if d < best_dist:
                best_dist = d
                best_node = node

        if best_node is None:
            best_node = min(nodes.values(), key=lambda n: np.linalg.norm(n.position - q))

        return best_node

    @staticmethod
    def graph_stats(nodes: Dict[int, Node3D], edges: List[Edge]) -> dict:
        n_trav = sum(1 for n in nodes.values() if n.risk < 0.85)
        risks  = [n.risk for n in nodes.values()]
        return {
            "total_nodes"      : len(nodes),
            "traversable_nodes": n_trav,
            "obstacle_nodes"   : len(nodes) - n_trav,
            "total_edges"      : len(edges),
            "mean_risk"        : float(np.mean(risks)),
            "max_risk"         : float(np.max(risks)),
            "std_risk"         : float(np.std(risks)),      # added
            "low_risk_frac"    : float((np.array(risks) < 0.2).mean()),  # added
        }