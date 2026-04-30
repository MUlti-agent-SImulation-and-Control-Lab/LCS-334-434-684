#!/usr/bin/env python3
"""
run_simulation_astar.py
-----------------------
Run plain A* (no risk awareness) on the outdoor simulation PCD.
Outputs saved as outdoor_A_*.
"""

import sys
from pathlib import Path

sys.path.insert(0, '.')

from lidar.simulation import load_simulation_cloud, suggest_start_goal
from lidar.preprocessor import preprocess, estimate_bounds
from terrain.feature_extractor import extract_features
from terrain.segmenter import TerrainSegmenter
from terrain.risk_estimator import RiskEstimator
from graph.builder import GraphBuilder
from pathfinding.algo import RiskAwarePlanner
from rich.console import Console
from rich.panel import Panel
import numpy as np


def main():
    console = Console()

    console.print(Panel.fit(
        "[bold yellow]Running Plain A* on Custom Simulation[/bold yellow]\n"
        "[dim]No risk awareness — distance only[/dim]",
        border_style="yellow"
    ))

    # ─────────────────────────────────────────────────────────────
    # 1. Load simulation data
    # ─────────────────────────────────────────────────────────────
    cloud_path = Path("simulation_lidar/full_map_outdoor_shifted.pcd")

    if not cloud_path.exists():
        console.print(f"[red]Error: {cloud_path} not found![/red]")
        sys.exit(1)

    console.print(f"\n[cyan]Loading: {cloud_path}[/cyan]")
    cloud = load_simulation_cloud(cloud_path)

    start_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    goal_pos  = np.array([29.0, 23.0, 0.0], dtype=np.float32)

    # ─────────────────────────────────────────────────────────────
    # 2. Preprocess
    # ─────────────────────────────────────────────────────────────
    console.print("\n[cyan]Preprocessing...[/cyan]")
    cloud_pp = preprocess(cloud, voxel_size=0.02, roi=None, nb_neighbors=0)
    bounds = estimate_bounds(cloud_pp)
    console.print(f"  ✓ {len(cloud_pp):,} points after preprocessing")
    console.print(f"  ✓ Extent: {bounds['extent_xy']:.1f} m")

    # ─────────────────────────────────────────────────────────────
    # 3. Feature extraction
    # ─────────────────────────────────────────────────────────────
    console.print("\n[cyan]Extracting features...[/cyan]")
    features = extract_features(cloud_pp, radius=0.5)
    console.print(f"  ✓ Features: {features.shape}")

    # ─────────────────────────────────────────────────────────────
    # 4. Terrain classification
    # ─────────────────────────────────────────────────────────────
    console.print("\n[cyan]Classifying terrain...[/cyan]")
    clf_weights = Path("/home/ayushman/ext_proj/ecs334/off_road_navig/models/weights/terrain_classifier.pt")
    if clf_weights.exists():
        segmenter = TerrainSegmenter(weights_path=str(clf_weights))
    else:
        segmenter = TerrainSegmenter()

    labels = segmenter.predict(features)
    from terrain.segmenter import LABEL_NAMES
    label_counts = {LABEL_NAMES[k]: int((labels==k).sum()) for k in range(5)}
    console.print(f"  ✓ Labels: " + "  ".join(f"{k}={v:,}" for k,v in label_counts.items()))

    # ─────────────────────────────────────────────────────────────
    # 5. Risk estimation
    # ─────────────────────────────────────────────────────────────
    console.print("\n[cyan]Estimating risk...[/cyan]")
    risk_weights = Path("/home/ayushman/ext_proj/ecs334/off_road_navig/models/weights/risk_predictor.pt")
    if risk_weights.exists():
        estimator = RiskEstimator(weights_path=str(risk_weights))
    else:
        estimator = RiskEstimator()

    risks = estimator.estimate(features, labels)

    # ─────────────────────────────────────────────────────────────
    # 6. Build graph
    # ─────────────────────────────────────────────────────────────
    console.print("\n[cyan]Building graph...[/cyan]")

    WATER_LABEL      = 2
    VEGETATION_LABEL = 3
    OBSTACLE_LABEL   = 1

    risks = np.zeros(len(labels), dtype=np.float32)
    risks[labels == 0]               = 0.10
    risks[labels == VEGETATION_LABEL]= 0.40
    risks[labels == OBSTACLE_LABEL]  = 0.90
    risks[labels == WATER_LABEL]     = 1.0
    risks[labels == 4]               = 0.15

    from matplotlib.path import Path as MplPath

    lake_polygon = np.array([
        [ 4,  2], [ 8,  6], [ 8, 14], [ 7, 28],
        [ 4, 29], [-2, 30], [-8, 29], [-14, 29],
        [-22, 26], [-26, 22], [-26, 12], [-22,  9],
        [-14,  7], [ 0,  2], [ 4,  2]
    ], dtype=np.float32)

    bridge_zone = np.array([
        [-3, 12], [4, 12], [4, 17], [-3, 17], [-3, 12]
    ], dtype=np.float32)

    xy           = cloud_pp[:, :2]
    lake_mask    = MplPath(lake_polygon).contains_points(xy)
    bridge_mask  = MplPath(bridge_zone).contains_points(xy)
    outside_mask = ~lake_mask & ~bridge_mask

    risks[lake_mask & ~bridge_mask] = 1.0
    risks[bridge_mask]              = 0.10

    np.random.seed(42)
    outside_risks        = np.random.beta(1.5, 5, outside_mask.sum()) * 0.4 + 0.05
    risks[outside_mask]  = outside_risks.astype(np.float32)

    builder = GraphBuilder(voxel_size=0.6, connect_radius=3.5, max_risk=0.85)
    nodes, edges = builder.build(cloud_pp, features, labels, risks)

    if not nodes:
        console.print("[bold red]CRITICAL: No nodes found![/bold red]")
        sys.exit(1)

    nodes, edges = filter_largest_component(nodes, edges)
    stats = GraphBuilder.graph_stats(nodes, edges)
    console.print(f"  ✓ Final graph: {stats['total_nodes']} nodes.")

    # ─────────────────────────────────────────────────────────────
    # 7. Find start/goal nodes
    # ─────────────────────────────────────────────────────────────
    console.print("\n[cyan]Locating start and goal...[/cyan]")
    start_node = GraphBuilder.find_nearest_node(nodes, start_pos)
    goal_node  = GraphBuilder.find_nearest_node(nodes, goal_pos)
    console.print(f"  ✓ Start → Node {start_node.node_id} at ({start_node.x:.1f}, {start_node.y:.1f}, {start_node.z:.1f})")
    console.print(f"  ✓ Goal  → Node {goal_node.node_id} at ({goal_node.x:.1f}, {goal_node.y:.1f}, {goal_node.z:.1f})")

    # ─────────────────────────────────────────────────────────────
    # 8. Plain A* (risk_lambda=0 → distance only)
    # ─────────────────────────────────────────────────────────────
    console.print("\n[cyan]Running plain A*...[/cyan]")

    planner = RiskAwarePlanner(
        window_size    = 5,
        risk_lambda    = 0.0,   # ← pure distance, no risk penalty
        max_window_risk= 0.9999 # ← no pruning
    )
    result = planner.plan(nodes, start_node.node_id, goal_node.node_id)

    if not result.found:
        console.print("\n[red]✗ No path found[/red]")
        sys.exit(1)

    console.print(Panel(result.summary(),
                        title="[bold green]A* Path Found[/bold green]",
                        border_style="green"))

    # ─────────────────────────────────────────────────────────────
    # 9. Save visualization
    # ─────────────────────────────────────────────────────────────
    console.print("\n[cyan]Generating visualization...[/cyan]")

    try:
        from ui.dashboard import Dashboard
        import matplotlib.pyplot as plt

        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        dash = Dashboard()
        fig  = dash.render(cloud_pp, features, labels, risks, nodes, edges, result)

        output_path = output_dir / "outdoor_A_dashboard.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        console.print(f"  ✓ Dashboard saved: {output_path}")
    except Exception as e:
        console.print(f"  [yellow]Visualization skipped: {e}[/yellow]")

    # ─────────────────────────────────────────────────────────────
    # 10. Save path coordinates
    # ─────────────────────────────────────────────────────────────
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    if result.found and len(result.path) > 0:
        if hasattr(result.path[0], 'x'):
            path_coords = np.array([[n.x, n.y, n.z] for n in result.path])
        else:
            path_coords = np.array([[nodes[nid].x, nodes[nid].y, nodes[nid].z]
                                     for nid in result.path])

        path_file = output_dir / "outdoor_A_path.csv"
        np.savetxt(path_file, path_coords, delimiter=',', header='x,y,z', comments='')
        console.print(f"  ✓ Path saved: {path_file}")


def filter_largest_component(nodes, edges):
    adj = {}
    for edge in edges:
        a, b = edge.src_id, edge.dst_id
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    visited    = set()
    components = []

    for start_id in nodes:
        if start_id in visited:
            continue
        comp  = set()
        queue = [start_id]
        while queue:
            n = queue.pop(0)
            if n in visited:
                continue
            visited.add(n)
            comp.add(n)
            for neighbor in adj.get(n, []):
                if neighbor not in visited:
                    queue.append(neighbor)
        components.append(comp)

    largest        = max(components, key=len)
    nodes_filtered = {nid: nodes[nid] for nid in largest}
    edges_filtered = [e for e in edges if e.src_id in largest and e.dst_id in largest]
    return nodes_filtered, edges_filtered


if __name__ == "__main__":
    main()