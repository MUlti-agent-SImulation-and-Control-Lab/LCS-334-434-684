"""
main.py
-------
CLI entry point for the 3D LiDAR off-road pathfinder.

Usage
-----
# Use a real point cloud file:
python main.py --cloud path/to/cloud.las --start 5 10 0 --goal 45 40 0

# Use synthetic terrain (no file needed):
python main.py --synthetic --start 2 2 0 --goal 48 48 0 --window 6

Full options:
  --cloud PATH        path to .las/.pcd/.ply/.npy file
  --synthetic         generate synthetic test terrain
  --start X Y Z       start position in world coords
  --goal  X Y Z       goal  position in world coords
  --window N          joint-risk window size (default 5)
  --voxel FLOAT       voxel size for graph construction (default 1.5m)
  --connect FLOAT     edge connection radius (default 4.0m)
  --max-risk FLOAT    maximum window risk threshold (default 0.80)
  --weights-clf PATH  terrain classifier weights (optional)
  --weights-risk PATH risk estimator weights (optional)
  --no-smooth         skip path smoothing
  --save-graph PATH   save graph to .npz for reuse
  --load-graph PATH   load prebuilt graph from .npz
"""

from __future__ import annotations
import argparse
import time
import sys
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="3D LiDAR Off-Road Pathfinder with Joint Risk",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--cloud",     type=str, help="Path to point cloud file")
    src.add_argument("--synthetic", action="store_true", help="Use synthetic terrain")

    p.add_argument("--start",    nargs=3, type=float, metavar=("X","Y","Z"),
                   required=True, help="Start position")
    p.add_argument("--goal",     nargs=3, type=float, metavar=("X","Y","Z"),
                   required=True, help="Goal position")
    p.add_argument("--window",   type=int,   default=3,
                   help="Joint-risk lookahead window (nodes)")
    p.add_argument("--voxel",    type=float, default=1.5,
                   help="Voxel size for graph nodes (m)")
    p.add_argument("--connect",  type=float, default=7.0,
                   help="Node connection radius (m)")
    p.add_argument("--max-risk", type=float, default=0.97,
                   help="Maximum allowed joint risk per window (joint P across window_size nodes)")
    p.add_argument("--weights-clf",  type=str, default=None)
    p.add_argument("--weights-risk", type=str, default=None)
    p.add_argument("--no-smooth",    action="store_true")
    p.add_argument("--save-graph",   type=str, default=None)
    p.add_argument("--load-graph",   type=str, default=None)
    p.add_argument("--alternatives", type=int, default=1,
                   help="Number of alternative paths to compute (1=best only)")
    p.add_argument(
        "--visualize",
        action="store_true",
        help="Open 6-panel matplotlib dashboard after planning"
    )
    p.add_argument("--n-synthetic",  type=int, default=50_000,
                   help="Points in synthetic terrain")
    return p


def run(args) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

    console = Console()
    console.print(Panel.fit(
        "[bold cyan]3D LiDAR Off-Road Pathfinder[/bold cyan]\n"
        "[dim]Joint Probability Risk · Terrain Segmentation · 3D A*[/dim]",
        border_style="cyan",
    ))

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(), console=console,
    ) as prog:

        # ── 1. Load or generate point cloud ─────────────────────────
        t = prog.add_task("Loading point cloud...", total=None)
        t0 = time.time()

        if args.synthetic:
            from lidar.loader import generate_synthetic_terrain
            cloud = generate_synthetic_terrain(n_points=args.n_synthetic)
            console.print(f"  [green]✓[/green] Synthetic terrain: {len(cloud):,} points")
        else:
            from lidar.loader import load_point_cloud
            cloud = load_point_cloud(args.cloud)
            console.print(f"  [green]✓[/green] Loaded {len(cloud):,} points from {args.cloud}")

        prog.update(t, description=f"Preprocessing... ({len(cloud):,} pts)")

        # ── 2. Preprocess ────────────────────────────────────────────
        from lidar.preprocessor import preprocess, estimate_bounds
        cloud_pp = preprocess(cloud, voxel_size=0.25)
        bounds   = estimate_bounds(cloud_pp)
        console.print(
            f"  [green]✓[/green] After preprocessing: {len(cloud_pp):,} points  |  "
            f"Extent {bounds['extent_xy']:.1f} m"
        )

        # ── 3. Feature extraction ────────────────────────────────────
        prog.update(t, description="Extracting terrain features...")
        from terrain.feature_extractor import extract_features
        features = extract_features(cloud_pp, radius=1.0)
        console.print(f"  [green]✓[/green] Features extracted ({features.shape[1]} dims per point)")

        # ── 4. Terrain segmentation ──────────────────────────────────
        prog.update(t, description="Segmenting terrain (ground/rock/water/veg)...")
        from terrain.segmenter import TerrainSegmenter, LABEL_NAMES
        segmenter = TerrainSegmenter(weights_path=args.weights_clf)
        labels = segmenter.predict(features)

        label_counts = {LABEL_NAMES[k]: int((labels==k).sum()) for k in range(5)}
        console.print(f"  [green]✓[/green] Terrain labels: " +
                      "  ".join(f"{k}={v:,}" for k,v in label_counts.items()))

        # ── 5. Risk estimation ───────────────────────────────────────
        prog.update(t, description="Estimating per-point risk...")
        from terrain.risk_estimator import RiskEstimator
        estimator = RiskEstimator(weights_path=args.weights_risk)
        risks = estimator.estimate(features, labels)
        console.print(
            f"  [green]✓[/green] Risk scores: "
            f"mean={risks.mean():.3f}  max={risks.max():.3f}  "
            f"high-risk(>0.7)={(risks>0.7).sum():,}"
        )

        # ── 6. Build / load graph ────────────────────────────────────
        if args.load_graph and Path(args.load_graph).exists():
            prog.update(t, description="Loading prebuilt graph...")
            nodes, edges = _load_graph(args.load_graph)
            console.print(f"  [green]✓[/green] Graph loaded from {args.load_graph}")
        else:
            prog.update(t, description="Building traversability graph...")
            from graph.builder import GraphBuilder
            builder = GraphBuilder(
                voxel_size    = args.voxel,
                connect_radius= args.connect,
                max_risk      = args.max_risk,
            )
            nodes, edges = builder.build(cloud_pp, features, labels, risks)
            stats = GraphBuilder.graph_stats(nodes, edges)
            console.print(
                f"  [green]✓[/green] Graph: {stats['total_nodes']} nodes "
                f"({stats['traversable_nodes']} traversable, "
                f"{stats['obstacle_nodes']} obstacles)  "
                f"{stats['total_edges']} edges"
            )

            if args.save_graph:
                _save_graph(nodes, edges, args.save_graph)
                console.print(f"  [green]✓[/green] Graph saved → {args.save_graph}")
        
        # After the graph building section, add:

        # ── Check connectivity ────────────────────────────────────────
        prog.update(t, description="Checking graph connectivity...")

        # Build adjacency list
        adj = {}
        for edge in edges:
            a, b = edge.src_id, edge.dst_id
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)

        # Find largest component using BFS
        def find_largest_component():
            visited = set()
            largest = set()
            
            for start_id in nodes:
                if start_id in visited:
                    continue
                # BFS from this node
                comp = set()
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
                if len(comp) > len(largest):
                    largest = comp
            return largest

        largest_comp = find_largest_component()
        console.print(
            f"  [green]✓[/green] Largest connected component: "
            f"{len(largest_comp)} / {len(nodes)} nodes "
            f"({100*len(largest_comp)/len(nodes):.1f}%)"
        )

        # Filter nodes to only the largest component
        nodes = {nid: nodes[nid] for nid in largest_comp}
        edges = [e for e in edges if e.src_id in largest_comp and e.dst_id in largest_comp]

        console.print(
            f"  [yellow]![/yellow] Using only largest component for pathfinding"
        )
        # ── 7. Find nearest start / goal nodes ───────────────────────
        prog.update(t, description="Locating start and goal nodes...")
        from graph.builder import GraphBuilder as GB
        start_node = GB.find_nearest_node(nodes, tuple(args.start))
        goal_node  = GB.find_nearest_node(nodes, tuple(args.goal))
        console.print(
            f"  [green]✓[/green] Start → Node {start_node.node_id}  "
            f"({start_node.x:.1f}, {start_node.y:.1f}, {start_node.z:.1f})  "
            f"risk={start_node.risk:.2f}"
        )
        console.print(
            f"  [green]✓[/green] Goal  → Node {goal_node.node_id}  "
            f"({goal_node.x:.1f}, {goal_node.y:.1f}, {goal_node.z:.1f})  "
            f"risk={goal_node.risk:.2f}"
        )

        # ── 8. Pathfinding ───────────────────────────────────────────
        prog.update(t, description=f"Running A* (window={args.window})...")
        t_plan = time.time()

        if args.alternatives > 1:
            from pathfinding.algo import plan_with_alternatives
            results = plan_with_alternatives(
                nodes, start_node.node_id, goal_node.node_id,
                window_size=args.window, n_alternatives=args.alternatives,
            )
        else:
            from pathfinding.algo import RiskAwarePlanner
            planner = RiskAwarePlanner(
                window_size    = args.window,
                max_window_risk= args.max_risk,
            )
            result  = planner.plan(nodes, start_node.node_id, goal_node.node_id)
            results = [result] if result.found else []

        plan_time = time.time() - t_plan
        prog.update(t, description="Done.", completed=True)

    # ── 9. Report results ────────────────────────────────────────────────
    console.print()
    if not results:
        console.print("[bold red]✗ No path found.[/bold red]  "
                      "Try relaxing --max-risk or check start/goal positions.")
        sys.exit(1)

    for idx, res in enumerate(results):
        label = f"Path {idx+1}" if len(results) > 1 else "Best Path"
        console.print(Panel(
            res.summary(),
            title=f"[bold green]{label}[/bold green]",
            border_style="green",
        ))

    best = results[0]

    # Insert this after 'best = results[0]'
    if best.found:
        # 1. Extract the (x, y, z) from each node in the path
        path_coords = np.array([[n.x, n.y, n.z] for n in best.path])
        
        # 2. Save to a CSV for your i-SENS Lab report
        output_csv = "final_path_coords.csv"
        np.savetxt(output_csv, path_coords, delimiter=",", 
                header="x,y,z", comments='')
        
        console.print(f"\n[bold green]✓[/bold green] Path Coordinates Saved: [cyan]{output_csv}[/cyan]")
        
        # 3. Print the first few for immediate verification
        console.print("[dim]First 3 Waypoints (World Coords):[/dim]")
        for i in range(min(3, len(path_coords))):
            c = path_coords[i]
            console.print(f"  {i}: x={c[0]:.2f}, y={c[1]:.2f}, z={c[2]:.2f}")

    # ── 10. Path smoothing ───────────────────────────────────────────────
    if not args.no_smooth and best.found:
        from pathfinding.path_smoother import smooth_path
        smoothed = smooth_path(best.path)
        # We will use the original best.path nodes for the 3D visualizer
        # but you could swap this with smoothed data if preferred.

    # ── 11. Custom Visualization ────────────────────────────────────────
    if args.visualize and best.found:
        # We use cloud_pp (preprocessed cloud) so it's not too heavy to render
        save_custom_visualizations(cloud_pp, best.path)

    total_time = time.time() - t0
    console.print(f"\n[dim]Pipeline completed in {total_time:.2f} s  |  "
                  f"Planning: {plan_time:.3f} s[/dim]")

    return results


# ── Graph serialisation helpers ───────────────────────────────────────────

def _save_graph(nodes, edges, path: str):
    import pickle
    with open(path, "wb") as f:
        import pickle
        pickle.dump({"nodes": nodes, "edges": edges}, f)


def _load_graph(path: str):
    import pickle
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["nodes"], data["edges"]


# ── __init__ stubs ────────────────────────────────────────────────────────

def _create_inits():
    for pkg in ["lidar", "terrain", "graph", "pathfinding", "models", "ui"]:
        p = Path(f"{pkg}/__init__.py")
        if not p.exists():
            p.write_text("")

def save_custom_visualizations(cloud, path_nodes, output_pfx="path_result"):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import matplotlib.patheffects as path_effects
    
    # 1. Prepare Data
    path_xyz = np.array([[n.x, n.y, n.z] for n in path_nodes])
    if path_xyz.shape[0] < 2:
        print("[red]✗[/red] Path too short to animate as an arrow.")
        return

    # 2. Setup Figure & static elements
    fig = plt.figure(figsize=(12, 10), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    
    # Plot Terrain (Static)
    ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], 
               c='skyblue', s=1, alpha=0.1, edgecolors='none', label='Terrain')
    
    # Initialize the Path Line (will be updated in animation)
    # We add a "neon glow" effect using path_effects
    line, = ax.plot([], [], [], 
                    c='#39FF14', linewidth=4, label='Path Traced', zorder=100)
    line.set_path_effects([path_effects.Stroke(linewidth=6, foreground='white'),
                           path_effects.Normal()])

    # Plot Start and Goal markers (Static)
    start = path_xyz[0]
    goal = path_xyz[-1]
    ax.scatter(start[0], start[1], start[2], c='#FF1493', s=100, label='Start') # Neon Pink
    ax.scatter(goal[0], goal[1], goal[2], c='#00BFFF', s=100, label='Goal')   # Neon Blue

    # Aesthetics
    ax.set_axis_off()
    ax.set_title("Path Tracing Animation", color='white', fontsize=15)
    
    # Save a clean static image of the complete path first
    line.set_data_3d(path_xyz[:, 0], path_xyz[:, 1], path_xyz[:, 2])
    plt.savefig(f"{output_pfx}_static.png", dpi=300, facecolor='black', bbox_inches='tight')
    print(f"\n[green]✓[/green] Static image of complete path saved: {output_pfx}_static.png")

    # 3. Create Animation (Tracing Path)
    # The number of frames is the number of points in our path
    num_frames = path_xyz.shape[0]

    def update(frame):
        # Update the line data to include points from start (0) to current frame
        current_path = path_xyz[:frame+1]
        line.set_data_3d(current_path[:, 0], current_path[:, 1], current_path[:, 2])
        return line,

    # Set a custom view angle that best shows the path
    ax.view_init(elev=30, azim=-45)

    # interval controls speed (lower = faster)
    ani = FuncAnimation(fig, update, frames=num_frames, interval=100, blit=True)
    
    # Save as GIF
    try:
        # 'pillow' writer is common, but 'ffmpeg' gives better compression for mp4
        ani.save(f"{output_pfx}_trace.gif", writer='pillow', fps=10)
        print(f"[green]✓[/green] Path tracing animation saved: {output_pfx}_trace.gif")
    except Exception as e:
        print(f"[red]✗[/red] Could not save GIF: {e}")
    
    plt.close()

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run(args)