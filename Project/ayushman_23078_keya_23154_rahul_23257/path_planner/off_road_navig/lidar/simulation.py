"""
lidar/simulation.py
-------------------
Simple loader for single PCD/PLY files from custom simulation environments.

Usage:
    from lidar.simulation import load_simulation_cloud
    
    cloud = load_simulation_cloud('simulation_lidar/full_map.pcd')
    # Returns: numpy array (N, 3) ready for main.py
"""

from __future__ import annotations
from pathlib import Path
import numpy as np


def load_simulation_cloud(filepath: str | Path) -> np.ndarray:
    """
    Load a single point cloud file from simulation.
    
    Args:
        filepath: Path to .pcd, .ply, or .npy file
        
    Returns:
        cloud: (N, 3) or (N, 4) numpy array [x, y, z] or [x, y, z, intensity]
    """
    from lidar.loader import load_point_cloud
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Cloud file not found: {filepath}")
    
    # Use existing loader (handles .pcd, .ply, .npy, .las)
    cloud = load_point_cloud(str(filepath))
    
    print(f"Loaded simulation cloud: {filepath.name}")
    print(f"  Points: {len(cloud):,}")
    print(f"  Shape: {cloud.shape}")
    print(f"  X range: [{cloud[:, 0].min():.1f}, {cloud[:, 0].max():.1f}] m")
    print(f"  Y range: [{cloud[:, 1].min():.1f}, {cloud[:, 1].max():.1f}] m")
    print(f"  Z range: [{cloud[:, 2].min():.1f}, {cloud[:, 2].max():.1f}] m")
    
    return cloud


def suggest_start_goal(cloud: np.ndarray, margin: float = 5.0) -> dict:
    x_range = cloud[:, 0].max() - cloud[:, 0].min()
    y_range = cloud[:, 1].max() - cloud[:, 1].min()

    if x_range < (margin * 2) or y_range < (margin * 2):
        print(f"[!] Warning: Map extent ({x_range:.1f}m) is too small for margin ({margin}m). Reducing margin.")
        margin = min(x_range, y_range) * 0.1  # Fallback to 10% of map size
    x_min, y_min = cloud[:, 0].min(), cloud[:, 1].min()
    x_max, y_max = cloud[:, 0].max(), cloud[:, 1].max()
    z_median = np.median(cloud[:, 2])
    
    # Start: bottom-left corner + margin
    start = (
        x_min + margin,
        y_min + margin,
        z_median
    )
    
    # Goal: top-right corner - margin
    goal = (
        x_max - margin,
        y_max - margin,
        z_median
    )
    
    return {'start': start, 'goal': goal}