#!/usr/bin/env python3
"""
Quick start script for TartanAir pathfinding.
Run this from ecs334/off_road_navig/

Usage:
  python quick_start.py
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, '.')

from lidar.tartanground import TartanAirDataset
from lidar.preprocessor import estimate_bounds
import numpy as np

# Dataset is in the parent directory's sibling folder
DATASET_ROOT = Path(__file__).parent.parent / "tartanair_data"

if not DATASET_ROOT.exists():
    print(f"ERROR: Dataset not found at {DATASET_ROOT.resolve()}")
    print("\nSearching for tartanair_data...")
    for candidate in [Path("../tartanair_data"), Path("../../tartanair_data"), 
                      Path.home() / "tartanair_data"]:
        if candidate.exists():
            print(f"  Found at: {candidate.resolve()}")
            DATASET_ROOT = candidate
            break
    else:
        print("Could not find tartanair_data folder!")
        sys.exit(1)

print(f"Using dataset: {DATASET_ROOT.resolve()}\n")

# Load dataset
ds = TartanAirDataset(DATASET_ROOT)
ds.print_summary()

# Get first trajectory
print("\nLoading first trajectory...")
scene = ds.scenes[0]
trajs = ds.trajectories_in_scene(scene)
if not trajs:
    print(f"No trajectories found in {scene}")
    sys.exit(1)

traj = trajs[0]
print(f"Trajectory: {traj}\n")

# Load one frame to test
print("Testing LiDAR loading...")
try:
    frame0 = traj.load_lidar_frame(0)
    print(f"✓ Loaded frame 0: {frame0.shape} - {len(frame0):,} points")
    print(f"  Min: {frame0.min(axis=0)}")
    print(f"  Max: {frame0.max(axis=0)}")
except Exception as e:
    print(f"✗ Failed to load frame 0: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Load merged cloud
print("\nLoading merged cloud (5 frames)...")
try:
    cloud = traj.load_lidar_merged(max_frames=5, stride=1)
    bounds = estimate_bounds(cloud)
    
    print(f"✓ Merged cloud: {len(cloud):,} points")
    print(f"  X: {bounds['x_min']:.1f} to {bounds['x_max']:.1f} m")
    print(f"  Y: {bounds['y_min']:.1f} to {bounds['y_max']:.1f} m")
    print(f"  Z: {bounds['z_min']:.1f} to {bounds['z_max']:.1f} m")
    print(f"  Extent: {bounds['extent_xy']:.1f} m")
    
    # Suggest coordinates
    sx = round(bounds['x_min'] + 2, 1)
    sy = round(bounds['y_min'] + 2, 1)
    gx = round(bounds['x_max'] - 2, 1)
    gy = round(bounds['y_max'] - 2, 1)
    
    print(f"\n✓ Suggested coordinates:")
    print(f"  --start {sx} {sy} 0")
    print(f"  --goal  {gx} {gy} 0")
    
    # Save merged cloud for easy testing
    output_path = Path("merged_cloud.npy")
    np.save(output_path, cloud)
    print(f"\n✓ Saved merged cloud to: {output_path.resolve()}")
    
    # Print full command
    print("\n" + "="*60)
    print("RUN THIS COMMAND:")
    print("="*60)
    print(f"""
python main.py \\
  --cloud merged_cloud.npy \\
  --start {sx} {sy} 0 \\
  --goal {gx} {gy} 0 \\
  --window 5 \\
  --visualize
""")
    
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("✓ All tests passed!")