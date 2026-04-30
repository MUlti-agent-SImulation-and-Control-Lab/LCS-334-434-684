# run_merged.py  --  loads 8 LiDAR frames and merges into one cloud
import sys, types
sys.path.insert(0, '.')
from lidar.tartanground import TartanAirDataset
from lidar.preprocessor import estimate_bounds
from main import run
 
ds   = TartanAirDataset('tartanair_data')  # <-- your path
traj = ds.trajectories_in_scene('ForestEnv')[0]
 
# Merge 8 frames into one cloud, save as temp .npy
import numpy as np
cloud = traj.load_lidar_merged(max_frames=8)
np.save('/tmp/merged_cloud.npy', cloud)
 
bounds = estimate_bounds(cloud)
sx = round(bounds['x_min'] + 2, 1)
sy = round(bounds['y_min'] + 2, 1)
gx = round(bounds['x_max'] - 2, 1)
gy = round(bounds['y_max'] - 2, 1)
 
args = types.SimpleNamespace(
    cloud='/tmp/merged_cloud.npy', synthetic=False,
    start=[sx, sy, 0.0], goal=[gx, gy, 0.0],
    window=2, voxel=2.0, connect=8.0, max_risk=0.97,
    weights_clf=None, weights_risk=None,
    no_smooth=False, save_graph=None, load_graph=None,
    alternatives=3, n_synthetic=50000,
    visualize=True, save_anim=None,
)
results = run(args)
print(f'Path: {len(results[0].path)} nodes, {results[0].total_dist:.1f}m')
