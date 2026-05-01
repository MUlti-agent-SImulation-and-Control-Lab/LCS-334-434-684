# inspect_traj.py
import sys, numpy as np
sys.path.insert(0, '.')
from lidar.tartanground import TartanAirDataset
from lidar.preprocessor import estimate_bounds
 
ds   = TartanAirDataset('../tartanair_data')  # <-- your path
traj = ds.trajectories_in_scene('ForestEnv')[0]   # first trajectory
 
print(f'Trajectory: {traj}')
print(f'LiDAR frames: {traj.n_lidar_frames}')
 
# Load merged cloud (5 frames combined)
cloud  = traj.load_lidar_merged(max_frames=5)
bounds = estimate_bounds(cloud)
print(f'Points (merged): {len(cloud):,}')
print(f'X: {bounds["x_min"]:.1f}  to  {bounds["x_max"]:.1f} m')
print(f'Y: {bounds["y_min"]:.1f}  to  {bounds["y_max"]:.1f} m')
print(f'Z: {bounds["z_min"]:.1f}  to  {bounds["z_max"]:.1f} m')
 
# IMU summary
imu = traj.load_imu()
print(f'IMU samples: {imu.n_samples}')
print(f'Duration: {imu.duration_s:.1f} s')
print(f'Mean risk (IMU): {imu.risk_from_imu().mean():.3f}')
 
sx = round(bounds['x_min'] + 2, 1)
sy = round(bounds['y_min'] + 2, 1)
gx = round(bounds['x_max'] - 2, 1)
gy = round(bounds['y_max'] - 2, 1)
print(f'Suggested --start  {sx} {sy} 0')
print(f'Suggested --goal   {gx} {gy} 0')
