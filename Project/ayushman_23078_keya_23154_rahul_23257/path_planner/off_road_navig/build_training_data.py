import sys
sys.path.insert(0, '.')
from lidar.tartanground import TartanAirDataset
 
ds = TartanAirDataset('tartanair_data')  # <-- your path
ds.build_training_dataset(
    output_dir    = 'off_road_navig/data/training',
    max_trajs     = 100,                   # process 100 trajectories
    frames_per_traj = 8,                   # merge 8 LiDAR frames per traj
    scenes        = None,                  # None = all scenes
)
