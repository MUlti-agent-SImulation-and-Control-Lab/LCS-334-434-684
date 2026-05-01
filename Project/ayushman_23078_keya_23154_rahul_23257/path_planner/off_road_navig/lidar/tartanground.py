"""
lidar/tartanground.py
---------------------
TartanAir Dataset Reader

EXACT folder structure based on your dataset:

  tartanair_data/
    ForestEnv/                      Scene name
      Data_diff/                    Difficulty level
        P1001/                      Trajectory ID
          lidar/
            000000.ply              LiDAR scans (PLY format)
            000001.ply
            ...
          image_lcam_front/         Left camera RGB (.png)
          image_rcam_front/         Right camera RGB (.png)
          depth_lcam_front/         Depth images (.png)
          depth_rcam_front/
          seg_lcam_front/           Semantic segmentation (.png)
          seg_rcam_front/
          imu/
            acc.npy / .txt          Raw accelerometer body frame [m/s²]
            acc_nograv.npy / .txt   Linear accel global frame
            gyro.npy / .txt         Angular velocity [rad/s]
            pos_global.npy / .txt   Position [x,y,z] metres
            vel_global.npy / .txt   Velocity global [m/s]
            vel_body.npy / .txt     Velocity body frame
            ori_global.npy / .txt   Quaternion [qw,qx,qy,qz]
            imu_time.npy / .txt     IMU timestamps
            cam_time.npy / .txt     Camera timestamps
            parameter.yaml
          pose_lcam_front.txt       Camera poses
          pose_rcam_front.txt
          P1001_metadata.json
          ForestEnv_rgb.pcd         Scene point cloud
          ForestEnv_sem.pcd
          seg_label_map.json
    Gascola/
      Data_diff/
        ...
    OldScandinavia/
      Data_diff/
        ...

Usage
-----
  from lidar.tartanground import TartanAirDataset

  ds = TartanAirDataset("/mnt/tartanair_data")
  ds.print_summary()

  for traj in ds.iter_trajectories(scenes=["ForestEnv"]):
      cloud = traj.load_lidar_frame(0)
      imu   = traj.load_imu()
"""

from __future__ import annotations
import numpy as np
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterator, List, Optional


@dataclass
class IMUData:
    """All IMU signals for one trajectory."""
    acc            : np.ndarray  # (T, 3)
    acc_nograv     : np.ndarray  # (T, 3)
    gyro           : np.ndarray  # (T, 3)
    pos_global     : np.ndarray  # (T, 3)
    vel_global     : np.ndarray  # (T, 3)
    vel_body       : np.ndarray  # (T, 3)
    ori_global     : np.ndarray  # (T, 4) quaternion
    timestamps     : np.ndarray  # (T,)
    cam_timestamps : Optional[np.ndarray] = None
    params         : dict = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return len(self.timestamps)

    @property
    def duration_s(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])

    def accel_magnitude(self) -> np.ndarray:
        return np.linalg.norm(self.acc, axis=1)

    def gyro_magnitude(self) -> np.ndarray:
        return np.linalg.norm(self.gyro, axis=1)

    def vibration_score(self) -> np.ndarray:
        GRAVITY = 9.81
        return np.abs(self.accel_magnitude() - GRAVITY)

    def slip_score(self) -> np.ndarray:
        speed = np.linalg.norm(self.vel_body, axis=1) + 1e-6
        return self.gyro_magnitude() / speed

    def risk_from_imu(self, window_s: float = 0.5) -> np.ndarray:
        dt = float(np.median(np.diff(self.timestamps))) if len(self.timestamps) > 1 else 0.01
        k  = max(1, int(window_s / dt))
        vib  = self._smooth(self.vibration_score(), k)
        slip = self._smooth(self.slip_score(), k)
        gyro = self._smooth(self.gyro_magnitude(), k)
        def _n(arr):
            rng = arr.max() - arr.min()
            return (arr - arr.min()) / (rng + 1e-9)
        risk = (0.45 * _n(vib) + 0.35 * _n(slip) + 0.20 * _n(gyro)).astype(np.float32)
        return np.clip(risk, 0.0, 1.0)

    @staticmethod
    def _smooth(arr: np.ndarray, k: int) -> np.ndarray:
        if k <= 1:
            return arr
        return np.convolve(arr, np.ones(k)/k, mode="same")


class Trajectory:
    """One trajectory: tartanair_data/<scene>/Data_diff/<traj_id>/"""

    def __init__(self, path: Path):
        self.path       = path
        self.traj_id    = path.name                    # P1001
        self.difficulty = path.parent.name             # Data_diff
        self.scene      = path.parent.parent.name      # ForestEnv
        self._lidar_frames: Optional[List[Path]] = None

    @property
    def lidar_dir(self) -> Path:
        d = self.path / "lidar"
        if not d.exists():
            raise FileNotFoundError(f"No lidar/ in {self.path}")
        return d

    @property
    def lidar_frames(self) -> List[Path]:
        if self._lidar_frames is None:
            frames = sorted(
                p for p in self.lidar_dir.iterdir()
                if p.suffix.lower() in (".ply", ".pcd", ".npy")
            )
            if not frames:
                raise FileNotFoundError(f"No LiDAR frames in {self.lidar_dir}")
            self._lidar_frames = frames
        return self._lidar_frames

    @property
    def n_lidar_frames(self) -> int:
        try:
            return len(self.lidar_frames)
        except FileNotFoundError:
            return 0

    def load_lidar_frame(self, frame_idx: int) -> np.ndarray:
        from lidar.loader import load_point_cloud
        frames = self.lidar_frames
        if frame_idx < 0 or frame_idx >= len(frames):
            raise IndexError(f"Frame {frame_idx} out of range")
        return load_point_cloud(frames[frame_idx])

    def load_lidar_merged(self, max_frames: int = 10, stride: int = 1) -> np.ndarray:
        from lidar.loader import load_point_cloud
        frames = self.lidar_frames[::stride][:max_frames]
        poses  = self.load_poses()
        clouds = []
        for i, fpath in enumerate(frames):
            cloud = load_point_cloud(fpath)
            if i < len(poses):
                cloud = self._transform_to_global(cloud, poses[i])
            clouds.append(cloud)
        return np.vstack(clouds)

    def _transform_to_global(self, cloud: np.ndarray, pose: np.ndarray) -> np.ndarray:
        xyz = cloud[:, :3]
        tx, ty, tz = pose[0], pose[1], pose[2]
        qw, qx, qy, qz = pose[3], pose[4], pose[5], pose[6]
        R = np.array([
            [1-2*(qy**2+qz**2),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
            [2*(qx*qy+qz*qw),   1-2*(qx**2+qz**2),   2*(qy*qz-qx*qw)],
            [2*(qx*qz-qy*qw),     2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)],
        ], dtype=np.float32)
        xyz_global = (R @ xyz.T).T + np.array([tx, ty, tz], dtype=np.float32)
        result = cloud.copy()
        result[:, :3] = xyz_global
        return result

    @property
    def imu_dir(self) -> Path:
        d = self.path / "imu"
        if not d.exists():
            raise FileNotFoundError(f"No imu/ in {self.path}")
        return d

    def load_imu(self) -> IMUData:
        imu_dir = self.imu_dir
        def _load(name):
            npy = imu_dir / f"{name}.npy"
            txt = imu_dir / f"{name}.txt"
            if npy.exists():
                arr = np.load(npy).astype(np.float32)
            elif txt.exists():
                arr = np.loadtxt(txt, dtype=np.float32)
            else:
                return None
            return arr if arr.ndim == 2 else arr.reshape(-1, arr.shape[0] if arr.ndim==1 else 1)
        def _load1d(name):
            npy = imu_dir / f"{name}.npy"
            txt = imu_dir / f"{name}.txt"
            if npy.exists():
                return np.load(npy).astype(np.float64).ravel()
            elif txt.exists():
                return np.loadtxt(txt, dtype=np.float64).ravel()
            return None

        acc        = _load("acc")
        gyro       = _load("gyro")
        pos_global = _load("pos_global")
        timestamps = _load1d("imu_time")
        if acc is None or gyro is None or pos_global is None:
            raise FileNotFoundError(f"Missing acc/gyro/pos_global in {imu_dir}")
        if timestamps is None:
            timestamps = np.arange(len(acc), dtype=np.float64) / 100.0

        T = len(acc)
        def _or_zeros(arr, cols=3):
            return arr if arr is not None and len(arr) == T else np.zeros((T, cols), np.float32)

        acc_nograv = _or_zeros(_load("acc_nograv"))
        vel_global = _or_zeros(_load("vel_global"))
        vel_body   = _or_zeros(_load("vel_body"))
        ori_global = _or_zeros(_load("ori_global"), cols=4)
        if ori_global.shape[1] == 3:
            ori_global = _euler_to_quat(ori_global)
        cam_timestamps = _load1d("cam_time")
        params = {}
        param_file = imu_dir / "parameter.yaml"
        if param_file.exists():
            with open(param_file) as f:
                params = yaml.safe_load(f) or {}
        return IMUData(
            acc=acc, acc_nograv=acc_nograv, gyro=gyro,
            pos_global=pos_global, vel_global=vel_global, vel_body=vel_body,
            ori_global=ori_global, timestamps=timestamps,
            cam_timestamps=cam_timestamps, params=params
        )

    def load_poses(self) -> np.ndarray:
        imu = self.load_imu()
        return np.hstack([imu.pos_global, imu.ori_global]).astype(np.float32)

    def __repr__(self) -> str:
        return f"Trajectory(scene={self.scene!r}, diff={self.difficulty!r}, id={self.traj_id!r}, frames={self.n_lidar_frames})"


class TartanAirDataset:
    """Dataset reader for tartanair_data/"""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")
        self._scenes: Optional[List[str]] = None

    @property
    def scenes(self) -> List[str]:
        if self._scenes is None:
            self._scenes = sorted(
                d.name for d in self.root.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            )
        return self._scenes

    def trajectories_in_scene(self, scene: str) -> List[Trajectory]:
        scene_dir = self.root / scene
        if not scene_dir.exists():
            raise FileNotFoundError(f"Scene not found: {scene_dir}")
        trajs = []
        diff_dir = scene_dir / "Data_diff"
        if not diff_dir.exists():
            for candidate in sorted(scene_dir.iterdir()):
                if candidate.is_dir() and not candidate.name.startswith("."):
                    diff_dir = candidate
                    break
            else:
                return trajs
        for traj_dir in sorted(diff_dir.iterdir()):
            if not traj_dir.is_dir() or traj_dir.name.startswith("."):
                continue
            if (traj_dir / "lidar").exists() or (traj_dir / "imu").exists():
                trajs.append(Trajectory(traj_dir))
        return trajs

    def iter_trajectories(
        self, scenes: List[str] | None = None, max_per_scene: int | None = None
    ) -> Iterator[Trajectory]:
        target_scenes = scenes if scenes is not None else self.scenes
        for scene in target_scenes:
            try:
                trajs = self.trajectories_in_scene(scene)
            except FileNotFoundError:
                continue
            if max_per_scene is not None:
                trajs = trajs[:max_per_scene]
            for traj in trajs:
                yield traj

    def count_trajectories(self) -> int:
        return sum(len(self.trajectories_in_scene(s)) for s in self.scenes)

    def print_summary(self):
        print(f"TartanAir Dataset @ {self.root}")
        print(f"  Scenes: {len(self.scenes)}")
        total_trajs = 0
        total_frames = 0
        for scene in self.scenes[:10]:
            trajs = self.trajectories_in_scene(scene)
            frames = sum(t.n_lidar_frames for t in trajs)
            total_trajs  += len(trajs)
            total_frames += frames
            print(f"  {scene:30s}: {len(trajs):3d} trajectories  {frames:6,} lidar frames")
        if len(self.scenes) > 10:
            print(f"  ... ({len(self.scenes) - 10} more scenes)")
        print(f"  TOTAL: {total_trajs} trajectories  {total_frames:,} lidar frames")

    def build_training_dataset(
        self, output_dir: str | Path = "data/training",
        max_trajs: int = 200, frames_per_traj: int = 5,
        scenes: List[str] | None = None
    ):
        from lidar.preprocessor import preprocess
        from terrain.feature_extractor import extract_features
        from terrain.segmenter import TerrainSegmenter
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        seg = TerrainSegmenter()
        saved = 0
        for traj in self.iter_trajectories(scenes=scenes):
            if saved >= max_trajs:
                break
            try:
                cloud  = traj.load_lidar_merged(max_frames=frames_per_traj)
                cloud  = preprocess(cloud, voxel_size=0.25)
                if len(cloud) < 100:
                    continue
                feats  = extract_features(cloud, radius=1.0)
                labels = seg.predict(feats)
                imu    = traj.load_imu()
                risks  = _imu_risk_to_cloud(imu, cloud, feats, labels)
                out_name = f"traj_{traj.scene}_{traj.traj_id}.npz"
                np.savez_compressed(
                    output_dir / out_name,
                    features=feats.astype(np.float32),
                    labels=labels.astype(np.int32),
                    risks=risks.astype(np.float32),
                )
                saved += 1
                print(f"  [{saved:>3d}] {traj.scene}/{traj.traj_id}: {len(cloud):,} pts → {out_name}")
            except Exception as e:
                print(f"  SKIP {traj}: {e}")
        print(f"\nBuilt {saved} training files in {output_dir}/")
        return output_dir


def _euler_to_quat(euler: np.ndarray) -> np.ndarray:
    r, p, y = euler[:, 0], euler[:, 1], euler[:, 2]
    cr, sr = np.cos(r/2), np.sin(r/2)
    cp, sp = np.cos(p/2), np.sin(p/2)
    cy, sy = np.cos(y/2), np.sin(y/2)
    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy
    return np.column_stack([qw, qx, qy, qz]).astype(np.float32)


def _imu_risk_to_cloud(imu: IMUData, cloud: np.ndarray, feats: np.ndarray, labels: np.ndarray) -> np.ndarray:
    from terrain.risk_estimator import RiskEstimator
    base_risk  = RiskEstimator._heuristic_estimate(feats, labels)
    imu_risk   = imu.risk_from_imu()
    imu_median = float(np.median(imu_risk))
    scale = 1.0 + 0.5 * (imu_median - 0.3)
    scale = float(np.clip(scale, 0.5, 2.0))
    return np.clip(base_risk * scale, 0.0, 1.0).astype(np.float32)