"""
models/feedback_loop.py
------------------------
Traversal Feedback Loop — Active Learning from TartanGround IMU Data.

Two modes
---------
ONLINE  — robot is driving live, logs events per node, updates graph in real time.
OFFLINE — process historical TartanGround trajectories (878 paths × IMU signals)
          to auto-generate per-node risk targets for model training.

TartanGround IMU files used
---------------------------
  acc.npy / .txt          Body-frame accelerometer (with gravity)
  acc_nograv.npy / .txt   Global-frame linear acceleration
  gyro.npy / .txt         Body-frame angular velocity [ωx, ωy, ωz] rad/s
  vel_body.npy / .txt     Body-frame velocity
  pos_global.npy / .txt   Ground-truth position [x, y, z]
  imu_time.npy / .txt     Timestamps for acc + gyro

Risk derivation logic
---------------------
  vibration_risk = |acc_body| - g  (deviation from gravity = rough terrain)
  slip_risk      = |gyro| / |vel_body|  (spinning relative to speed = slipping)
  final_risk     = 0.45×vibration + 0.35×slip + 0.20×|gyro| — smoothed over 0.5s

Usage
-----
  # OFFLINE: auto-label all TartanGround trajectories
  from models.feedback_loop import build_imu_risk_dataset
  build_imu_risk_dataset(
      dataset_root = "/mnt/tartanground",
      output_dir   = "data/training",
      max_trajs    = 300,
  )

  # ONLINE: live robot feedback
  from models.feedback_loop import TraversalFeedbackLoop
  loop = TraversalFeedbackLoop()
  loop.log_event(node_id, "slip", severity=0.8)
  loop.apply_to_graph(nodes)
  loop.save("logs/events.json")
"""

from __future__ import annotations
import json
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

# ═══════════════════════════════════════════════════════════════════
#  Event multipliers
# ═══════════════════════════════════════════════════════════════════

EVENT_RISK_MULTIPLIERS = {
    "clean"            : 0.70,
    "vibration"        : 1.30,
    "slip"             : 1.50,
    "stuck"            : 2.00,
    "rollover_warning" : 3.00,
}
EVENT_DECAY_HALF_LIFE = 3600.0   # seconds


@dataclass
class TraversalEvent:
    node_id   : int
    event_type: str
    severity  : float
    timestamp : float = field(default_factory=time.time)
    xyz       : Optional[tuple] = None


# ═══════════════════════════════════════════════════════════════════
#  Online feedback loop (live robot)
# ═══════════════════════════════════════════════════════════════════

class TraversalFeedbackLoop:
    """
    Online risk updater. Log events while the robot drives, apply to graph.
    """

    def __init__(
        self,
        alpha              : float = 0.15,
        min_events_retrain : int   = 200,
        decay_half_life    : float = EVENT_DECAY_HALF_LIFE,
    ):
        self.alpha              = alpha
        self.min_events_retrain = min_events_retrain
        self.decay_half_life    = decay_half_life
        self._events      : List[TraversalEvent] = []
        self._node_updates: Dict[int, List[float]] = {}
        self._events_since_retrain = 0

    def log_event(
        self, node_id: int, event_type: str,
        severity: float = 1.0, xyz: tuple | None = None,
    ):
        if event_type not in EVENT_RISK_MULTIPLIERS:
            raise ValueError(f"Unknown event type '{event_type}'. "
                             f"Use: {list(EVENT_RISK_MULTIPLIERS)}")
        ev = TraversalEvent(node_id=node_id, event_type=event_type,
                            severity=float(np.clip(severity, 0, 1)), xyz=xyz)
        self._events.append(ev)
        self._events_since_retrain += 1
        self._node_updates.setdefault(node_id, []).append(
            EVENT_RISK_MULTIPLIERS[event_type] * ev.severity
        )

    def apply_to_graph(self, nodes: dict) -> int:
        updated = 0
        for node_id, adjustments in self._node_updates.items():
            if node_id not in nodes:
                continue
            node = nodes[node_id]
            for adj in adjustments:
                target = float(np.clip(node.risk * adj, 0.0, 1.0))
                node.risk = float(np.clip(
                    node.risk * (1 - self.alpha) + target * self.alpha,
                    0.0, 1.0
                ))
            updated += 1
        self._node_updates.clear()
        self._apply_decay(nodes)
        return updated

    def _apply_decay(self, nodes: dict):
        NEUTRAL = 0.30
        ALPHA   = 0.01
        now     = time.time()
        recent  = {e.node_id for e in self._events
                   if (now - e.timestamp) < self.decay_half_life * 2}
        for nid, node in nodes.items():
            if nid not in recent:
                node.risk = float(node.risk * (1 - ALPHA) + NEUTRAL * ALPHA)

    def should_retrain(self) -> bool:
        return self._events_since_retrain >= self.min_events_retrain

    def retrain(self, features_map: dict, labels_map: dict,
                save_path: str | Path = "models/weights/uncertainty_risk.pt"):
        feats, labels, risks = self.get_training_data(features_map, labels_map)
        if len(feats) < 10:
            print("[FeedbackLoop] Not enough data yet.")
            return
        from models.uncertainty_risk import train as _train
        print(f"[FeedbackLoop] Retraining on {len(feats)} events…")
        _train(feats, labels, risks, save_path=save_path, epochs=30, lr=5e-4)
        self._events_since_retrain = 0

    def get_training_data(self, features_map: dict, labels_map: dict) -> tuple:
        per_node: Dict[int, List[float]] = {}
        for e in self._events:
            age_h  = (time.time() - e.timestamp) / 3600.0
            weight = np.exp(-age_h / (self.decay_half_life / 3600.0))
            per_node.setdefault(e.node_id, []).append(
                EVENT_RISK_MULTIPLIERS[e.event_type] * e.severity * weight
            )
        fl, ll, rl = [], [], []
        for nid, mults in per_node.items():
            if nid not in features_map:
                continue
            target = float(np.clip(0.3 * float(np.mean(mults)), 0.0, 1.0))
            fl.append(features_map[nid])
            ll.append(labels_map.get(nid, 0))
            rl.append(target)
        if not fl:
            return np.zeros((0, 8)), np.zeros(0, int), np.zeros(0)
        return (np.array(fl, np.float32),
                np.array(ll, np.int32),
                np.array(rl, np.float32))

    def stats(self) -> dict:
        from collections import Counter
        return {
            "total_events"    : len(self._events),
            "unique_nodes"    : len({e.node_id for e in self._events}),
            "event_breakdown" : dict(Counter(e.event_type for e in self._events)),
            "pending_updates" : sum(len(v) for v in self._node_updates.values()),
            "ready_to_retrain": self.should_retrain(),
        }

    def save(self, path: str | Path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([asdict(e) for e in self._events], f, indent=2)
        print(f"[FeedbackLoop] {len(self._events)} events → {path}")

    def load(self, path: str | Path):
        path = Path(path)
        if not path.exists():
            return
        with open(path) as f:
            self._events = [TraversalEvent(**d) for d in json.load(f)]
        print(f"[FeedbackLoop] Loaded {len(self._events)} events from {path}")


# ═══════════════════════════════════════════════════════════════════
#  TartanGround IMU risk extractor
# ═══════════════════════════════════════════════════════════════════

class TartanIMURiskExtractor:
    """
    Extracts per-timestep risk scores from TartanGround IMU files.

    Reads the exact files produced by the dataset:
      acc.npy, gyro.npy, vel_body.npy, imu_time.npy
    (falls back to .txt variants automatically)

    Example
    -------
      extractor = TartanIMURiskExtractor("/mnt/tartanground/scene/P000/imu")
      risks     = extractor.risk_timeseries()   # (T,) float32 in [0,1]
      mean_risk = extractor.mean_risk()
    """

    GRAVITY = 9.81

    def __init__(self, imu_dir: str | Path):
        self.imu_dir = Path(imu_dir)
        if not self.imu_dir.exists():
            raise FileNotFoundError(f"IMU directory not found: {self.imu_dir}")
        self._cache: dict = {}

    # ── File loading ──────────────────────────────────────────────

    def _load(self, name: str) -> Optional[np.ndarray]:
        """Load .npy or .txt, return float32 array or None."""
        if name in self._cache:
            return self._cache[name]
        for ext in (".npy", ".txt"):
            p = self.imu_dir / (name + ext)
            if p.exists():
                arr = (np.load(p).astype(np.float32)
                       if ext == ".npy"
                       else np.loadtxt(p, dtype=np.float32))
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                self._cache[name] = arr
                return arr
        return None

    @property
    def acc(self) -> np.ndarray:
        """(T,3) raw accelerometer in body frame."""
        a = self._load("acc")
        if a is None:
            raise FileNotFoundError(f"acc.npy/.txt not found in {self.imu_dir}")
        return a

    @property
    def acc_nograv(self) -> np.ndarray:
        """(T,3) linear acceleration, gravity removed, global frame."""
        a = self._load("acc_nograv")
        return a if a is not None else np.zeros_like(self.acc)

    @property
    def acc_nograv_body(self) -> np.ndarray:
        """(T,3) linear acceleration, gravity removed, body frame."""
        a = self._load("acc_nograv_body")
        return a if a is not None else np.zeros_like(self.acc)

    @property
    def gyro(self) -> np.ndarray:
        """(T,3) angular velocity [ωx, ωy, ωz] rad/s, body frame."""
        g = self._load("gyro")
        if g is None:
            raise FileNotFoundError(f"gyro.npy/.txt not found in {self.imu_dir}")
        return g

    @property
    def vel_body(self) -> np.ndarray:
        """(T,3) velocity in body frame."""
        v = self._load("vel_body")
        return v if v is not None else np.zeros_like(self.acc)

    @property
    def vel_global(self) -> np.ndarray:
        """(T,3) velocity in global frame."""
        v = self._load("vel_global")
        return v if v is not None else np.zeros_like(self.acc)

    @property
    def pos_global(self) -> np.ndarray:
        """(T,3) ground-truth position [x,y,z]."""
        p = self._load("pos_global")
        if p is None:
            raise FileNotFoundError(f"pos_global.npy/.txt not in {self.imu_dir}")
        return p

    @property
    def ori_global(self) -> np.ndarray:
        """(T,4) orientation quaternion [qw,qx,qy,qz]."""
        o = self._load("ori_global")
        return o if (o is not None and o.shape[1] == 4) else np.tile([1,0,0,0], (len(self.acc), 1)).astype(np.float32)

    @property
    def timestamps(self) -> np.ndarray:
        """(T,) IMU timestamps in seconds."""
        t = self._load("imu_time")
        if t is not None:
            return t.ravel().astype(np.float64)
        # Synthesise at 100 Hz
        return np.arange(len(self.acc), dtype=np.float64) / 100.0

    @property
    def cam_timestamps(self) -> Optional[np.ndarray]:
        """(C,) camera timestamps. Lower freq than IMU."""
        t = self._load("cam_time")
        return t.ravel() if t is not None else None

    def n_samples(self) -> int:
        return len(self.acc)

    def duration_s(self) -> float:
        ts = self.timestamps
        return float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0

    # ── Risk computation ───────────────────────────────────────────

    def vibration_signal(self) -> np.ndarray:
        """
        (T,) vibration = |acc_body| - g
        Measures how much the body acceleration deviates from gravity.
        High → rocky / rough ground.
        """
        return np.abs(np.linalg.norm(self.acc, axis=1) - self.GRAVITY)

    def slip_signal(self) -> np.ndarray:
        """
        (T,) slip = |gyro| / (|vel_body| + ε)
        High angular rate relative to forward speed = wheel slip / mud.
        Uses vel_body because it's in the robot's own reference frame.
        """
        speed = np.linalg.norm(self.vel_body, axis=1) + 0.1
        return np.linalg.norm(self.gyro, axis=1) / speed

    def instability_signal(self) -> np.ndarray:
        """
        (T,) raw |gyro| — captures sudden rotations regardless of speed.
        """
        return np.linalg.norm(self.gyro, axis=1)

    def risk_timeseries(self, smooth_s: float = 0.5) -> np.ndarray:
        """
        (T,) risk score in [0,1] derived from all three signals.

        Weights:
          45% vibration   (roughness / rocks)
          35% slip        (wetness / mud)
          20% instability (generic instability)

        smooth_s : smoothing window in seconds (reduces noise).
        """
        ts = self.timestamps
        dt = float(np.median(np.diff(ts))) if len(ts) > 1 else 0.01
        k  = max(1, int(smooth_s / dt))

        vib  = self._smooth(self.vibration_signal(),   k)
        slip = self._smooth(self.slip_signal(),         k)
        inst = self._smooth(self.instability_signal(),  k)

        # Normalise each to [0,1] relative to this trajectory
        def _norm(a):
            lo, hi = a.min(), a.max()
            return (a - lo) / (hi - lo + 1e-9)

        risk = (0.45 * _norm(vib)
                + 0.35 * _norm(slip)
                + 0.20 * _norm(inst)).astype(np.float32)

        return np.clip(risk, 0.0, 1.0)

    def mean_risk(self) -> float:
        return float(self.risk_timeseries().mean())

    def p75_risk(self) -> float:
        return float(np.percentile(self.risk_timeseries(), 75))

    def risk_at_position(self, query_xyz: np.ndarray, radius: float = 2.0) -> np.ndarray:
        """
        For each query position, return the mean risk of IMU samples
        whose ground-truth position is within radius metres.

        Parameters
        ----------
        query_xyz : (M, 3) positions to query (e.g. graph node positions)
        radius    : spatial neighbourhood in metres

        Returns
        -------
        (M,) float32 risk per query position
        """
        pos      = self.pos_global                    # (T, 3)
        risk_ts  = self.risk_timeseries()             # (T,)
        M        = len(query_xyz)
        out      = np.full(M, 0.3, dtype=np.float32)  # neutral prior

        for i, qp in enumerate(query_xyz):
            d    = np.linalg.norm(pos - qp, axis=1)  # (T,)
            mask = d < radius
            if mask.sum() > 0:
                out[i] = float(risk_ts[mask].mean())

        return out

    @staticmethod
    def _smooth(arr: np.ndarray, k: int) -> np.ndarray:
        if k <= 1:
            return arr
        return np.convolve(arr, np.ones(k)/k, mode="same")


# ═══════════════════════════════════════════════════════════════════
#  Offline bulk dataset builder (uses TartanGroundDataset)
# ═══════════════════════════════════════════════════════════════════

def build_imu_risk_dataset(
    dataset_root : str | Path,
    output_dir   : str | Path = "data/training",
    max_trajs    : int = 300,
    frames_per_traj: int = 8,
    scenes       : list | None = None,
    save_interval: int = 50,
):
    """
    Process TartanGround trajectories to build a training dataset.

    For each trajectory:
      1. Load merged LiDAR point cloud (multiple frames)
      2. Preprocess & extract terrain features
      3. Auto-label with heuristic segmenter
      4. Load IMU → compute risk_at_position() per graph node
      5. Save features.npy, labels.npy, risks.npy

    This converts your 1.5 TB of raw data into a compact
    supervised training set without any manual labelling.

    Parameters
    ----------
    dataset_root    : path to TartanGround root directory
    output_dir      : where to save .npz training files
    max_trajs       : max trajectories to process
    frames_per_traj : LiDAR frames to merge per trajectory
    scenes          : list of scene names (None = all)
    save_interval   : print progress every N trajectories

    Output
    ------
    output_dir/
      traj_<scene>_<id>.npz  →  features(N,8), labels(N,), risks(N,)
    summary.json              →  stats per trajectory
    """
    from lidar.tartanground import TartanGroundDataset
    from lidar.loader import load_point_cloud
    from lidar.preprocessor import preprocess
    from terrain.feature_extractor import extract_features
    from terrain.segmenter import TerrainSegmenter

    dataset_root = Path(dataset_root)
    output_dir   = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds  = TartanGroundDataset(dataset_root)
    seg = TerrainSegmenter()
    summary = []
    saved   = 0

    print(f"[IMURiskDataset] Building training data from {dataset_root}")
    print(f"  Scenes available: {len(ds.scenes)}")
    print(f"  Max trajectories: {max_trajs}")
    print(f"  Output: {output_dir}")
    print()

    for traj in ds.iter_trajectories(scenes=scenes):
        if saved >= max_trajs:
            break

        try:
            # ── 1. Load merged LiDAR ──────────────────────────────
            cloud = traj.load_lidar_merged(max_frames=frames_per_traj)
            if len(cloud) < 200:
                continue

            # ── 2. Preprocess ─────────────────────────────────────
            cloud_pp = preprocess(cloud, voxel_size=0.25)

            # ── 3. Features + heuristic labels ────────────────────
            feats  = extract_features(cloud_pp, radius=1.0)
            labels = seg.predict(feats)

            # ── 4. IMU-derived risk per point ─────────────────────
            imu_ext = TartanIMURiskExtractor(traj.imu_dir)
            risks   = imu_ext.risk_at_position(cloud_pp[:, :3], radius=2.0)

            # Where IMU gave neutral prior (no nearby trajectory point),
            # fall back to feature-based risk
            from terrain.risk_estimator import RiskEstimator
            feat_risks = RiskEstimator._heuristic_estimate(feats, labels)
            neutral    = np.abs(risks - 0.3) < 0.01   # unvisited regions
            risks[neutral] = feat_risks[neutral]

            # ── 5. Save ───────────────────────────────────────────
            fname = f"traj_{traj.scene}_{traj.traj_id}.npz"
            np.savez_compressed(
                output_dir / fname,
                features = feats.astype(np.float32),
                labels   = labels.astype(np.int32),
                risks    = risks.astype(np.float32),
            )
            saved += 1

            entry = {
                "scene"      : traj.scene,
                "traj_id"    : traj.traj_id,
                "n_points"   : int(len(cloud_pp)),
                "mean_risk"  : float(risks.mean()),
                "imu_duration": imu_ext.duration_s(),
                "file"       : fname,
            }
            summary.append(entry)

            if saved % save_interval == 0 or saved <= 5:
                print(f"  [{saved:>4d}/{max_trajs}] {traj.scene}/{traj.traj_id}: "
                      f"{len(cloud_pp):>7,} pts  mean_risk={risks.mean():.3f}  "
                      f"imu_dur={imu_ext.duration_s():.1f}s")

        except Exception as e:
            print(f"  SKIP {traj.scene}/{traj.traj_id}: {e}")
            continue

    # Save summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[IMURiskDataset] Done. {saved} trajectories → {output_dir}")
    print(f"  Total points: {sum(e['n_points'] for e in summary):,}")
    print(f"  Mean risk across dataset: {np.mean([e['mean_risk'] for e in summary]):.3f}")
    return output_dir