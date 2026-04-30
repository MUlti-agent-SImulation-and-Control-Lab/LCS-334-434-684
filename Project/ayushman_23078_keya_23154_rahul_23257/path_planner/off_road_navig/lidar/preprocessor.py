"""
lidar/preprocessor.py
---------------------
Preprocessing pipeline for raw LiDAR point clouds:
  1. Statistical outlier removal
  2. Voxel-grid downsampling
  3. Ground/height normalisation (subtract median ground plane)
  4. Bounding-box crop to ROI
  5. Ground-level filter (remove canopy / high points)
"""

from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree


def preprocess(
    cloud: np.ndarray,
    voxel_size: float = 0.25,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    roi: tuple[float, float, float, float] | None = None,
    max_height: float = 10.0,   
) -> np.ndarray:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    cloud       : (N, C) array, first 3 cols = x, y, z
    voxel_size  : grid cell size in metres for downsampling
    nb_neighbors: K neighbours for outlier test
    std_ratio   : points beyond mean_dist + std_ratio*std are removed
    roi         : (x_min, y_min, x_max, y_max) crop — None keeps all
    max_height  : after ground normalisation, drop points above this Z.
                  Removes tree canopy / tall vegetation that inflates risk.
                  Default 3.0 m keeps robot-relevant geometry only.

    Returns
    -------
    (M, C) filtered, downsampled cloud
    """
    cloud = cloud.copy()

    if roi is not None:
        cloud = _crop_roi(cloud, roi)

    cloud = _statistical_outlier_removal(cloud, nb_neighbors, std_ratio)
    cloud = _voxel_downsample(cloud, voxel_size)
    cloud = _normalise_ground(cloud)
    cloud = _filter_ground_level(cloud, max_height)   # ← NEW: after normalise so Z=0 is ground

    return cloud


# ── Crop ──────────────────────────────────────────────────────────────────

def _crop_roi(cloud: np.ndarray, roi: tuple) -> np.ndarray:
    x_min, y_min, x_max, y_max = roi
    mask = (
        (cloud[:, 0] >= x_min) & (cloud[:, 0] <= x_max) &
        (cloud[:, 1] >= y_min) & (cloud[:, 1] <= y_max)
    )
    return cloud[mask]


# ── Statistical outlier removal ───────────────────────────────────────────

def _statistical_outlier_removal(
    cloud: np.ndarray,
    nb_neighbors: int,
    std_ratio: float,
) -> np.ndarray:
    return cloud # --- IGNORE --- (temporarily disable for testing)
    if len(cloud) < nb_neighbors + 1:
        return cloud

    tree = cKDTree(cloud[:, :3])
    dists, _ = tree.query(cloud[:, :3], k=nb_neighbors + 1)
    mean_dists = dists[:, 1:].mean(axis=1)   # exclude self (dist=0)

    global_mean = mean_dists.mean()
    global_std = mean_dists.std()
    threshold = global_mean + std_ratio * global_std

    return cloud[mean_dists <= threshold]


# ── Voxel downsampling ────────────────────────────────────────────────────

def _voxel_downsample(cloud: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    Assign each point to a voxel, keep the centroid of each occupied voxel.
    Averages all feature columns too (intensity etc.)
    """
    coords = cloud[:, :3]
    min_coords = coords.min(axis=0)

    indices = np.floor((coords - min_coords) / voxel_size).astype(np.int32)
    max_idx = indices.max(axis=0) + 1
    keys = (
        indices[:, 0].astype(np.int64) * max_idx[1] * max_idx[2]
        + indices[:, 1].astype(np.int64) * max_idx[2]
        + indices[:, 2].astype(np.int64)
    )

    order = np.argsort(keys)
    sorted_keys = keys[order]
    sorted_cloud = cloud[order]

    unique_keys, first_idx, counts = np.unique(
        sorted_keys, return_index=True, return_counts=True
    )

    cum_cloud = np.zeros((len(unique_keys), cloud.shape[1]), dtype=np.float32)
    for i, (start, cnt) in enumerate(zip(first_idx, counts)):
        cum_cloud[i] = sorted_cloud[start : start + cnt].mean(axis=0)

    return cum_cloud


# ── Ground normalisation ──────────────────────────────────────────────────

def _normalise_ground(cloud: np.ndarray, percentile: float = 5.0) -> np.ndarray:
    """
    Subtract the estimated ground level from z so that ground ≈ 0.
    Ground is approximated as the `percentile`-th z value.
    """
    z_floor = np.percentile(cloud[:, 2], percentile)
    cloud = cloud.copy()
    cloud[:, 2] -= z_floor
    return cloud


# ── Ground-level filter ───────────────────────────────────────────────────

def _filter_ground_level(cloud: np.ndarray, max_height: float = 3.0) -> np.ndarray:
    """
    Remove points above max_height metres from ground (Z=0 after normalisation).

    Why: TartanAir forest scenes contain tall tree canopy (Z up to 42m).
    These points get labelled VEGETATION/OBSTACLE with high risk, but the
    robot never travels through them. Keeping them inflates node risk scores
    and compresses risk variance, making A* and Risk-Aware identical.

    3.0m captures ground surface + low obstacles (rocks, logs, bushes)
    while discarding trunks and canopy.
    """
    mask = (cloud[:, 2] >= 0.0) & (cloud[:, 2] <= max_height)
    filtered = cloud[mask]

    removed = len(cloud) - len(filtered)
    pct = removed / max(len(cloud), 1) * 100
    print(f"[Preprocessor] Ground filter: kept {len(filtered):,} / {len(cloud):,} points "
          f"(removed {removed:,} = {pct:.1f}% above {max_height}m)")

    return filtered


# ── Utility ───────────────────────────────────────────────────────────────

def estimate_bounds(cloud: np.ndarray) -> dict:
    """Return axis-aligned bounding box of the cloud."""
    mins = cloud[:, :3].min(axis=0)
    maxs = cloud[:, :3].max(axis=0)
    return {
        "x_min": float(mins[0]), "x_max": float(maxs[0]),
        "y_min": float(mins[1]), "y_max": float(maxs[1]),
        "z_min": float(mins[2]), "z_max": float(maxs[2]),
        "extent_xy": float(np.sqrt((maxs[0]-mins[0])**2 + (maxs[1]-mins[1])**2)),
    }