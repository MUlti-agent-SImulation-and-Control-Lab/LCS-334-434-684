"""
terrain/feature_extractor.py
"""

from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree


def extract_features(
    cloud: np.ndarray,
    radius: float = 1.0,
    min_neighbours: int = 5,
) -> np.ndarray:
    """
    Parameters
    ----------
    cloud          : (N, C) preprocessed cloud; cols 0-2 = x,y,z;
                     col 3 = intensity (optional)
    radius         : neighbourhood search radius in metres
    min_neighbours : minimum points required for feature calc

    Returns
    -------
    features : (N, 8) float32 array
    """
    xyz = cloud[:, :3].astype(np.float64)
    has_intensity = cloud.shape[1] > 3
    intensity = cloud[:, 3].astype(np.float64) if has_intensity else np.ones(len(cloud))

    tree = cKDTree(xyz)
    neighbour_lists = tree.query_ball_point(xyz, r=radius)

    n = len(cloud)
    slope_arr     = np.zeros(n, dtype=np.float32)
    roughness_arr = np.zeros(n, dtype=np.float32)
    curvature_arr = np.zeros(n, dtype=np.float32)
    density_arr   = np.zeros(n, dtype=np.float32)
    int_std_arr   = np.zeros(n, dtype=np.float32)

    HIGH_RISK_SLOPE     = 45.0
    HIGH_RISK_ROUGHNESS = 0.5

    for i, neighbours in enumerate(neighbour_lists):
        k = len(neighbours)
        density_arr[i] = k

        if k < min_neighbours:
            slope_arr[i]     = HIGH_RISK_SLOPE
            roughness_arr[i] = HIGH_RISK_ROUGHNESS
            curvature_arr[i] = 1.0
            int_std_arr[i]   = 0.0
            continue

        pts  = xyz[neighbours]
        ints = intensity[neighbours]

        centroid = pts.mean(axis=0)
        centred  = pts - centroid
        cov      = np.cov(centred.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal   = eigvecs[:, 0]
        normal  /= np.linalg.norm(normal) + 1e-9

        cos_angle     = abs(normal[2])
        slope_arr[i]  = float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))

        residuals        = np.abs(centred @ normal)
        roughness_arr[i] = float(np.sqrt((residuals ** 2).mean()))

        total_var        = eigvals.sum()
        curvature_arr[i] = float(eigvals[0] / (total_var + 1e-9))

        int_std_arr[i] = float(ints.std())

    height_arr = cloud[:, 2].astype(np.float32)
    int_arr    = intensity.astype(np.float32)

    def _norm(arr):
        rng = arr.max() - arr.min()
        return (arr - arr.min()) / (rng + 1e-9)

    norm_int   = _norm(int_arr)
    norm_rough = _norm(roughness_arr)
    norm_h     = _norm(height_arr)

    wetness_proxy = (
        (1.0 - norm_int)   * 0.5
        + (1.0 - norm_rough) * 0.3
        + (1.0 - norm_h)     * 0.2
    ).astype(np.float32)

    features = np.column_stack([
        slope_arr,      # 0
        roughness_arr,  # 1
        curvature_arr,  # 2
        height_arr,     # 3
        int_arr,        # 4
        int_std_arr,    # 5
        wetness_proxy,  # 6
        density_arr,    # 7
    ]).astype(np.float32)

    return features


# ── top-level — NOT inside extract_features ───────────────────────────────
def add_rgb_features(
    features: np.ndarray,
    xyz: np.ndarray,
    image: np.ndarray,
    K: np.ndarray,
    T_cam_lidar: np.ndarray,
) -> np.ndarray:
    """
    Project LiDAR points into RGB image, sample color, append HSV features.

    Parameters
    ----------
    features    : (N, 8)  output of extract_features()
    xyz         : (N, 3)  point coordinates
    image       : (H, W, 3) BGR image (OpenCV convention)
    K           : (3, 3)  camera intrinsic matrix
    T_cam_lidar : (4, 4)  LiDAR-to-camera extrinsic transform

    Returns
    -------
    (N, 11) — original 8 features + hue, saturation, value
    """
    import cv2
    N    = len(xyz)
    ones = np.ones((N, 1))
    pts_cam = (T_cam_lidar @ np.hstack([xyz, ones]).T).T[:, :3]

    uvw = (K @ pts_cam.T).T
    u   = (uvw[:, 0] / uvw[:, 2]).astype(int)
    v   = (uvw[:, 1] / uvw[:, 2]).astype(int)

    H, W  = image.shape[:2]
    valid = (uvw[:, 2] > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)

    hsv       = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
    rgb_feats = np.zeros((N, 3), dtype=np.float32)
    rgb_feats[valid] = hsv[v[valid], u[valid]]

    return np.hstack([features, rgb_feats])


FEATURE_NAMES = [
    "slope_deg",
    "roughness",
    "curvature",
    "height_above_ground",
    "intensity",
    "intensity_std",
    "wetness_proxy",
    "point_density",
]