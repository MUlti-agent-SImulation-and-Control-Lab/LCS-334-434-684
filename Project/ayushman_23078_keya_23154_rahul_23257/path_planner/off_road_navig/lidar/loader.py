"""
lidar/loader.py
---------------
Unified point cloud loader.

Supported formats
-----------------
  .pcd          PCD (TartanGround primary LiDAR format)
  .ply          PLY (TartanGround secondary LiDAR format)
  .npy / .npz   NumPy arrays
  .las / .laz   Survey-grade LiDAR
  .txt / .csv   Space/comma-separated xyz

Output
------
All loaders return (N, C) float32 where columns 0-2 are x, y, z.
Column 3 is intensity when available (normalised to [0,1]).
"""

from __future__ import annotations
import numpy as np
from pathlib import Path


def load_point_cloud(filepath: str | Path) -> np.ndarray:
    """
    Auto-detect format and load. Returns (N, C) float32, cols = x,y,z[,intensity].
    """
    path = Path(filepath)
    ext  = path.suffix.lower()

    dispatch = {
        ".pcd": _load_pcd,
        ".ply": _load_ply,
        ".npy": _load_npy,
        ".npz": _load_npz,
        ".las": _load_las,
        ".laz": _load_las,
        ".txt": _load_text,
        ".csv": _load_text,
    }
    if ext not in dispatch:
        raise ValueError(
            f"Unsupported format '{ext}'. "
            f"Supported: {list(dispatch)}"
        )
    return dispatch[ext](path)


# ── PCD (TartanGround primary format) ──────────────────────────────

def _load_pcd(path: Path) -> np.ndarray:
    """
    Parse PCD files (ASCII, binary, binary_compressed).
    TartanGround LiDAR scans are stored as .pcd.
    Falls back to Open3D for binary_compressed files.
    """
    header = {}
    header_lines = 0
    with open(path, "rb") as f:
        for line in f:
            header_lines += 1
            line_str = line.decode("utf-8", errors="ignore").strip()
            if line_str.startswith("#"):
                continue
            parts = line_str.split()
            if not parts:
                continue
            key = parts[0].upper()
            if key in ("FIELDS", "SIZE", "TYPE", "COUNT", "WIDTH",
                       "HEIGHT", "POINTS", "DATA"):
                header[key] = parts[1:]
            if key == "DATA":
                data_type = parts[1].lower()
                break

    if "FIELDS" not in header:
        # Malformed or unreadable — fall back to Open3D
        return _load_open3d(path)

    fields    = header["FIELDS"]
    n_points  = int(header.get("POINTS", ["0"])[0])
    data_type = header.get("DATA", ["ascii"])[0].lower()

    if data_type == "ascii":
        try:
            arr = np.loadtxt(path, comments=("#", *list(header.keys())),
                             dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
        except Exception:
            return _load_open3d(path)
    else:
        # binary or binary_compressed → Open3D handles this cleanly
        return _load_open3d(path)

    return _extract_xyz_intensity(arr, fields)


def _extract_xyz_intensity(arr: np.ndarray, fields: list) -> np.ndarray:
    """
    Pull x, y, z columns and optionally intensity from a raw field array.
    """
    fields_lower = [f.lower() for f in fields]

    def _col(name):
        try:
            return arr[:, fields_lower.index(name)]
        except ValueError:
            return None

    x = _col("x");  y = _col("y");  z = _col("z")
    if x is None or y is None or z is None:
        # No explicit x/y/z labels — assume first 3 columns
        return arr[:, :3].astype(np.float32)

    cols = [x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)]

    intensity = _col("intensity") or _col("i")
    if intensity is not None:
        mx = intensity.max()
        if mx > 0:
            intensity = intensity / mx
        cols.append(intensity.astype(np.float32))

    return np.column_stack(cols)


# ── PLY (TartanGround secondary format) ───────────────────────────

def _load_ply(path: Path) -> np.ndarray:
    return _load_open3d(path)


# ── Open3D fallback (handles binary PCD, PLY, etc.) ───────────────

def _load_open3d(path: Path) -> np.ndarray:
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError(
            "open3d is required for .pcd/.ply files: pip install open3d"
        )
    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points, dtype=np.float32)
    
    # Check if we got any points
    if len(points) == 0:
        raise ValueError(f"Loaded 0 points from {path}. File may be corrupt or empty.")

    if pcd.has_colors():
        colors    = np.asarray(pcd.colors, dtype=np.float32)
        intensity = colors.mean(axis=1, keepdims=True)
        return np.hstack([points, intensity])

    return points


# ── NumPy ──────────────────────────────────────────────────────────

def _load_npy(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    if arr.ndim == 1:
        # flat 1-D → try reshape
        if len(arr) % 3 == 0:
            arr = arr.reshape(-1, 3)
        elif len(arr) % 4 == 0:
            arr = arr.reshape(-1, 4)
        else:
            raise ValueError(f"Cannot reshape 1-D array of length {len(arr)} into xyz")
    if arr.shape[1] < 3:
        raise ValueError(f"Expected ≥3 columns (x,y,z), got {arr.shape}")
    return arr.astype(np.float32)


def _load_npz(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    for key in ("points", "cloud", "xyz", "data", "lidar"):
        if key in data:
            return data[key].astype(np.float32)
    first = list(data.files)[0]
    return data[first].astype(np.float32)


# ── LAS / LAZ ─────────────────────────────────────────────────────

def _load_las(path: Path) -> np.ndarray:
    try:
        import laspy
    except ImportError:
        raise ImportError("pip install laspy[lazrs]")
    las  = laspy.read(str(path))
    cols = [np.array(las.x, np.float32),
            np.array(las.y, np.float32),
            np.array(las.z, np.float32)]
    if hasattr(las, "intensity"):
        i = np.array(las.intensity, np.float32)
        mx = i.max()
        if mx > 0:
            i /= mx
        cols.append(i)
    return np.column_stack(cols)


# ── Text / CSV ────────────────────────────────────────────────────

def _load_text(path: Path) -> np.ndarray:
    # Try comma first, then space
    for delim in (",", None):
        try:
            arr = np.loadtxt(path, delimiter=delim, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] >= 3:
                return arr
        except Exception:
            continue
    raise ValueError(f"Could not parse text file as xyz: {path}")


# ── Synthetic terrain generator ───────────────────────────────────

def generate_synthetic_terrain(
    n_points : int   = 50_000,
    size     : float = 50.0,
    n_rocks  : int   = 15,
    n_puddles: int   = 8,
    seed     : int   = 42,
) -> np.ndarray:
    """
    Generate a synthetic off-road terrain point cloud (N, 4): x, y, z, intensity.
    Used for testing without real data.
    """
    rng = np.random.default_rng(seed)
    x   = rng.uniform(0, size, n_points).astype(np.float32)
    y   = rng.uniform(0, size, n_points).astype(np.float32)
    z   = (
        2.0 * np.sin(x / 8.0) * np.cos(y / 8.0)
        + 1.0 * np.sin(x / 3.0 + 1.0)
        + 0.3 * rng.standard_normal(n_points)
    ).astype(np.float32)

    intensity = np.clip(
        0.6 + rng.standard_normal(n_points).astype(np.float32) * 0.05,
        0.0, 1.0
    )

    rock_cx = rng.uniform(5, size-5, n_rocks)
    rock_cy = rng.uniform(5, size-5, n_rocks)
    for cx, cy, r, h in zip(rock_cx, rock_cy,
                              rng.uniform(0.5, 2.0, n_rocks),
                              rng.uniform(0.5, 2.5, n_rocks)):
        d = np.sqrt((x-cx)**2 + (y-cy)**2)
        m = d < r
        z[m] += h * np.cos((np.pi/2.0) * d[m] / r)
        intensity[m] *= 0.85

    puddle_cx = rng.uniform(5, size-5, n_puddles)
    puddle_cy = rng.uniform(5, size-5, n_puddles)
    for cx, cy, r in zip(puddle_cx, puddle_cy, rng.uniform(1.0, 3.5, n_puddles)):
        d = np.sqrt((x-cx)**2 + (y-cy)**2)
        m = d < r
        base = z[m].mean() - 0.15
        z[m] = np.where(z[m] > base, base, z[m])
        intensity[m] = rng.uniform(0.05, 0.2, m.sum()).astype(np.float32)

    return np.column_stack([x, y, z, intensity])