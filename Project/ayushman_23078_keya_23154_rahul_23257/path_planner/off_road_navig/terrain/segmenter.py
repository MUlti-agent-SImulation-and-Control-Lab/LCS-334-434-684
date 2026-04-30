"""
terrain/segmenter.py
--------------------
Terrain segmentation using a lightweight PointNet-style MLP.

Label map
---------
  0 = GROUND       (traversable, low risk)
  1 = ROCK/OBSTACLE (non-traversable, high risk)
  2 = WATER/PUDDLE  (traversable but risky, slippery)
  3 = VEGETATION    (moderate risk, may entangle)
  4 = UNKNOWN       (sparse / edge region)

The model is a 3-layer MLP operating on the 8 per-point features produced
by feature_extractor.py.  It can be trained with labelled data or used in
rule-based (heuristic) mode out of the box.
"""

from __future__ import annotations
import numpy as np
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Label constants
LABEL_GROUND     = 0
LABEL_OBSTACLE   = 1
LABEL_WATER      = 2
LABEL_VEGETATION = 3
LABEL_UNKNOWN    = 4

LABEL_NAMES = {
    0: "ground",
    1: "obstacle",
    2: "water",
    3: "vegetation",
    4: "unknown",
}

LABEL_COLORS = {   # RGB, 0-1 floats for visualisation
    0: (0.40, 0.75, 0.35),   # green
    1: (0.85, 0.20, 0.10),   # red
    2: (0.10, 0.45, 0.90),   # blue
    3: (0.10, 0.55, 0.20),   # dark green
    4: (0.70, 0.70, 0.70),   # grey
}


# ── Neural model definition ────────────────────────────────────────────────

if HAS_TORCH:
    class TerrainMLP(nn.Module):
        """
        Input:  (B, 8) feature vector per point
        Output: (B, 5) class logits
        """
        def __init__(self, in_dim: int = 8, num_classes: int = 5):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, num_classes),
            )

        def forward(self, x):
            return self.net(x)


class TerrainSegmenter:
    """
    High-level wrapper. Falls back to heuristic classification if no
    pretrained weights are provided or torch is unavailable.
    """

    def __init__(self, weights_path: str | Path | None = None):
        self._model = None
        self._device = "cpu"

        if weights_path is not None and HAS_TORCH:
            weights_path = Path(weights_path)
            if weights_path.exists():
                self._model = TerrainMLP()
                state = torch.load(weights_path, map_location="cpu")
                self._model.load_state_dict(state)
                self._model.eval()

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        features : (N, 8) from feature_extractor.extract_features()

        Returns
        -------
        labels : (N,) int array with values in 0..4
        """
        if self._model is not None and HAS_TORCH:
            return self._neural_predict(features)
        return self._heuristic_predict(features)

    # ── Neural path ───────────────────────────────────────────────────────

    def _neural_predict(self, features: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32)
            logits = self._model(x)
            labels = logits.argmax(dim=-1).numpy().astype(np.int32)
        return labels

    # ── Heuristic path (no weights needed) ───────────────────────────────

    @staticmethod
    def _heuristic_predict(features: np.ndarray) -> np.ndarray:
        """
        Column indices (see feature_extractor.FEATURE_NAMES):
        0 slope_deg | 1 roughness | 2 curvature | 3 height_above_ground
        4 intensity | 5 intensity_std | 6 wetness_proxy | 7 point_density

        Tuned for dense forest environments (TartanAir-style).
        """
        n      = len(features)
        labels = np.full(n, LABEL_GROUND, dtype=np.int32)

        slope         = features[:, 0]
        roughness     = features[:, 1]
        curvature     = features[:, 2]
        height        = features[:, 3]
        intensity     = features[:, 4]
        intensity_std = features[:, 5]
        wetness       = features[:, 6]
        density       = features[:, 7]

        # Percentile-normalise density so thresholds are dataset-agnostic
        d10, d90      = np.percentile(density, [10, 90])
        density_norm  = np.clip((density - d10) / (d90 - d10 + 1e-9), 0, 1)

        # ── Unknown: edge of scan / extremely sparse ───────────────────
        labels[(density < 3) | (density_norm < 0.05)] = LABEL_UNKNOWN

        # ── Obstacle: steep OR rough+elevated OR high curvature ────────
        obstacle = (
            (slope > 25)                              # was 15
            | ((roughness > 0.15) & (height > 0.30)) # was 0.20, 0.15
            | ((curvature > 0.25) & (height > 0.30)) # was 0.10
        )
        labels[obstacle] = LABEL_OBSTACLE

        # ── Water: flat, smooth, low, high wetness ─────────────────────
        water = (
            (wetness   > 0.75)    # was 0.50 — needs stronger wetness signal
            & (slope   < 5)       # was 8
            & (roughness < 0.05)  # was 0.10
            & (height  < 0.10)    # was 0.20 — must be very close to ground
            & (intensity < 0.25)  # ADD: water has low LiDAR reflectance
        )
        labels[water] = LABEL_WATER

        # ── Vegetation: dense returns + mixed reflectance + elevated ───
        veg = (
            (roughness    >  0.08) & (roughness    <= 0.35)
            & (intensity  <  0.55)
            & (density_norm  > 0.35)
            & (intensity_std > 0.05)
            & (height     >  0.05)
            & (labels     == LABEL_GROUND)
        )
        labels[veg] = LABEL_VEGETATION

        # ── Ground refinement: high curvature + rough = root/rock ─────
        labels[
            (labels == LABEL_GROUND)
            & (curvature > 0.30)
            & (roughness > 0.12)
        ] = LABEL_OBSTACLE

        # ── Final unknown sweep ────────────────────────────────────────
        labels[(density < 2) & (labels == LABEL_GROUND)] = LABEL_UNKNOWN

        return labels


# ── Training helper (use when you have labelled data) ─────────────────────

def train(
    features: np.ndarray,
    labels: np.ndarray,
    save_path: str | Path = "models/weights/terrain_classifier.pt",
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 512,
):
    """
    Train the TerrainMLP on labelled feature arrays.

    Parameters
    ----------
    features  : (N, 8) float32
    labels    : (N,) int  (0-4)
    save_path : where to save the trained weights
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for training.")

    import torch
    from torch.utils.data import TensorDataset, DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TerrainMLP().to(device)

    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(X, y)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    crit  = torch.nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
        sched.step()
        if epoch % 10 == 0:
            avg = total_loss / len(dataset)
            print(f"  Epoch {epoch:>3d}/{epochs}  loss={avg:.4f}")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), save_path)
    print(f"Saved weights → {save_path}")
    return model