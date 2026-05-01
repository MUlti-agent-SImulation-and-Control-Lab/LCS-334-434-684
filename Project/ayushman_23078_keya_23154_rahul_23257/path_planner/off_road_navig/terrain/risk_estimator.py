"""
terrain/risk_estimator.py
--------------------------
Produces a scalar risk score in [0, 1] for every point in the cloud.

Risk is a weighted combination of:
  - terrain_label_risk : base risk by terrain class
  - slope_risk         : increases with angle
  - roughness_risk     : increases with surface roughness
  - wetness_risk       : puddle / slippery surface probability
  - height_risk        : points significantly above ground → obstacle
  - curvature_risk     : abrupt curvature changes → edge of obstacle

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

from .segmenter import (
    LABEL_GROUND, LABEL_OBSTACLE, LABEL_WATER,
    LABEL_VEGETATION, LABEL_UNKNOWN,
)

# Base risk per label class
_LABEL_BASE_RISK = {
    LABEL_GROUND:     0.05,   
    LABEL_OBSTACLE:   0.95,   
    LABEL_WATER:      0.70,   
    LABEL_VEGETATION: 0.25,   # lower from 0.40
    LABEL_UNKNOWN:    0.45,   # lower from 0.60
}

# Blend weights — must sum to 1
_WEIGHTS = {
    "label":     0.50,   # was 0.30 — make label dominate
    "slope":     0.20,
    "roughness": 0.10,   # was 0.15
    "wetness":   0.10,   # was 0.20
    "height":    0.07,
    "curvature": 0.03,
}


# ── Neural risk head ────────────────────────────────────────────────────
if HAS_TORCH:
    class RiskMLP(nn.Module):
        def __init__(self, in_dim: int = 9):
            super().__init__()
            self.encoder = nn.Sequential(
                # Block 1: Index 0, 1, 2, 3
                nn.Linear(in_dim, 128),      # encoder.0
                nn.BatchNorm1d(128),         # encoder.1
                nn.ReLU(),                   # encoder.2
                nn.Dropout(0.1),             # encoder.3
                
                # Block 2: Index 4, 5, 6, 7
                nn.Linear(128, 256),         # encoder.4
                nn.BatchNorm1d(256),         # encoder.5
                nn.ReLU(),                   # encoder.6
                nn.Dropout(0.1),             # encoder.7
                
                # Block 3: Index 8, 9, 10, 11
                nn.Linear(256, 128),         # encoder.8
                nn.BatchNorm1d(128),         # encoder.9
                nn.ReLU(),                   # encoder.10
                nn.Dropout(0.1),             # encoder.11
                
                # Block 4: Index 12, 13
                nn.Linear(128, 64),          # encoder.12
                nn.ReLU()                    # encoder.13
            )
            
            self.head = nn.Sequential(
                nn.Linear(64, 1),            # head.0
                nn.Sigmoid()
            )

        def forward(self, x):
            # Ensure we handle 1D input for single points if needed
            if x.ndim == 1:
                x = x.unsqueeze(0)
            x = self.encoder(x)
            return self.head(x).squeeze()
# ── Main class ──────────────────────────────────────────────────────────

class RiskEstimator:
    """
    Assign a per-point risk score.

    Usage
    -----
    estimator = RiskEstimator(weights_path=None)   # heuristic mode
    risk_scores = estimator.estimate(features, labels)
    """

    def __init__(self, weights_path: str | Path | None = None):
        self._model = None
        if weights_path is not None and HAS_TORCH:
            p = Path(weights_path)
            if p.exists():
                self._model = RiskMLP()
                state_dict = torch.load(p, map_location="cpu")
                self._model.load_state_dict(state_dict, strict = False)
                self._model.eval()

    def estimate(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        features : (N, 8)  from feature_extractor
        labels   : (N,)    from segmenter

        Returns
        -------
        risk : (N,) float32 in [0, 1]
        """
        if self._model is not None and HAS_TORCH:
            return self._neural_estimate(features, labels)
        return self._heuristic_estimate(features, labels)

    # ── Neural path ──────────────────────────────────────────────────────

    def _neural_estimate(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        import torch
        label_col = labels.reshape(-1, 1).astype(np.float32) / 4.0   # normalise 0-1
        x = np.hstack([features, label_col])
        with torch.no_grad():
            risk = self._model(torch.tensor(x, dtype=torch.float32)).numpy()
        return risk.astype(np.float32)

    # ── Heuristic path ───────────────────────────────────────────────────

    @staticmethod
    def _heuristic_estimate(
        features: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """
        Feature column indices:
          0 slope_deg | 1 roughness | 2 curvature | 3 height
          4 intensity | 5 int_std  | 6 wetness    | 7 density
        """
        n = len(features)

        # ── Label risk ──────────────────────────────────────────────────
        label_risk = np.array([_LABEL_BASE_RISK.get(int(l), 0.6) for l in labels],
                              dtype=np.float32)

        # ── Slope risk (sigmoid centred at 20°, saturates at 40°) ───────
        slope_norm = np.clip(features[:, 0] / 40.0, 0, 1)
        slope_risk = _sigmoid(slope_norm, midpoint=0.5, steepness=6.0)

        # ── Roughness risk ───────────────────────────────────────────────
        max_rough = features[:, 1].max() + 1e-9
        rough_risk = np.clip(features[:, 1] / 0.3, 0, 1)

        # ── Wetness risk ─────────────────────────────────────────────────
        wetness_risk = features[:, 6]   # already [0,1]

        # ── Height risk (tall protrusions = obstacle) ────────────────────
        max_h = max(features[:, 3].max(), 0.1)
        height_risk = np.clip(features[:, 3] / 2.0, 0, 1)

        # ── Curvature risk ───────────────────────────────────────────────
        max_c = max(features[:, 2].max(), 1e-9)
        curv_risk = np.clip(features[:, 2] / 0.5, 0, 1)

        w = _WEIGHTS
        risk = (
            w["label"]     * label_risk
            + w["slope"]     * slope_risk
            + w["roughness"] * rough_risk
            + w["wetness"]   * wetness_risk
            + w["height"]    * height_risk
            + w["curvature"] * curv_risk
        ).astype(np.float32)

        return np.clip(risk, 0.0, 1.0)


def _sigmoid(x: np.ndarray, midpoint: float = 0.5, steepness: float = 8.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-steepness * (x - midpoint)))


# ── Training helper ─────────────────────────────────────────────────────

def train(
    features: np.ndarray,
    labels: np.ndarray,
    risk_targets: np.ndarray,
    save_path: str | Path = "models/weights/risk_predictor.pt",
    epochs: int = 60,
    lr: float = 1e-3,
):
    """
    Train RiskMLP given ground-truth risk annotations.

    risk_targets : (N,) float in [0,1]  (human-labelled or simulation-derived)
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required.")

    import torch
    from torch.utils.data import TensorDataset, DataLoader

    label_col = labels.reshape(-1, 1).astype(np.float32) / 4.0
    X = np.hstack([features, label_col])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RiskMLP().to(device)
    opt  = torch.optim.Adam(model.parameters(), lr=lr)
    crit = torch.nn.MSELoss()

    ds     = TensorDataset(torch.tensor(X, dtype=torch.float32),
                            torch.tensor(risk_targets, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=512, shuffle=True)

    for epoch in range(1, epochs + 1):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}/{epochs}  loss={loss.item():.5f}")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), save_path)
    print(f"Saved → {save_path}")