"""
models/uncertainty_risk.py
---------------------------
Uncertainty-Aware Risk Estimation using Monte Carlo Dropout.

Why this matters
----------------
Standard risk estimation gives a single point estimate: risk=0.4.
But that could mean "we're very confident it's 0.4" OR "we have no
idea, it could be 0.1 or 0.9". In off-road navigation, high
*uncertainty* should itself be treated as high risk (unknown = unsafe).

MC Dropout approach
-------------------
During inference, we run the model T=30 times with Dropout ACTIVE.
Each run gives a slightly different prediction. The variance across
runs gives us the epistemic uncertainty.

Final output per point:
  risk_mean : float  — average predicted risk
  risk_std  : float  — std across T forward passes (uncertainty)
  risk_upper: float  — risk_mean + k*risk_std  (conservative estimate)

The planner uses risk_upper, so uncertain regions are treated conservatively.

Usage
-----
  est = UncertaintyRiskEstimator(weights_path="models/weights/uncertainty_risk.pt")
  result = est.estimate(features, labels)
  # result.risk_mean  (N,)
  # result.risk_std   (N,)
  # result.risk_upper (N,)  ← feed this into graph builder
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ═══════════════════════════════════════════════════════════════════
#  Result container
# ═══════════════════════════════════════════════════════════════════

@dataclass
class UncertaintyResult:
    risk_mean  : np.ndarray   # (N,) mean risk estimate
    risk_std   : np.ndarray   # (N,) epistemic uncertainty
    risk_upper : np.ndarray   # (N,) conservative: mean + k*std
    high_uncertainty_mask: np.ndarray  # (N,) bool — std > threshold


# ═══════════════════════════════════════════════════════════════════
#  Model definition
# ═══════════════════════════════════════════════════════════════════

if HAS_TORCH:
    class MCDropoutRiskNet(nn.Module):
        """
        Risk regression MLP with Dropout at every layer.
        Dropout stays ACTIVE during inference for MC sampling.

        Input:  (B, 9) = 8 features + normalised label
        Output: (B, 1) risk in [0,1] via sigmoid
        """
        def __init__(self, in_dim: int = 9, dropout_p: float = 0.3):
            super().__init__()
            self.dropout_p = dropout_p
            self.net = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x).squeeze(-1)

        def enable_dropout(self):
            """Force dropout layers on even in eval mode (for MC sampling)."""
            for m in self.modules():
                if isinstance(m, nn.Dropout):
                    m.train()


# ═══════════════════════════════════════════════════════════════════
#  Main estimator class
# ═══════════════════════════════════════════════════════════════════

class UncertaintyRiskEstimator:
    """
    Parameters
    ----------
    weights_path   : path to saved .pt weights (None = heuristic fallback)
    n_mc_samples   : number of stochastic forward passes (default 30)
    uncertainty_k  : risk_upper = mean + k * std  (default 1.5)
    uncertainty_thr: std above this → flagged as high-uncertainty
    """

    def __init__(
        self,
        weights_path    : str | Path | None = None,
        n_mc_samples    : int   = 30,
        uncertainty_k   : float = 1.5,
        uncertainty_thr : float = 0.12,
        dropout_p       : float = 0.3,
    ):
        self.n_mc         = n_mc_samples
        self.k            = uncertainty_k
        self.unc_thr      = uncertainty_thr
        self._model       = None
        self._device      = "cpu"

        if weights_path is not None and HAS_TORCH:
            p = Path(weights_path)
            if p.exists():
                self._model = MCDropoutRiskNet(dropout_p=dropout_p)
                self._model.load_state_dict(torch.load(p, map_location="cpu"))
                self._model.eval()
                print(f"[UncertaintyRisk] Loaded weights from {p}")
            else:
                print(f"[UncertaintyRisk] Weights not found at {p}. Heuristic fallback.")

    def estimate(
        self,
        features : np.ndarray,   # (N, 8)
        labels   : np.ndarray,   # (N,) int
    ) -> UncertaintyResult:
        """
        Returns UncertaintyResult with mean, std, upper-bound risk arrays.
        """
        if self._model is not None and HAS_TORCH:
            return self._mc_estimate(features, labels)
        return self._heuristic_with_uncertainty(features, labels)

    # ── MC Dropout path ──────────────────────────────────────────────

    def _mc_estimate(self, features: np.ndarray, labels: np.ndarray) -> UncertaintyResult:
        import torch

        label_col = labels.reshape(-1, 1).astype(np.float32) / 4.0
        x = torch.tensor(np.hstack([features, label_col]), dtype=torch.float32)

        self._model.eval()
        self._model.enable_dropout()   # activate dropout for MC

        samples = []
        with torch.no_grad():
            for _ in range(self.n_mc):
                samples.append(self._model(x).numpy())

        samples    = np.stack(samples, axis=0)   # (T, N)
        risk_mean  = samples.mean(axis=0).astype(np.float32)
        risk_std   = samples.std(axis=0).astype(np.float32)
        risk_upper = np.clip(risk_mean + self.k * risk_std, 0.0, 1.0).astype(np.float32)
        hi_unc     = risk_std > self.unc_thr

        return UncertaintyResult(
            risk_mean=risk_mean,
            risk_std=risk_std,
            risk_upper=risk_upper,
            high_uncertainty_mask=hi_unc,
        )

    # ── Heuristic fallback with synthetic uncertainty ─────────────────

    def _heuristic_with_uncertainty(
        self, features: np.ndarray, labels: np.ndarray
    ) -> UncertaintyResult:
        from terrain.risk_estimator import RiskEstimator

        base_risk = RiskEstimator._heuristic_estimate(features, labels)

        # Synthetic uncertainty: low point density and unknown label → high std
        density   = features[:, 7]
        max_d     = density.max() + 1e-9
        norm_d    = density / max_d

        unknown_mask = (labels == 4).astype(np.float32)
        risk_std  = (
            0.05                               # base noise
            + (1.0 - norm_d) * 0.08           # sparse = uncertain
            + unknown_mask * 0.15             # unknown label = very uncertain
            + features[:, 6] * 0.05           # wet = uncertain
        ).astype(np.float32)
        risk_std  = np.clip(risk_std, 0.0, 0.3)

        risk_upper = np.clip(base_risk + self.k * risk_std, 0.0, 1.0)

        return UncertaintyResult(
            risk_mean=base_risk,
            risk_std=risk_std,
            risk_upper=risk_upper,
            high_uncertainty_mask=(risk_std > self.unc_thr),
        )


# ═══════════════════════════════════════════════════════════════════
#  Training helper
# ═══════════════════════════════════════════════════════════════════

def train(
    features     : np.ndarray,
    labels       : np.ndarray,
    risk_targets : np.ndarray,
    save_path    : str | Path = "models/weights/uncertainty_risk.pt",
    epochs       : int   = 80,
    lr           : float = 1e-3,
    dropout_p    : float = 0.3,
    batch_size   : int   = 512,
):
    """
    Train MCDropoutRiskNet.

    risk_targets: (N,) float in [0,1] — ground-truth or simulation-derived risk.
    Can be collected from:
      - Human annotation of traversability
      - Robot IMU: high vibration / slip → high risk replay
      - Simulation: assign risk from known terrain types
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required.")

    import torch
    from torch.utils.data import TensorDataset, DataLoader

    label_col = labels.reshape(-1, 1).astype(np.float32) / 4.0
    X = np.hstack([features, label_col])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = MCDropoutRiskNet(dropout_p=dropout_p).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr * 10, total_steps=epochs * (len(X) // batch_size + 1)
    )
    crit   = nn.MSELoss()

    ds     = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(risk_targets, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    best = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(xb)

        avg = total / len(ds)
        if avg < best:
            best = avg
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.cpu().state_dict(), save_path)
            model.to(device)

        if epoch % 20 == 0:
            print(f"  Epoch {epoch}/{epochs}  mse={avg:.5f}  best={best:.5f}")

    print(f"Saved → {save_path}")