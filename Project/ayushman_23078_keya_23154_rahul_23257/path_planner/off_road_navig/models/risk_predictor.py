"""
models/risk_predictor.py
-------------------------
Risk prediction training pipeline.

Three sources of risk supervision signal:

  1. SIMULATION-DERIVED
     Run terrain in a physics simulator, extract contact forces / slip
     at each visited node → ground-truth risk label.

  2. IMU/TELEMETRY-DERIVED  
     Parse historical robot logs: high vibration, slip events, stuck events
     → automatically generate (feature, risk) pairs via feedback_loop.py

  3. HUMAN ANNOTATION
     Label a small subset of nodes in CloudCompare / QGIS → risk 0-1.

This module:
  - RiskPredictorNet : the regression architecture (same as uncertainty_risk
                       MCDropoutRiskNet but optionally without MC Dropout for
                       deterministic deployment)
  - RiskDataset      : dataset from (features, labels, risk_targets)
  - RiskTrainer      : full training loop with validation
  - RiskEvaluator    : MAE, RMSE, per-label breakdown
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Tuple

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ═══════════════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════════════

if HAS_TORCH:
    class RiskPredictorNet(nn.Module):
        """
        Input:  (B, 9) = 8 terrain features + terrain label (normalised)
        Output: (B, 1) risk score in [0, 1]

        Two modes:
          deterministic : standard MLP, fast inference
          mc_dropout    : Dropout active at inference (see uncertainty_risk.py)
        """
        def __init__(self, in_dim: int = 9, dropout_p: float = 0.3):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(dropout_p),

                nn.Linear(128, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(dropout_p),

                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(dropout_p),

                nn.Linear(128, 64),
                nn.GELU(),
            )
            self.head = nn.Sequential(
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.encoder(x)).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════════════════

_BaseDataset = Dataset if HAS_TORCH else object

class RiskDataset(_BaseDataset):
    """
    Parameters
    ----------
    features    : (N, 8) float32
    labels      : (N,)   int — terrain class 0-4
    risk_targets: (N,)   float — ground-truth risk in [0,1]
    augment     : add Gaussian noise to features during training
    """

    def __init__(
        self,
        features    : np.ndarray,
        labels      : np.ndarray,
        risk_targets: np.ndarray,
        augment     : bool = True,
        noise_std   : float = 0.015,
    ):
        label_col       = labels.reshape(-1, 1).astype(np.float32) / 4.0
        self.X          = np.hstack([features, label_col]).astype(np.float32)
        self.y          = risk_targets.astype(np.float32)
        self.augment    = augment
        self.noise_std  = noise_std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        y = self.y[idx]
        if self.augment:
            x[:-1] += np.random.normal(0, self.noise_std, x[:-1].shape).astype(np.float32)
        if HAS_TORCH:
            return torch.tensor(x), torch.tensor(y)
        return x, y

    @classmethod
    def from_imu_logs(
        cls,
        features        : np.ndarray,
        labels          : np.ndarray,
        imu_data        : np.ndarray,   # (T, 6) [ax,ay,az,gx,gy,gz]
        node_assignments: np.ndarray,   # (T,) node index per IMU sample
        augment         : bool = True,
    ) -> "RiskDataset":
        """
        Build a dataset directly from IMU logs synced with node assignments.
        Great for exploiting the 1.5 TB archive automatically.

        For each node, aggregates all IMU samples to produce a risk target:
          - High linear accel variance → rough/rocky
          - High angular rate          → instability / slip
          - Sustained high accel       → stuck
        """
        N      = len(features)
        risks  = np.full(N, 0.25, dtype=np.float32)   # neutral prior

        accel_mag = np.linalg.norm(imu_data[:, :3], axis=1)
        gyro_mag  = np.linalg.norm(imu_data[:, 3:],  axis=1)

        # Accumulate per-node
        per_node_accel = {}
        per_node_gyro  = {}
        for t, nid in enumerate(node_assignments):
            if nid < 0 or nid >= N:
                continue
            per_node_accel.setdefault(nid, []).append(accel_mag[t])
            per_node_gyro.setdefault(nid,  []).append(gyro_mag[t])

        GRAVITY = 9.81
        for nid in per_node_accel:
            a_arr = np.array(per_node_accel[nid])
            g_arr = np.array(per_node_gyro[nid])

            a_mean = a_arr.mean()
            a_std  = a_arr.std()
            g_mean = g_arr.mean()

            # Heuristic mapping → risk
            accel_risk  = np.clip((a_mean - GRAVITY) / GRAVITY, 0, 1)
            jitter_risk = np.clip(a_std / (GRAVITY * 0.3), 0, 1)
            slip_risk   = np.clip(g_mean / 2.0, 0, 1)

            risks[nid] = float(np.clip(
                0.4 * accel_risk + 0.3 * jitter_risk + 0.3 * slip_risk, 0, 1
            ))

        return cls(features, labels, risks, augment=augment)


# ═══════════════════════════════════════════════════════════════════
#  Trainer
# ═══════════════════════════════════════════════════════════════════

class RiskTrainer:
    """
    Parameters
    ----------
    save_path  : where to checkpoint best model
    epochs     : training epochs
    lr         : learning rate
    batch_size : batch size
    patience   : early-stopping patience (epochs without improvement)
    """

    def __init__(
        self,
        save_path : str | Path = "models/weights/risk_predictor.pt",
        epochs    : int   = 80,
        lr        : float = 1e-3,
        batch_size: int   = 512,
        patience  : int   = 12,
    ):
        self.save_path  = Path(save_path)
        self.epochs     = epochs
        self.lr         = lr
        self.batch_size = batch_size
        self.patience   = patience

    def fit(
        self,
        model,
        train_ds : "RiskDataset",
        val_ds   : "RiskDataset",
    ) -> float:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model  = model.to(device)

        opt   = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=5, factor=0.5, min_lr=1e-5
        )

        # Huber loss: less sensitive to outlier risk labels than MSE
        crit  = nn.HuberLoss(delta=0.1)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                  shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=self.batch_size * 2,
                                  shuffle=False, num_workers=0)

        best_val   = float("inf")
        no_improve = 0

        for epoch in range(1, self.epochs + 1):
            model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = crit(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                train_loss += loss.item() * len(xb)
            train_loss /= len(train_ds)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    val_loss += crit(model(xb.to(device)), yb.to(device)).item() * len(xb)
            val_loss /= len(val_ds)

            sched.step(val_loss)

            marker = ""
            if val_loss < best_val:
                best_val   = val_loss
                no_improve = 0
                self.save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.cpu().state_dict(), self.save_path)
                model.to(device)
                marker = " ← best"
            else:
                no_improve += 1

            if epoch % 10 == 0:
                print(f"  Epoch {epoch:>3d}/{self.epochs}  "
                      f"train={train_loss:.5f}  val={val_loss:.5f}{marker}")

            if no_improve >= self.patience:
                print(f"  Early stopping at epoch {epoch}.")
                break

        print(f"Best val Huber loss: {best_val:.5f}  →  {self.save_path}")
        return best_val


# ═══════════════════════════════════════════════════════════════════
#  Evaluator
# ═══════════════════════════════════════════════════════════════════

class RiskEvaluator:
    @staticmethod
    def evaluate(
        model    ,
        val_ds   : "RiskDataset",
        label_names: list | None = None,
    ) -> dict:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required.")

        label_names = label_names or ["ground","obstacle","water","vegetation","unknown"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model  = model.to(device).eval()
        loader = DataLoader(val_ds, batch_size=1024, shuffle=False)

        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in loader:
                preds.append(model(xb.to(device)).cpu().numpy())
                trues.append(yb.numpy())

        pred = np.concatenate(preds)
        true = np.concatenate(trues)
        # Recover label from last column of X
        labels = (val_ds.X[:, -1] * 4).round().astype(int)

        mae   = float(np.abs(pred - true).mean())
        rmse  = float(np.sqrt(((pred - true) ** 2).mean()))
        r2    = float(1 - np.var(pred - true) / (np.var(true) + 1e-9))

        per_label = {}
        for lbl_id, name in enumerate(label_names):
            mask = labels == lbl_id
            if mask.sum() == 0:
                continue
            per_label[name] = {
                "mae" : float(np.abs(pred[mask] - true[mask]).mean()),
                "rmse": float(np.sqrt(((pred[mask] - true[mask]) ** 2).mean())),
                "n"   : int(mask.sum()),
                "mean_pred": float(pred[mask].mean()),
                "mean_true": float(true[mask].mean()),
            }

        return {
            "mae"      : mae,
            "rmse"     : rmse,
            "r2"       : r2,
            "per_label": per_label,
        }


# ═══════════════════════════════════════════════════════════════════
#  Quick-start helper
# ═══════════════════════════════════════════════════════════════════

def quick_train(
    features     : np.ndarray,
    labels       : np.ndarray,
    risk_targets : np.ndarray,
    save_path    : str | Path = "models/weights/risk_predictor.pt",
    val_fraction : float = 0.15,
    epochs       : int   = 80,
):
    """
    One-liner training entry point.

    Example
    -------
    from models.risk_predictor import quick_train
    quick_train(features, labels, risk_targets)
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required.")

    N     = len(features)
    perm  = np.random.permutation(N)
    n_val = int(N * val_fraction)

    train_ds = RiskDataset(features[perm[n_val:]], labels[perm[n_val:]],
                            risk_targets[perm[n_val:]], augment=True)
    val_ds   = RiskDataset(features[perm[:n_val]], labels[perm[:n_val]],
                            risk_targets[perm[:n_val]], augment=False)

    model   = RiskPredictorNet()
    trainer = RiskTrainer(save_path=save_path, epochs=epochs)
    trainer.fit(model, train_ds, val_ds)

    metrics = RiskEvaluator.evaluate(model, val_ds)
    print(f"  MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  R²={metrics['r2']:.4f}")
    return model, metrics