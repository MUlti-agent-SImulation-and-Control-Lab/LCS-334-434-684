"""
models/pointnet_segmenter.py
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
import sys

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

NUM_CLASSES = 5
CLASS_NAMES  = ["ground", "obstacle", "water", "vegetation", "unknown"]
CLASS_WEIGHTS = [1.0, 3.0, 12.0, 4.0, 6.0]   # mirrors terrain_classifier.py


if HAS_TORCH:

    # ─────────────────────────────────────────────────────────────────
    #  Geometry helpers
    # ─────────────────────────────────────────────────────────────────

    def _square_dist(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        """(B,N,C) × (B,M,C) → (B,N,M) squared L2"""
        dist  = -2 * torch.bmm(src, dst.transpose(1, 2))
        dist +=  (src ** 2).sum(-1, keepdim=True)
        dist +=  (dst ** 2).sum(-1, keepdim=True).transpose(1, 2)
        return dist.clamp(min=0)

    def _fps(xyz: torch.Tensor, n_points: int) -> torch.Tensor:
        """
        Farthest Point Sampling.
        xyz: (B, N, 3) → indices: (B, n_points)

        Bug fix from original: `B, N, _ = xyz.device, ...` unpacked
        device into B — now correctly unpacked from xyz.shape.
        """
        B, N, _ = xyz.shape          # ← fixed (was xyz.device, xyz.shape[0], ...)
        device  = xyz.device
        sampled  = torch.zeros(B, n_points, dtype=torch.long, device=device)
        dist     = torch.full((B, N), 1e10, device=device)
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)

        for i in range(n_points):
            sampled[:, i] = farthest
            centroid = xyz[torch.arange(B, device=device), farthest].unsqueeze(1)
            d    = ((xyz - centroid) ** 2).sum(-1)
            dist = torch.minimum(dist, d)
            farthest = dist.argmax(dim=1)

        return sampled

    def _ball_query(
        radius: float, k: int,
        xyz: torch.Tensor, query: torch.Tensor
    ) -> torch.Tensor:
        """
        Ball query — returns (B, M, K) indices.
        Out-of-radius slots are filled with the first valid neighbour.
        """
        sq_dists  = _square_dist(query, xyz)             # (B, M, N)
        idx       = sq_dists.argsort(dim=-1)[:, :, :k]  # (B, M, K)
        first_idx = idx[:, :, 0:1].expand_as(idx)
        mask      = sq_dists.gather(-1, idx) > radius ** 2
        idx[mask] = first_idx[mask]
        return idx

    # ─────────────────────────────────────────────────────────────────
    #  Building blocks
    # ─────────────────────────────────────────────────────────────────

    class SharedMLP(nn.Module):
        """
        Conv1d shared MLP: (B, C_in, N) → (B, C_out, N)
        Uses GELU instead of ReLU — better gradient flow on smooth features.
        Optional dropout for regularisation inside SA layers.
        """
        def __init__(self, in_ch: int, out_ch: int,
                     bn: bool = True, dropout: float = 0.0):
            super().__init__()
            self.conv = nn.Conv1d(in_ch, out_ch, 1, bias=not bn)
            self.bn   = nn.BatchNorm1d(out_ch) if bn else nn.Identity()
            self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        def forward(self, x):
            return self.drop(F.gelu(self.bn(self.conv(x))))

    class SetAbstraction(nn.Module):
        """
        PointNet++ Set Abstraction layer.
        Added per-layer dropout (0.1) to regularise encoder.
        """
        def __init__(self, n_center: int, radius: float, k: int,
                     in_ch: int, mlp_dims: list, dropout: float = 0.1):
            super().__init__()
            self.n_center = n_center
            self.radius   = radius
            self.k        = k

            layers  = []
            last_ch = in_ch + 3
            for i, dim in enumerate(mlp_dims):
                # only apply dropout on non-final layers in each SA block
                d = dropout if i < len(mlp_dims) - 1 else 0.0
                layers.append(SharedMLP(last_ch, dim, dropout=d))
                last_ch = dim
            self.mlp = nn.Sequential(*layers)

        def forward(self, xyz: torch.Tensor, features: "torch.Tensor | None"):
            B, N, _ = xyz.shape

            idx_fps = _fps(xyz, self.n_center)
            new_xyz = xyz[torch.arange(B, device=xyz.device).unsqueeze(1), idx_fps]

            idx_ball    = _ball_query(self.radius, self.k, xyz, new_xyz)
            grouped_xyz = xyz.unsqueeze(1).expand(-1, self.n_center, -1, -1)
            grouped_xyz = grouped_xyz.gather(
                2, idx_ball.unsqueeze(-1).expand(-1, -1, -1, 3)
            )
            grouped_xyz -= new_xyz.unsqueeze(2)

            if features is not None:
                feat_t  = features.transpose(1, 2)
                grouped = feat_t.unsqueeze(1).expand(-1, self.n_center, -1, -1)
                grouped = grouped.gather(
                    2, idx_ball.unsqueeze(-1).expand(-1, -1, -1, feat_t.shape[-1])
                )
                grouped = torch.cat([grouped_xyz, grouped], dim=-1)
            else:
                grouped = grouped_xyz

            B2, nc = B, self.n_center
            Ch     = grouped.shape[-1]
            grouped = grouped.permute(0, 3, 2, 1)       # (B, Ch, K, nc)
            grouped = grouped.reshape(B2 * nc, Ch, self.k)
            grouped = self.mlp(grouped)                  # (B*nc, mlp_out, K)
            new_feat = grouped.max(dim=-1)[0]            # (B*nc, mlp_out)
            new_feat = new_feat.reshape(B2, nc, -1).transpose(1, 2)
            return new_xyz, new_feat

    class FeaturePropagation(nn.Module):
        """Inverse-distance interpolation upsample + skip connection."""
        def __init__(self, in_ch: int, mlp_dims: list):
            super().__init__()
            layers, last_ch = [], in_ch
            for dim in mlp_dims:
                layers.append(SharedMLP(last_ch, dim))
                last_ch = dim
            self.mlp = nn.Sequential(*layers)

        def forward(self, xyz1, xyz2, feat1, feat2):
            B, N, _ = xyz1.shape
            M = xyz2.shape[1]

            if M == 1:
                interp = feat2.expand(-1, -1, N)
            else:
                dists, idx = _square_dist(xyz1, xyz2).topk(3, dim=-1, largest=False)
                dists   = dists.clamp(min=1e-10)
                weights = 1.0 / dists
                weights = weights / weights.sum(-1, keepdim=True)
                feat2_t = feat2.transpose(1, 2)
                interp  = (
                    feat2_t.unsqueeze(1)
                    .expand(-1, N, -1, -1)
                    .gather(2, idx.unsqueeze(-1).expand(-1, -1, -1, feat2_t.shape[-1]))
                    * weights.unsqueeze(-1)
                ).sum(2).transpose(1, 2)

            new_feat = torch.cat([interp, feat1], dim=1) if feat1 is not None else interp
            return self.mlp(new_feat)

    class PointNetPP(nn.Module):
        """
        PointNet++ segmentation.
        Input:  (B, N, 3) xyz + optional (B, N, C) features
        Output: (B, num_classes, N) logits

        Changes from original:
          - _fps bug fixed (device unpacking)
          - SharedMLP uses GELU
          - SA layers have encoder dropout (0.1)
          - Head dropout reduced 0.5 → 0.3 (0.5 was too aggressive for 5-class)
          - Wider SA3 (64 centres instead of 32) for better global context
        """
        def __init__(self, in_feat_dim: int = 1, num_classes: int = NUM_CLASSES):
            super().__init__()
            self.sa1 = SetAbstraction(512, 0.5,  32,  in_feat_dim, [32,  32,  64],  dropout=0.1)
            self.sa2 = SetAbstraction(128, 1.5,  64,  64,          [64,  64,  128], dropout=0.1)
            self.sa3 = SetAbstraction(64,  4.0,  128, 128,         [128, 128, 256], dropout=0.1)

            self.fp3 = FeaturePropagation(256 + 128, [256, 256])
            self.fp2 = FeaturePropagation(256 + 64,  [256, 128])
            self.fp1 = FeaturePropagation(128 + in_feat_dim, [128, 128, 128])

            self.head = nn.Sequential(
                nn.Conv1d(128, 128, 1),
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.Dropout(0.3),          # reduced from 0.5
                nn.Conv1d(128, 64,  1),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Conv1d(64, num_classes, 1),
            )

        def forward(self, xyz, features=None):
            feat0 = features.transpose(1, 2) if features is not None else None
            xyz1, f1 = self.sa1(xyz,  feat0)
            xyz2, f2 = self.sa2(xyz1, f1)
            xyz3, f3 = self.sa3(xyz2, f2)
            f2 = self.fp3(xyz2, xyz3, f2, f3)
            f1 = self.fp2(xyz1, xyz2, f1, f2)
            f0 = self.fp1(xyz,  xyz1, feat0, f1)
            return self.head(f0)


# ═══════════════════════════════════════════════════════════════════
#  High-level wrapper
# ═══════════════════════════════════════════════════════════════════

class PointNetPPSegmenter:
    def __init__(
        self,
        weights_path : str | Path | None = None,
        in_feat_dim  : int  = 8,      # changed default: use all 8 features not just intensity
        num_classes  : int  = NUM_CLASSES,
        chunk_size   : int  = 8192,
    ):
        self._model       = None
        self._chunk_size  = chunk_size
        self._in_feat_dim = in_feat_dim
        self._device      = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"

        if weights_path is not None and HAS_TORCH:
            p = Path(weights_path)
            if p.exists():
                self._model = PointNetPP(in_feat_dim, num_classes).to(self._device)
                self._model.load_state_dict(
                    torch.load(p, map_location=self._device)
                )
                self._model.eval()
                print(f"[PointNetPP] Loaded weights from {p}")
            else:
                print(f"[PointNetPP] Weights not found at {p} — heuristic fallback.")

    def predict(self, cloud: np.ndarray,
                features: "np.ndarray | None" = None) -> np.ndarray:
        if self._model is None or not HAS_TORCH:
            from terrain.segmenter import TerrainSegmenter
            if features is None:
                raise ValueError("features required for heuristic fallback")
            return TerrainSegmenter._heuristic_predict(features)
        return self._chunked_predict(cloud, features)

    def _chunked_predict(self, cloud, features):
        N      = len(cloud)
        labels = np.zeros(N, dtype=np.int32)

        for start in range(0, N, self._chunk_size):
            end       = min(start + self._chunk_size, N)
            chunk_xyz = cloud[start:end, :3]
            pad_n     = self._chunk_size - (end - start)

            if pad_n > 0:
                chunk_xyz = np.vstack([chunk_xyz,
                                       np.tile(chunk_xyz[-1:], (pad_n, 1))])

            xyz_t  = torch.tensor(chunk_xyz[None],
                                  dtype=torch.float32, device=self._device)
            feat_t = None

            if features is not None:
                chunk_feat = features[start:end, :self._in_feat_dim]
                if pad_n > 0:
                    chunk_feat = np.vstack([chunk_feat,
                                            np.tile(chunk_feat[-1:], (pad_n, 1))])
                feat_t = torch.tensor(chunk_feat[None],
                                      dtype=torch.float32, device=self._device)
            elif cloud.shape[1] > 3:
                intensity = cloud[start:end, 3:4]
                if pad_n > 0:
                    intensity = np.vstack([intensity,
                                           np.tile(intensity[-1:], (pad_n, 1))])
                feat_t = torch.tensor(intensity[None],
                                      dtype=torch.float32, device=self._device)

            with torch.no_grad():
                logits = self._model(xyz_t, feat_t)
                preds  = logits[0].argmax(dim=0).cpu().numpy()

            labels[start:end] = preds[:end - start]

            pct = end / N
            bar = "█" * int(30 * pct) + "░" * (30 - int(30 * pct))
            sys.stdout.write(f"\r  [{bar}] {pct*100:.0f}%  ({end:,}/{N:,})")
            sys.stdout.flush()

        print()
        return labels


# ═══════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════

def train(
    model        : "PointNetPP",
    clouds       : list,
    label_arrays : list,
    save_path    : str | Path = "models/weights/pointnetpp.pt",
    epochs       : int   = 80,
    lr           : float = 5e-4,        # lowered from 1e-3 — PointNet++ is sensitive
    chunk_size   : int   = 8192,
    in_feat_dim  : int   = 8,           # use all features by default
    patience     : int   = 15,
    label_smoothing: float = 0.05,
):
    """
    Train PointNet++ with:
      - Class-weighted + label-smoothed loss
      - AdamW optimiser
      - Warmup (5 ep) + cosine LR decay
      - Early stopping
      - Progress bar per epoch
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required.")

    from torch.utils.data import Dataset, DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[PointNetPP train] device={device}  epochs={epochs}  lr={lr}")
    model = model.to(device)

    # ── Loss: weighted + label smoothing ──────────────────────────────
    weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device)
    crit    = nn.CrossEntropyLoss(weight=weights,
                                  label_smoothing=label_smoothing,
                                  ignore_index=-1)

    # ── Optimiser + schedule ──────────────────────────────────────────
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    warmup = 5

    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        t = (ep - warmup) / max(epochs - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # ── Dataset ───────────────────────────────────────────────────────
    class CloudDataset(Dataset):
        def __init__(self, clouds, labels, chunk_size, in_feat_dim):
            self.clouds      = clouds
            self.labels      = labels
            self.chunk_size  = chunk_size
            self.in_feat_dim = in_feat_dim

        def __len__(self):
            return len(self.clouds)

        def __getitem__(self, idx):
            cloud = self.clouds[idx]
            label = self.labels[idx]
            N     = len(cloud)

            if N >= self.chunk_size:
                sel = np.random.choice(N, self.chunk_size, replace=False)
            else:
                sel = np.concatenate([
                    np.arange(N),
                    np.random.choice(N, self.chunk_size - N, replace=True)
                ])

            xyz = torch.tensor(cloud[sel, :3], dtype=torch.float32)
            lbl = torch.tensor(label[sel],     dtype=torch.long)

            if cloud.shape[1] > 3:
                feat = torch.tensor(
                    cloud[sel, 3:3 + self.in_feat_dim], dtype=torch.float32
                )
            else:
                feat = torch.zeros(self.chunk_size, self.in_feat_dim)

            # ── Point cloud augmentation ───────────────────────────────
            # 1. Random jitter
            xyz  += torch.randn_like(xyz) * 0.01
            # 2. Random scale (0.9–1.1×)
            xyz  *= torch.empty(1).uniform_(0.9, 1.1).item()
            # 3. Random rotation around Z axis (gravity axis stays fixed)
            angle = torch.empty(1).uniform_(0, 2 * np.pi).item()
            c, s  = np.cos(angle), np.sin(angle)
            R     = torch.tensor([[c, -s, 0],
                                   [s,  c, 0],
                                   [0,  0, 1]], dtype=torch.float32)
            xyz   = (R @ xyz.T).T

            return xyz, feat, lbl

    ds     = CloudDataset(clouds, label_arrays, chunk_size, in_feat_dim)
    loader = DataLoader(ds, batch_size=4, shuffle=True,
                        num_workers=0, pin_memory=(device == "cuda"))

    save_path  = Path(save_path)
    best_loss  = float("inf")
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_i, (xyz_b, feat_b, lbl_b) in enumerate(loader):
            xyz_b  = xyz_b.to(device)
            feat_b = feat_b.to(device)
            lbl_b  = lbl_b.to(device)

            opt.zero_grad()
            logits = model(xyz_b, feat_b)     # (B, C, N)
            loss   = crit(logits, lbl_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

            # per-batch progress
            pct = (batch_i + 1) / len(loader)
            bar = "█" * int(20 * pct) + "░" * (20 - int(20 * pct))
            sys.stdout.write(
                f"\r  Epoch {epoch:>3d}/{epochs}  [{bar}]  "
                f"loss={loss.item():.4f}"
            )
            sys.stdout.flush()

        sched.step()
        avg    = total_loss / len(loader)
        lr_now = opt.param_groups[0]["lr"]

        if avg < best_loss:
            best_loss, no_improve = avg, 0
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.cpu().state_dict(), save_path)
            model.to(device)
            marker = " ← best"
        else:
            no_improve += 1
            marker = ""

        print(f"\r  Epoch {epoch:>3d}/{epochs}  "
              f"loss={avg:.4f}  best={best_loss:.4f}  "
              f"lr={lr_now:.2e}{marker}          ")

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}.")
            break

    print(f"Training complete. Best model → {save_path}")