"""
models/terrain_classifier.py
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import List, Tuple

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


NUM_CLASSES = 5
CLASS_NAMES = ["ground", "obstacle", "water", "vegetation", "unknown"]

# Recomputed from typical TartanAir forest distributions.
# Higher weight = rarer class = penalise misses more.
DEFAULT_CLASS_WEIGHTS = [1.0, 3.0, 12.0, 4.0, 6.0]


# ═══════════════════════════════════════════════════════════════════
#  Improved MLP — deeper, with residual skip connection
# ═══════════════════════════════════════════════════════════════════

if HAS_TORCH:
    class TerrainMLP(nn.Module):
        """
        Deeper MLP with a residual connection and label smoothing support.
        Input:  (B, in_dim)      — 8 geometric features, or 11 with RGB
        Output: (B, num_classes) — raw logits
        """
        def __init__(self, in_dim: int = 8, num_classes: int = NUM_CLASSES):
            super().__init__()

            # ── Stem: project input to hidden dim ─────────────────────
            self.stem = nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.BatchNorm1d(64),
                nn.GELU(),
            )

            # ── Block 1 ───────────────────────────────────────────────
            self.block1 = nn.Sequential(
                nn.Linear(64, 128),
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(128, 128),
                nn.BatchNorm1d(128),
                nn.GELU(),
            )
            # project stem output to block1 output dim for skip
            self.skip1 = nn.Linear(64, 128, bias=False)

            # ── Block 2 ───────────────────────────────────────────────
            self.block2 = nn.Sequential(
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.GELU(),
            )
            # skip: identity since dims match (128 → 128)

            # ── Head ──────────────────────────────────────────────────
            self.head = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, num_classes),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            s   = self.stem(x)
            b1  = self.block1(s) + self.skip1(s)   # residual
            b2  = self.block2(b1) + b1              # residual (same dim)
            return self.head(b2)


# ═══════════════════════════════════════════════════════════════════
#  Dataset  —  richer augmentation
# ═══════════════════════════════════════════════════════════════════

_BaseDataset = Dataset if HAS_TORCH else object

class TerrainDataset(_BaseDataset):
    """
    Per-point feature dataset with stronger augmentation.

    Feature columns (must match feature_extractor output):
      0 slope_deg | 1 roughness | 2 curvature | 3 height
      4 intensity | 5 intensity_std | 6 wetness | 7 density
    """

    # Columns where sign-flip makes physical sense
    _FLIP_COLS = []          # none for these features
    # Columns that can be scaled without changing class meaning
    _SCALE_COLS = [1, 2, 4, 5, 6]   # roughness, curv, intensity, int_std, wetness

    def __init__(
        self,
        features : np.ndarray,
        labels   : np.ndarray,
        augment  : bool = True,
        noise_std: float = 0.015,    # tighter than before — 0.02 was too noisy
    ):
        self.features  = features.astype(np.float32)
        self.labels    = labels.astype(np.int64)
        self.augment   = augment
        self.noise_std = noise_std

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx].copy()
        lbl  = self.labels[idx]

        if self.augment:
            # 1. Gaussian noise on all features
            feat += np.random.normal(0, self.noise_std, feat.shape).astype(np.float32)

            # 2. Random scaling of non-geometric features (±15%)
            scale = np.ones(feat.shape, dtype=np.float32)
            for c in self._SCALE_COLS:
                scale[c] = np.random.uniform(0.85, 1.15)
            feat *= scale

            # 3. Randomly zero out intensity (simulates sensor dropout)
            if np.random.rand() < 0.05:
                feat[4] = 0.0
                feat[5] = 0.0

            # 4. Slope jitter: small additive perturbation (± 2°)
            feat[0] += np.random.uniform(-2.0, 2.0)
            feat[0]  = np.clip(feat[0], 0.0, 90.0)

            feat = np.clip(feat, -10, 10)

        if HAS_TORCH:
            return torch.tensor(feat), torch.tensor(lbl)
        return feat, lbl


# ═══════════════════════════════════════════════════════════════════
#  Dataset builder  —  unchanged API, better auto-label rebalancing
# ═══════════════════════════════════════════════════════════════════

class DatasetBuilder:
    def __init__(
        self,
        data_root           : str | Path,
        label_suffix        : str   = "_labels.npy",
        auto_label          : bool  = True,
        val_fraction        : float = 0.15,
        max_points_per_scan : int   = 100_000,
    ):
        self.data_root    = Path(data_root)
        self.label_suffix = label_suffix
        self.auto_label   = auto_label
        self.val_fraction = val_fraction
        self.max_pts      = max_points_per_scan

    def build(self) -> Tuple["TerrainDataset", "TerrainDataset"]:
        from terrain.feature_extractor import extract_features
        from terrain.segmenter import TerrainSegmenter

        all_features, all_labels = [], []
        seg = TerrainSegmenter() if self.auto_label else None

        scan_paths = sorted(
            p for p in self.data_root.iterdir()
            if p.suffix in (".las", ".laz", ".pcd", ".ply", ".npy", ".txt")
            and not p.stem.endswith("_labels")
        )
        if not scan_paths:
            raise FileNotFoundError(f"No scans in {self.data_root}")
        print(f"[DatasetBuilder] {len(scan_paths)} scans found.")

        for scan_path in scan_paths:
            try:
                cloud = np.load(scan_path)
                cloud = cloud[~np.isnan(cloud).any(axis=1)]
                feats = extract_features(cloud, radius=1.0)
                N     = len(cloud)

                if N > self.max_pts:
                    idx   = np.random.choice(N, self.max_pts, replace=False)
                    feats = feats[idx]
                    N     = self.max_pts

                label_path = scan_path.with_name(scan_path.stem + self.label_suffix)
                if label_path.exists():
                    lbls = np.load(label_path).astype(np.int32)
                    if len(lbls) > N:
                        lbls = lbls[idx]
                elif self.auto_label and seg is not None:
                    lbls = seg.predict(feats)
                else:
                    print(f"  Skipping {scan_path.name} — no labels")
                    continue

                all_features.append(feats)
                all_labels.append(lbls)
                print(f"  {scan_path.name}: {N:,} pts")
            except Exception as e:
                print(f"  Error — {scan_path.name}: {e}")

        if not all_features:
            raise RuntimeError("No valid scans loaded.")

        X, y = self._balance_and_split(
            np.vstack(all_features), np.concatenate(all_labels)
        )
        return X, y

    def _balance_and_split(self, X, y):
        """
        Oversample rare classes (water, unknown) to at most 3× median count
        before splitting, so val set also sees rare classes.
        """
        counts  = np.bincount(y, minlength=NUM_CLASSES).astype(float)
        median  = np.median(counts[counts > 0])
        cap     = int(median * 3)

        X_bal, y_bal = [X], [y]
        for c in range(NUM_CLASSES):
            idx = np.where(y == c)[0]
            deficit = cap - len(idx)
            if deficit > 0 and len(idx) > 0:
                extra = np.random.choice(idx, deficit, replace=True)
                X_bal.append(X[extra])
                y_bal.append(y[extra])

        X = np.vstack(X_bal)
        y = np.concatenate(y_bal)
        perm = np.random.permutation(len(X))
        X, y = X[perm], y[perm]

        n_val    = int(len(X) * self.val_fraction)
        train_ds = TerrainDataset(X[n_val:], y[n_val:], augment=True)
        val_ds   = TerrainDataset(X[:n_val],  y[:n_val],  augment=False)
        print(f"[DatasetBuilder] Train: {len(train_ds):,}  Val: {len(val_ds):,}")
        return train_ds, val_ds

    @classmethod
    def from_tartanground_npz(
        cls,
        npz_dir         : str,
        val_fraction    : float = 0.15,
        max_pts_per_file: int   = 80_000,
    ) -> tuple:
        npz_dir = Path(npz_dir)
        files   = sorted(npz_dir.glob("traj_*.npz"))
        if not files:
            raise FileNotFoundError(f"No traj_*.npz in {npz_dir}")

        print(f"[DatasetBuilder] {len(files)} npz files...")
        all_f, all_l = [], []
        for fpath in files:
            data  = np.load(fpath)
            feats = data["features"]
            lbls  = data["labels"]
            N     = len(feats)
            if N > max_pts_per_file:
                idx   = np.random.choice(N, max_pts_per_file, replace=False)
                feats = feats[idx]
                lbls  = lbls[idx]
            all_f.append(feats)
            all_l.append(lbls)

        X    = np.vstack(all_f)
        y    = np.concatenate(all_l)
        perm = np.random.permutation(len(X))
        X, y = X[perm], y[perm]
        n_val    = int(len(X) * val_fraction)
        train_ds = TerrainDataset(X[n_val:], y[n_val:], augment=True)
        val_ds   = TerrainDataset(X[:n_val],  y[:n_val],  augment=False)
        print(f"[DatasetBuilder] Train: {len(train_ds):,}  Val: {len(val_ds):,}")
        return train_ds, val_ds


# ═══════════════════════════════════════════════════════════════════
#  Trainer  —  label smoothing + warmup + better LR schedule
# ═══════════════════════════════════════════════════════════════════

class Trainer:
    def __init__(
        self,
        save_path     : str | Path    = "models/weights/terrain_classifier.pt",
        epochs        : int           = 80,
        lr            : float         = 3e-4,
        batch_size    : int           = 1024,
        class_weights : List[float] | None = None,
        patience      : int           = 15,
        label_smoothing: float        = 0.05,
        warmup_epochs : int           = 5,
    ):
        self.save_path       = Path(save_path)
        self.epochs          = epochs
        self.lr              = lr
        self.batch_size      = batch_size
        self.class_weights   = class_weights or DEFAULT_CLASS_WEIGHTS
        self.patience        = patience
        self.label_smoothing = label_smoothing
        self.warmup_epochs   = warmup_epochs

    def fit(self, model, train_ds, val_ds):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Trainer] device={device}  epochs={self.epochs}  "
              f"batch={self.batch_size}  lr={self.lr}")
        model = model.to(device)

        weights = torch.tensor(self.class_weights, dtype=torch.float32, device=device)
        # Label smoothing reduces overconfidence — helps with noisy auto-labels
        crit = nn.CrossEntropyLoss(weight=weights,
                                   label_smoothing=self.label_smoothing)

        opt   = torch.optim.AdamW(model.parameters(),
                                  lr=self.lr, weight_decay=1e-3)

        # Warmup then cosine decay
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / self.warmup_epochs
            progress = (epoch - self.warmup_epochs) / (self.epochs - self.warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        # Balanced sampler
        label_arr  = train_ds.labels
        class_freq = np.bincount(label_arr, minlength=NUM_CLASSES).astype(float)
        class_freq = np.where(class_freq == 0, 1, class_freq)
        sample_w   = (1.0 / class_freq)[label_arr]
        sampler    = WeightedRandomSampler(
            torch.tensor(sample_w, dtype=torch.float32),
            num_samples=len(train_ds), replacement=True
        )

        train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                  sampler=sampler, num_workers=0)
        val_loader   = DataLoader(val_ds, batch_size=self.batch_size * 2,
                                  shuffle=False, num_workers=0)

        best_val, no_improve = float("inf"), 0

        for epoch in range(1, self.epochs + 1):
            # ── Train ─────────────────────────────────────────────────
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

            # ── Validate ──────────────────────────────────────────────
            model.eval()
            val_loss = 0.0
            all_preds, all_true = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits  = model(xb)
                    val_loss += crit(logits, yb).item() * len(xb)
                    all_preds.append(logits.argmax(1).cpu().numpy())
                    all_true.append(yb.cpu().numpy())
            val_loss /= len(val_ds)
            sched.step()

            preds = np.concatenate(all_preds)
            true  = np.concatenate(all_true)
            miou  = float(np.nanmean(_compute_iou(preds, true, NUM_CLASSES)))
            acc   = float((preds == true).mean())

            if val_loss < best_val:
                best_val, no_improve = val_loss, 0
                self.save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.cpu().state_dict(), self.save_path)
                model.to(device)
                marker = " ← best"
            else:
                no_improve += 1
                marker = ""

            if epoch % 5 == 0 or epoch == 1:
                lr_now = opt.param_groups[0]["lr"]
                print(f"  Epoch {epoch:>3d}/{self.epochs}  "
                      f"train={train_loss:.4f}  val={val_loss:.4f}  "
                      f"mIoU={miou:.3f}  acc={acc:.3f}  "
                      f"lr={lr_now:.2e}{marker}")

            if no_improve >= self.patience:
                print(f"  Early stopping at epoch {epoch}.")
                break

        print(f"Best val loss: {best_val:.4f}  →  {self.save_path}")
        return best_val


# ═══════════════════════════════════════════════════════════════════
#  Evaluator  —  vectorised confusion matrix
# ═══════════════════════════════════════════════════════════════════

class Evaluator:
    @staticmethod
    def evaluate(model, val_ds, batch_size: int = 2048) -> dict:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model  = model.to(device).eval()
        loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        all_preds, all_true = [], []
        with torch.no_grad():
            for xb, yb in loader:
                logits = model(xb.to(device))
                all_preds.append(logits.argmax(1).cpu().numpy())
                all_true.append(yb.numpy())

        preds = np.concatenate(all_preds)
        true  = np.concatenate(all_true)
        iou   = _compute_iou(preds, true, NUM_CLASSES)
        miou  = float(np.nanmean(iou))
        acc   = float((preds == true).mean())

        per_class = {
            CLASS_NAMES[i]: {
                "iou": float(iou[i]) if not np.isnan(iou[i]) else None,
                "f1" : float(_f1(preds, true, i)),
                "n"  : int((true == i).sum()),
            }
            for i in range(NUM_CLASSES)
        }

        print(f"\n{'Class':<14} {'IoU':>6} {'F1':>6} {'N':>8}")
        print("─" * 38)
        for name, m in per_class.items():
            iou_s = f"{m['iou']:.3f}" if m['iou'] is not None else "  N/A"
            print(f"  {name:<12} {iou_s:>6} {m['f1']:>6.3f} {m['n']:>8,}")
        print(f"  {'mIoU':<12} {miou:>6.3f}")
        print(f"  {'accuracy':<12} {acc:>6.3f}")

        return {
            "mean_iou"        : miou,
            "accuracy"        : acc,
            "per_class"       : per_class,
            "confusion_matrix": _confusion_matrix(true, preds, NUM_CLASSES).tolist(),
        }


# ── Metric helpers ─────────────────────────────────────────────────────────

def _compute_iou(pred, true, n_cls):
    iou = np.full(n_cls, np.nan)
    for c in range(n_cls):
        tp = ((pred == c) & (true == c)).sum()
        fp = ((pred == c) & (true != c)).sum()
        fn = ((pred != c) & (true == c)).sum()
        if tp + fp + fn > 0:
            iou[c] = tp / (tp + fp + fn)
    return iou

def _f1(pred, true, cls):
    tp = ((pred == cls) & (true == cls)).sum()
    fp = ((pred == cls) & (true != cls)).sum()
    fn = ((pred != cls) & (true == cls)).sum()
    if tp == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 2 * p * r / (p + r + 1e-9)

def _confusion_matrix(true, pred, n_cls):
    # vectorised — O(N) not O(N²)
    mask = (true >= 0) & (true < n_cls) & (pred >= 0) & (pred < n_cls)
    cm   = np.zeros((n_cls, n_cls), dtype=np.int64)
    np.add.at(cm, (true[mask], pred[mask]), 1)
    return cm