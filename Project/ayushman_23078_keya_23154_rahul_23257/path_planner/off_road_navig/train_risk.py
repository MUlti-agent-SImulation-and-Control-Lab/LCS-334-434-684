import sys, numpy as np
sys.path.insert(0, '.')
from pathlib import Path
from models.risk_predictor import RiskDataset, RiskPredictorNet, RiskTrainer
 
# Load .npz files
npz_dir = Path('off_road_navig/data/training')
files   = sorted(npz_dir.glob('traj_*.npz'))
print(f'Loading {len(files)} files...')
 
all_f, all_l, all_r = [], [], []
for fpath in files:
    data = np.load(fpath)
    all_f.append(data['features'])
    all_l.append(data['labels'])
    all_r.append(data['risks'])
 
X = np.vstack(all_f)
y_l = np.concatenate(all_l)
y_r = np.concatenate(all_r)
 
perm  = np.random.permutation(len(X))
X, y_l, y_r = X[perm], y_l[perm], y_r[perm]
n_val = int(len(X) * 0.15)
 
train_ds = RiskDataset(X[n_val:], y_l[n_val:], y_r[n_val:], augment=True)
val_ds   = RiskDataset(X[:n_val],  y_l[:n_val],  y_r[:n_val],  augment=False)
 
model   = RiskPredictorNet()
trainer = RiskTrainer(
    save_path = 'off_road_navig/models/weights/risk_predictor.pt',
    epochs    = 80,
    batch_size= 1024,
)
trainer.fit(model, train_ds, val_ds)
