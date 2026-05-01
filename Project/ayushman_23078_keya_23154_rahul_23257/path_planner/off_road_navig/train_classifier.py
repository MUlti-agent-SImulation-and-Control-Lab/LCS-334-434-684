import sys
sys.path.insert(0, '.')
from models.terrain_classifier import DatasetBuilder, Trainer
from terrain.segmenter import TerrainMLP
 
# Load from the .npz files built in Phase 4
train_ds, val_ds = DatasetBuilder.from_tartanground_npz(
    npz_dir      = 'off_road_navig/data/training',
    val_fraction = 0.15,
)
 
model   = TerrainMLP()
trainer = Trainer(
    save_path  = 'off_road_navig/models/weights/terrain_classifier.pt',
    epochs     = 60,
    batch_size = 2048,
)
trainer.fit(model, train_ds, val_ds)
