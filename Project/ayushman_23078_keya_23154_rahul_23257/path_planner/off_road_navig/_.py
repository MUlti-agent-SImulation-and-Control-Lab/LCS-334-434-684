import numpy as np
from terrain.feature_extractor import extract_features

raw = np.load("/home/ayushman/ext_proj/ecs334/merged_cloud.npy")
raw = raw[~np.isnan(raw).any(axis=1)]
xyz = raw[:, :3]

print(f"XYZ range:  x=[{xyz[:,0].min():.2f}, {xyz[:,0].max():.2f}]")
print(f"            y=[{xyz[:,1].min():.2f}, {xyz[:,1].max():.2f}]")
print(f"            z=[{xyz[:,2].min():.2f}, {xyz[:,2].max():.2f}]")

feats = extract_features(raw, radius=1.0)
names = ["slope_deg","roughness","curvature","height","intensity","intensity_std","wetness","density"]
print(f"\n{'Feature':<15} {'min':>8} {'p10':>8} {'p50':>8} {'p90':>8} {'max':>8}")
print("─" * 55)
for i, name in enumerate(names):
    col = feats[:, i]
    p10, p50, p90 = np.percentile(col, [10, 50, 90])
    print(f"{name:<15} {col.min():>8.3f} {p10:>8.3f} {p50:>8.3f} {p90:>8.3f} {col.max():>8.3f}")

print(f"\nLabel distribution with CURRENT thresholds:")
from terrain.segmenter import TerrainSegmenter, LABEL_NAMES
seg = TerrainSegmenter()
labels = seg.predict(feats)
N = len(labels)
for lbl, name in LABEL_NAMES.items():
    cnt = (labels == lbl).sum()
    bar = "█" * int(40 * cnt / N)
    print(f"  {name:<12} {bar:<40} {100*cnt/N:5.1f}%")