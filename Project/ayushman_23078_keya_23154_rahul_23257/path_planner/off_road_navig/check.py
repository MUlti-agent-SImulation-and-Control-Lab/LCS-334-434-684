
import numpy as np
from lidar.simulation import load_simulation_cloud
from lidar.preprocessor import preprocess
from terrain.feature_extractor import extract_features
from terrain.segmenter import TerrainSegmenter, LABEL_NAMES

cloud = load_simulation_cloud('simulation_lidar/full_map_outdoor_shifted.pcd')
cloud_pp = preprocess(cloud, voxel_size=0.02, roi=None, nb_neighbors=0)
features = extract_features(cloud_pp, radius=0.5)

seg = TerrainSegmenter()
labels = seg.predict(features)

unique, counts = np.unique(labels, return_counts=True)
for u, c in zip(unique, counts):
    print(f'{LABEL_NAMES[u]}: {c} ({100*c/len(labels):.1f}%)')

print(f'Slope: mean={features[:,0].mean():.1f} median={np.median(features[:,0]):.1f} p90={np.percentile(features[:,0],90):.1f}')
print(f'Roughness: mean={features[:,1].mean():.3f} median={np.median(features[:,1]):.3f} p90={np.percentile(features[:,1],90):.3f}')
print(f'Height: mean={features[:,3].mean():.3f} median={np.median(features[:,3]):.3f} p90={np.percentile(features[:,3],90):.3f}')
