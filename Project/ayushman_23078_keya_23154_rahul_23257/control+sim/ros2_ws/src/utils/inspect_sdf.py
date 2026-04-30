import trimesh
import numpy as np

mesh_path = '/home/ayushman/ext_proj/ecs334/simulation/ros2_ws/src/clearpath_simulator/clearpath_gz/meshes/office/office_construction.dae'

loaded = trimesh.load(mesh_path, force='mesh')

if isinstance(loaded, trimesh.Scene):
    print(f"Scene with {len(loaded.geometry)} sub-meshes:")
    for name, geom in loaded.geometry.items():
        print(f"  '{name}': {len(geom.vertices)} vertices, bounds: {geom.bounds}")
else:
    print(f"Single mesh: {len(loaded.vertices)} vertices")
    print(f"Bounds (min): {loaded.bounds[0]}")
    print(f"Bounds (max): {loaded.bounds[1]}")
    print(f"Extents (size): {loaded.extents}")