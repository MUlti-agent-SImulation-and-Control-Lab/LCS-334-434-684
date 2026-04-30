# sdf_to_pcd.py
import xml.etree.ElementTree as ET
import numpy as np
import open3d as o3d
import trimesh
import os

# ── CONFIGURE THESE ──────────────────────────────────────────
SDF_PATH    = '/root/ros2_ws/outdoor.sdf'  # your exported world sdf
OUTPUT_PCD  = '/root/ros2_ws/full_map_outdoor.pcd'
POINTS_PER_MESH = 100000

# Skip the robot itself — only want the environment
SKIP_MODELS = ['a200_0000/robot']
# ─────────────────────────────────────────────────────────────

def resolve_uri(uri):
    uri = uri.strip()

    # file:// → strip prefix and use path directly
    if uri.startswith('file://'):
        path = uri.replace('file://', '')
        if os.path.exists(path):
            print(f"  ✓ {path}")
            return path
        else:
            print(f"  ✗ Not found: {path}")
            return None

    # model:// → not expected in exported SDF but handle anyway
    if uri.startswith('model://'):
        print(f"  ✗ Unresolved model:// URI: {uri}")
        return None

    # bare path
    if os.path.exists(uri):
        print(f"  ✓ {uri}")
        return uri

    print(f"  ✗ Not found: {uri}")
    return None


def load_mesh(mesh_path):
    try:
        loaded = trimesh.load(mesh_path, force='mesh')
        if isinstance(loaded, trimesh.Scene):
            meshes = list(loaded.geometry.values())
            if not meshes:
                return None
            print(f"  → Scene with {len(meshes)} sub-meshes, merging...")
            return trimesh.util.concatenate(meshes)
        return loaded
    except Exception as e:
        print(f"  ✗ Load error: {e}")
        return None


def sdf_to_pcd(sdf_path, output_path, points_per_mesh):
    tree = ET.parse(sdf_path)
    root = tree.getroot()
    all_points = []

    for model in root.iter('model'):
        model_name = model.get('name')

        if model_name in SKIP_MODELS:
            print(f"\n── Skipping robot: '{model_name}'")
            continue

        pose_elem = model.find('pose')
        pose_str  = pose_elem.text.strip() if pose_elem is not None else "0 0 0 0 0 0"
        x, y, z, roll, pitch, yaw = map(float, pose_str.split())
        print(f"\n── Model: '{model_name}' | pose: [{pose_str}]")

        seen_uris = set()
        for uri_elem in model.iter('uri'):
            uri = uri_elem.text.strip()
            if uri in seen_uris:
                continue
            seen_uris.add(uri)

            mesh_path = resolve_uri(uri)
            if mesh_path is None:
                continue

            # Only process mesh files
            if not mesh_path.endswith(('.dae', '.stl', '.obj', '.ply')):
                print(f"  ~ Skipping non-mesh: {os.path.basename(mesh_path)}")
                continue

            mesh = load_mesh(mesh_path)
            if mesh is None or len(mesh.vertices) == 0:
                print(f"  ✗ Empty mesh")
                continue

            print(f"  Sampling {points_per_mesh} points from {os.path.basename(mesh_path)}...", end='', flush=True)
            points, _ = trimesh.sample.sample_surface(mesh, points_per_mesh)
            print(f" done ✓ ({len(points)} pts)")

            # Apply world pose transform
            R = trimesh.transformations.euler_matrix(roll, pitch, yaw)[:3, :3]
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3]  = [x, y, z]
            points_h  = np.hstack([points, np.ones((len(points), 1))])
            points    = (T @ points_h.T).T[:, :3]

            all_points.append(points)

    print(f"\n{'='*50}")
    if not all_points:
        print("ERROR: No points collected.")
        return None

    combined = np.vstack(all_points)
    print(f"Total points before downsample: {combined.shape[0]}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined)
    pcd = pcd.voxel_down_sample(voxel_size=0.02)
    print(f"Total points after downsample:  {len(pcd.points)}")

    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Saved → {output_path}")
    return pcd


pcd = sdf_to_pcd(SDF_PATH, OUTPUT_PCD, POINTS_PER_MESH)
if pcd:
    print("\nVisualizing... (press Q to quit)")
    o3d.visualization.draw_geometries([pcd])