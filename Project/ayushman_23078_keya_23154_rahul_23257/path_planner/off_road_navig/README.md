# 3D LiDAR Off-Road Pathfinder
### Risk-Aware Navigation with Joint Probability Lookahead

---

## Overview

READ THE PDF README TO RUN THE CODE, THIS FILE EXPLAINS THE IMPORTANT FOLDER STRUCTURED WHICH MUCH BE PRESERVED WHEN RUNNING ANY FILE.

This system takes a **LiDAR point cloud** of an off-road terrain and produces an optimal, safety-aware 3D path from a start to a goal. It accounts for rocks, puddles, slippery surfaces, steep slopes, and sharp turns — not just at individual waypoints, but **jointly across a configurable sliding window** of consecutive nodes.

```
LiDAR Cloud (.las/.pcd/.ply/.npy)
       │
       ▼
  Preprocessor          (downsample, denoise, normalise)
       │
       ▼
  Feature Extractor     (slope, roughness, wetness, curvature…)
       │
       ▼
  Terrain Segmenter     (ML/heuristic: ground | rock | water | veg)
       │
       ▼
  Risk Estimator        (per-point scalar risk [0,1])
       │
       ▼
  Graph Builder         (voxel-cluster → Node3D graph)
       │
       ▼
  Risk-Aware A*         (3D A* + joint-probability lookahead)
       │
       ▼
  Path Smoother         (cubic spline resampling)
       │
       ▼
  Visualiser / Output
```

---

## Folder Structure

```
lidar_pathfinder/
├── main.py                          # CLI entry point
├── requirements.txt
│
├── lidar/
│   ├── loader.py                    # Load .pcd/.las/.npy/.ply/.txt
│   ├── preprocessor.py              # Voxel downsample, outlier removal, normalise
│   └── visualizer.py                # Matplotlib / Open3D plots
│
├── terrain/
│   ├── feature_extractor.py         # Slope, roughness, curvature, wetness proxy
│   ├── segmenter.py                 # TerrainMLP (PyTorch) or heuristic fallback
│   └── risk_estimator.py            # RiskMLP or weighted heuristic → [0,1] score
│
├── graph/
│   ├── node.py                      # Node3D dataclass
│   ├── edge.py                      # Edge dataclass with traversal cost
│   └── builder.py                   # Point cloud → traversability graph
│
├── models/
│   ├── terrain_classifier.py        # PointNet-lite (training wrapper)
│   ├── risk_predictor.py            # Risk regression head (training wrapper)
│   └── weights/                     # Saved .pt model weights
│
├── pathfinding/
│   ├── algo.py                      # RiskAwarePlanner (A* + joint risk)
│   ├── joint_risk.py                # Joint probability over N-node windows
│   └── path_smoother.py             # Cubic spline smoothing
│
└── ui/
    └── dashboard.py                 # Rich CLI dashboard (optional)
```

---

## Quick Start

### Install
```bash
pip install -r requirements.txt
```

### Run on synthetic terrain (no data needed)
```bash
python main.py --synthetic --start 2 2 0 --goal 48 48 0
```

### Run on a real LiDAR file
```bash
python main.py --cloud terrain.las --start 5 3 0 --goal 80 90 0 --window 6
```

### Get 3 alternative paths
```bash
python main.py --synthetic --start 2 2 0 --goal 48 48 0 --alternatives 3
```

---

## Key Concepts

### Joint Probability Risk Window

Rather than evaluating each node in isolation, the algorithm maintains a
**sliding window of N consecutive nodes** and computes the joint probability
that at least one of them is a failure event:

```
JointRisk(window) = 1 - ∏ (1 - risk_i)
```

A **turn penalty** is added when the path requires sharp heading changes,
especially on sloped terrain — because turning on a hillside or near a puddle
is far more dangerous than a straight traverse.

This means:
- Node A (risk=0.2) followed by 4 puddle nodes (risk=0.7 each) → window risk ≈ 0.99
- Node B (risk=0.4) followed by 4 flat ground nodes (risk=0.1 each) → window risk ≈ 0.54

**The planner correctly prefers the path through B.**

### Terrain Classification

| Label | Class       | Heuristic Signal                      |
|-------|-------------|---------------------------------------|
| 0     | Ground      | Low slope, low roughness, mid intensity |
| 1     | Rock/Obstacle | High slope OR roughness, elevated z  |
| 2     | Water/Puddle | Low intensity, smooth, slight depression |
| 3     | Vegetation  | Moderate roughness, low intensity     |
| 4     | Unknown     | Sparse neighbourhood                  |

### Wetness Proxy

LiDAR intensity drops significantly on wet/muddy surfaces (water absorbs IR).
The wetness proxy combines:
- **Low intensity** → likely absorbing surface
- **Low roughness** → flat (pooled) surface
- **Low height** → depression (puddle collects in dips)

---

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--cloud PATH` | — | Input point cloud file |
| `--synthetic` | — | Use generated test terrain |
| `--start X Y Z` | required | Start world position |
| `--goal X Y Z` | required | Goal world position |
| `--window N` | 5 | Joint-risk lookahead window |
| `--voxel FLOAT` | 1.5 | Graph node voxel size (m) |
| `--connect FLOAT` | 4.0 | Node connection radius (m) |
| `--max-risk FLOAT` | 0.80 | Max joint risk before pruning |
| `--alternatives N` | 1 | Number of paths to return |
| `--weights-clf PATH` | None | Terrain classifier `.pt` weights |
| `--weights-risk PATH` | None | Risk estimator `.pt` weights |
| `--no-smooth` | off | Skip spline smoothing |
| `--save-graph PATH` | None | Cache graph to disk |
| `--load-graph PATH` | None | Load cached graph |

---

## Training Your Own Models

### Terrain Classifier
```python
from terrain.segmenter import train
# features: (N, 8), labels: (N,) with values 0-4
train(features, labels, save_path="models/weights/terrain_classifier.pt", epochs=50)
```

### Risk Estimator
```python
from terrain.risk_estimator import train
# risk_targets: (N,) float in [0,1] from human annotation or simulation
train(features, labels, risk_targets, save_path="models/weights/risk_predictor.pt")
```

---

## Supported File Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| NumPy | `.npy`, `.npz` | (N, 3+) arrays |
| LAS/LAZ | `.las`, `.laz` | Full LiDAR attributes |
| PCD | `.pcd` | Point Cloud Library format |
| PLY | `.ply` | Polygon File Format |
| Text | `.txt`, `.csv` | Comma-separated x,y,z[,intensity] |