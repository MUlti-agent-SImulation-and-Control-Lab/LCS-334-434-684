"""
pipeline_visualizer.py
----------------------
Saves 8 diagnostic PNGs for the full terrain segmentation pipeline.

Outputs (all in pipeline_outputs/):
  01_raw_cloud.png
  02_voxels.png
  03_sphere.png
  04_covariance.png
  05_surface_fit.png
  06_slope.png
  07_roughness.png
  08_segmentation.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
import sys, time

# ── config ────────────────────────────────────────────────────────
CLOUD_PATH  = "/home/ayushman/ext_proj/ecs334/merged_cloud.npy"
OUT_DIR     = Path("pipeline_outputs")
RADIUS      = 1.0
VOXEL_SIZE  = 0.5
MAX_DISP    = 40_000      # max points for scatter plots
DARK_BG     = "#0d0d0d"
SEED        = 42
# ─────────────────────────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED)

def _progress(msg):
    print(f"  {msg}", flush=True)

def _savefig(fig, name):
    p = OUT_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved → {p}")

def _dark_fig(*args, **kwargs):
    fig = plt.figure(*args, facecolor=DARK_BG, **kwargs)
    return fig

def _subsample(xyz, n=MAX_DISP):
    if len(xyz) > n:
        idx = rng.choice(len(xyz), n, replace=False)
        return xyz[idx]
    return xyz

# ══════════════════════════════════════════════════════════════════
# 0. Load
# ══════════════════════════════════════════════════════════════════
print("\n[0/8] Loading cloud...")
raw = np.load(CLOUD_PATH)
raw = raw[~np.isnan(raw).any(axis=1)]
xyz = raw[:, :3]
N   = len(xyz)
print(f"      {N:,} points")

# ══════════════════════════════════════════════════════════════════
# 1. Raw point cloud
# ══════════════════════════════════════════════════════════════════
print("[1/8] Raw cloud...")
sub = _subsample(xyz)
z   = sub[:, 2]
z_n = (z - z.min()) / (z.max() - z.min() + 1e-9)          # height → colour

fig = _dark_fig(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection="3d", facecolor=DARK_BG)
ax1.scatter(sub[:,0], sub[:,1], sub[:,2],
            c=z_n, cmap="viridis", s=0.4, linewidths=0)
ax1.set_title("Perspective", color="white", fontsize=9)
ax1.tick_params(colors="#555", labelsize=6)

ax2 = fig.add_subplot(122, facecolor=DARK_BG)
sc = ax2.scatter(sub[:,0], sub[:,1], c=z_n, cmap="viridis", s=0.3, linewidths=0)
ax2.set_title("Top-down", color="white", fontsize=9)
ax2.set_aspect("equal")
ax2.tick_params(colors="#555", labelsize=6)
cbar = fig.colorbar(sc, ax=ax2, fraction=0.03, pad=0.04)
cbar.set_label("Height (normalised)", color="white", fontsize=8)
cbar.ax.yaxis.set_tick_params(color="white", labelsize=6)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
fig.suptitle("01 — Raw LiDAR point cloud (merged 5 frames)", color="white", fontsize=11)
_savefig(fig, "01_raw_cloud.png")

# ══════════════════════════════════════════════════════════════════
# 2. Voxel grid
# ══════════════════════════════════════════════════════════════════
print("[2/8] Voxel grid...")
origin   = xyz.min(axis=0)
vox_idx  = ((xyz - origin) / VOXEL_SIZE).astype(int)
# unique voxel centres
unique_v = np.unique(vox_idx, axis=0)
centres  = unique_v * VOXEL_SIZE + origin + VOXEL_SIZE / 2
print(f"      {len(centres):,} voxels from {N:,} points  (size={VOXEL_SIZE}m)")

sub_v = _subsample(centres, 20_000)
zv    = sub_v[:, 2]
zv_n = (zv - zv.min()) / (zv.max() - zv.min() + 1e-9)


fig = _dark_fig(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection="3d", facecolor=DARK_BG)
ax1.scatter(sub_v[:,0], sub_v[:,1], sub_v[:,2],
            c=zv_n, cmap="plasma", s=1.5, linewidths=0, marker="s")
ax1.set_title("Perspective", color="white", fontsize=9)
ax1.tick_params(colors="#555", labelsize=6)

ax2 = fig.add_subplot(122, facecolor=DARK_BG)
ax2.scatter(sub_v[:,0], sub_v[:,1], c=zv_n, cmap="plasma",
            s=1.0, linewidths=0, marker="s")
ax2.set_title("Top-down", color="white", fontsize=9)
ax2.set_aspect("equal")
ax2.tick_params(colors="#555", labelsize=6)
fig.suptitle(f"02 — Voxel grid  (voxel size = {VOXEL_SIZE} m)", color="white", fontsize=11)
_savefig(fig, "02_voxels.png")

# ══════════════════════════════════════════════════════════════════
# 3. Sphere neighbourhood for one random point
# ══════════════════════════════════════════════════════════════════
print("[3/8] Sphere neighbourhood...")

# Pick a random point that has enough neighbours
for _ in range(200):
    pi = rng.integers(0, N)
    p  = xyz[pi]
    dists = np.linalg.norm(xyz - p, axis=1)
    nbr_mask = dists <= RADIUS
    nbrs     = xyz[nbr_mask]
    if len(nbrs) >= 10:
        break
print(f"      centre point index={pi}  neighbours={len(nbrs)}")

fig = _dark_fig(figsize=(8, 7))
ax  = fig.add_subplot(111, projection="3d", facecolor=DARK_BG)

# Draw a translucent sphere surface
u, v  = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
sx = p[0] + RADIUS * np.cos(u) * np.sin(v)
sy = p[1] + RADIUS * np.sin(u) * np.sin(v)
sz = p[2] + RADIUS * np.cos(v)
ax.plot_surface(sx, sy, sz, alpha=0.08, color="#4488ff",
                linewidth=0, antialiased=False)
ax.plot_wireframe(sx, sy, sz, color="#2255aa", linewidth=0.2, alpha=0.15)

# Background cloud (tiny, faint)
bg_idx = rng.choice(N, min(8000, N), replace=False)
ax.scatter(xyz[bg_idx,0], xyz[bg_idx,1], xyz[bg_idx,2],
           c="#333333", s=0.2, linewidths=0, alpha=0.4)

# Neighbour points
ax.scatter(nbrs[:,0], nbrs[:,1], nbrs[:,2],
           c="#00ffcc", s=8, linewidths=0, zorder=5, label=f"Neighbours ({len(nbrs)})")

# Centre point
ax.scatter(*p, c="#ff4444", s=60, zorder=10, label="Query point")

ax.set_title(f"03 — Sphere neighbourhood  (r = {RADIUS} m)", color="white", fontsize=10)
ax.tick_params(colors="#555", labelsize=6)
leg = ax.legend(fontsize=8, framealpha=0.2, labelcolor="white")
fig.suptitle("", color="white")
_savefig(fig, "03_sphere.png")

# ══════════════════════════════════════════════════════════════════
# 4. Covariance matrix
# ══════════════════════════════════════════════════════════════════
print("[4/8] Covariance matrix...")
cov = np.cov(nbrs.T)           # (3, 3)
eigenvalues, eigenvectors = np.linalg.eigh(cov)
idx_sort = np.argsort(eigenvalues)
eigenvalues  = eigenvalues[idx_sort]
eigenvectors = eigenvectors[:, idx_sort]

# ── terminal print ────────────────────────────────────────────────
print("\n      Covariance matrix:")
labels_xyz = ["X", "Y", "Z"]
header = "        " + "".join(f"{'  '+lbl:>14}" for lbl in labels_xyz)
print(header)
for i, row in enumerate(cov):
    print(f"      {labels_xyz[i]}  " + "  ".join(f"{v:12.6f}" for v in row))
print(f"\n      Eigenvalues: λ0={eigenvalues[0]:.6f}  λ1={eigenvalues[1]:.6f}  λ2={eigenvalues[2]:.6f}")
print(f"      (λ0 smallest → normal direction)")

# ── image ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=DARK_BG)

# Left: heatmap
ax = axes[0]
ax.set_facecolor(DARK_BG)
im = ax.imshow(cov, cmap="coolwarm", aspect="auto")
ax.set_xticks([0,1,2]); ax.set_xticklabels(["X","Y","Z"], color="white")
ax.set_yticks([0,1,2]); ax.set_yticklabels(["X","Y","Z"], color="white")
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{cov[i,j]:.4f}", ha="center", va="center",
                color="white", fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(colors="white")
ax.set_title("Covariance matrix", color="white", fontsize=10)

# Right: table with eigenvalues
ax2 = axes[1]
ax2.set_facecolor(DARK_BG)
ax2.axis("off")
rows = []
for i in range(3):
    ev = eigenvectors[:, i]
    rows.append([f"λ{i} = {eigenvalues[i]:.6f}",
                 f"[{ev[0]:.3f}, {ev[1]:.3f}, {ev[2]:.3f}]",
                 "← normal" if i == 0 else ""])
col_labels = ["Eigenvalue", "Eigenvector", ""]
tbl = ax2.table(cellText=rows, colLabels=col_labels,
                loc="center", cellLoc="left")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor("#1a1a2e" if r % 2 == 0 else "#16213e")
    cell.set_text_props(color="white")
    cell.set_edgecolor("#333355")
ax2.set_title("Eigenvalues & eigenvectors", color="white", fontsize=10)

fig.suptitle("04 — Covariance matrix of sphere neighbourhood", color="white", fontsize=11)
_savefig(fig, "04_covariance.png")

# ══════════════════════════════════════════════════════════════════
# 5. Surface fit from eigenvectors
# ══════════════════════════════════════════════════════════════════
print("[5/8] Surface fit...")
normal   = eigenvectors[:, 0]          # smallest eigenvalue = normal
centroid = nbrs.mean(axis=0)

# Build tangent plane grid
t1 = eigenvectors[:, 1]
t2 = eigenvectors[:, 2]
grid_range = RADIUS * 0.9
gs = np.linspace(-grid_range, grid_range, 20)
G1, G2 = np.meshgrid(gs, gs)
plane_pts = (centroid[None,None,:]
             + G1[:,:,None] * t1[None,None,:]
             + G2[:,:,None] * t2[None,None,:])

fig = _dark_fig(figsize=(9, 7))
ax  = fig.add_subplot(111, projection="3d", facecolor=DARK_BG)

# Plane surface
surf = ax.plot_surface(plane_pts[:,:,0], plane_pts[:,:,1], plane_pts[:,:,2],
                       alpha=0.25, color="#44aaff", linewidth=0)

# Neighbourhood points
ax.scatter(nbrs[:,0], nbrs[:,1], nbrs[:,2],
           c="#00ffcc", s=10, zorder=5, label="Neighbours")
ax.scatter(*p, c="#ff4444", s=60, zorder=10, label="Query point")
ax.scatter(*centroid, c="#ffff00", s=40, zorder=10, label="Centroid")

# Normal vector
scale = RADIUS * 0.7
ax.quiver(*centroid, *(normal * scale),
          color="#ff8800", linewidth=2, arrow_length_ratio=0.2, label="Surface normal")

ax.set_title("05 — Surface fit via PCA (smallest eigenvector = normal)",
             color="white", fontsize=9)
ax.tick_params(colors="#555", labelsize=6)
leg = ax.legend(fontsize=8, framealpha=0.2, labelcolor="white")
_savefig(fig, "05_surface_fit.png")

# ══════════════════════════════════════════════════════════════════
# 6. Slope: angle between normal and gravity
# ══════════════════════════════════════════════════════════════════
print("[6/8] Slope...")
gravity  = np.array([0.0, 0.0, -1.0])
cos_a    = np.clip(np.dot(normal, -gravity), -1, 1)
slope_deg = np.degrees(np.arccos(cos_a))
# Ensure normal points generally upward for display
if normal[2] < 0:
    normal_disp = -normal
else:
    normal_disp = normal

fig = _dark_fig(figsize=(9, 7))
ax  = fig.add_subplot(111, projection="3d", facecolor=DARK_BG)

# Fitted plane
ax.plot_surface(plane_pts[:,:,0], plane_pts[:,:,1], plane_pts[:,:,2],
                alpha=0.20, color="#44aaff", linewidth=0)

# Points scattered on surface
ax.scatter(nbrs[:,0], nbrs[:,1], nbrs[:,2],
           c="#00ffcc", s=8, zorder=5, alpha=0.7)

scale = RADIUS * 0.85

# Surface normal (orange)
ax.quiver(*centroid, *(normal_disp * scale),
          color="#ff8800", linewidth=2.5, arrow_length_ratio=0.15,
          label=f"Surface normal")

# Gravity axis (cyan, straight down)
grav_start = centroid + normal_disp * scale
ax.quiver(*grav_start, 0, 0, -scale,
          color="#00ccff", linewidth=2.5, arrow_length_ratio=0.15,
          label="Gravity axis (–Z)")

# Arc to show the angle — parametric arc in the plane of the two vectors
v1 = normal_disp / np.linalg.norm(normal_disp)
v2 = np.array([0., 0., 1.])
if abs(np.dot(v1, v2)) > 0.999:
    v2 = np.array([1., 0., 0.])
perp = np.cross(v1, v2)
if np.linalg.norm(perp) > 1e-6:
    perp /= np.linalg.norm(perp)
    arc_t  = np.linspace(0, np.radians(slope_deg), 40)
    arc_r  = scale * 0.5
    arc_pts = (centroid[None,:]
               + arc_r * np.cos(arc_t)[:,None] * v1[None,:]
               + arc_r * np.sin(arc_t)[:,None] * perp[None,:])
    ax.plot(arc_pts[:,0], arc_pts[:,1], arc_pts[:,2],
            color="#ffff44", linewidth=1.5, label=f"Slope = {slope_deg:.1f}°")

ax.set_title(f"06 — Slope estimation  (slope = {slope_deg:.1f}°)", color="white", fontsize=10)
ax.tick_params(colors="#555", labelsize=6)
ax.legend(fontsize=8, framealpha=0.2, labelcolor="white")
_savefig(fig, "06_slope.png")

# ══════════════════════════════════════════════════════════════════
# 7. Roughness
# ══════════════════════════════════════════════════════════════════
print("[7/8] Roughness...")
# Project each neighbour onto the plane, measure residual distance
centred  = nbrs - centroid
residuals = np.abs(centred @ normal)     # signed distance to plane
roughness = residuals.std()

fig = _dark_fig(figsize=(12, 5))

# Left: 3-D residuals
ax1 = fig.add_subplot(121, projection="3d", facecolor=DARK_BG)
ax1.plot_surface(plane_pts[:,:,0], plane_pts[:,:,1], plane_pts[:,:,2],
                 alpha=0.15, color="#44aaff", linewidth=0)

res_n = (residuals - residuals.min()) / (residuals.max() - residuals.min() + 1e-9)
sc = ax1.scatter(nbrs[:,0], nbrs[:,1], nbrs[:,2],
                 c=res_n, cmap="hot", s=18, zorder=5)

# Draw residual lines from point to plane projection
for pt, res, rn in zip(nbrs, residuals, res_n):
    proj = pt - (centred[nbrs.tolist().index(pt.tolist()) if False else 0] @ normal) * normal
    # simplified: project each point onto plane
    c2p  = pt - centroid
    proj = pt - (c2p @ normal) * normal
    col  = plt.cm.hot(rn)
    ax1.plot([pt[0], proj[0]], [pt[1], proj[1]], [pt[2], proj[2]],
             color=col, linewidth=0.6, alpha=0.6)

ax1.set_title("Residual distances to fitted plane", color="white", fontsize=9)
ax1.tick_params(colors="#555", labelsize=6)
fig.colorbar(sc, ax=ax1, fraction=0.03, pad=0.04,
             label="Residual (norm)").ax.tick_params(colors="white", labelsize=6)

# Right: histogram of residuals
ax2 = fig.add_subplot(122, facecolor=DARK_BG)
ax2.hist(residuals, bins=20, color="#ff8800", edgecolor="#cc6600", alpha=0.85)
ax2.axvline(roughness, color="#ffff44", linewidth=1.5,
            label=f"σ (roughness) = {roughness:.4f} m")
ax2.axvline(residuals.mean(), color="#00ffcc", linewidth=1.5, linestyle="--",
            label=f"mean = {residuals.mean():.4f} m")
ax2.set_xlabel("Residual distance to plane (m)", color="white", fontsize=9)
ax2.set_ylabel("Count", color="white", fontsize=9)
ax2.tick_params(colors="white", labelsize=8)
ax2.legend(fontsize=8, framealpha=0.2, labelcolor="white")
ax2.spines[:].set_color("#333")
fig.suptitle(f"07 — Roughness  (σ of residuals = {roughness:.4f} m)", color="white", fontsize=11)
_savefig(fig, "07_roughness.png")

# ══════════════════════════════════════════════════════════════════
# 8. Terrain segmentation (UPDATED TO USE TRAINED MODEL)
# ══════════════════════════════════════════════════════════════════
print("[8/8] Terrain segmentation...")
from terrain.feature_extractor import extract_features
from terrain.segmenter import TerrainSegmenter, LABEL_COLORS, LABEL_NAMES

# Path to your trained weights (must match your main.py path)
WEIGHTS_PATH = "models/weights/terrain_classifier.pt" 

print(f"      Loading model weights from: {WEIGHTS_PATH}")
seg = TerrainSegmenter(weights_path=WEIGHTS_PATH) 
# ------------------------

t0 = time.time()
print("      Extracting features (full cloud)...")
# ... (keep the rest of the extraction and prediction loop as is)
features = extract_features(xyz, radius=RADIUS)
print(f"      Done in {time.time()-t0:.1f}s")

seg    = TerrainSegmenter()
CHUNK  = 50_000
all_labels = []
t0 = time.time()
for start in range(0, N, CHUNK):
    end = min(start + CHUNK, N)
    all_labels.append(seg.predict(features[start:end]))
    pct = end / N
    bar = "█" * int(30*pct) + "░" * (30 - int(30*pct))
    sys.stdout.write(f"\r      [{bar}] {pct*100:.0f}%")
    sys.stdout.flush()
print(f"\n      Segmented in {time.time()-t0:.1f}s")
labels_all = np.concatenate(all_labels)

# Label distribution
print("      Label distribution:")
for lbl, name in LABEL_NAMES.items():
    cnt = (labels_all == lbl).sum()
    print(f"        {name:<12} {100*cnt/N:5.1f}%  ({cnt:,})")

seg_colors = np.array([LABEL_COLORS[l] for l in labels_all])

# subsample for display
idx = rng.choice(N, min(MAX_DISP, N), replace=False)

fig = _dark_fig(figsize=(14, 6))
ax1 = fig.add_subplot(121, projection="3d", facecolor=DARK_BG)
ax1.scatter(xyz[idx,0], xyz[idx,1], xyz[idx,2],
            c=seg_colors[idx], s=0.5, linewidths=0)
ax1.set_title("Perspective", color="white", fontsize=9)
ax1.tick_params(colors="#555", labelsize=6)

ax2 = fig.add_subplot(122, facecolor=DARK_BG)
ax2.scatter(xyz[idx,0], xyz[idx,1],
            c=seg_colors[idx], s=0.4, linewidths=0)
ax2.set_title("Top-down", color="white", fontsize=9)
ax2.set_aspect("equal")
ax2.tick_params(colors="#555", labelsize=6)

patches = [mpatches.Patch(color=LABEL_COLORS[l], label=LABEL_NAMES[l])
           for l in sorted(LABEL_NAMES)]
fig.legend(handles=patches, loc="lower center", ncol=5,
           framealpha=0, labelcolor="white", fontsize=9,
           bbox_to_anchor=(0.5, 0.01))
fig.suptitle("08 — Terrain segmentation", color="white", fontsize=11)
plt.tight_layout(rect=[0, 0.06, 1, 1])
_savefig(fig, "08_segmentation.png")

print(f"\nAll 8 outputs saved to {OUT_DIR.resolve()}/")