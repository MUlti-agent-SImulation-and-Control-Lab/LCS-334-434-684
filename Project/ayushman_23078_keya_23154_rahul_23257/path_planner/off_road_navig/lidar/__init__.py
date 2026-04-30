"""lidar/ — Point cloud I/O, preprocessing, and visualisation."""
from .loader import load_point_cloud, generate_synthetic_terrain
from .preprocessor import preprocess, estimate_bounds
from .visualizer import plot_cloud_risk, plot_terrain_labels, plot_graph, plot_joint_risk_profile, show_all

__all__ = [
    "load_point_cloud", "generate_synthetic_terrain",
    "preprocess", "estimate_bounds",
    "plot_cloud_risk", "plot_terrain_labels", "plot_graph",
    "plot_joint_risk_profile", "show_all",
]