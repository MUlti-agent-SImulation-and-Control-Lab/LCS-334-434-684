"""terrain/ — Feature extraction, segmentation, risk estimation."""
from .feature_extractor import extract_features, FEATURE_NAMES
from .segmenter import TerrainSegmenter, LABEL_GROUND, LABEL_OBSTACLE, LABEL_WATER, LABEL_VEGETATION, LABEL_UNKNOWN, LABEL_NAMES
from .risk_estimator import RiskEstimator

__all__ = [
    "extract_features", "FEATURE_NAMES",
    "TerrainSegmenter", "LABEL_GROUND", "LABEL_OBSTACLE", "LABEL_WATER",
    "LABEL_VEGETATION", "LABEL_UNKNOWN", "LABEL_NAMES",
    "RiskEstimator",
]