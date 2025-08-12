# deep_marker_estimation/defaults.py
import os
import numpy as np

PKG_DIR = os.path.dirname(os.path.abspath(__file__))

def _p(*parts):
    return os.path.join(PKG_DIR, *parts)

DEFAULTS = {
    "marker": {
        # Default marker image & geometry
        "dictionary": "AprilTag 36h11",
        "id": 0,
        "marker_length_with_border": 0.10,  # meters
        "marker_length_without_border": 0.08,  # meters
        "num_squares": 10,
        "marker_image_path": _p("utils", "tag36h11_0.png"),
        # keypoints_* will be computed lazily on first use from marker_image_path
    },
    "camera": {
        # sensible placeholders; pass real intrinsics at runtime
        "camera_matrix": np.array([[906.995, 0, 638.235],
                                   [0, 906.995, 360.533],
                                   [0, 0, 1]], dtype=float),
        "dist_coeffs": np.array([0, 0, 0, 0, 0], dtype=float),
    },
    "segmentation": {
        "checkpoint_path": _p("segmentation_model", "my_checkpoint_minimodel_epoch_47_batch_0.pth.tar"),
        "input_size": (480, 640),
        "segmentation_threshold": 0.1,
    },
    "keypoints": {
        "checkpoint_path": _p("keypoints_model", "keypoints_checkpoint_20250330.pth.tar"),
        "roi_size": 128,
    },
    "pattern_based": {
        "max_iterations": 10,
        "max_keypoints_est_2d": 72,
    },
}
