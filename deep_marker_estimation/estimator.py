# deep_marker_estimation/estimator.py
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import cv2

from .defaults import DEFAULTS
from .utils.image_utils import overlay_points_on_image  # optional use
from .utils.seg_utils import segment_marker
from .utils.keypoint_utils import estimate_keypoints
from .utils.pose_estimation_utils import estimate_pose_from_keypoints
from .utils.pattern_based_estimation_utils import pattern_based_pose_estimation
from .utils.homography_utils import convert_marker_keypoints_to_cartesian
from .utils.image_utils import find_keypoints

def _resolve(path: str) -> str:
    """Return absolute path; if already absolute, keep it; else resolve relative to this package."""
    if os.path.isabs(path):
        return path
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, path)


def _quat_from_R(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Minimal, stable-enough conversion for right-handed, proper R."""
    t = np.trace(R)
    if t > 0.0:
        s = (t + 1.0) ** 0.5 * 2.0
        w = 0.25 * s
        x = (R[2,1] - R[1,2]) / s
        y = (R[0,2] - R[2,0]) / s
        z = (R[1,0] - R[0,1]) / s
    else:
        # pick the largest diagonal
        i = int(np.argmax([R[0,0], R[1,1], R[2,2]]))
        if i == 0:
            s = ((1.0 + R[0,0] - R[1,1] - R[2,2]) ** 0.5) * 2.0
            w = (R[2,1] - R[1,2]) / s
            x = 0.25 * s
            y = (R[0,1] + R[1,0]) / s
            z = (R[0,2] + R[2,0]) / s
        elif i == 1:
            s = ((1.0 + R[1,1] - R[0,0] - R[2,2]) ** 0.5) * 2.0
            w = (R[0,2] - R[2,0]) / s
            x = (R[0,1] + R[1,0]) / s
            y = 0.25 * s
            z = (R[1,2] + R[2,1]) / s
        else:
            s = ((1.0 + R[2,2] - R[0,0] - R[1,1]) ** 0.5) * 2.0
            w = (R[1,0] - R[0,1]) / s
            x = (R[0,2] + R[2,0]) / s
            y = (R[1,2] + R[2,1]) / s
            z = 0.25 * s
    return float(w), float(x), float(y), float(z)


class MarkerEstimator:
    """
    Easy-to-use estimator that runs:
      1) segmentation → keypoints → PnP (LBCV)
      2) optional pattern-based refinement (PBCV)

    Usage:
      est = MarkerEstimator()  # or pass configs overrides
      result = est.estimate_pose(image_bgr, camera_matrix=K, dist_coeffs=D)
    """

    def __init__(
        self,
        config_marker: Optional[Dict[str, Any]] = None,
        config_segmentation: Optional[Dict[str, Any]] = None,
        config_keypoints: Optional[Dict[str, Any]] = None,
        config_pattern_based: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Merge configs with defaults
        self.cfg_marker = {**DEFAULTS["marker"], **(config_marker or {})}
        self.cfg_seg = {**DEFAULTS["segmentation"], **(config_segmentation or {})}
        self.cfg_kp = {**DEFAULTS["keypoints"], **(config_keypoints or {})}
        self.cfg_pbcv = {**DEFAULTS["pattern_based"], **(config_pattern_based or {})}

        # Resolve paths
        self.cfg_marker["marker_image_path"] = _resolve(self.cfg_marker["marker_image_path"])
        self.cfg_seg["checkpoint_path"] = _resolve(self.cfg_seg["checkpoint_path"])
        self.cfg_kp["checkpoint_path"] = _resolve(self.cfg_kp["checkpoint_path"])

        # Load marker image and precompute keypoints for the marker (lazy on first use)
        self._marker_img = None
        self._marker_kp_image = None
        self._marker_kp_cart = None


        img_marker_path = self.cfg_marker["marker_image_path"]
        img_marker = cv2.imread(img_marker_path)
        
        self.cfg_marker["marker_image"] = img_marker
        
        keypoints_marker_image_space = find_keypoints(img_marker)
        keypoints_marker_cartesian_space = convert_marker_keypoints_to_cartesian(
            keypoints_marker_image_space, image_size=(img_marker.shape[0], img_marker.shape[1]), marker_size=(0.1, 0.1)
        )
        self.cfg_marker["keypoints_marker_cartesian_space"] = keypoints_marker_cartesian_space 

    # --- internal helpers ---
    def _ensure_marker_kps(self) -> None:
        if self._marker_img is not None:
            return
        marker_path = self.cfg_marker["marker_image_path"]
        img_marker = cv2.imread(marker_path, cv2.IMREAD_COLOR)
        if img_marker is None:
            raise FileNotFoundError(f"Marker image not found: {marker_path}")

        kps_img = find_keypoints(img_marker)
        kps_cart = convert_marker_keypoints_to_cartesian(
            kps_img,
            image_size=(img_marker.shape[0], img_marker.shape[1]),
            marker_size=(
                self.cfg_marker["marker_length_without_border"],
                self.cfg_marker["marker_length_without_border"],
            ),
        )
        self._marker_img = img_marker
        self._marker_kp_image = kps_img
        self._marker_kp_cart = kps_cart

    # --- public API ---
    def estimate_pose(
        self,
        image_bgr: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        return_debug: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Args:
            image_bgr: HxWx3 BGR image (OpenCV style)
            camera_matrix: 3x3 intrinsics
            dist_coeffs:  (k1,k2,p1,p2[,k3,...])
            return_debug: include intermediate products

        Returns:
            dict with keys:
                'lbcv': {
                    'R': 3x3, 't': (3,), 'keypoints_2d': Nx2, 'segmentation': HxW (uint8 or float),
                }
                'pbcv': {
                    'R': 3x3, 't': (3,), 'image_similarity': float,
                }
                'quaternion_wxyz': (w,x,y,z)   # from PBCV if present else from LBCV
            or None if nothing detected.
        """
        self._ensure_marker_kps()

        # --- 1) segmentation ---
        seg = segment_marker(image_bgr, self.cfg_seg)  # HxW mask (float or uint8)

        # --- 2) keypoint estimation ---
        kps_2d = estimate_keypoints(image_bgr, seg, self.cfg_kp)  # Nx2
        if kps_2d is None or len(kps_2d) == 0:
            return None

        # --- 3) PnP pose from keypoints (LBCV) ---
        # The original code used estimate_pose_from_keypoints(..., config_camera, config_marker, config_camera)
        # We pass camera intrinsics explicitly here:
        lbcv_pose = estimate_pose_from_keypoints(
            kps_2d,
            {"camera_matrix": camera_matrix, "dist_coeffs": dist_coeffs},
            {
                **self.cfg_marker,
                "keypoints_marker_image_space": self._marker_kp_image,
                "keypoints_marker_cartesian_space": self._marker_kp_cart,
                "marker_image": self._marker_img,
            },
            {"camera_matrix": camera_matrix, "dist_coeffs": dist_coeffs},  # kept to match your util signature
        )
        # lbcv_pose expected to be a 4x4 or (R,t). Normalize to (R,t):
        R_lbcv, t_lbcv = _normalize_pose(lbcv_pose)

        out: Dict[str, Any] = {
            "lbcv": {
                "R": R_lbcv,
                "t": t_lbcv,
                "keypoints_2d": kps_2d,
                "segmentation": seg,
            }
        }

        # --- 4) pattern-based refinement (PBCV) ---
        try:
            T_pbcv, sim = pattern_based_pose_estimation(
                image_bgr, seg, (R_lbcv, t_lbcv),  # provide initial guess
                self.cfg_marker,
                {"camera_matrix": camera_matrix, "dist_coeffs": dist_coeffs},
                self.cfg_pbcv
            )
            R_pbcv, t_pbcv = _normalize_pose(T_pbcv)
            out["pbcv"] = {"R": R_pbcv, "t": t_pbcv, "image_similarity": float(sim)}
            qwxyz = _quat_from_R(R_pbcv)
        except Exception:
            # If refinement fails, keep LBCV
            qwxyz = _quat_from_R(R_lbcv)

        out["quaternion_wxyz"] = qwxyz

        if not return_debug:
            # Drop heavyweight intermediates
            out["lbcv"].pop("segmentation", None)

        return out


# --- helper to normalize pose output shapes ---
def _normalize_pose(pose_any: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Accepts:
      - 4x4 homogeneous matrix
      - (R, t) where R is 3x3 and t is (3,) or (3,1)
    Returns:
      (R:3x3, t:(3,))
    """
    if isinstance(pose_any, (list, tuple)) and len(pose_any) == 2:
        R, t = pose_any
        R = np.asarray(R, dtype=float).reshape(3,3)
        t = np.asarray(t, dtype=float).reshape(3,)
        return R, t
    M = np.asarray(pose_any, dtype=float)
    if M.shape == (4,4):
        R = M[:3,:3]
        t = M[:3,3]
        return R, t
    raise ValueError(f"Unrecognized pose format: shape {getattr(pose_any,'shape',None)}")
