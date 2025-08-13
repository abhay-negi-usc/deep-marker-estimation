# deep_marker_estimation/estimator.py
from __future__ import annotations
import os, inspect, threading
from typing import Any, Dict, Optional, Tuple
import numpy as np
import cv2

from .defaults import DEFAULTS
from .utils.homography_utils import convert_marker_keypoints_to_cartesian
from .utils.image_utils import find_keypoints
# IMPORTANT: import the MODULES, not the functions:
from .utils import seg_utils as _seg
from .utils import keypoint_utils as _kp
from .utils import pose_estimation_utils as _pose
from .utils import pattern_based_estimation_utils as _pbcv

def _resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, path)

def _quat_from_R(R: np.ndarray) -> Tuple[float, float, float, float]:
    t = np.trace(R)
    if t > 0.0:
        s = (t + 1.0) ** 0.5 * 2.0
        w = 0.25 * s
        x = (R[2,1] - R[1,2]) / s
        y = (R[0,2] - R[2,0]) / s
        z = (R[1,0] - R[0,1]) / s
    else:
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

def _normalize_pose(pose_any: Any) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(pose_any, (list, tuple)) and len(pose_any) == 2:
        R, t = pose_any
        R = np.asarray(R, dtype=float).reshape(3,3)
        t = np.asarray(t, dtype=float).reshape(3,)
        return R, t
    M = np.asarray(pose_any, dtype=float)
    if M.shape == (4,4):
        return M[:3,:3], M[:3,3]
    raise ValueError(f"Unrecognized pose format: shape {getattr(pose_any,'shape',None)}")

class MarkerEstimator:
    """
    One-shot API:
        est = MarkerEstimator(preload_models=True, device="cpu")
        out = est.estimate_pose(img_bgr, K, D)
    """

    def __init__(
        self,
        config_marker: Optional[Dict[str, Any]] = None,
        config_segmentation: Optional[Dict[str, Any]] = None,
        config_keypoints: Optional[Dict[str, Any]] = None,
        config_pattern_based: Optional[Dict[str, Any]] = None,
        *,
        preload_models: bool = True,
        device: str = "cpu",
    ) -> None:
        # Merge configs
        self.cfg_marker = {**DEFAULTS["marker"], **(config_marker or {})}
        self.cfg_seg    = {**DEFAULTS["segmentation"], **(config_segmentation or {})}
        self.cfg_kp     = {**DEFAULTS["keypoints"], **(config_keypoints or {})}
        self.cfg_pbcv   = {**DEFAULTS["pattern_based"], **(config_pattern_based or {})}
        self.device = device

        # Resolve paths
        self.cfg_marker["marker_image_path"] = _resolve(self.cfg_marker["marker_image_path"])
        self.cfg_seg["checkpoint_path"] = _resolve(self.cfg_seg["checkpoint_path"])
        self.cfg_kp["checkpoint_path"]  = _resolve(self.cfg_kp["checkpoint_path"])

        # Marker keypoints (lazy)
        self._marker_img = None
        self._marker_kp_image = None
        self._marker_kp_cart  = None

        # Model caches
        self._seg_model = None
        self._kp_model  = None
        self._sig_seg_accepts_model = None  # cached bool
        self._sig_kp_accepts_model  = None  # cached bool
        self._load_lock = threading.Lock()

        # Preload once (optional)
        if preload_models:
            self._ensure_models_loaded()

        # Eagerly compute marker-space keypoints into cfg (only once)
        img_marker_path = self.cfg_marker["marker_image_path"]
        img_marker = cv2.imread(img_marker_path, cv2.IMREAD_COLOR)
        if img_marker is None:
            raise FileNotFoundError(f"Marker image not found: {img_marker_path}")
        self.cfg_marker["marker_image"] = img_marker
        keypoints_marker_image_space = find_keypoints(img_marker)
        keypoints_marker_cartesian_space = convert_marker_keypoints_to_cartesian(
            keypoints_marker_image_space,
            image_size=(img_marker.shape[0], img_marker.shape[1]),
            marker_size=(self.cfg_marker["marker_length_with_border"],
                         self.cfg_marker["marker_length_with_border"]),
        )
        self.cfg_marker["keypoints_marker_cartesian_space"] = keypoints_marker_cartesian_space

    # ---- internal helpers ---------------------------------------------------
    def _ensure_marker_kps(self) -> None:
        if self._marker_img is not None:
            return
        img_marker = self.cfg_marker["marker_image"]
        kps_img = find_keypoints(img_marker)
        kps_cart = convert_marker_keypoints_to_cartesian(
            kps_img,
            image_size=(img_marker.shape[0], img_marker.shape[1]),
            marker_size=(self.cfg_marker["marker_length_with_border"],
                         self.cfg_marker["marker_length_with_border"]),
        )
        self._marker_img = img_marker
        self._marker_kp_image = kps_img
        self._marker_kp_cart  = kps_cart

    def _ensure_models_loaded(self):
        if self._seg_model is not None and self._kp_model is not None:
            return
        with self._load_lock:
            if self._seg_model is None:
                if hasattr(_seg, "load_segmentation_model"):
                    self._seg_model = _seg.load_segmentation_model(self.cfg_seg["checkpoint_path"], device=self.device)
                else:
                    # last resort: remember to add this function in seg_utils
                    raise RuntimeError("seg_utils.load_segmentation_model is missing; add it to avoid reloads per frame.")
            if self._kp_model is None:
                if hasattr(_kp, "load_keypoint_model"):
                    self._kp_model = _kp.load_keypoint_model(self.cfg_kp["checkpoint_path"], device=self.device)
                else:
                    raise RuntimeError("keypoint_utils.load_keypoint_model is missing; add it to avoid reloads per frame.")

            # cache whether functions accept `model=` to avoid introspection per frame
            if self._sig_seg_accepts_model is None:
                try:
                    self._sig_seg_accepts_model = "model" in inspect.signature(_seg.segment_marker).parameters
                except Exception:
                    self._sig_seg_accepts_model = False
            if self._sig_kp_accepts_model is None:
                try:
                    self._sig_kp_accepts_model = "model" in inspect.signature(_kp.estimate_keypoints).parameters
                except Exception:
                    self._sig_kp_accepts_model = False

    # ---- public API ---------------------------------------------------------
    def estimate_pose(
        self,
        image_bgr: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        return_debug: bool = False,
    ) -> Optional[Dict[str, Any]]:

        self._ensure_marker_kps()
        self._ensure_models_loaded()

        # print model identities once
        if not hasattr(self, "_printed_ids"):
            print("[DME] seg_model id:", id(self._seg_model), "kp_model id:", id(self._kp_model))
            self._printed_ids = True


        # 1) segmentation (pass cached model if function supports it)
        if self._sig_seg_accepts_model:
            seg = _seg.segment_marker(image_bgr, self.cfg_seg, model=self._seg_model)
        else:
            # function doesn’t accept a model; it may be reloading internally (warn once?)
            seg = _seg.segment_marker(image_bgr, self.cfg_seg)

        # 2) keypoints
        if self._sig_kp_accepts_model:
            kps_2d = _kp.estimate_keypoints(image_bgr, seg, self.cfg_kp, model=self._kp_model)
        else:
            kps_2d = _kp.estimate_keypoints(image_bgr, seg, self.cfg_kp)

        if kps_2d is None or len(kps_2d) == 0:
            return None

        # 3) PnP from keypoints (LBCV)
        lbcv_pose = _pose.estimate_pose_from_keypoints(
            kps_2d,
            {"camera_matrix": camera_matrix, "dist_coeffs": dist_coeffs},
            {
                **self.cfg_marker,
                "keypoints_marker_image_space": self._marker_kp_image,
                "keypoints_marker_cartesian_space": self._marker_kp_cart,
                "marker_image": self._marker_img,
            },
            {"camera_matrix": camera_matrix, "dist_coeffs": dist_coeffs},
        )
        R_lbcv, t_lbcv = _normalize_pose(lbcv_pose)

        out: Dict[str, Any] = {
            "lbcv": {
                "R": R_lbcv,
                "t": t_lbcv,
                "keypoints_2d": kps_2d,
                "segmentation": seg,
            }
        }

        # 4) PBCV refinement
        try:
            T_pbcv, sim = _pbcv.pattern_based_pose_estimation(
                image_bgr, seg, (R_lbcv, t_lbcv),
                self.cfg_marker,
                {"camera_matrix": camera_matrix, "dist_coeffs": dist_coeffs},
                self.cfg_pbcv
            )
            R_pbcv, t_pbcv = _normalize_pose(T_pbcv)
            out["pbcv"] = {"R": R_pbcv, "t": t_pbcv, "image_similarity": float(sim)}
            qwxyz = _quat_from_R(R_pbcv)
        except Exception:
            qwxyz = _quat_from_R(R_lbcv)

        out["quaternion_wxyz"] = qwxyz

        if not return_debug:
            out["lbcv"].pop("segmentation", None)

        return out
