import numpy as np 
import cv2 
import torch 
import albumentations as A
from albumentations.pytorch import ToTensorV2

from deep_marker_estimation.keypoints_model.utils import load_checkpoint
from deep_marker_estimation.keypoints_model.model import RegressorMobileNetV3

import os, threading
import torch
try:
    DEVICE
except NameError:
    DEVICE = "cpu"

_KP_MODEL_CACHE = {}
_KP_LOCK = threading.Lock()

def load_keypoint_model(checkpoint_path: str, device: str = DEVICE):
    ckpt_abs = os.path.abspath(checkpoint_path)
    key = (ckpt_abs, str(device))
    with _KP_LOCK:
        m = _KP_CACHE.get(key)
        if m is not None:
            return m

        # 1) TorchScript
        try:
            m = torch.jit.load(ckpt_abs, map_location=device)
            m.eval()
            _KP_CACHE[key] = m
            print(f"[DME] KP TorchScript loaded once from: {ckpt_abs}")
            return m
        except Exception:
            pass

        obj = torch.load(ckpt_abs, map_location=device)

        # 2) pickled nn.Module
        if hasattr(obj, "forward"):
            m = obj.to(device).eval()
            for p in m.parameters(): p.requires_grad = False
            _KP_CACHE[key] = m
            print(f"[DME] KP nn.Module (pickled) loaded once from: {ckpt_abs}")
            return m

        # 3) state_dict (raw or in 'state_dict')
        if isinstance(obj, dict):
            state = obj.get("state_dict", obj) if _looks_like_state_dict(obj) or ("state_dict" in obj) else None
            if state is not None:
                m = RegressorMobileNetV3().to(device)
                m.load_state_dict(state, strict=False)
                m.eval()
                for p in m.parameters(): p.requires_grad = False
                _KP_CACHE[key] = m
                print(f"[DME] KP state_dict loaded once into RegressorMobileNetV3 from: {ckpt_abs}")
                return m

        raise RuntimeError(f"Unsupported keypoint checkpoint format at {ckpt_abs}: {type(obj)}")

import os, threading, torch

try:
    DEVICE
except NameError:
    DEVICE = "cpu"

_KP_CACHE = {}
_KP_LOCK = threading.Lock()

def _looks_like_state_dict(obj):
    if not isinstance(obj, dict) or not obj:
        return False
    import numpy as _np
    return all(isinstance(k, str) for k in obj.keys()) and \
           all(torch.is_tensor(v) or isinstance(v, _np.ndarray) for v in obj.values())


def _get_kp_model(config_keypoint, device=DEVICE):
    ckpt = os.path.abspath(config_keypoint["checkpoint_path"])
    key = (ckpt, str(device))
    with _KP_LOCK:
        m = _KP_CACHE.get(key)
        if m is not None:
            return m
        model = RegressorMobileNetV3().to(device)
        state = torch.load(ckpt, map_location=device)
        load_checkpoint(state, model)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        _KP_CACHE[key] = model
        print(f"[DME] Keypoint model loaded once from: {ckpt} (device={device})")
        return model

def compute_roi(seg, rgb, roi_size=128):
    seg = np.array(seg)
    padding = 5
    image_border_size = max(seg.shape)

    seg = cv2.copyMakeBorder(seg, image_border_size, image_border_size, image_border_size, image_border_size, cv2.BORDER_CONSTANT, value=0)
    rgb = cv2.copyMakeBorder(rgb, image_border_size, image_border_size, image_border_size, image_border_size, cv2.BORDER_CONSTANT, value=0)

    tag_pixels = np.argwhere(seg != 0)
    if tag_pixels.size == 0:
        return None, None

    min_x, max_x = tag_pixels[:, 1].min(), tag_pixels[:, 1].max()
    min_y, max_y = tag_pixels[:, 0].min(), tag_pixels[:, 0].max()
    center_x = int(np.floor((min_x + max_x) / 2))
    center_y = int(np.floor((min_y + max_y) / 2))
    side = max(max_x - min_x, max_y - min_y) + 2 * padding
    half_side = side // 2

    x0 = max(0, center_x - half_side)
    x1 = center_x + half_side
    y0 = max(0, center_y - half_side)
    y1 = center_y + half_side

    roi = rgb[y0:y1, x0:x1]
    roi = cv2.resize(roi, (roi_size, roi_size))  # always resize
    coords = np.array([x0, x1, y0, y1]) - image_border_size

    return roi, coords

def estimate_keypoints(image, marker_segmentation, config_keypoint, model=None): 

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None:
        device = config_keypoint.get("device", DEVICE)
        model  = load_keypoint_model(config_keypoint["checkpoint_path"], device)

    # Load config
    roi_size = config_keypoint.get('roi_size', 128)

    # Setup model
    torch.cuda.empty_cache()
    model.eval()

    roi_img, roi_coords = compute_roi(marker_segmentation, image, roi_size=roi_size)

    transform = A.Compose([ToTensorV2()])

    roi_tensor = transform(image=roi_img)["image"].unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        keypoints_roi = model(roi_tensor).cpu().numpy().reshape(-1, 2)

    # === Fixed reprojection ===
    s = np.array(roi_img.shape[:2])  # (H, W)
    x0, x1, y0, y1 = roi_coords
    roi_center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    w = x1 - x0
    h = y1 - y0
    m = s / np.array([w, h])  # scale from image -> ROI

    keypoints_img = (keypoints_roi - s / 2) / m + roi_center

    keypoints_orig = np.stack([
        keypoints_img[:, 0],
        keypoints_img[:, 1]
    ], axis=1)

    return keypoints_orig