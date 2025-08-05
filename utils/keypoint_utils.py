import numpy as np 
import cv2 
import torch 
import albumentations as A
from albumentations.pytorch import ToTensorV2

from keypoints_model.utils import load_checkpoint
from keypoints_model.model import RegressorMobileNetV3

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

def estimate_keypoints(image, marker_segmentation, config_keypoint): 

    # Load config
    kp_model_path = config_keypoint['checkpoint_path']
    checkpoint_path = config_keypoint['checkpoint_path']
    roi_size = config_keypoint.get('roi_size', 128)

    # Setup model
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    kp_model = RegressorMobileNetV3().to(DEVICE)
    load_checkpoint(torch.load(kp_model_path, map_location=DEVICE), kp_model)
    kp_model.eval()

    roi_img, roi_coords = compute_roi(marker_segmentation, image, roi_size=roi_size)

    transform = A.Compose([ToTensorV2()])

    roi_tensor = transform(image=roi_img)["image"].unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        keypoints_roi = kp_model(roi_tensor).cpu().numpy().reshape(-1, 2)

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