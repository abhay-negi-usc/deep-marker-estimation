import os
import sys
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
import cv2

from keypoints_model.model import RegressorMobileNetV3
from keypoints_model.utils import load_checkpoint, overlay_points_on_image
import pdb 

def _normalize_imagenet(img_chw: torch.Tensor) -> torch.Tensor:
	"""Normalize CHW float image (0..255) using ImageNet mean/std.

	Args:
		img_chw: torch.Tensor of shape (3, H, W), dtype float32, range [0, 255]
	Returns:
		Normalized tensor in float32.
	"""
	mean = torch.tensor([0.485, 0.456, 0.406], device=img_chw.device).view(3, 1, 1)
	std = torch.tensor([0.229, 0.224, 0.225], device=img_chw.device).view(3, 1, 1)
	return (img_chw / 255.0 - mean) / std


def _heuristic_map_to_pixels(pred_xy: np.ndarray, w: int, h: int) -> np.ndarray:
	"""Map model outputs to pixel coordinates if they appear normalized.

	If predictions are in [0,1], scale by (w, h).
	If predictions are roughly in [-1,1], map to [0,w]/[0,h].
	Otherwise, assume they are already in pixel units for the resized image.
	"""
	pmin = float(np.nanmin(pred_xy))
	pmax = float(np.nanmax(pred_xy))
	out = pred_xy.copy().astype(np.float32)
	if pmax <= 1.0 and pmin >= 0.0:
		out *= np.array([w, h], dtype=np.float32)
	elif pmax <= 1.5 and pmin >= -1.5:
		out = (out + 1.0) * 0.5 * np.array([w, h], dtype=np.float32)
	return out


def predict_keypoints_on_image(
	model: torch.nn.Module,
	img_path: str,
	input_size: Tuple[int, int],
	device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Run keypoint prediction on a single image.

	Returns:
		- orig_bgr (H,W,3) uint8 original image in BGR for OpenCV drawing
		- pred_xy_orig (K,2) float32 predictions mapped to original image size (pixels)
		- pred_xy_resized (K,2) float32 predictions in resized image coordinates (pixels)
	"""
	# Load original image as RGB, keep a BGR copy for drawing later
	orig_rgb = np.array(Image.open(img_path).convert("RGB"))
	orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
	H0, W0 = orig_rgb.shape[:2]

	in_h, in_w = input_size
	# Resize to model input
	resized = cv2.resize(orig_rgb, (in_w, in_h), interpolation=cv2.INTER_AREA)
	# To tensor CHW
	x = torch.from_numpy(resized).permute(2, 0, 1).float().to(device)
	x = _normalize_imagenet(x).unsqueeze(0)  # (1,3,H,W)

	with torch.no_grad():
		pred = model(x)  # (1, 2K)
	pred = pred[0].detach().cpu().numpy()
	K = pred.shape[0] // 2
	pred_xy = pred.reshape(K, 2)

	# Map to pixel coords of resized image if they look normalized
	pred_xy_resized = _heuristic_map_to_pixels(pred_xy, in_w, in_h)

	# Scale back to original resolution
	sx, sy = float(W0) / float(in_w), float(H0) / float(in_h)
	pred_xy_orig = pred_xy_resized * np.array([sx, sy], dtype=np.float32)

	return orig_bgr, pred_xy_orig.astype(np.float32), pred_xy_resized.astype(np.float32)


def find_images(input_dir: str, exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg")) -> List[str]:
	files = []
	for name in sorted(os.listdir(input_dir)):
		if name.lower().endswith(exts):
			files.append(os.path.join(input_dir, name))
	return files


def ensure_out_dir(base_input_dir: str, out_dir: str = None) -> str:
	if out_dir:
		os.makedirs(out_dir, exist_ok=True)
		return out_dir
	# default: create a new subdirectory inside the input directory
	d = os.path.join(base_input_dir, "keypoints_overlay")
	os.makedirs(d, exist_ok=True)
	return d


def main():
	# Configuration dictionary (edit these paths and options before running)
	CONFIG = {
		"input_dir": "/home/nom4d/deep-marker-estimation/keypoints_model/test_images/",  # e.g., "/path/to/images"
		"checkpoint": "/home/nom4d/deep-marker-estimation/keypoints_model/checkpoints/my_checkpoint.pth.tar_step_10000.pth.tar",  # e.g., "/path/to/checkpoint.pth.tar"
		"out_dir": None,  # if None, a subfolder "keypoints_overlay" inside input_dir will be created
		"height": 300,
		"width": 480,
		"device": "cuda" if torch.cuda.is_available() else "cpu",
		"radius": 2,
	}

	input_dir = CONFIG.get("input_dir")
	checkpoint = CONFIG.get("checkpoint")
	out_dir_cfg = CONFIG.get("out_dir")
	in_h = int(CONFIG.get("height", 300))
	in_w = int(CONFIG.get("width", 480))
	device = CONFIG.get("device", "cpu")
	radius = int(CONFIG.get("radius", 2))

	# Basic validation and helpful messages
	if not input_dir or not os.path.isdir(input_dir):
		print("[info] Please set CONFIG['input_dir'] to a directory with images (png/jpg/jpeg). Script exiting.")
		return
	if not checkpoint or not os.path.isfile(checkpoint):
		print("[info] Please set CONFIG['checkpoint'] to a valid model checkpoint (.pth.tar). Script exiting.")
		return

	input_size = (in_h, in_w)

	# Model
	model = RegressorMobileNetV3().to(device)
	try:
		ckpt = torch.load(checkpoint, map_location=device)
		load_checkpoint(ckpt, model)
	except Exception as e:
		print(f"[error] failed to load checkpoint: {e}")
		sys.exit(1)
	model.eval()

	images = find_images(input_dir)
	if not images:
		print(f"[warn] no images found in: {input_dir}")
		return

	out_dir = ensure_out_dir(input_dir, out_dir_cfg)
	print(f"Writing overlays to: {out_dir}")

	saved = 0
	
	for img_path in images:
		try:
			orig_bgr, pred_xy_orig, _ = predict_keypoints_on_image(
				model, img_path, input_size=input_size, device=device
			)        			
	
			# Optionally filter negative coords (if any)
			valid = pred_xy_orig[:, 0] >= 0
			valid &= pred_xy_orig[:, 1] >= 0
			pts = pred_xy_orig[valid]

			overlay = overlay_points_on_image(orig_bgr.copy(), pts, radius=radius)
			out_name = os.path.basename(img_path)
			out_path = os.path.join(out_dir, out_name)
			cv2.imwrite(out_path, overlay)
			saved += 1
		except Exception as e:
			print(f"[skip] {img_path}: {e}")

	print(f"Done. Saved {saved} overlaid images to {out_dir}")
	pdb.set_trace()
	

if __name__ == "__main__":
	main()

