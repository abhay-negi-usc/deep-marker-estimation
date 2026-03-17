#!/usr/bin/env python3
"""
Validate keypoints JSON by overlaying points on corresponding images.

This script pairs images with keypoints files (by numeric id in filenames),
draws all keypoints onto each image using `overlay_points_on_image`, and saves
the visualizations for quick inspection.

Configuration-only version (no CLI):
    - Edit the CONFIG dict below. It mirrors the data_dir logic used in train.py.
    - The script will auto-pick images folder as either `images/` or `rgb/` under the split.

Notes:
    - Colors are assigned per tag if the JSON groups keypoints by tag.
    - Points with negative coords or NaNs are ignored.
    - Expects pixel coordinates in the original image frame.
"""

import os
import re
import json
from typing import Dict, List, Tuple

import numpy as np
import cv2

from keypoints_model.utils import overlay_points_on_image

# ------------- Config -------------
# Default uses the same data_dir style as in keypoints_model/train.py
# You can change split to "train" or "val". Output overlays will be written
# under <data_dir>/<split>_overlays by default.
CONFIG: Dict = {
    # Copy from train.py (update if needed on your machine)
    "data_dir": "/home/nom4d/deep-marker-estimation/data_generation/multi_marker_augmented_output/multi_marker_augmented_20251021-205104/",
    # "train" or "val"
    "split": "val",
    # Optional: override images/keypoints subfolders; images will auto-choose between candidates
    "images_candidates": ["images", "rgb"],
    "keypoints_subdir": "keypoints",
    # Output directory; if empty, will default to <data_dir>/<split>_overlays
    "out": "",
    # Visualization controls
    "limit": 0,                 # 0 = all
    "radius": 3,
    "thickness": -1,
    "draw_tag_labels": False,
    "grid": False,
    "fail_on_missing": False,
}


def numeric_id(name: str) -> str:
    m = re.search(r"(\d+)", name)
    return m.group(1) if m else ""


def list_files_by_ext(folder: str, exts: Tuple[str, ...]) -> List[str]:
    return [f for f in sorted(os.listdir(folder)) if f.lower().endswith(exts)]


def distinct_colors_bgr(n: int) -> List[Tuple[int, int, int]]:
    # Some distinct BGR colors
    palette = [
        (0, 0, 255),      # red
        (0, 255, 0),      # green
        (255, 0, 0),      # blue
        (0, 255, 255),    # yellow
        (255, 0, 255),    # magenta
        (255, 255, 0),    # cyan
        (0, 165, 255),    # orange
        (147, 20, 255),   # pink-ish
        (203, 192, 255),  # beige-ish
        (128, 128, 128),  # gray
    ]
    if n <= len(palette):
        return palette[:n]
    # Repeat if more tags than palette size
    out = []
    for i in range(n):
        out.append(palette[i % len(palette)])
    return out


def load_keypoints_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def to_point_list(arr_like) -> List[Tuple[float, float]]:
    arr = np.array(arr_like, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    if arr.shape[-1] != 2:
        raise ValueError(f"Keypoints array must have shape (N,2), got {arr.shape}")
    pts = []
    for x, y in arr:
        if np.isnan(x) or np.isnan(y) or x < 0 or y < 0:
            pts.append(None)
        else:
            pts.append((float(x), float(y)))
    return pts


def draw_tag_label(img: np.ndarray, text: str, anchor_xy: Tuple[int, int], color=(255, 255, 255)):
    x, y = int(anchor_xy[0]), int(anchor_xy[1])
    cv2.putText(img, text, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def make_grid(img_left: np.ndarray, img_right: np.ndarray) -> np.ndarray:
    h = max(img_left.shape[0], img_right.shape[0])
    w = img_left.shape[1] + img_right.shape[1]
    c = img_left.shape[2]
    grid = np.zeros((h, w, c), dtype=img_left.dtype)
    grid[: img_left.shape[0], : img_left.shape[1]] = img_left
    grid[: img_right.shape[0], img_left.shape[1] : img_left.shape[1] + img_right.shape[1]] = img_right
    return grid


def choose_subdir(base: str, candidates: List[str]) -> str:
    for c in candidates:
        p = os.path.join(base, c)
        if os.path.isdir(p):
            return p
    # default to first candidate path even if missing (to surface a clear error later)
    return os.path.join(base, candidates[0])


def main():
    cfg = CONFIG

    data_dir = cfg["data_dir"]
    split = cfg.get("split", "val")
    images_candidates = cfg.get("images_candidates", ["images", "rgb"])
    keypoints_subdir = cfg.get("keypoints_subdir", "keypoints")

    split_dir = os.path.join(data_dir, split)
    img_dir = choose_subdir(split_dir, images_candidates)
    kp_dir = os.path.join(split_dir, keypoints_subdir)

    out_dir = cfg.get("out") or os.path.join(data_dir, f"{split}_overlays")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(split_dir):
        print(f"[ERROR] Split folder not found: {split_dir}")
        return
    if not os.path.isdir(img_dir):
        print(f"[ERROR] Images folder not found: {img_dir}")
        return
    if not os.path.isdir(kp_dir):
        print(f"[ERROR] Keypoints folder not found: {kp_dir}")
        return

    img_exts = (".png", ".jpg", ".jpeg")
    images = list_files_by_ext(img_dir, img_exts)
    jsons = list_files_by_ext(kp_dir, (".json",))

    # Build id -> json file map
    id_to_json: Dict[str, str] = {}
    for jf in jsons:
        nid = numeric_id(jf)
        if nid:
            id_to_json[nid] = jf

    missing = 0
    processed = 0
    total_points = 0
    total_drawn = 0

    for i, im_name in enumerate(images):
        if cfg["limit"] and processed >= cfg["limit"]:
            break

        nid = numeric_id(im_name)
        jf = id_to_json.get(nid)
        if jf is None:
            missing += 1
            if cfg["fail_on_missing"]:
                raise FileNotFoundError(f"No keypoints json for image {im_name} (id={nid})")
            continue

        im_path = os.path.join(img_dir, im_name)
        kp_path = os.path.join(kp_dir, jf)

        # Read with cv2 to keep BGR channel order consistent with overlay util
        img_bgr = cv2.imread(im_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[WARN] Failed to read image: {im_path}")
            continue

        raw = img_bgr.copy()
        data = load_keypoints_json(kp_path)

        # Two possible formats supported:
        # 1) { "tag1": [[x,y],...], "tag2": [[x,y],...] }
        # 2) { "keypoints": [[x,y], ...] }
        if isinstance(data, dict) and any(isinstance(v, list) for v in data.values()):
            tags = list(data.keys())
            colors = distinct_colors_bgr(len(tags))
            for tag, color in zip(tags, colors):
                try:
                    pts = to_point_list(data[tag])
                except Exception as e:
                    print(f"[WARN] {jf}:{tag} invalid keypoints: {e}")
                    continue
                total_points += sum(1 for p in pts if p is not None)
                # draw
                out = overlay_points_on_image(img_bgr, pts, radius=cfg["radius"], color=color, thickness=cfg["thickness"])
                # optionally label the first valid point for this tag
                if cfg["draw_tag_labels"]:
                    for p in pts:
                        if p is not None:
                            draw_tag_label(out, tag, p, color=(255, 255, 255))
                            break
        else:
            # Fallback: try to interpret as a flat list under 'keypoints'
            kp_list = data.get("keypoints") if isinstance(data, dict) else None
            if kp_list is None:
                print(f"[WARN] Unrecognized JSON structure in {kp_path}; skipping")
                continue
            pts = to_point_list(kp_list)
            total_points += sum(1 for p in pts if p is not None)
            overlay_points_on_image(img_bgr, pts, radius=cfg["radius"], color=(0, 0, 255), thickness=cfg["thickness"])

        total_drawn += np.count_nonzero((img_bgr != raw).any(axis=2)) > 0

        # Save outputs
        out_name = os.path.splitext(im_name)[0] + "_overlay.png"
        out_path = os.path.join(out_dir, out_name)
        if cfg["grid"]:
            grid = make_grid(raw, img_bgr)
            cv2.imwrite(out_path, grid)
        else:
            cv2.imwrite(out_path, img_bgr)

        processed += 1
        if (processed % 50) == 0:
            print(f"Processed {processed} images...")

    print("\nSummary:")
    print(f"  Images processed: {processed}")
    print(f"  Missing pairs:    {missing}")
    print(f"  Total points:     {total_points}")
    print(f"  Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
