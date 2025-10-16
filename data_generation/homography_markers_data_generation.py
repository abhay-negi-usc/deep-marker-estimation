#!/usr/bin/env python3
"""
Render a square marker image onto a predetermined-size canvas.
Background image is zoomed-in to fill the canvas ("cover" fit), then the
marker is projected using camera intrinsics and pose, and composited by homography.

Inputs (CONFIG):
- output_size_wh: [W, H] final image size in pixels
- background_dir: directory containing source background images (any sizes)
- marker_dir: directory containing square marker PNGs (alpha supported)
- K: 3x3 intrinsics
- dist: distortion coeffs (k1,k2,p1,p2,k3) or empty/zeros if none
- (optional) K_size_wh: [Wk, Hk] resolution that K corresponds to; if given and
    different from output_size_wh, K is scaled accordingly
- marker_length_m: real edge length of the square marker (meters)
- bg_zoom_extra: multiplicative zoom beyond "cover" fit (>=1.0 to zoom in)
- bg_center_xy: [cx, cy] crop center in normalized coords (0..1), default center
- draw_axes / axes_len_m: optional projected axes overlay
- marker_border_probability: probability [0..1] of drawing a border around a marker
- marker_border_frac_image_width: border thickness as a fraction of output image width
 - white_rect_enlarge_frac: fraction by which the marker's side length is increased to draw a white rectangle behind it
- num_markers or num_markers_range: number of markers per image (fixed or [min,max])
- euler_deg_ranges_xyz: [[min,max] per axis in degrees] for rotations (x,y,z)
- tvec_ranges_m: [[min,max] per axis in meters] for translations (x,y,z, z>0)
- output_dir: where to save generated images
- n_images: how many images to generate
 - output directory naming: images are saved under '<output_dir>/multi_marker_{timestamp}/...'
 - save_bw: if True, save RGB images as single-channel grayscale instead of BGR

Coordinate frames and conventions:
- OpenCV camera coords: +x right, +y down, +z forward
- Marker coords: +x right, +y down, +z into marker 

"""

import cv2
import numpy as np
from pathlib import Path
import math
import random
import re
import json
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
try:
    from tqdm import tqdm  # progress bar
except Exception:  # fallback if tqdm not installed
    def tqdm(iterable, **kwargs):
        return iterable
from typing import List, Tuple

# ------------------------ Configuration ------------------------
CONFIG = {
    # Canvas / output
    "output_size_wh": [1920, 1200],  # width, height (px)

    # IO directories
    "background_dir": "/home/anegi/Downloads/test2017/",
    "marker_dir": "data_generation/assets/tag36h11_no_border_64",

    # Camera intrinsics (example; replace with yours)
    "K": [[1400.0,   0.0, 960.0],
          [  0.0, 1400.0, 600.0],
          [  0.0,    0.0,   1.0]],
    "dist": [0, 0, 0, 0, 0],         # zeros if no distortion

    # Marker real size
    "marker_length_m": 0.10,         # meters

    # Background zoom/crop behavior
    "bg_zoom_extra": 1.0,            # >=1.0 zooms in more than "cover"
    "bg_center_xy": [0.5, 0.5],      # normalized crop center (0..1, 0..1)

    # Rendering
    "draw_axes": False,
    "axes_len_m": 0.06,

    # Optional marker border overlay
    # Probability to draw a border around each projected marker, and thickness
    # as a fraction of the output image width.
    "marker_border_probability": 1.0,
    "marker_border_frac_image_width": 0.125,
    # White rectangle backdrop (projected square) size increase over the marker's side length
    "white_rect_enlarge_frac": 0.25,

    # Dataset generation controls
    # Use either fixed num_markers or a range (inclusive)
    # "num_markers": 2,
    "num_markers_range": [8, 16],
    "n_images": 1_000,
    # Base output directory; a timestamped subdir 'multi_marker_{YYYYmmdd-HHMMSS}' will be created inside
    "output_dir": "data_generation/multi_marker_output",

    # Save RGB images as grayscale (BW) if True; segmentation remains colored
    "save_bw": False,

    # Pose randomization ranges
    "tvec_ranges_m": [[-1.0, 1.0], [-1.0, 1.0], [0.2, 2.0]],
    "euler_deg_ranges_xyz": [[-60, 60], [-60, 60], [-180, 180]],

    # Optional seed for reproducibility
    # "seed": 42,

    "frustum_sampling": {
        "enabled": True,          # turn frustum enforcement on/off
        "margin_px": 0,           # keep this many pixels away from the image border
        "keep_partial": True,    # if True, allow partial visibility; if False, require full marker inside
        "min_quad_area_px": 10000, # reject poses that make the marker too tiny
        "max_quad_area_frac": 0.25, # as a fraction of the image area, reject if too large (0 disables)
        "max_tries_per_marker": 200  # how many attempts before giving up on that marker
    }, 

    "num_workers": 24,  # how many parallel worker processes to use (adjust to your CPU)
}

# ------------------------ Helpers ------------------------
def ensure_float_array(x, shape=None):
    a = np.asarray(x, dtype=np.float64)
    if shape is not None:
        a = a.reshape(shape)
    return a

def se3_to_rt(T_cam_marker: np.ndarray):
    """Convert 4x4 pose to (rvec,tvec) in OpenCV convention."""
    R = T_cam_marker[:3, :3]
    t = T_cam_marker[:3, 3]
    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    return rvec.reshape(3), t.reshape(3)

def scale_intrinsics(K, from_wh, to_wh):
    """Scale intrinsics from one image size to another (aspect-preserving per axis)."""
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    fw, fh = float(from_wh[0]), float(from_wh[1])
    tw, th = float(to_wh[0]), float(to_wh[1])
    sx, sy = (tw / fw), (th / fh)
    K_scaled = K.copy()
    K_scaled[0,0] = fx * sx
    K_scaled[1,1] = fy * sy
    K_scaled[0,2] = cx * sx
    K_scaled[1,2] = cy * sy
    return K_scaled

def list_images_in_dir(directory: Path, exts=(".png", ".jpg", ".jpeg", ".bmp")) -> List[Path]:
    """Recursively list image files under a directory."""
    directory = Path(directory)
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"Directory not found or not a directory: {directory}")
    files = [p for p in sorted(directory.rglob("*")) if p.suffix.lower() in exts]
    if not files:
        raise FileNotFoundError(f"No image files with extensions {exts} found under: {directory}")
    return files

def fit_background_to_canvas(bg_bgr, out_w, out_h, extra_zoom=1.0, center_xy=(0.5, 0.5)):
    """
    Resize background to COVER the canvas (like CSS background-size: cover),
    then center-crop to (out_w, out_h). 'extra_zoom' (>1) zooms further in.
    'center_xy' chooses crop center in normalized coords (0..1).
    """
    H, W = bg_bgr.shape[:2]
    if W == 0 or H == 0:
        raise ValueError("Background image has invalid dimensions.")
    cover_scale = max(out_w / W, out_h / H) * float(extra_zoom)
    new_w = max(1, int(math.ceil(W * cover_scale)))
    new_h = max(1, int(math.ceil(H * cover_scale)))

    interp = cv2.INTER_LINEAR if cover_scale >= 1.0 else cv2.INTER_AREA
    resized = cv2.resize(bg_bgr, (new_w, new_h), interpolation=interp)

    # compute crop box centered at (cx, cy) in resized space
    cx = np.clip(center_xy[0], 0.0, 1.0) * (new_w - 1)
    cy = np.clip(center_xy[1], 0.0, 1.0) * (new_h - 1)
    x0 = int(round(cx - out_w / 2))
    y0 = int(round(cy - out_h / 2))
    x0 = np.clip(x0, 0, max(0, new_w - out_w))
    y0 = np.clip(y0, 0, max(0, new_h - out_h))
    x1, y1 = x0 + out_w, y0 + out_h

    canvas = resized[y0:y1, x0:x1].copy()
    if canvas.shape[1] != out_w or canvas.shape[0] != out_h:
        # guard (shouldn't happen)
        canvas = cv2.resize(canvas, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return canvas

def project_marker_corners(marker_len_m, K, dist, rvec, tvec):
    """
    Marker corners in marker frame (z=0), centered at origin.
    Order: [(-L/2,-L/2), (L/2,-L/2), (L/2,L/2), (-L/2,L/2)]
    """
    L = float(marker_len_m)
    corners_3d = np.array([
        [-L/2, -L/2, 0.0],
        [ L/2, -L/2, 0.0],
        [ L/2,  L/2, 0.0],
        [-L/2,  L/2, 0.0],
    ], dtype=np.float32)
    corners_2d, _ = cv2.projectPoints(corners_3d, rvec, tvec, K, dist)
    return corners_2d.reshape(-1, 2)  # (4,2)

def warp_marker_onto_canvas(base_bgr, marker_img, quad_2d, out_w, out_h):
    """
    Warp 'marker_img' onto 'base_bgr' at 'quad_2d' (4x2). Supports 3-channel BGR or 4-channel BGRA.
    Returns composited BGR image (out_h, out_w, 3).
    """
    assert base_bgr.shape[0] == out_h and base_bgr.shape[1] == out_w
    h, w = out_h, out_w

    # Prepare source corners (full marker image)
    mh, mw = marker_img.shape[:2]
    src = np.array([[0, 0], [mw - 1, 0], [mw - 1, mh - 1], [0, mh - 1]], dtype=np.float32)
    dst = quad_2d.astype(np.float32)

    H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC)

    if marker_img.shape[2] == 4:
        # Separate color and alpha
        marker_bgr = marker_img[:, :, :3]
        alpha = marker_img[:, :, 3] / 255.0
        warped_bgr = cv2.warpPerspective(marker_bgr, H, (w, h))
        warped_alpha = cv2.warpPerspective(alpha, H, (w, h))
        warped_alpha_3c = np.dstack([warped_alpha]*3)

        out = (warped_bgr * warped_alpha_3c + base_bgr * (1.0 - warped_alpha_3c)).astype(np.uint8)
        return out
    else:
        warped = cv2.warpPerspective(marker_img, H, (w, h))
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        mask = (gray > 0).astype(np.uint8)
        mask_3c = np.dstack([mask]*3)
        out = base_bgr.copy()
        out[mask_3c == 1] = warped[mask_3c == 1]
        return out

def warp_marker_mask(marker_img, quad_2d, out_w, out_h):
    """
    Create a warped binary mask (uint8 0/255) of the marker region onto the canvas.
    If marker has alpha, uses the alpha; else assumes the entire square image is the marker.
    """
    h, w = out_h, out_w
    mh, mw = marker_img.shape[:2]
    src = np.array([[0, 0], [mw - 1, 0], [mw - 1, mh - 1], [0, mh - 1]], dtype=np.float32)
    dst = quad_2d.astype(np.float32)
    H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC)

    if marker_img.shape[2] == 4:
        alpha = marker_img[:, :, 3]
        mask = cv2.warpPerspective(alpha, H, (w, h))
        # Binarize
        _, mask_bin = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    else:
        ones = np.full((mh, mw), 255, dtype=np.uint8)
        mask_bin = cv2.warpPerspective(ones, H, (w, h))
    return mask_bin

def extract_marker_id_from_path(p: Path) -> int:
    """Extract trailing integer id from filename stem; raises if none found."""
    s = p.stem
    m = re.search(r"(\d+)$", s)
    if m:
        return int(m.group(1))
    # Fallback: search in full filename if stem fails
    m2 = re.search(r"(\d+)$", p.name)
    if m2:
        return int(m2.group(1))
    raise ValueError(f"Could not extract numeric marker id from path: {p}")

def id_to_color_bgr(marker_id: int) -> Tuple[int, int, int]:
    """Deterministically map an integer id to a vivid BGR color (OpenCV HSV space)."""
    # OpenCV HSV: H in [0,179]
    h = int((marker_id * 37) % 180)  # spread ids around hue circle
    s = 200
    v = 255
    hsv = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])

def draw_axes(image_bgr, K, dist, rvec, tvec, axes_len_m=0.05, thickness=2):
    origin = np.array([[0, 0, 0]], dtype=np.float32)
    axes = np.float32([
        [axes_len_m, 0, 0],
        [0, axes_len_m, 0],
        [0, 0, axes_len_m],
    ])
    pts, _ = cv2.projectPoints(np.vstack((origin, axes)), rvec, tvec, K, dist)
    pts = pts.reshape(-1, 2)
    o, x, y, z = pts[0], pts[1], pts[2], pts[3]
    cv2.line(image_bgr, tuple(o.astype(int)), tuple(x.astype(int)), (0, 0, 255), thickness)   # X red
    cv2.line(image_bgr, tuple(o.astype(int)), tuple(y.astype(int)), (0, 255, 0), thickness)   # Y green
    cv2.line(image_bgr, tuple(o.astype(int)), tuple(z.astype(int)), (255, 0, 0), thickness)   # Z blue

def euler_deg_to_rvec(rx_deg: float, ry_deg: float, rz_deg: float, order: str = 'xyz') -> np.ndarray:
    """Convert Euler angles in degrees to Rodrigues rotation vector.
    order 'xyz' means R = Rz @ Ry @ Rx (apply Rx, then Ry, then Rz)."""
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx),  np.cos(rx)]], dtype=np.float64)
    Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                   [ 0,          1, 0         ],
                   [-np.sin(ry), 0, np.cos(ry)]], dtype=np.float64)
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz),  np.cos(rz), 0],
                   [0,           0,          1]], dtype=np.float64)
    if order.lower() == 'xyz':
        R = Rz @ Ry @ Rx
    elif order.lower() == 'zyx':
        R = Rx @ Ry @ Rz
    else:
        raise ValueError(f"Unsupported Euler order: {order}")
    rvec, _ = cv2.Rodrigues(R)
    return rvec.reshape(3)

def put_text_with_outline(img: np.ndarray, text: str, org=(10, 30),
                          font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.0,
                          color=(255, 255, 255), thickness=2,
                          outline_color=(0, 0, 0), outline_thickness=3):
    """Draw anti-aliased text with an outline for readability."""
    cv2.putText(img, text, org, font, font_scale, outline_color, outline_thickness, cv2.LINE_AA)
    cv2.putText(img, text, org, font, font_scale, color, thickness, cv2.LINE_AA)

def quad_area_px(quad: np.ndarray) -> float:
    """Area of a convex quad in pixels via shoelace (sum of two triangles)."""
    p = quad.astype(np.float64)
    # split into two triangles (0,1,2) and (0,2,3)
    def tri_area(a,b,c):
        return 0.5 * abs(
            a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1])
        )
    return tri_area(p[0],p[1],p[2]) + tri_area(p[0],p[2],p[3])

def quad_inside_image(quad: np.ndarray, w: int, h: int, margin: int = 0) -> bool:
    """Check if all points are within [margin, w-1-margin] x [margin, h-1-margin]."""
    x_ok = (quad[:,0] >= margin) & (quad[:,0] <= (w-1-margin))
    y_ok = (quad[:,1] >= margin) & (quad[:,1] <= (h-1-margin))
    return bool(np.all(x_ok) and np.all(y_ok))

def sample_pose_in_frustum(
    K: np.ndarray,
    dist: np.ndarray,
    marker_length_m: float,
    out_w: int,
    out_h: int,
    euler_ranges,
    t_ranges,
    frustum_cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rejection-sample a pose so that the *projected square marker* lies inside the image
    (optionally allowing partial visibility), with size constraints.
    Returns (rvec, tvec, quad_2d).
    """
    enabled = frustum_cfg.get("enabled", True)
    tries  = int(frustum_cfg.get("max_tries_per_marker", 200))
    margin = int(frustum_cfg.get("margin_px", 0))
    keep_partial = bool(frustum_cfg.get("keep_partial", False))
    min_area = float(frustum_cfg.get("min_quad_area_px", 0))
    max_area_frac = float(frustum_cfg.get("max_quad_area_frac", 0.0))
    img_area = float(out_w * out_h)
    max_area = (max_area_frac * img_area) if max_area_frac > 0 else float("inf")

    for _ in range(tries):
        # 1) sample Euler (degrees) and keep them
        rx = random.uniform(euler_ranges[0][0], euler_ranges[0][1])
        ry = random.uniform(euler_ranges[1][0], euler_ranges[1][1])
        rz = random.uniform(euler_ranges[2][0], euler_ranges[2][1])
        euler_deg_xyz = np.array([rx, ry, rz], dtype=np.float64)

        # 2) convert to rvec and sample translation
        rvec = euler_deg_to_rvec(rx, ry, rz, order='xyz')
        tx = random.uniform(t_ranges[0][0], t_ranges[0][1])
        ty = random.uniform(t_ranges[1][0], t_ranges[1][1])
        tz = max(1e-3, random.uniform(t_ranges[2][0], t_ranges[2][1]))
        tvec = np.array([tx, ty, tz], dtype=np.float64)

        if not enabled:
            quad = project_marker_corners(marker_length_m, K, dist, rvec, tvec)
            return rvec, tvec, quad, euler_deg_xyz

        # project marker quad and run checks
        quad = project_marker_corners(marker_length_m, K, dist, rvec, tvec)

        # Keep only quads with finite coordinates
        if not np.isfinite(quad).all():
            continue

        # Size checks
        area_px = quad_area_px(quad)
        if area_px < min_area or area_px > max_area:
            continue

        # Visibility checks
        if keep_partial:
            # allow partial: require at least one point to be inside (and none NaN)
            any_inside = np.any(
                (quad[:,0] >= 0) & (quad[:,0] <= out_w-1) &
                (quad[:,1] >= 0) & (quad[:,1] <= out_h-1)
            )
            if not any_inside:
                continue
        else:
            if not quad_inside_image(quad, out_w, out_h, margin):
                continue

        return rvec, tvec, quad, euler_deg_xyz

    # Fallback single unconstrained sample
    rx = random.uniform(euler_ranges[0][0], euler_ranges[0][1])
    ry = random.uniform(euler_ranges[1][0], euler_ranges[1][1])
    rz = random.uniform(euler_ranges[2][0], euler_ranges[2][1])
    euler_deg_xyz = np.array([rx, ry, rz], dtype=np.float64)
    rvec = euler_deg_to_rvec(rx, ry, rz, order='xyz')
    tx = random.uniform(t_ranges[0][0], t_ranges[0][1])
    ty = random.uniform(t_ranges[1][0], t_ranges[1][1])
    tz = max(1e-3, random.uniform(t_ranges[2][0], t_ranges[2][1]))
    tvec = np.array([tx, ty, tz], dtype=np.float64)
    quad = project_marker_corners(marker_length_m, K, dist, rvec, tvec)
    return rvec, tvec, quad, euler_deg_xyz

def rvec_to_euler_deg_xyz(rvec: np.ndarray) -> np.ndarray:
    import math
    R, _ = cv2.Rodrigues(rvec.astype(np.float64).reshape(3,1))
    sy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    singular = sy < 1e-6
    if not singular:
        rx = math.atan2(R[2,1], R[2,2])
        ry = math.atan2(-R[2,0], sy)
        rz = math.atan2(R[1,0], R[0,0])
    else:
        # Gimbal lock (sy ~ 0): set rz = 0
        rx = math.atan2(-R[1,2], R[1,1])
        ry = math.atan2(-R[2,0], sy)
        rz = 0.0
    return np.degrees([rx, ry, rz]).astype(np.float64)


# ------------------------ Per-image worker ------------------------
def _render_one_image(
    i: int,
    cfg: dict,
    K: np.ndarray,
    dist: np.ndarray,
    bg_files: list,
    marker_files: list,
    out_w: int,
    out_h: int,
    images_dir: str,
    seg_dir: str,
    metadata_dir: str,
):
    # Seed randomness per-process and per-index for diversity
    try:
        base_seed = int(cfg.get("seed", 0))
    except Exception:
        base_seed = 0
    rnd_seed = (base_seed + i + os.getpid()) & 0x7FFFFFFF
    random.seed(rnd_seed)
    np.random.seed(rnd_seed % (2**32 - 1))

    images_dir = Path(images_dir)
    seg_dir = Path(seg_dir)
    metadata_dir = Path(metadata_dir)

    # Shallow copies of ranges from cfg
    euler_ranges = cfg.get("euler_deg_ranges_xyz", [[0, 0], [0, 0], [0, 0]])
    t_ranges = cfg.get("tvec_ranges_m", [[0, 0], [0, 0], [1.0, 1.0]])

    # Background
    bg_path = Path(random.choice(bg_files))
    bg = cv2.imread(str(bg_path), cv2.IMREAD_COLOR)
    if bg is None:
        raise FileNotFoundError(f"Could not read background image: {bg_path}")
    canvas = fit_background_to_canvas(
        bg, out_w, out_h,
        extra_zoom=float(cfg.get("bg_zoom_extra", 1.0)),
        center_xy=tuple(cfg.get("bg_center_xy", [0.5, 0.5]))
    )

    # Initialize outputs for this image
    out_img = canvas
    seg_img = np.zeros((out_h, out_w, 3), dtype=np.uint8)  # colored per marker id
    markers_info = []

    # Marker count
    fixed_n = cfg.get("num_markers")
    n_range = cfg.get("num_markers_range", [1, 1])
    n_range = [int(n_range[0]), int(n_range[1])]
    if fixed_n is not None:
        n_markers = int(fixed_n)
    else:
        n_markers = int(random.randint(min(n_range), max(n_range)))

    # Choose markers (with replacement)
    chosen = [Path(random.choice(marker_files)) for _ in range(n_markers)]
    random.shuffle(chosen)

    for m_path in chosen:
        marker = cv2.imread(str(m_path), cv2.IMREAD_UNCHANGED)
        if marker is None:
            continue
        # Ensure square marker by resizing to min dimension if needed
        if marker.shape[0] != marker.shape[1]:
            s = min(marker.shape[0], marker.shape[1])
            marker = cv2.resize(marker, (s, s), interpolation=cv2.INTER_AREA)

        # Sample pose constrained to the camera frustum (optionally partial)
        rvec, tvec, quad, euler_deg_xyz = sample_pose_in_frustum(
            K=K,
            dist=dist,
            marker_length_m=float(cfg["marker_length_m"]),
            out_w=out_w,
            out_h=out_h,
            euler_ranges=euler_ranges,
            t_ranges=t_ranges,
            frustum_cfg=cfg.get("frustum_sampling", {"enabled": True})
        )

        # Project marker corners for this pose
        quad = project_marker_corners(cfg["marker_length_m"], K, dist, rvec, tvec)

        # Optionally draw a white rectangle behind the marker by projecting a larger square
        border_prob = float(cfg.get("marker_border_probability", 0.0))  # reuse probability to control backdrop
        enlarge_frac = float(cfg.get("white_rect_enlarge_frac", 0.25))
        white_rect_drawn = False
        if enlarge_frac > 0.0 and random.random() < max(0.0, min(1.0, border_prob)):
            L_white = (1.0 + float(enlarge_frac)) * float(cfg["marker_length_m"])
            quad_white = project_marker_corners(L_white, K, dist, rvec, tvec)
            if np.isfinite(quad_white).all():
                quad_white_int = quad_white.astype(np.int32)
                cv2.fillConvexPoly(out_img, quad_white_int, color=(255, 255, 255), lineType=cv2.LINE_8)
                white_rect_drawn = True

        # Composite marker over the background (and any border already drawn)
        out_img = warp_marker_onto_canvas(out_img, marker, quad, out_w, out_h)

        # Update segmentation map
        try:
            marker_id = extract_marker_id_from_path(Path(m_path))
        except Exception:
            # If id cannot be parsed, place into a default bucket 0
            marker_id = 0
        color_bgr = id_to_color_bgr(marker_id)
        mask = warp_marker_mask(marker, quad, out_w, out_h)
        if mask is not None:
            # Apply color where mask is present; later markers will overwrite earlier ones
            seg_img[mask > 0] = color_bgr
        if cfg.get("draw_axes", False):
            draw_axes(out_img, K, dist, rvec, tvec, axes_len_m=cfg.get("axes_len_m", 0.05))

        # Build 4x4 transform T_cam_marker from rvec/tvec
        Rm, _ = cv2.Rodrigues(rvec)
        T = np.eye(4, dtype=float)
        T[:3, :3] = Rm
        T[:3, 3] = tvec.reshape(3)

        # Record per-marker metadata
        markers_info.append({
            "marker_id": int(marker_id),
            "rvec": rvec.reshape(-1).tolist(),
            "tvec": tvec.reshape(-1).tolist(),
            "euler_deg_xyz": [float(euler_deg_xyz[0]),
                               float(euler_deg_xyz[1]),
                               float(euler_deg_xyz[2])],
            "quad_px": quad.astype(float).tolist(),
            "seg_color_bgr": [int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2])],
            "T_cam_marker": T.astype(float).tolist(),
            # Backdrop (white rectangle) metadata
            "white_rect_drawn": bool(white_rect_drawn),
            "white_rect_enlarge_frac": float(enlarge_frac if white_rect_drawn else 0.0),
            # Legacy keys preserved for compatibility
            "border_drawn": bool(white_rect_drawn),
            "border_thickness_px": 0,
        })

    # Overlay number of markers on the RGB image
    put_text_with_outline(out_img, f"{n_markers} markers", org=(10, 36), font_scale=1.0)

    # Save RGB image (optionally grayscale) and segmentation for this iteration
    out_path = images_dir / f"rendered_{i:05d}.png"
    if bool(cfg.get("save_bw", False)):
        out_to_save = cv2.cvtColor(out_img, cv2.COLOR_BGR2GRAY)
    else:
        out_to_save = out_img
    cv2.imwrite(str(out_path), out_to_save)

    seg_path = seg_dir / f"rendered_{i:05d}_seg.png"
    cv2.imwrite(str(seg_path), seg_img)

    # Save metadata JSON
    meta = {
        "index": int(i),
        "n_markers": int(n_markers),
        "output_size_wh": [int(out_w), int(out_h)],
        "background_path": str(bg_path),
        "camera": {
            "K": K.tolist(),
            "dist": dist.tolist(),
        },
        "marker_length_m": float(cfg["marker_length_m"]),
        "outputs": {
            "rgb": str(out_path.relative_to(images_dir.parent)),
            "segmentation": str(seg_path.relative_to(seg_dir.parent)),
            "image_mode": "bw" if bool(cfg.get("save_bw", False)) else "rgb",
        },
        "markers": markers_info,
    }
    meta_path = metadata_dir / f"rendered_{i:05d}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return True


# ------------------------ Main ------------------------
def main(cfg):
    # RNG seed (base). Workers will reseed for diversity.
    if cfg.get("seed") is not None:
        try:
            np.random.seed(int(cfg["seed"]))
            random.seed(int(cfg["seed"]))
        except Exception:
            pass

    out_w, out_h = int(cfg["output_size_wh"][0]), int(cfg["output_size_wh"][1])

    # Camera parameters
    K = ensure_float_array(cfg["K"], shape=(3, 3))
    dist = ensure_float_array(cfg.get("dist", [0, 0, 0, 0, 0])).ravel()
    if "K_size_wh" in cfg:
        K = scale_intrinsics(K, from_wh=cfg["K_size_wh"], to_wh=cfg["output_size_wh"]) 

    # IO
    bg_files = list_images_in_dir(Path(cfg["background_dir"]))
    marker_files = list_images_in_dir(Path(cfg["marker_dir"]))

    # Build timestamped output directory
    timestamp_str = time.strftime("%Y%m%d-%H%M%S")
    out_dir_cfg = cfg.get("output_dir")
    if out_dir_cfg:
        if "{timestamp}" in out_dir_cfg or "{time}" in out_dir_cfg:
            out_dir = Path(out_dir_cfg.format(timestamp=timestamp_str, time=timestamp_str))
        else:
            out_dir = Path(out_dir_cfg) / f"multi_marker_{timestamp_str}"
    else:
        out_dir = Path("data_generation/multi_marker_output") / f"multi_marker_{timestamp_str}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Subdirectories
    images_dir = out_dir / "images"
    seg_dir = out_dir / "segmentations"
    metadata_dir = out_dir / "metadata"
    images_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    n_images = int(cfg.get("n_images", 1))
    start_time = time.time()

    # Prepare immutable args for workers
    bg_files_str = [str(p) for p in bg_files]
    marker_files_str = [str(p) for p in marker_files]

    workers = int(cfg.get("num_workers", os.cpu_count() or 1))

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(
                _render_one_image,
                i,
                cfg,
                K,
                dist,
                bg_files_str,
                marker_files_str,
                out_w,
                out_h,
                str(images_dir),
                str(seg_dir),
                str(metadata_dir),
            ) for i in range(n_images)
        ]
        for _ in tqdm(as_completed(futures), total=n_images, desc="Rendering"):
            pass

    end_time = time.time()
    print(f"Generated {n_images} images in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main(CONFIG)
