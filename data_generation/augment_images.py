import numpy as np 
import pandas as pd 
import os 
import json 
from PIL import Image
import random 
import time 
import cv2 
import math 
import albumentations as A 
import shutil
import copy
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, *args, **kwargs):
        return x
from scipy.spatial.transform import Rotation as R 
import matplotlib.pyplot as plt 

from math import gcd, lcm
from random import randint, uniform
from random import random as random_function 
from math import cos, sin, radians 
from perlin_numpy import generate_fractal_noise_2d
import re

def to_grayscale(x, **kwargs):
    # Accept **kwargs because Albumentations may pass extra params (e.g., shape)
    # Always return 3-channel grayscale to satisfy transforms expecting 3 channels.
    if x.ndim == 2:
        gray = x
    elif x.ndim == 3:
        # Input from PIL is RGB; convert to single-channel gray first
        gray = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    else:
        return x
    # Convert back to 3-channel RGB grayscale
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

def project_point_to_image(C,T,P): 
    P_H = np.array([[P[0]],[P[1]],[P[2]],[1]]) 
    T_H = T[:3,:4]  
    uv = C @ T_H @ P_H 
    if uv[2] != 0: 
        uv = uv / uv[2] # NOTE: check if the alternative case of not dividing when z=0 is valid 
    uv = uv[:2] 
    uv = uv.reshape((2)) 
    return uv 

def project_point_list_to_image(C,T,P_list): 
    uv_list = []  
    for P in P_list: 
        uv = project_point_to_image(C,T,P) 
        uv_list.append(uv) 
    return uv_list   

def transform_pts(pts, T):  
    pts_transformed = [] 
    for pt in pts: 
        pt = pt.reshape(3,1) 
        pt = np.vstack((pt, 1))  
        pt_transformed = T @ pt  
        pts_transformed.append(pt_transformed[:3]) 
    return pts_transformed

def overlay_points_on_image(image, pixel_points, radius=5, color=(0, 0, 255), thickness=-1):
    """
    Overlays a list of pixel points on the input image.

    Parameters:
    - image: The input image (a NumPy array).
    - pixel_points: A list of 2D pixel coordinates [(x1, y1), (x2, y2), ...].
    - radius: The radius of the circle to draw around each point. Default is 5.
    - color: The color of the circle (BGR format). Default is red (0, 0, 255).
    - thickness: The thickness of the circle. Default is -1 to fill the circle.

    Returns:
    - The image with points overlaid.
    """
    # Iterate over each pixel point and overlay it on the image
    for point in pixel_points:
        if point is not None:  # Only overlay valid points
            x, y = int(point[0]), int(point[1])
            # check if the point is within the image bounds
            if x < 0 or x >= image.shape[1] or y < 0 or y >= image.shape[0]:
                continue
            # Draw a filled circle at the pixel coordinates
            cv2.circle(image, (x, y), radius, color, thickness)
    return image

def compute_2D_gridpoints(N=10,s=0.1): 
    # N = num squares, s = side length  
    u = np.linspace(-s/2, +s/2, N+1) 
    v = np.linspace(-s/2, +s/2, N+1) 
    gridpoints = [] 
    for uu in u:
        for vv in v: 
            gridpoints.append(np.array([uu,vv,0])) 
    return gridpoints 


def rotate3d(pic, rot_x, rot_y, rot_z, f_mult = 1.0, fill_color = (0,0,0)):

    height, width = [(2 * i) for i in pic.shape[0:2]]

    pic_exp = np.zeros((height, width, 4), dtype = np.uint8)
    pic_exp[:,:,:3] = fill_color
    pic_exp[pic.shape[0]//2:(height + pic.shape[0])//2,
            pic.shape[1]//2:(width + pic.shape[1])//2, :] = pic

    alpha = radians(rot_x)
    beta = radians(rot_y)
    gamma = radians(rot_z)

    f = (width / 2) * f_mult

    # 2d -> 3d
    proj2d3d = np.asarray([[1, 0, -width / 2],
                           [0, 1, -height / 2],
                           [0, 0, 0],
                           [0, 0, 1]])

    # Rotation matrices
    rx = np.asarray([[1, 0, 0, 0],
                     [0, cos(alpha), -sin(alpha), 0],
                     [0, sin(alpha), cos(alpha), 0],
                     [0, 0, 0, 1]])
    
    ry = np.asarray([[cos(beta), 0, sin(beta), 0],
                     [0, 1, 0, 0],
                     [-sin(beta), 0, cos(beta), 0],
                     [0, 0, 0, 1]])
    
    rz = np.asarray([[cos(gamma), -sin(gamma), 0, 0],
                     [sin(gamma), cos(gamma), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    
    # Translation
    T = np.asarray([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, f],
                    [0, 0, 0, 1]])
    
    # 3d -> 2d
    proj3d2d = np.asarray([[f, 0, width / 2, 0],
                           [0, f, height / 2, 0],
                           [0, 0, 1, 0]])
    
    # Combine all
    transform = proj3d2d @ (T @ ((rx @ ry @ rz) @ proj2d3d))
    pic_exp = cv2.warpPerspective(pic_exp, transform, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=fill_color)

    return pic_exp, transform

def gradient(width, height):

    t_size = max(width, height)
    size = t_size * 2

    # Vectorized vertical gradient [0..1]
    col = np.linspace(0.0, 1.0, num=size, dtype=np.float32)[:, None]
    grad = np.broadcast_to(col, (size, size)).copy()

    center = grad.shape[0] // 2
    mat = cv2.getRotationMatrix2D((center, center), random_function() * 360, 1.0)
    pic = cv2.warpAffine(grad, mat, (size, size))

    # Final crop
    center = grad.shape[0] // 2
    pic = pic[center - height // 2:center + height // 2, center - width // 2:center + width // 2]

    # Re-range (guard against flat image)
    pmin = float(pic.min())
    pmax = float(pic.max())
    if pmax - pmin < 1e-12:
        return np.zeros_like(pic, dtype=np.float32)
    pic = (pic - pmin) / (pmax - pmin)

    return pic

def lines(width, height, num_patterns = 3):

    t_size = max(width, height)
    size = t_size * 2

    pic = np.ones((size, size))
    center = pic.shape[0]//2

    for i in range(num_patterns):

        curr = 0

        while curr < size:
            paint = randint(1, max((size - curr)//2, 1))#min(randint(0, 16), size - curr)
            skip = randint(1, max((size - curr - paint)//2, 1))#min(randint(0, 16), size - curr - paint)
            pic[curr:curr + paint] *= uniform(0.0, 2.0)#random()
            curr = curr + paint + skip

        # Rotate

        mat = cv2.getRotationMatrix2D((center, center), random_function() * 360, 1.0)
        pic = cv2.warpAffine(pic, mat, (pic.shape[0], pic.shape[1]))

    # Re-range

    pic = (pic - np.min(pic)) / (np.max(pic) - np.min(pic) + 1e-6)

    # Perspective

    pic = cv2.merge((pic, pic, pic, np.ones(pic.shape))) * 255.0
    pic, _ = rotate3d(pic, randint(-30,30), randint(-30,30), 0)
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY) / 255.0

    # Final crop

    center = pic.shape[0]//2
    pic = pic[center - height//2:center + height//2, center - width//2:center + width//2]

    return pic

def circular(width, height):

    # Vectorized radial falloff around a random center
    cy = randint(0, height - 1)
    cx = randint(0, width - 1)

    diag = int((width**2 + height**2) ** 0.5)
    radius = max(1, randint(max(1, diag // 4), diag))

    yy, xx = np.ogrid[:height, :width]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.float32)
    pic = 1.0 - (dist / float(radius))
    np.clip(pic, 0.0, 1.0, out=pic)

    # Normalize for safety (usually already in [0,1])
    pmin = float(pic.min())
    pmax = float(pic.max())
    if pmax - pmin > 1e-12:
        pic = (pic - pmin) / (pmax - pmin)
    else:
        pic = np.zeros_like(pic, dtype=np.float32)
    return pic

def perlin(width, height, bins = 0, octaves = 4):

    t_width = lcm(width, 2 ** (octaves - 1))
    t_height = lcm(height, 2 ** (octaves - 1))

    res_x = t_width//gcd(t_width, t_height)
    res_y = t_height//gcd(t_width, t_height)

    # Fractal noise

    pic = generate_fractal_noise_2d((t_height, t_width), 
                                    (res_y, res_x), 
                                    octaves)

    # Re-range

    pic = (pic - np.min(pic)) / (np.max(pic) - np.min(pic) + 1e-6)

    # Threshold

    if bins > 1:
        pic = np.digitize(pic, [(i + 1) / bins for i in range(bins - 1)]) / (bins - 1)
    return pic

def lighting_augmentation(image): 
    # check if image is 0-1 or 0-255, convert to 0-1 
    # final image outputted is 0-255 
    image = np.array(image, dtype=np.float32)
    if image.size == 0:
        return image
    if image.max() > 1.0:
        # image is 0-255  
        image = image / 255.0 

    # Ensure 3D array (H, W, C)
    was_grayscale = (image.ndim == 2)
    if was_grayscale:
        image = image[:, :, None]

    height, width = image.shape[:2] 
    augmented_image = image.copy()

    # Apply effects with broadcasting to avoid large temporary repeats
    if np.random.rand() < 0.8:
        eff = lines(width, height).astype(np.float32)
        augmented_image *= eff[..., None]
    if np.random.rand() < 0.1:
        eff = perlin(width, height).astype(np.float32)
        augmented_image *= eff[..., None]
    if np.random.rand() < 0.8:
        eff = gradient(width, height).astype(np.float32)
        augmented_image *= eff[..., None]
    if np.random.rand() < 0.2:
        eff = circular(width, height).astype(np.float32)
        augmented_image *= eff[..., None]

    # Renormalize if too dark or out of range
    amax = float(augmented_image.max())
    amin = float(augmented_image.min())
    if (amax < 0.4) or (amax > 1.0):  # NOTE: HYPERPARAMETER
        pixel_max = np.random.uniform(0.9, 1.0)
        denom = max(1e-6, (amax - amin))
        augmented_image = pixel_max * (augmented_image - amin) / denom

    augmented_image *= 255.0

    # If original was grayscale, squeeze channel back out
    if was_grayscale:
        augmented_image = augmented_image[:, :, 0]
    
    return augmented_image 

class datapoint:
    def __init__(self, metadata_filepath, image_filepath, seg_png_filepath=None):
        # Store the filepaths (pose and seg json now consolidated in metadata)
        self.metadata_filepath = metadata_filepath
        self.image_filepath = image_filepath
        self.rgb_filepath = image_filepath  # maintain backward compatibility naming
        self.seg_png_filepath = seg_png_filepath  # optional; mask can be inferred from image if needed
        # Infer dataset root (two levels up from metadata file: .../metadata/<file>.json)
        try:
            self.dataset_root = os.path.dirname(os.path.dirname(self.metadata_filepath))
        except Exception:
            self.dataset_root = None
        # Caches and lazy attrs
        self._img_size = None  # (W, H) cached on first use
        self._color_map_rgb = {}

        # Load metadata and pose data now
        self.read_files()
        self.read_pose_data()
    

    def read_files(self):
        # Read the actual data from files and store it
        self.metadata = self._read_json(self.metadata_filepath) if self.metadata_filepath else None 
        self.rgb = self._read_rgb(self.rgb_filepath) if self.rgb_filepath else None
        self.seg_png = self._read_segmentation_png(self.seg_png_filepath) if self.seg_png_filepath else None


    def read_pose_data(self):
        # Reset structures
        self.marker_poses: dict[str, np.ndarray] = {}  # here stores T_cam_marker
        self.marker_info: dict[str, dict] = {}
        self.marker_quads: dict[str, np.ndarray] = {}
        self.active_markers: list[str] = []
        self.light = {}

        md = self.metadata if isinstance(self.metadata, dict) else {}

        # Camera intrinsics from metadata
        cam = md.get("camera", {})
        K = np.array(cam.get("K", []), dtype=float)
        if K.size == 9:
            self.camera_matrix = K.reshape(3, 3)
        # Lens distortion if needed later
        self.dist_coeffs = np.array(cam.get("dist", []), dtype=float) if cam else None

        # Marker side length
        if "marker_length_m" in md:
            try:
                self.marker_side_length = float(md["marker_length_m"])  # meters
            except Exception:
                pass

        # Outputs (paths)
        self.outputs = md.get("outputs", {})

        # Markers list
        markers = md.get("markers", [])
        total = 0
        if isinstance(markers, list):
            for m in markers:
                try:
                    mid = m.get("marker_id")
                    tag_name = f"tag{mid}" if mid is not None else f"tag{total}"
                    T = np.array(m.get("T_cam_marker", []), dtype=float)
                    if T.size == 16:
                        T = T.reshape(4, 4)
                    else:
                        continue
                    quad = np.array(m.get("quad_px", []), dtype=float)
                    if quad.shape == (4, 2):
                        self.marker_quads[tag_name] = quad
                    self.marker_poses[tag_name] = T
                    self.marker_info[tag_name] = {
                        "marker_id": mid,
                        "seg_color_bgr": m.get("seg_color_bgr"),
                        "rvec": m.get("rvec"),
                        "tvec": m.get("tvec"),
                        "euler_deg_xyz": m.get("euler_deg_xyz"),
                    }
                    self.active_markers.append(tag_name)
                    total += 1
                except Exception:
                    continue
        self.num_active_markers = len(self.active_markers)
        self.total_markers = total
        # Cache color map (BGR -> RGB) for segmentation matching
        self._color_map_rgb = {}
        try:
            for tag in self.active_markers:
                info = (self.marker_info or {}).get(tag)
                if info and info.get('seg_color_bgr') is not None:
                    bgr = info['seg_color_bgr']
                    if isinstance(bgr, (list, tuple)) and len(bgr) >= 3:
                        self._color_map_rgb[tag] = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
        except Exception:
            self._color_map_rgb = {}

    def compute_keypoints(self, keypoints_tag_frame, camera_matrix=None):
        # Project keypoints defined in marker frame using T_cam_marker directly.
        K = self.camera_matrix if camera_matrix is None else camera_matrix
        keypoints_all = {}
        for tag_name, T_c_m in self.marker_poses.items():
            uv_list = []
            for kp_t in keypoints_tag_frame:
                pt = np.hstack((kp_t, np.array([1.0]))).reshape(4, 1)
                Pc = (T_c_m @ pt)[:3].reshape(3)
                if Pc[2] == 0:
                    uv_list.append(np.array([np.nan, np.nan]))
                    continue
                uv_h = K @ Pc.reshape(3, 1)
                uv = (uv_h[:2] / Pc[2]).reshape(2)
                uv_list.append(uv)
            keypoints_all[tag_name] = uv_list
        self.keypoints_image_space = keypoints_all
        return keypoints_all
    
    def _read_json(self, filepath):
        """Read and parse JSON files."""
        with open(filepath, 'r') as file:
            return json.load(file)

    def _read_rgb(self, filepath):
        """Placeholder for reading RGB image files."""
        return filepath  # Placeholder: returning the file path to avoid memory overload

    def _read_segmentation_png(self, filepath):
        """Placeholder for reading segmentation PNG image files."""
        return filepath  # Placeholder: returning the file path to avoid memory overload

    def _read_segmentation_json(self, filepath):
        """Deprecated: segmentation mapping resides in metadata now. Kept for compatibility if needed."""
        if not filepath:
            return None
        with open(filepath, 'r') as file:
            return json.load(file)

    def compute_diffusion_reflectance(self):
        """Compute a simple diffuse reflection proxy based on camera/marker normals and light exposure.
        For multi-markers, compute the max across markers as a heuristic.
        """
        light_exposure = 0.0
        I_incident = 2 ** light_exposure
        values = []
        for tf in self.marker_poses.values():
            # Treat camera z-axis as viewing direction; marker normal is tf[:3,2] in camera frame
            N = np.array(tf)[:3, 2]
            L = np.array([0, 0, 1], dtype=float)
            values.append(max(float(np.dot(N, L)), 0.0))
        self.diffuse_reflection = I_incident * (max(values) if values else 0.0)

    def preprocess_seg_img(self, tag_name: str | None = None):
        """Return a binary segmentation mask from quad polygons in metadata.
        This is for internal use (ROI/filtering). Saved segmentation files remain unchanged.
        """
        # Determine image size from cached metadata or read once
        if not self._img_size:
            with Image.open(self.rgb_filepath) as _im_sz:
                self._img_size = _im_sz.size  # (W, H)
        W, H = self._img_size
        mask = np.zeros((H, W), dtype=np.uint8)

        # Decide which quads to render
        include_tags = [tag_name] if tag_name else list(self.marker_quads.keys())
        for tname in include_tags:
            quad = self.marker_quads.get(tname)
            if quad is None or quad.shape != (4, 2):
                continue
            poly = np.round(quad).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [poly], color=255)
        return Image.fromarray(mask)
    
    def get_roi_image(self, seg=None, roi_size=128, padding=5, tag_name: str | None = None):
        # Fallback to metadata-derived mask when seg not provided
        if seg is None:
            seg = self.preprocess_seg_img(tag_name=tag_name)

        # Border padding large enough to safely crop around bbox
        seg_arr = np.array(seg, dtype=np.uint8)
        image_border_size = int(np.max([seg_arr.shape[0], seg_arr.shape[1]]))

        # Fast bbox on mask
        seg_np = cv2.copyMakeBorder(
            seg_arr,
            image_border_size,
            image_border_size,
            image_border_size,
            image_border_size,
            cv2.BORDER_CONSTANT,
            value=0,
        )
        x, y, w, h = cv2.boundingRect(seg_np)
        seg_tag_min_x = int(x)
        seg_tag_max_x = int(x + w)
        seg_tag_min_y = int(y)
        seg_tag_max_y = int(y + h)
        seg_height = int(h)
        seg_width = int(w)
        seg_center_x = int(x + w // 2)
        seg_center_y = int(y + h // 2)

        # Load RGB and prepare same border
        with Image.open(self.rgb_filepath) as _im_rgb:
            rgb = np.array(_im_rgb)
        rgb = cv2.copyMakeBorder(
            rgb,
            image_border_size,
            image_border_size,
            image_border_size,
            image_border_size,
            cv2.BORDER_CONSTANT,
            value=0,
        )
        rgb_side = int(max(seg_height, seg_width) + 2 * padding)
        rgb_tag_min_x = int(seg_center_x - rgb_side // 2)
        rgb_tag_max_x = int(seg_center_x + rgb_side // 2)
        rgb_tag_min_y = int(seg_center_y - rgb_side // 2)
        rgb_tag_max_y = int(seg_center_y + rgb_side // 2)
        roi_img = rgb[rgb_tag_min_y:rgb_tag_max_y, rgb_tag_min_x:rgb_tag_max_x, :]

        # Resize ROI to fixed size
        try:
            self.roi_img = cv2.resize(roi_img, (roi_size, roi_size))
        except Exception:
            self.roi_img = roi_img

        W = int(rgb.shape[1])
        H = int(rgb.shape[0])
        # image (x,y) coordinates (origin at image center)
        self.roi_coordinates = np.array([
            rgb_tag_min_x - W / 2,
            rgb_tag_max_x - W / 2,
            rgb_tag_min_y - H / 2,
            rgb_tag_max_y - H / 2,
        ])

        self.roi_center = np.array([seg_center_x, seg_center_y]) - np.array([image_border_size, image_border_size])

        self.W_img = W
        self.H_img = H
        self.img_center = np.array([W / 2, H / 2])

        return self.roi_img, self.roi_coordinates, self.roi_center
    
    def get_roi_keypoints(self, tag_name: str | None = None):

        # check if keypoints and roi have been computed, else return None 
        if not hasattr(self, 'keypoints_image_space') or not hasattr(self, 'roi_img'):
            return None 
        
        # get keypoints in roi space
        roi_keypoints = []
        # Determine which keypoints to use
        if isinstance(self.keypoints_image_space, dict):
            # Use requested tag or flatten all
            kps_iter = []
            if tag_name and tag_name in self.keypoints_image_space:
                kps_iter = self.keypoints_image_space[tag_name]
            else:
                for lst in self.keypoints_image_space.values():
                    kps_iter.extend(lst)
        else:
            kps_iter = self.keypoints_image_space

        for kp in kps_iter:
            s = np.array(self.roi_img.shape[:2]) 
            w = self.roi_coordinates[1] - self.roi_coordinates[0]
            h = self.roi_coordinates[3] - self.roi_coordinates[2]
            m = s / np.array([w, h])     
            # m = s / np.array([self.W_img, self.H_img])     
            kp_roi = m*(kp - self.roi_center) + s/2 
            roi_keypoints.append(kp_roi) 

        self.roi_keypoints = roi_keypoints 

        return self.roi_keypoints  

    # def __repr__(self):
    #     """Custom representation for the datapoint object."""
    #     # return f"datapoint(metadata_filepath={self.metadata_filepath}, pose_filepath={self.pose_filepath}, rgb_filepath={self.rgb_filepath}, seg_png_filepath={self.seg_png_filepath}, seg_json_filepath={self.seg_json_filepath})"
    #     description = [
    #         f"lighting_exposure={self.metadata["light"]["exposure"]:.2f}",
    #         # f"lighting_color={str(self.metadata["light"]["color"]) }" # FIXME: reduce to two decimal places 
    #         f"lighting_color=({self.metadata["light"]["color"][0]:.2f},{self.metadata["light"]["color"][1]:.2f},{self.metadata["light"]["color"][2]:.2f})" # FIXME: reduce to two decimal places 
    #     ]
    #     return "\n".join(description) 

class DataProcessor:
    def __init__(self, data_folders, out_dir, fast_mode: bool | None = None, max_datapoints_total: int | None = None, resume: bool | None = None):
        self.data_folders = data_folders
        self.out_dir = out_dir
        self.datapoints = []
        self.datapoints_train = []
        self.datapoints_val = []
        # Fast mode toggle: arg overrides env; default False
        if fast_mode is None:
            env_fast = os.environ.get("DME_FAST") or os.environ.get("DME_FAST_MODE")
            self.fast_mode = str(env_fast).lower() in {"1", "true", "yes"}
        else:
            self.fast_mode = bool(fast_mode)
        # Max total datapoints (train + val): arg overrides env; default None (unlimited)
        if max_datapoints_total is None:
            env_max = os.environ.get("DME_MAX_DATAPOINTS") or os.environ.get("DME_MAX_SAMPLES")
            try:
                self.max_datapoints_total = int(env_max) if env_max not in (None, "", "None") else None
                if self.max_datapoints_total is not None and self.max_datapoints_total <= 0:
                    self.max_datapoints_total = None
            except Exception:
                self.max_datapoints_total = None
        else:
            self.max_datapoints_total = int(max_datapoints_total)
            if self.max_datapoints_total <= 0:
                self.max_datapoints_total = None
        # Resume toggle: arg overrides env; default True
        if resume is None:
            env_resume = os.environ.get("DME_RESUME", "1")
            self.resume = str(env_resume).lower() in {"1", "true", "yes"}
        else:
            self.resume = bool(resume)
        # Initialize transforms and camera
        self.set_augmentation_transforms()
        self.set_camera(camera_name="homography")

    # Resume helpers
    def _resume_state_path(self, dataset_dir: str) -> str:
        return os.path.join(dataset_dir, ".resume_state.json")

    def _load_resume_state(self, dataset_type: str, dataset_dir: str) -> dict:
        path = self._resume_state_path(dataset_dir)
        try:
            if os.path.isfile(path):
                with open(path, "r") as f:
                    state = json.load(f)
                return state.get(dataset_type, {"done_ids": [], "complete": False})
        except Exception:
            pass
        return {"done_ids": [], "complete": False}

    def _write_full_resume_state(self, dataset_dir: str, dataset_type: str, data: dict):
        path = self._resume_state_path(dataset_dir)
        try:
            full = {}
            if os.path.isfile(path):
                with open(path, "r") as f:
                    try:
                        full = json.load(f) or {}
                    except Exception:
                        full = {}
            full[dataset_type] = data
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(full, f)
            os.replace(tmp, path)
        except Exception:
            # Best-effort; ignore write errors
            pass

    def _save_resume_state(self, dataset_type: str, dataset_dir: str, done_ids: set, complete: bool = False):
        data = {
            "done_ids": sorted(list(done_ids)),
            "complete": bool(complete),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._write_full_resume_state(dataset_dir, dataset_type, data)

    def _datapoint_uid(self, dp) -> str:
        """Stable id for a datapoint across runs, based on trailing numeric id in filenames."""
        def _get_numeric_id(filename: str) -> str:
            base = os.path.splitext(os.path.basename(filename))[0]
            if base.endswith('_seg'):
                base = base[:-4]
            m = re.search(r'(\d+)$', base)
            return m.group(1) if m else base
        try:
            if getattr(dp, "metadata_filepath", None):
                return _get_numeric_id(dp.metadata_filepath)
        except Exception:
            pass
        try:
            if getattr(dp, "rgb_filepath", None):
                return _get_numeric_id(dp.rgb_filepath)
        except Exception:
            pass
        return f"dp_{id(dp)}"

    def _get_files_in_subfolder(self, folder, file_extension=None):
        """Helper method to get files in a subfolder, with an optional file extension filter."""
        files_list = os.listdir(folder)
        if file_extension:
            files_list = [file for file in files_list if file.endswith(file_extension)]
        # Order files_list by date created
        files_list = sorted(files_list, key=lambda x: os.path.getctime(os.path.join(folder, x)))  # Assumes creation dates are synchronized
        return files_list
    
    def set_marker(self, image_path, num_squares, side_length): 
        self.marker_path = image_path 
        # self.marker_image = Image.open(image_path) 
        self.marker_num_squares = num_squares 
        self.marker_side_length = side_length 
        self.keypoints_tag_frame = compute_2D_gridpoints(N=self.marker_num_squares, s=self.marker_side_length) 

    def set_camera(self, camera_name="homography", camera_matrix=None):  
        # default camera is isaac 
        if camera_name == "homography": 
            # camera parameters 
            width = 1920 
            height = 1200 
            # focal_length = 12.5 
            # horiz_aperture = 20.955
            # # Pixels are square so we can do:
            # vert_aperture = height/width * horiz_aperture
            # fov = 2 * math.atan(horiz_aperture / (2 * focal_length))
            # compute focal point and center
            fx = 1400 # width * focal_length / horiz_aperture
            fy = 1400 # height * focal_length / vert_aperture
            cx = width / 2
            cy = height /2 

            self.camera_matrix = np.array([
                [fx,0,cx],
                [0,fy,cy],
                [0,0,1]
            ])
        if camera_matrix is not None: 
            self.camera_matrix = camera_matrix 

    def process_folders(self):
        """Process the folders and create datapoint objects.
        Supports new layout (images, metadata, segmentations) with consolidated metadata,
        and falls back to legacy (rgb, seg, pose) if needed.
        """
        print("[phase] process_folders: start scanning folders")
        def get_numeric_id(filename):
            # Extract trailing numeric ID from base name, handling optional '_seg' suffix.
            # Examples:
            #  - 'rendered_00000.png' -> '00000'
            #  - 'rendered_00000_seg.png' -> '00000'
            #  - 'metadata_00000.json' -> '00000'
            base = os.path.splitext(filename)[0]
            if base.endswith('_seg'):
                base = base[:-4]
            m = re.search(r'(\d+)$', base)
            return m.group(1) if m else base

        def resolve_subdir(base, candidates):
            for c in candidates:
                p = os.path.join(base, c)
                if os.path.isdir(p):
                    return p
            return os.path.join(base, candidates[0])

        for data_folder in tqdm(self.data_folders, desc="scan folders", unit="dir"):
            print(f"[info] scanning: {data_folder}")
            metadata_subfolder = resolve_subdir(data_folder, ["metadata"])  # required
            images_subfolder = resolve_subdir(data_folder, ["images", "rgb"])  # new -> legacy
            seg_subfolder = resolve_subdir(data_folder, ["segmentations", "seg"])  # new -> legacy
            pose_subfolder = os.path.join(data_folder, "pose")  # legacy only

            # list files
            metadata_files = self._get_files_in_subfolder(metadata_subfolder, file_extension=".json")

            # support .png and .jpg images
            image_files_png = self._get_files_in_subfolder(images_subfolder, file_extension=".png") if os.path.isdir(images_subfolder) else []
            image_files_jpg = self._get_files_in_subfolder(images_subfolder, file_extension=".jpg") if os.path.isdir(images_subfolder) else []
            rgb_files = sorted(list(set(image_files_png + image_files_jpg)))

            seg_png_files = self._get_files_in_subfolder(seg_subfolder, file_extension=".png") if os.path.isdir(seg_subfolder) else []

            # Create dictionaries keyed by file index
            metadata_dict = {get_numeric_id(f): f for f in metadata_files}
            rgb_dict = {get_numeric_id(f): f for f in rgb_files}
            seg_png_dict = {get_numeric_id(f): f for f in seg_png_files}

            # Only use keys common to metadata and images; segmentation optional
            common_ids = sorted(set(metadata_dict) & set(rgb_dict))
            print(f"[info] found: metadata={len(metadata_files)} images={len(rgb_files)} segs={len(seg_png_files)} -> pairs={len(common_ids)}")

            if not common_ids:
                print(f"No matching file IDs found in metadata/images for: {data_folder}")
                continue

            for file_id in tqdm(common_ids, desc=f"pair {os.path.basename(data_folder)}", unit="file", leave=False):
                metadata_filepath = os.path.join(metadata_subfolder, metadata_dict[file_id])
                image_filepath = os.path.join(images_subfolder, rgb_dict[file_id])
                seg_png_filepath = os.path.join(seg_subfolder, seg_png_dict[file_id]) if file_id in seg_png_dict else None

                # Create and store datapoint
                dp = datapoint(metadata_filepath, image_filepath, seg_png_filepath)
                # Propagate camera intrinsics and marker size to datapoint for synthetic masks
                dp.camera_matrix = getattr(self, 'camera_matrix', None)
                dp.marker_side_length = getattr(self, 'marker_side_length', 0.100)
                self.datapoints.append(dp)
        print("[phase] process_folders: completed")


    def get_datapoints(self):
        """Return the list of datapoint objects."""
        return self.datapoints
    
    def get_datapoints_filtered(self):
        """Return the list of filtered datapoint objects."""
        return self.datapoints_filtered 
    
    def check_image_okay(self, rgb_img, seg_img, min_tag_area=1000, min_tag_pix_mean=25, max_tag_pix_mean=250, min_tag_pix_contrast=10): 
        if rgb_img is None: 
            return False 
        seg_img = np.array(seg_img) 
        # compute pixel area of tag segmentation 
        tag_pix_area = np.sum(seg_img == 255) 

        # create list of marker pixels using segmentation 
        marker_pixels = np.argwhere(seg_img != 0)  # Get the indices of pixels where the tag is present 
        # compute contrast of marker pixels using rgb image 
        rgb_img = np.array(rgb_img) 
        if rgb_img.max() <= 1.0: 
            rgb_img *= 255.0 
        marker_rgb_values = rgb_img[marker_pixels[:, 0], marker_pixels[:, 1]]  # Get the RGB values of the marker pixels 
        marker_grey_values = np.mean(marker_rgb_values, axis=1)  # Compute the mean RGB values of the marker pixels 
        # compute contrast as the difference in magnitude of the RGB values of the marker pixels 
        tag_pix_contrast = marker_grey_values.max() - marker_grey_values.min()  
        tag_pix_mean = marker_grey_values.mean()

        if (
            # tag_pix_area > min_tag_area # NOTE: disabled area check 
            tag_pix_mean > min_tag_pix_mean 
            and tag_pix_mean > min_tag_pix_mean 
            and tag_pix_mean < max_tag_pix_mean
            and tag_pix_contrast >= min_tag_pix_contrast
        ):  # enforce min contrast and reasonable brightness
            bool_image_ok = True 
        else:
            bool_image_ok = False
            # Optional debug prints; keep quiet by default
            # if not (tag_pix_mean > min_tag_pix_mean): 
            #     print("tag pix mean too low ") 
            # if not (tag_pix_mean < max_tag_pix_mean):
            #     print("tag pix mean too high")      
            # if not (tag_pix_contrast >= min_tag_pix_contrast):
            #     print("tag pix contrast too low")
        return bool_image_ok

    def check_image_okay_multi_marker(self,
                                      rgb_img,
                                      dp,
                                      seg_img=None,
                                      min_tag_pix_mean=25,
                                      max_tag_pix_mean=250,
                                      min_tag_pix_contrast=10):
        """Validate that every visible marker passes brightness and contrast thresholds.

        Parameters:
        - rgb_img: numpy array (H,W,3) or PIL Image
        - dp: datapoint instance containing marker metadata (seg colors and quads)
        - seg_img: optional PIL Image or numpy array of the segmentation; if None, will try to load
                   from dp.seg_png_filepath or dp.outputs['segmentation']; if still unavailable,
                   falls back to quad-based per-marker masks via dp.preprocess_seg_img(tag_name).
        - min_tag_pix_mean, max_tag_pix_mean, min_tag_pix_contrast: thresholds

        Returns: (all_ok: bool, tag_stats: dict)
        tag_stats[tag] = {area, mean, contrast, ok}
        """
        # Normalize inputs
        if isinstance(rgb_img, Image.Image):
            rgb = np.array(rgb_img)
        else:
            rgb = np.array(rgb_img)
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[:, :, None], 3, axis=2)
        if rgb.max() <= 1.0:
            rgb = rgb * 255.0

        # Try to get a color segmentation array (prefer provided to avoid IO)
        seg_arr = None
        if seg_img is not None:
            seg_arr = np.array(seg_img)
        else:
            seg_path = None
            if getattr(dp, 'seg_png_filepath', None) and os.path.isfile(dp.seg_png_filepath):
                seg_path = dp.seg_png_filepath
            else:
                try:
                    seg_rel = (getattr(dp, 'outputs', {}) or {}).get('segmentation')
                    if seg_rel and getattr(dp, 'dataset_root', None):
                        cand = os.path.join(dp.dataset_root, seg_rel)
                        if os.path.isfile(cand):
                            seg_path = cand
                except Exception:
                    pass
            if seg_path:
                try:
                    seg_arr = np.array(Image.open(seg_path).convert('RGB'))
                except Exception:
                    seg_arr = None

        # Build color map from metadata (BGR -> RGB)
        # Use cached color map if available
        color_map_rgb = getattr(dp, '_color_map_rgb', None) or {}
        marker_tags = getattr(dp, 'active_markers', list(getattr(dp, 'marker_poses', {}).keys()))
        if not color_map_rgb:
            # Fallback build once
            for tag in marker_tags:
                info = (getattr(dp, 'marker_info', {}) or {}).get(tag)
                if info and info.get('seg_color_bgr') is not None:
                    bgr = info['seg_color_bgr']
                    if isinstance(bgr, (list, tuple)) and len(bgr) >= 3:
                        color_map_rgb[tag] = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
            try:
                dp._color_map_rgb = color_map_rgb
            except Exception:
                pass

        tag_stats = {}
        all_ok = True

        for tag in marker_tags:
            # Derive per-marker mask
            mask = None
            if seg_arr is not None and seg_arr.ndim == 3 and tag in color_map_rgb:
                color = np.array(color_map_rgb[tag], dtype=np.uint8)
                mask = np.all(seg_arr == color, axis=-1).astype(np.uint8) * 255
            else:
                # Fallback: use quad-derived mask
                try:
                    mask_img = dp.preprocess_seg_img(tag_name=tag)
                    mask = np.array(mask_img)
                except Exception:
                    mask = None

            if mask is None:
                tag_stats[tag] = {'area': 0, 'mean': 0.0, 'contrast': 0.0, 'ok': False}
                all_ok = False
                continue

            m = (mask == 255)
            if not np.any(m):
                tag_stats[tag] = {'area': 0, 'mean': 0.0, 'contrast': 0.0, 'ok': False}
                all_ok = False
                continue

            vals = rgb[m]
            gray_vals = np.mean(vals, axis=1)
            contrast = float(np.ptp(gray_vals))
            mean_val = float(gray_vals.mean())
            area = int(m.sum())

            ok = (mean_val > min_tag_pix_mean) and (mean_val < max_tag_pix_mean) and (contrast >= min_tag_pix_contrast)
            tag_stats[tag] = {
                'area': area,
                'mean': mean_val,
                'contrast': contrast,
                'ok': bool(ok),
            }
            if not ok:
                all_ok = False

        # Attach stats on dp for downstream inspection
        try:
            dp.tag_stats_multi = tag_stats
        except Exception:
            pass

        return all_ok, tag_stats

    def filter_datapoints(self, min_tag_area=1000, min_tag_pix_mean=25, max_tag_pix_mean=250, min_tag_pix_contrast=10): 
        """Keep datapoints only if ALL active markers are viewable.
        Viewable means per-tag mask passes area, mean, and contrast thresholds.
        Also stores per-tag stats on the datapoint for debugging.
        """
        self.datapoints_filtered = [] 
        self.datapoints_filtered_out = [] 
        for idx, dp in enumerate(self.datapoints):
            dp.compute_diffusion_reflectance() 
            # Load RGB once
            rgb_img = np.array(Image.open(dp.rgb_filepath))

            # Evaluate each active marker separately
            all_ok = True
            tag_stats = {}
            tags = getattr(dp, 'active_markers', list(getattr(dp, 'marker_poses', {}).keys()))
            if not tags:
                # No markers listed; treat as failure
                all_ok = False
            for tag_name in tags:
                try:
                    seg_img_tag = dp.preprocess_seg_img(tag_name=tag_name)
                    ok = self.check_image_okay(
                        rgb_img,
                        seg_img_tag,
                        min_tag_area=min_tag_area,
                        min_tag_pix_mean=min_tag_pix_mean,
                        max_tag_pix_mean=max_tag_pix_mean,
                        min_tag_pix_contrast=min_tag_pix_contrast,
                    )
                    seg_np = np.array(seg_img_tag)
                    marker_pixels = np.argwhere(seg_np == 255)
                    if marker_pixels.size > 0:
                        vals = rgb_img[marker_pixels[:, 0], marker_pixels[:, 1]]
                        gray_vals = np.mean(vals, axis=1)
                        contrast = float(gray_vals.max() - gray_vals.min())
                        mean_val = float(gray_vals.mean())
                        area = int(marker_pixels.shape[0])
                    else:
                        contrast = 0.0
                        mean_val = 0.0
                        area = 0
                        ok = False
                    tag_stats[tag_name] = {
                        'area': area,
                        'mean': mean_val,
                        'contrast': contrast,
                        'ok': bool(ok),
                    }
                    if not ok:
                        all_ok = False
                except Exception as e:
                    print(f"[warning] error processing tag '{tag_name}' in datapoint idx={idx}: {e}")
                    tag_stats[tag_name] = {'area': 0, 'mean': 0.0, 'contrast': 0.0, 'ok': False}
                    all_ok = False

                    # show image
                    import matplotlib.pyplot as plt
                    plt.imshow(rgb_img)
                    plt.show()

            # Attach stats for downstream debugging/inspection
            dp.tag_stats = tag_stats
            dp.all_tags_viewable = all_ok

            if all_ok:
                self.datapoints_filtered.append(dp)
            else:
                self.datapoints_filtered_out.append(dp)

            # if dp.diffuse_reflection > min_diffuse_reflection and dp.tag_pix_area > min_tag_area and dp.tag_pix_mean > min_tag_pix_mean and dp.tag_pix_mean < max_tax_pix_mean:  # FIXME: hardcoded threshold for tag area and diffuse reflection 
            #     self.datapoints_filtered.append(dp) 
            # else: 
            #     self.datapoints_filtered_out.append(dp) 

            if len(self.datapoints) > 0 and (idx % max(1, int(len(self.datapoints)/10)) == 0): 
                print(f"[filter] Processed {idx} / {len(self.datapoints)} | kept={len(self.datapoints_filtered)} dropped={len(self.datapoints_filtered_out)}") 
                
    def split_train_val_data(self, filter=True, frac_train=0.8, num_points_max=None):
        """Split the datapoints into training and validation datasets.
        - If num_points_max is provided, cap total (train + val) to this value.
        - Else, if self.max_datapoints_total is set, use that cap.
        - Else, use all available datapoints.
        """
        # Source pool
        pool = self.datapoints_filtered if filter else self.datapoints
        N_available = len(pool)

        # Resolve cap
        cap = None
        if num_points_max is not None:
            try:
                cap = int(num_points_max)
            except Exception:
                cap = None
        if cap is None:
            cap = self.max_datapoints_total

        if cap is None or cap < 0:
            num_points = N_available
        else:
            num_points = min(cap, N_available)

        if num_points <= 0:
            self.datapoints_train = []
            self.datapoints_val = []
            return

        # Exact split counts (ensure train + val == num_points)
        n_train = int(round(frac_train * num_points))
        n_train = max(0, min(n_train, num_points))
        n_val = num_points - n_train

        # Sample without replacement
        if num_points == N_available:
            base = pool.copy()
        else:
            base = random.sample(pool, num_points)

        if n_train > 0:
            self.datapoints_train = random.sample(base, n_train)
        else:
            self.datapoints_train = []
        remaining = [dp for dp in base if dp not in self.datapoints_train]
        if n_val > 0 and remaining:
            n_val = min(n_val, len(remaining))
            self.datapoints_val = random.sample(remaining, n_val)
        else:
            self.datapoints_val = []

    def create_directories(self):
        """Create directories for training and validation data."""
        dir_train = os.path.join(self.out_dir, "train")
        dir_val = os.path.join(self.out_dir, "val")
        dir_train_rgb = os.path.join(dir_train, "rgb")
        dir_train_seg = os.path.join(dir_train, "seg")
        dir_val_rgb = os.path.join(dir_val, "rgb")
        dir_val_seg = os.path.join(dir_val, "seg")

        os.makedirs(dir_train_rgb, exist_ok=True)
        os.makedirs(dir_train_seg, exist_ok=True)
        os.makedirs(dir_val_rgb, exist_ok=True)
        os.makedirs(dir_val_seg, exist_ok=True)

        return dir_train_rgb, dir_train_seg, dir_val_rgb, dir_val_seg

    def preprocess_rgb(self, img_path):  
        """Preprocess RGB image by resizing it."""
        # new_size = (480, 270)  # Define the new size
        # new_size = (480*2, 270*2)  # Define the new size
        img = Image.open(img_path)
        # img_resized = img.resize(new_size)
        img_resized = img 
        return img_resized

    def preprocess_seg_img(self, seg_img_path, seg_json_path, tag_seg_color=None):
        """
        Preprocesses the segmentation image by resizing and converting it to a binary mask based on tag color.
        """
        # Validate that the segmentation image file exists
        if not os.path.exists(seg_img_path):
            raise FileNotFoundError(f"Segmentation image file not found: {seg_img_path}")

        # Validate that the JSON file exists
        if not os.path.exists(seg_json_path):
            raise FileNotFoundError(f"Segmentation JSON file not found: {seg_json_path}")

        # Load the segmentation JSON data if tag_seg_color is not provided
        if tag_seg_color is None:
            with open(seg_json_path, 'r') as json_file:
                seg_json = json.load(json_file)

            # Find the tag color from the JSON data
            for key, val in seg_json.items(): 
                if val.get("class") == "tag0":  
                    # Convert the key (which is a string representing a tuple) into an actual tuple
                    tag_seg_color = tuple(map(int, key.strip('()').split(', ')))  # Convert string '(140, 25, 255, 255)' into a tuple (140, 25, 255, 255)
                    break
            else:
                # raise ValueError("Tag with class 'tag0' not found in JSON.")
                tag_seg_color = tuple([-1,-1,-1,-1]) # impossible color value # FIXME: this is a workaround which can be turned into something more elegant 

        # Load and resize the segmentation image
        seg_img = Image.open(seg_img_path)
        # new_size = (480, 270)
        # new_size = (480*2, 270*2)
        # seg_img_resized = seg_img.resize(new_size)
        seg_img_resized = seg_img

        # Convert the resized image to a NumPy array
        seg_img_resized = np.array(seg_img_resized)

        # Check if the image is RGB (3 channels) or RGBA (4 channels) or grayscale (1 channel)
        if len(seg_img_resized.shape) == 3:
            if seg_img_resized.shape[2] == 3:  # RGB image
                # Compare each pixel to the tag color (e.g., RGB triplet)
                seg_img_resized = np.all(seg_img_resized == tag_seg_color[:3], axis=-1)  # Create binary mask for RGB image
            elif seg_img_resized.shape[2] == 4:  # RGBA image
                # Compare each pixel to the tag color (RGBA)
                seg_img_resized = np.all(seg_img_resized == tag_seg_color, axis=-1)  # Create binary mask for RGBA image
        else:  # If it's a single channel (grayscale), use it directly
            seg_img_resized = seg_img_resized == tag_seg_color  # Compare pixel values directly

        # Convert the binary mask to uint8 type (0 or 1)
        seg_img_resized = (seg_img_resized).astype(np.uint8) * 255  # Multiply by 255 to match image range

        # Convert the binary mask back to an image
        seg_img_resized = Image.fromarray(seg_img_resized)

        return seg_img_resized

    def set_augmentation_transforms(self):
        if getattr(self, 'fast_mode', False):
            transform = A.Compose([
                A.Lambda(image=to_grayscale, p=1.0),
                A.GaussNoise(var_limit=(0, 0.002), per_channel=False, p=0.6),
                A.AdvancedBlur(blur_limit=(3, 9), p=0.4),
                A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.6, 0.6), p=0.6),
            ])
        else:
            transform = A.Compose([
                # A.RandomShadow(shadow_roi=(0,0,1,1), ...),
                A.Lambda(image=to_grayscale, p=1.0),
                A.RandomSunFlare(flare_roi=(0, 0, 1, 1), num_flare_circles_range=(10, 30), src_radius=80, src_color=(175, 175, 175), method="physics_based", p=0.5),
                A.GaussNoise(var_limit=(0, 0.005), per_channel=True, p=1.0),
                A.AdvancedBlur(blur_limit=(5, 31), p=0.7),
                A.RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.95, 0.95), p=0.8),
            ])
        self.albumentations_transform = transform 

    def augment_image(self, image, dp, seg=None, 
                      max_attempts_lighting=10, 
                      max_attempts_combined=20,
                      min_tag_area=1000,
                      min_tag_pix_mean=25,
                      max_tag_pix_mean=250,
                      min_tag_pix_contrast=10,
                      allow_partial_marker_accept=False,
                      partial_accept_min_fraction=0.80):
        """Augment an image and validate it with filtering after augmentation.
        Returns an augmented image (np.ndarray) if it passes the filter within the
        allowed attempts; otherwise returns None."""
        base = np.array(image)[:, :, :3]
        augmented_image = None
        for attempt in range(max_attempts_combined):
            # photometric augmentation
            aug = self.albumentations_transform(image=base)["image"]
            # if attempt < max_attempts_lighting:
            #     aug = lighting_augmentation(aug)
            # apply filter on augmented image
            all_ok, tag_stats = self.check_image_okay_multi_marker(
                aug, dp, seg_img=seg,
                min_tag_pix_mean=min_tag_pix_mean,
                max_tag_pix_mean=max_tag_pix_mean,
                min_tag_pix_contrast=min_tag_pix_contrast,
            )
            # Derive pass/fail tag sets
            passed_tags = [t for t, s in tag_stats.items() if s.get('ok')]
            failed_tags = [t for t in tag_stats.keys() if t not in passed_tags]
            total_tags = max(1, len(tag_stats))
            pass_fraction = len(passed_tags) / total_tags

            # Accept if all pass, or if partial is allowed and enough pass
            if all_ok or (allow_partial_marker_accept and pass_fraction >= partial_accept_min_fraction):
                augmented_image = {
                    'image': aug,
                    'passed_tags': passed_tags,
                    'failed_tags': failed_tags,
                    'pass_fraction': pass_fraction,
                    'all_ok': all_ok,
                }
                break
            # else: try again up to max_attempts_combined
        if augmented_image is None:
            return None
        return augmented_image 
    
    def save_train_val_data(self, 
                            save_rgb=True, 
                            save_seg=True, 
                            save_keypoints=True, 
                            save_metadata=True, 
                            num_augmentations=1,
                            save_summary_image=False,
                            save_roi=False, 
                            max_tries_per_aug=10,
                            filter_base_images=False,
                            allow_partial_marker_accept=False,
                            partial_accept_min_fraction=0.80,
                            segmentation_background_color=(0, 0, 0),
                            ):
        
        # Create directories for both training and validation datasets
        self.train_dir = os.path.join(self.out_dir, "train")
        self.val_dir = os.path.join(self.out_dir, "val")
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.val_dir, exist_ok=True)

        # Loop over train and val
        for dataset_type in ['train', 'val']:
            dataset_dir = self.train_dir if dataset_type == 'train' else self.val_dir
            datapoints = self.datapoints_train if dataset_type == 'train' else self.datapoints_val

            # Create specific directories for RGB, Segmentation, Keypoints, Metadata, and Summary Images
            if save_rgb:
                os.makedirs(os.path.join(dataset_dir, "images"), exist_ok=True)
            if save_seg:
                os.makedirs(os.path.join(dataset_dir, "segmentations"), exist_ok=True)
            if save_keypoints:
                os.makedirs(os.path.join(dataset_dir, "keypoints"), exist_ok=True)
            if save_metadata:
                os.makedirs(os.path.join(dataset_dir, "metadata"), exist_ok=True)
            if save_summary_image:
                os.makedirs(os.path.join(dataset_dir, "summary_images"), exist_ok=True)
            if save_roi:
                os.makedirs(os.path.join(dataset_dir, "roi_rgb"), exist_ok=True)
                os.makedirs(os.path.join(dataset_dir, "roi_keypoints"), exist_ok=True) 

            # Load resume progress
            done_ids = set()
            if getattr(self, 'resume', False):
                st = self._load_resume_state(dataset_type, dataset_dir)
                done_ids = set(st.get('done_ids', []))
                # mark as active/incomplete at start
                self._save_resume_state(dataset_type, dataset_dir, done_ids, complete=False)

            # Process each datapoint in the current dataset
            print(f"[phase] Saving {dataset_type} data (N={len(datapoints)})")
            for i, dp in enumerate(tqdm(datapoints, desc=f"{dataset_type} save", unit="img")):
                dp_uid = self._datapoint_uid(dp)
                if getattr(self, 'resume', False) and dp_uid in done_ids:
                    continue
                try:
                    # Internal segmentation for checks/ROI only (may be synthesized)
                    seg_img = dp.preprocess_seg_img()
                    # Prefer copying the original segmentation file as-is when saving
                    seg_src_path = dp.seg_png_filepath if (dp.seg_png_filepath and os.path.isfile(dp.seg_png_filepath)) else None
                    if seg_src_path is None:
                        # Try metadata outputs path
                        try:
                            seg_rel = (dp.outputs or {}).get("segmentation")
                            if seg_rel and dp.dataset_root:
                                cand = os.path.join(dp.dataset_root, seg_rel)
                                if os.path.isfile(cand):
                                    seg_src_path = cand
                        except Exception:
                            pass
                    # Preload segmentation for checks (avoid repeated IO in retries)
                    seg_for_checks = None
                    if seg_src_path is not None and os.path.isfile(seg_src_path):
                        try:
                            with Image.open(seg_src_path) as _im_seg:
                                seg_for_checks = np.array(_im_seg.convert('RGB'))
                        except Exception:
                            seg_for_checks = np.array(seg_img)
                    else:
                        seg_for_checks = np.array(seg_img)
                    # Load RGB once for this datapoint (used by saving and summaries)
                    rgb_img = Image.open(dp.rgb_filepath)
                    augmented_img = None
                    if save_rgb:
                        if num_augmentations == 0:
                            # Optionally apply filter on the base image before saving
                            if (not filter_base_images) or self.check_image_okay(np.array(rgb_img), seg_img):
                                rgb_img.save(os.path.join(dataset_dir, "images", f"img_{i}.png"))
                                if save_seg:
                                    seg_out = os.path.join(dataset_dir, "segmentations", f"seg_{i}.png")
                                    if seg_src_path is not None:
                                        shutil.copyfile(seg_src_path, seg_out)
                                    else:
                                        # Fallback: save synthesized binary mask
                                        Image.fromarray(np.array(seg_img)).save(seg_out)
                            else:
                                # base image rejected by filter; skip saving
                                pass
                        else:
                            # For each requested augmentation, try up to max_tries_per_aug times until one passes the filter
                            accepted = 0
                            for j in range(num_augmentations):
                                aug_result = self.augment_image(
                                    rgb_img, dp, seg=seg_for_checks,
                                    max_attempts_lighting=5,
                                    max_attempts_combined=max_tries_per_aug,
                                    min_tag_area=1000,
                                    min_tag_pix_mean=25,
                                    max_tag_pix_mean=250,
                                    min_tag_pix_contrast=10,
                                    allow_partial_marker_accept=allow_partial_marker_accept,
                                    partial_accept_min_fraction=partial_accept_min_fraction,
                                )
                                if aug_result is None:
                                    # rejected after retries; skip this augmentation index
                                    continue
                                aug_np = aug_result['image'] if isinstance(aug_result, dict) else aug_result
                                passed_tags = aug_result.get('passed_tags', getattr(dp, 'active_markers', [])) if isinstance(aug_result, dict) else getattr(dp, 'active_markers', [])
                                failed_tags = aug_result.get('failed_tags', []) if isinstance(aug_result, dict) else []
                                augmented_img = Image.fromarray(aug_np.astype(np.uint8))
                                img_out_path = os.path.join(dataset_dir, "images", f"img_{i}_{accepted}.png")
                                augmented_img.save(img_out_path)
                                # Save matching segmentation for each augmentation by copying original if available
                                if save_seg:
                                    seg_out = os.path.join(dataset_dir, "segmentations", f"seg_{i}_{accepted}.png")
                                    # Prepare a segmentation image, potentially removing failed tags
                                    seg_modified = None
                                    # Try to start from original colored segmentation if available
                                    base_seg_arr = None
                                    if seg_src_path is not None and os.path.isfile(seg_src_path):
                                        try:
                                            im = Image.open(seg_src_path)
                                            # Use RGB for color comparisons; keep alpha if present when saving
                                            base_seg_arr = np.array(im.convert('RGB'))
                                            seg_mode = 'RGB'
                                        except Exception:
                                            base_seg_arr = None
                                    if base_seg_arr is None:
                                        # Fallback: use synthesized combined mask
                                        base_seg_arr = np.array(dp.preprocess_seg_img())
                                        seg_mode = 'L'

                                    # Remove failed tags
                                    if len(failed_tags) == 0:
                                        # No changes; copy original if available to keep file unchanged
                                        if seg_src_path is not None and os.path.isfile(seg_src_path):
                                            try:
                                                shutil.copyfile(seg_src_path, seg_out)
                                                seg_modified = None  # mark as already saved
                                            except Exception:
                                                seg_modified = base_seg_arr
                                        else:
                                            seg_modified = base_seg_arr
                                    else:
                                        if base_seg_arr.ndim == 3 and base_seg_arr.shape[2] == 3:
                                            # Color-coded segmentation; zero out pixels of failed tag colors
                                            # Build color map (BGR -> RGB)
                                            color_map_rgb = {}
                                            for tag_name in getattr(dp, 'active_markers', []):
                                                info = (getattr(dp, 'marker_info', {}) or {}).get(tag_name)
                                                if info and info.get('seg_color_bgr') is not None:
                                                    bgr = info['seg_color_bgr']
                                                    if isinstance(bgr, (list, tuple)) and len(bgr) >= 3:
                                                        color_map_rgb[tag_name] = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
                                            seg_modified = base_seg_arr.copy()
                                            bg = np.array(segmentation_background_color, dtype=np.uint8)
                                            for t in failed_tags:
                                                if t in color_map_rgb:
                                                    color = np.array(color_map_rgb[t], dtype=np.uint8)
                                                    mask = np.all(seg_modified == color, axis=-1)
                                                    seg_modified[mask] = bg
                                        else:
                                            # Binary/grayscale segmentation: subtract each failed tag's quad mask
                                            seg_modified = base_seg_arr.copy()
                                            if seg_modified.ndim == 3:
                                                # convert to single channel if needed
                                                seg_modified = cv2.cvtColor(seg_modified, cv2.COLOR_RGB2GRAY)
                                            for t in failed_tags:
                                                try:
                                                    mask_img = dp.preprocess_seg_img(tag_name=t)
                                                    mask = np.array(mask_img)
                                                    seg_modified[mask == 255] = 0
                                                except Exception:
                                                    pass

                                    # Save modified segmentation (if not already copied)
                                    if seg_modified is not None:
                                        try:
                                            Image.fromarray(seg_modified.astype(np.uint8)).save(seg_out)
                                        except Exception:
                                            # Fallback to copying original if modification fails
                                            if seg_src_path is not None:
                                                shutil.copyfile(seg_src_path, seg_out)
                                            else:
                                                Image.fromarray(np.array(seg_img)).save(seg_out)
                                # Save filtered metadata per accepted augmentation, removing failed tags
                                if save_metadata:
                                    try:
                                        meta = copy.deepcopy(dp.metadata) if isinstance(dp.metadata, dict) else {}
                                        markers_list = meta.get('markers', []) if isinstance(meta, dict) else []
                                        # Build keep set from passed tag names
                                        keep_ids = set()
                                        keep_colors = set()
                                        for t in passed_tags:
                                            info = (getattr(dp, 'marker_info', {}) or {}).get(t)
                                            if info is None:
                                                continue
                                            mid = info.get('marker_id')
                                            if mid is not None:
                                                keep_ids.add(mid)
                                            col = info.get('seg_color_bgr')
                                            if isinstance(col, (list, tuple)):
                                                keep_colors.add(tuple(col))
                                        def marker_keep(m):
                                            try:
                                                mid = m.get('marker_id')
                                                col = tuple(m.get('seg_color_bgr')) if isinstance(m.get('seg_color_bgr'), list) else m.get('seg_color_bgr')
                                                return (mid in keep_ids) or (col in keep_colors)
                                            except Exception:
                                                return False
                                        filtered_markers = [m for m in markers_list if marker_keep(m)]
                                        if isinstance(meta, dict):
                                            meta['markers'] = filtered_markers
                                        meta_out_path = os.path.join(dataset_dir, 'metadata', f"metadata_{i}_{accepted}.json")
                                        with open(meta_out_path, 'w') as f:
                                            json.dump(meta, f)
                                    except Exception:
                                        # Fallback: write original metadata with suffix
                                        meta_out_path = os.path.join(dataset_dir, 'metadata', f"metadata_{i}_{accepted}.json")
                                        with open(meta_out_path, 'w') as f:
                                            json.dump(dp.metadata, f)
                                accepted += 1
                    # If not saving RGB, but still want seg (rare), copy base seg unchanged if available
                    if save_seg and num_augmentations == 0 and not save_rgb:
                        seg_out = os.path.join(dataset_dir, "segmentations", f"seg_{i}.png")
                        if seg_src_path is not None:
                            shutil.copyfile(seg_src_path, seg_out)
                        else:
                            Image.fromarray(np.array(seg_img)).save(seg_out)

                    if save_keypoints:
                        keypoints = dp.compute_keypoints(self.keypoints_tag_frame, self.camera_matrix)
                        # Save as nested dict: { tag: [ [u,v], ... ] }
                        keypoints_json = {tag: [kp.tolist() for kp in uv_list] for tag, uv_list in keypoints.items()}
                        with open(os.path.join(dataset_dir, "keypoints", f"keypoints_{i}.json"), 'w') as f:
                            json.dump(keypoints_json, f)

                    if save_metadata:
                        # Base (no augmentation): original metadata
                        if num_augmentations == 0:
                            metadata = dp.metadata
                            with open(os.path.join(dataset_dir, "metadata", f"metadata_{i}.json"), 'w') as f:
                                json.dump(metadata, f)
                        else:
                            # For augmented outputs, write one metadata file per accepted augmentation index
                            # We need to mirror the accepted count; re-derive filenames present in images dir
                            # Simpler: regenerate metadata alongside image/seg save inside the augmentation loop
                            pass

                    if save_roi:
                        # Save one combined ROI based on combined mask, plus optional per-marker ROIs
                        roi_image, roi_coordinates, roi_center = dp.get_roi_image(seg=seg_img)
                        roi_image_pil = Image.fromarray(roi_image)
                        roi_image_pil.save(os.path.join(dataset_dir, "roi_rgb", f"roi_{i}.png"))
                        roi_keypoints = dp.get_roi_keypoints()
                        if roi_keypoints is not None:
                            roi_keypoints_json = {f"keypoints_{k}": kp.tolist() for k, kp in enumerate(roi_keypoints)}
                            with open(os.path.join(dataset_dir, "roi_keypoints", f"roi_keypoints_{i}.json"), 'w') as f:
                                json.dump(roi_keypoints_json, f)
                        # Additionally, per-marker ROIs if segmentation allows
                        for tag_name in getattr(dp, "active_markers", []):
                            try:
                                tag_seg = dp.preprocess_seg_img(tag_name=tag_name)
                                tag_roi_img, _, _ = dp.get_roi_image(seg=tag_seg, tag_name=tag_name)
                                Image.fromarray(tag_roi_img).save(os.path.join(dataset_dir, "roi_rgb", f"roi_{i}_{tag_name}.png"))
                                tag_roi_kps = dp.get_roi_keypoints(tag_name=tag_name)
                                if tag_roi_kps is not None:
                                    roi_kp_json = {f"keypoints_{k}": kp.tolist() for k, kp in enumerate(tag_roi_kps)}
                                    with open(os.path.join(dataset_dir, "roi_keypoints", f"roi_keypoints_{i}_{tag_name}.json"), 'w') as f:
                                        json.dump(roi_kp_json, f)
                            except Exception:
                                # If per-tag color not available, skip silently
                                pass

                    if save_summary_image:
                        # Check if images are loaded correctly
                        if rgb_img is None:
                            raise ValueError(f"RGB image at {dp.rgb_filepath} could not be loaded.")
                        if seg_img is None:
                            raise ValueError(f"Segmentation image at {dp.seg_png_filepath} could not be loaded.")

                        # Convert from BGR (OpenCV default) to RGB (for matplotlib)
                        image_rgb = np.array(rgb_img)
                        if augmented_img is None:
                            augmented_img_rgb = image_rgb
                        else:
                            augmented_img_rgb = np.array(augmented_img)

                        # Create a new figure for each image
                        plt.figure(figsize=(12, 8))  # Adjust figure size to make space for metadata and the new ROI subplot

                        # Subplot for original RGB image
                        plt.subplot(2, 3, 1)  # 2 rows, 3 columns, 1st subplot
                        plt.imshow(image_rgb)
                        plt.axis('off')  # Hide axes
                        plt.title(f'Original Image {i}')

                        # Subplot for augmented RGB image
                        plt.subplot(2, 3, 2)  # 2 rows, 3 columns, 2nd subplot
                        plt.imshow(augmented_img_rgb)
                        plt.axis('off')  # Hide axes
                        plt.title(f'Augmented Image {i}')

                        # Subplot for segmentation image
                        plt.subplot(2, 3, 3)  # 2 rows, 3 columns, 3rd subplot
                        plt.imshow(seg_img, cmap='viridis')  # Use a colormap for better visualization
                        plt.axis('off')  # Hide axes
                        plt.title(f'Segmentation Image {i}')

                        # Subplot for RGB image - keypoints
                        # Flatten keypoints across markers for visualization
                        flat_kps = []
                        if isinstance(keypoints, dict):
                            for lst in keypoints.values():
                                flat_kps.extend(lst)
                        else:
                            flat_kps = keypoints
                        keypoints_image = overlay_points_on_image(image=np.array(augmented_img_rgb), pixel_points=flat_kps, radius=1)
                        plt.subplot(2, 3, 4)  # 2 rows, 3 columns, 4th subplot
                        plt.imshow(keypoints_image)
                        plt.axis('off')  # Hide axes
                        plt.title(f'Keypoints Image {i}')

                        if True: #save_roi:

                            # Ensure ROI computed locally for summary
                            roi_image, _, _ = dp.get_roi_image(seg=seg_img)
                            roi_keypoints = dp.get_roi_keypoints()

                            # Subplot for ROI image
                            plt.subplot(2, 3, 5)  # 2 rows, 3 columns, 5th subplot
                            plt.imshow(roi_image)
                            plt.axis('off')  # Hide axes
                            plt.title(f'ROI Image {i}')

                            # Subplot for ROI image with keypoints 
                            roi_keypoints_image = overlay_points_on_image(image=np.array(roi_image), pixel_points=roi_keypoints, radius=1)
                            plt.subplot(2, 3, 6)  # 2 rows, 3 columns, 6th subplot
                            plt.imshow(roi_keypoints_image)
                            plt.axis('off')  # Hide axes
                            plt.title(f'ROI Keypoints Image {i}')
                        
                        # Display metadata as text in a separate area
                        metadata_str = dp.__repr__()

                        # Create a new subplot for metadata
                        plt.text(1.05, 0.5, metadata_str, fontsize=12, ha='left', va='center', transform=plt.gca().transAxes,
                                bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=1'))

                        # Adjust layout to avoid overlap and make space for metadata
                        plt.tight_layout()  # Adjust layout
                        plt.subplots_adjust(right=0.8)  # Make space for metadata on the right

                        # Save the image to the summary_images folder
                        save_path = os.path.join(dataset_dir, "summary_images", f"summary_image_{i}.png")
                        plt.savefig(save_path, bbox_inches='tight', dpi=300)  # Save with high resolution
                        plt.close()  # Close the plot to free up memory

                    # Mark datapoint as done and persist periodically
                    if getattr(self, 'resume', False):
                        done_ids.add(dp_uid)
                        if (i % max(1, len(datapoints)//20 + 1)) == 0:
                            self._save_resume_state(dataset_type, dataset_dir, done_ids, complete=False)
                except Exception as e:
                    print(f"[warn] Skipping datapoint due to error ({dataset_type} idx={i}, uid={dp_uid}): {e}")
                    continue


            print(f"[phase] Completed saving {dataset_type} data.") 
            if getattr(self, 'resume', False):
                self._save_resume_state(dataset_type, dataset_dir, done_ids, complete=True)

if __name__ == "__main__":
    print("[phase] Initialization")
    DATA_ROOT = "/home/nom4d/deep-marker-estimation/data_generation/multi_marker_output/"
    # DATA_ROOT = "/home/nom4d/deep-marker-estimation/data_generation/test/"

    def is_dataset_dir(path: str) -> bool:
        return (
            os.path.isdir(os.path.join(path, "metadata"))
            and (
                os.path.isdir(os.path.join(path, "images"))
                or os.path.isdir(os.path.join(path, "rgb"))
            )
        )

    def pick_latest_dataset_dir(base: str) -> str | None:
        if not os.path.isdir(base):
            print(f"[warn] DATA_ROOT does not exist: {base}")
            return None
        subdirs = [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
        candidates = [d for d in sorted(subdirs) if is_dataset_dir(d)]
        return candidates[-1] if candidates else None

    def dataset_overview(dataset_dir: str):
        print(f"[info] Dataset directory: {dataset_dir}")
        images_dir = os.path.join(dataset_dir, "images")
        rgb_dir = os.path.join(dataset_dir, "rgb")
        seg_dir = os.path.join(dataset_dir, "segmentations")
        meta_dir = os.path.join(dataset_dir, "metadata")
        def count_files(p, exts=(".png", ".jpg", ".jpeg")):
            if not os.path.isdir(p):
                return 0
            return sum(1 for f in os.listdir(p) if f.lower().endswith(exts))
        print(f"[info] images: {count_files(images_dir)} | rgb: {count_files(rgb_dir)} | seg: {count_files(seg_dir, ('.png',))} | metadata: {count_files(meta_dir, ('.json',))}")
        try:
            metas = [f for f in sorted(os.listdir(meta_dir)) if f.endswith('.json')]
            if metas:
                with open(os.path.join(meta_dir, metas[0]), 'r') as f:
                    md = json.load(f)
                keys = list(md.keys())
                mtype = type(md.get('markers')).__name__
                print(f"[info] metadata keys: {keys}")
                print(f"[info] markers field type: {mtype}")
        except Exception as e:
            print(f"[warn] Could not inspect metadata: {e}")

    dataset_dir = pick_latest_dataset_dir(DATA_ROOT)
    if not dataset_dir:
        print("[error] No valid dataset found under DATA_ROOT. Exiting.")
        raise SystemExit(1)

    dataset_overview(dataset_dir)
    data_folders = [dataset_dir]

    print("[phase] Preparing output directory")
    out_root = os.path.join("/home/nom4d/deep-marker-estimation/", "data_generation", "multi_marker_augmented_output")
    os.makedirs(out_root, exist_ok=True)
    OUT_DIR = os.path.join(out_root, f"multi_marker_augmented_{time.strftime('%Y%m%d-%H%M%S')}")
    # OUT_DIR = "/home/nom4d/deep-marker-estimation/data_generation/multi_marker_augmented_output/multi_marker_augmented_20251016-211723/"
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[info] Output directory: {OUT_DIR}")

    print("[phase] Initializing processor and camera/marker configs")
    processor = DataProcessor(data_folders, OUT_DIR, max_datapoints_total=50_000)
    processor.set_marker(image_path=None, num_squares=8, side_length=0.100)

    print("[phase] Scanning and pairing files")
    processor.process_folders()

    print(f"[info] Total datapoints discovered: {len(processor.datapoints)}")
    if processor.datapoints:
        # Skip pre-filtering; we'll filter post-augmentation per sample
        print("[phase] Splitting train/val")
        processor.split_train_val_data(filter=False, frac_train=0.8)
        print("[phase] Augmentation and saving (with post-augmentation filtering)")
        processor.save_train_val_data(num_augmentations=1, save_summary_image=False, max_tries_per_aug=10, filter_base_images=False)
        print("[phase] Done")