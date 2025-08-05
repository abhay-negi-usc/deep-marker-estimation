import numpy as np 
import torch
import cv2
import numpy as np
from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
from PIL import Image
from scipy.spatial.transform import Rotation as R

from segmentation_model.model import UNETWithDropout
from segmentation_model.utils import load_checkpoint as load_seg_ckpt
from keypoints_model.model import RegressorMobileNetV3
from keypoints_model.utils import load_checkpoint as load_kp_ckpt
# from keypoints_model.utils import xyzabc_to_tf, rvectvec_to_xyzabc
from albumentations.pytorch import ToTensorV2
import albumentations as A

def rvectvec_to_xyzabc(rvec, tvec): 
    rot = cv2.Rodrigues(rvec)[0] 
    tvec = tvec.reshape(3)
    xyzabc = np.concatenate((tvec, R.from_matrix(rot).as_euler("xyz",degrees=True))) 
    return xyzabc 

def xyzabc_to_tf(xyzabc): 
    tvec = xyzabc[:3] 
    rot = R.from_euler("xyz",xyzabc[3:],degrees=True).as_matrix()
    tf = np.eye(4) 
    tf[:3,:3] = rot 
    tf[:3,3] = tvec 
    return tf

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

def MarkerPoseEstimator(
    image: np.ndarray,
    seg_model,
    kp_model,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    keypoints_tag_frame: np.ndarray,
    tf_W_Ccv: np.ndarray,
):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    seg_size = (image.shape[1], image.shape[0])  # width, height
    resized_rgb = cv2.resize(image, seg_size)

    seg_transform = Compose([Normalize(max_pixel_value=1.0), ToTensorV2()])
    img_tensor = seg_transform(image=resized_rgb)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        seg_mask = torch.sigmoid(seg_model(img_tensor))
        seg_mask = (seg_mask > 0.1).float().cpu()
        seg_mask_img = Image.fromarray(seg_mask.squeeze().numpy().astype(np.uint8) * 255)

    def compute_roi(seg, rgb):
        seg = np.array(seg)
        padding = 5
        roi_size = 128
        image_border_size = max(seg.shape)

        seg = cv2.copyMakeBorder(seg, image_border_size, image_border_size, image_border_size, image_border_size, cv2.BORDER_CONSTANT, value=0)
        rgb = cv2.copyMakeBorder(rgb, image_border_size, image_border_size, image_border_size, image_border_size, cv2.BORDER_CONSTANT, value=0)

        tag_pixels = np.argwhere(seg == 255)
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

    roi_img, roi_coords = compute_roi(seg_mask_img, resized_rgb)
    if roi_img is None:
        return None, seg_mask_img, Image.fromarray(image)

    transform = Compose([ToTensorV2()])
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

    # Optional: remap to original image resolution (if resized_rgb differs)
    H_orig, W_orig = image.shape[:2]
    H_resized, W_resized = resized_rgb.shape[:2]
    scale_x_back = W_orig / W_resized
    scale_y_back = H_orig / H_resized

    keypoints_orig = np.stack([
        keypoints_img[:, 0] * scale_x_back,
        keypoints_img[:, 1] * scale_y_back
    ], axis=1)

    success, rvec, tvec = cv2.solvePnP(
        objectPoints=keypoints_tag_frame,
        imagePoints=keypoints_orig,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs
    )
    if not success:
        return None, seg_mask_img, Image.fromarray(image)

    pose = rvectvec_to_xyzabc(rvec, tvec)
    tf_marker = tf_W_Ccv @ xyzabc_to_tf(pose)

    overlay = overlay_points_on_image(image.copy(), keypoints_orig, radius=3)
    overlay_pil = Image.fromarray(overlay)

    return tf_marker, seg_mask_img, overlay_pil

def compute_2D_gridpoints(N=10,s=0.1): 
    # N = num squares, s = side length  
    u = np.linspace(-s/2, +s/2, N+1) 
    v = np.linspace(-s/2, +s/2, N+1) 
    gridpoints = [] 
    for uu in u:
        for vv in v: 
            gridpoints.append(np.array([uu,vv,0])) 
    return gridpoints 

def build_lbcv_predictor(
    seg_model_path: str,
    kp_model_path: str,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_side_length: float,
    keypoints_tag_frame: np.ndarray,
):

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    seg_model = UNETWithDropout(in_channels=3, out_channels=1).to(DEVICE)
    load_seg_ckpt(torch.load(seg_model_path, map_location=DEVICE), seg_model)
    seg_model.eval()

    torch.cuda.empty_cache()
    kp_model = RegressorMobileNetV3().to(DEVICE)
    load_kp_ckpt(torch.load(kp_model_path, map_location=DEVICE), kp_model)
    kp_model.eval()

    # tf_W_Ccv = np.array([
    #     [1, 0, 0, 0],
    #     [0, -1, 0, 0],
    #     [0, 0, -1, 0],
    #     [0, 0, 0, 1]
    # ])
    tf_W_Ccv = np.array([
        [-1,0,0,0],
        [0,-1,0,0],
        [0,0,1,0],
        [0,0,0,1],
    ]) # FIXME: don't know why the original tf_W_Ccv is not working 
    # tf_W_Ccv = np.eye(4)

    def compute_roi(seg, rgb):

        padding = 5
        roi_size = 128
        image_border_size = np.max([np.array(seg).shape[0], np.array(seg).shape[1]])

        seg = np.array(seg)
        seg = cv2.copyMakeBorder(seg, image_border_size, image_border_size, image_border_size, image_border_size, cv2.BORDER_CONSTANT, value=0)
        tag_pixels = np.argwhere(seg == 255)
        if tag_pixels.size == 0:
            return None, None

        seg_tag_min_x = np.min(tag_pixels[:, 1])
        seg_tag_max_x = np.max(tag_pixels[:, 1])
        seg_tag_min_y = np.min(tag_pixels[:, 0])
        seg_tag_max_y = np.max(tag_pixels[:, 0])
        seg_height = seg_tag_max_y - seg_tag_min_y
        seg_width = seg_tag_max_x - seg_tag_min_x
        seg_center_x = (seg_tag_min_x + seg_tag_max_x) // 2
        seg_center_y = (seg_tag_min_y + seg_tag_max_y) // 2

        if isinstance(rgb, str):
            rgb = np.array(cv2.imread(rgb))
        if isinstance(rgb, Image.Image):
            rgb = np.array(rgb)
        if isinstance(rgb, np.ndarray):
            rgb = rgb
        rgb = cv2.copyMakeBorder(rgb, image_border_size, image_border_size, image_border_size, image_border_size, cv2.BORDER_CONSTANT, value=0)

        rgb_side = max(seg_height, seg_width) + 2 * padding
        rgb_tag_min_x = seg_center_x - rgb_side // 2
        rgb_tag_max_x = seg_center_x + rgb_side // 2
        rgb_tag_min_y = seg_center_y - rgb_side // 2
        rgb_tag_max_y = seg_center_y + rgb_side // 2
        roi_img = rgb[rgb_tag_min_y:rgb_tag_max_y, rgb_tag_min_x:rgb_tag_max_x, :]
        roi_img = cv2.resize(roi_img, (roi_size, roi_size))
        roi_coordinates = np.array([rgb_tag_min_x, rgb_tag_max_x, rgb_tag_min_y, rgb_tag_max_y]) - image_border_size 

        return roi_img, roi_coordinates

    def predict_pose_from_image(image: np.ndarray):
        from keypoints_model.utils import xyzabc_to_tf, rvectvec_to_xyzabc

        tf_marker = None
        # seg_size = (640, 480)
        seg_size = (image.shape[0], image.shape[1])  # Use original image size for segmentation

        # Resize RGB image to match segmentation input
        resized_rgb = cv2.resize(image, seg_size)  # shape (H, W, 3)

        # Segmentation transform: normalized for model
        seg_transform = A.Compose([
            A.Normalize(max_pixel_value=1.0),
            ToTensorV2()
        ])
        transformed = seg_transform(image=resized_rgb)
        img_tensor = transformed["image"].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            seg_mask = torch.sigmoid(seg_model(img_tensor))
            seg_mask = (seg_mask > 0.5).float().cpu()

        seg_mask_img = transforms.ToPILImage()(seg_mask.squeeze(0))  # shape matches resized_rgb

        seg_transform_full = A.Compose([
            A.Normalize(max_pixel_value=1.0),
            ToTensorV2()
        ])
        transformed_full = seg_transform_full(image=image)
        img_tensor_full = transformed_full["image"].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            seg_mask_full = torch.sigmoid(seg_model(img_tensor_full))
            seg_mask_full = (seg_mask_full > 0.5).float().cpu()

        seg_mask_img_full = transforms.ToPILImage()(seg_mask_full.squeeze(0))  # shape matches resized_rgb

        # No tag detected
        if np.array(seg_mask_img).max() == 0:
            return None, None, seg_mask_img, seg_mask_img_full

        # --- Compute ROI using resized RGB and seg ---
        roi_img, roi_coords = compute_roi(seg_mask_img, resized_rgb)
        if roi_img is None:
            return None, None, seg_mask_img, seg_mask_img_full

        # Keypoint transform (no resize)
        kp_transform = A.Compose([ToTensorV2()])
        roi_tensor = kp_transform(image=roi_img)["image"].unsqueeze(0).float().to(DEVICE)

        with torch.no_grad():
            keypoints_roi = kp_model(roi_tensor).cpu().numpy().reshape(-1, 2)

        # Convert from ROI (128×128) to image coordinates
        s = np.array(roi_img.shape[:2])  # (H, W)
        img_roi_center_x = (roi_coords[0] + roi_coords[1]) / 2
        img_roi_center_y = (roi_coords[2] + roi_coords[3]) / 2
        roi_center = np.array([img_roi_center_x, img_roi_center_y])
        w = roi_coords[1] - roi_coords[0]
        h = roi_coords[3] - roi_coords[2]
        m = s / np.array([w, h])  # per-axis scale

        keypoints_img = (keypoints_roi - s / 2) / m + roi_center

        success, rvec, tvec = cv2.solvePnP(
            objectPoints=keypoints_tag_frame,
            imagePoints=keypoints_img,
            cameraMatrix=camera_matrix,
            distCoeffs=dist_coeffs,
        )
        if not success:
            return None, None, seg_mask_img, seg_mask_img_full 
        
        pose_marker = rvectvec_to_xyzabc(rvec, tvec)
        tf_marker = tf_W_Ccv @ xyzabc_to_tf(pose_marker)

        return tf_marker, keypoints_img, seg_mask_img, seg_mask_img_full 

    return predict_pose_from_image, seg_model, kp_model
