import cv2
import numpy as np
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional, Tuple
from utils import MarkerPoseEstimator, compute_2D_gridpoints, build_lbcv_predictor

def detect_pose_opencv_marker(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    aruco_dict_type: int,
    marker_length: float
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

    vis_image = image.copy()
    corners, ids, _ = detector.detectMarkers(image)

    if ids is not None and len(ids) > 0:
        # Draw bold borders
        for corner in corners:
            pts = corner[0].astype(int)
            for i in range(4):
                pt1 = tuple(pts[i])
                pt2 = tuple(pts[(i + 1) % 4])
                cv2.line(vis_image, pt1, pt2, color=(0, 255, 0), thickness=5)

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

        if rvecs is not None and tvecs is not None:
            tf_c_m = np.eye(4)
            tf_c_m[:3, :3] = cv2.Rodrigues(rvecs[0])[0]
            tf_c_m[:3, 3] = tvecs[0].reshape(3)
            return tf_c_m, vis_image

    return None, vis_image

def run_learning_based_marker_estimation(
    image_path: str,
    seg_model_path: str,
    kp_model_path: str,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_length: float,
    num_squares: int = 10,
    tf_W_Ccv: Optional[np.ndarray] = None
) -> Tuple[Optional[np.ndarray], Optional[Image.Image], Optional[Image.Image]]:
    """
    Runs learning-based marker pose estimation and returns the pose, segmentation mask,
    and overlay image with predicted keypoints.

    Returns:
        - 4x4 pose matrix or None
        - Segmentation mask as PIL.Image or None
        - Keypoint overlay image as PIL.Image or None
    """
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    keypoints_tag_frame = np.array(
        compute_2D_gridpoints(N=num_squares, s=marker_length)
    )

    # Build predictor
    predict_fn, seg_model, kp_model = build_lbcv_predictor(
        seg_model_path=seg_model_path,
        kp_model_path=kp_model_path,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        marker_side_length=marker_length,
        keypoints_tag_frame=keypoints_tag_frame,
    )

    # Set default tf_W_Ccv if not provided
    if tf_W_Ccv is None:
        tf_W_Ccv = np.eye(4)

    # Call the functional estimator
    pose, seg_mask_img, overlay_img = MarkerPoseEstimator(
        image=image_rgb,
        seg_model=seg_model,
        kp_model=kp_model,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        keypoints_tag_frame=keypoints_tag_frame,
        tf_W_Ccv=tf_W_Ccv,
    )

    return pose, seg_mask_img, overlay_img

# === Parameters ===
fx, fy, cx, cy = 1363.85, 1365.40, 958.58, 552.25
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
dist_coeffs = np.array([0.1, -0.2, 0.0, 0.0, 0.0])
aruco_dict_type = cv2.aruco.DICT_APRILTAG_36h11
marker_length = 0.0798 
# tf_W_Ccv = np.array([
#     [-1, 0, 0, 0],
#     [0, -1, 0, 0],
#     [0,  0, 1, 0],
#     [0,  0, 0, 1]
# ])
tf_W_Ccv = np.eye(4) 

# === Load and process image ===
# image_path = "./example2_small.png"
image_path = "./aruco6x6/IMG_6287.jpg"
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # For matplotlib

# --- OpenCV Detection ---
pose_opencv, vis_img_bgr = detect_pose_opencv_marker(
    image_bgr, camera_matrix, dist_coeffs, aruco_dict_type, marker_length
)
vis_img_rgb = cv2.cvtColor(vis_img_bgr, cv2.COLOR_BGR2RGB)

# --- Learning-Based Detection ---
pose_lbcv, seg_mask_img, overlay_img = run_learning_based_marker_estimation(
    image_path=image_path,
    seg_model_path="/home/nom4d/marker_ws/segmentation_checkpoints/my_checkpoint_multimarker_epoch_0_batch_10000.pth.tar",
    kp_model_path= "./keypoints_model/my_checkpoint_keypoints_20250330.pth.tar", 
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    marker_length=marker_length * 10/8,
    num_squares=10,
    tf_W_Ccv=tf_W_Ccv
)

# Convert PIL images to np.array for plotting
seg_mask_np = np.array(seg_mask_img) if seg_mask_img is not None else np.zeros(image_rgb.shape[:2], dtype=np.uint8)
overlay_np = np.array(overlay_img) if overlay_img is not None else image_rgb

# === Show side-by-side ===
fig, axs = plt.subplots(1, 4, figsize=(20, 6))
axs[0].imshow(image_rgb)
axs[0].set_title("Original Image")
axs[0].axis("off")

axs[1].imshow(vis_img_rgb)
axs[1].set_title("OpenCV Detection")
axs[1].axis("off")

axs[2].imshow(seg_mask_np, cmap='gray')
axs[2].set_title("Segmentation Mask")
axs[2].axis("off")

axs[3].imshow(overlay_np)
axs[3].set_title("Keypoint Overlay")
axs[3].axis("off")

plt.tight_layout()
plt.show()

# === Print pose if detected ===
if pose_lbcv is not None:
    print("LBCV Estimated Pose:\n", pose_lbcv)
else:
    print("No marker detected (LBCV).")

if pose_opencv is not None:
    print("OpenCV Estimated Pose:\n", pose_opencv)
else:
    print("No marker detected (OpenCV).")