# imports 
import numpy as np
from matplotlib import pyplot as plt
import cv2 
import os 
from PIL import Image

# utility functions 
from utils.image_utils import *
from utils.seg_utils import * 
from utils.keypoint_utils import * 
from utils.pose_estimation_utils import * 
from utils.homography_utils import *
from utils.pattern_based_estimation_utils import *

# configurations 
# configure marker properties 
img_marker_path = "./utils/tag36h11_0.png"
img_marker = cv2.imread(img_marker_path)
keypoints_marker_image_space = find_keypoints(img_marker)
keypoints_marker_cartesian_space = convert_marker_keypoints_to_cartesian(
    keypoints_marker_image_space, image_size=(img_marker.shape[0], img_marker.shape[1]), marker_size=(0.1, 0.1)
)

config_marker = {
    "dictionary": 'AprilTag 36h11', 
    "id": 0, 
    "marker_length_with_border": 0.1, # units: meters 
    "marker_length_without_border": 0.08, # units: meters
    "num_squares": 10, # number of squares in the grid
    "keypoints_marker_image_space": keypoints_marker_image_space,
    "keypoints_marker_cartesian_space": keypoints_marker_cartesian_space,
    "marker_image": img_marker,
    "marker_image_path": img_marker_path,
}
# configure camera properties 
config_camera = {
    "camera_matrix": np.array([[906.995, 0, 638.235], [0, 906.995, 360.533], [0, 0, 1]]), # example camera matrix
    "dist_coeffs": np.array([0,0,0,0,0], dtype=float), # example distortion coefficients
}
# configure segmentation model  
config_segmentation = {
    # "checkpoint_path": "./segmentation_model/segmentation_checkpoint_20250329.pth.tar",
    "checkpoint_path": "./segmentation_model/my_checkpoint_minimodel_epoch_47_batch_0.pth.tar",
    "input_size": (480, 640), 
    "segmentation_threshold": 0.5,  
}
# configure keypoint model
config_keypoint = {
    "checkpoint_path": "./keypoints_model/keypoints_checkpoint_20250330.pth.tar", 
    "roi_size": 128 # size to which the ROI will be resized, 
}
# configure pattern-based pose estimation
config_pattern_based = {
    "max_iterations": 10,  # maximum number of iterations for ICP
    "max_keypoints_est_2d": 72,  # maximum number of keypoints to estimate in 2D
}

# marker segmentation 
# read image
image_path = "./test_images/picture_54.png"
# check if image exists
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found at {image_path}")
# read image
image = cv2.imread(image_path) 

# segment marker from image
marker_segmentation = segment_marker(image, config_segmentation) 

# visualize segmentation
plt.imshow(marker_segmentation, cmap='gray')
plt.title('Segmented Marker')
plt.axis('off')
plt.show()

import pdb; pdb.set_trace()  # Debugging breakpoint


# learning based pose estimation 
# perform keypoint estimation
keypoints = estimate_keypoints(image, marker_segmentation, config_keypoint)
overlay = overlay_points_on_image(image.copy(), keypoints, radius=5)
# use PnP to solve for pose 
tf_cam_marker_LBCV = estimate_pose_from_keypoints(keypoints, config_camera, config_marker, config_camera)

print("Estimated Pose (LBCV):\n", tf_cam_marker_LBCV)

# visualize original image and segmentation and keypoints 
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")
plt.subplot(1, 3, 2)
plt.imshow(marker_segmentation, cmap='gray')
plt.title("Segmented Marker")
plt.axis("off")
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title("Keypoints Overlay")
plt.axis("off")
plt.tight_layout()
plt.show()

# pattern-based pose estimation
tf_cam_marker_PBCV, image_similarity_score = pattern_based_pose_estimation(
    image, marker_segmentation, tf_cam_marker_LBCV, config_marker, config_camera, config_pattern_based
)
print("Estimated Pose (PBCV):\n", tf_cam_marker_PBCV)
print("Image Similarity Score:", image_similarity_score)
