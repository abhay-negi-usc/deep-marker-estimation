import numpy as np 
import matplotlib.pyplot as plt
import cv2 
from scipy.spatial.transform import Rotation as R
import torch 

from utils.image_utils import * 
from utils.homography_utils import * 

def convert_marker_keypoints_to_cartesian(keypoints_image_space, image_size, marker_size=(0.1, 0.1)): 
    """
    Convert keypoints from image space to Cartesian coordinates based on the marker size and image dimensions.
    
    Args:
        keypoints_image_space (list): List of keypoints in image space (2D coordinates).
        image_size (tuple): Size of the image (height, width).
        marker_size (tuple): Size of the marker in meters (width, height).
    
    Returns:
        list: List of keypoints in Cartesian coordinates (3D coordinates).
    """
    height, width = image_size
    cartesian_keypoints = []
    for kp in keypoints_image_space:
        # Normalize the keypoint coordinates to the range [-1, 1]
        kp = kp.reshape(2)
        x_norm = (kp[0] / width) * 2 - 1
        y_norm = (kp[1] / height) * 2 - 1
        x_norm = -x_norm  # rotate about y-axis to convert from image coordinates to marker coordinates # NOTE: make this configurable
        # Convert normalized coordinates to Cartesian space
        x_cartesian = x_norm * marker_size[0] / 2
        y_cartesian = y_norm * marker_size[1] / 2
        cartesian_keypoints.append([x_cartesian, y_cartesian, 0])  # Z-coordinate is set to 0 for a flat marker
    cartesian_keypoints = np.array(cartesian_keypoints, dtype=np.float64)
    return cartesian_keypoints


def overlay_3D_points_on_image(image, points_3d, camera_matrix, tf_est, color=(0, 255, 0), radius=5):
    """
    Overlay 3D points on an image using the estimated transformation matrix and camera intrinsic parameters.
    
    Args:
        image (numpy.ndarray): The input image (BGR format).
        points_3d (list): List of 3D points to overlay on the image.
        camera_matrix (numpy.ndarray): Camera intrinsic matrix (3x3).
        tf_est (numpy.ndarray): Estimated transformation matrix (4x4).
        color (tuple): Color for the overlay points in BGR format.
        radius (int): Radius of the overlay points.
    
    Returns:
        numpy.ndarray: The image with 3D points overlaid.
    """
    # Transform 3D points to camera space
    points_3d_homogeneous = np.hstack((points_3d, np.ones((len(points_3d), 1))))
    points_camera_space = tf_est @ points_3d_homogeneous.T
    points_camera_space = points_camera_space[:3, :].T  # Convert back to 3D coordinates

    # Project 3D points to 2D image space
    projected_points = cv2.projectPoints(points_camera_space, np.zeros(3), np.zeros(3), camera_matrix, None)[0].reshape(-1, 2)

    # Overlay points on the image
    for point in projected_points:
        cv2.circle(image, tuple(point.astype(int)), radius, color, -1)

    return image

def refine_pose_icp_3d2d_auto_match(
    image_np, keypoints_ref_3d, keypoints_est_2d, camera_matrix, tf_init=None, dist_coeffs=None, 
    max_iterations=20, max_keypoints_est_2d=100, outlier_percentile=90,
    show_iteration_images=False, plot_residual=False, plot_estimate=False, output_final_image=False
):
    """
    Refine camera pose using 3D points and 2D image points, with optional outlier removal.
    """

    # --- Initial estimate ---
    if tf_init is not None:
        R_init = tf_init[:3, :3]
        t_init = tf_init[:3, 3]
        rvec, _ = cv2.Rodrigues(R_init)
        tvec = t_init.reshape(3, 1).astype(np.float32)
    else:
        rvec = np.zeros((3, 1), dtype=np.float32)
        tvec = np.zeros((3, 1), dtype=np.float32)

    if dist_coeffs is None:
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float32)
    keypoints_ref_3d = np.array(keypoints_ref_3d).reshape(-1, 3)

    # --- Clean and filter 2D points ---
    keypoints_est_2d = np.asarray(keypoints_est_2d, dtype=np.float32).reshape(-1, 2)
    if len(keypoints_est_2d) > max_keypoints_est_2d:
        center = np.mean(keypoints_est_2d, axis=0)
        distances = np.linalg.norm(keypoints_est_2d - center, axis=1)
        sorted_indices = np.argsort(distances)
        keypoints_est_2d = keypoints_est_2d[sorted_indices[:max_keypoints_est_2d]]

    residual_history, eul_history, tvec_history = [], [], []

    tf_est = tf_init 

    for i in range(max_iterations):
        projected_points, _ = cv2.projectPoints(keypoints_ref_3d, rvec.copy(), tvec.copy(), camera_matrix, dist_coeffs)
        projected_points = projected_points.reshape(-1, 2)

        if show_iteration_images and i ==0:
            img_overlay = overlay_points_on_image(image_np.copy(), projected_points, radius=5, color=(255, 0, 0))
            plt.imshow(cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB))
            plt.title(f'Initial estimate keypoint overlay')
            plt.axis('off')
            plt.show()

        # Find nearest neighbors
        distances = np.linalg.norm(projected_points[:, None, :] - keypoints_est_2d[None, :, :], axis=2)
        nearest_indices = np.argmin(distances, axis=1)

        matched_3d = keypoints_ref_3d
        matched_2d = keypoints_est_2d[nearest_indices]

        # Compute residuals
        residuals = np.linalg.norm(projected_points - matched_2d, axis=1)

        # Outlier rejection
        threshold = np.percentile(residuals, outlier_percentile)
        inlier_mask = residuals <= threshold

        matched_3d_inliers = matched_3d[inlier_mask]
        matched_2d_inliers = matched_2d[inlier_mask]

        if matched_3d_inliers.shape[0] >= 4:
            success, rvec_new, tvec_new = cv2.solvePnP(
                matched_3d_inliers, matched_2d_inliers, camera_matrix, dist_coeffs,
                rvec.copy(), tvec.copy(), useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not success:
                print(f"Iteration {i}: solvePnP failed, stopping early.")
                break

            rvec, tvec = rvec_new, tvec_new

            rot_matrix, _ = cv2.Rodrigues(rvec)
            tf_est = np.eye(4)
            tf_est[:3, :3] = rot_matrix
            tf_est[:3, 3] = tvec.flatten()

            mean_residual = np.mean(residuals[inlier_mask])
            residual_history.append(mean_residual)

            eul = R.from_matrix(rot_matrix).as_euler('xyz', degrees=True)
            eul_history.append(eul)
            tvec_history.append(tvec.flatten())

            if show_iteration_images:
                img_overlay = overlay_3D_points_on_image(
                    # np.zeros((720, 1280, 3), dtype=np.uint8),
                    image_np.copy(),  
                    matched_3d_inliers, camera_matrix, tf_est, 
                    color=(0, 255, 0), radius=5
                )
                plt.imshow(cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB))
                plt.title(f'Iteration {i}, Residual: {mean_residual:.4f}')
                plt.axis('off')
                plt.show()

            if output_final_image and i == max_iterations - 1:
                img_overlay = overlay_3D_points_on_image(
                    image_np.copy(), matched_3d_inliers, camera_matrix, tf_est, 
                    color=(0, 255, 0), radius=5
                )
                # save image
                output_image_path = "./ablations/analysis/real_exp/final_overlay_image.png"
                cv2.imwrite(output_image_path, img_overlay)
                print(f"Final overlay image saved to {output_image_path}")

        else:
            print(f"Iteration {i}: Not enough inlier points to solvePnP.")
            break

    # --- Plots ---
    if plot_residual and residual_history:
        plt.figure()
        plt.plot(residual_history, marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('Mean Residual')
        plt.title('ICP Residuals Over Iterations')
        plt.grid(True)
        plt.show()

    if plot_estimate and eul_history:
        fig, axs = plt.subplots(2, 3, figsize=(15, 8))
        eul_history = np.array(eul_history)
        tvec_history = np.array(tvec_history)
        axs[0, 0].plot(eul_history[:, 0], marker='o'); axs[0, 0].set_title('Rotation X')
        axs[0, 1].plot(eul_history[:, 1], marker='o'); axs[0, 1].set_title('Rotation Y')
        axs[0, 2].plot(eul_history[:, 2], marker='o'); axs[0, 2].set_title('Rotation Z')
        axs[1, 0].plot(tvec_history[:, 0], marker='o'); axs[1, 0].set_title('Translation X')
        axs[1, 1].plot(tvec_history[:, 1], marker='o'); axs[1, 1].set_title('Translation Y')
        axs[1, 2].plot(tvec_history[:, 2], marker='o'); axs[1, 2].set_title('Translation Z')
        for ax in axs.flat:
            ax.set_xlabel('Iteration')
            ax.grid(True)
        plt.tight_layout()
        plt.show()

    return tf_est, residual_history[-1] if residual_history else None

def _image_similarity_score(image, rendered, rendered_seg, threshold=0.25): 
    if rendered_seg.sum() == 0:
        return 0 
    # turn images black and white 
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rendered_gray = cv2.cvtColor(rendered, cv2.COLOR_BGR2GRAY)
    seg = rendered_seg 
    # get coordinates of non-zero pixels in the rendered segmentation
    non_zero_coords = np.column_stack(np.where(seg > 0))
    # normalize images at non_zero_coords 
    image_marker_region = image_gray[non_zero_coords[:, 0], non_zero_coords[:, 1]]
    render_marker_region = rendered_gray[non_zero_coords[:, 0], non_zero_coords[:, 1]]
    image_marker_region_normalized = image_marker_region / (image_marker_region.max() - image_marker_region.min())  # Normalize to [0, 1]
    render_marker_region_normalized = render_marker_region / (render_marker_region.max() - render_marker_region.min()) # Normalize to [0, 1]
    similarity_score = 0 
    for idx in range(len(image_marker_region_normalized)):
        if abs(image_marker_region_normalized[idx] - render_marker_region_normalized[idx]) < threshold:
            similarity_score += 1 
    return similarity_score 

def compute_image_similarity_score(image, image_marker, marker_length, tf, camera_matrix, dist_coeffs): 
    marker_corners_2d = np.array([
        [0, 0],
        [0, image_marker.shape[0]],
        [image_marker.shape[1], image_marker.shape[0]],
        [image_marker.shape[1], 0]
    ], dtype=np.float32)
    marker_corners_3d = np.array([
        [0, 0, 0],
        [marker_length, 0, 0],
        [marker_length, marker_length, 0],
        [0, marker_length, 0]
    ], dtype=np.float32)
    pose = np.zeros(6)
    pose[:3] = tf[:3, 3]
    pose[3:] = R.from_matrix(tf[:3, :3]).as_euler('xyz', degrees=True)
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    pose_tensor = torch.tensor(pose, dtype=torch.float32, device=device)
    marker_seg = image.copy()
    marker_seg[:, :, :] = marker_seg.max()
    # FIXME: remove unnecessary GPU operations 
    rendered = marker_reprojection_differentiable(image_marker, marker_corners_2d, marker_corners_3d, pose_tensor, camera_matrix, image_size=(image.shape[0],image.shape[1])) 
    rendered_seg = marker_reprojection_differentiable(marker_seg, marker_corners_2d, marker_corners_3d, pose_tensor, camera_matrix, image_size=(image.shape[0],image.shape[1]))
    rendered = rendered.cpu().numpy().squeeze(0).transpose(1, 2, 0) * 255
    rendered_seg = rendered_seg.cpu().numpy().squeeze(0).transpose(1, 2, 0) * 255 
    rendered_seg = rendered_seg[:,:,0] # Use only one channel for segmentation 
    similarity_score = _image_similarity_score(image, rendered, rendered_seg) 
    return similarity_score

def pattern_based_pose_estimation(image, seg, tf_init, marker_config, camera_config, pattern_based_config):  

    camera_matrix = np.array(camera_config['camera_matrix'])
    dist_coeffs = np.array(camera_config['dist_coeffs'])
    marker_length = marker_config.get('marker_length_with_border', 0.1)  # Default tag size in meters
    img_marker = marker_config.get('marker_image', None)
    keypoints_marker_cartesian_space = marker_config.get('keypoints_marker_cartesian_space', None)

    max_iterations = pattern_based_config.get('max_iterations', 10)
    max_keypoints_est_2d = pattern_based_config.get('max_keypoints_est_2d', 100)

    seg = segmentation_biggest_blob_filter(seg, min_area=1000)

    keypoints_rgb_image_space = find_keypoints(image, seg, maxCorners=100, qualityLevel=0.1, minDistance=10, blockSize=7)    

    # visualize keypoints on the image
    overlay = overlay_points_on_image(image.copy(), keypoints_rgb_image_space, radius=5, color=(0, 255, 0))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title('Keypoints on Image')
    plt.axis('off')
    plt.show()

    tf_cam_marker, residual = refine_pose_icp_3d2d_auto_match(
        np.array(image), keypoints_marker_cartesian_space, keypoints_rgb_image_space, camera_matrix,
        tf_init, max_iterations=max_iterations, show_iteration_images=True, max_keypoints_est_2d=max_keypoints_est_2d, output_final_image=False,
    )
    image_similarity_score = compute_image_similarity_score(image, img_marker, marker_length, tf_cam_marker, camera_matrix, dist_coeffs)
    return tf_cam_marker, image_similarity_score 

