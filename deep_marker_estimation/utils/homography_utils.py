import numpy as np 
import cv2 
import torch 
import torch.nn.functional as F

def project_points_torch(points_3d, rvec, tvec, K):
    """
    Differentiable implementation of cv2.projectPoints.

    Args:
        points_3d: (N, 3) torch tensor
        rvec: (3,) torch tensor, axis-angle
        tvec: (3,) torch tensor
        K: (3, 3) torch tensor

    Returns:
        points_2d: (N, 2) torch tensor in pixel space
    """
    # Convert rvec (axis-angle) to rotation matrix using Rodrigues formula
    theta = torch.norm(rvec)
    if theta < 1e-6:
        R = torch.eye(3, device=rvec.device, dtype=rvec.dtype)
    else:
        k = rvec / theta
        K_cross = torch.tensor([
            [0, -k[2], k[1]],
            [k[2], 0, -k[0]],
            [-k[1], k[0], 0]
        ], device=rvec.device, dtype=rvec.dtype)
        R = torch.eye(3, device=rvec.device) + torch.sin(theta) * K_cross + (1 - torch.cos(theta)) * (K_cross @ K_cross)

    points_cam = (R @ points_3d.T).T + tvec  # (N, 3)
    points_proj = (K @ points_cam.T).T  # (N, 3)
    points_2d = points_proj[:, :2] / points_proj[:, 2:3]  # (N, 2)

    return points_2d

def euler_to_rot_matrix(euler_deg: torch.Tensor) -> torch.Tensor:
    """ Convert xyz Euler angles in degrees to a rotation matrix (3x3) in a differentiable way. """
    euler_rad = torch.deg2rad(euler_deg)
    x, y, z = euler_rad

    Rx = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(x), -torch.sin(x)],
        [0, torch.sin(x), torch.cos(x)]
    ], device=euler_deg.device)

    Ry = torch.tensor([
        [torch.cos(y), 0, torch.sin(y)],
        [0, 1, 0],
        [-torch.sin(y), 0, torch.cos(y)]
    ], device=euler_deg.device)

    Rz = torch.tensor([
        [torch.cos(z), -torch.sin(z), 0],
        [torch.sin(z), torch.cos(z), 0],
        [0, 0, 1]
    ], device=euler_deg.device)

    return Rz @ Ry @ Rx  # XYZ order


def compute_homography_dlt_batched(src_pts: torch.Tensor, dst_pts: torch.Tensor) -> torch.Tensor:
    B = src_pts.shape[0]
    assert src_pts.shape == (B, 4, 2)
    assert dst_pts.shape == (B, 4, 2)

    A_list = []
    for i in range(4):
        x = src_pts[:, i, 0]
        y = src_pts[:, i, 1]
        u = dst_pts[:, i, 0]
        v = dst_pts[:, i, 1]

        zeros = torch.zeros_like(x)

        row1 = torch.stack([-x, -y, -torch.ones_like(x), zeros, zeros, zeros, u * x, u * y, u], dim=1)
        row2 = torch.stack([zeros, zeros, zeros, -x, -y, -torch.ones_like(x), v * x, v * y, v], dim=1)

        A_list.extend([row1, row2])

    A = torch.stack(A_list, dim=1)  # (B, 8, 9)
    _, _, V = torch.linalg.svd(A)
    h = V[:, -1, :]  # (B, 9)
    H = h.view(B, 3, 3)
    H = H / H[:, 2:3, 2:3]
    return H

def warp_perspective_torch(image: torch.Tensor, H: torch.Tensor, out_size: tuple) -> torch.Tensor:
    B, C, H_img, W_img = image.shape
    H_out, W_out = out_size

    y, x = torch.meshgrid(
        torch.linspace(0, H_out - 1, H_out, device=image.device),
        torch.linspace(0, W_out - 1, W_out, device=image.device),
        indexing="ij"
    )
    grid = torch.stack([x, y, torch.ones_like(x)], dim=-1)
    grid = grid.view(-1, 3).permute(1, 0).unsqueeze(0)

    H_inv = torch.inverse(H)
    warped = H_inv @ grid
    warped = warped[:, :2, :] / warped[:, 2:3, :]

    norm_x = 2 * warped[:, 0, :] / (W_img - 1) - 1
    norm_y = 2 * warped[:, 1, :] / (H_img - 1) - 1
    grid = torch.stack([norm_x, norm_y], dim=-1).view(1, H_out, W_out, 2)

    out = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return out

def marker_reprojection_differentiable(marker_image_np, marker_corners_2d, marker_corners_3d, xyzabc, camera_matrix, image_size=(480, 640)):
    if marker_image_np.ndim == 2:
        marker_tensor = torch.from_numpy(marker_image_np).float() / 255.0
        marker_tensor = marker_tensor.unsqueeze(0).unsqueeze(0)
    elif marker_image_np.ndim == 3:
        marker_tensor = torch.from_numpy(marker_image_np).permute(2, 0, 1).float() / 255.0
        marker_tensor = marker_tensor.unsqueeze(0)
    else:
        raise ValueError("Unexpected marker image shape")

    device = xyzabc.device
    marker_tensor = marker_tensor.to(device)

    t = xyzabc[:3]
    R_mat = euler_to_rot_matrix(xyzabc[3:])
    rvec, _ = cv2.Rodrigues(R_mat.cpu().numpy())
    rvec = torch.from_numpy(rvec).squeeze().float().to(device)

    marker_corners_3d = torch.from_numpy(marker_corners_3d).float().to(device)
    K = torch.from_numpy(camera_matrix).float().to(device)
    tvec = t.float().unsqueeze(0)

    marker_corners_2d_proj = project_points_torch(marker_corners_3d, rvec, tvec.squeeze(0), K)  # (4, 2)

    src_pts = torch.from_numpy(marker_corners_2d).unsqueeze(0).float().to(device)
    dst_pts = marker_corners_2d_proj.unsqueeze(0)
    H = compute_homography_dlt_batched(src_pts, dst_pts)

    warped = warp_perspective_torch(marker_tensor, H, image_size)
    return warped

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