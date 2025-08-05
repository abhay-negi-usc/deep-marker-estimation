import numpy as np 
from scipy.spatial.transform import Rotation as R
import cv2 

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

def compute_2D_gridpoints(N=10,s=0.1): 
    # N = num squares, s = side length  
    u = np.linspace(-s/2, +s/2, N+1) 
    v = np.linspace(-s/2, +s/2, N+1) 
    gridpoints = [] 
    for uu in u:
        for vv in v: 
            gridpoints.append(np.array([uu,vv,0])) 
    return gridpoints 

def estimate_pose_from_keypoints(keypoints, config_camera, config_marker, config_keypoint):
    """
    Estimate pose from keypoints using PnP.
    """
    camera_matrix = np.array(config_camera['camera_matrix'])
    dist_coeffs = np.array(config_camera['dist_coeffs'])

    # Assuming keypoints are in the format (x, y) and we have a predefined tag size
    marker_length = config_marker.get('marker_length_with_border', 0.1)  # Default tag size in meters
    num_squares = config_marker.get('num_squares', 10)
    keypoints_tag_frame = np.array(
        compute_2D_gridpoints(N=num_squares, s=marker_length)
    )

    # Convert keypoints to the correct format for PnP
    keypoints_orig = np.array(keypoints).reshape(-1, 2).astype(np.float32)

    # Solve PnP
    success, rvec, tvec = cv2.solvePnP(
        objectPoints=keypoints_tag_frame,
        imagePoints=keypoints_orig,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs
    )
    if not success:
        return None 

    pose = rvectvec_to_xyzabc(rvec, tvec)
    tf_markerimage_markercoord = np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]) # accounting for convention difference

    tf_cam_marker = xyzabc_to_tf(pose) @ tf_markerimage_markercoord

    return tf_cam_marker 

