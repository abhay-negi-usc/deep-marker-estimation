import cv2, numpy as np
from deep_marker_estimation import MarkerEstimator

# Init once
est = MarkerEstimator()

# Per-frame
img = cv2.imread("/path/to/frame.png", cv2.IMREAD_COLOR)
K = np.array([[906.995, 0, 638.235],[0, 906.995, 360.533],[0, 0, 1]], dtype=float)
D = np.zeros(5)

result = est.estimate_pose(img, camera_matrix=K, dist_coeffs=D)
if result:
    R = result["pbcv"]["R"] if "pbcv" in result else result["lbcv"]["R"]
    t = result["pbcv"]["t"] if "pbcv" in result else result["lbcv"]["t"]
    qwxyz = result["quaternion_wxyz"]
    print("R=\n", R, "\nt=", t, "\nquat(wxyz)=", qwxyz)
