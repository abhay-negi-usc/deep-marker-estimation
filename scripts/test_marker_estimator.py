#!/usr/bin/env python3
import os, cv2, numpy as np, yaml
from deep_marker_estimation import MarkerEstimator

def pose_to_T(R, t):
    T = np.eye(4, dtype=float)
    T[:3,:3] = R
    T[:3, 3] = t.reshape(3,)
    return T

def save_pose_yaml(path, R, t, quat_wxyz):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "R": R.tolist(),
        "t": t.reshape(3,).tolist(),
        "T_4x4": pose_to_T(R, t).tolist(),
        "quaternion_wxyz": list(map(float, quat_wxyz)),
    }
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

def main():
    img_path = "./test_images/marker_ablation_examples/skew_negative.png"
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")

    # Example intrinsics; replace with your calibrated values
    K = np.array([[906.995, 0, 638.235],
                  [0, 906.995, 360.533],
                  [0, 0, 1.0]], dtype=float)
    D = np.zeros(5, dtype=float)

    est = MarkerEstimator()
    res = est.estimate_pose(img, camera_matrix=K, dist_coeffs=D, return_debug=True)

    if not res:
        print("No marker detected.")
        return

    # Prefer PBCV if available; otherwise use LBCV
    if "pbcv" in res:
        R = res["pbcv"]["R"]; t = res["pbcv"]["t"]
        source = "PBCV"
    else:
        print("PBCV failed, falling back to LBCV.")
        R = res["lbcv"]["R"]; t = res["lbcv"]["t"]
        source = "LBCV (fallback)"

    T = pose_to_T(R, t)

    print(f"\n=== {source} Pose ===")
    print("R:\n", R)
    print("t:", t.reshape(3,))
    print("T (4x4):\n", T)
    print("Quaternion (w,x,y,z):", res["quaternion_wxyz"])

    # Save outputs
    out_dir = "./test_images/_out"
    os.makedirs(out_dir, exist_ok=True)

    # Save images
    cv2.imwrite(os.path.join(out_dir, "original.png"), img)
    seg = res["lbcv"]["segmentation"]
    seg_vis = (seg * 255).astype(np.uint8) 
    cv2.imwrite(os.path.join(out_dir, "segmentation.png"), seg_vis)

    overlay = img.copy()
    for x, y in res["lbcv"]["keypoints_2d"]:
        cv2.circle(overlay, (int(x), int(y)), 5, (0, 255, 0), -1)
    cv2.imwrite(os.path.join(out_dir, "keypoints_overlay.png"), overlay)

    # Save pose to YAML
    save_pose_yaml(os.path.join(out_dir, "pose_pbcv.yaml"), R, t, res["quaternion_wxyz"])
    print(f"\nSaved pose to {os.path.join(out_dir, 'pose_pbcv.yaml')}")

if __name__ == "__main__":
    main()
