import numpy as np 
import cv2 

def overlay_points_on_image(image, pixel_points, radius=5, color=(0, 255, 0), thickness=-1):
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

def segmentation_biggest_blob_filter(segmentation_mask, min_area=1000):
    """
    Filters the segmentation mask to keep only the largest connected component (blob).
    
    Args:
        segmentation_mask (numpy.ndarray): Binary segmentation mask.
        min_area (int): Minimum area of the blob to keep.
    
    Returns:
        numpy.ndarray: Filtered segmentation mask with only the largest blob.
    """
    if segmentation_mask is None or not np.any(segmentation_mask):
        return None
    
    # Find contours
    contours, _ = cv2.findContours(segmentation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    
    if cv2.contourArea(largest_contour) < min_area:
        return None
    
    # Create a new mask for the largest blob
    filtered_mask = np.zeros_like(segmentation_mask)
    cv2.drawContours(filtered_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    return filtered_mask

def find_keypoints(img_rgb, img_seg=None, maxCorners=100, qualityLevel=0.01, minDistance=10, blockSize=7):
    """
    Find keypoints in an image using Shi-Tomasi corner detection.

    Parameters:
    - img_rgb: The input RGB image (a NumPy array).
    - img_seg: Optional segmentation mask (a NumPy array). If provided, keypoints will be detected only in the segmented area.
    - maxCorners: Maximum number of corners to return.
    - qualityLevel: Parameter characterizing the minimal accepted quality of image corners.
    - minDistance: Minimum possible Euclidean distance between the returned corners.
    - blockSize: Size of an average block for computing a derivative covariation matrix over each pixel neighborhood.           
    Returns:
    - keypoints: Detected keypoints in the image. If no keypoints are found, returns None.
    """

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    if img_seg is None:
        keypoints = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10, blockSize=7)
    else: 
        # If img_seg is 3-channel, convert it
        if len(img_seg.shape) == 3 and img_seg.shape[2] == 3:
            img_seg_gray = cv2.cvtColor(img_seg, cv2.COLOR_BGR2GRAY)
        else:
            img_seg_gray = img_seg.copy()
        if img_seg_gray.max() <= 1.0:  # If the mask is normalized
            img_seg_gray = (img_seg_gray * 255).astype(np.uint8)
        _, mask_binary = cv2.threshold(img_seg_gray, 127, 255, cv2.THRESH_BINARY)
        keypoints = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10, blockSize=7, mask=mask_binary)
    if keypoints is not None:
        keypoints = keypoints.reshape(-1, 2)  # Reshape to (N, 2) where N is the number of keypoints
    return keypoints