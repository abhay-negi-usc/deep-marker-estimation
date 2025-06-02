import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn 
import numpy as np 
from scipy.spatial.transform import Rotation as R 
import cv2 

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_keypointsdir,
    val_dir,
    val_keypointsdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = MarkersDataset(
        image_dir=train_dir,
        keypoints_dir=train_keypointsdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = MarkersDataset(
        image_dir=val_dir,
        keypoints_dir=val_keypointsdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def evaluate_mse_loss(loader, model, device): 
    model.eval() 
    total_mse_loss = 0.0 
    with torch.no_grad(): 
        for data, targets in loader: 
            data, targets = data.to(device), targets.to(device) 
            data = data.to(torch.float32).permute(0,3,1,2) 
            outputs = model(data) 

            # compute MSE loss 
            mse_loss = nn.MSELoss()(outputs, targets) 
            total_mse_loss += mse_loss.item() 
        avg_mse_loss = total_mse_loss / len(loader) 
        return avg_mse_loss

def evaluate_l1_loss(loader, model, device):
    model.eval()
    total_l1_loss = 0.0
    with torch.no_grad():   
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            data = data.to(torch.float32).permute(0,3,1,2)
            outputs = model(data)

            # compute L1 loss
            l1_loss = nn.L1Loss()(outputs, targets)
            total_l1_loss += l1_loss.item()
        avg_l1_loss = total_l1_loss / len(loader)
        return avg_l1_loss

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(x, f"{folder}/rgb_{idx}.png", normalize=True)
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/seg_{idx}.png", normalize=True)

    model.train()

def xyzabc_to_rvectvec(xyzabc): 
    tvec = xyzabc[:3] 
    rot = R.from_euler("xyz",xyzabc[3:],degrees=True).as_matrix()
    rvec = cv2.Rodrigues(rot)[0] 
    return rvec, tvec 

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

def tf_to_xyzabc(tf): 
    rot = tf[:3,:3] 
    tvec = tf[:3,3] 
    xyzabc = np.concatenate((tvec, R.from_matrix(rot).as_euler("xyz",degrees=True))) 
    return xyzabc 

def compute_2D_gridpoints(N=10,s=0.1): 
    # N = num squares, s = side length  
    u = np.linspace(-s/2, +s/2, N+1) 
    v = np.linspace(-s/2, +s/2, N+1) 
    gridpoints = [] 
    for uu in u:
        for vv in v: 
            gridpoints.append(np.array([uu,vv,0])) 
    return gridpoints 

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