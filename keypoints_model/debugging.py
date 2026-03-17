import faulthandler
faulthandler.enable()

import os
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["WANDB_START_METHOD"]   = "thread"

import cv2
import numpy as np
import torch

print("=== Step 1: CUDA init ===")
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.zeros(1).cuda()
print("ok")

print("=== Step 2: wandb init ===")
import wandb
wandb.init(project="debug-test", mode="disabled")  # disabled = no network calls
print("ok")

print("=== Step 3: model ===")
from keypoints_model.model import RegressorMobileNetV3
model = RegressorMobileNetV3().to(device)
print("ok")

print("=== Step 4: dataset __init__ ===")
from keypoints_model.train import MarkerKeypointDataset, resolve_split
data_dir = "/home/anegi/abhay_ws/deep-marker-estimation/data_generation/multi_marker_output/multi_marker_20260218-201012/"
train_stems, val_stems = resolve_split(data_dir)
ds = MarkerKeypointDataset(
    image_dir=os.path.join(data_dir, "images"),
    metadata_dir=os.path.join(data_dir, "metadata"),
    stems=train_stems[:100],   # only 100 stems to keep it fast
    transform=None,
    roi_size=(300, 480),
    target_k=81,
)
print(f"ok — {len(ds)} samples")

print("=== Step 5: single __getitem__ ===")
crop, target = ds[0]
print(f"ok — crop={crop.shape} target={target.shape}")

print("=== Step 6: DataLoader, num_workers=0 ===")
from torch.utils.data import DataLoader
loader = DataLoader(ds, batch_size=4, num_workers=0)
batch_img, batch_tgt = next(iter(loader))
print(f"ok — batch={batch_img.shape}")

# In debugging.py, wrap the spawn test:
if __name__ == '__main__':
    print("=== Step 7: DataLoader, num_workers=1 spawn ===")
    loader2 = DataLoader(ds, batch_size=4, num_workers=1,
                         multiprocessing_context="spawn")
    batch_img2, batch_tgt2 = next(iter(loader2))
    print(f"ok — batch={batch_img2.shape}")

    print("=== Step 8: forward pass ===")
    x = batch_img.float().permute(0,3,1,2).cuda()
    mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).cuda()
    std  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).cuda()
    x = (x/255.0 - mean) / std
    out = model(x)
    print(f"ok — output={out.shape}")

    print("=== Step 9: backward pass ===")
    loss = ((out - batch_tgt.float().cuda())**2).mean()
    loss.backward()
    print(f"ok — loss={loss.item():.4f}")

    print("\n=== ALL STEPS PASSED ===")