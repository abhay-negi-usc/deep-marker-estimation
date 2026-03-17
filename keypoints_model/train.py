#!/usr/bin/env python3
"""
Train a keypoint regressor (MobileNetV3-based) on marker keypoints produced by
homography_markers_data_generation.py.

Data layout expected under config["data_dir"]:

    <data_dir>/
        images/       rendered_NNNNN.png   (or .jpg / .jpeg)
        metadata/     rendered_NNNNN.json

Each metadata JSON contains a list of "markers", each with:
    - quad_px: [[x0,y0], ..., [x3,y3]]   — 4 corners in full-image pixel space
    - keypoints: [{row, col, image_xy_px:[u,v], visible:bool}, ...]
    - keypoints_grid_size: int            — N for the N×N grid

One dataset sample = one marker crop from one image.
The model predicts all N×N keypoint coordinates in ROI pixel space.
Invisible keypoints (visible=False) are stored as (−1, −1) and excluded from
the loss via masked MSE.

Train / val split
─────────────────
On the first run the script shuffles all available stems (with a fixed random
seed) and writes a split manifest to

    <data_dir>/split.json

On every subsequent run (including restarts from a checkpoint) the exact same
manifest is reloaded automatically, guaranteeing that no val image ever leaks
into the training set.  Delete split.json to force a fresh split.
"""


import os
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["WANDB_START_METHOD"]   = "thread"

import faulthandler
faulthandler.enable()

import json

import albumentations as A
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import wandb
from keypoints_model.model import RegressorMobileNetV3
from keypoints_model.utils import load_checkpoint, overlay_points_on_image, save_checkpoint

matplotlib.use("Agg")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# Train / val split
# ─────────────────────────────────────────────────────────────────────────────

def resolve_split(
    data_dir: str,
    val_fraction: float = 0.1,
    seed: int = 42,
    split_filename: str = "split.json",
) -> tuple[list[str], list[str]]:
    """Return (train_stems, val_stems), loading or creating a split manifest.

    The manifest is saved to <data_dir>/<split_filename> on first run and
    reloaded verbatim on every subsequent run, so restarting from a checkpoint
    always uses the identical split.

    Each "stem" is the bare filename without extension, e.g. "rendered_00042".
    Both the images/ and metadata/ directories are expected to share the same
    stems (the generator guarantees this).

    Args:
        data_dir:       Root data directory containing images/ and metadata/.
        val_fraction:   Fraction of images held out for validation (default 0.1).
        seed:           RNG seed used when creating a new split (stored in the
                        manifest so it is always recoverable).
        split_filename: Name of the manifest JSON written into data_dir.

    Returns:
        (train_stems, val_stems) — two lists of stem strings.
    """
    split_path = os.path.join(data_dir, split_filename)

    # ── Reload existing split ────────────────────────────────────────────────
    if os.path.isfile(split_path):
        with open(split_path) as f:
            manifest = json.load(f)
        train_stems = manifest["train"]
        val_stems   = manifest["val"]
        print(
            f"[split] Loaded existing split from {split_path}  "
            f"(train={len(train_stems):,}  val={len(val_stems):,}  "
            f"seed={manifest.get('seed')})"
        )
        return train_stems, val_stems

    # ── Discover all stems from the metadata directory ──────────────────────
    meta_dir = os.path.join(data_dir, "metadata")
    if not os.path.isdir(meta_dir):
        raise FileNotFoundError(
            f"metadata/ directory not found under data_dir: {data_dir}"
        )
    all_stems = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(meta_dir)
        if f.endswith(".json")
    )
    if not all_stems:
        raise FileNotFoundError(f"No JSON files found in {meta_dir}")

    # ── Shuffle deterministically and split ─────────────────────────────────
    rng = np.random.default_rng(seed)
    indices    = rng.permutation(len(all_stems)).tolist()
    n_val      = max(1, int(round(len(all_stems) * val_fraction)))
    val_idx    = set(indices[:n_val])
    train_stems = [all_stems[i] for i in range(len(all_stems)) if i not in val_idx]
    val_stems   = [all_stems[i] for i in range(len(all_stems)) if i in     val_idx]

    # ── Persist manifest ─────────────────────────────────────────────────────
    manifest = {
        "seed":         seed,
        "val_fraction": val_fraction,
        "n_total":      len(all_stems),
        "n_train":      len(train_stems),
        "n_val":        len(val_stems),
        "train":        train_stems,
        "val":          val_stems,
    }
    with open(split_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(
        f"[split] Created new split → {split_path}  "
        f"(train={len(train_stems):,}  val={len(val_stems):,}  seed={seed})"
    )
    return train_stems, val_stems


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalize_batch(data: torch.Tensor, device: str) -> torch.Tensor:
    """ImageNet-normalize a float32 NCHW tensor whose values are in [0, 255]."""
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=device).view(1, 3, 1, 1)
    return (data / 255.0 - mean) / std


def masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE loss that ignores slots where target == −1 (invisible / out-of-frame).

    pred, target: (B, 2K) — interleaved u0 v0 u1 v1 …
    """
    valid = (target[:, 0::2] >= 0) & (target[:, 1::2] >= 0)   # (B, K) bool
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    valid_coords = valid.unsqueeze(2).expand(-1, -1, 2).reshape(pred.shape)
    diff2 = (pred - target) ** 2
    return (diff2 * valid_coords.float()).sum() / (valid_coords.float().sum() + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Training / evaluation steps
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(loader, model, optimizer, scaler, device,
                    global_step=0, cfg=None, val_loader=None, epoch=None):
    """Train for one epoch.

    Returns (avg_train_loss, updated_global_step).
    Performs step-level checkpointing, prediction snapshots, and wandb logging
    at the intervals specified in cfg.
    """
    model.train()
    loop = tqdm(loader, desc=f"Epoch {epoch}")
    total_loss, total_samples = 0.0, 0

    ckpt_steps  = int(cfg.get("checkpoint_interval_steps", 0))  if cfg else 0
    pred_steps  = int(cfg.get("prediction_interval_steps", 0))  if cfg else 0
    log_steps   = int(cfg.get("wandb_log_interval_steps",  10)) if cfg else 10
    n_pred_imgs = int(cfg.get("num_prediction_images",     8))  if cfg else 8

    for data, targets in loop:
        data    = normalize_batch(data.to(device).float().permute(0, 3, 1, 2), device)
        targets = targets.float().to(device)

        with torch.amp.autocast(device_type=device):
            loss = masked_mse(model(data), targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_loss     = float(loss.item())
        total_loss    += batch_loss * data.size(0)
        total_samples += data.size(0)
        global_step   += 1

        # Step-level wandb logging
        if log_steps > 0 and global_step % log_steps == 0:
            wandb.log(
                {"train/loss_step": batch_loss, "epoch": epoch, "global_step": global_step},
                step=global_step,
            )

        # Step-level checkpoint
        if ckpt_steps > 0 and global_step % ckpt_steps == 0:
            save_checkpoint(
                {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "global_step": global_step},
                f"{cfg['checkpoint_path']}_step_{global_step}.pth.tar",
            )

        # Step-level prediction snapshot
        if pred_steps > 0 and global_step % pred_steps == 0 and val_loader is not None:
            save_predictions(val_loader, model, cfg=cfg, device=device,
                             n=n_pred_imgs, step=global_step)

        loop.set_postfix(loss=f"{batch_loss:.4f}", step=global_step)

    return total_loss / max(1, total_samples), global_step


@torch.no_grad()
def evaluate_loss(loader, model, device):
    model.eval()
    total_loss, total_samples = 0.0, 0
    for data, targets in loader:
        data    = normalize_batch(data.to(device).float().permute(0, 3, 1, 2), device)
        targets = targets.float().to(device)
        loss    = masked_mse(model(data), targets)
        total_loss    += loss.item() * data.size(0)
        total_samples += data.size(0)
    model.train()
    return total_loss / max(1, total_samples)


# ─────────────────────────────────────────────────────────────────────────────
# Prediction snapshots
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def save_predictions(loader, model, cfg, device, n=8, step=None):
    """Save n sample images with predicted keypoints overlaid and log to wandb.

    When cfg["save_keypoint_json"] is True, also writes a JSON file containing
    the predicted ROI-pixel coordinates for each sample, which is then uploaded
    to wandb as an artifact.

    Args:
        loader:  DataLoader yielding (images_uint8_HWC, targets).
        model:   The regressor.
        cfg:     Config dict (uses "save_dir", "save_keypoint_json",
                 "image_width", "image_height").
        device:  Torch device string.
        n:       Number of images to save.
        step:    Current global training step (used for file naming and wandb).
    """
    folder    = cfg.get("save_dir", "saved_predictions")
    save_json = bool(cfg.get("save_keypoint_json", True))
    dot_r     = max(2, cfg["image_width"] // 120)   # scale dot radius to ROI size
    os.makedirs(folder, exist_ok=True)

    model.eval()
    saved        = 0
    wandb_images = {}
    json_records = []

    for x_raw, _ in loader:
        if saved >= n:
            break

        x_np = x_raw.cpu().numpy() if isinstance(x_raw, torch.Tensor) else x_raw
        B    = x_np.shape[0]

        # Batch inference
        imgs_f = (x_np.astype(np.float32) / 255.0 - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
        inp    = torch.tensor(imgs_f).permute(0, 3, 1, 2).float().to(device)
        preds_batch = model(inp).cpu().numpy()   # (B, 2K)

        for j in range(B):
            if saved >= n:
                break

            raw_img = x_np[j].astype(np.uint8)
            preds   = preds_batch[j].reshape(-1, 2)   # (K, 2) in ROI pixel coords

            vis_img   = overlay_points_on_image(raw_img.copy(), preds, radius=dot_r)
            save_path = os.path.join(folder, f"pred_{step or 0:07d}_{saved:03d}.png")
            plt.imsave(save_path, vis_img)

            wandb_images[f"predictions/{saved:03d}"] = wandb.Image(
                save_path,
                caption=f"step={step} sample={saved}",
            )

            if save_json:
                json_records.append({
                    "sample_id": saved,
                    "step":      step,
                    "keypoints_roi_px": preds.tolist(),
                })
            saved += 1

    # Log all images in one call
    log_payload = dict(wandb_images)
    if step is not None:
        log_payload["global_step"] = step
    wandb.log(log_payload, step=step)

    # Save and upload keypoint JSON
    if save_json and json_records:
        json_path = os.path.join(folder, f"keypoints_{step or 0:07d}.json")
        with open(json_path, "w") as f:
            json.dump(json_records, f, indent=2)
        wandb.save(json_path)

    model.train()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MarkerKeypointDataset(Dataset):
    """One sample = one marker crop from one rendered image.

    Reads full-image PNGs from <image_dir> and annotations from the JSON files
    in <metadata_dir> produced by homography_markers_data_generation.py.

    Only the image stems listed in *stems* are loaded — pass train_stems or
    val_stems from resolve_split() to guarantee a clean train / val separation.

    For each marker in each JSON:
    - A square ROI is cropped around the marker's bounding quad (with padding),
      then resized to (roi_h, roi_w).
    - All N×N keypoints are transformed into ROI pixel coordinates.
    - Keypoints with visible=False are stored as (−1, −1) and excluded from
      the loss via masked_mse.

    Target shape: (2 * target_k,) — flattened (u0,v0, u1,v1, …) in ROI pixels,
    ordered row-major by the grid (index = row*N + col).
    """

    def __init__(
        self,
        image_dir: str,
        metadata_dir: str,
        stems: list[str],           # explicit list of stems to include
        transform=None,
        roi_size: tuple = (300, 480),
        target_k: int = 81,
        padding: int = 16,
        min_side: int = 32,
    ):
        self.image_dir    = image_dir
        self.metadata_dir = metadata_dir
        self.transform    = transform
        self.roi_h, self.roi_w = roi_size
        self.target_k = int(target_k)
        self.padding  = int(padding)
        self.min_side = int(min_side)

        self.samples = []   # list of dicts, one per (image, marker) pair

        exts = (".png", ".jpg", ".jpeg")
        for stem in stems:
            # Find the image file
            img_path = next(
                (os.path.join(image_dir, stem + e) for e in exts
                 if os.path.isfile(os.path.join(image_dir, stem + e))),
                None,
            )
            if img_path is None:
                continue

            meta_path = os.path.join(metadata_dir, stem + ".json")
            if not os.path.isfile(meta_path):
                continue

            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except Exception:
                continue

            grid_size = int(meta.get("keypoints_grid_size", 9))
            for marker in meta.get("markers", []):
                kps  = marker.get("keypoints", [])
                quad = marker.get("quad_px")
                if not kps or quad is None:
                    continue
                # self.samples.append({
                #     "image_path": img_path,
                #     "quad_px":    np.array(quad, dtype=np.float32),   # (4, 2)
                #     "keypoints":  kps,                                  # list of dicts
                #     "grid_size":  grid_size,
                # })
                kp_array = np.full((target_k, 3), -1.0, dtype=np.float32)

                for kp in kps:
                    # Skip malformed entries from older-format JSONs
                    if not isinstance(kp, dict):
                        continue
                    if "row" not in kp or "col" not in kp or "image_xy_px" not in kp:
                        continue
                    i = kp["row"] * grid_size + kp["col"]
                    if i < target_k:
                        u, v = kp["image_xy_px"]
                        kp_array[i, 0] = float(u)
                        kp_array[i, 1] = float(v)
                        kp_array[i, 2] = float(kp.get("visible", False))

                self.samples.append({
                    "image_path": img_path,
                    "quad_px":    np.array(quad, dtype=np.float32),
                    "keypoints_array": kp_array,   # replaces "keypoints": kps
                    "grid_size":  grid_size,
                })
        
        self._img_cache: dict[str, np.ndarray] = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s        = self.samples[idx]
        kp_array = s["keypoints_array"]   # (K, 3): [u, v, visible]

        raw = cv2.imread(s["image_path"], cv2.IMREAD_COLOR)
        if raw is None:
            raise FileNotFoundError(f"Could not read image: {s['image_path']}")
        im = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        H, W = im.shape[:2]

        quad = s["quad_px"]   # (4, 2)
        cx = float((quad[:, 0].min() + quad[:, 0].max()) / 2.0)
        cy = float((quad[:, 1].min() + quad[:, 1].max()) / 2.0)

        # Fixed-size ROI top-left corner (may be outside image bounds — that's fine)
        x0 = int(round(cx - self.roi_w / 2.0))
        y0 = int(round(cy - self.roi_h / 2.0))
        x1 = x0 + self.roi_w
        y1 = y0 + self.roi_h

        # Black canvas of exactly the output size
        crop = np.zeros((self.roi_h, self.roi_w, 3), dtype=np.uint8)

        # Intersection of ROI with actual image
        src_x0 = max(0, x0);  src_x1 = min(W, x1)
        src_y0 = max(0, y0);  src_y1 = min(H, y1)

        if src_x1 > src_x0 and src_y1 > src_y0:
            # Where in the canvas does this land
            dst_x0 = src_x0 - x0;  dst_x1 = src_x1 - x0
            dst_y0 = src_y0 - y0;  dst_y1 = src_y1 - y0
            crop[dst_y0:dst_y1, dst_x0:dst_x1] = im[src_y0:src_y1, src_x0:src_x1]

        # Keypoints: just shift by ROI top-left — no scaling at all
        target = np.full((self.target_k * 2,), -1.0, dtype=np.float32)
        for i in range(self.target_k):
            u, v, vis = kp_array[i]
            if vis > 0.5:
                u_roi = float(u) - x0   # pixel shift only, no scale
                v_roi = float(v) - y0
                target[i * 2]     = u_roi
                target[i * 2 + 1] = v_roi

        # Optional augmentation (colour only — no spatial transforms)
        if self.transform is not None:
            try:
                crop = self.transform(image=crop)["image"]
            except Exception as e:
                print(f"[warn] transform failed: {e}")

        if crop.dtype != np.uint8:
            crop = np.clip(crop, 0, 255).astype(np.uint8)

        assert crop.shape == (self.roi_h, self.roi_w, 3), \
            f"Unexpected crop shape {crop.shape} at idx {idx}"

        return crop, target

def worker_init_fn(worker_id):
    cv2.setNumThreads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    config = {
        # ── data ────────────────────────────────────────────────────────────
        # Flat directory produced by homography_markers_data_generation.py,
        # containing images/ and metadata/ subdirectories.
        "data_dir": "/home/anegi/abhay_ws/deep-marker-estimation/data_generation/multi_marker_output/multi_marker_20260218-201012/",
        # Val fraction and seed used to create (or verify) split.json.
        # Once split.json exists these values are ignored — the file takes
        # precedence so restarts always reproduce the same split.
        "val_fraction": 0.1,
        "split_seed":   42,

        # ── model / input ────────────────────────────────────────────────────
        "image_height": 300,
        "image_width":  480,

        # ── optimisation ─────────────────────────────────────────────────────
        "learning_rate":       1e-6,
        "weight_decay":        1e-4,
        "batch_size":          128,
        "num_epochs":          1_000,
        "num_workers":         8,
        "pin_memory":          True,

        # ── checkpoint loading ───────────────────────────────────────────────
        "load_model":           False,
        "load_checkpoint_path": "",
        "num_epoch_dont_save":  0,      # don't overwrite best until this epoch

        # ── checkpointing ────────────────────────────────────────────────────
        "checkpoint_path":             "checkpoints/keypoints.pth.tar",
        "checkpoint_interval_steps":   0,   # 0 → disable step-based saving
        "checkpoint_interval_epochs":  10,

        # ── prediction snapshots (always drawn from val set) ─────────────────
        "prediction_interval_steps":   0,   # 0 → disable step-based snapshots
        "prediction_interval_epochs":  1,
        "num_prediction_images":        1,
        # Also save predicted keypoint coords as a JSON file (logged to wandb)
        "save_keypoint_json":           True,
        "save_dir":                    "saved_predictions",

        # ── wandb ────────────────────────────────────────────────────────────
        "wandb_project":           "keypoint-regression",
        "wandb_run_name":          "mobilenetv3-keypoints",
        "wandb_log_interval_steps": 10,   # 0 → disable step-level loss logging
    }

    os.makedirs(config["save_dir"], exist_ok=True)
    ckpt_parent = os.path.dirname(config["checkpoint_path"])
    if ckpt_parent:
        os.makedirs(ckpt_parent, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        torch.zeros(1, device=device)   # forces CUDA context creation now
        
    wandb.init(
        project=config["wandb_project"],
        name=config["wandb_run_name"],
        config=config,
    )


    # ── train / val split ────────────────────────────────────────────────────
    train_stems, val_stems = resolve_split(
        data_dir=config["data_dir"],
        val_fraction=config["val_fraction"],
        seed=config["split_seed"],
    )
    wandb.log({
        "dataset/n_train_images": len(train_stems),
        "dataset/n_val_images":   len(val_stems),
    })

    # ── transforms ──────────────────────────────────────────────────────────
    train_transform = A.Compose([
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.8),
        A.GaussianBlur(p=0.2),
    ])
    val_transform = None 

    # ── model ───────────────────────────────────────────────────────────────
    model = RegressorMobileNetV3().to(device)
    try:
        K_expected = model.output_layer.out_features // 2
    except Exception:
        K_expected = 81   # default: 9×9 grid

    # ── datasets ────────────────────────────────────────────────────────────
    img_dir  = next(
        (os.path.join(config["data_dir"], d) for d in ("images", "rgb")
         if os.path.isdir(os.path.join(config["data_dir"], d))),
        os.path.join(config["data_dir"], "images"),
    )
    meta_dir = os.path.join(config["data_dir"], "metadata")

    def make_dataset(stems, transform):
        return MarkerKeypointDataset(
            image_dir=img_dir,
            metadata_dir=meta_dir,
            stems=stems,
            transform=transform,
            roi_size=(config["image_height"], config["image_width"]),
            target_k=K_expected,
        )

    train_ds = make_dataset(train_stems, train_transform)
    val_ds   = make_dataset(val_stems,   val_transform)
    print(f"Dataset — train: {len(train_ds):,} samples  val: {len(val_ds):,} samples  K={K_expected}")
    wandb.log({
        "dataset/train_samples": len(train_ds),
        "dataset/val_samples":   len(val_ds),
    })

    train_loader = DataLoader(
        train_ds,
        batch_size=128,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,               # was 4, reduce to cut RAM usage
        multiprocessing_context="spawn",
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=256,
        shuffle=False,
        num_workers=4,                   # val doesn't need as many workers
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        multiprocessing_context="spawn",
        worker_init_fn=worker_init_fn,
    )

    # ── optimiser ───────────────────────────────────────────────────────────
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scaler = torch.amp.GradScaler(device="cuda")

    if config["load_model"] and config["load_checkpoint_path"]:
        load_checkpoint(torch.load(config["load_checkpoint_path"]), model)

    best_val_loss = float("inf")
    global_step   = 0

    # ── training loop ────────────────────────────────────────────────────────
    for epoch in range(config["num_epochs"]):
        train_loss, global_step = train_one_epoch(
            train_loader, model, optimizer, scaler, device,
            global_step=global_step, cfg=config, val_loader=val_loader, epoch=epoch,
        )
        wandb.log(
            {"train/loss_epoch": train_loss, "epoch": epoch, "global_step": global_step},
            step=global_step,
        )

        val_loss = evaluate_loss(val_loader, model, device)
        wandb.log(
            {"val/loss_epoch": val_loss, "epoch": epoch, "global_step": global_step},
            step=global_step,
        )
        print(f"Epoch {epoch+1}/{config['num_epochs']} | "
              f"train {train_loss:.5f} | val {val_loss:.5f}")

        # Save best model
        if val_loss < best_val_loss and epoch >= config["num_epoch_dont_save"]:
            best_val_loss = val_loss
            save_checkpoint(
                {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()},
                config["checkpoint_path"],
            )

        # Periodic epoch-level checkpoint
        if epoch % config.get("checkpoint_interval_epochs", 5) == 0:
            save_checkpoint(
                {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()},
                f"{config['checkpoint_path']}_epoch_{epoch}.pth.tar",
            )

        # Periodic epoch-level prediction snapshot
        if epoch % config.get("prediction_interval_epochs", 5) == 0:
            save_predictions(
                val_loader, model, cfg=config, device=device,
                n=config["num_prediction_images"], step=global_step,
            )


if __name__ == "__main__":
    main()