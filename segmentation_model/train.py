import torch 
import albumentations as A 
from albumentations.pytorch import ToTensorV2 
from tqdm import tqdm 
import torch.nn as nn 
import torch.optim as optim 
# from segmentation_model.model import UNETWithDropout
from segmentation_model.model import UNETWithDropoutMini
from segmentation_model.utils import (
    load_checkpoint, 
    save_checkpoint, 
    get_loaders, 
    check_accuracy,
    save_predictions_as_imgs,
)
import os 
import wandb
from contextlib import nullcontext
import gc
import time
import traceback

DATA_DIR = "/home/nom4d/deep-marker-estimation/data_generation/multi_marker_augmented_output/multi_marker_augmented_20251021-205104/"
TRAIN_IMG_DIR = f"{DATA_DIR}/train/images"
TRAIN_MASK_DIR = f"{DATA_DIR}/train/segmentations"
VAL_IMG_DIR = f"{DATA_DIR}/val/images"
VAL_MASK_DIR = f"{DATA_DIR}/val/segmentations"
# Save checkpoints under this repo for portability
SAVE_DIR = "/home/nom4d/deep-marker-estimation/segmentation_model/binary_segmentation_checkpoints"
SAVE_FREQ = 1000 
# How often (in batches) to attempt an explicit memory cleanup
MEM_CLEAR_FREQ = 50

LEARNING_RATE = 1e-5 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
BATCH_SIZE = 8
NUM_EPOCHS = 1000
num_epoch_dont_save = 0 
NUM_WORKERS = 8
IMAGE_HEIGHT = 300 
IMAGE_WIDTH = 480 
PIN_MEMORY = False 
LOAD_MODEL = True   
LOAD_CHECKPOINT_PATH = "/home/nom4d/deep-marker-estimation/segmentation_model/binary_segmentation_checkpoints/my_checkpoint_minimodel_epoch_0_batch_0.pth.tar"             

# Binary segmentation configuration (single-logit)
NUM_CLASSES = 1  # out_channels=1 for BCEWithLogitsLoss

# Handling large images: choose one strategy
# If True, randomly crop patches of (IMAGE_HEIGHT, IMAGE_WIDTH) for training.
# If False, resize entire image to (IMAGE_HEIGHT, IMAGE_WIDTH).
USE_RANDOM_CROP = False

# Stability/Performance toggles
USE_AMP = False  # Mixed precision can trigger cuDNN internal errors on some setups
CUDNN_BENCHMARK = False
CUDNN_DETERMINISTIC = True
DISABLE_CUDNN = True  # Disable cuDNN by default to avoid CUDNN_STATUS_INTERNAL_ERROR

# Apply backend settings
try:
    import torch.backends.cudnn as cudnn
    cudnn.enabled = not DISABLE_CUDNN
    cudnn.benchmark = CUDNN_BENCHMARK
    cudnn.deterministic = CUDNN_DETERMINISTIC
    # Disable TF32 paths which sometimes interact poorly on certain drivers
    cudnn.allow_tf32 = False
    try:
        import torch.backends.cuda.matmul as matmul  # type: ignore
        matmul.allow_tf32 = False
    except Exception:
        pass
except Exception as e:
    print(f"[warn] Failed to set cuDNN backend settings: {e}")
    pass

def check_accuracy_binary(loader, model, device="cpu"):
    """Compute pixel accuracy and Dice for binary segmentation.
    Thresholds sigmoid(model(x)) at 0.5.
    """
    model.eval()
    num_correct = 0
    num_pixels = 0
    dice_sum = 0.0
    batches = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            # y expected (N, H, W) or (N,1,H,W), values in {0,1}
            if y.ndim == 3:
                y = y.unsqueeze(1)
            y = y.float().to(device)

            preds = model(x)
            preds = torch.sigmoid(preds)
            preds_bin = (preds > 0.5).float()

            num_correct += (preds_bin == y).sum().item()
            num_pixels += torch.numel(preds_bin)

            # Dice coefficient per batch
            intersection = (preds_bin * y).sum(dim=(1, 2, 3))
            union = preds_bin.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
            dice = (2 * intersection + 1e-8) / (union + 1e-8)
            dice_sum += dice.mean().item()
            batches += 1

    model.train()
    acc = num_correct / max(1, num_pixels)
    mean_dice = dice_sum / max(1, batches)
    print(f"Accuracy: {acc*100:.2f}%, Mean Dice: {mean_dice:.4f}")
    return acc


def _to_cpu(obj):
    """Recursively move any torch.Tensor in a nested structure to CPU.
    Handles dict, list, tuple and torch.Tensor. Leaves other objects alone.
    """
    if torch.is_tensor(obj):
        try:
            return obj.cpu()
        except Exception:
            return obj
    elif isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_to_cpu(v) for v in obj)
    else:
        return obj


def safe_save_checkpoint(state, filename, max_retries=2, backoff_s=1.0):
    """Save checkpoint with tensors moved to CPU first. If the target
    directory (possibly remote) is unavailable, fall back to a local
    repo-relative directory and save there. Returns the path where the
    checkpoint was actually written.
    """
    # Prepare CPU-only checkpoint
    cpu_state = {}
    try:
        if "state_dict" in state:
            cpu_state["state_dict"] = _to_cpu(state["state_dict"])
        if "optimizer" in state:
            cpu_state["optimizer"] = _to_cpu(state["optimizer"])
        # Copy other keys as-is (they are usually small metadata)
        for k, v in state.items():
            if k in ("state_dict", "optimizer"):
                continue
            cpu_state[k] = v
    except Exception:
        # Fall back: attempt to save the original state if conversion fails
        cpu_state = state

    dest_dir = os.path.dirname(filename)
    basename = os.path.basename(filename)

    # Try primary target first with retries
    for attempt in range(max_retries + 1):
        try:
            if dest_dir:
                os.makedirs(dest_dir, exist_ok=True)
            torch.save(cpu_state, filename)
            print(f"=> Saved checkpoint to: {filename}")
            return filename
        except Exception as e:
            print(f"[warn] Failed to save checkpoint to {filename} (attempt {attempt}): {e}")
            traceback.print_exc()
            # Backoff before retrying
            time.sleep(backoff_s * (2 ** attempt))

    # If we get here, primary save failed. Save to a local fallback dir.
    fallback_dir = os.path.join(os.path.dirname(__file__), "local_checkpoints")
    try:
        os.makedirs(fallback_dir, exist_ok=True)
    except Exception as e:
        print(f"[error] Unable to create local fallback checkpoint dir {fallback_dir}: {e}")
        traceback.print_exc()
        # Last resort: try saving to current working directory
        fallback_dir = os.getcwd()

    fallback_path = os.path.join(fallback_dir, basename)
    try:
        torch.save(cpu_state, fallback_path)
        print(f"=> Saved checkpoint to fallback location: {fallback_path}")
        return fallback_path
    except Exception as e:
        print(f"[error] Failed to save checkpoint to fallback location {fallback_path}: {e}")
        traceback.print_exc()
        # Re-raise to let caller know saving failed completely
        raise

def train_fn(loader, model, optimizer, loss_fn, scaler, epoch): 
    loop = tqdm(loader) # progress bar 
    epoch_loss = 0

    for batch_idx, (data, targets) in enumerate(loop): 
        data = data.to(device=DEVICE) 
        # targets expected shape (N, 1, H, W) for BCE; if (N,H,W), unsqueeze
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float().to(device=DEVICE)

        # forward 
        if USE_AMP and DEVICE == "cuda":
            ac_ctx = torch.amp.autocast(device_type="cuda")
        else:
            ac_ctx = nullcontext()
        with ac_ctx:
            predictions = model(data) 
            loss = loss_fn(predictions, targets) 

        # backward 
        optimizer.zero_grad() 
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # accumulate loss
        epoch_loss += loss.item()

        # Log batch loss to wandb
        wandb.log({"batch_loss": loss.item(), "epoch": epoch, "batch_idx": batch_idx})

        # update tqdm loop 
        loop.set_postfix(loss=loss.item())         

        # Save checkpoint every SAVE_FREQ batches
        if batch_idx % SAVE_FREQ == 0: 
            # Attempt to save a CPU-only checkpoint to the configured SAVE_DIR.
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "batch_idx": batch_idx,
            }
            target_path = os.path.join(SAVE_DIR, f"my_checkpoint_minimodel_epoch_{epoch}_batch_{batch_idx}.pth.tar")
            try:
                saved_path = safe_save_checkpoint(checkpoint, target_path)
            except Exception:
                print("[error] checkpoint save failed completely")
                saved_path = None
            # Free any transient GPU memory that may have been used during
            # checkpoint creation/serialization. This is a best-effort call
            # and safe to run on CPU-only systems.
            try:
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
            except Exception:
                pass

        # Free local references to large tensors to make them collectible
        # by CPython GC as soon as possible.
        try:
            del predictions
        except Exception:
            pass
        try:
            del loss
        except Exception:
            pass

        # Periodic forced GC + CUDA cache cleanup to avoid gradual memory growth.
        # Run every MEM_CLEAR_FREQ batches (skip the first batch to avoid
        # excessively frequent cleanup at startup).
        if (MEM_CLEAR_FREQ is not None) and (MEM_CLEAR_FREQ > 0) and (batch_idx % MEM_CLEAR_FREQ == 0) and (batch_idx != 0):
            try:
                gc.collect()
            except Exception:
                pass
            try:
                if DEVICE == "cuda":
                    # Best-effort: free cached device memory
                    torch.cuda.empty_cache()
                    # Free any CUDA IPC resources if the function exists
                    ipc_collect = getattr(torch.cuda, "ipc_collect", None)
                    if callable(ipc_collect):
                        try:
                            ipc_collect()
                        except Exception:
                            pass
            except Exception:
                pass

    # Log average training loss to wandb
    avg_loss = epoch_loss / len(loader)
    wandb.log({"train_loss": avg_loss, "epoch": epoch})

def main(): 
    train_transforms_list = []
    if USE_RANDOM_CROP:
        # Pad small images then crop a fixed-size patch for training
        train_transforms_list.extend([
            A.PadIfNeeded(min_height=IMAGE_HEIGHT, min_width=IMAGE_WIDTH),
            A.RandomCrop(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        ])
    else:
        # Uniformly resize to a manageable size
        train_transforms_list.append(
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
        )

    # Optional geometric augs could be re-enabled if desired
    # train_transforms_list.extend([
    #     A.Rotate(limit=15, p=0.25),
    #     A.HorizontalFlip(p=0.5),
    # ])

    train_transforms_list.extend([

        # randomly change brightness and contrast
        A.RandomBrightnessContrast(
            brightness_limit=0.5,   # +/- 20% brightness
            contrast_limit=0.5,     # +/- 20% contrast
            p=1.0                   # 50% probability
        ),
        
        # simulate highlight/shadow shifts
        A.RandomToneCurve(
            scale=0.5,  # how strong the tone curve deformation is
            p=1.0
        ),

        # optional: more stylized global brightness change
        A.CLAHE(
            clip_limit=(1, 4), 
            tile_grid_size=(8, 8), 
            p=1.0
        ),

        # finally normalize and convert to tensor
        A.Normalize(max_pixel_value=255.0, normalization="standard"),
        ToTensorV2(),
        
    ])

    train_transform = A.Compose(train_transforms_list)

    # For validation, use deterministic resizing for consistent metrics
    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH), 
            
            # randomly change brightness and contrast
            A.RandomBrightnessContrast(
                brightness_limit=0.5,   # +/- 20% brightness
                contrast_limit=0.5,     # +/- 20% contrast
                p=1.0                   # 50% probability
            ),
            
            # simulate highlight/shadow shifts
            A.RandomToneCurve(
                scale=0.5,  # how strong the tone curve deformation is
                p=1.0
            ),

            # optional: more stylized global brightness change
            A.CLAHE(
                clip_limit=(1, 4), 
                tile_grid_size=(8, 8), 
                p=1.0
            ),

            # finally normalize and convert to tensor
            A.Normalize(max_pixel_value=255.0, normalization="standard"),
            ToTensorV2(),
        ]
    )

    # Build loaders (label mapping from utils is ignored for binary)
    train_loader, val_loader, *_ = get_loaders(
        TRAIN_IMG_DIR, 
        TRAIN_MASK_DIR,
        VAL_IMG_DIR, 
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY
    )

    # Single-logit binary segmentation model
    model = UNETWithDropoutMini(in_channels=1, out_channels=1).to(DEVICE) 
    
    # For binary segmentation, use BCEWithLogitsLoss
    loss_fn = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if LOAD_MODEL: 
        if LOAD_CHECKPOINT_PATH is not None:
            # Load checkpoint onto CPU first to avoid transiently inflating GPU memory
            # if the checkpoint contains CUDA tensors. Loading to CPU and then
            # calling load_state_dict will copy parameters to the model device.
            chk_path = LOAD_CHECKPOINT_PATH
            if not os.path.isabs(chk_path):
                chk_path = os.path.join(chk_path)
            checkpoint = torch.load(chk_path, map_location="cpu")
            load_checkpoint(checkpoint, model)
        accuracy = 0.0
    else: 
        accuracy = 0.0 

    # check_accuracy(val_loader, model, device=DEVICE) 

    scaler = torch.amp.GradScaler(enabled=USE_AMP and DEVICE == "cuda")
    for epoch in range(NUM_EPOCHS): 
        train_fn(train_loader, model, optimizer, loss_fn, scaler, epoch)

        # check accuracy (binary)
        new_accuracy = check_accuracy_binary(val_loader, model, device=DEVICE)

        # Log accuracy to wandb
        wandb.log({"val_accuracy": new_accuracy, "epoch": epoch})

        # Run GC and release any cached CUDA memory after validation to avoid
        # gradual memory growth across epochs.
        try:
            gc.collect()
        except Exception:
            pass
        try:
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

        if epoch == 0: 
            accuracy = new_accuracy

        if new_accuracy > accuracy and epoch > num_epoch_dont_save: 
            accuracy = new_accuracy 
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "val_accuracy": new_accuracy,
            }
            target_path = os.path.join(SAVE_DIR, f"my_checkpoint_minimodel_epoch_{epoch}.pth.tar")
            try:
                saved_path = safe_save_checkpoint(checkpoint, target_path)
            except Exception:
                print("[error] epoch checkpoint save failed completely")
                saved_path = None

            # Optionally save some predictions
            saved_images_dir = os.path.join(os.path.dirname(__file__), "sample_preds")
            os.makedirs(saved_images_dir, exist_ok=True)
            try:
                save_predictions_as_imgs(
                    val_loader, model, folder=saved_images_dir, device=DEVICE, num_datapoints=10
                )
            except Exception as e:
                print(f"[warn] save_predictions_as_imgs failed: {e}")
                traceback.print_exc()

if __name__ == "__main__":
    # Initialize wandb
    wandb.init(
        config={
            "wandb_key": "9336a0a286df1f392970fb1192519ef0191ba865",
            "wandb_project": "segmentation_mini_model", 
            "wandb_entity": "abhay-negi-usc-university-of-southern-california", 
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": NUM_EPOCHS,
            "image_height": IMAGE_HEIGHT,
            "image_width": IMAGE_WIDTH,
            "num_workers": NUM_WORKERS,
            "pin_memory": PIN_MEMORY,
            "train_img_dir": TRAIN_IMG_DIR,
            "train_mask_dir": TRAIN_MASK_DIR,
            "val_img_dir": VAL_IMG_DIR,
            "val_mask_dir": VAL_MASK_DIR,
            "num_classes": "binary(1-logit)",
            "use_random_crop": USE_RANDOM_CROP,
        }
    )

    main()
