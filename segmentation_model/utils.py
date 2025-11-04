import torch
import torchvision
from .dataset import MarkersDatasetGrayscale
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def _collect_unique_labels(mask_dir, max_files=400):
    import os
    import numpy as np
    from PIL import Image
    from os.path import isfile, join
    IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    uniques = set()
    count = 0
    for name in os.listdir(mask_dir):
        if not name.lower().endswith(IMAGE_EXTS):
            continue
        p = join(mask_dir, name)
        if not isfile(p):
            continue
        arr = np.array(Image.open(p).convert("L"))
        vals = np.unique(arr)
        uniques.update(map(int, vals))
        count += 1
        if count >= max_files:
            break
    return sorted(uniques)


def _build_label_to_index(train_maskdir, val_maskdir, max_files_each=400):
    uniques = set(_collect_unique_labels(train_maskdir, max_files_each))
    uniques.update(_collect_unique_labels(val_maskdir, max_files_each))
    uniques = sorted(uniques)
    label_to_index = {}
    next_idx = 0
    if 0 in uniques:
        label_to_index[0] = 0
        next_idx = 1
    for v in uniques:
        if v == 0 and 0 in label_to_index:
            continue
        label_to_index[v] = next_idx
        next_idx += 1
    return label_to_index


def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    # Build a consistent label mapping across train and val
    label_to_index = _build_label_to_index(train_maskdir, val_maskdir)

    train_ds = MarkersDatasetGrayscale(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
        label_to_index=label_to_index,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = MarkersDatasetGrayscale(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
        label_to_index=label_to_index,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    # Share a consistent number of classes
    num_classes = len(label_to_index)
    return train_loader, val_loader, num_classes, label_to_index

def check_accuracy(loader, model, device="cuda"):
    """Compute per-pixel accuracy and mean Dice score.
    Supports binary (1-channel logits) and multi-class (C-channel logits).
    """
    num_correct = 0
    num_pixels = 0
    dice_sum = 0.0
    model.eval()

    def compute_mean_dice(pred_labels: torch.Tensor, true_labels: torch.Tensor) -> torch.Tensor:
        # pred_labels, true_labels: (N, H, W) long
        with torch.no_grad():
            max_label = int(torch.max(torch.stack([pred_labels.max(), true_labels.max()])).item())
            # Exclude background=0
            classes = list(range(1, max_label + 1)) if max_label >= 1 else []
            if not classes:
                return torch.tensor(1.0, device=pred_labels.device)
            dices = []
            for c in classes:
                pred_c = (pred_labels == c).float()
                true_c = (true_labels == c).float()
                intersect = (pred_c * true_c).sum()
                denom = pred_c.sum() + true_c.sum()
                dice_c = (2.0 * intersect) / (denom + 1e-8)
                dices.append(dice_c)
            return torch.stack(dices).mean()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).long()  # (N, H, W)

            logits = model(x)
            if logits.ndim == 4 and logits.shape[1] > 1:
                # Multi-class: argmax over channels
                preds = torch.argmax(logits, dim=1)  # (N, H, W)
            else:
                # Binary: threshold sigmoid and squeeze channel
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long().squeeze(1)  # (N, H, W)

            num_correct += (preds == y).sum()
            num_pixels += y.numel()
            dice_sum += compute_mean_dice(preds, y)

    accuracy = num_correct.float() / float(num_pixels)
    print(f"Got {num_correct}/{num_pixels} with acc {accuracy*100:.2f}")
    mean_dice = (dice_sum/len(loader)).item() if hasattr(dice_sum, "item") else float(dice_sum/len(loader))
    print(f"Mean Dice (no background): {mean_dice:.4f}")
    model.train()
    return accuracy 

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda", num_datapoints=None
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        if num_datapoints is not None and idx >= num_datapoints:
            break

        # Move input to device for inference, but ensure images saved to disk are moved
        # back to CPU and detached to avoid holding GPU memory/larger tensors.
        x = x.to(device=device)
        with torch.no_grad():
            logits = model(x)
            if logits.ndim == 4 and logits.shape[1] > 1:
                pred_labels = torch.argmax(logits, dim=1).float()  # (N, H, W)
                # Normalize for visualization (grayscale by class id)
                max_label = max(1, int(pred_labels.max().item()))
                preds_vis = (pred_labels / float(max_label)).unsqueeze(1)
            else:
                preds_vis = (torch.sigmoid(logits) > 0.5).float()  # (N,1,H,W)

        # Detach and move to CPU before saving to reduce GPU memory pressure.
        try:
            preds_vis_cpu = preds_vis.detach().cpu()
        except Exception:
            preds_vis_cpu = preds_vis.cpu()
        try:
            x_cpu = x.detach().cpu()
        except Exception:
            x_cpu = x.cpu()

        torchvision.utils.save_image(preds_vis_cpu, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(x_cpu, f"{folder}/rgb_{idx}.png", normalize=True)
        # Save GT visualization similarly (scale labels if multi-class)
        if y.ndim == 3:  # (N,H,W)
            max_label_y = max(1, int(y.max().item()))
            y_vis = (y.float() / float(max_label_y)).unsqueeze(1)
        else:  # (N,1,H,W) or already float
            y_vis = y

        # Ensure ground-truth visualization is on CPU
        try:
            y_vis_cpu = y_vis.detach().cpu()
        except Exception:
            y_vis_cpu = y_vis.cpu()
        torchvision.utils.save_image(y_vis_cpu, f"{folder}/seg_{idx}.png", normalize=True)

    model.train()
