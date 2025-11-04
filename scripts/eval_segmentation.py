import os
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import colorsys

# Local imports
from segmentation_model.model import UNETWithDropoutMini
from segmentation_model.utils import get_loaders, load_checkpoint
from PIL import Image as PILImage
import tempfile


def _latest_checkpoint(ckpt_dir: str) -> str | None:
    if not os.path.isdir(ckpt_dir):
        return None
    files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth.tar')]
    if not files:
        return None
    files = sorted(files)
    return os.path.join(ckpt_dir, files[-1])


def _infer_num_classes_from_ckpt(state_dict: dict) -> int | None:
    # Try to read the final conv weight shape
    for k, v in state_dict.items():
        if k.endswith('final_conv.weight') and isinstance(v, torch.Tensor):
            return int(v.shape[0])
    return None


def _to_vis_rgb(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert CHW tensor in [0,1] to uint8 HWC RGB for saving."""
    if img_tensor.ndim == 3 and img_tensor.shape[0] == 1:
        img_tensor = img_tensor.repeat(3, 1, 1)
    img_clamped = img_tensor.clamp(0, 1)
    img_np = (img_clamped.cpu().numpy() * 255.0).astype(np.uint8)
    img_np = np.transpose(img_np, (1, 2, 0))  # HWC
    return img_np


def _label_to_vis_gray(label_tensor: torch.Tensor) -> np.ndarray:
    """Convert (H,W) long tensor of class indices to uint8 grayscale visualization."""
    labels = label_tensor.cpu().numpy()
    max_label = int(labels.max()) if labels.size > 0 else 1
    max_label = max(1, max_label)
    vis = (labels.astype(np.float32) / float(max_label))
    vis = (vis * 255.0).astype(np.uint8)
    vis = np.stack([vis, vis, vis], axis=-1)  # to 3-channel for easy concat
    return vis


def _make_palette(num_classes: int) -> np.ndarray:
    """Create a simple color palette (num_classes x 3) in uint8 RGB.
    Class 0 (background) is black; others are distinct hues.
    """
    num_classes = max(int(num_classes), 1)
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    if num_classes <= 1:
        palette[0] = (0, 0, 0)
        return palette
    for c in range(1, num_classes):
        h = (c - 1) / max(1, num_classes - 1)
        r, g, b = colorsys.hsv_to_rgb(h, 0.9, 1.0)
        palette[c] = (int(r * 255), int(g * 255), int(b * 255))
    return palette


def _label_to_color(label_tensor: torch.Tensor, palette: np.ndarray) -> np.ndarray:
    """Map (H,W) long tensor of class indices to color image using palette.
    palette: (C,3) uint8 array where C >= max(label)+1.
    Returns HxWx3 uint8.
    """
    labels = label_tensor.cpu().numpy().astype(np.int64)
    max_idx = int(labels.max()) if labels.size > 0 else 0
    if max_idx >= len(palette):
        # Extend palette if needed
        extra = _make_palette(max_idx + 1)
        palette = extra
    color = palette[labels]
    return color


def _hstack_three(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Horizontally stack three HxWx3 arrays (pads if necessary)."""
    h = max(a.shape[0], b.shape[0], c.shape[0])
    w_a, w_b, w_c = a.shape[1], b.shape[1], c.shape[1]

    def pad_to_h(img: np.ndarray) -> np.ndarray:
        if img.shape[0] == h:
            return img
        pad_h = h - img.shape[0]
        return np.pad(img, ((0, pad_h), (0, 0), (0, 0)), mode='constant', constant_values=0)

    a, b, c = pad_to_h(a), pad_to_h(b), pad_to_h(c)
    return np.concatenate([a, b, c], axis=1)


def build_val_transform(img_h: int, img_w: int) -> A.Compose:
    return A.Compose([
        A.Resize(height=img_h, width=img_w),
        A.Normalize(max_pixel_value=255.0),
        ToTensorV2(),
    ])


def evaluate(
    data_dir: str,
    checkpoint_path: str | None,
    out_dir: str,
    num_samples: int,
    image_height: int,
    image_width: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    device: str,
):
    train_img = os.path.join(data_dir, 'train', 'images')
    train_seg = os.path.join(data_dir, 'train', 'segmentations')
    val_img = os.path.join(data_dir, 'val', 'images')
    val_seg = os.path.join(data_dir, 'val', 'segmentations')

    os.makedirs(out_dir, exist_ok=True)

    val_transform = build_val_transform(image_height, image_width)

    # Build loaders (and consistent label mapping)
    # Ensure mask directories exist and contain a mask for each image. If masks
    # are missing or the folder is empty, create zero-valued masks so the
    # dataset yields a segmentation of all-zeros (background).
    def _ensure_zero_masks(mask_dir: str, image_dir: str):
        """Create mask_dir if missing and add a zero mask for every image in
        image_dir that does not already have a corresponding mask.
        Masks are created with the same stem as the image and saved as PNG.
        """
        IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        os.makedirs(mask_dir, exist_ok=True)
        # List images
        imgs = [f for f in os.listdir(image_dir) if f.lower().endswith(IMAGE_EXTS) and os.path.isfile(os.path.join(image_dir, f))]
        for img_name in imgs:
            stem = os.path.splitext(img_name)[0]
            # Prefer .png mask
            mask_path = os.path.join(mask_dir, stem + '.png')
            if os.path.exists(mask_path):
                continue
            # Create zero mask with same size as image (grayscale)
            try:
                img = PILImage.open(os.path.join(image_dir, img_name)).convert('L')
                size = img.size  # (W, H)
            except Exception:
                # Fallback to small default size if image can't be opened
                size = (64, 64)
            zero = PILImage.new('L', size, 0)
            try:
                zero.save(mask_path)
            except Exception:
                # If saving to the requested mask_dir fails (permissions),
                # create a temporary masks dir and populate there, then return
                # that path so callers use the temp dir instead.
                tmp = tempfile.mkdtemp(prefix='dummy_masks_')
                for img_name2 in imgs:
                    stem2 = os.path.splitext(img_name2)[0]
                    p2 = os.path.join(tmp, stem2 + '.png')
                    try:
                        img2 = PILImage.open(os.path.join(image_dir, img_name2)).convert('L')
                        size2 = img2.size
                    except Exception:
                        size2 = (64, 64)
                    PILImage.new('L', size2, 0).save(p2)
                return tmp
        return mask_dir

    # Possibly replace train_seg/val_seg with directories populated with zero masks
    train_seg = _ensure_zero_masks(train_seg, train_img) if os.path.isdir(train_img) else train_seg
    val_seg = _ensure_zero_masks(val_seg, val_img) if os.path.isdir(val_img) else val_seg

    train_loader, val_loader, derived_num_classes, label_to_index = get_loaders(
        train_img,
        train_seg,
        val_img,
        val_seg,
        batch_size,
        val_transform,
        val_transform,
        num_workers,
        pin_memory,
    )

    # Decide checkpoint
    if checkpoint_path is None:
        # Try latest in default checkpoints dir near project root
        default_ckpt_dir = os.path.join(os.path.dirname(__file__), '..', 'segmentation_model', 'checkpoints')
        checkpoint_path = _latest_checkpoint(default_ckpt_dir)
    if checkpoint_path is None or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Prepare model
    # Infer classes from checkpoint if possible
    state = torch.load(checkpoint_path, map_location='cpu')
    sd = state.get('state_dict', state)
    ckpt_num_classes = _infer_num_classes_from_ckpt(sd)
    num_classes = ckpt_num_classes or derived_num_classes

    model = UNETWithDropoutMini(in_channels=1, out_channels=num_classes).to(device)
    load_checkpoint(state, model)
    model.eval()

    # Build color palette for visualization
    palette = _make_palette(num_classes)

    saved = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(val_loader):
            x = x.to(device)
            y = y.long().to(device)  # (N,H,W)
            logits = model(x)
            if logits.shape[1] > 1:
                pred = torch.argmax(logits, dim=1)  # (N,H,W)
            else:
                pred = (torch.sigmoid(logits).squeeze(1) > 0.1).long()

            # Build summary per sample
            n = x.shape[0]
            for i in range(n):
                if saved >= num_samples:
                    return
                img_vis = _to_vis_rgb(x[i].cpu())
                pred_vis = _label_to_color(pred[i], palette)
                gt_vis = _label_to_color(y[i], palette)
                summary = _hstack_three(img_vis, pred_vis, gt_vis)
                out_path = os.path.join(out_dir, f'sample_{saved:04d}.png')
                Image.fromarray(summary).save(out_path)
                saved += 1


def main():
    # Config dictionary (modify values here as needed)
    CONFIG = {
        # "data_dir": "/home/nom4d/deep-marker-estimation/data_generation/multi_marker_augmented_output/multi_marker_augmented_20251021-205104/",
        "data_dir": "/home/anegi/Downloads/EE_cam_images-20251028T002152Z-1-001/EE_cam_images/",
                # "data_dir": "/home/anegi/Downloads/multi_tag_samples-20251029T034204Z-1-001/multi_tag_samples/",
        "checkpoint": "/home/nom4d/deep-marker-estimation/segmentation_model/binary_segmentation_checkpoints/my_checkpoint_minimodel_epoch_0_batch_0.pth.tar",  # Path to .pth.tar; if None, uses latest from default checkpoints dir
        "out_dir": "/home/nom4d/deep-marker-estimation/segmentation_model/eval_outputs_2/",
        "num": 77,  # number of samples to save
        "img_h": 300,
        "img_w": 480,
        "batch_size": 4,
        "num_workers": 0,
        "pin_memory": False,
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Safer defaults to avoid cuDNN issues
    try:
        import torch.backends.cudnn as cudnn
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True
        cudnn.allow_tf32 = False
    except Exception:
        pass

    evaluate(
        data_dir=CONFIG["data_dir"],
        checkpoint_path=CONFIG["checkpoint"],
        out_dir=CONFIG["out_dir"],
        num_samples=CONFIG["num"],
        image_height=CONFIG["img_h"],
        image_width=CONFIG["img_w"],
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
        device=device,
    )


if __name__ == '__main__':
    main()
