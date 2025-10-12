import cv2
import numpy as np
from math import gcd
import torch 
from albumentations.pytorch import ToTensorV2
import albumentations as A
import matplotlib.pyplot as plt

from deep_marker_estimation.segmentation_model.model import UNETWithDropout, UNETWithDropoutMini
from deep_marker_estimation.segmentation_model.utils import load_checkpoint

import numpy as np
from math import gcd

import os, threading
import torch

try:
    DEVICE
except NameError:
    DEVICE = "cpu"

_SEG_MODEL_CACHE = {}
_SEG_LOCK = threading.Lock()

def _looks_like_state_dict(obj):
    if not isinstance(obj, dict) or not obj:
        return False
    # keys are strings, values are tensors/arrays
    if not all(isinstance(k, str) for k in obj.keys()):
        return False
    # tolerate both torch Tensors and numpy arrays
    import numpy as _np
    return all(torch.is_tensor(v) or isinstance(v, _np.ndarray) for v in obj.values())


def load_segmentation_model(checkpoint_path: str, device: str = DEVICE,
                            *, in_channels: int = 1, out_channels: int = 1):
    """
    Load exactly once per (abs_path, device, in/out chans).
    Supports:
      • TorchScript (torch.jit.load)
      • pickled nn.Module (torch.save(model))
      • bare state_dict (torch.save(model.state_dict()))
      • dict with key 'state_dict'
    """
    ckpt_abs = os.path.abspath(checkpoint_path)
    key = (ckpt_abs, str(device), int(in_channels), int(out_channels))
    with _SEG_LOCK:
        m = _SEG_CACHE.get(key)
        if m is not None:
            return m

        # 1) TorchScript fast path
        try:
            m = torch.jit.load(ckpt_abs, map_location=device)
            m.eval()
            _SEG_CACHE[key] = m
            print(f"[DME] Seg TorchScript loaded once from: {ckpt_abs}")
            return m
        except Exception:
            pass

        obj = torch.load(ckpt_abs, map_location=device)

        # 2) pickled nn.Module
        if hasattr(obj, "forward"):
            m = obj.to(device).eval()
            for p in m.parameters(): p.requires_grad = False
            _SEG_CACHE[key] = m
            print(f"[DME] Seg nn.Module (pickled) loaded once from: {ckpt_abs}")
            return m

        # 3) state_dict container or raw state_dict
        if isinstance(obj, dict):
            state = obj.get("state_dict", obj) if _looks_like_state_dict(obj) or ("state_dict" in obj) else None
            if state is not None:
                # instantiate your known net here
                # m = UNETWithDropoutMini(in_channels=in_channels, out_channels=out_channels).to(device)
                m = UNETWithDropout(in_channels=3, out_channels=out_channels).to(device)
                # IMPORTANT: load directly; do NOT call your load_checkpoint expecting 'state_dict'
                m.load_state_dict(state, strict=False)
                m.eval()
                for p in m.parameters(): p.requires_grad = False
                _SEG_CACHE[key] = m
                print(f"[DME] Seg state_dict loaded once into UNETWithDropoutMini from: {ckpt_abs}")
                return m

        raise RuntimeError(f"Unsupported segmentation checkpoint format at {ckpt_abs}: {type(obj)}")

# seg_utils.py (top-level near imports)
import os, threading, torch

# Use your existing DEVICE or allow override via config
try:
    DEVICE
except NameError:
    DEVICE = "cpu"

_SEG_CACHE = {}
_SEG_LOCK = threading.Lock()


def split_image_by_aspect_ratio(image, M, N, stride=None):
    """
    Splits an image into overlapping tiles of size P×Q such that:
    - P/Q = M/N (aspect ratio match)
    - P ≥ M, Q ≥ N
    - Entire image is covered by tiles (possibly overlapping)

    Args:
        image (np.ndarray): Input image of shape (H, W, C) or (H, W)
        M (int): Height component of desired aspect ratio
        N (int): Width component of desired aspect ratio
        stride (tuple, optional): (vertical_stride, horizontal_stride). If None, defaults to (P//2, Q//2)

    Returns:
        tiles (List[np.ndarray]): List of image tiles
        coords (List[Tuple[int, int]]): List of (y, x) top-left coordinates in original image
    """
    H, W = image.shape[:2]

    # Simplify aspect ratio
    d = gcd(M, N)
    m, n = M // d, N // d

    # Choose largest P, Q that fit the aspect ratio and image
    scale = min(H // m, W // n)
    P, Q = m * scale, n * scale

    # Define stride
    if stride is None:
        stride_y, stride_x = P // 2, Q // 2
    else:
        stride_y, stride_x = stride

    tiles = []
    coords = []

    # Compute y and x positions to ensure full coverage
    y_positions = list(range(0, H - P + 1, stride_y))
    x_positions = list(range(0, W - Q + 1, stride_x))

    # Add final row/column if needed to reach image edge
    if y_positions[-1] + P < H:
        y_positions.append(H - P)
    if x_positions[-1] + Q < W:
        x_positions.append(W - Q)

    for y in y_positions:
        for x in x_positions:
            tile = image[y:y + P, x:x + Q]
            tiles.append(tile)
            coords.append((y, x))

    return tiles, coords

def combine_tiles_and_coords(tiles, coords, image_shape):
    """
    Combines tiles back into a full image using given coordinates.
    Handles overlapping tiles by averaging pixel values.

    Args:
        tiles (List[np.ndarray]): List of image tiles (H, W) or (H, W, C) or (1, H, W)
        coords (List[Tuple[int, int]]): Top-left (y, x) coordinates for each tile
        image_shape (Tuple[int, int] or Tuple[int, int, int]): Shape of final image (H, W[, C])

    Returns:
        np.ndarray: Combined image
    """
    # Determine if grayscale or color
    is_color = tiles[0].ndim == 3 and tiles[0].shape[-1] in [1, 3]

    # Prepare accumulation and count arrays
    combined_image = np.zeros(image_shape, dtype=np.float32)
    count_image = np.zeros(image_shape, dtype=np.float32)

    for tile, (y, x) in zip(tiles, coords):
        # Convert (1, H, W) → (H, W)
        if tile.ndim == 3 and tile.shape[0] == 1:
            tile = tile[0]  # from (1, H, W) to (H, W)

        h, w = tile.shape[:2]

        # Add tile to combined image and update count for averaging
        combined_image[y:y + h, x:x + w] += tile
        count_image[y:y + h, x:x + w] += 1.0

    # Avoid divide-by-zero
    count_image[count_image == 0] = 1.0
    combined_image = combined_image / count_image

    return combined_image.astype(tiles[0].dtype)

def combine_tiles_and_coords_max(tiles, coords, image_shape):
    """
    Combines tiles back into a full image using the maximum value for overlapping pixels.

    Args:
        tiles (List[np.ndarray]): List of image tiles (H, W) or (H, W, C) or (1, H, W)
        coords (List[Tuple[int, int]]): Top-left (y, x) coordinates for each tile
        image_shape (Tuple[int, int] or Tuple[int, int, int]): Shape of final image (H, W[, C])

    Returns:
        np.ndarray: Combined image
    """
    # Prepare output array
    combined_image = np.zeros(image_shape, dtype=tiles[0].dtype)

    for tile, (y, x) in zip(tiles, coords):
        # Convert (1, H, W) → (H, W)
        if tile.ndim == 3 and tile.shape[0] == 1:
            tile = tile[0]

        h, w = tile.shape[:2]

        # Take elementwise max
        combined_image[y:y + h, x:x + w] = np.maximum(
            combined_image[y:y + h, x:x + w],
            tile
        )

    return combined_image

import torch
import numpy as np
import cv2

def segment_marker(image, config_segmentation, model=None):
    """
    Segments the marker in the input image by tiling, resizing to model input size,
    performing inference, and stitching the segmentation mask back together.

    Args:
        image (np.ndarray): RGB input image (H, W, 3)
        config_segmentation (dict): Configuration dict with keys:
            - 'checkpoint_path': path to model weights
            - 'input_size': (H_input, W_input)
            - 'segmentation_threshold': optional float threshold for binary mask

    Returns:
        np.ndarray: Full-size segmentation mask (H, W)
    """
    if model is None:
        device  = config_segmentation.get("device", DEVICE)
        in_ch   = int(config_segmentation.get("in_channels", 1))
        out_ch  = int(config_segmentation.get("out_channels", 1))
        model   = load_segmentation_model(config_segmentation["checkpoint_path"], device,
                                          in_channels=in_ch, out_channels=out_ch)

    # Load config
    seg_threshold = config_segmentation.get('segmentation_threshold', 0.5)
    input_size = config_segmentation['input_size']

    # Setup model
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    model.eval()

    # Tile original image
    image_tiles, image_tiles_coords = split_image_by_aspect_ratio(
        image, M=input_size[0], N=input_size[1]
    )

    seg_tiles = []

    for tile in image_tiles:
        orig_h, orig_w = tile.shape[:2]

        # Resize to model input size
        tile_resized = cv2.resize(tile, (input_size[1], input_size[0]), interpolation=cv2.INTER_LINEAR)

        # Prepare input tensor
        seg_transform = A.Compose([
            A.Normalize(max_pixel_value=1.0),
            ToTensorV2(),
        ])
        transformed = seg_transform(image=tile_resized)
        tile_tensor = transformed["image"].unsqueeze(0).to(DEVICE)

        # # convert to grayscale # FIXME: uncomment for mini model, comment for regular segmentation model 
        # if tile_tensor.shape[1] == 3:
        #     tile_tensor = tile_tensor.mean(dim=1, keepdim=True)

        # Predict
        with torch.no_grad():
            seg_output = torch.sigmoid(model(tile_tensor))  # shape: (1, 1, H, W)
            seg_mask = seg_output.squeeze().cpu().numpy()  # shape: (H, W)
            torch.cuda.empty_cache()

        # Threshold (optional)
        seg_mask = (seg_mask > seg_threshold).astype(np.uint8)

        # Resize segmentation output back to original tile size
        seg_mask_resized = cv2.resize(seg_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        seg_tiles.append(seg_mask_resized)

    # Stitch back
    seg = combine_tiles_and_coords_max(seg_tiles, image_tiles_coords, image.shape[:2])

    return seg

