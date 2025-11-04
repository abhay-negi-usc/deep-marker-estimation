import os 
from PIL import Image 
from torch.utils.data import Dataset 
import numpy as np 
from torchvision import transforms
from os.path import splitext, isfile, join
import torch
import re
try:
    import cv2
    # Avoid OpenCV multi-threading inside DataLoader workers which can segfault
    cv2.setNumThreads(0)
except Exception:
    cv2 = None

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

convert_tensor = transforms.ToTensor()

def load_image(filename):
    ext = splitext(filename)[1]
    # import pdb; pdb.set_trace()
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def _list_files_with_ext(path, exts):
    return [f for f in os.listdir(path) if isfile(join(path, f)) and f.lower().endswith(exts)]


def _mask_candidates_for_image(stem):
    """Generate plausible mask filename stems for a given image stem.
    Examples:
    - img_0001 -> seg_0001
    - image_0001 -> segmentation_0001
    - frame_0001_rgb -> frame_0001
    - foo_0001_0 -> foo_0001
    Returns a list of stems (no extension).
    """
    candidates = []

    def add(s):
        if s and s not in candidates:
            candidates.append(s)

    add(stem)

    # Common token swaps
    swaps = [
        ("img", "seg"),
        ("image", "segmentation"),
        ("rgb", "seg"),
    ]
    for src, dst in swaps:
        if src in stem:
            add(stem.replace(src, dst))

    # Remove common trailing channel/frame suffixes
    suffixes = ["_rgb", "_image", "_img", "_color", "_0", "_1"]
    for suf in suffixes:
        if stem.endswith(suf):
            add(stem[: -len(suf)])

    # If starts with img_ or image_, try seg_ or segmentation_
    if stem.startswith("img_"):
        add("seg_" + stem[len("img_"):])
    if stem.startswith("image_"):
        add("segmentation_" + stem[len("image_"):])

    # Also try prefixing with seg_
    add("seg_" + stem)

    return candidates


def _build_mask_index(mask_dir):
    files = _list_files_with_ext(mask_dir, IMAGE_EXTS)
    stems = {os.path.splitext(f)[0] for f in files}
    return stems


def _resolve_mask_path(mask_dir, img_filename, mask_stem_index):
    stem, _ = os.path.splitext(img_filename)
    candidates = _mask_candidates_for_image(stem)

    # 1) Direct stem match
    for c in candidates:
        if c in mask_stem_index:
            # choose an extension preferring .png
            for ext in (".png", ".jpg", ".jpeg", ".tif", ".bmp"):
                p = join(mask_dir, c + ext)
                if isfile(p):
                    return p

    # 2) Fallback: numeric id match
    nums = re.findall(r"\d+", stem)
    if nums:
        key = nums[-1]
        for s in mask_stem_index:
            if s.endswith("_" + key) or re.search(rf"(^|[^0-9]){key}($|[^0-9])", s):
                for ext in (".png", ".jpg", ".jpeg", ".tif", ".bmp"):
                    p = join(mask_dir, s + ext)
                    if isfile(p):
                        return p

    return None

class MarkersDatasetGrayscale(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, label_to_index=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        all_images = _list_files_with_ext(image_dir, IMAGE_EXTS)
        self._mask_stems = _build_mask_index(mask_dir)
        # For binary segmentation we collapse all non-zero labels to 1
        self.label_to_index = label_to_index  # kept for compatibility, not used in binary path
        self.index_to_label = None
        self.num_classes = 2

        self.images = []
        self._mask_paths = {}
        for f in sorted(all_images):
            mp = _resolve_mask_path(mask_dir, f, self._mask_stems)
            if mp is not None:
                self.images.append(f)
                self._mask_paths[f] = mp
        if len(self.images) == 0:
            raise RuntimeError(f"No image/mask pairs found under {image_dir} and {mask_dir}")

        # Keep simple binary meta for compatibility
        self.index_to_label = [0, 1]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_filename = self.images[index]
        img_path = os.path.join(self.image_dir, img_filename)
        mask_path = self._mask_paths[img_filename]

        # Load image (grayscale) as numpy array (0..255)
        image = np.array(load_image(img_path).convert("L"))

        # Load mask and convert to binary: foreground = 1 where mask > 0
        mask_raw = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        mask = (mask_raw > 0).astype(np.float32)

        # Apply transforms if specified (Albumentations will keep mask shape HxW)
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        # Ensure mask is a tensor of dtype float with shape (H, W)
        if isinstance(mask, np.ndarray):
            # If transforms didn't convert to tensor
            mask = torch.from_numpy(mask.astype(np.float32))
        else:
            # Albumentations ToTensorV2 may return mask as float; ensure float
            mask = mask.float()

        return image, mask


class MarkersDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, label_to_index=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # For binary segmentation we collapse all non-zero labels to 1
        self.label_to_index = label_to_index  # kept for compatibility, not used in binary path
        self.index_to_label = [0, 1]
        self.num_classes = 2
        all_images = _list_files_with_ext(image_dir, IMAGE_EXTS)
        self._mask_stems = _build_mask_index(mask_dir)

        self.images = []
        self._mask_paths = {}
        for f in sorted(all_images):
            mp = _resolve_mask_path(mask_dir, f, self._mask_stems)
            if mp is not None:
                self.images.append(f)
                self._mask_paths[f] = mp
        if len(self.images) == 0:
            raise RuntimeError(f"No image/mask pairs found under {image_dir} and {mask_dir}")

        # No need to build multi-class mapping for binary use-case

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_filename = self.images[index]
        img_path = os.path.join(self.image_dir, img_filename)
        mask_path = self._mask_paths[img_filename]

        # Load image and convert to numpy array
        image = np.array(Image.open(img_path).convert("RGB"))

        # Load mask and convert to binary: foreground = 1 where mask > 0
        mask_raw = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        mask = (mask_raw > 0).astype(np.float32)

        # Apply transforms if specified
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        # Ensure mask is torch.float (H, W)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask.astype(np.float32))
        else:
            mask = mask.float()

        return image, mask


