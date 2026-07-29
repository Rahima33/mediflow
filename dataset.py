"""
MediFlowXrayDataset: a custom PyTorch Dataset replacing torchvision's
ImageFolder, so that lung-field cropping and CLAHE can run on every
image *before* the standard resize/augment/normalize/tensor pipeline.

Why ImageFolder can't do this: ImageFolder only knows how to load an
image and apply a single torchvision `transform`. It has no hook for
calling custom OpenCV functions (lung cropping, CLAHE) as an
intermediate step. This class re-implements ImageFolder's folder-scanning
behavior, then inserts the custom preprocessing before handing the result
off to a normal torchvision transform pipeline -- so your existing
train_transform / val_transform (Resize, RandomRotation, ColorJitter,
ToTensor, Normalize) still work unchanged as the `transform` argument.

Usage is designed to be a drop-in replacement:

    # Before:
    full_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)

    # After:
    full_dataset = MediFlowXrayDataset(root=TRAIN_DIR, transform=train_transform)

Everything downstream (random_split, DataLoader, WeightedRandomSampler,
.samples, .classes, .class_to_idx) continues to work the same way, since
this class provides the same attributes ImageFolder does.
"""

import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from preprocessing.lung import crop_to_lung_fields
from preprocessing.clahe import apply_clahe

def _make_samples(root_dir):
    """
    Recreate ImageFolder's folder-scanning behavior: assumes a structure
    like root_dir/NORMAL/*.jpg, root_dir/PNEUMONIA/*.jpg, and assigns
    integer labels alphabetically by subfolder name.

    Returns:
        samples: list of (filepath, label) tuples
        classes: sorted list of class names, e.g. ["NORMAL", "PNEUMONIA"]
        class_to_idx: dict mapping class name -> integer label
    """
    classes = sorted(
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    )
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    valid_extensions = (".jpg", ".jpeg", ".png")
    samples = []
    for cls_name in classes:
        cls_dir = os.path.join(root_dir, cls_name)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(valid_extensions):
                samples.append((os.path.join(cls_dir, fname), class_to_idx[cls_name]))

    return samples, classes, class_to_idx


class MediFlowXrayDataset(Dataset):
    def __init__(self, root, transform=None, use_lung_crop=True, use_clahe=True,
                 cache_dir=None, cache_size=256):
        """
        Args:
            root: path to a folder containing one subfolder per class
                  (e.g. "train/NORMAL", "train/PNEUMONIA")
            transform: a torchvision transform (Resize, ToTensor,
                       Normalize, etc.) applied AFTER lung-cropping and
                       CLAHE. Same transforms you already built for
                       ImageFolder work unchanged here.
            use_lung_crop: toggle lung-field segmentation on/off (useful
                       for quick before/after comparisons during
                       experimentation)
            use_clahe: toggle CLAHE on/off, same reasoning
            cache_dir: if provided, preprocessed images (after lung-crop
                       + CLAHE, resized to cache_size x cache_size) are
                       saved here on first access and loaded from disk on
                       every subsequent access. Lung segmentation is a
                       neural network forward pass, so running it fresh
                       on every image on every epoch is very slow -- this
                       cache means the expensive part only ever runs once
                       per image, not once per epoch. Strongly
                       recommended whenever use_lung_crop=True.
            cache_size: side length (pixels) to resize to BEFORE caching.
                       Caching at full original resolution (X-rays are
                       often 1000-3000px per side) wastes huge amounts of
                       disk space for no benefit, since the training
                       pipeline resizes to 224x224 anyway -- this can
                       fill up disk quotas (e.g. on Kaggle/Colab) after
                       enough images get cached. 256 gives a small margin
                       over the typical 224 training size while keeping
                       cached files small (~30-60x smaller than full-res).
        """
        self.samples, self.classes, self.class_to_idx = _make_samples(root)
        self.transform = transform
        self.use_lung_crop = use_lung_crop
        self.use_clahe = use_clahe
        self.cache_dir = cache_dir
        self.cache_size = cache_size

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.samples)

    def _cache_path(self, path):
        # Build a unique, flat cache filename from the original path so
        # images from different class subfolders never collide.
        safe_name = path.replace(os.sep, "__").replace("/", "__")
        return os.path.join(self.cache_dir, safe_name + ".npy")

    def _preprocess(self, path):
        """Run lung-crop + CLAHE, then resize down before returning."""
        image = cv2.imread(path)
        if self.use_lung_crop:
            image = crop_to_lung_fields(image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.use_clahe:
            gray = apply_clahe(gray)

        # Resize down BEFORE converting to 3-channel / caching. This is
        # the key fix: caching at full original resolution (often
        # 1000-3000px) can silently fill up disk quotas after enough
        # images are cached, since the training pipeline only ever needs
        # 224x224 anyway. Doing this resize once, here, also makes cache
        # reads faster (smaller files) on every subsequent epoch.
        gray = cv2.resize(gray, (self.cache_size, self.cache_size))

        # DenseNet121 expects 3-channel input (pretrained on RGB
        # ImageNet images), even though X-rays are inherently grayscale.
        # We replicate the single channel across R, G, B.
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return rgb

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        if self.cache_dir:
            cache_path = self._cache_path(path)
            if os.path.exists(cache_path):
                rgb = np.load(cache_path)
            else:
                rgb = self._preprocess(path)
                np.save(cache_path, rgb)
        else:
            rgb = self._preprocess(path)

        # ---- Hand off to the standard torchvision pipeline ----
        pil_image = Image.fromarray(rgb)

        if self.transform:
            pil_image = self.transform(pil_image)

        return pil_image, label