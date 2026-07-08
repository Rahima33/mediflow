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
    def __init__(self, root, transform=None, use_lung_crop=True, use_clahe=True):
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
        """
        self.samples, self.classes, self.class_to_idx = _make_samples(root)
        self.transform = transform
        self.use_lung_crop = use_lung_crop
        self.use_clahe = use_clahe

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # ---- Load with OpenCV (BGR by default) ----
        image = cv2.imread(path)

        # ---- Step 1: lung-field segmentation ----
        # Suppresses the mediastinum/heart/spine region that Grad-CAM
        # showed the model was incorrectly using as a shortcut.
        if self.use_lung_crop:
            image = crop_to_lung_fields(image)

        # ---- Step 2: CLAHE (needs a single-channel grayscale image) ----
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.use_clahe:
            gray = apply_clahe(gray)

        # ---- Convert back to 3 channels ----
        # DenseNet121 expects 3-channel input (it was pretrained on RGB
        # ImageNet images), even though X-rays are inherently grayscale.
        # We replicate the single channel across R, G, B.
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        # ---- Hand off to the standard torchvision pipeline ----
        # PIL.Image is what torchvision transforms expect as input.
        pil_image = Image.fromarray(rgb)

        if self.transform:
            pil_image = self.transform(pil_image)

        return pil_image, label


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision import transforms

    # Point this at your actual local copy of the Kaggle dataset's
    # train folder, e.g. a small subset you've copied down for testing.
    TEST_DIR = "sample_images_by_class"  # expects TEST_DIR/NORMAL, TEST_DIR/PNEUMONIA

    preview_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = MediFlowXrayDataset(root=TEST_DIR, transform=preview_transform)

    print(f"Found {len(dataset)} images across classes: {dataset.classes}")
    print(f"class_to_idx: {dataset.class_to_idx}")

    # Show the first few images after the full preprocessing pipeline
    fig, axes = plt.subplots(1, min(4, len(dataset)), figsize=(16, 4))
    if len(dataset) == 1:
        axes = [axes]

    for i in range(min(4, len(dataset))):
        image_tensor, label = dataset[i]
        image_np = image_tensor.permute(1, 2, 0).numpy()  # CHW -> HWC for matplotlib
        axes[i].imshow(image_np)
        axes[i].set_title(dataset.classes[label])
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()