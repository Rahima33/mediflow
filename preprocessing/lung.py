"""
Lung-field cropping for MediFlow, using a pretrained segmentation model.

Motivation: Grad-CAM analysis on earlier model versions revealed two
distinct problems:
  1. (v3) The model sometimes fixated on the central mediastinum/thymus
     region rather than actual lung tissue on misclassified NORMAL cases.
  2. (frozen + lung-crop + CLAHE) The SAME-SIZE, black-masked crop
     approach introduced a NEW artifact: the hard boundary between real
     pixels and masked-black pixels became a learnable shortcut of its
     own, with attention landing on mask edges / lateral margins /
     the diaphragm curve rather than lung interior.

This version defaults to mode="crop": a TRUE bounding-box crop (the
returned image is smaller, containing only real photographic content --
no masked-black pixels, no artificial internal edge). This targets
problem #2 directly. Note it does not resuppress the mediastinum (the
space between the two lungs falls inside the bounding box), so it is a
different, complementary intervention to the original mask-based
approach, not a strict improvement on every axis.

Uses torchxrayvision's pretrained PSPNet segmentation model, trained
specifically to segment anatomical structures (including "Left Lung"
and "Right Lung" as distinct classes) on chest X-rays.
"""

import numpy as np
import torch
import torchvision
import cv2
import torchxrayvision as xrv

# Run on GPU if available.
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load once, at import time, so repeated calls don't reload the model.
# Weights are downloaded automatically on first use and cached locally
# (in ~/.torchxrayvision).
_seg_model = xrv.baseline_models.chestx_det.PSPNet()
_seg_model.eval()
_seg_model.to(_DEVICE)

_LEFT_LUNG_IDX = _seg_model.targets.index("Left Lung")
_RIGHT_LUNG_IDX = _seg_model.targets.index("Right Lung")

_transform = torchvision.transforms.Compose([
    xrv.datasets.XRayCenterCrop(),
    xrv.datasets.XRayResizer(512)
])


def get_lung_mask(image_gray, threshold=0.5):
    """
    Run the pretrained segmentation model and combine the Left Lung +
    Right Lung channels into a single binary mask.

    Args:
        image_gray: 2D numpy array, grayscale X-ray, any resolution
        threshold: probability cutoff for "this pixel is lung" (0-1)

    Returns:
        mask: 2D numpy array (uint8, 0 or 255), 512x512 (the model's
              internal processing resolution) -- resize back to your
              original image size before applying.
    """
    img = xrv.datasets.normalize(image_gray, 255)
    img = img[None, ...]
    img = _transform(img)
    img_tensor = torch.from_numpy(img).unsqueeze(0).float().to(_DEVICE)

    with torch.no_grad():
        output = _seg_model(img_tensor)
        output = torch.sigmoid(output)

    lung_prob = output[0, _LEFT_LUNG_IDX] + output[0, _RIGHT_LUNG_IDX]
    mask = (lung_prob > threshold).cpu().numpy().astype(np.uint8) * 255
    return mask


def get_lung_bbox(mask, padding_frac=0.05):
    """
    Find the bounding box of the (already dilated) lung mask, with a
    small padding margin so we don't cut directly against the lung edge.

    Returns (x_min, y_min, x_max, y_max), or None if the mask is empty
    (segmentation failed to find any lung pixels at all).
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    h, w = mask.shape
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    pad_x = int((x_max - x_min) * padding_frac)
    pad_y = int((y_max - y_min) * padding_frac)

    x_min = max(0, x_min - pad_x)
    x_max = min(w, x_max + pad_x)
    y_min = max(0, y_min - pad_y)
    y_max = min(h, y_max + pad_y)

    return x_min, y_min, x_max, y_max


def crop_to_lung_fields(image_bgr_or_gray, dilate_mask_px=10, mode="crop", padding_frac=0.05):
    """
    Restrict the image to the lung-field region using the pretrained
    segmentation mask.

    Args:
        image_bgr_or_gray: original image, either grayscale or BGR
            (as read by cv2.imread), any resolution
        dilate_mask_px: expand the raw segmentation mask by this many
            pixels before computing the crop/mask boundary, so we don't
            clip real lung tissue right at the edge of the (approximate)
            segmentation
        mode: "crop" (default) -- return a true, smaller crop containing
            only the lung bounding box. No non-lung pixels are present
            at all, so there is no artificial masked-region edge for a
            downstream CNN to latch onto as a shortcut. NOTE: since the
            bounding box spans both lungs, the mediastinum (the space
            between them) is still included -- this mode targets
            peripheral/boundary artifacts, not the mediastinal shortcut
            specifically.
            "mask" -- the original behavior: same-size output, non-lung
            pixels zeroed out (black). Kept for comparison/backward
            compatibility; produces a hard internal edge that can itself
            become a learnable shortcut.
        padding_frac: (crop mode only) fraction of the bounding box's
            width/height to pad on each side.

    Returns:
        result: cropped or masked image. In "crop" mode the output size
            varies per image; downstream resizing (in your training
            transforms) handles this the same way it already handles
            variably-sized source X-rays.
    """
    if len(image_bgr_or_gray.shape) == 3:
        gray = cv2.cvtColor(image_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr_or_gray

    orig_h, orig_w = gray.shape

    mask = get_lung_mask(gray)
    mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    if dilate_mask_px > 0:
        kernel = np.ones((dilate_mask_px, dilate_mask_px), np.uint8)
        mask = cv2.dilate(mask, kernel)

    if mode == "crop":
        bbox = get_lung_bbox(mask, padding_frac=padding_frac)
        if bbox is None:
            # Segmentation found nothing -- fall back to the original,
            # uncropped image rather than risk returning an empty array.
            return image_bgr_or_gray
        x_min, y_min, x_max, y_max = bbox
        return image_bgr_or_gray[y_min:y_max, x_min:x_max]

    # mode == "mask": original same-size, black-masked behavior
    if len(image_bgr_or_gray.shape) == 3:
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        result = cv2.bitwise_and(image_bgr_or_gray, mask_3ch)
    else:
        result = cv2.bitwise_and(image_bgr_or_gray, mask)

    return result


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    IMAGE_PATH = "sample_images/IM-0015-0001.jpeg"  # swap in a real test image

    original = cv2.imread(IMAGE_PATH)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    cropped = crop_to_lung_fields(original, mode="crop")
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

    masked = crop_to_lung_fields(original, mode="mask")
    masked_rgb = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(cropped_rgb)
    axes[1].set_title(f"mode='crop' (new default)\nshape: {cropped.shape[:2]}")
    axes[1].axis("off")

    axes[2].imshow(masked_rgb)
    axes[2].set_title("mode='mask' (old behavior)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()