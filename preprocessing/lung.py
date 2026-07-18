"""
Lung-field cropping for MediFlow, using a pretrained segmentation model.

Motivation: Grad-CAM analysis on the v2 DenseNet121 model revealed that
several false positives (NORMAL X-rays misclassified as PNEUMONIA) were
driven by the model attending to the central mediastinum/thymus region
rather than actual lung tissue. This module segments out the two lung
fields and suppresses everything else (mediastinum, heart, spine,
background), so the model can no longer use that region as a shortcut.

NOTE: An earlier version of this file used classical OpenCV thresholding
(Otsu + contours) to approximate lung segmentation. That approach failed
in practice -- global thresholding can't cleanly separate lung tissue from
background/bone on chest X-rays, since their intensity ranges overlap. This
version instead uses torchxrayvision's pretrained PSPNet segmentation
model, which was trained specifically to segment anatomical structures
(including "Left Lung" and "Right Lung" as distinct classes) on chest
X-rays, and is far more reliable.
"""

import numpy as np
import torch
import torchvision
import cv2
import torchxrayvision as xrv

# Run on GPU if available -- this model previously ran silently on CPU
# even when a GPU was available elsewhere in the pipeline (e.g. for
# DenseNet121 training), which made per-image lung segmentation roughly
# an order of magnitude slower than necessary.
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
        mask: 2D numpy array (uint8, 0 or 255), same size as the model's
              internal 512x512 processing resolution -- resize back to
              your original image size before applying.
    """
    img = xrv.datasets.normalize(image_gray, 255)  # scale to model's expected range
    img = img[None, ...]  # add channel dim -> [1, H, W]
    img = _transform(img)
    img_tensor = torch.from_numpy(img).unsqueeze(0).float().to(_DEVICE)  # [1, 1, 512, 512]

    with torch.no_grad():
        output = _seg_model(img_tensor)  # [1, 14, 512, 512]
        output = torch.sigmoid(output)   # raw scores -> 0-1 probabilities

    lung_prob = output[0, _LEFT_LUNG_IDX] + output[0, _RIGHT_LUNG_IDX]
    mask = (lung_prob > threshold).cpu().numpy().astype(np.uint8) * 255
    return mask


def crop_to_lung_fields(image_bgr_or_gray, dilate_mask_px=10):
    """
    Suppress non-lung regions (mediastinum, heart, spine, background)
    using the pretrained lung segmentation mask.

    Args:
        image_bgr_or_gray: original image, either grayscale or BGR
            (as read by cv2.imread), any resolution
        dilate_mask_px: expand the mask by this many pixels so we don't
            clip real lung tissue near the segmentation boundary

    Returns:
        result: same shape/channels as input, with non-lung regions
            suppressed (set to 0 / black)
    """
    if len(image_bgr_or_gray.shape) == 3:
        gray = cv2.cvtColor(image_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr_or_gray

    orig_h, orig_w = gray.shape

    mask = get_lung_mask(gray)
    # The model works at 512x512 internally -- resize the mask back to
    # match the original image's actual dimensions before applying it.
    mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    if dilate_mask_px > 0:
        kernel = np.ones((dilate_mask_px, dilate_mask_px), np.uint8)
        mask = cv2.dilate(mask, kernel)

    if len(image_bgr_or_gray.shape) == 3:
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        result = cv2.bitwise_and(image_bgr_or_gray, mask_3ch)
    else:
        result = cv2.bitwise_and(image_bgr_or_gray, mask)

    return result


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    IMAGE_PATH = "sample_images/sample_xray.jpg"  # swap in a real test image

    original = cv2.imread(IMAGE_PATH)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    cropped = crop_to_lung_fields(original)
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

    mask = get_lung_mask(gray)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Segmented Lung Mask (512x512)")
    axes[1].axis("off")

    axes[2].imshow(cropped_rgb)
    axes[2].set_title("Mediastinum Suppressed")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()