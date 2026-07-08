"""
CLAHE (Contrast Limited Adaptive Histogram Equalization) for MediFlow.

Chest X-rays often have uneven contrast -- some regions washed out,
others too dark to see subtle detail -- because exposure settings and
patient body composition vary. CLAHE redistributes pixel intensities
locally (in small tiles), rather than globally, so it boosts contrast
where needed without over-brightening regions that are already well
exposed. The "contrast limited" part caps how aggressively any single
tile can be boosted, which prevents noise from being amplified into
something that could look like a false pathology signal.
"""

import cv2
import numpy as np


def apply_clahe(image_gray, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE to a grayscale chest X-ray.

    Args:
        image_gray: 2D numpy array, grayscale image (uint8)
        clip_limit: caps contrast amplification per tile. Higher = more
            aggressive contrast boosting, but more noise amplification too.
            2.0 is a common, conservative default for medical imaging.
        tile_grid_size: how many tiles to divide the image into (rows, cols).
            Smaller tiles = more local/aggressive adaptation; larger tiles
            behave closer to plain (non-adaptive) histogram equalization.

    Returns:
        2D numpy array, same shape as input, contrast-enhanced
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image_gray)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    IMAGE_PATH = "sample_images/IM-0011-0001-0002.jpeg"  # swap in a real test image

    original = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    enhanced = apply_clahe(original)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original (grayscale)")
    axes[0].axis("off")

    axes[1].imshow(enhanced, cmap="gray")
    axes[1].set_title("After CLAHE")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()