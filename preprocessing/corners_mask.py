"""
Corner artifact masking.

Motivation: Grad-CAM showed repeated false positives (IM-0011, and a
second case) both driven by attention on burned-in PACS text/timestamp
overlays in image corners ("17:28:51", "KV68"), not lung tissue. This
masks those corner regions BEFORE lung segmentation runs, so the
watermark can neither confuse segmentation nor survive into the crop.
"""

import cv2
import numpy as np


def mask_corners(image, corner_frac=0.08):
    """
    Zero out square regions in all four corners of the image.

    Args:
        image: BGR or grayscale, any resolution
        corner_frac: side length of each masked square, as a fraction
            of min(height, width). 0.08 is a conservative starting
            point -- large enough to cover typical PACS overlay text,
            small enough to be very unlikely to touch lung tissue
            (which is centered, not cornered).

    Returns:
        image with corner squares set to 0 (black)
    """
    h, w = image.shape[:2]
    s = int(min(h, w) * corner_frac)

    result = image.copy()
    for y0, y1 in [(0, s), (h - s, h)]:
        for x0, x1 in [(0, s), (w - s, w)]:
            result[y0:y1, x0:x1] = 0
    return result


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    IMAGE_PATH = "sample_images/IM-0011-0001-0002.jpeg"  # swap in a real test image

    original = cv2.imread(IMAGE_PATH)
    masked = mask_corners(original)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Corners masked")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()