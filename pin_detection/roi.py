"""
ROI extraction for large images: find pin cluster region from masked image.
Crop to ROI before training/inference to reduce computation.
"""
import numpy as np
from pathlib import Path
from typing import Tuple

from .annotation import extract_green_mask, cluster_to_bbox


def extract_pin_roi(
    masked_path: str | Path,
    margin_ratio: float = 0.15,
    min_roi_size: int = 200,
) -> Tuple[int, int, int, int]:
    """
    Extract ROI (x1, y1, x2, y2) from masked image where green pins are.
    Adds margin around the union of all pin bboxes.
    Returns pixel coordinates for cropping.
    """
    from PIL import Image
    img = np.array(Image.open(masked_path).convert("RGB"))
    h, w = img.shape[:2]
    mask = extract_green_mask(img)
    bboxes = cluster_to_bbox(mask)
    if not bboxes:
        return 0, 0, w, h  # full image fallback

    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[2] for b in bboxes)
    y2 = max(b[3] for b in bboxes)

    roi_w = x2 - x1
    roi_h = y2 - y1
    margin_x = max(int(roi_w * margin_ratio), 20)
    margin_y = max(int(roi_h * margin_ratio), 20)

    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w, x2 + margin_x)
    y2 = min(h, y2 + margin_y)

    out_w = x2 - x1
    out_h = y2 - y1
    if out_w < min_roi_size or out_h < min_roi_size:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        half = max(min_roi_size // 2, out_w // 2, out_h // 2)
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, cx + half)
        y2 = min(h, cy + half)

    return x1, y1, x2, y2


def crop_to_roi(
    img: np.ndarray,
    roi: Tuple[int, int, int, int],
) -> np.ndarray:
    """Crop image to ROI (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = roi
    return img[y1:y2, x1:x2].copy()


