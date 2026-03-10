"""
ROI extraction for large images: find pin cluster region from masked image.
Crop to ROI before training/inference to reduce computation.
Connector is horizontal (wide, short) → ROI min: width ≥ 30%, height ≥ 10% of image.
"""
import numpy as np
from pathlib import Path
from typing import Tuple

from .annotation import extract_red_mask, extract_green_mask, cluster_to_bbox

# Connector is horizontal: ROI must be wide and short
MIN_ROI_WIDTH_RATIO = 0.30   # ≥ 30% of image width
MIN_ROI_HEIGHT_RATIO = 0.10  # ≥ 10% of image height


def extract_pin_roi(
    masked_path: str | Path,
    margin_ratio: float = 0.15,
    min_roi_size: int = 200,
    min_width_ratio: float = MIN_ROI_WIDTH_RATIO,
    min_height_ratio: float = MIN_ROI_HEIGHT_RATIO,
) -> Tuple[int, int, int, int]:
    """
    Extract ROI (x1, y1, x2, y2) from masked image where green pins are.
    Adds margin around the union of all pin bboxes.
    Enforces min ROI: width ≥ min_width_ratio*w, height ≥ min_height_ratio*h (connector is horizontal).
    Returns pixel coordinates for cropping.
    """
    from PIL import Image
    img = np.array(Image.open(masked_path).convert("RGB"))
    h, w = img.shape[:2]
    # RED (ROI Editor target) 우선, 없으면 GREEN (legacy). Action #33
    mask_red = extract_red_mask(img)
    mask_green = extract_green_mask(img)
    mask = mask_red if mask_red.any() else mask_green
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

    min_w = max(min_roi_size, int(w * min_width_ratio))
    min_h = max(min_roi_size, int(h * min_height_ratio))
    out_w = x2 - x1
    out_h = y2 - y1

    if out_w < min_w or out_h < min_h:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        half_w = max(min_w // 2, out_w // 2)
        half_h = max(min_h // 2, out_h // 2)
        x1 = max(0, cx - half_w)
        y1 = max(0, cy - half_h)
        x2 = min(w, cx + half_w)
        y2 = min(h, cy + half_h)
        out_w = x2 - x1
        out_h = y2 - y1
        if out_w < min_w:
            dx = min_w - out_w
            x1 = max(0, x1 - dx // 2)
            x2 = min(w, x2 + (dx - dx // 2))
        if out_h < min_h:
            dy = min_h - out_h
            y1 = max(0, y1 - dy // 2)
            y2 = min(h, y2 + (dy - dy // 2))

    return x1, y1, x2, y2


def crop_to_roi(
    img: np.ndarray,
    roi: Tuple[int, int, int, int],
) -> np.ndarray:
    """Crop image to ROI (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = roi
    return img[y1:y2, x1:x2].copy()


