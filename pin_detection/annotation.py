"""
Extract pin bounding boxes from masked image.
Output: YOLO format annotations.
Supports: red (Action #33, ROI Editor target), green (legacy), cross-shaped (+) markers.
"""
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple

# Relaxed green detection for cross-shaped (+) markers (thin lines, slight color variation)
GREEN_G_MIN = 120
GREEN_R_MAX_DIFF = 50   # r < g - this
GREEN_B_MAX_DIFF = 50   # b < g - this

# Red target marker (ROI Editor square/brush, avoids YOLO confusion with original green)
RED_R_MIN = 200
RED_G_MAX_DIFF = 60   # g < r - this
RED_B_MAX_DIFF = 60   # b < r - this


def extract_red_mask(img: np.ndarray) -> np.ndarray:
    """Extract binary mask of red pixels (R high, G and B low). Action #33 target marker."""
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    mask = ((r > RED_R_MIN) & (g < r - RED_G_MAX_DIFF) & (b < r - RED_B_MAX_DIFF)).astype(np.uint8) * 255
    return mask


def extract_green_mask(img: np.ndarray, relaxed: bool = True) -> np.ndarray:
    """
    Extract binary mask of green pixels (G high, R and B low).
    relaxed=True: looser thresholds for cross-shaped (+) markers, slight color variation.
    """
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    if relaxed:
        mask = ((g > GREEN_G_MIN) & (r < g - GREEN_R_MAX_DIFF) & (b < g - GREEN_B_MAX_DIFF)).astype(np.uint8) * 255
    else:
        mask = ((g > 150) & (r < g - 60) & (b < g - 60)).astype(np.uint8) * 255
    return mask


def _dilate_for_thin_markers(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Morphological dilation to connect thin cross (+) arms into single blobs.
    Thin 1–2px lines may fragment; dilation merges them before clustering.
    """
    from scipy import ndimage
    struct = ndimage.generate_binary_structure(2, 2)  # 3x3 cross-like
    for _ in range(kernel_size // 2):
        mask = ndimage.binary_dilation(mask, structure=struct).astype(np.uint8) * 255
    return mask


def cluster_to_bbox(mask: np.ndarray, min_area: int = 2, dilate_thin: bool = True) -> List[Tuple[int, int, int, int]]:
    """
    Find connected green regions and return bounding boxes (x1, y1, x2, y2).
    dilate_thin: apply dilation before clustering to merge thin cross (+) arms.
    min_area: minimum pixels per cluster (2 for very thin crosses).
    """
    from scipy import ndimage
    if dilate_thin and mask.any():
        mask = _dilate_for_thin_markers(mask)
    labeled, num_features = ndimage.label(mask)
    bboxes = []
    for i in range(1, num_features + 1):
        ys, xs = np.where(labeled == i)
        if len(ys) < min_area:
            continue
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        bboxes.append((x1, y1, x2, y2))
    return bboxes


def bboxes_to_yolo_format(bboxes: List[Tuple[int, int, int, int]], img_w: int, img_h: int) -> List[Tuple[float, float, float, float]]:
    """Convert (x1,y1,x2,y2) to YOLO normalized (x_center, y_center, width, height)."""
    out = []
    for x1, y1, x2, y2 in bboxes:
        xc = (x1 + x2) / 2 / img_w
        yc = (y1 + y2) / 2 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        out.append((xc, yc, w, h))
    return out


def masked_image_to_annotations(masked_path: str | Path) -> Tuple[np.ndarray, List[Tuple[float, float, float, float]]]:
    """
    Load masked image, extract green regions, return image and YOLO annotations.
    Returns: (image_array, list of (xc, yc, w, h) normalized)
    """
    img = np.array(Image.open(masked_path).convert("RGB"))
    return masked_array_to_annotations(img)


def masked_array_to_annotations(img: np.ndarray) -> Tuple[np.ndarray, List[Tuple[float, float, float, float]]]:
    """
    Extract YOLO annotations from masked image array.
    RED (target marker) takes priority over GREEN (legacy) for YOLO training. Action #33.
    """
    h, w = img.shape[:2]
    mask_red = extract_red_mask(img)
    mask_green = extract_green_mask(img)
    # Prefer red (ROI Editor target); fallback to green (legacy)
    mask = mask_red if mask_red.any() else mask_green
    bboxes = cluster_to_bbox(mask)
    yolo_anns = bboxes_to_yolo_format(bboxes, w, h)
    return img, yolo_anns
