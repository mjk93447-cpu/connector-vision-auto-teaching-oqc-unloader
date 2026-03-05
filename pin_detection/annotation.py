"""
Extract pin bounding boxes from masked image (green dots).
Output: YOLO format annotations.
"""
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple


def extract_green_mask(img: np.ndarray) -> np.ndarray:
    """Extract binary mask of green pixels (G high, R and B low)."""
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    mask = ((g > 150) & (r < g - 60) & (b < g - 60)).astype(np.uint8) * 255
    return mask


def cluster_to_bbox(mask: np.ndarray, min_area: int = 4) -> List[Tuple[int, int, int, int]]:
    """
    Find connected green regions and return bounding boxes (x1, y1, x2, y2).
    Uses simple flood-fill or contour detection.
    """
    from scipy import ndimage
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
    h, w = img.shape[:2]
    mask = extract_green_mask(img)
    bboxes = cluster_to_bbox(mask)
    yolo_anns = bboxes_to_yolo_format(bboxes, w, h)
    return img, yolo_anns
