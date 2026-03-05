"""
Inference: detect pins, output masked image and Excel.
"""
import numpy as np
from pathlib import Path
from typing import List, Tuple

from PIL import Image


def run_inference(
    model_path: str | Path,
    image_path: str | Path,
    output_image_path: str | Path | None = None,
    conf_threshold: float = 0.25,
    cap_precision: bool = True,
) -> Tuple[np.ndarray, List[Tuple[float, float, float, float]], np.ndarray]:
    """
    Run YOLO inference on connector image.
    Returns: (original_image, list of (x_center, y_center, w, h) normalized, masked_image)
    cap_precision: 위/아래 각 20개 초과 시 confidence 상위 20개만 유지 (Precision 보장)
    """
    from ultralytics import YOLO

    model = YOLO(model_path)
    img = np.array(Image.open(image_path).convert("RGB"))
    h, w = img.shape[:2]

    results = model.predict(image_path, conf=conf_threshold, verbose=False)
    if not results:
        return img, [], img.copy()

    r = results[0]
    boxes = r.boxes
    detections = []
    confidences = []
    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        x1, y1, x2, y2 = xyxy
        xc = (x1 + x2) / 2 / w
        yc = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        detections.append((xc, yc, bw, bh))
        confidences.append(conf)

    if cap_precision and confidences:
        detections = cap_at_20_per_row(detections, confidences)

    masked = draw_green_dots(img, detections, w, h)

    if output_image_path:
        Image.fromarray(masked).save(output_image_path)

    return img, detections, masked


def draw_green_dots(img: np.ndarray, detections: List[Tuple[float, float, float, float]], w: int, h: int, dot_radius: int = 3) -> np.ndarray:
    """Draw green dots at pin centers."""
    out = img.copy()
    green = np.array([0, 255, 0], dtype=np.uint8)
    for xc, yc, bw, bh in detections:
        px = int(xc * w)
        py = int(yc * h)
        for dy in range(-dot_radius, dot_radius + 1):
            for dx in range(-dot_radius, dot_radius + 1):
                if dx * dx + dy * dy <= dot_radius * dot_radius:
                    ny, nx = py + dy, px + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        out[ny, nx] = green
    return out


def split_upper_lower(detections: List[Tuple[float, float, float, float]]) -> Tuple[List, List]:
    """Split detections into upper and lower by y coordinate (normalized 0-1)."""
    upper = []
    lower = []
    mid_y = 0.5
    for d in detections:
        if d[1] < mid_y:
            upper.append(d)
        else:
            lower.append(d)
    return upper, lower


def cap_at_20_per_row(
    detections: List[Tuple[float, float, float, float]],
    confidences: List[float],
) -> List[Tuple[float, float, float, float]]:
    """
    Precision: 20개 초과 감지 금지. 위/아래 각각 20개 초과 시 confidence 상위 20개만 유지.
    """
    if len(confidences) != len(detections):
        return detections
    upper = [(d, c) for d, c in zip(detections, confidences) if d[1] < 0.5]
    lower = [(d, c) for d, c in zip(detections, confidences) if d[1] >= 0.5]
    upper_sorted = sorted(upper, key=lambda x: -x[1])[:20]
    lower_sorted = sorted(lower, key=lambda x: -x[1])[:20]
    return [d for d, _ in upper_sorted] + [d for d, _ in lower_sorted]


def compute_spacing_mm(detections: List[Tuple[float, float, float, float]], w: int, pin_width_mm: float = 0.5) -> List[float]:
    """Compute left-right spacing between adjacent pins in mm. Sort by x, then diff."""
    if len(detections) < 2:
        return []
    sorted_by_x = sorted(detections, key=lambda d: d[0])
    pixel_widths = [d[2] * w for d in sorted_by_x]
    avg_pixel_width = sum(pixel_widths) / len(pixel_widths) or 1
    mm_per_pixel = pin_width_mm / avg_pixel_width

    spacings = []
    for i in range(len(sorted_by_x) - 1):
        x1 = sorted_by_x[i][0] * w
        x2 = sorted_by_x[i + 1][0] * w
        gap_px = x2 - x1 - (sorted_by_x[i][2] * w + sorted_by_x[i + 1][2] * w) / 2
        spacings.append(max(0, gap_px * mm_per_pixel))
    return spacings
