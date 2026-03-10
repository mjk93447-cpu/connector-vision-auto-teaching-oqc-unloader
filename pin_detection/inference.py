"""
Inference: detect pins, output masked image and Excel.

Design: YOLO is primary (full model performance). Masked prior supplements FN.
Geometry refinement is auxiliary for edge cases (FN interpolate, FP verify/remove).
"""
import numpy as np
from pathlib import Path
from typing import List, Tuple

from PIL import Image


def _merge_yolo_with_masked_prior(
    yolo_dets: List[Tuple[float, float, float, float]],
    yolo_confs: List[float],
    masked_dets: List[Tuple[float, float, float, float]],
    w: int,
    h: int,
    merge_dist_px: float = 15,
) -> Tuple[List[Tuple[float, float, float, float]], List[float]]:
    """
    Merge YOLO (primary) with masked prior (supplement FN).
    - YOLO detections are primary.
    - Masked positions that have no nearby YOLO detection fill FN (confidence=0.9 as prior).
    - Dedupe: masked within merge_dist_px of YOLO → keep YOLO.
    """
    if not masked_dets:
        return yolo_dets, yolo_confs
    out_d = list(yolo_dets)
    out_c = list(yolo_confs)
    merge_dist_norm = merge_dist_px / min(w, h)

    for md in masked_dets:
        mx, my = md[0], md[1]
        has_nearby = False
        for yd in yolo_dets:
            dx = abs(yd[0] - mx)
            dy = abs(yd[1] - my)
            if dx * w <= merge_dist_px and dy * h <= merge_dist_px:
                has_nearby = True
                break
        if not has_nearby:
            out_d.append(md)
            out_c.append(0.9)
    return out_d, out_c


def run_inference(
    model_path: str | Path,
    image_path: str | Path,
    output_image_path: str | Path | None = None,
    conf_threshold: float = 0.25,
    cap_precision: bool = True,
    use_geometry_refinement: bool = True,
    model: "YOLO | None" = None,
    masked_path: str | Path | None = None,
    roi_margin: float = 0.15,
) -> Tuple[np.ndarray, List[Tuple[float, float, float, float]], np.ndarray]:
    """
    Run YOLO inference on connector image.
    Returns: (original_image, list of (x_center, y_center, w, h) normalized, masked_image)

    Design:
    - YOLO is primary (full model performance).
    - masked_path: supplement FN when YOLO misses (cross markers from masked image).
    - Geometry refinement: auxiliary for FN interpolate, FP verify/remove.
    """
    from ultralytics import YOLO

    img = np.array(Image.open(image_path).convert("RGB"))
    h, w = img.shape[:2]

    # 1. Primary: YOLO inference
    if model is None:
        model = YOLO(model_path)
    predict_input = img
    results = model.predict(predict_input, conf=conf_threshold, verbose=False)
    detections = []
    confidences = []
    if results:
        r = results[0]
        boxes = r.boxes
        det_w = predict_input.shape[1] if predict_input.ndim >= 2 else w
        det_h = predict_input.shape[0] if predict_input.ndim >= 2 else h
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            bx1, by1, bx2, by2 = xyxy
            xc = (bx1 + bx2) / 2 / det_w
            yc = (by1 + by2) / 2 / det_h
            bw = (bx2 - bx1) / det_w
            bh = (by2 - by1) / det_h
            detections.append((xc, yc, bw, bh))
            confidences.append(conf)

    # 2. Supplement: masked prior for FN (when YOLO missed)
    if (
        masked_path is not None
        and Path(masked_path).exists()
        and use_geometry_refinement
    ):
        from .annotation import masked_image_to_annotations
        try:
            _, masked_dets = masked_image_to_annotations(masked_path)
            if masked_dets:
                detections, confidences = _merge_yolo_with_masked_prior(
                    detections, confidences, masked_dets, w, h, merge_dist_px=15
                )
        except Exception:
            pass

    # 3. Auxiliary: geometry refinement (FN interpolate, FP verify/remove)
    if use_geometry_refinement and detections:
        from .geometry_refinement import refine_to_fixed_grid
        detections = refine_to_fixed_grid(detections, confidences, w, h, n_per_row=20)
    elif cap_precision and confidences:
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
