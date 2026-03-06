"""
Geometry-based refinement for pin detection.
Leverages fixed layout: 20 upper + 20 lower, horizontal alignment, ~uniform spacing.
Goal: Precision/Recall → 100% via interpolation (FN) and slot capping (FP).
Includes: Y-band filter, outlier removal, bounds clamp, empty-row handling.
"""
from typing import List, Tuple

import numpy as np

# Expected pin row y bands (normalized). Detections outside are filtered.
UPPER_Y_MIN, UPPER_Y_MAX = 0.03, 0.52
LOWER_Y_MIN, LOWER_Y_MAX = 0.42, 0.97
# Valid x range for interpolated positions
X_MIN_VALID, X_MAX_VALID = 0.02, 0.98


def _filter_by_y_band(
    detections: List[Tuple[float, float, float, float]],
    confidences: List[float] | None,
    upper: bool,
) -> Tuple[List[Tuple], List[float] | None]:
    """Remove detections outside expected y band for row."""
    y_min, y_max = (UPPER_Y_MIN, UPPER_Y_MAX) if upper else (LOWER_Y_MIN, LOWER_Y_MAX)
    out_d, out_c = [], [] if confidences else None
    for i, d in enumerate(detections):
        if y_min <= d[1] <= y_max:
            out_d.append(d)
            if confidences:
                out_c.append(confidences[i])
    return out_d, out_c


def _remove_y_outliers(
    row_dets: List[Tuple[float, float, float, float]],
    row_confs: List[float] | None,
    sigma: float = 3.0,
) -> Tuple[List[Tuple], List[float] | None]:
    """Remove detections with y far from row median (only extreme outliers)."""
    if len(row_dets) < 5:
        return row_dets, row_confs
    ys = [d[1] for d in row_dets]
    med, std = np.median(ys), np.std(ys) or 1e-6
    thresh = med + sigma * std
    out_d, out_c = [], [] if row_confs else None
    for i, d in enumerate(row_dets):
        if abs(d[1] - med) <= sigma * std:
            out_d.append(d)
            if row_confs:
                out_c.append(row_confs[i])
    return out_d or row_dets, out_c or row_confs


def _clamp_bbox(xc: float, yc: float, w: float, h: float) -> Tuple[float, float, float, float]:
    """Clamp to valid [0,1] range."""
    return (
        max(0, min(1, xc)),
        max(0, min(1, yc)),
        max(1e-4, min(1, w)),
        max(1e-4, min(1, h)),
    )


def split_upper_lower(
    detections: List[Tuple[float, float, float, float]],
    confidences: List[float] | None,
    mid_y: float = 0.5,
) -> Tuple[List[Tuple], List[Tuple], List[float] | None, List[float] | None]:
    """Split detections into upper/lower by y. Returns (upper, lower, upper_conf, lower_conf)."""
    upper, lower = [], []
    uc, lc = [], [] if confidences else None
    for i, d in enumerate(detections):
        if d[1] < mid_y:
            upper.append(d)
            if confidences:
                uc.append(confidences[i])
        else:
            lower.append(d)
            if confidences:
                lc.append(confidences[i])
    return upper, lower, uc, lc


def _refine_row(
    row_dets: List[Tuple[float, float, float, float]],
    row_confs: List[float] | None,
    w: int,
    h: int,
    n_slots: int,
    max_gap_ratio: float = 1.8,
) -> List[Tuple[float, float, float, float]]:
    """
    Refine one row to exactly n_slots positions.
    - If len < n_slots: interpolate missing from gaps (large gap = missing pin between)
    - If len > n_slots: keep top n_slots by confidence
    """
    if not row_dets:
        return []

    # Sort by x
    paired = list(zip(row_dets, row_confs or [1.0] * len(row_dets)))
    paired.sort(key=lambda p: p[0][0])
    sorted_dets = [p[0] for p in paired]
    sorted_confs = [p[1] for p in paired]

    if len(sorted_dets) > n_slots:
        # FP: keep top n_slots by confidence
        idx = np.argsort([-c for c in sorted_confs])[:n_slots]
        return [sorted_dets[i] for i in sorted(idx)]

    if len(sorted_dets) == n_slots:
        return sorted_dets

    # FN: interpolate missing. Find large gaps and insert.
    xs = [d[0] * w for d in sorted_dets]
    ys = [d[1] * h for d in sorted_dets]
    bw = np.mean([d[2] * w for d in sorted_dets]) if sorted_dets else 12
    bh = np.mean([d[3] * h for d in sorted_dets]) if sorted_dets else 4

    out_x = list(xs)
    out_y = list(ys)
    n_missing = n_slots - len(sorted_dets)

    while len(out_x) < n_slots and len(out_x) >= 2:
        gaps = [out_x[i + 1] - out_x[i] for i in range(len(out_x) - 1)]
        med_gap = np.median(gaps) if gaps else (out_x[-1] - out_x[0]) / (len(out_x) - 1)
        max_gap = med_gap * max_gap_ratio
        inserted = False
        for i in range(len(out_x) - 1):
            gap = out_x[i + 1] - out_x[i]
            if gap > max_gap and n_missing > 0:
                new_x = (out_x[i] + out_x[i + 1]) / 2
                new_y = (out_y[i] + out_y[i + 1]) / 2
                out_x.insert(i + 1, new_x)
                out_y.insert(i + 1, new_y)
                n_missing -= 1
                inserted = True
                break
        if not inserted:
            break

    # If still missing (e.g. at ends), extend uniformly
    while len(out_x) < n_slots and len(out_x) >= 2:
        dx = out_x[1] - out_x[0]
        out_x.insert(0, out_x[0] - dx)
        out_y.insert(0, out_y[0])
        n_missing -= 1
        if n_missing <= 0:
            break
    while len(out_x) < n_slots and len(out_x) >= 2:
        dx = out_x[-1] - out_x[-2]
        out_x.append(out_x[-1] + dx)
        out_y.append(out_y[-1])
        n_missing -= 1
        if n_missing <= 0:
            break

    # Convert back to normalized (xc, yc, w, h), clamp to valid range
    result = []
    for x, y in zip(out_x[:n_slots], out_y[:n_slots]):
        xc = x / w
        yc = y / h
        xc = max(X_MIN_VALID, min(X_MAX_VALID, xc))
        result.append(_clamp_bbox(xc, yc, bw / w, bh / h))
    return result


def _template_row_from_other(
    other_row: List[Tuple[float, float, float, float]],
    w: int,
    h: int,
    n_per_row: int,
    target_y: float,
) -> List[Tuple[float, float, float, float]]:
    """Generate template row from other row's x spacing when this row is empty."""
    if not other_row or len(other_row) < 2:
        # Fallback: uniform x across image
        xs = [X_MIN_VALID + i * (X_MAX_VALID - X_MIN_VALID) / (n_per_row - 1) for i in range(n_per_row)]
        bw = 0.02
        bh = 0.01
        return [_clamp_bbox(x, target_y, bw, bh) for x in xs]
    sorted_other = sorted(other_row, key=lambda d: d[0])
    xs = [d[0] for d in sorted_other[:n_per_row]]
    if len(xs) < n_per_row:
        dx = (xs[-1] - xs[0]) / max(1, len(xs) - 1) if len(xs) > 1 else 0.02
        while len(xs) < n_per_row:
            xs.append(xs[-1] + dx)
    bw = np.mean([d[2] for d in sorted_other])
    bh = np.mean([d[3] for d in sorted_other])
    return [_clamp_bbox(x, target_y, bw, bh) for x in xs[:n_per_row]]


def refine_to_fixed_grid(
    detections: List[Tuple[float, float, float, float]],
    confidences: List[float] | None,
    w: int,
    h: int,
    n_per_row: int = 20,
    mid_y: float = 0.5,
    min_per_row_for_interp: int = 8,
) -> List[Tuple[float, float, float, float]]:
    """
    Refine detections to exactly n_per_row * 2 (upper + lower).
    Uses geometry: uniform spacing, interpolate FN, cap FP.
    Applies: Y-band filter, outlier removal, bounds clamp, empty-row template.
    """
    # Pre-filter: Y band (remove out-of-range detections), optional outlier removal
    upper_raw, lower_raw, uc_raw, lc_raw = split_upper_lower(detections, confidences, mid_y)
    upper_f, uc_f = _filter_by_y_band(upper_raw, uc_raw, upper=True)
    lower_f, lc_f = _filter_by_y_band(lower_raw, lc_raw, upper=False)
    # Fallback: if filter removed all, use raw (avoid over-filtering)
    if not upper_f:
        upper_f, uc_f = upper_raw, uc_raw
    if not lower_f:
        lower_f, lc_f = lower_raw, lc_raw
    # Outlier removal: only when many detections (avoid removing valid edge pins)
    if len(upper_f) >= 12:
        upper_f, uc_f = _remove_y_outliers(upper_f, uc_f)
    if len(lower_f) >= 12:
        lower_f, lc_f = _remove_y_outliers(lower_f, lc_f)

    def _safe_refine(row_dets, row_confs, is_upper):
        if not row_dets:
            # Empty row: use template from other row
            other = lower_f if is_upper else upper_f
            ty = 0.2 if is_upper else 0.65
            return _template_row_from_other(other, w, h, n_per_row, ty)
        if len(row_dets) < min_per_row_for_interp:
            if len(row_dets) > n_per_row and row_confs:
                paired = sorted(zip(row_dets, row_confs), key=lambda p: -p[1])[:n_per_row]
                return [_clamp_bbox(d[0], d[1], d[2], d[3]) for d, _ in paired]
            return [_clamp_bbox(d[0], d[1], d[2], d[3]) for d in row_dets]
        return _refine_row(row_dets, row_confs, w, h, n_per_row)

    upper_ref = _safe_refine(upper_f, uc_f, is_upper=True)
    lower_ref = _safe_refine(lower_f, lc_f, is_upper=False)
    return upper_ref + lower_ref
