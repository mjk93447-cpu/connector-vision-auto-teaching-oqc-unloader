"""
Score penalty factor analysis: synthetic metrics + synthetic images.
- Sensitivity: for each factor, measure score drop when that factor goes from good to bad.
- Image test: generate images designed to stress each factor; run detector and check
  if the factor discriminates good vs bad boundaries.
Factors with minimal impact (score drop < IMPACT_THRESHOLD) are candidates for removal.
"""
import os
import sys
import json

import numpy as np
from PIL import Image

# Ensure project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sobel_edge_detection import (
    SobelEdgeDetector,
    compute_auto_score,
    evaluate_one_candidate_mp,
    PARAM_DEFAULTS,
    AUTO_DEFAULTS,
)

# Good/bad metric values (aligned with score sigmoid targets in compute_auto_score)
METRIC_GOOD = {
    "coverage": 0.92,
    "gap": 0.08,
    "continuity": 0.06,
    "intrusion": 0.01,
    "outside": 0.02,
    "thickness": 0.04,
    "band_ratio": 0.92,
    "endpoints": 0.02,
    "wrinkle": 0.06,
    "branch": 0.03,
    "excess_dots": 0.0,
}
METRIC_BAD = {
    "coverage": 0.5,
    "gap": 0.5,
    "continuity": 0.5,
    "intrusion": 0.25,
    "outside": 0.25,
    "thickness": 0.25,
    "band_ratio": 0.5,
    "endpoints": 0.35,
    "wrinkle": 0.5,
    "branch": 0.35,
    "excess_dots": 0.5,
}

IMPACT_THRESHOLD_PCT = 2.0  # factors contributing < this % to score drop are "minimal impact"
OUT_DIR = "score_penalty_analysis_out"
SYNTH_DIR = os.path.join(OUT_DIR, "synthetic_images")


def run_sensitivity_analysis():
    """Measure score drop when each factor is set to 'bad' (others good)."""
    weights = dict(AUTO_DEFAULTS)
    base_metrics = dict(METRIC_GOOD)
    score_all_good, _ = compute_auto_score(base_metrics, weights, return_details=True)

    results = []
    for key in METRIC_GOOD:
        if key not in METRIC_BAD:
            continue
        metrics = dict(base_metrics)
        metrics[key] = METRIC_BAD[key]
        score_bad, details = compute_auto_score(metrics, weights, return_details=True)
        drop = score_all_good - score_bad
        drop_pct = 100.0 * (drop / max(score_all_good, 1e-9))
        results.append({
            "factor": key,
            "score_all_good": score_all_good,
            "score_when_bad": score_bad,
            "score_drop": drop,
            "score_drop_pct": drop_pct,
            "has_impact": drop_pct >= IMPACT_THRESHOLD_PCT,
        })
    return score_all_good, results


def make_band_and_boundary(h, w, band_radius, margin=2):
    """Rectangle boundary and band (dilated)."""
    boundary = np.zeros((h, w), dtype=bool)
    boundary[margin, margin:w - margin] = True
    boundary[h - 1 - margin, margin:w - margin] = True
    boundary[margin:h - margin, margin] = True
    boundary[margin:h - margin, w - 1 - margin] = True
    det = SobelEdgeDetector()
    band = det.dilate_binary(boundary, band_radius) if band_radius > 0 else boundary.copy()
    return boundary, band


def build_data_item(image, boundary, band, band_radius):
    mask = np.ones_like(boundary, dtype=bool)
    band_pixels = int(band.sum())
    return {
        "image": np.asarray(image, dtype=np.float32),
        "mask": mask,
        "boundary": boundary,
        "bands": {band_radius: band, 0: boundary},
        "band_pixels": {band_radius: band_pixels, 0: int(boundary.sum())},
        "weight": 1.0,
    }


def generate_synthetic_images():
    """Create images designed to stress: endpoints, branch, wrinkle, intrusion, outside, thickness, gap."""
    os.makedirs(SYNTH_DIR, exist_ok=True)
    h, w = 64, 64
    band_radius = 3
    boundary, band = make_band_and_boundary(h, w, band_radius)
    det = SobelEdgeDetector()
    settings = dict(PARAM_DEFAULTS)
    settings["boundary_band_radius"] = band_radius
    settings["use_boundary_band_filter"] = False
    settings["nms_relax"] = 0.96
    settings["high_ratio"] = 0.10
    settings["low_ratio"] = 0.04

    cases = []

    # 1) Clean rectangle: baseline (low endpoints, low branch if closed contour)
    img_clean = np.zeros((h, w), dtype=np.float32)
    img_clean[8:56, 8:56] = 200.0
    data_clean = [build_data_item(img_clean, boundary, band, band_radius)]
    score_clean, sum_clean, _ = evaluate_one_candidate_mp(data_clean, settings, AUTO_DEFAULTS)
    cases.append(("clean_rect", data_clean, score_clean, sum_clean))

    # 2) Open boundary: two parallel lines (many endpoints)
    img_open = np.zeros((h, w), dtype=np.float32)
    img_open[10:54, 12:52] = 200.0
    b_open = np.zeros((h, w), dtype=bool)
    b_open[10, 12:52] = True
    b_open[53, 12:52] = True
    band_open = det.dilate_binary(b_open, band_radius)
    data_open = [build_data_item(img_open, b_open, band_open, band_radius)]
    score_open, sum_open, _ = evaluate_one_candidate_mp(data_open, settings, AUTO_DEFAULTS)
    cases.append(("open_boundary", data_open, score_open, sum_open))

    # 3) Cross / T-junction: more branch points
    img_cross = np.zeros((h, w), dtype=np.float32)
    img_cross[8:56, 8:56] = 200.0
    img_cross[28:36, :] = 200.0
    img_cross[:, 28:36] = 200.0
    data_cross = [build_data_item(img_cross, boundary, band, band_radius)]
    score_cross, sum_cross, _ = evaluate_one_candidate_mp(data_cross, settings, AUTO_DEFAULTS)
    cases.append(("cross_branch", data_cross, score_cross, sum_cross))

    # 4) Jagged boundary: more non-isolated pixels (wrinkle-like)
    img_jag = np.zeros((h, w), dtype=np.float32)
    for i in range(8, 56):
        j = 8 + (i % 5) * 2
        if j < 56:
            img_jag[i, j:j + 4] = 200.0
    img_jag[8:56, 8:12] = 200.0
    img_jag[8:56, 52:56] = 200.0
    data_jag = [build_data_item(img_jag, boundary, band, band_radius)]
    score_jag, sum_jag, _ = evaluate_one_candidate_mp(data_jag, settings, AUTO_DEFAULTS)
    cases.append(("jagged", data_jag, score_jag, sum_jag))

    # 5) Interior texture: may increase intrusion
    img_intr = np.zeros((h, w), dtype=np.float32)
    img_intr[8:56, 8:56] = 200.0
    for _ in range(80):
        r, c = np.random.randint(12, 52, size=2)
        img_intr[r:r+2, c:c+2] = 80.0
    data_intr = [build_data_item(img_intr, boundary, band, band_radius)]
    score_intr, sum_intr, _ = evaluate_one_candidate_mp(data_intr, settings, AUTO_DEFAULTS)
    cases.append(("interior_texture", data_intr, score_intr, sum_intr))

    # 6) Thick edge: wider bright band to simulate thick boundary
    img_thick = np.zeros((h, w), dtype=np.float32)
    img_thick[6:58, 6:58] = 200.0
    img_thick[8:56, 8:56] = 220.0
    data_thick = [build_data_item(img_thick, boundary, band, band_radius)]
    score_thick, sum_thick, _ = evaluate_one_candidate_mp(data_thick, settings, AUTO_DEFAULTS)
    cases.append(("thick_edge", data_thick, score_thick, sum_thick))

    # Save one sample image per case for inspection
    for name, data_list, _, _ in cases:
        arr = np.clip(data_list[0]["image"], 0, 255).astype(np.uint8)
        path = os.path.join(SYNTH_DIR, f"{name}.png")
        Image.fromarray(arr).save(path)

    return cases


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== 1) Sensitivity: score drop when each factor set to 'bad' ===\n")
    score_good, sensitivity = run_sensitivity_analysis()
    print(f"Score (all good): {score_good:.4f}\n")
    for r in sensitivity:
        flag = "KEEP" if r["has_impact"] else "LOW IMPACT"
        print(f"  {r['factor']:14s}  drop={r['score_drop']:.4f} ({r['score_drop_pct']:.1f}%)  -> {flag}")
    low_impact = [r["factor"] for r in sensitivity if not r["has_impact"]]
    print(f"\nLow-impact factors (drop < {IMPACT_THRESHOLD_PCT}%): {low_impact}")

    print("\n=== 2) Synthetic image metrics (same params) ===\n")
    metric_names = ["endpoints", "branch", "wrinkle", "thickness", "intrusion", "outside", "coverage", "gap", "continuity", "band_ratio", "excess_dots"]
    image_metrics = {k: [] for k in metric_names}
    try:
        cases = generate_synthetic_images()
        for name, _, score, summary in cases:
            print(f"  {name:20s} score={score:.4f}  end={summary['endpoints']:.3f} br={summary['branch']:.3f} "
                  f"wrk={summary['wrinkle']:.3f} thick={summary['thickness']:.3f} intr={summary['intrusion']:.3f}")
            for k in metric_names:
                if k in summary:
                    image_metrics[k].append(summary[k])
    except Exception as e:
        print(f"  Synthetic image run failed: {e}")

    # Discriminative power: factors with near-zero variance across images don't help choose params
    print("\n=== 3) Metric variance across synthetic images (discriminative power) ===\n")
    variances = {}
    for k in metric_names:
        vals = image_metrics.get(k, [])
        if len(vals) >= 2:
            variances[k] = float(np.var(vals))
        else:
            variances[k] = 0.0
    for k in sorted(variances, key=lambda x: -variances[x]):
        print(f"  {k:14s} var={variances[k]:.6f}")
    # Suggest removing factors that have very low variance (never discriminate in our tests)
    VAR_THRESHOLD = 0.002
    low_variance = [k for k, v in variances.items() if v < VAR_THRESHOLD]
    print(f"\nLow-variance (weak discriminative power, var < {VAR_THRESHOLD}): {low_variance}")

    # Also: gap is redundant with coverage (gap = 1 - coverage). Prefer removing gap from score.
    redundant = ["gap"] if "gap" in metric_names else []

    report = {
        "score_all_good": score_good,
        "impact_threshold_pct": IMPACT_THRESHOLD_PCT,
        "sensitivity": sensitivity,
        "factors_to_remove": low_impact,
        "metric_variances": variances,
        "low_variance_factors": low_variance,
        "redundant_factors": redundant,
    }
    out_path = os.path.join(OUT_DIR, "penalty_analysis_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved: {out_path}")
    return low_impact


if __name__ == "__main__":
    main()
