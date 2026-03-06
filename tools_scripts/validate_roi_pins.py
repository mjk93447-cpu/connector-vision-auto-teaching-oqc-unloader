"""
Validate ROI extraction: ensure no pins are lost when cropping large images.
Compares annotation count before/after ROI crop.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pin_detection.annotation import masked_image_to_annotations, masked_array_to_annotations
from pin_detection.roi import extract_pin_roi, crop_to_roi
from PIL import Image
import numpy as np


def validate_roi(masked_path: Path, expected_pins: int = 40, margin_ratio: float = 0.15) -> tuple[bool, str]:
    """
    Returns (ok, message).
    """
    _, anns_before = masked_image_to_annotations(masked_path)
    n_before = len(anns_before)

    img = np.array(Image.open(masked_path).convert("RGB"))
    h, w = img.shape[:2]
    roi = extract_pin_roi(masked_path, margin_ratio=margin_ratio)
    cropped = crop_to_roi(img, roi)
    _, anns_after = masked_array_to_annotations(cropped)
    # Actually: masked_array_to_annotations takes array, so we pass cropped. But cropped is the masked region.
    # The annotations are extracted from the cropped masked image - so we're comparing:
    # - anns_before: from full masked image
    # - anns_after: from cropped masked image (only pins inside ROI)
    # If ROI cuts off pins, anns_after would have fewer. So we need to count pins that fall INSIDE roi.
    x1, y1, x2, y2 = roi
    n_inside = 0
    for xc, yc, bw, bh in anns_before:
        px = int(xc * w)
        py = int(yc * h)
        if x1 <= px < x2 and y1 <= py < y2:
            n_inside += 1
    # Pins inside ROI should all be in anns_after. anns_after is from cropped image so coords are relative to crop.
    # Actually the issue: after crop, we re-run masked_array_to_annotations on the cropped image.
    # The cropped image contains only the ROI region. So any pin that was inside ROI in the original
    # will appear in the cropped image. Pins outside ROI are physically cut off - they won't appear in cropped.
    # So n_after should equal number of pins that were inside ROI. If ROI is correct, n_inside should equal n_before
    # (all pins inside ROI). So we need: n_before == n_inside (ROI contains all pins) and n_after == n_inside.
    n_after = len(anns_after)

    if n_before != n_after:
        return False, f"Pin count mismatch: before={n_before} after_crop={n_after} (expected {expected_pins})"
    if n_before != expected_pins:
        return False, f"Unexpected pin count: {n_before} (expected {expected_pins})"
    if n_inside < n_before:
        return False, f"ROI cuts off pins: {n_inside} inside ROI, {n_before} total"
    return True, f"OK: {n_before} pins preserved"


def main() -> int:
    base = Path(__file__).resolve().parent.parent / "test_data"
    # Test pin_synthetic (640x480) - no ROI used (below threshold)
    synthetic = base / "pin_synthetic" / "masked"
    if synthetic.exists():
        print("=== pin_synthetic (small, ROI not applied) ===")
        for p in sorted(synthetic.glob("*.jpg"))[:3]:
            _, anns = masked_image_to_annotations(p)
            print(f"  {p.name}: {len(anns)} pins")

    # Test pin_large_5496x3672 - ROI applied
    large_dir = base / "pin_large_5496x3672" / "masked"
    if not large_dir.exists():
        print("pin_large_5496x3672 not found. Run: python tools_scripts/generate_large_pin_data.py")
        return 1

    print("\n=== pin_large_5496x3672 ROI validation (expected 40 pins each) ===")
    fails = []
    for p in sorted(large_dir.glob("*.jpg")):
        ok, msg = validate_roi(p, expected_pins=40)
        status = "PASS" if ok else "FAIL"
        print(f"  {p.name}: {status} - {msg}")
        if not ok:
            fails.append((p.name, msg))

    # Also test with smaller margin (edge case)
    print("\n=== Margin 0.10 (tighter) ===")
    for p in sorted(large_dir.glob("*.jpg"))[:3]:
        ok, msg = validate_roi(p, expected_pins=40, margin_ratio=0.10)
        print(f"  {p.name}: {'PASS' if ok else 'FAIL'} - {msg}")
        if not ok:
            fails.append((p.name, f"margin=0.10: {msg}"))

    if fails:
        print(f"\n{len(fails)} validation failures")
        return 1
    print("\nAll ROI validations passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
