"""
YOLO EXE validation: generate synthetic data, train 10 epochs, run inference on noisy test set.
Saves results to yolo_result/ and analyzes 40-pin detection accuracy.
Run: python tools_scripts/run_yolo_exe_validation.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def generate_train_data():
    """Generate 20 train pairs (unmasked + masked)."""
    from tools_scripts.generate_pin_test_data import generate_connector_image, bbox_to_green_region

    out = ROOT / "test_data" / "yolo_exe_validation" / "train"
    u_dir = out / "unmasked"
    m_dir = out / "masked"
    u_dir.mkdir(parents=True, exist_ok=True)
    m_dir.mkdir(parents=True, exist_ok=True)

    for i in range(20):
        img, bboxes, _ = generate_connector_image(seed=100 + i, blur_prob=0.2, n_fake_pins=4)
        stem = f"{i+1:02d}"
        Image.fromarray(img).save(u_dir / f"{stem}.jpg")
        masked = img.copy()
        for x1, y1, x2, y2 in bboxes:
            bbox_to_green_region(masked, x1, y1, x2, y2)
        Image.fromarray(masked).save(m_dir / f"{stem}.jpg")
    print(f"Generated 20 train pairs in {out}")
    return out


def generate_noisy_test_data():
    """Generate 10 noisy test pairs (high blur, more fake pins)."""
    from tools_scripts.generate_pin_test_data import generate_connector_image, bbox_to_green_region

    out = ROOT / "test_data" / "yolo_exe_validation" / "test_noisy"
    u_dir = out / "unmasked"
    m_dir = out / "masked"
    u_dir.mkdir(parents=True, exist_ok=True)
    m_dir.mkdir(parents=True, exist_ok=True)

    for i in range(10):
        img, bboxes, _ = generate_connector_image(
            seed=200 + i, blur_prob=0.6, n_fake_pins=12
        )
        stem = f"noisy_{i+1:02d}"
        Image.fromarray(img).save(u_dir / f"{stem}.jpg")
        masked = img.copy()
        for x1, y1, x2, y2 in bboxes:
            bbox_to_green_region(masked, x1, y1, x2, y2)
        Image.fromarray(masked).save(m_dir / f"{stem}.jpg")
    print(f"Generated 10 noisy test pairs in {out}")
    return out


def train_model(train_dir: Path, epochs: int = 10) -> Path:
    """Train YOLO model. Returns path to best.pt."""
    from pin_detection.train import train_pin_model

    u_dir = train_dir / "unmasked"
    m_dir = train_dir / "masked"
    out_dir = ROOT / "pin_models" / "yolo_exe_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = train_pin_model(
        unmasked_dir=str(u_dir),
        masked_dir=str(m_dir),
        output_dir=str(out_dir),
        epochs=epochs,
        imgsz=640,
        val_split=0.2,
        use_roi=True,
    )
    print(f"Model saved: {model_path}")
    return model_path


def run_inference_and_save(model_path: Path, test_dir: Path, result_dir: Path) -> dict:
    """Run inference on test images, save to yolo_result/, return analysis."""
    from pin_detection.inference import run_inference, split_upper_lower
    from pin_detection.annotation import masked_array_to_annotations

    result_dir.mkdir(parents=True, exist_ok=True)
    u_dir = test_dir / "unmasked"
    m_dir = test_dir / "masked"

    analysis = {"images": [], "summary": {"ok": 0, "ng": 0, "total": 0}}

    for u_path in sorted(u_dir.glob("*.jpg")):
        stem = u_path.stem
        m_path = m_dir / f"{stem}.jpg"
        if not m_path.exists():
            m_path = m_dir / u_path.name
        if not m_path.exists():
            continue

        img, detections, masked = run_inference(
            model_path=str(model_path),
            image_path=str(u_path),
            output_image_path=str(result_dir / f"{stem}_masked.jpg"),
            conf_threshold=0.01,
        )

        upper, lower = split_upper_lower(detections)
        n_upper, n_lower = len(upper), len(lower)
        ok = n_upper == 20 and n_lower == 20

        # Ground truth from masked
        _, gt_anns = masked_array_to_annotations(np.array(Image.open(m_path).convert("RGB")))
        n_gt = len(gt_anns)

        analysis["images"].append({
            "file": u_path.name,
            "detected_upper": n_upper,
            "detected_lower": n_lower,
            "detected_total": len(detections),
            "gt_total": n_gt,
            "ok": ok,
        })
        analysis["summary"]["total"] += 1
        if ok:
            analysis["summary"]["ok"] += 1
        else:
            analysis["summary"]["ng"] += 1

        print(f"  {u_path.name}: upper={n_upper}, lower={n_lower} -> {'OK' if ok else 'NG'}")

    return analysis


def main() -> int:
    print("=== YOLO EXE Validation ===\n")

    # 1. Generate data
    train_dir = generate_train_data()
    test_dir = generate_noisy_test_data()

    # 2. Train
    print("\n--- Training (10 epochs) ---")
    model_path = train_model(train_dir, epochs=10)

    # 3. Inference
    result_dir = ROOT / "yolo_result"
    print("\n--- Inference on noisy test set ---")
    analysis = run_inference_and_save(model_path, test_dir, result_dir)

    # 4. Summary
    s = analysis["summary"]
    print(f"\n=== Result: {s['ok']}/{s['total']} OK ===")
    if s["ng"] > 0:
        print("NG images:")
        for img in analysis["images"]:
            if not img["ok"]:
                print(f"  {img['file']}: upper={img['detected_upper']}, lower={img['detected_lower']}")

    # Save analysis JSON
    with open(result_dir / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nResults saved to {result_dir}")
    print(f"Analysis: {result_dir / 'analysis.json'}")

    return 0 if s["ok"] == s["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
