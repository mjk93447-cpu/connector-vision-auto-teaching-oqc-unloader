"""
Reproduce EXE train crash (Action #24): Building dataset (21/21) → silent exit.

Environment: PyInstaller EXE, GUI Train tab, 21 images.
Crash: When dataset prep completes, model.train() spawns workers → EXE exits.

Fix: workers=0 when sys.frozen (train.py). Run this script to verify the flow
completes without crash. In Python, workers>0 works; in EXE, workers=0 is forced.

Usage:
  python tools_scripts/repro_exe_train_crash.py
  # Simulates EXE: set sys.frozen=True, run train flow with 21 images
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    # Simulate EXE: frozen mode forces workers=0
    sys.frozen = True
    if not hasattr(sys, "_MEIPASS"):
        sys._MEIPASS = str(ROOT)  # get_yolo26n_path needs this when frozen
    try:
        from tools_scripts.generate_pin_test_data import generate_connector_image, bbox_to_green_region
        from pin_detection.train import train_pin_model, _default_workers
        from PIL import Image

        # Verify workers=0 in frozen mode
        assert _default_workers() == 0, "frozen mode must use workers=0"

        # Generate 21 pairs (matching user's 21 images)
        data_dir = ROOT / "test_data" / "exe_crash_repro"
        u_dir = data_dir / "unmasked"
        m_dir = data_dir / "masked"
        u_dir.mkdir(parents=True, exist_ok=True)
        m_dir.mkdir(parents=True, exist_ok=True)

        for i in range(21):
            img, bboxes, _ = generate_connector_image(seed=300 + i, blur_prob=0.2, n_fake_pins=4)
            stem = f"{i+1:02d}"
            Image.fromarray(img).save(u_dir / f"{stem}.jpg")
            masked = img.copy()
            for x1, y1, x2, y2 in bboxes:
                bbox_to_green_region(masked, x1, y1, x2, y2)
            Image.fromarray(masked).save(m_dir / f"{stem}.jpg")

        out_dir = ROOT / "pin_models_exe_crash_repro"
        out_dir.mkdir(parents=True, exist_ok=True)

        def on_progress(c, t, p):
            print(f"Building dataset ({c}/{t}) — {p.name}")

        model_path = train_pin_model(
            unmasked_dir=str(u_dir),
            masked_dir=str(m_dir),
            output_dir=str(out_dir),
            epochs=2,
            imgsz=640,
            workers=4,  # Will be forced to 0 in frozen mode
            val_split=0.2,
            use_roi=True,
            on_progress=on_progress,
        )
        print(f"OK: Model saved {model_path}")
        (out_dir / "repro_ok.txt").write_text("OK")
        return 0
    finally:
        if hasattr(sys, "frozen"):
            del sys.frozen


if __name__ == "__main__":
    raise SystemExit(main())
