"""
EXE stress test: 5000×4000 large factory-like images with ROI.
Simulates realistic crash scenario: 20 images, 40 pins each, complex background, roi_map.
Runs with sys.frozen=True (cache=disk, workers=0). Verifies no crash.
Usage: python tools_scripts/test_exe_large_train.py
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Simulate EXE environment
sys.frozen = True
sys._MEIPASS = str(ROOT)


def main() -> int:
    data_dir = ROOT / "test_data" / "pin_large_factory"
    unmasked_dir = data_dir / "unmasked"
    masked_dir = data_dir / "masked"
    roi_src = data_dir / "roi_map.json"
    out_dir = ROOT / "pin_models_exe_large_test"

    # Generate data if missing
    if not unmasked_dir.exists() or not list(unmasked_dir.glob("*.jpg")):
        print("Generating 5000×4000 factory-like data...")
        from tools_scripts.generate_pin_test_data import _generate_large_factory
        data_dir.mkdir(parents=True, exist_ok=True)
        _generate_large_factory(data_dir, n_pairs=20)
    else:
        print(f"Using existing data: {unmasked_dir}")

    if not roi_src.exists():
        print("ERROR: roi_map.json not found. Run generate with --large-factory first.")
        return 1

    # Copy roi_map to output dir (training loads from output_dir.parent or output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(roi_src, out_dir / "roi_map.json")
    print(f"roi_map.json copied to {out_dir}")

    # Train (frozen mode: cache=disk, workers=0)
    from pin_detection.train import train_pin_model
    from pin_detection.debug_log import clear_log

    clear_log()
    print("Starting train (frozen mode, 2 epochs, imgsz=640)...")
    try:
        result = train_pin_model(
            unmasked_dir=str(unmasked_dir),
            masked_dir=str(masked_dir),
            output_dir=str(out_dir),
            epochs=2,
            imgsz=640,
            val_split=0.2,
            use_roi=True,
        )
        print(f"SUCCESS: best.pt at {result}")
        return 0
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
