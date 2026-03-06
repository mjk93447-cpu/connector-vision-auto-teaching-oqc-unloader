#!/usr/bin/env python
"""
Pin Detection GUI launcher — for PyInstaller EXE build.
Usage: python run_pin_gui.py

Offline-first: Set YOLO_OFFLINE before any ultralytics import to block
network calls (model download, pip check, font fetch, etc.).

EXE fix: When console=False (GUI mode), sys.stdout/stderr are None.
Ultralytics YOLO training writes progress to stdout → AttributeError.
Redirect None streams to devnull before any imports.
"""
import os
import sys

os.environ["YOLO_OFFLINE"] = "true"

# Fix PyInstaller GUI: stdout/stderr are None when console=False
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test-train":
        # Headless training test (for EXE verification). Run from project root.
        from pathlib import Path
        root = Path.cwd()
        unmasked = root / "test_data" / "pin_synthetic" / "unmasked"
        masked = root / "test_data" / "pin_synthetic" / "masked"
        out = root / "pin_models_exe_test"
        if unmasked.exists() and masked.exists():
            from pin_detection.train import train_pin_model
            train_pin_model(unmasked_dir=unmasked, masked_dir=masked, output_dir=out, epochs=3)
            (out / "exe_test_ok.txt").write_text("OK")
        sys.exit(0)
    from pin_detection.gui import main
    main()
