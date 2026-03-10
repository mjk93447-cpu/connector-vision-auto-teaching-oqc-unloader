#!/usr/bin/env python
"""
Pin Detection GUI launcher — for PyInstaller EXE build.
Usage: python run_pin_gui.py

Offline-first: Set YOLO_OFFLINE before any ultralytics import to block
network calls (model download, pip check, font fetch, etc.).

EXE fix (ROADMAP 10.11a, 2026-03-06): When console=False (GUI mode),
sys.stdout/stderr are None. Ultralytics YOLO training writes progress
to stdout → AttributeError. Redirect None streams to devnull before
any imports. See docs/EXE_NONETYPE_FIX_PLAN.md, docs/TROUBLESHOOTING.md.
"""
import os
import sys

os.environ["YOLO_OFFLINE"] = "true"

# Fix PyInstaller GUI: stdout/stderr are None when console=False
# PIN_DEBUG=1: redirect stderr to %TEMP%/pin_train_debug.log to capture crash tracebacks
_debug = os.environ.get("PIN_DEBUG", "").strip() in ("1", "true", "yes")
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    if _debug:
        try:
            _log = os.path.join(os.environ.get("TEMP", os.environ.get("TMP", ".")), "pin_train_debug.log")
            sys.stderr = open(_log, "a", encoding="utf-8")
        except Exception:
            sys.stderr = open(os.devnull, "w")
    else:
        sys.stderr = open(os.devnull, "w")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Windows multiprocessing 필수 (EXE workers 가능성)
    if len(sys.argv) > 1 and sys.argv[1] == "--test-train":
        # Headless training test (for EXE verification). Run from project root.
        from pathlib import Path
        root = Path.cwd()
        unmasked = root / "test_data" / "pin_synthetic" / "unmasked"
        masked = root / "test_data" / "pin_synthetic" / "masked"
        if not unmasked.exists():
            unmasked = root / "test_data" / "pin_synthetic" / "train" / "unmasked"
            masked = root / "test_data" / "pin_synthetic" / "train" / "masked"
        out = root / "pin_models_exe_test"
        if unmasked.exists() and masked.exists():
            from pin_detection.train import train_pin_model
            train_pin_model(unmasked_dir=unmasked, masked_dir=masked, output_dir=out, epochs=3)
            (out / "exe_test_ok.txt").write_text("OK")
        sys.exit(0)
    from pin_detection.gui import main
    main()
