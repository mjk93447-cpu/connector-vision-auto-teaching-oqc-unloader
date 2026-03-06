#!/usr/bin/env python
"""
Pin Detection GUI launcher — for PyInstaller EXE build.
Usage: python run_pin_gui.py

Offline-first: Set YOLO_OFFLINE before any ultralytics import to block
network calls (model download, pip check, font fetch, etc.).
"""
import os
os.environ["YOLO_OFFLINE"] = "true"

from pin_detection.gui import main

if __name__ == "__main__":
    main()
