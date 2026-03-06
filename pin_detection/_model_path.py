"""Resolve YOLO26n model path for offline EXE (bundled) or dev (cache)."""
import sys
from pathlib import Path


def get_yolo26n_path() -> str:
    """
    Return path to yolo26n.pt for training.
    - PyInstaller EXE: use bundled models/yolo26n.pt from _MEIPASS
    - Dev: use "yolo26n.pt" (ultralytics cache, requires prior download when online)
    """
    if getattr(sys, "frozen", False):
        base = Path(sys._MEIPASS)
        bundled = base / "models" / "yolo26n.pt"
        if bundled.exists():
            return str(bundled)
    return "yolo26n.pt"
