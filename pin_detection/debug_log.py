"""
EXE crash diagnosis: file-based debug log.
Writes to %TEMP%/pin_train_debug.log so we can see the last step before crash.
Survives segfaults and silent exits.
"""
import os
import sys
from datetime import datetime
from pathlib import Path


def _log_path() -> Path:
    return Path(os.environ.get("TEMP", os.environ.get("TMP", "/tmp"))) / "pin_train_debug.log"


def log_step(step: str, detail: str = "") -> None:
    """Append a step marker to debug log. Call at each phase."""
    try:
        p = _log_path()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {step}" + (f" - {detail}" if detail else "") + "\n"
        with open(p, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


def clear_log() -> None:
    """Clear debug log at start of training."""
    try:
        _log_path().write_text("", encoding="utf-8")
    except Exception:
        pass
