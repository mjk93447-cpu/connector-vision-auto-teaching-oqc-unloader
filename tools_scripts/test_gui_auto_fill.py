"""
Verify GUI auto-fills test data on startup (avoids Select folder first during EXE testing).
Run from project root. Requires test_data to exist.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Ensure test_data exists
if not (ROOT / "test_data" / "pin_large_factory" / "unmasked").exists():
    if not (ROOT / "test_data" / "pin_synthetic" / "train" / "unmasked").exists():
        print("Generating test_data...")
        import subprocess
        subprocess.run([
            sys.executable, "tools_scripts/generate_pin_test_data.py",
            "--output-dir", "test_data/pin_large_factory",
            "--large-factory", "--large-factory-n", "5"
        ], cwd=ROOT, check=True)

def main():
    import tkinter as tk
    from pin_detection.gui import PinDetectionGUI, _resolve_synthetic_paths

    paths = _resolve_synthetic_paths()
    if not paths:
        print("SKIP: test_data not found")
        return 0

    root = tk.Tk()
    root.withdraw()
    app = PinDetectionGUI(root)
    u = app.unmasked_dir.get().strip()
    m = app.masked_dir.get().strip()
    o = app.output_dir.get().strip()
    root.destroy()

    if u and m and o:
        print("OK: GUI auto-filled paths on startup")
        print(f"  Unmasked: {u}")
        print(f"  Masked: {m}")
        print(f"  Output: {o}")
        return 0
    print("FAIL: Paths not auto-filled")
    return 1

if __name__ == "__main__":
    sys.exit(main())
