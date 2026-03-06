"""
Reproduce EXE training with None stdout/stderr.
Simulates PyInstaller console=False: stdout/stderr are None.
Runs actual training to verify no 'NoneType' object has no attribute 'write'.
"""
import os
import sys

# Simulate EXE: set stdout/stderr to None BEFORE any other imports
sys.stdout = None
sys.stderr = None

# Now import run_pin_gui - it will replace None with devnull
import run_pin_gui  # noqa: F401

# After import, stdout/stderr are fixed. Run training.
# We need to run train_pin_model directly (run_pin_gui only has main() for GUI)
from pathlib import Path

# Use test data
root = Path(__file__).resolve().parent.parent
unmasked = root / "test_data" / "pin_synthetic" / "unmasked"
masked = root / "test_data" / "pin_synthetic" / "masked"
out = root / "pin_models_exe_repro_test"

if __name__ == "__main__":
    from pin_detection.train import train_pin_model
    model_path = train_pin_model(
        unmasked_dir=unmasked,
        masked_dir=masked,
        output_dir=out,
        epochs=3,
    )
    # Report success (stdout is devnull; write to file for verification)
    (out / "repro_success.txt").write_text(f"Model: {model_path}\n")
