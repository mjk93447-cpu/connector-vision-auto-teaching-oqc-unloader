"""
Test that run_pin_gui handles None stdout/stderr (PyInstaller GUI mode).
Simulates: sys.stdout=None, sys.stderr=None before import.
"""
import os
import sys


def test_stdout_stderr_fix():
    """Simulate EXE: set stdout/stderr to None, then import run_pin_gui.
    Should not raise AttributeError."""
    # Save originals
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    try:
        sys.stdout = None
        sys.stderr = None
        # This would normally crash when ultralytics tries to write
        # run_pin_gui fixes this at import time
        import run_pin_gui  # noqa: F401
        # If we get here without crash, fix is applied
        assert sys.stdout is not None, "stdout should be replaced"
        assert sys.stderr is not None, "stderr should be replaced"
        return 0
    except AttributeError as e:
        return 1
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr


def main():
    code = test_stdout_stderr_fix()
    # stdout restored in test's finally
    if code == 0:
        print("OK: run_pin_gui handles None stdout/stderr")
    else:
        print("FAIL: AttributeError when stdout/stderr are None")
    return code


if __name__ == "__main__":
    sys.exit(main())
