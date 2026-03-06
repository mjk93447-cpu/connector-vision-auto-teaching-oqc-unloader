"""
Test that run_pin_gui handles None stdout/stderr (PyInstaller GUI mode).
Simulates: sys.stdout=None, sys.stderr=None before import.
"""
import os
import sys

# Ensure project root in path (CI may run from different cwd)
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
os.chdir(_root)


def test_stdout_stderr_fix():
    """Simulate EXE: set stdout/stderr to None, then import run_pin_gui.
    Should not raise AttributeError."""
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    try:
        sys.stdout = None
        sys.stderr = None
        import run_pin_gui  # noqa: F401
        if sys.stdout is None or sys.stderr is None:
            return 1, "stdout/stderr not replaced by run_pin_gui"
        return 0, None
    except AttributeError:
        return 1, "AttributeError when stdout/stderr are None"
    except Exception as e:
        return 1, str(e)
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr


def main():
    code, msg = test_stdout_stderr_fix()
    if code == 0:
        print("OK: run_pin_gui handles None stdout/stderr")
    else:
        print(f"FAIL: {msg or 'unknown'}")
    return code


if __name__ == "__main__":
    sys.exit(main())
