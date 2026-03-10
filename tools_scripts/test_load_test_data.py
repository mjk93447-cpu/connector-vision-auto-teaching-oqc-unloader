"""Test Load test data path resolution. Run from project root."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pin_detection.gui import _resolve_synthetic_paths

# From project root
p = _resolve_synthetic_paths()
assert p, "Should find test_data from project root"
u, m, o = p
assert Path(u).exists(), f"Unmasked not found: {u}"
assert Path(m).exists(), f"Masked not found: {m}"
print("OK: Load test data paths resolved")
print(f"  Unmasked: {u}")
print(f"  Masked: {m}")
print(f"  Output: {o}")
