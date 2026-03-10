"""
Run all verification steps for pin detection development.
Use when agent needs to validate changes before commit or handoff.

Usage:
  python tools_scripts/run_all_verification.py [--quick] [--skip-exe]

  --quick: Skip slow tests (exe_large_train, benchmark)
  --skip-exe: Skip EXE simulation scripts (repro_exe_train_crash, test_exe_large_train)
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run(cmd: list[str], desc: str) -> tuple[int, str]:
    """Run command, return (exit_code, output)."""
    try:
        r = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        out = (r.stdout or "") + (r.stderr or "")
        return r.returncode, out
    except subprocess.TimeoutExpired:
        return 1, "Timeout"
    except Exception as e:
        return 1, str(e)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Skip slow tests")
    parser.add_argument("--skip-exe", action="store_true", help="Skip EXE simulation")
    args = parser.parse_args()

    failed = []
    steps = [
        ("pytest tests/", [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"]),
        ("test_exe_stdout_fix", [sys.executable, "tools_scripts/test_exe_stdout_fix.py"]),
    ]

    if not args.skip_exe:
        steps.append(("repro_exe_train_crash", [sys.executable, "tools_scripts/repro_exe_train_crash.py"]))
        if not args.quick:
            steps.append(("test_exe_large_train", [sys.executable, "tools_scripts/test_exe_large_train.py"]))

    if not args.quick:
        # Use train subdir (generate_pin_test_data creates train/unmasked, train/masked)
        bench_unmasked = ROOT / "test_data" / "pin_synthetic" / "train" / "unmasked"
        bench_masked = ROOT / "test_data" / "pin_synthetic" / "train" / "masked"
        if bench_unmasked.exists() and bench_masked.exists():
            steps.append(("benchmark_train_speed (2 epochs)", [
                sys.executable, "tools_scripts/benchmark_train_speed.py",
                "--epochs", "2", "--mosaic", "0",
                "--unmasked", str(bench_unmasked), "--masked", str(bench_masked),
            ]))
        else:
            print("\n--- benchmark_train_speed (skipped: test_data/pin_synthetic/train not found) ---")

    for desc, cmd in steps:
        print(f"\n--- {desc} ---")
        code, out = run(cmd, desc)
        if code != 0:
            print(out[:2000] if len(out) > 2000 else out)
            failed.append(desc)
        else:
            print("OK")

    print("\n" + "=" * 50)
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print("All verification steps passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
