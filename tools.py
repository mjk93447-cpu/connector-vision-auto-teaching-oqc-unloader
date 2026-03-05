"""
Unified CLI for eval, tuning, benchmark, and analysis tools.
Usage: python tools.py <subcommand> [args...]
"""
import argparse
import subprocess
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPT_MAP = {
    "eval": "tools_scripts.edge_performance_eval",
    "boundary": "tools_scripts.boundary_score_eval",
    "tune": "tools_scripts.run_target_score_tuning",
    "fast-tune": "tools_scripts.run_fast_target_score_test",
    "penalty": "tools_scripts.score_penalty_analysis",
    "branch": "tools_scripts.branch_endpoint_impact_test",
    "gpu": "tools_scripts.gpu_benchmark",
    "test-gui": "tools_scripts.test_auto_gui",
    "test-automated": "tools_scripts.test_auto_automated",
    "test-perf": "tools_scripts.test_auto_performance",
}


def _run_script(module_name: str, extra_args: list) -> int:
    """Run a script module with optional extra args."""
    os.chdir(PROJECT_ROOT)
    cmd = [sys.executable, "-m", module_name] + extra_args
    return subprocess.run(cmd, cwd=PROJECT_ROOT).returncode


def main():
    parser = argparse.ArgumentParser(
        description="Connector Vision Auto Teaching - eval, tune, benchmark, analysis"
    )
    sub = parser.add_subparsers(dest="cmd", help="Subcommand")
    for name, module in SCRIPT_MAP.items():
        sub.add_parser(name, help=f"Run {module}")
    pin_p = sub.add_parser("pin", help="Pin detection: train, inference (YOLO26)")
    pin_p.add_argument("extra", nargs=argparse.REMAINDER, help="train/inference subcommand and args")
    args, extra = parser.parse_known_args()

    if not args.cmd:
        parser.print_help()
        return 0

    if args.cmd == "pin":
        pin_extra = getattr(args, "extra", extra)
        return _run_script("pin_detection.cli", pin_extra)

    module = SCRIPT_MAP[args.cmd]
    return _run_script(module, extra)


if __name__ == "__main__":
    sys.exit(main() or 0)
