"""
Full pin detection pipeline: generate data -> train -> eval (mAP50, confusion matrix).
Usage: python -m tools_scripts.run_full_pin_pipeline [--skip-train] [--epochs N]
"""
import argparse
import subprocess
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-generate", action="store_true", help="Skip data generation")
    parser.add_argument("--skip-train", action="store_true", help="Skip training")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--train-pairs", type=int, default=40)
    parser.add_argument("--test-pairs", type=int, default=20)
    args = parser.parse_args()

    root = _project_root()
    data_dir = root / "test_data" / "pin_synthetic"
    train_unmasked = data_dir / "train" / "unmasked"
    train_masked = data_dir / "train" / "masked"
    test_unmasked = data_dir / "test" / "unmasked"
    test_masked = data_dir / "test" / "masked"
    output_dir = root / "pin_models"
    model_path = root / "runs" / "detect" / "pin_models" / "pin_run" / "weights" / "best.pt"
    eval_save = root / "pin_eval_results"

    if not args.skip_generate:
        print("Step 1: Generating train + test data...")
        r = subprocess.run([
            sys.executable, "-m", "tools_scripts.generate_pin_test_data",
            "--output-dir", str(data_dir),
            "--train-pairs", str(args.train_pairs),
            "--test-pairs", str(args.test_pairs),
        ], cwd=root)
        if r.returncode != 0:
            return r.returncode

    if not args.skip_train:
        print("Step 2: Training YOLO model (train/val split 80/20)...")
        r = subprocess.run([
            sys.executable, "tools.py", "pin", "train",
            "--unmasked-dir", str(train_unmasked),
            "--masked-dir", str(train_masked),
            "--output-dir", str(output_dir),
            "--epochs", str(args.epochs),
            "--val-split", "0.2",
        ], cwd=root)
        if r.returncode != 0:
            return r.returncode
    else:
        if not model_path.exists():
            model_path = root / "runs" / "detect" / "pin_models_test" / "pin_run" / "weights" / "best.pt"
        if not model_path.exists():
            print("ERROR: No model found. Run without --skip-train.")
            return 1

    if not model_path.exists():
        model_path = root / "runs" / "detect" / "pin_models" / "pin_run" / "weights" / "best.pt"
    if not model_path.exists():
        print("ERROR: Model not found at", model_path)
        return 1

    print("Step 3: Evaluating on test set (Recall, Precision, mAP50, Confusion Matrix)...")
    r = subprocess.run([
        sys.executable, "tools.py", "pin", "eval",
        "--model", str(model_path),
        "--unmasked-dir", str(test_unmasked),
        "--masked-dir", str(test_masked),
        "--conf", "0.01",
        "--max-dist", "40",
        "--run-map50",
        "--save-dir", str(eval_save),
    ], cwd=root)
    if r.returncode != 0:
        return r.returncode

    print("\nPipeline complete. Results saved to", eval_save)
    return 0


if __name__ == "__main__":
    sys.exit(main())
