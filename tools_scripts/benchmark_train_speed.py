"""
Benchmark training speed: measure training time for pin dataset.
Usage: python tools_scripts/benchmark_train_speed.py [--unmasked DIR] [--masked DIR] [--epochs N]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pin_detection.train import train_pin_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unmasked", default="test_data/pin_synthetic/unmasked", help="Unmasked image dir")
    parser.add_argument("--masked", default="test_data/pin_synthetic/masked", help="Masked image dir")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--workers", type=int, default=4, help="Workers")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--mosaic", type=float, default=0.0, help="Mosaic aug (0=fast)")
    parser.add_argument("--output", default="pin_models_benchmark", help="Output dir")
    args = parser.parse_args()

    u = Path(args.unmasked)
    m = Path(args.masked)
    if not u.exists() or not m.exists():
        print(f"Error: dirs not found: {u}, {m}")
        return 1

    print(f"Training: {u} + {m}")
    print(f"  epochs={args.epochs}, imgsz={args.imgsz}, workers={args.workers}, batch={args.batch}, mosaic={args.mosaic}")
    t0 = time.perf_counter()
    model_path = train_pin_model(
        unmasked_dir=u,
        masked_dir=m,
        output_dir=args.output,
        epochs=args.epochs,
        imgsz=args.imgsz,
        workers=args.workers,
        batch=args.batch,
        mosaic=args.mosaic,
        cache=True,
        rect=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"Done: {model_path}")
    print(f"Time: {elapsed:.1f} sec ({elapsed/60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
