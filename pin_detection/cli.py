"""
CLI for pin detection: train and inference.
"""
import argparse
from pathlib import Path

from .inference import run_inference, split_upper_lower, compute_spacing_mm
from .excel_io import load_excel_format, write_result_excel


def _run_gui() -> int:
    from .gui import main
    main()
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .train import train_pin_model
    if args.unmasked_dir and args.masked_dir:
        model_path = train_pin_model(
            unmasked_dir=args.unmasked_dir,
            masked_dir=args.masked_dir,
            excel_path=args.excel,
            output_dir=args.output_dir,
            epochs=args.epochs,
            imgsz=args.imgsz,
            workers=args.workers,
            val_split=getattr(args, "val_split", 0.2),
        )
    elif args.unmasked and args.masked:
        model_path = train_pin_model(
            unmasked_path=args.unmasked,
            masked_path=args.masked,
            excel_path=args.excel,
            output_dir=args.output_dir,
            epochs=args.epochs,
            imgsz=args.imgsz,
            workers=args.workers,
        )
    else:
        raise SystemExit("Use --unmasked/--masked or --unmasked-dir/--masked-dir")
    print(f"Model saved: {model_path}")
    return 0


def cmd_inference(args: argparse.Namespace) -> int:
    img, detections, masked = run_inference(
        model_path=args.model,
        image_path=args.image,
        output_image_path=args.output_image,
        conf_threshold=args.conf,
    )
    h, w = img.shape[:2]
    upper, lower = split_upper_lower(detections)
    upper_spacings = compute_spacing_mm(upper, w)
    lower_spacings = compute_spacing_mm(lower, w)

    print(f"Upper pins: {len(upper)}, Lower pins: {len(lower)}")
    print(f"Judgment: {'OK' if len(upper) == 20 and len(lower) == 20 else 'NG'}")

    format_ref = None
    if args.excel_format:
        format_ref = load_excel_format(args.excel_format)

    excel_out = args.output_excel or (Path(args.image).parent / "result.xlsx")
    write_result_excel(
        excel_out,
        upper_count=len(upper),
        lower_count=len(lower),
        upper_spacings=upper_spacings,
        lower_spacings=lower_spacings,
        format_ref=format_ref,
    )
    print(f"Excel saved: {excel_out}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    import subprocess
    import sys
    root = Path(__file__).resolve().parent.parent
    cmd = [
        sys.executable, "-m", "tools_scripts.run_pin_experiment",
        "--model", str(args.model),
        "--unmasked-dir", str(args.unmasked_dir),
        "--masked-dir", str(args.masked_dir),
        "--conf", str(args.conf),
        "--iou", str(args.iou),
        "--max-dist", str(args.max_dist),
    ]
    if getattr(args, "run_map50", False):
        cmd.append("--run-map50")
    if getattr(args, "save_dir", None):
        cmd.extend(["--save-dir", str(args.save_dir)])
    return subprocess.run(cmd, cwd=root).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Connector pin detection (YOLO26)")
    sub = parser.add_subparsers(dest="cmd", help="Command")

    train_p = sub.add_parser("train", help="Train from masked/unmasked pair(s) + Excel")
    train_p.add_argument("--unmasked", help="Unmasked image (single pair)")
    train_p.add_argument("--masked", help="Masked image (single pair)")
    train_p.add_argument("--unmasked-dir", dest="unmasked_dir", help="Dir with 10 unmasked images")
    train_p.add_argument("--masked-dir", dest="masked_dir", help="Dir with 10 masked images")
    train_p.add_argument("--excel", help="Reference Excel or dir (format)")
    train_p.add_argument("--output-dir", default="pin_models", help="Output directory")
    train_p.add_argument("--epochs", type=int, default=100)
    train_p.add_argument("--imgsz", type=int, default=640)
    train_p.add_argument("--workers", type=int, default=None, help="Data loading workers (default: cpu_count)")
    train_p.add_argument("--val-split", type=float, default=0.2, dest="val_split", help="Validation split (0.2 = 80%% train)")
    train_p.set_defaults(func=cmd_train)

    gui_p = sub.add_parser("gui", help="Launch GUI (train/inference)")
    gui_p.set_defaults(func=lambda a: _run_gui())

    inf_p = sub.add_parser("inference", help="Run inference on image")
    inf_p.add_argument("--model", required=True, help="Trained model .pt path")
    inf_p.add_argument("--image", required=True, help="Input connector image")
    inf_p.add_argument("--output-image", help="Output masked image path")
    inf_p.add_argument("--output-excel", help="Output Excel path")
    inf_p.add_argument("--excel-format", help="Reference Excel for format (from training)")
    inf_p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    inf_p.set_defaults(func=cmd_inference)

    eval_p = sub.add_parser("eval", help="Evaluate Recall/Precision vs GT (masked images)")
    eval_p.add_argument("--model", required=True, help="Trained model .pt path")
    eval_p.add_argument("--unmasked-dir", dest="unmasked_dir", required=True, help="Unmasked images")
    eval_p.add_argument("--masked-dir", dest="masked_dir", required=True, help="Masked images (GT)")
    eval_p.add_argument("--conf", type=float, default=0.01, help="Confidence threshold")
    eval_p.add_argument("--iou", type=float, default=0, help="IoU threshold (0=use max-dist)")
    eval_p.add_argument("--max-dist", type=float, default=50, help="Max center distance (px) for matching")
    eval_p.add_argument("--run-map50", action="store_true", dest="run_map50", help="Run YOLO val for mAP50")
    eval_p.add_argument("--save-dir", dest="save_dir", help="Save metrics and confusion matrix JSON")
    eval_p.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
