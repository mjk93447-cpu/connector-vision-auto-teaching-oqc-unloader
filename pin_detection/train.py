"""
Train YOLO26 model from masked/unmasked image pairs + Excel.
Supports 1 pair or 10 pairs (unmasked-dir + masked-dir).
"""
import os
from pathlib import Path

from ultralytics import YOLO


def _default_workers() -> int:
    """Use multi-core for data loading. Cap at 8 on Windows."""
    n = os.cpu_count() or 4
    return min(n, 8)


def train_pin_model(
    unmasked_path: str | Path | None = None,
    masked_path: str | Path | None = None,
    unmasked_dir: str | Path | None = None,
    masked_dir: str | Path | None = None,
    excel_path: str | Path | None = None,
    output_dir: str | Path = "pin_models",
    epochs: int = 100,
    imgsz: int = 640,
    workers: int | None = None,
) -> Path:
    """
    Train pin detection model.
    - Single pair: --unmasked X --masked Y
    - 10 pairs: --unmasked-dir X --masked-dir Y (paired by filename)
    Excel is for format reference; training uses image annotations.
    """
    from .dataset import prepare_yolo_dataset, prepare_yolo_dataset_from_dirs

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_dir / "dataset"

    if unmasked_dir and masked_dir:
        data_yaml = prepare_yolo_dataset_from_dirs(
            Path(unmasked_dir), Path(masked_dir), dataset_dir
        )
    elif unmasked_path and masked_path:
        data_yaml = prepare_yolo_dataset(
            Path(unmasked_path), Path(masked_path), dataset_dir
        )
    else:
        raise ValueError("Use --unmasked/--masked or --unmasked-dir/--masked-dir")

    model = YOLO("yolo26n.pt")  # nano for small objects, fast
    n_workers = workers if workers is not None else _default_workers()

    # Recall: 20+20 pins must not be missed. Precision: no over-detection.
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        workers=n_workers,
        project=str(output_dir),
        name="pin_run",
        exist_ok=True,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5,
        translate=0.05,
        scale=0.3,
        fliplr=0.5,
        mosaic=0.5,  # 소형 객체: mosaic 낮춰 localization 정확도 유지
        copy_paste=0.1,  # 소형 객체 증강
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = Path(results.save_dir) / "weights" / "last.pt"
    return best_pt
