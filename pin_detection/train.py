"""
Train YOLO26 model from masked/unmasked image pairs.
Supports 1 pair or 10 pairs (unmasked-dir + masked-dir).

EXE fix (Action #24): PyInstaller EXE + workers>0 causes silent crash on Windows
when dataset prep completes and model.train() spawns DataLoader workers.
Use workers=0 in frozen/EXE mode to avoid multiprocessing spawn.
"""
import os
import sys
import threading
from pathlib import Path
from typing import Callable

from ultralytics import YOLO


def _default_workers() -> int:
    """Use multi-core for data loading. Cap at 4 on Windows for spawn stability."""
    if getattr(sys, "frozen", False):
        return 0  # EXE: multiprocessing spawn crashes silently (Action #24)
    n = os.cpu_count() or 4
    return min(n, 4)


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
    val_split: float = 0.2,
    stop_event: threading.Event | None = None,
    batch: int = 16,
    cache: str | bool = True,
    mosaic: float = 0.0,
    rect: bool = True,
    use_roi: bool = True,
    on_progress: Callable[[int, int, Path], None] | None = None,
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
            Path(unmasked_dir), Path(masked_dir), dataset_dir,
            val_split=val_split, use_roi=use_roi, on_progress=on_progress,
        )
    elif unmasked_path and masked_path:
        data_yaml = prepare_yolo_dataset(
            Path(unmasked_path), Path(masked_path), dataset_dir, use_roi=use_roi
        )
    else:
        raise ValueError("Use --unmasked/--masked or --unmasked-dir/--masked-dir")

    from ._model_path import get_yolo26n_path
    model = YOLO(get_yolo26n_path())  # nano, bundled in EXE for offline
    n_workers = workers if workers is not None else _default_workers()
    if getattr(sys, "frozen", False) and n_workers > 0:
        n_workers = 0  # EXE: force 0 to avoid multiprocessing crash (Action #24)

    def _on_epoch_end(trainer):
        if stop_event and stop_event.is_set():
            trainer.stop_training = True
    if stop_event:
        model.add_callback("on_train_epoch_end", _on_epoch_end)

    # Recall: 20+20 pins must not be missed. Precision: no over-detection.
    # Speed: batch, cache, mosaic=0, rect=True, plots=False for faster training.
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        workers=n_workers,
        batch=batch,
        cache=cache,
        rect=rect,
        plots=False,
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
        mosaic=mosaic,
        copy_paste=0.0 if mosaic == 0 else 0.1,
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = Path(results.save_dir) / "weights" / "last.pt"
    return best_pt
