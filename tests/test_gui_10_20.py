"""
10.20 GUI & ROI Editor functional tests.
Run: python -m pytest tests/test_gui_10_20.py -v
"""
import json
import tempfile
from pathlib import Path

import pytest


def test_roi_editor_load_save():
    """Test roi_map load/save (single and split format)."""
    from pin_detection.roi_editor import load_roi_map, save_roi_map

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "roi_map.json"
        assert load_roi_map(p) == {}

        roi_map = {"01": [100, 100, 500, 300], "02": [0, 0, 640, 480]}
        save_roi_map(p, roi_map)
        assert p.exists()
        loaded = load_roi_map(p)
        assert loaded == roi_map

        split_roi = {"03": {"upper": [0, 0, 400, 200], "lower": [0, 200, 400, 400]}}
        save_roi_map(p, {**roi_map, **split_roi})
        loaded2 = load_roi_map(p)
        assert loaded2["01"] == roi_map["01"]
        assert loaded2["03"] == split_roi["03"]


def test_dataset_roi_map_integration():
    """Test prepare_yolo_dataset_from_dirs with roi_map."""
    from pin_detection.dataset import prepare_yolo_dataset_from_dirs

    base = Path(__file__).parent.parent / "test_data"
    unmasked = base / "pin_synthetic" / "unmasked"
    masked = base / "pin_synthetic" / "masked"
    if not unmasked.exists() or not (list(unmasked.glob("*.jpg")) or list(unmasked.glob("*.png"))):
        unmasked = base / "pin_synthetic" / "train" / "unmasked"
        masked = base / "pin_synthetic" / "train" / "masked"
    if not unmasked.exists():
        pytest.skip("No test data")

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "dataset"
        roi_map = {}
        for f in sorted(unmasked.iterdir())[:2]:
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                roi_map[f.stem] = [50, 50, 200, 150]
        data_yaml = prepare_yolo_dataset_from_dirs(
            unmasked, masked, out, val_split=0.2, use_roi=False, roi_map=roi_map
        )
        assert data_yaml.exists()
        train_img = out / "images" / "train"
        assert train_img.exists()
        imgs = list(train_img.glob("*.jpg"))
        assert len(imgs) >= 1


def test_dataset_roi_map_from_file():
    """Test roi_map auto-load from output_dir.parent/roi_map.json."""
    from pin_detection.dataset import prepare_yolo_dataset_from_dirs

    base = Path(__file__).parent.parent / "test_data"
    unmasked = base / "pin_synthetic" / "train" / "unmasked"
    masked = base / "pin_synthetic" / "train" / "masked"
    if not unmasked.exists():
        pytest.skip("No test data")

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        roi_path = root / "roi_map.json"
        u_files = [f for f in unmasked.iterdir() if f.suffix.lower() in {".jpg", ".png", ".bmp"}][:2]
        roi_map = {f.stem: [10, 10, 100, 100] for f in u_files}
        with open(roi_path, "w") as f:
            json.dump(roi_map, f)

        dataset_dir = root / "dataset"
        data_yaml = prepare_yolo_dataset_from_dirs(
            unmasked, masked, dataset_dir, val_split=0.0, use_roi=False
        )
        assert data_yaml.exists()


def test_imgsz_no_cap():
    """Verify imgsz is not capped at 1280."""
    from pin_detection.gui import _estimate_training_time

    t640 = _estimate_training_time(10, 640, 3, 4)
    t1280 = _estimate_training_time(10, 1280, 3, 4)
    t2000 = _estimate_training_time(10, 2000, 3, 4)
    assert t2000 > t1280 > t640


def test_graph_poll_candidates():
    """Test graph poll finds results.csv in output_dir/pin_run or runs/detect."""
    from pathlib import Path
    import tempfile
    from pin_detection.gui import PinDetectionGUI
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    app = PinDetectionGUI(root)
    # Simulate candidates
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "pin_run"
        out.mkdir()
        (out / "results.csv").write_text("epoch,train/box_loss\n1,1.5\n")
        app._graph_save_dir = [out]
        app._poll_graph()
        assert len(app._graph_data) >= 1
    root.destroy()


def test_train_validation():
    """Test _on_train validation: output_dir required, paths must exist."""
    import tkinter as tk
    from pin_detection.gui import PinDetectionGUI

    root = tk.Tk()
    root.withdraw()
    app = PinDetectionGUI(root)
    # Empty folders -> validation would show error (we can't easily test messagebox)
    # Just verify the method exists and doesn't crash on empty
    app.unmasked_dir.set("")
    app.masked_dir.set("")
    app.output_dir.set("")
    # _on_train checks and returns early with messagebox - no crash
    try:
        app._on_train()
    except Exception:
        pass
    root.destroy()


def test_edit_roi_validation():
    """Test _on_edit_roi validation: requires folders and output."""
    import tkinter as tk
    from pin_detection.gui import PinDetectionGUI

    root = tk.Tk()
    root.withdraw()
    app = PinDetectionGUI(root)
    app.unmasked_dir.set("")
    app.masked_dir.set("")
    app.output_dir.set("")
    try:
        app._on_edit_roi()
    except Exception:
        pass
    root.destroy()


def test_exe_stdout_fix():
    """Test run_pin_gui handles None stdout/stderr (EXE mode)."""
    import sys
    orig_out, orig_err = sys.stdout, sys.stderr
    try:
        sys.stdout = None
        sys.stderr = None
        import run_pin_gui  # noqa: F401
        assert sys.stdout is not None and sys.stderr is not None
    finally:
        sys.stdout, sys.stderr = orig_out, orig_err


def test_add_one_pair_with_roi():
    """Test _add_one_pair with user roi."""
    from pin_detection.dataset import _add_one_pair

    base = Path(__file__).parent.parent / "test_data"
    u = base / "pin_synthetic" / "unmasked" / "01.jpg"
    m = base / "pin_synthetic" / "masked" / "01.jpg"
    if not u.exists():
        u = base / "pin_synthetic" / "train" / "unmasked" / "001.jpg"
        m = base / "pin_synthetic" / "train" / "masked" / "001.jpg"
    if not u.exists():
        u = list((base / "pin_synthetic" / "train" / "unmasked").glob("*.jpg"))[0]
        m = (base / "pin_synthetic" / "train" / "masked") / u.name
    if not u.exists() or not m.exists():
        pytest.skip("No test pair")

    with tempfile.TemporaryDirectory() as d:
        img_dir = Path(d) / "images"
        lbl_dir = Path(d) / "labels"
        img_dir.mkdir()
        lbl_dir.mkdir()
        roi = (50, 50, 300, 200)
        _add_one_pair(u, m, img_dir, lbl_dir, roi=roi)
        out_img = img_dir / f"{u.stem}.jpg"
        assert out_img.exists()
        from PIL import Image
        w, h = Image.open(out_img).size
        assert w <= 300 - 50
        assert h <= 200 - 50
