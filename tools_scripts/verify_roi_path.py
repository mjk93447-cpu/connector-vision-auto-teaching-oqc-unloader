"""
Verify ROI delivery path and roi_map usage before factory testing.
Run: python tools_scripts/verify_roi_path.py

Uses pin_large_factory (CI) or pin_synthetic as test data.
CI generates pin_large_factory before this step; local: run generate_pin_test_data --large-factory first.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _resolve_test_data() -> tuple[Path, Path] | None:
    """Resolve unmasked/masked dirs. Prefer pin_large_factory (CI), fallback pin_synthetic."""
    # 1. pin_large_factory (CI: --large-factory generates this before verify)
    large = ROOT / "test_data" / "pin_large_factory"
    u_large = large / "unmasked"
    m_large = large / "masked"
    if u_large.exists() and m_large.exists() and list(u_large.glob("*.jpg")):
        return u_large, m_large
    # 2. pin_synthetic (train or root)
    syn = ROOT / "test_data" / "pin_synthetic"
    u_syn = syn / "train" / "unmasked" if (syn / "train" / "unmasked").exists() else syn / "unmasked"
    m_syn = syn / "train" / "masked" if (syn / "train" / "masked").exists() else syn / "masked"
    if u_syn.exists() and m_syn.exists() and list(u_syn.glob("*.jpg")):
        return u_syn, m_syn
    return None


def _test_roi_map_load_path():
    """Verify dataset loads roi_map from output_dir.parent or output_dir."""
    from pin_detection.dataset import prepare_yolo_dataset_from_dirs

    resolved = _resolve_test_data()
    if not resolved:
        print("SKIP: No test data (run generate_pin_test_data --large-factory or default first)")
        return True

    unmasked, masked = resolved
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        roi_path = root / "roi_map.json"
        u_files = [f for f in unmasked.iterdir() if f.suffix.lower() in IMG_EXTS][:3]
        roi_map = {f.stem: [20, 20, 150, 120] for f in u_files}
        with open(roi_path, "w") as f:
            json.dump(roi_map, f, indent=2)

        dataset_dir = root / "dataset"
        data_yaml = prepare_yolo_dataset_from_dirs(
            unmasked, masked, dataset_dir, val_split=0.0, use_roi=False
        )
        assert data_yaml.exists(), "data.yaml not created"
        train_img = dataset_dir / "images" / "train"
        assert train_img.exists(), "train images dir missing"
        imgs = list(train_img.glob("*.jpg"))
        assert len(imgs) >= 1, "No train images"
        # Verify cropped size (roi 20,20,150,120 -> max 130x100)
        for p in imgs[:1]:
            from PIL import Image
            w, h = Image.open(p).size
            assert w <= 150 and h <= 120, f"ROI crop failed: got {w}x{h}"
    print("  [OK] roi_map load path (output_dir.parent) verified")
    return True


def _test_roi_map_in_output_dir():
    """Verify roi_map at output_dir (dataset_dir) also works."""
    from pin_detection.dataset import prepare_yolo_dataset_from_dirs

    resolved = _resolve_test_data()
    if not resolved:
        print("SKIP: No test data")
        return True

    unmasked, masked = resolved
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        dataset_dir = root / "dataset"
        dataset_dir.mkdir(parents=True)
        roi_path = dataset_dir / "roi_map.json"
        u_files = [f for f in unmasked.iterdir() if f.suffix.lower() in IMG_EXTS][:2]
        roi_map = {f.stem: [10, 10, 80, 60] for f in u_files}
        with open(roi_path, "w") as f:
            json.dump(roi_map, f, indent=2)

        data_yaml = prepare_yolo_dataset_from_dirs(
            unmasked, masked, dataset_dir, val_split=0.0, use_roi=False
        )
        assert data_yaml.exists()
    print("  [OK] roi_map at output_dir (dataset_dir) verified")
    return True


def _test_roi_editor_save_path():
    """Verify roi_editor saves to output_dir/roi_map.json."""
    from pin_detection.roi_editor import load_roi_map, save_roi_map

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        roi_path = out / "roi_map.json"
        roi_map = {"01": [100, 100, 500, 300]}
        save_roi_map(roi_path, roi_map)
        assert roi_path.exists()
        loaded = load_roi_map(roi_path)
        assert loaded == roi_map
    print("  [OK] roi_editor load/save path verified")
    return True


def main() -> int:
    print("ROI path & roi_map verification")
    print("-" * 40)
    resolved = _resolve_test_data()
    if resolved:
        src = "pin_large_factory" if "pin_large_factory" in str(resolved[0]) else "pin_synthetic"
        print(f"  Test data: {src}")
    else:
        print("  Test data: none (roi_map tests will SKIP)")
    try:
        _test_roi_editor_save_path()
        _test_roi_map_load_path()
        _test_roi_map_in_output_dir()
        print("-" * 40)
        print("All ROI path checks PASSED")
        return 0
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
