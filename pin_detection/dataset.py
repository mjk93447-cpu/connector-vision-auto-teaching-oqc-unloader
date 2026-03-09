"""
Build YOLO dataset from masked/unmasked image pairs.
Supports 1 pair, dir (unmasked-dir + masked-dir), train/val split.
ROI crop for large images (5496×3672): extract pin region, crop, then train.
Cell-ID pairing: filenames with YYYYMMDD_HHMMSS_A2HD{cell_no} → pair by A2HD{cell_no}.
"""
import re
import random
import shutil
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
from PIL import Image

from .annotation import masked_array_to_annotations
from .roi import extract_pin_roi, crop_to_roi

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
ROI_SIZE_THRESHOLD = 2000  # max(w,h) > this → use ROI crop

# Cell ID: A2HD + alphanumeric (e.g. A2HD001, A2HD12345)
CELL_ID_PATTERN = re.compile(r"(A2HD[A-Za-z0-9]+)", re.IGNORECASE)


def extract_cell_id(path: Path) -> str | None:
    """
    Extract cell ID (A2HDxxxx) from filename.
    Example: 20250101_120000_A2HD001.jpg → A2HD001
    """
    m = CELL_ID_PATTERN.search(path.stem)
    return m.group(1).upper() if m else None


def validate_pair_dimensions(unmasked_path: Path, masked_path: Path) -> None:
    """
    Ensure unmasked and masked images have the same dimensions.
    Annotations are extracted from masked image; they must align with unmasked.
    """
    with Image.open(unmasked_path) as u, Image.open(masked_path) as m:
        uw, uh = u.size
        mw, mh = m.size
    if (uw, uh) != (mw, mh):
        raise ValueError(
            f"Image size mismatch: unmasked {unmasked_path.name} ({uw}x{uh}) "
            f"!= masked {masked_path.name} ({mw}x{mh}). "
            "Both must have identical dimensions for correct annotation alignment."
        )


def _add_one_pair(
    unmasked_path: Path,
    masked_path: Path,
    images_dir: Path,
    labels_dir: Path,
    class_id: int = 0,
    use_roi: bool = False,
    roi_margin: float = 0.15,
    roi: tuple[int, int, int, int] | None = None,
) -> None:
    """Add one image pair to dataset. Optionally crop to ROI (user roi_map or auto)."""
    validate_pair_dimensions(unmasked_path, masked_path)
    u_img = np.array(Image.open(unmasked_path).convert("RGB"))
    m_img, yolo_anns = masked_array_to_annotations(
        np.array(Image.open(masked_path).convert("RGB"))
    )
    if not yolo_anns:
        raise ValueError(f"No green pin regions found in {masked_path}")

    h, w = u_img.shape[:2]
    if roi is not None:
        # User-specified ROI (roi_map from ROI Editor, ROADMAP 10.20)
        x1, y1, x2, y2 = roi
        x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
        roi_clamped = (x1, y1, x2, y2)
        u_img = crop_to_roi(u_img, roi_clamped)
        m_img = crop_to_roi(m_img, roi_clamped)
        _, yolo_anns = masked_array_to_annotations(m_img)
    elif use_roi and max(w, h) > ROI_SIZE_THRESHOLD:
        roi_auto = extract_pin_roi(masked_path, margin_ratio=roi_margin)
        u_img = crop_to_roi(u_img, roi_auto)
        m_img = crop_to_roi(m_img, roi_auto)
        _, yolo_anns = masked_array_to_annotations(m_img)

    stem = unmasked_path.stem
    dst_img = images_dir / f"{stem}.jpg"
    Image.fromarray(u_img).save(dst_img, quality=95)
    stem_out = dst_img.stem

    label_path = labels_dir / f"{stem_out}.txt"
    with open(label_path, "w") as f:
        for xc, yc, bw, bh in yolo_anns:
            f.write(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


def prepare_yolo_dataset(
    unmasked_path: Path,
    masked_path: Path,
    output_dir: Path,
    class_id: int = 0,
    use_roi: bool = False,
    roi_margin: float = 0.15,
) -> Path:
    """
    Create YOLO dataset from one image pair.
    Returns path to dataset yaml for training.
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images" / "train"
    labels_dir = output_dir / "labels" / "train"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    _add_one_pair(unmasked_path, masked_path, images_dir, labels_dir, class_id, use_roi=use_roi, roi_margin=roi_margin)

    data_yaml = output_dir / "data.yaml"
    with open(data_yaml, "w") as f:
        f.write(f"path: {output_dir.absolute()}\n")
        f.write("train: images/train\n")
        f.write("val: images/train\n")
        f.write("names:\n")
        f.write("  0: pin\n")

    return data_yaml


def analyze_dataset_for_training(
    unmasked_dir: Path,
    masked_dir: Path,
    max_samples: int = 5,
) -> dict:
    """
    Quick scan of dataset to suggest optimal training parameters.
    Samples up to max_samples images, extracts image size and pin bbox sizes.
    Returns: {imgsz, epochs, val_split, note, pin_avg_w, pin_avg_h, n_images, img_w, img_h}.
    """
    unmasked_dir = Path(unmasked_dir)
    masked_dir = Path(masked_dir)
    u_files = sorted([f for f in unmasked_dir.iterdir() if f.suffix.lower() in IMG_EXTS], key=lambda p: p.name)
    if not u_files:
        return {"imgsz": 640, "epochs": 100, "val_split": 0.2, "note": "No images"}

    n_images = len(u_files)
    samples = u_files[: min(max_samples, len(u_files))]
    img_sizes = []
    pin_widths = []
    pin_heights = []

    for u_path in samples:
        try:
            with Image.open(u_path) as im:
                img_sizes.append(im.size)
        except Exception:
            continue
        try:
            m_path = _find_masked_pair(u_path, masked_dir)
            m_img = np.array(Image.open(m_path).convert("RGB"))
            _, yolo_anns = masked_array_to_annotations(m_img)
            w, h = m_img.shape[1], m_img.shape[0]
            for xc, yc, bw, bh in yolo_anns:
                pin_widths.append(bw * w)
                pin_heights.append(bh * h)
        except Exception:
            pass

    img_w = img_sizes[0][0] if img_sizes else 0
    img_h = img_sizes[0][1] if img_sizes else 0
    max_dim = max(img_w, img_h) if img_sizes else 0

    # imgsz: suggestion only, no hard cap (ROADMAP 10.20). User can override 320–4096.
    imgsz = 640
    note_parts = []

    if max_dim > ROI_SIZE_THRESHOLD:
        imgsz = 640
        note_parts.append("full image (ROI off)")
    elif max_dim > 0:
        if pin_widths and pin_heights:
            avg_w = sum(pin_widths) / len(pin_widths)
            avg_h = sum(pin_heights) / len(pin_heights)
            if avg_w * avg_h < 120 or avg_w < 12 or avg_h < 8:
                imgsz = min(4096, max(640, max_dim))
                note_parts.append("small pins")
            elif max_dim < 640:
                imgsz = min(4096, max(640, max_dim))
                note_parts.append("match resolution")
        else:
            imgsz = min(4096, max(640, max_dim)) if max_dim > 400 else 640

    epochs = 100  # Higher epochs for better P/R (EXE_TEST_FEEDBACK 10.20.2)

    val_split = 0.2
    if n_images < 5:
        val_split = 0.0
        note_parts.append("no val split")

    note = ", ".join(note_parts) if note_parts else "default"

    mosaic = 0.0 if "full image" in note or n_images >= 20 else 0.5
    return {
        "imgsz": imgsz,
        "epochs": epochs,
        "val_split": val_split,
        "mosaic": mosaic,
        "note": note,
        "pin_avg_w": sum(pin_widths) / len(pin_widths) if pin_widths else 0,
        "pin_avg_h": sum(pin_heights) / len(pin_heights) if pin_heights else 0,
        "n_images": n_images,
        "img_w": img_w,
        "img_h": img_h,
    }


def get_dataset_info(data_yaml: Path) -> dict:
    """Return n_images, img_w, img_h from dataset."""
    from PIL import Image
    data_yaml = Path(data_yaml)
    base = data_yaml.parent
    img_dir = base / "images" / "train"
    if not img_dir.exists():
        return {"n_images": 0, "img_w": 0, "img_h": 0}
    files = [f for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXTS]
    n = len(files)
    w, h = 0, 0
    if files:
        try:
            im = Image.open(files[0])
            w, h = im.size
            im.close()
        except Exception:
            pass
    return {"n_images": n, "img_w": w, "img_h": h}


def _find_masked_pair(unmasked_path: Path, masked_dir: Path) -> Path:
    """
    Find masked file for unmasked image.
    Cell-ID pairing: 20250101_120000_A2HD001.jpg <-> any masked file containing A2HD001.
    Fallback: stem-based (exact name, stem_masked, stem + IMG_EXTS).
    """
    stem = unmasked_path.stem
    cell_id = extract_cell_id(unmasked_path)

    # 1) Cell-ID pairing: find masked file with same cell ID
    if cell_id:
        masked_files = [f for f in masked_dir.iterdir() if f.suffix.lower() in IMG_EXTS]
        for mf in masked_files:
            if cell_id in mf.stem.upper():
                # Prefer same stem, then stem_masked
                if mf.stem == stem:
                    return mf
                if mf.stem == f"{stem}_masked":
                    return mf
        for mf in masked_files:
            if cell_id in mf.stem.upper():
                return mf  # any match

    # 2) Exact name
    m = masked_dir / unmasked_path.name
    if m.exists():
        return m
    # 3) stem_masked.suffix (e.g. 2_masked.bmp)
    m2 = masked_dir / f"{stem}_masked{unmasked_path.suffix}"
    if m2.exists():
        return m2
    # 4) stem + other extensions (bmp/jpg cross-pairing)
    for ext in IMG_EXTS:
        cand = masked_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
        cand2 = masked_dir / f"{stem}_masked{ext}"
        if cand2.exists():
            return cand2
    raise FileNotFoundError(f"Masked pair not found for {unmasked_path.name}")


def prepare_yolo_dataset_from_dirs(
    unmasked_dir: Path,
    masked_dir: Path,
    output_dir: Path,
    class_id: int = 0,
    val_split: float = 0.2,
    seed: int = 42,
    use_roi: bool = True,
    roi_margin: float = 0.15,
    roi_map: dict[str, list[int]] | None = None,
    on_progress: Callable[[int, int, Path], None] | None = None,
) -> Path:
    """
    Create YOLO dataset from directories.
    Pairs by matching filename. unmasked/01.jpg <-> masked/01.jpg
    val_split: fraction for validation (0.2 = 80% train, 20% val). 0 = no split (train=val).
    use_roi: when max(w,h)>2000, crop to pin region (ignored if roi_map has entry). Default True.
    roi_margin: margin around pin bbox (0.15 = 15%).
    roi_map: user ROI per stem {stem: [x1,y1,x2,y2]}. If None, loads output_dir/roi_map.json if exists.
    on_progress: optional callback(current, total, path) during dataset build.
    """
    output_dir = Path(output_dir)
    train_img = output_dir / "images" / "train"
    train_lbl = output_dir / "labels" / "train"
    val_img = output_dir / "images" / "val"
    val_lbl = output_dir / "labels" / "val"
    train_img.mkdir(parents=True, exist_ok=True)
    train_lbl.mkdir(parents=True, exist_ok=True)
    val_img.mkdir(parents=True, exist_ok=True)
    val_lbl.mkdir(parents=True, exist_ok=True)

    unmasked_dir = Path(unmasked_dir)
    masked_dir = Path(masked_dir)
    u_files = sorted([f for f in unmasked_dir.iterdir() if f.suffix.lower() in IMG_EXTS], key=lambda p: p.name)

    # Load roi_map from file if not provided (ROADMAP 10.20)
    # roi_map.json lives in output folder: output_dir.parent (dataset_dir's parent) or output_dir
    if roi_map is None:
        roi_map_path = output_dir.parent / "roi_map.json"
        if not roi_map_path.exists():
            roi_map_path = output_dir / "roi_map.json"
        if roi_map_path.exists():
            try:
                import json
                with open(roi_map_path) as f:
                    roi_map = {k: list(v) for k, v in json.load(f).items() if isinstance(v, (list, tuple)) and len(v) == 4}
            except Exception:
                roi_map = {}
        else:
            roi_map = {}

    if not u_files:
        raise ValueError(
            f"No unmasked images found in {unmasked_dir}. "
            f"Expected images with extensions: {', '.join(IMG_EXTS)}"
        )

    if val_split > 0 and len(u_files) >= 2:
        rng = random.Random(seed)
        shuffled = u_files.copy()
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_split))
        val_files = set(shuffled[:n_val])
        train_files = shuffled[n_val:]
    else:
        val_files = set()
        train_files = u_files

    def _get_roi(u_path: Path) -> tuple[int, int, int, int] | None:
        stem = u_path.stem
        if stem in roi_map and len(roi_map[stem]) == 4:
            return tuple(roi_map[stem])
        return None

    all_files = list(train_files) + list(val_files)
    total = len(all_files)
    done = [0]

    def _add_with_progress(u_path, img_dir, lbl_dir):
        if on_progress:
            on_progress(done[0] + 1, total, u_path)
        m_path = _find_masked_pair(u_path, masked_dir)
        roi = _get_roi(u_path)
        _add_one_pair(u_path, m_path, img_dir, lbl_dir, class_id, use_roi=use_roi and roi is None, roi_margin=roi_margin, roi=roi)
        done[0] += 1

    for u_path in train_files:
        _add_with_progress(u_path, train_img, train_lbl)
    for u_path in val_files:
        _add_with_progress(u_path, val_img, val_lbl)

    data_yaml = output_dir / "data.yaml"
    val_path = "images/val" if val_files else "images/train"
    with open(data_yaml, "w") as f:
        f.write(f"path: {output_dir.absolute()}\n")
        f.write("train: images/train\n")
        f.write(f"val: {val_path}\n")
        f.write("names:\n")
        f.write("  0: pin\n")

    return data_yaml


def prepare_yolo_test_dataset(
    unmasked_dir: Path,
    masked_dir: Path,
    output_dir: Path,
    class_id: int = 0,
    use_roi: bool = False,
    roi_margin: float = 0.15,
) -> Path:
    """
    Create YOLO-format test dataset (for model.val() mAP50).
    All images go to images/test, labels/test.
    """
    output_dir = Path(output_dir)
    img_dir = output_dir / "images" / "test"
    lbl_dir = output_dir / "labels" / "test"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    unmasked_dir = Path(unmasked_dir)
    masked_dir = Path(masked_dir)
    u_files = sorted([f for f in unmasked_dir.iterdir() if f.suffix.lower() in IMG_EXTS], key=lambda p: p.name)
    if not u_files:
        raise ValueError(f"No unmasked images in {unmasked_dir}")

    for u_path in u_files:
        m_path = _find_masked_pair(u_path, masked_dir)
        _add_one_pair(u_path, m_path, img_dir, lbl_dir, class_id, use_roi=use_roi, roi_margin=roi_margin)

    data_yaml = output_dir / "data.yaml"
    with open(data_yaml, "w") as f:
        f.write(f"path: {output_dir.absolute()}\n")
        f.write("train: images/test\n")
        f.write("val: images/test\n")
        f.write("names:\n")
        f.write("  0: pin\n")
    return data_yaml
