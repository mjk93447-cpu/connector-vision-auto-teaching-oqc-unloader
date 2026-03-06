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
from typing import List, Tuple

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
) -> None:
    """Add one image pair to dataset. Optionally crop to ROI for large images."""
    validate_pair_dimensions(unmasked_path, masked_path)
    u_img = np.array(Image.open(unmasked_path).convert("RGB"))
    m_img, yolo_anns = masked_array_to_annotations(
        np.array(Image.open(masked_path).convert("RGB"))
    )
    if not yolo_anns:
        raise ValueError(f"No green pin regions found in {masked_path}")

    h, w = u_img.shape[:2]
    if use_roi and max(w, h) > ROI_SIZE_THRESHOLD:
        roi = extract_pin_roi(masked_path, margin_ratio=roi_margin)
        u_img = crop_to_roi(u_img, roi)
        m_img = crop_to_roi(m_img, roi)
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
    use_roi: bool = True,
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
) -> Path:
    """
    Create YOLO dataset from directories.
    Pairs by matching filename. unmasked/01.jpg <-> masked/01.jpg
    val_split: fraction for validation (0.2 = 80% train, 20% val). 0 = no split (train=val).
    use_roi: when max(w,h)>2000, crop to pin region before training (saves compute).
    roi_margin: margin around pin bbox (0.15 = 15%).
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

    for u_path in train_files:
        m_path = _find_masked_pair(u_path, masked_dir)
        _add_one_pair(u_path, m_path, train_img, train_lbl, class_id, use_roi=use_roi, roi_margin=roi_margin)
    for u_path in val_files:
        m_path = _find_masked_pair(u_path, masked_dir)
        _add_one_pair(u_path, m_path, val_img, val_lbl, class_id, use_roi=use_roi, roi_margin=roi_margin)

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
    use_roi: bool = True,
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
