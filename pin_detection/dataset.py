"""
Build YOLO dataset from masked/unmasked image pairs.
Supports 1 pair, dir (unmasked-dir + masked-dir), train/val split.
"""
import random
import shutil
from pathlib import Path
from typing import List, Tuple

from PIL import Image

from .annotation import masked_image_to_annotations

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


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
) -> None:
    """Add one image pair to dataset."""
    validate_pair_dimensions(unmasked_path, masked_path)
    _, yolo_anns = masked_image_to_annotations(masked_path)
    if not yolo_anns:
        raise ValueError(f"No green pin regions found in {masked_path}")

    stem = unmasked_path.stem
    dst_img = images_dir / f"{stem}{unmasked_path.suffix}"
    if dst_img.suffix.lower() not in IMG_EXTS:
        dst_img = images_dir / f"{stem}.jpg"
    shutil.copy2(unmasked_path, dst_img)
    stem_out = dst_img.stem

    label_path = labels_dir / f"{stem_out}.txt"
    with open(label_path, "w") as f:
        for xc, yc, w, h in yolo_anns:
            f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def prepare_yolo_dataset(
    unmasked_path: Path,
    masked_path: Path,
    output_dir: Path,
    class_id: int = 0,
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

    _add_one_pair(unmasked_path, masked_path, images_dir, labels_dir, class_id)

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
    Supports cross-format pairing: unmasked 2.bmp <-> masked 2.jpg
    Priority: 1) exact name 2) stem_masked.suffix 3) stem + any IMG_EXTS
    """
    stem = unmasked_path.stem
    # 1) Exact match
    m = masked_dir / unmasked_path.name
    if m.exists():
        return m
    # 2) stem_masked.suffix (e.g. 2_masked.bmp)
    m2 = masked_dir / f"{stem}_masked{unmasked_path.suffix}"
    if m2.exists():
        return m2
    # 3) stem + other extensions (bmp/jpg cross-pairing)
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
) -> Path:
    """
    Create YOLO dataset from directories.
    Pairs by matching filename. unmasked/01.jpg <-> masked/01.jpg
    val_split: fraction for validation (0.2 = 80% train, 20% val). 0 = no split (train=val).
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
        _add_one_pair(u_path, m_path, train_img, train_lbl, class_id)
    for u_path in val_files:
        m_path = _find_masked_pair(u_path, masked_dir)
        _add_one_pair(u_path, m_path, val_img, val_lbl, class_id)

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
        _add_one_pair(u_path, m_path, img_dir, lbl_dir, class_id)

    data_yaml = output_dir / "data.yaml"
    with open(data_yaml, "w") as f:
        f.write(f"path: {output_dir.absolute()}\n")
        f.write("train: images/test\n")
        f.write("val: images/test\n")
        f.write("names:\n")
        f.write("  0: pin\n")
    return data_yaml
