"""
Build YOLO dataset from masked/unmasked image pairs.
Supports 1 pair or 10 pairs (unmasked-dir + masked-dir).
"""
import shutil
from pathlib import Path
from typing import List, Tuple

from .annotation import masked_image_to_annotations

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _add_one_pair(
    unmasked_path: Path,
    masked_path: Path,
    images_dir: Path,
    labels_dir: Path,
    class_id: int = 0,
) -> None:
    """Add one image pair to dataset."""
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
    """Find masked file for unmasked (same name or stem_masked.suffix)."""
    m = masked_dir / unmasked_path.name
    if m.exists():
        return m
    m2 = masked_dir / f"{unmasked_path.stem}_masked{unmasked_path.suffix}"
    if m2.exists():
        return m2
    raise FileNotFoundError(f"Masked pair not found for {unmasked_path.name}")


def prepare_yolo_dataset_from_dirs(
    unmasked_dir: Path,
    masked_dir: Path,
    output_dir: Path,
    class_id: int = 0,
) -> Path:
    """
    Create YOLO dataset from directories (10 pairs).
    Pairs by matching filename. unmasked/01.jpg <-> masked/01.jpg
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images" / "train"
    labels_dir = output_dir / "labels" / "train"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    unmasked_dir = Path(unmasked_dir)
    masked_dir = Path(masked_dir)
    u_files = sorted([f for f in unmasked_dir.iterdir() if f.suffix.lower() in IMG_EXTS], key=lambda p: p.name)

    for u_path in u_files:
        m_path = _find_masked_pair(u_path, masked_dir)
        _add_one_pair(u_path, m_path, images_dir, labels_dir, class_id)

    data_yaml = output_dir / "data.yaml"
    with open(data_yaml, "w") as f:
        f.write(f"path: {output_dir.absolute()}\n")
        f.write("train: images/train\n")
        f.write("val: images/train\n")
        f.write("names:\n")
        f.write("  0: pin\n")

    return data_yaml
