"""
Generate synthetic connector pin test data for pin_detection.
Mimics 20p FPC/FFC adapter: grayscale, rectangular pads, 20 upper + 20 lower facing each other.
Ref: https://hubtronics.in/image/cache/catalog/sagar/20p-fpc-ffc-adapter-board-550x550.jpg
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def generate_connector_image(
    width: int = 640,
    height: int = 480,
    n_upper: int = 20,
    n_lower: int = 20,
    pad_w: int = 12,
    pad_h: int = 4,
    pitch_px: int = 18,
    blur_prob: float = 0.25,
    n_fake_pins: int = 6,
    seed: int = 42,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], list[tuple[int, int]]]:
    """
    Generate synthetic FPC/FFC connector top-view (grayscale).
    - Grayscale: black bg (0), white pads (200-255)
    - Pins: rectangular pads (wider than tall), horizontal alignment
    - 20 upper + 20 lower, facing each other
    - Optional blur, fake dots (dust/scratches)
    Returns (image_array RGB for compatibility, bboxes, fake_centers).
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Upper row: horizontal pads
    y_upper = 90
    x_start = 60
    x_positions = [x_start + i * pitch_px for i in range(n_upper)]

    bboxes = []
    for xc in x_positions:
        x1 = max(0, xc - pad_w // 2)
        y1 = y_upper
        x2 = min(width, x1 + pad_w)
        y2 = min(height, y1 + pad_h)
        val = rng.integers(210, 256)
        img[y1:y2, x1:x2] = [val, val, val]
        bboxes.append((x1, y1, x2, y2))

    # Lower row: facing upper (same horizontal layout)
    y_lower = height // 2 + 60
    for xc in x_positions:
        x1 = max(0, xc - pad_w // 2)
        y1 = y_lower
        x2 = min(width, x1 + pad_w)
        y2 = min(height, y1 + pad_h)
        val = rng.integers(210, 256)
        img[y1:y2, x1:x2] = [val, val, val]
        bboxes.append((x1, y1, x2, y2))

    # Optional blur (factory camera)
    if blur_prob > 0 and rng.random() < blur_prob:
        img = np.array(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=0.6)))

    # Fake pins: dust, scratches, reflections (NOT real pads)
    fake_centers = []
    for _ in range(n_fake_pins):
        cx = rng.integers(40, width - 40)
        cy = rng.integers(40, height - 40)
        if any(abs(cx - (b[0] + b[2]) // 2) < 50 and abs(cy - (b[1] + b[3]) // 2) < 50 for b in bboxes):
            continue
        r = rng.integers(1, 4)
        val = rng.integers(180, 256)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        img[ny, nx] = [val, val, val]
        fake_centers.append((cx, cy))

    # Grayscale sensor noise
    noise = rng.integers(-4, 5, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img, bboxes, fake_centers


def bbox_to_green_region(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, radius: int = 6) -> None:
    """Draw green dot at pad center (masked = GT)."""
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    h, w = img.shape[:2]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w:
                    img[ny, nx] = [0, 255, 0]


def _generate_to_dir(
    out_dir: Path,
    n_pairs: int,
    seed_offset: int,
    blur_prob: float,
    n_fake_pins: int,
) -> None:
    unmasked_dir = out_dir / "unmasked"
    masked_dir = out_dir / "masked"
    unmasked_dir.mkdir(parents=True, exist_ok=True)
    masked_dir.mkdir(parents=True, exist_ok=True)
    meta = {}
    for i in range(n_pairs):
        img, bboxes, fake_centers = generate_connector_image(
            seed=seed_offset + i,
            blur_prob=blur_prob,
            n_fake_pins=n_fake_pins,
        )
        stem = f"{i+1:03d}"
        unmasked_path = unmasked_dir / f"{stem}.jpg"
        Image.fromarray(img).save(unmasked_path)
        masked_img = img.copy()
        for x1, y1, x2, y2 in bboxes:
            bbox_to_green_region(masked_img, x1, y1, x2, y2)
        masked_path = masked_dir / f"{stem}.jpg"
        Image.fromarray(masked_img).save(masked_path)
        meta[stem] = {
            "n_real": len(bboxes),
            "n_fake": len(fake_centers),
            "fake_centers": [[int(x), int(y)] for x, y in fake_centers],
        }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate FPC/FFC-style grayscale pin test data")
    parser.add_argument("--output-dir", default="test_data/pin_synthetic", help="Output directory")
    parser.add_argument("--n-pairs", type=int, default=0, help="Number of pairs (legacy, use --train/--test)")
    parser.add_argument("--train-pairs", type=int, default=40, help="Train set pairs")
    parser.add_argument("--test-pairs", type=int, default=20, help="Test set pairs (never used for training)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--blur-prob", type=float, default=0.25)
    parser.add_argument("--n-fake-pins", type=int, default=6)
    args = parser.parse_args()

    root = _project_root()
    out = root / args.output_dir

    if args.n_pairs > 0:
        # Legacy: single set
        _generate_to_dir(out, args.n_pairs, 42, args.blur_prob, args.n_fake_pins)
        print(f"Generated {args.n_pairs} pairs in {out}")
    else:
        train_dir = out / "train"
        test_dir = out / "test"
        _generate_to_dir(train_dir, args.train_pairs, 0, args.blur_prob, args.n_fake_pins)
        _generate_to_dir(test_dir, args.test_pairs, 1000, args.blur_prob, args.n_fake_pins)
        print(f"Generated train: {args.train_pairs} pairs in {train_dir}")
        print(f"Generated test:  {args.test_pairs} pairs in {test_dir}")
    print(f"  unmasked: grayscale, 20 upper + 20 lower rectangular pads, {args.n_fake_pins} fake dots")
    print(f"  masked:   green only on real pads (40 total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
