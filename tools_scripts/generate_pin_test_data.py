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


def _scale_for_resolution(width: int, height: int) -> tuple[int, int, int, int, int, int]:
    """Scale pad/pitch/positions for target resolution (base 640x480)."""
    base_w, base_h = 640, 480
    sw, sh = width / base_w, height / base_h
    pad_w = max(8, int(12 * sw))
    pad_h = max(3, int(4 * sh))
    pitch_px = max(14, int(18 * sw))
    y_upper = int(90 * sh)
    y_lower = height // 2 + int(60 * sh)
    x_start = int(60 * sw)
    return pad_w, pad_h, pitch_px, y_upper, y_lower, x_start


def generate_connector_image(
    width: int = 640,
    height: int = 480,
    n_upper: int = 20,
    n_lower: int = 20,
    pad_w: int | None = None,
    pad_h: int | None = None,
    pitch_px: int | None = None,
    blur_prob: float = 0.25,
    n_fake_pins: int = 6,
    seed: int = 42,
    complex_bg: bool = False,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], list[tuple[int, int]]]:
    """
    Generate synthetic FPC/FFC connector top-view (grayscale).
    - Grayscale: black bg (0), white pads (200-255)
    - Pins: rectangular pads (wider than tall), horizontal alignment
    - 20 upper + 20 lower, facing each other
    - Optional blur, fake dots (dust/scratches)
    Returns (image_array RGB for compatibility, bboxes, fake_centers).
    complex_bg: add PCB texture, dust, scratches for factory-like realism.
    """
    rng = np.random.default_rng(seed)
    spad, spadh, spitch, y_upper, y_lower, x_start = _scale_for_resolution(width, height)
    _pad_w = pad_w if pad_w is not None else spad
    _pad_h = pad_h if pad_h is not None else spadh
    _pitch = pitch_px if pitch_px is not None else spitch

    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Complex background: PCB-like base, subtle texture
    if complex_bg:
        base = rng.integers(15, 35, (height, width, 3), dtype=np.uint8)
        img[:] = base
        # Fine grain noise
        noise = rng.integers(-8, 9, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        # PCB trace-like lines (horizontal)
        for _ in range(rng.integers(20, 50)):
            y = rng.integers(0, height)
            val = rng.integers(25, 55)
            thickness = rng.integers(1, 4)
            for t in range(thickness):
                yy = min(height - 1, y + t)
                img[yy, :] = np.clip(img[yy, :].astype(np.int16) + val, 0, 255).astype(np.uint8)

    # Upper row: horizontal pads
    x_positions = [x_start + i * _pitch for i in range(n_upper)]

    bboxes = []
    for xc in x_positions:
        x1 = max(0, xc - _pad_w // 2)
        y1 = y_upper
        x2 = min(width, x1 + _pad_w)
        y2 = min(height, y1 + _pad_h)
        val = rng.integers(210, 256)
        img[y1:y2, x1:x2] = [val, val, val]
        bboxes.append((x1, y1, x2, y2))

    # Lower row: facing upper (same horizontal layout)
    for xc in x_positions:
        x1 = max(0, xc - _pad_w // 2)
        y1 = y_lower
        x2 = min(width, x1 + _pad_w)
        y2 = min(height, y1 + _pad_h)
        val = rng.integers(210, 256)
        img[y1:y2, x1:x2] = [val, val, val]
        bboxes.append((x1, y1, x2, y2))

    # Optional blur (factory camera)
    if blur_prob > 0 and rng.random() < blur_prob:
        img = np.array(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=0.6)))

    # Fake pins: dust, scratches, reflections (NOT real pads)
    avoid_dist = max(50, int(0.08 * min(width, height)))
    n_fake = n_fake_pins * max(1, (width * height) // (640 * 480)) if complex_bg else n_fake_pins
    fake_centers = []
    for _ in range(min(n_fake, 80)):  # cap 80 for large images
        cx = rng.integers(avoid_dist, width - avoid_dist)
        cy = rng.integers(avoid_dist, height - avoid_dist)
        if any(abs(cx - (b[0] + b[2]) // 2) < avoid_dist and abs(cy - (b[1] + b[3]) // 2) < avoid_dist for b in bboxes):
            continue
        r = rng.integers(1, max(2, min(8, _pad_w // 4)))
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


def bbox_to_green_region(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, radius: int | None = None) -> None:
    """Draw green dot at pad center (masked = GT). radius defaults to half of min(w,h) of bbox, min 6."""
    if radius is None:
        bw, bh = x2 - x1, y2 - y1
        radius = max(6, min(bw, bh) // 2)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    h, w = img.shape[:2]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w:
                    img[ny, nx] = [0, 255, 0]


def bbox_to_cross_marker(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, arm_len: int | None = None, thickness: int = 2) -> None:
    """Draw green cross (+) at pad center (masked = GT). For factory-like thin markers (annotation _dilate_for_thin_markers)."""
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    h, w = img.shape[:2]
    if arm_len is None:
        bw, bh = x2 - x1, y2 - y1
        arm_len = max(4, min(bw, bh) // 2)
    green = [0, 255, 0]
    half_t = thickness // 2
    for dy in range(-arm_len, arm_len + 1):
        for t in range(thickness):
            ny = cy + dy
            nx = cx - half_t + t
            if 0 <= ny < h and 0 <= nx < w:
                img[ny, nx] = green
    for dx in range(-arm_len, arm_len + 1):
        for t in range(thickness):
            ny = cy - half_t + t
            nx = cx + dx
            if 0 <= ny < h and 0 <= nx < w:
                img[ny, nx] = green


def _generate_to_dir(
    out_dir: Path,
    n_pairs: int,
    seed_offset: int,
    blur_prob: float,
    n_fake_pins: int,
    width: int = 640,
    height: int = 480,
    complex_bg: bool = False,
    cross_markers: bool = False,
) -> None:
    unmasked_dir = out_dir / "unmasked"
    masked_dir = out_dir / "masked"
    unmasked_dir.mkdir(parents=True, exist_ok=True)
    masked_dir.mkdir(parents=True, exist_ok=True)
    meta = {}
    for i in range(n_pairs):
        img, bboxes, fake_centers = generate_connector_image(
            width=width,
            height=height,
            seed=seed_offset + i,
            blur_prob=blur_prob,
            n_fake_pins=n_fake_pins,
            complex_bg=complex_bg,
        )
        stem = f"{i+1:03d}"
        unmasked_path = unmasked_dir / f"{stem}.jpg"
        Image.fromarray(img).save(unmasked_path)
        masked_img = img.copy()
        draw_fn = bbox_to_cross_marker if cross_markers else bbox_to_green_region
        for x1, y1, x2, y2 in bboxes:
            draw_fn(masked_img, x1, y1, x2, y2)
        masked_path = masked_dir / f"{stem}.jpg"
        Image.fromarray(masked_img).save(masked_path)
        meta[stem] = {
            "n_real": len(bboxes),
            "n_fake": len(fake_centers),
            "fake_centers": [[int(x), int(y)] for x, y in fake_centers],
        }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def _generate_large_factory(out_dir: Path, n_pairs: int = 20, cross_markers: bool = False) -> None:
    """
    Generate 5000×4000 factory-like connector images (40 pins, complex background).
    Creates roi_map.json with pin-region ROI per stem (simulates GUI rectangle tool).
    cross_markers: draw thin cross (+) instead of filled dots (factory masked style).
    """
    width, height = 5000, 4000
    unmasked_dir = out_dir / "unmasked"
    masked_dir = out_dir / "masked"
    unmasked_dir.mkdir(parents=True, exist_ok=True)
    masked_dir.mkdir(parents=True, exist_ok=True)
    roi_map: dict[str, list[int]] = {}
    margin_ratio = 0.15

    for i in range(n_pairs):
        img, bboxes, fake_centers = generate_connector_image(
            width=width,
            height=height,
            seed=5000 + i,
            blur_prob=0.2,
            n_fake_pins=25,
            complex_bg=True,
        )
        stem = f"{i+1:02d}"
        unmasked_path = unmasked_dir / f"{stem}.jpg"
        Image.fromarray(img).save(unmasked_path, quality=95)
        masked_img = img.copy()
        draw_fn = bbox_to_cross_marker if cross_markers else bbox_to_green_region
        for x1, y1, x2, y2 in bboxes:
            draw_fn(masked_img, x1, y1, x2, y2)
        masked_path = masked_dir / f"{stem}.jpg"
        Image.fromarray(masked_img).save(masked_path, quality=95)

        # ROI = union of pin bboxes + margin (simulates GUI rectangle selection)
        if bboxes:
            x1 = min(b[0] for b in bboxes)
            y1 = min(b[1] for b in bboxes)
            x2 = max(b[2] for b in bboxes)
            y2 = max(b[3] for b in bboxes)
            mw = max(int((x2 - x1) * margin_ratio), 100)
            mh = max(int((y2 - y1) * margin_ratio), 100)
            roi_map[stem] = [
                max(0, x1 - mw),
                max(0, y1 - mh),
                min(width, x2 + mw),
                min(height, y2 + mh),
            ]

    roi_path = out_dir / "roi_map.json"
    with open(roi_path, "w") as f:
        json.dump(roi_map, f, indent=2)
    print(f"  roi_map.json saved to {roi_path}")


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
    parser.add_argument("--large-factory", action="store_true", help="5000×4000 factory-like data + roi_map.json")
    parser.add_argument("--large-factory-n", type=int, default=20, help="Number of pairs for --large-factory (default 20, use 5 for CI)")
    parser.add_argument("--cross-markers", action="store_true", help="Draw thin cross (+) markers instead of filled dots (factory masked style)")
    args = parser.parse_args()

    root = _project_root()
    out = root / args.output_dir

    if args.large_factory:
        _generate_large_factory(out, n_pairs=args.large_factory_n, cross_markers=args.cross_markers)
        print(f"Generated {args.large_factory_n} pairs (5000×4000, 40 pins, complex bg) in {out}")
        return 0

    if args.n_pairs > 0:
        # Legacy: single set
        _generate_to_dir(out, args.n_pairs, 42, args.blur_prob, args.n_fake_pins,
                        width=args.width, height=args.height, cross_markers=args.cross_markers)
        print(f"Generated {args.n_pairs} pairs in {out}")
    else:
        train_dir = out / "train"
        test_dir = out / "test"
        _generate_to_dir(train_dir, args.train_pairs, 0, args.blur_prob, args.n_fake_pins,
                         width=args.width, height=args.height, cross_markers=args.cross_markers)
        _generate_to_dir(test_dir, args.test_pairs, 1000, args.blur_prob, args.n_fake_pins,
                         width=args.width, height=args.height, cross_markers=args.cross_markers)
        print(f"Generated train: {args.train_pairs} pairs in {train_dir}")
        print(f"Generated test:  {args.test_pairs} pairs in {test_dir}")
    print(f"  unmasked: grayscale, 20 upper + 20 lower rectangular pads, {args.n_fake_pins} fake dots")
    marker_type = "cross (+)" if args.cross_markers else "filled dots"
    print(f"  masked:   green {marker_type} on real pads (40 total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
