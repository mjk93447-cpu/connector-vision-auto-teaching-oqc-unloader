"""
Action #36: Realistic factory-like synthetic data for P/R 0.999 training.
- Very small pins (6-12px), high-res (5000x4000)
- Mold, other connector parts, PCB traces, dust
- Red markers for YOLO target (Square tool compatible)
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools_scripts.generate_pin_test_data import (
    RED_RGB,
    bbox_to_red_region,
    generate_connector_image,
)


def generate_realistic_factory_image(
    width: int = 5000,
    height: int = 4000,
    seed: int = 42,
    pin_scale: float = 0.5,
    easy: bool = False,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], dict]:
    """
    Factory-like: tiny pins, mold, PCB, dust. pin_scale<1 = smaller pins.
    Returns (img, bboxes, meta).
    """
    rng = np.random.default_rng(seed)
    base_w, base_h = 640, 480
    sw, sh = width / base_w, height / base_h
    scale = pin_scale
    pad_w = max(6, int(12 * sw * scale))
    pad_h = max(4, int(4 * sh * scale))
    pitch_px = max(12, int(18 * sw))
    y_upper = int(120 * sh)
    y_lower = height // 2 + int(80 * sh)
    x_start = int(80 * sw)

    img = np.zeros((height, width, 3), dtype=np.uint8)

    # PCB base + texture
    base = rng.integers(18, 40, (height, width, 3), dtype=np.uint8)
    img[:] = base
    noise = rng.integers(-6, 7, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Mold / connector body (dark rectangle, confuser)
    mold_margin = int(0.1 * min(width, height))
    mold_x1 = mold_margin
    mold_x2 = width - mold_margin
    mold_y1 = int(height * 0.2)
    mold_y2 = int(height * 0.8)
    mold_val = rng.integers(40, 80, (mold_y2 - mold_y1, mold_x2 - mold_x1, 3), dtype=np.uint8)
    img[mold_y1:mold_y2, mold_x1:mold_x2] = np.clip(
        img[mold_y1:mold_y2, mold_x1:mold_x2].astype(np.int16) + mold_val - 50, 0, 255
    ).astype(np.uint8)

    # PCB traces
    for _ in range(rng.integers(30, 80)):
        y = rng.integers(0, height)
        val = rng.integers(30, 70)
        for t in range(rng.integers(1, 3)):
            yy = min(height - 1, y + t)
            img[yy, :] = np.clip(img[yy, :].astype(np.int16) + val, 0, 255).astype(np.uint8)

    # Pins (20+20)
    x_positions = [x_start + i * pitch_px for i in range(20)]
    bboxes = []
    for xc in x_positions:
        x1 = max(0, xc - pad_w // 2)
        y1 = y_upper
        x2 = min(width, x1 + pad_w)
        y2 = min(height, y1 + pad_h)
        val = rng.integers(200, 256)
        img[y1:y2, x1:x2] = [val, val, val]
        bboxes.append((x1, y1, x2, y2))
    for xc in x_positions:
        x1 = max(0, xc - pad_w // 2)
        y1 = y_lower
        x2 = min(width, x1 + pad_w)
        y2 = min(height, y1 + pad_h)
        val = rng.integers(200, 256)
        img[y1:y2, x1:x2] = [val, val, val]
        bboxes.append((x1, y1, x2, y2))

    # Fake pins / dust (confusers). --easy: fewer for Action #36 P/R experiments
    avoid = max(80, int(0.05 * min(width, height)))
    n_fake = rng.integers(10, 25) if getattr(generate_realistic_factory_image, "_easy", False) else rng.integers(40, 120)
    for _ in range(n_fake):
        cx = rng.integers(avoid, width - avoid)
        cy = rng.integers(avoid, height - avoid)
        if any(abs(cx - (b[0] + b[2]) // 2) < avoid and abs(cy - (b[1] + b[3]) // 2) < avoid for b in bboxes):
            continue
        r = rng.integers(2, max(4, pad_w))
        val = rng.integers(160, 240)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        img[ny, nx] = [val, val, val]

    # Blur
    if rng.random() < 0.3:
        img = np.array(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=0.8)))

    noise = rng.integers(-3, 4, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img, bboxes, {"pad_w": pad_w, "pad_h": pad_h}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--pairs", type=int, default=20, help="Number of pairs")
    parser.add_argument("--fast", action="store_true", help="Smaller 2000x1500 for quick test")
    parser.add_argument("--easy", action="store_true", help="Fewer fake pins for Action #36 P/R experiments")
    args = parser.parse_args()
    out_dir = ROOT / "test_data" / "pin_realistic_factory"
    n_pairs = args.pairs
    width, height = (2000, 1500) if args.fast else (5000, 4000)
    unmasked_dir = out_dir / "unmasked"
    masked_dir = out_dir / "masked"
    unmasked_dir.mkdir(parents=True, exist_ok=True)
    masked_dir.mkdir(parents=True, exist_ok=True)
    roi_map = {}
    margin = 0.15

    for i in range(n_pairs):
        img, bboxes, _ = generate_realistic_factory_image(
            width=width, height=height, seed=3600 + i, pin_scale=0.6, easy=args.easy
        )
        stem = f"{i+1:02d}"
        Image.fromarray(img).save(unmasked_dir / f"{stem}.jpg", quality=95)
        masked = img.copy()
        # Action #36: min 14px red marker for better YOLO learning (was 10)
        for x1, y1, x2, y2 in bboxes:
            bbox_to_red_region(masked, x1, y1, x2, y2, size=max(14, min(x2 - x1, y2 - y1)))
        Image.fromarray(masked).save(masked_dir / f"{stem}.jpg", quality=95)

        if bboxes:
            x1 = min(b[0] for b in bboxes)
            y1 = min(b[1] for b in bboxes)
            x2 = max(b[2] for b in bboxes)
            y2 = max(b[3] for b in bboxes)
            mw = max(int((x2 - x1) * margin), 150)
            mh = max(int((y2 - y1) * margin), 150)
            roi_map[stem] = [
                max(0, x1 - mw), max(0, y1 - mh),
                min(width, x2 + mw), min(height, y2 + mh),
            ]

    (out_dir / "roi_map.json").write_text(json.dumps(roi_map, indent=2), encoding="utf-8")
    print(f"Generated {n_pairs} pairs in {out_dir}")
    print(f"  pin_realistic_factory: {width}x{height}, tiny pins, mold, dust, red markers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
