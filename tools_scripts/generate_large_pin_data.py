"""
Generate 5496x3672 large-format pin test data for ROI benchmark.
Places connector (640x480) at center of large canvas.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools_scripts.generate_pin_test_data import generate_connector_image, bbox_to_green_region

W, H = 5496, 3672
CONNECTOR_W, CONNECTOR_H = 640, 480


def generate_large_image(seed: int = 42) -> tuple:
    """Generate 5496x3672 image with connector at center."""
    conn_img, bboxes, fake_centers = generate_connector_image(
        width=CONNECTOR_W,
        height=CONNECTOR_H,
        seed=seed,
        blur_prob=0.2,
        n_fake_pins=4,
    )
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    ox = (W - CONNECTOR_W) // 2
    oy = (H - CONNECTOR_H) // 2
    canvas[oy : oy + CONNECTOR_H, ox : ox + CONNECTOR_W] = conn_img
    offset_bboxes = [(x1 + ox, y1 + oy, x2 + ox, y2 + oy) for x1, y1, x2, y2 in bboxes]
    offset_fake = [(cx + ox, cy + oy) for cx, cy in fake_centers]
    return canvas, offset_bboxes, offset_fake


def main() -> int:
    out = Path(__file__).resolve().parent.parent / "test_data" / "pin_large_5496x3672"
    unmasked_dir = out / "unmasked"
    masked_dir = out / "masked"
    unmasked_dir.mkdir(parents=True, exist_ok=True)
    masked_dir.mkdir(parents=True, exist_ok=True)

    for i in range(20):
        img, bboxes, _ = generate_large_image(seed=100 + i)
        stem = f"{i+1:02d}"
        Image.fromarray(img).save(unmasked_dir / f"{stem}.jpg", quality=95)
        masked = img.copy()
        for x1, y1, x2, y2 in bboxes:
            bbox_to_green_region(masked, x1, y1, x2, y2)
        Image.fromarray(masked).save(masked_dir / f"{stem}.jpg", quality=95)

    print(f"Generated 20 pairs at {out}")
    print(f"  Size: {W}x{H}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
