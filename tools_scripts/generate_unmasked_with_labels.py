"""
Generate unmasked images + YOLO-format labels for Pin Masking / EXE testing.
No masked images. Labels in output_dir/labels/<stem>.txt (YOLO v26 format).
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools_scripts.generate_realistic_factory_data import generate_realistic_factory_image


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--pairs", type=int, default=5)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("-o", "--output", default="test_data/pin_unmasked_labels")
    args = parser.parse_args()

    out_dir = ROOT / args.output
    unmasked_dir = out_dir / "unmasked"
    labels_dir = out_dir / "labels"
    unmasked_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    width, height = (2000, 1500) if args.fast else (5000, 4000)
    roi_map = {}
    margin = 0.15

    for i in range(args.pairs):
        img, bboxes, _ = generate_realistic_factory_image(
            width=width, height=height, seed=3600 + i, pin_scale=0.6, easy=True
        )
        stem = f"{i+1:02d}"
        Image.fromarray(img).save(unmasked_dir / f"{stem}.jpg", quality=95)

        # YOLO format: class x_center y_center width height (normalized)
        with open(labels_dir / f"{stem}.txt", "w") as f:
            for x1, y1, x2, y2 in bboxes:
                xc = (x1 + x2) / 2 / width
                yc = (y1 + y2) / 2 / height
                w = max(0.02, (x2 - x1) / width)
                h = max(0.02, (y2 - y1) / height)
                f.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

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
    print(f"Generated {args.pairs} pairs: {unmasked_dir}, {labels_dir}")
    print("  Use: unmasked + output_dir/labels for Pin Masking train (no masked folder)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
