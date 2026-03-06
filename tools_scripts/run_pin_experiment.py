"""
Run pin detection experiment: inference on test images, mAP50, Recall/Precision, confusion matrix.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from pin_detection.inference import run_inference
from pin_detection.annotation import masked_image_to_annotations


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def iou_bbox(b1: tuple, b2: tuple, w: int, h: int) -> float:
    """Compute IoU between two YOLO-format bboxes (xc,yc,w,h normalized)."""
    def to_xyxy(xc, yc, bw, bh):
        x1 = (xc - bw/2) * w
        y1 = (yc - bh/2) * h
        x2 = (xc + bw/2) * w
        y2 = (yc + bh/2) * h
        return x1, y1, x2, y2
    a1 = to_xyxy(*b1)
    a2 = to_xyxy(*b2)
    x1 = max(a1[0], a2[0])
    y1 = max(a1[1], a2[1])
    x2 = min(a1[2], a2[2])
    y2 = min(a1[3], a2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (a1[2] - a1[0]) * (a1[3] - a1[1])
    area2 = (a2[2] - a2[0]) * (a2[3] - a2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def center_distance(g: tuple, p: tuple, w: int, h: int) -> float:
    """Pixel distance between bbox centers."""
    gx = g[0] * w
    gy = g[1] * h
    px = p[0] * w
    py = p[1] * h
    return ((gx - px) ** 2 + (gy - py) ** 2) ** 0.5


def match_detections(gt: list, pred: list, w: int, h: int, iou_thresh: float = 0.5, max_dist_px: float = 0) -> tuple:
    """Match pred to gt. Use IoU if iou_thresh>0; else use center distance when max_dist_px>0."""
    used_gt = [False] * len(gt)
    tp = 0
    for p in pred:
        best_j = -1
        if iou_thresh > 0:
            best_iou = 0.0
            for j, g in enumerate(gt):
                if used_gt[j]:
                    continue
                iou = iou_bbox(g, p, w, h)
                if iou >= iou_thresh and iou > best_iou:
                    best_iou, best_j = iou, j
        elif max_dist_px > 0:
            best_dist = 1e9
            for j, g in enumerate(gt):
                if used_gt[j]:
                    continue
                dist = center_distance(g, p, w, h)
                if dist <= max_dist_px and dist < best_dist:
                    best_dist, best_j = dist, j
        if best_j >= 0:
            used_gt[best_j] = True
            tp += 1
    fn = sum(1 for u in used_gt if not u)
    fp = len(pred) - tp
    return tp, fp, fn


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model .pt path")
    parser.add_argument("--unmasked-dir", required=True, help="Unmasked images")
    parser.add_argument("--masked-dir", required=True, help="Masked images (GT)")
    parser.add_argument("--conf", type=float, default=0.01, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.25, help="IoU threshold for matching (0=use dist)")
    parser.add_argument("--max-dist", type=float, default=0, help="Max center distance (px) for fallback matching")
    parser.add_argument("--run-map50", action="store_true", help="Run YOLO val() for mAP50 (needs test dataset)")
    parser.add_argument("--test-dataset-dir", help="Prepare YOLO test dataset here and run mAP50")
    parser.add_argument("--save-dir", help="Save metrics and confusion matrix JSON")
    args = parser.parse_args()

    root = _project_root()
    model_path = Path(args.model)
    unmasked_dir = Path(args.unmasked_dir)
    masked_dir = Path(args.masked_dir)

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
    u_files = sorted([f for f in unmasked_dir.iterdir() if f.suffix.lower() in IMG_EXTS])

    total_tp, total_fp, total_fn = 0, 0, 0
    total_fp_on_fake = 0  # detections on fake pins (noise)
    latencies = []

    meta_path = unmasked_dir.parent / "meta.json"
    meta = {}
    if meta_path.exists():
        import json
        meta = json.load(meta_path.open())

    for u_path in u_files:
        m_path = masked_dir / u_path.name
        if not m_path.exists():
            m_path = masked_dir / f"{u_path.stem}_masked{u_path.suffix}"
        if not m_path.exists():
            print(f"Skip {u_path.name}: no masked pair")
            continue

        # GT from masked
        _, gt_anns = masked_image_to_annotations(m_path)
        from PIL import Image
        with Image.open(u_path) as im:
            w, h = im.size

        # Inference
        t0 = time.perf_counter()
        _, detections, _ = run_inference(
            model_path=model_path,
            image_path=u_path,
            conf_threshold=args.conf,
            cap_precision=True,
        )
        latencies.append((time.perf_counter() - t0) * 1000)

        tp, fp, fn = match_detections(gt_anns, detections, w, h, args.iou, args.max_dist)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Count FP on fake pins (noise) if meta available
        stem = u_path.stem
        if stem in meta and meta[stem].get("fake_centers"):
            fake_centers = meta[stem]["fake_centers"]
            used_gt = [False] * len(gt_anns)
            for p in detections:
                best_j = -1
                best_dist = 1e9
                for j, g in enumerate(gt_anns):
                    if used_gt[j]:
                        continue
                    dist = center_distance(g, p, w, h)
                    if dist <= args.max_dist and dist < best_dist:
                        best_dist, best_j = dist, j
                if best_j >= 0:
                    used_gt[best_j] = True
                else:
                    # p is FP - check if near fake center
                    px, py = int(p[0] * w), int(p[1] * h)
                    for fc in fake_centers:
                        if ((px - fc[0]) ** 2 + (py - fc[1]) ** 2) ** 0.5 <= 15:
                            total_fp_on_fake += 1
                            break

    n = len(latencies)
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0

    # Confusion matrix (object detection: TP, FP, FN)
    cm = {"TP": total_tp, "FP": total_fp, "FN": total_fn}
    total_gt = total_tp + total_fn
    total_pred = total_tp + total_fp

    map50 = None
    if args.run_map50 or args.test_dataset_dir:
        from pin_detection.dataset import prepare_yolo_test_dataset
        from ultralytics import YOLO
        test_dir = Path(args.test_dataset_dir) if args.test_dataset_dir else unmasked_dir.parent / "yolo_test"
        test_dir.mkdir(parents=True, exist_ok=True)
        data_yaml = prepare_yolo_test_dataset(
            unmasked_dir, masked_dir, test_dir, class_id=0
        )
        model = YOLO(model_path)
        metrics = model.val(data=str(data_yaml), split="val", verbose=False)
        map50 = float(getattr(metrics.box, "map50", 0) or 0)
        if map50 is not None:
            print(f"\nmAP50 (YOLO val): {map50:.4f}")

    print("=" * 50)
    print("PIN DETECTION EXPERIMENT RESULTS")
    print("=" * 50)
    print(f"Images: {n}")
    print(f"TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    if total_fp_on_fake > 0:
        print(f"FP on fake (noise): {total_fp_on_fake}")
    print(f"Recall:    {recall:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"F1:        {f1:.2%}")
    if map50 is not None:
        print(f"mAP50:     {map50:.4f}")
    print(f"Inference: {sum(latencies)/n:.1f} ms/img avg (conf={args.conf})")
    print("\nConfusion Matrix (pin detection):")
    print("  |              | Predicted Pin | Predicted BG |")
    print("  | Actual Pin   |     %5d      |     %5d     |" % (total_tp, total_fn))
    print("  | Actual BG    |     %5d      |      -      |" % total_fp)
    print("=" * 50)

    if args.save_dir:
        save_path = Path(args.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        metrics_out = {
            "n_images": n,
            "TP": total_tp, "FP": total_fp, "FN": total_fn,
            "Recall": recall, "Precision": precision, "F1": f1,
            "confusion_matrix": cm,
            "fp_on_fake": total_fp_on_fake,
            "inference_ms_avg": sum(latencies) / n if latencies else 0,
        }
        if map50 is not None:
            metrics_out["mAP50"] = map50
        with open(save_path / "metrics.json", "w") as f:
            json.dump(metrics_out, f, indent=2)
        with open(save_path / "confusion_matrix.json", "w") as f:
            json.dump(cm, f, indent=2)
        print(f"Saved to {save_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
