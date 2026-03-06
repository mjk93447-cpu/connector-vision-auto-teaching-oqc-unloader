"""
Self-tests for pin_detection: data pair connectivity, annotation extraction.
"""
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


class TestAnnotation(unittest.TestCase):
    """Test annotation extraction from synthetic masked images."""

    def test_extract_green_mask(self):
        from pin_detection.annotation import extract_green_mask
        # Pure green pixel
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[2:4, 2:4] = [0, 255, 0]  # broadcasts to 2x2 region
        mask = extract_green_mask(img)
        self.assertEqual(mask[2, 2], 255)
        self.assertEqual(mask[0, 0], 0)

    def test_cluster_to_bbox(self):
        from pin_detection.annotation import extract_green_mask, cluster_to_bbox
        img = np.zeros((20, 20, 3), dtype=np.uint8)
        img[5:8, 5:8] = [0, 255, 0]  # 3x3 green block
        mask = extract_green_mask(img)
        bboxes = cluster_to_bbox(mask)
        self.assertGreaterEqual(len(bboxes), 1)
        x1, y1, x2, y2 = bboxes[0]
        self.assertLessEqual(x1, 5)
        self.assertGreaterEqual(x2, 8)
        self.assertLessEqual(y1, 5)
        self.assertGreaterEqual(y2, 8)

    def test_masked_to_annotations_synthetic(self):
        """Create synthetic masked image with green dots, verify annotations."""
        from pin_detection.annotation import masked_image_to_annotations
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "masked.png"
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            img[:, :] = [50, 50, 50]  # background
            # Two green regions
            img[10:15, 10:15] = [0, 255, 0]
            img[50:55, 50:55] = [0, 255, 0]
            Image.fromarray(img).save(path)
            _, anns = masked_image_to_annotations(path)
            self.assertGreaterEqual(len(anns), 2, "Should find at least 2 pin regions")
            for xc, yc, w, h in anns:
                self.assertGreaterEqual(xc, 0)
                self.assertLessEqual(xc, 1)
                self.assertGreaterEqual(yc, 0)
                self.assertLessEqual(yc, 1)
                self.assertGreater(w, 0)
                self.assertGreater(h, 0)


class TestDatasetPairing(unittest.TestCase):
    """Test data pair connectivity and validation."""

    def test_find_masked_pair_same_name(self):
        from pin_detection.dataset import _find_masked_pair
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            unmasked = tmp / "unmasked"
            masked = tmp / "masked"
            unmasked.mkdir()
            masked.mkdir()
            (unmasked / "01.jpg").touch()
            (masked / "01.jpg").touch()
            m = _find_masked_pair(unmasked / "01.jpg", masked)
            self.assertEqual(m.name, "01.jpg")

    def test_find_masked_pair_suffix(self):
        from pin_detection.dataset import _find_masked_pair
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            unmasked = tmp / "unmasked"
            masked = tmp / "masked"
            unmasked.mkdir()
            masked.mkdir()
            (unmasked / "01.jpg").touch()
            (masked / "01_masked.jpg").touch()
            m = _find_masked_pair(unmasked / "01.jpg", masked)
            self.assertEqual(m.name, "01_masked.jpg")

    def test_find_masked_pair_missing_raises(self):
        from pin_detection.dataset import _find_masked_pair
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            unmasked = tmp / "unmasked"
            masked = tmp / "masked"
            unmasked.mkdir()
            masked.mkdir()
            (unmasked / "01.jpg").touch()
            with self.assertRaises(FileNotFoundError):
                _find_masked_pair(unmasked / "01.jpg", masked)

    def test_pair_dimension_validation(self):
        """Unmasked and masked must have same dimensions for correct annotation."""
        from pin_detection.dataset import _add_one_pair
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            unmasked = tmp / "unmasked"
            masked = tmp / "masked"
            unmasked.mkdir()
            masked.mkdir()
            # Create images
            u_img = np.zeros((100, 100, 3), dtype=np.uint8)
            u_img[:, :] = 128
            m_img = np.zeros((100, 100, 3), dtype=np.uint8)
            m_img[:, :] = 50
            m_img[10:15, 10:15] = [0, 255, 0]
            Image.fromarray(u_img).save(unmasked / "01.jpg")
            Image.fromarray(m_img).save(masked / "01.jpg")
            out_img = tmp / "out" / "images" / "train"
            out_lbl = tmp / "out" / "labels" / "train"
            out_img.mkdir(parents=True)
            out_lbl.mkdir(parents=True)
            _add_one_pair(unmasked / "01.jpg", masked / "01.jpg", out_img, out_lbl)
            self.assertTrue((out_img / "01.jpg").exists())
            self.assertTrue((out_lbl / "01.txt").exists())
            with open(out_lbl / "01.txt") as f:
                lines = f.readlines()
            self.assertGreater(len(lines), 0)

    def test_prepare_from_dirs_empty_unmasked_raises(self):
        from pin_detection.dataset import prepare_yolo_dataset_from_dirs
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            u_dir = tmp / "unmasked"
            m_dir = tmp / "masked"
            u_dir.mkdir()
            m_dir.mkdir()
            # No images in unmasked
            out = tmp / "out"
            with self.assertRaises(ValueError) as ctx:
                prepare_yolo_dataset_from_dirs(u_dir, m_dir, out)
            self.assertIn("No unmasked images", str(ctx.exception))

    def test_prepare_from_dirs_dimension_mismatch_raises(self):
        """If unmasked and masked have different sizes, validation must raise."""
        from pin_detection.dataset import prepare_yolo_dataset_from_dirs
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            u_dir = tmp / "unmasked"
            m_dir = tmp / "masked"
            u_dir.mkdir()
            m_dir.mkdir()
            # Unmasked 100x100
            Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(u_dir / "01.jpg")
            # Masked 200x200 - DIFFERENT!
            m_img = np.zeros((200, 200, 3), dtype=np.uint8)
            m_img[20:30, 20:30] = [0, 255, 0]
            Image.fromarray(m_img).save(m_dir / "01.jpg")
            out = tmp / "out"
            with self.assertRaises(ValueError) as ctx:
                prepare_yolo_dataset_from_dirs(u_dir, m_dir, out)
            self.assertIn("size mismatch", str(ctx.exception))

    def test_prepare_from_dirs_val_split(self):
        """With val_split=0.2, train and val should have different images."""
        from pin_detection.dataset import prepare_yolo_dataset_from_dirs
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            u_dir = tmp / "unmasked"
            m_dir = tmp / "masked"
            u_dir.mkdir()
            m_dir.mkdir()
            base = np.zeros((100, 100, 3), dtype=np.uint8)
            m_img = base.copy()
            m_img[20:30, 20:30] = [0, 255, 0]
            for i in range(5):
                Image.fromarray(base).save(u_dir / f"{i+1:02d}.jpg")
                Image.fromarray(m_img).save(m_dir / f"{i+1:02d}.jpg")
            out = tmp / "out"
            prepare_yolo_dataset_from_dirs(u_dir, m_dir, out, val_split=0.2, seed=42)
            train_imgs = list((out / "images" / "train").iterdir())
            val_imgs = list((out / "images" / "val").iterdir())
            self.assertGreater(len(train_imgs), 0)
            self.assertGreater(len(val_imgs), 0)
            self.assertEqual(len(train_imgs) + len(val_imgs), 5)


class TestDimensionValidation(unittest.TestCase):
    """Verify dimension check is added and works."""

    def test_validate_pair_dimensions(self):
        from pin_detection.dataset import validate_pair_dimensions
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            u = tmp / "u.jpg"
            m = tmp / "m.jpg"
            Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(u)
            Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(m)
            validate_pair_dimensions(u, m)  # same - should pass

    def test_validate_pair_dimensions_mismatch_raises(self):
        from pin_detection.dataset import validate_pair_dimensions
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            u = tmp / "u.jpg"
            m = tmp / "m.jpg"
            Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(u)
            Image.fromarray(np.zeros((200, 200, 3), dtype=np.uint8)).save(m)
            with self.assertRaises(ValueError):
                validate_pair_dimensions(u, m)


class TestRealisticGenerator(unittest.TestCase):
    """Test realistic connector pin generator."""
    def test_generator_black_bg_white_pins(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from tools_scripts.generate_pin_test_data import generate_connector_image
        img, bboxes, fake_centers = generate_connector_image(
            seed=42, blur_prob=0, n_fake_pins=4
        )
        self.assertEqual(img.shape[2], 3)
        self.assertGreaterEqual(img.min(), 0)
        self.assertLessEqual(img.max(), 255)
        # Background should be black (with possible slight sensor noise)
        self.assertLessEqual(img[0, 0, 0], 10)
        self.assertLessEqual(img[0, 0, 1], 10)
        self.assertLessEqual(img[0, 0, 2], 10)
        # Pins should be white
        y1, x1 = bboxes[0][1], bboxes[0][0]
        self.assertGreaterEqual(img[y1, x1, 0], 200)
        self.assertGreaterEqual(img[y1, x1, 1], 200)
        self.assertEqual(len(bboxes), 40)
        self.assertLessEqual(len(fake_centers), 4, "Fake pins may be fewer if overlapping real pins")

    def test_masked_green_only_on_real_pins(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from tools_scripts.generate_pin_test_data import generate_connector_image, bbox_to_green_region
        img, bboxes, fake_centers = generate_connector_image(seed=42, blur_prob=0, n_fake_pins=2)
        masked = img.copy()
        for x1, y1, x2, y2 in bboxes:
            bbox_to_green_region(masked, x1, y1, x2, y2)
        from pin_detection.annotation import extract_green_mask, cluster_to_bbox
        mask = extract_green_mask(masked)
        bboxes_out = cluster_to_bbox(mask)
        self.assertEqual(len(bboxes_out), 40, "Green only on real pins")


if __name__ == "__main__":
    unittest.main()
