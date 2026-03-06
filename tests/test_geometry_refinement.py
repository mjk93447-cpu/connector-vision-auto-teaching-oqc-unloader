"""
Tests for geometry_refinement: 20+20 fixed grid, interpolation, capping, edge cases.
"""
import unittest
from pin_detection.geometry_refinement import (
    split_upper_lower,
    refine_to_fixed_grid,
    _refine_row,
    _clamp_bbox,
    _filter_by_y_band,
    _remove_y_outliers,
)


class TestClampBbox(unittest.TestCase):
    def test_clamp_in_range(self):
        out = _clamp_bbox(0.5, 0.5, 0.02, 0.01)
        self.assertAlmostEqual(out[0], 0.5)
        self.assertAlmostEqual(out[1], 0.5)

    def test_clamp_out_of_range(self):
        out = _clamp_bbox(1.5, -0.1, 0.02, 0.01)
        self.assertLessEqual(out[0], 1)
        self.assertGreaterEqual(out[1], 0)


class TestFilterByYBand(unittest.TestCase):
    def test_keeps_in_band(self):
        dets = [(0.3, 0.2, 0.02, 0.01), (0.5, 0.65, 0.02, 0.01)]
        u, uc = _filter_by_y_band(dets, [0.9, 0.8], upper=True)
        self.assertEqual(len(u), 1)
        self.assertAlmostEqual(u[0][1], 0.2)


class TestSplitUpperLower(unittest.TestCase):
    def test_split_by_mid_y(self):
        # (xc, yc, w, h) normalized
        dets = [(0.3, 0.2, 0.02, 0.01), (0.5, 0.7, 0.02, 0.01)]
        u, l, uc, lc = split_upper_lower(dets, [0.9, 0.8])
        self.assertEqual(len(u), 1)
        self.assertEqual(len(l), 1)
        self.assertEqual(u[0][1], 0.2)
        self.assertEqual(l[0][1], 0.7)


class TestRefineRow(unittest.TestCase):
    def test_cap_over_20(self):
        w, h = 640, 480
        dets = [(0.1 + i * 0.04, 0.2, 0.02, 0.01) for i in range(25)]
        confs = [0.5 + i * 0.02 for i in range(25)]
        out = _refine_row(dets, confs, w, h, n_slots=20)
        self.assertEqual(len(out), 20)

    def test_interpolate_under_20(self):
        w, h = 640, 480
        # 19 positions with one gap (simulate missing between 9 and 11)
        dets = [(0.1 + i * 0.04, 0.2, 0.02, 0.01) for i in range(19)]
        out = _refine_row(dets, None, w, h, n_slots=20)
        self.assertEqual(len(out), 20)


class TestRefineToFixedGrid(unittest.TestCase):
    def test_exactly_40_output(self):
        w, h = 640, 480
        # 38 detections (19 upper + 19 lower)
        dets = [(0.1 + i * 0.04, 0.2, 0.02, 0.01) for i in range(19)]
        dets += [(0.1 + i * 0.04, 0.6, 0.02, 0.01) for i in range(19)]
        out = refine_to_fixed_grid(dets, [0.9] * 38, w, h, n_per_row=20)
        self.assertEqual(len(out), 40)

    def test_empty_row_uses_template(self):
        w, h = 640, 480
        # Only upper row (20), lower empty
        dets = [(0.1 + i * 0.04, 0.2, 0.02, 0.01) for i in range(20)]
        out = refine_to_fixed_grid(dets, [0.9] * 20, w, h, n_per_row=20)
        self.assertEqual(len(out), 40)
        # Lower row should have y ~ 0.65
        lower = [d for d in out if d[1] >= 0.5]
        self.assertEqual(len(lower), 20)

    def test_out_of_range_filtered(self):
        w, h = 640, 480
        # One detection at y=0.9 (upper band) - should go to lower
        dets = [(0.3, 0.2, 0.02, 0.01)] * 19 + [(0.3, 0.65, 0.02, 0.01)] * 19
        dets.append((0.5, 0.01, 0.02, 0.01))  # y too high for upper - edge
        out = refine_to_fixed_grid(dets, [0.9] * len(dets), w, h, n_per_row=20)
        self.assertEqual(len(out), 40)

    def test_output_bounds_clamped(self):
        w, h = 640, 480
        dets = [(0.1 + i * 0.04, 0.2, 0.02, 0.01) for i in range(20)]
        dets += [(0.1 + i * 0.04, 0.6, 0.02, 0.01) for i in range(20)]
        out = refine_to_fixed_grid(dets, [0.9] * 40, w, h, n_per_row=20)
        for d in out:
            self.assertGreaterEqual(d[0], 0)
            self.assertLessEqual(d[0], 1)
            self.assertGreaterEqual(d[1], 0)
            self.assertLessEqual(d[1], 1)
