"""
Imaging tests — require pyvips (skipped wholesale when unavailable, e.g. on
dev machines without the wheel; they run in the container / CI).

Covers: grid layout solver bounds, compose round-trip with separators, cell
mapping + bleed rejection, headshot crop, tile grid coords.
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    import pyvips  # noqa: F401
    HAS_PYVIPS = True
except Exception:
    HAS_PYVIPS = False



# Silence console log HANDLERS for the whole suite: the failure-path
# tests exercise refusal/retry/fallback branches whose designed
# behavior includes ERROR/WARNING logs, and those lines made build
# logs read as if the build were failing. Handlers only — record flow
# is untouched, so assertLogs() still sees everything.
import logging as _logging


def _silence_console_logging():
    _logging.getLogger().handlers = [_logging.NullHandler()]
    for _name in list(_logging.root.manager.loggerDict):
        _lg = _logging.getLogger(_name)
        if _lg.handlers:
            _lg.handlers = [_logging.NullHandler()]


_silence_console_logging()


class TestLayoutSolver(unittest.TestCase):
    """Pure math — no pyvips needed."""

    def test_landscape_1080p(self):
        from lib.grid_image import solve_layout
        cols, rows = solve_layout(1920, 1080)
        self.assertLessEqual(cols * (1920 + 2) - 2, 9800)
        self.assertLessEqual(rows * (1080 + 2) - 2, 9800)
        self.assertLessEqual(cols * rows, 20)
        self.assertGreaterEqual(cols * rows, 16)  # decent packing

    def test_portrait(self):
        from lib.grid_image import solve_layout
        cols, rows = solve_layout(1080, 1920)
        self.assertLessEqual(cols * (1080 + 2) - 2, 9800)
        self.assertLessEqual(rows * (1920 + 2) - 2, 9800)
        self.assertEqual(cols * rows, 20)  # portrait still reaches the cap

    def test_sd(self):
        from lib.grid_image import solve_layout
        cols, rows = solve_layout(720, 576)
        self.assertEqual(cols * rows, 20)  # cap reached


class TestCellMapping(unittest.TestCase):
    """Pure math — bbox ratios in, cell + frame-relative box out."""

    LAYOUT = dict(n_frames=20, cols=5, rows=4, cell_w=1920, cell_h=1080)

    def _map(self, left, top, w, h):
        from lib.grid_image import map_face_to_cell
        return map_face_to_cell(
            {"Left": left, "Top": top, "Width": w, "Height": h},
            **self.LAYOUT)

    def test_center_of_cell_0(self):
        grid_w = 5 * 1920 + 4 * 2
        grid_h = 4 * 1080 + 3 * 2
        # face centred in cell (0,0)
        r = self._map(800 / grid_w, 400 / grid_h, 200 / grid_w, 250 / grid_h)
        self.assertIsNotNone(r)
        idx, box = r
        self.assertEqual(idx, 0)
        self.assertAlmostEqual(box["l"], 800 / 1920, places=3)

    def test_cell_round_trip_all_cells(self):
        grid_w = 5 * 1920 + 4 * 2
        grid_h = 4 * 1080 + 3 * 2
        for idx_expect in range(20):
            row, col = divmod(idx_expect, 5)
            cx = col * (1920 + 2) + 900
            cy = row * (1080 + 2) + 500
            r = self._map((cx - 60) / grid_w, (cy - 75) / grid_h,
                          120 / grid_w, 150 / grid_h)
            self.assertIsNotNone(r, f"cell {idx_expect}")
            self.assertEqual(r[0], idx_expect)

    def test_bleed_rejected(self):
        grid_w = 5 * 1920 + 4 * 2
        grid_h = 4 * 1080 + 3 * 2
        # face straddling the cell 0 / cell 1 border: centre in cell 0 but
        # 40% of the width hangs into cell 1
        left = (1920 - 120) / grid_w
        r = self._map(left, 400 / grid_h, 200 / grid_w, 200 / grid_h)
        self.assertIsNone(r)

    def test_empty_trailing_cell_rejected(self):
        from lib.grid_image import map_face_to_cell
        # only 3 frames in a 5x4 grid; face centre in cell 4
        grid_w = 5 * 1920 + 4 * 2
        r = map_face_to_cell(
            {"Left": (4 * 1922 + 900) / grid_w, "Top": 0.1,
             "Width": 0.01, "Height": 0.05},
            n_frames=3, cols=5, rows=4, cell_w=1920, cell_h=1080)
        self.assertIsNone(r)


@unittest.skipUnless(HAS_PYVIPS, "pyvips not installed")
class TestCompose(unittest.TestCase):
    def _jpeg(self, w, h, val):
        import numpy as np
        import pyvips
        arr = np.full((h, w, 3), val, dtype="uint8")
        img = pyvips.Image.new_from_memory(arr.tobytes(), w, h, 3, "uchar")
        return img.jpegsave_buffer(Q=85)

    def test_compose_and_separator(self):
        import pyvips
        from lib.grid_image import compose_grid
        frames = [self._jpeg(320, 180, 40 + i * 10) for i in range(4)]
        out = compose_grid(frames, cols=2, rows=2, cell_w=320, cell_h=180)
        img = pyvips.Image.jpegload_buffer(out)
        self.assertEqual(img.width, 2 * 320 + 2)
        self.assertEqual(img.height, 2 * 180 + 2)
        # separator column is black
        sep = img.crop(320, 0, 2, 180).avg()
        self.assertLess(sep, 12)

    def test_headshot_crop(self):
        import pyvips
        from lib.grid_image import crop_headshot
        frame = self._jpeg(640, 360, 128)
        out = crop_headshot(frame, {"l": 0.4, "t": 0.3, "w": 0.2, "h": 0.3},
                            margin=0.3, out_w=120, out_h=160)
        img = pyvips.Image.jpegload_buffer(out)
        self.assertEqual((img.width, img.height), (120, 160))

    def test_downscale_fit_never_upscales(self):
        import pyvips
        from lib.grid_image import downscale_fit
        big = self._jpeg(2560, 1440, 100)
        out = pyvips.Image.jpegload_buffer(downscale_fit(big, 1280))
        self.assertEqual((out.width, out.height), (1280, 720))
        small = self._jpeg(320, 240, 100)
        out2 = pyvips.Image.jpegload_buffer(downscale_fit(small, 1280))
        self.assertEqual((out2.width, out2.height), (320, 240))

    def test_prepare_tile_letterbox_and_placement(self):
        import pyvips
        from lib.index_batch import prepare_tile, tile_box_to_source
        from lib.input_schema import BATCH_TILE_DIM
        tile_w, tile_h = BATCH_TILE_DIM
        # Square tile: fit-inside scale is tile_edge/long_edge for ANY
        # orientation (the minimax property that motivated square).
        # Landscape 3200x1800 -> scale = tile_w/3200, letterboxed vertically.
        src = self._jpeg(3200, 1800, 200)
        tile, placement = prepare_tile(src)
        img = pyvips.Image.jpegload_buffer(tile)
        self.assertEqual((img.width, img.height), (tile_w, tile_h))
        expected_scale = tile_w / 3200
        self.assertAlmostEqual(placement["scale"], expected_scale, places=3)
        self.assertEqual(placement["off_x"], 0)
        self.assertGreater(placement["off_y"], 0)
        content_h = round(1800 * expected_scale)
        # Letterbox bands are black; content band is bright.
        self.assertLess(img.crop(0, 0, tile_w, placement["off_y"]).avg(), 8)
        band = img.crop(0, placement["off_y"], tile_w, content_h).avg()
        self.assertGreater(band, 150)
        # Round-trip a face box through the recorded placement: a face whose
        # top sits 27 tile-px below the content origin.
        fb = {"l": 0.1, "t": (placement["off_y"] + 27) / tile_h,
              "w": 0.2, "h": 54 / tile_h}
        src_box = tile_box_to_source(fb, placement)
        self.assertAlmostEqual(src_box["t"], 27 / expected_scale / 1800,
                               places=3)
        # Portrait source: same scale law on the long edge (height).
        portrait = self._jpeg(1800, 3200, 180)
        _, p_port = prepare_tile(portrait)
        self.assertAlmostEqual(p_port["scale"], tile_h / 3200, places=3)
        self.assertEqual(p_port["off_y"], 0)
        self.assertGreater(p_port["off_x"], 0)  # pillarboxed
        # Small source must NOT be upscaled (scale stays 1.0).
        small = self._jpeg(300, 400, 90)
        _, p2 = prepare_tile(small)
        self.assertEqual(p2["scale"], 1.0)

    def test_native_crop_never_upscales(self):
        import pyvips
        from lib.grid_image import crop_face_native
        # Face box 128x108 px on a 640x360 frame; +30% margin ≈ 204x172 crop.
        frame = self._jpeg(640, 360, 128)
        out = crop_face_native(frame, {"l": 0.4, "t": 0.3, "w": 0.2, "h": 0.3},
                               margin=0.3, max_edge_px=1024)
        img = pyvips.Image.jpegload_buffer(out)
        # Native size retained (no forced tile resize, no upscale to the cap)
        self.assertLess(max(img.width, img.height), 1024)
        self.assertGreater(img.width, 150)   # ~204 px, not 180-tile-forced
        self.assertGreater(img.height, 150)  # ~172 px

    def test_native_crop_caps_long_edge(self):
        import pyvips
        from lib.grid_image import crop_face_native
        # Huge face: crop region far exceeds the cap; must downscale to cap
        # while preserving the crop's aspect ratio.
        frame = self._jpeg(1920, 1080, 128)
        out = crop_face_native(frame, {"l": 0.05, "t": 0.05, "w": 0.9, "h": 0.85},
                               margin=0.0, max_edge_px=512)
        img = pyvips.Image.jpegload_buffer(out)
        self.assertEqual(max(img.width, img.height), 512)
        self.assertAlmostEqual(img.width / img.height, (0.9 * 1920) / (0.85 * 1080),
                               places=1)

    def test_headshot_native_aspect_and_no_upscale(self):
        import pyvips
        from lib.grid_image import crop_headshot_native
        # Small face (128x108 px + 30% margin ≈ 166x140 crop rect, expanded
        # to 3:4 ≈ 166x222): native pixels kept — 3:4 exactly, NOT resized
        # up to the cap.
        frame = self._jpeg(640, 360, 128)
        out = crop_headshot_native(frame, {"l": 0.4, "t": 0.3, "w": 0.2, "h": 0.3},
                                   margin=0.3, ratio_w=3, ratio_h=4,
                                   max_edge_px=512)
        img = pyvips.Image.jpegload_buffer(out)
        self.assertAlmostEqual(img.width / img.height, 3 / 4, places=1)
        self.assertLess(max(img.width, img.height), 512)   # never upscaled
        self.assertGreater(img.height, 200)                # ~222 px native

    def test_headshot_native_caps_long_edge(self):
        import pyvips
        from lib.grid_image import crop_headshot_native
        # Huge face: 3:4 rect exceeds the cap; downscale to it, keep 3:4.
        frame = self._jpeg(1920, 1080, 128)
        out = crop_headshot_native(frame, {"l": 0.3, "t": 0.1, "w": 0.35, "h": 0.75},
                                   margin=0.0, ratio_w=3, ratio_h=4,
                                   max_edge_px=512)
        img = pyvips.Image.jpegload_buffer(out)
        self.assertEqual(max(img.width, img.height), 512)
        self.assertAlmostEqual(img.width / img.height, 3 / 4, places=1)

    def test_tile_grid_coords(self):
        from lib.grid_image import compose_tile_grid, point_in_box
        tiles = [self._jpeg(120, 160, 60 + i) for i in range(5)]
        _, coords = compose_tile_grid(tiles, 120, 160, cols=4)
        self.assertEqual(len(coords), 5)
        # centre of tile 4 (row 1, col 0)
        cx = coords[4]["l"] + coords[4]["w"] / 2
        cy = coords[4]["t"] + coords[4]["h"] / 2
        matches = [i for i, c in enumerate(coords) if point_in_box(cx, cy, c)]
        self.assertEqual(matches, [4])


if __name__ == "__main__":
    unittest.main(verbosity=1)
