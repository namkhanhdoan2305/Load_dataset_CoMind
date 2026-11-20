"""Clock identity, unit conversion and timestamp consistency checks."""

import unittest
from unittest import mock
from types import SimpleNamespace
import tempfile

from time_contract import clock_id, graph_time_us, microseconds
from sgg_pipeline import VideoReader


class TimeContractTests(unittest.TestCase):
    def test_units_and_recording_scopes(self):
        self.assertEqual(microseconds(0.2), 200000)
        self.assertNotEqual(clock_id("a"), clock_id("b"))

    def test_invalid_times_fail(self):
        for value in (-1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                microseconds(value)

    def test_mixed_clock_or_mixed_units_fail(self):
        for row in ({"time_s": 0.2, "time_us": 200, "clock_id": clock_id("a")},
                    {"time_s": 0.2, "time_us": 200000, "clock_id": clock_id("b")}):
            with self.assertRaises(ValueError):
                graph_time_us(row, "a")

    def test_valid_clock_preserves_time(self):
        self.assertEqual(graph_time_us({"time_s": 0.2, "time_us": 200000,
                                        "clock_id": clock_id("a")}, "a"), 200000)

    def video(self, fps=30, offset=0):
        class Capture:
            cursor = 0
            last = 0
            def isOpened(self):
                return True
            def get(self, key):
                return {1: fps, 2: 301, 4: self.last / fps * 1000 + offset}.get(key, 0)
            def set(self, key, value):
                self.cursor = value
            def grab(self):
                self.cursor += 1
                return True
            def read(self):
                self.last = self.cursor
                self.cursor += 1
                return True, SimpleNamespace(shape=(960, 960, 3))
            def release(self):
                pass
        return SimpleNamespace(VideoCapture=lambda path: Capture(), CAP_PROP_FPS=1,
                               CAP_PROP_FRAME_COUNT=2, CAP_PROP_POS_FRAMES=3, CAP_PROP_POS_MSEC=4,
                               imwrite=lambda path, image: True)

    def test_decoder_records_pts_and_a_shared_integer_grid(self):
        with mock.patch.dict("sys.modules", {"cv2": self.video()}), tempfile.TemporaryDirectory() as folder:
            reader = VideoReader("unused", tracking_fps=10)
            rows = reader.decode(0, 2, folder, 960)
        self.assertEqual([r["source_frame_index"] for r in rows], [0, 3, 6])
        self.assertEqual([r["time_us"] for r in rows], [0, 100000, 200000])
        self.assertEqual(rows[2]["source_pts_us"], 200000)
        self.assertTrue(all(r["pts_verified"] for r in rows))

    def test_decoder_rejects_bad_pts_instead_of_silently_using_frame_number(self):
        with mock.patch.dict("sys.modules", {"cv2": self.video(offset=10)}), tempfile.TemporaryDirectory() as folder:
            reader = VideoReader("unused", tracking_fps=10)
            with self.assertRaisesRegex(RuntimeError, "timestamp disagrees"):
                reader.decode(0, 2, folder, 960)

    def test_fractional_fps_samples_a_past_frame_for_the_grid(self):
        with mock.patch.dict("sys.modules", {"cv2": self.video(fps=29.97)}), tempfile.TemporaryDirectory() as folder:
            reader = VideoReader("unused", tracking_fps=10)
            rows = reader.decode(0, 2, folder, 960)
        self.assertEqual(rows[2]["source_frame_index"], 5)
        self.assertLessEqual(rows[2]["source_pts_us"], rows[2]["time_us"])


if __name__ == "__main__":
    unittest.main()
