"""Checks for subset stability, persistent progress and resuming Drive downloads."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import comind_download as source
import comind_to_hf as downloader
import load_comind_drive as preset


class SubsetTests(unittest.TestCase):
    def test_30_is_prefix_of_50_and_never_contains_official_test(self):
        ids = preset.select_sessions(30)
        self.assertEqual(ids, preset.select_sessions(50)[:30])
        self.assertEqual(len(set(ids)), 30)
        self.assertFalse(set(ids).intersection(source.TEST_IDS))

    def test_invalid_session_counts_are_rejected(self):
        for count in (0, 56):
            with self.assertRaises(ValueError):
                preset.select_sessions(count)

    def test_unmounted_default_drive_stops_before_network(self):
        with mock.patch.object(preset.os.path, "ismount", return_value=False), \
                mock.patch.object(downloader, "make_plan") as plan:
            with self.assertRaises(SystemExit):
                preset.main([])
        plan.assert_not_called()


class DirectoryProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.target = Path(self.temp.name)
        self.manifest = {
            "url_id": "recording", "dest_subpath": "recordings/recording",
            "hash_algorithm": "sha256", "raw": b"{}",
            "files": [{"path": name, "size": len(name),
                       "hash": hashlib.sha256(name.encode()).hexdigest()}
                      for name in ("a.mp4", "b.csv")],
        }

    def fake_verified(self, man, entry, dest, base_url):
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(entry["path"].encode())

    def test_budget_stops_between_files_and_next_run_finishes(self):
        with mock.patch.object(downloader, "download_verified", self.fake_verified), \
                mock.patch.object(downloader.time, "monotonic", side_effect=[0, 0, 1, 3601]):
            result = downloader.save_to_directory([self.manifest], self.target, max_hours=1)
        self.assertEqual(result["state"], "budget_reached")
        self.assertEqual(result["verified_files"], 1)
        self.assertTrue((self.target / "recordings/recording/a.mp4").is_file())
        self.assertFalse((self.target / "recordings/recording/b.csv").exists())
        with mock.patch.object(downloader, "download_verified", self.fake_verified):
            result = downloader.save_to_directory([self.manifest], self.target)
        self.assertEqual(result["state"], "complete")
        self.assertEqual(result["verified_files"], 2)
        persisted = json.loads((self.target / "download-status.json").read_text())
        self.assertEqual(persisted["state"], "complete")

    def test_download_failure_is_reported_without_marking_complete(self):
        with mock.patch.object(downloader, "download_verified", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                downloader.save_to_directory([self.manifest], self.target)
        result = json.loads((self.target / "download-status.json").read_text())
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["verified_files"], 0)
        self.assertEqual(result["current_file"], "recordings/recording/a.mp4")


if __name__ == "__main__":
    unittest.main()
