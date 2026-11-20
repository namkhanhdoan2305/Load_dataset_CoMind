"""Current subset selection and verified local download integrity."""

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import comind_data as data
import comind_source as source


class SubsetTests(unittest.TestCase):
    def test_30_is_prefix_of_50_and_never_contains_official_test(self):
        ids = data.select_sessions(30)
        self.assertEqual(ids, data.select_sessions(50)[:30])
        self.assertEqual(len(set(ids)), 30)
        self.assertFalse(set(ids).intersection(source.TEST_IDS))
        self.assertEqual(ids[0], "8ee6c694-38be-4a90-ad5a-75ef4449e972")

    def test_invalid_session_counts_are_rejected(self):
        for count in (0, 56):
            with self.assertRaises(ValueError):
                data.select_sessions(count)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.target = Path(self.temp.name) / "payload"
        self.payload = b"synthetic source contents"
        self.entry = {"path": "mp4s/helper_trimmed_sync.mp4", "size": len(self.payload),
                      "hash": hashlib.sha256(self.payload).hexdigest()}
        self.manifest = {"url_id": "example", "hash_algorithm": "sha256"}

    def test_same_size_corrupt_file_is_replaced_after_hash_check(self):
        self.target.write_bytes(b"x" * len(self.payload))
        original = source.download_file
        def fetch(url, path, *args, **kwargs):
            if Path(path).exists():
                return original(url, path, *args, **kwargs)
            Path(path).write_bytes(self.payload)
        with mock.patch.object(source, "download_file", side_effect=fetch) as download:
            data.download_verified(self.manifest, self.entry, self.target, "https://example.invalid")
        self.assertEqual(self.target.read_bytes(), self.payload)
        self.assertEqual(download.call_count, 2)

    def test_complete_partial_is_promoted_without_network(self):
        Path(str(self.target) + ".part").write_bytes(self.payload)
        with mock.patch.object(source.urllib.request, "urlopen") as request:
            data.download_verified(self.manifest, self.entry, self.target, "https://example.invalid")
        request.assert_not_called()
        self.assertEqual(self.target.read_bytes(), self.payload)


if __name__ == "__main__":
    unittest.main()
