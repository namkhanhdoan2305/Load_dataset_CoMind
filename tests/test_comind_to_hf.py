"""Synthetic integrity and interruption checks. No external uploads or data."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import comind_to_hf as bridge


class FakeHub:
    def __init__(self):
        self.files = {}
        self.corrupt_upload = False
        self.interrupt_after_commit = False

    def get_paths_info(self, paths, **kwargs):
        return [SimpleNamespace(path=p, size=len(self.files[p]),
                                lfs=SimpleNamespace(sha256=hashlib.sha256(self.files[p]).hexdigest()))
                for p in paths if p in self.files]

    def upload_file(self, path_in_repo, path_or_fileobj, **kwargs):
        payload = Path(path_or_fileobj).read_bytes()
        if self.corrupt_upload:
            payload = b"x" * len(payload)
        self.files[path_in_repo] = payload
        if self.interrupt_after_commit:
            raise ConnectionError("Simulated dropped response after successful commit")
        return SimpleNamespace(oid="verified-commit")


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.staging = Path(self.temp.name)
        self.payload = b"synthetic MP4 contents"
        self.entry = {"path": "mp4s/helper_trimmed_sync.mp4", "size": len(self.payload),
                      "hash": hashlib.sha256(self.payload).hexdigest()}
        self.man = {"url_id": "example", "dest_subpath": "recordings/example",
                    "hash_algorithm": "sha256", "files": [self.entry], "raw": b"{}"}
        self.remote_path = "recordings/example/" + self.entry["path"]
        self.hub = FakeHub()

    def fake_download(self, url, dest, expected_size, callback, **kwargs):
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(self.payload)

    def transfer(self):
        bridge.transfer_file(self.hub, "user/dataset", self.man, self.entry,
                             self.staging, "https://example.invalid")

    def test_success_verifies_and_removes_temporary_payload(self):
        with mock.patch.object(bridge.source, "download_file", self.fake_download):
            self.transfer()
        self.assertEqual(self.hub.files[self.remote_path], self.payload)
        self.assertEqual(list(self.staging.iterdir()), [])

    def test_same_size_corrupt_remote_is_not_treated_as_complete(self):
        self.hub.files[self.remote_path] = b"x" * len(self.payload)
        with mock.patch.object(bridge.source, "download_file", self.fake_download):
            self.transfer()
        self.assertEqual(self.hub.files[self.remote_path], self.payload)

    def test_bad_uploaded_hash_keeps_local_payload(self):
        self.hub.corrupt_upload = True
        with mock.patch.object(bridge.source, "download_file", self.fake_download):
            with self.assertRaisesRegex(RuntimeError, "Hub hash verification failed"):
                self.transfer()
        self.assertEqual(len(list(self.staging.iterdir())), 1)

    def test_lost_commit_response_then_restart_skips_download(self):
        self.hub.interrupt_after_commit = True
        with mock.patch.object(bridge.source, "download_file", self.fake_download), \
                mock.patch.object(bridge, "retry", lambda operation: operation()):
            with self.assertRaises(ConnectionError):
                self.transfer()
        self.assertEqual(len(list(self.staging.iterdir())), 1)
        with mock.patch.object(bridge.source, "download_file") as download:
            self.transfer()
        download.assert_not_called()
        self.assertEqual(list(self.staging.iterdir()), [])

    def test_directory_resume_repairs_same_size_corruption(self):
        dest = self.staging / self.remote_path
        dest.parent.mkdir(parents=True)
        dest.write_bytes(b"x" * len(self.payload))
        with mock.patch.object(bridge.source, "download_file", wraps=bridge.source.download_file) as download:
            # First call detects existing bad bytes; second call replaces them.
            original = download._mock_wraps
            def fetch(url, path, *args, **kwargs):
                if Path(path).exists():
                    return original(url, path, *args, **kwargs)
                return self.fake_download(url, path, *args, **kwargs)
            download.side_effect = fetch
            bridge.save_to_directory([self.man], self.staging, "https://example.invalid")
        self.assertEqual(dest.read_bytes(), self.payload)
        self.assertEqual(download.call_count, 2)
        selection = json.loads((dest.parents[1] / "selection.json").read_text())
        self.assertEqual(selection["files"], [self.entry])

    def test_complete_partial_is_promoted_without_network(self):
        dest = self.staging / "payload"
        Path(str(dest) + ".part").write_bytes(self.payload)
        with mock.patch.object(bridge.source.urllib.request, "urlopen") as request:
            bridge.download_verified(self.man, self.entry, dest, "https://example.invalid")
        request.assert_not_called()
        self.assertEqual(dest.read_bytes(), self.payload)


if __name__ == "__main__":
    unittest.main()
