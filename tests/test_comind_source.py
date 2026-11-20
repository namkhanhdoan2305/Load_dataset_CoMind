"""Exercise actual resume/retry byte handling through the retained source downloader."""

import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import comind_source as source


class Response(io.BytesIO):
    status = 206


class SourceDownloadTests(unittest.TestCase):
    def test_http_range_resumes_existing_partial(self):
        payload = b"0123456789abcdef"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "payload.mp4"
            Path(str(path) + ".part").write_bytes(payload[:5])
            with mock.patch.object(source.urllib.request, "urlopen", return_value=Response(payload[5:])) as request:
                source.download_file("https://example.invalid/video", str(path), len(payload), lambda *args: None, 1, 60)
            self.assertEqual(request.call_args.args[0].get_header("Range"), "bytes=5-")
            self.assertEqual(path.read_bytes(), payload)
            self.assertFalse(Path(str(path) + ".part").exists())

    def test_interrupted_response_retries_from_committed_partial_bytes(self):
        payload = b"0123456789abcdef"
        class Interrupted(Response):
            status = 200
            def read(self, size=-1):
                if self.tell():
                    raise ConnectionError("simulated interrupted source transfer")
                return super().read(5)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "payload.mp4"
            with mock.patch.object(source.urllib.request, "urlopen", side_effect=[Interrupted(payload), Response(payload[5:])]) as request, \
                    mock.patch.object(source.time, "sleep"):
                source.download_file("https://example.invalid/video", str(path), len(payload), lambda *args: None, 1, 60)
            self.assertEqual(request.call_args_list[1].args[0].get_header("Range"), "bytes=5-")
            self.assertEqual(path.read_bytes(), payload)


if __name__ == "__main__":
    unittest.main()
