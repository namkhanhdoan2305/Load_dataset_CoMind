"""Read-only audit distinguishes deleted raw video from missing/corrupt source data."""

import json
from pathlib import Path
import unittest

import comind_download as source
from comind_to_hf import GROUPS, ANNOTATIONS, split_annotations
from belief_data import export_recording
from review_recording import review
import test_sensor_sync


class ReviewTests(unittest.TestCase):
    def setUp(self):
        fixture = test_sensor_sync.ExportIntegrationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        # Move the fixture into the production root/recordings/<ID> layout.
        self.root = fixture.recording.parent
        self.recording = self.root / "recordings" / fixture.recording.name
        self.recording.parent.mkdir()
        fixture.recording.rename(self.recording)
        rid = self.recording.name
        entries = []
        for group, paths in GROUPS.items():
            for name in paths:
                file = self.recording / group / Path(name).name
                entries.append({"path": name, "size": file.stat().st_size if file.exists() else 123,
                                "hash": source.file_digest(file) if file.exists() else "video-sha"})
        meta = self.root / "metadata" / rid
        meta.mkdir(parents=True)
        (meta / "source-healthcheck.json").write_text(json.dumps({"files": entries}))
        common = self.root / "annotations"
        common.mkdir()
        common_entries = []
        for name in ANNOTATIONS:
            file = common / name
            file.write_text(json.dumps({"data": {rid: {"event": {"start_time": 0.1, "end_time": 0.2}}}}))
            common_entries.append({"path": name, "size": file.stat().st_size, "hash": source.file_digest(file)})
        (common / "source-healthcheck.json").write_text(json.dumps({"files": common_entries}))
        split_annotations([{"url_id": rid}], self.root)
        export_recording(self.recording)

    def test_completed_graph_can_replace_retention_of_raw_video_in_audit(self):
        report = review(self.recording, work_dir=self.root / "no-work", source_dir=self.root / "no-source")
        self.assertEqual(report["errors"], [])
        status = report["source_files"]["mp4s/leader_trimmed_sync.mp4"]["status"]
        self.assertEqual(status, "graph_provenance_verified_raw_not_retained")
        self.assertFalse(report["physical_audio_video_alignment_verified"])

    def test_tampered_transcript_is_reported(self):
        path = self.recording / "transcripts/helper_trimmed_sync_transcript.json"
        path.write_text(path.read_text() + " ")
        report = review(self.recording, work_dir=self.root / "no-work", source_dir=self.root / "no-source")
        self.assertTrue(any("Source file hash mismatch" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
