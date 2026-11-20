"""Explicit label timestamps, short events between graph samples, and no target leakage."""

import gzip
import json
from pathlib import Path
import tempfile
import unittest

from annotation_timing import AnnotationTimeline, KINDS
from belief_data import export_recording
import test_sensor_sync


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.recording = Path(self.temp.name) / "session_one"
        folder = self.recording / "annotations"
        folder.mkdir(parents=True)
        for name in KINDS.values():
            (folder / name).write_text(json.dumps({"recording_id": self.recording.name,
                                                 "recording_present_in_source": True, "data": {}}))

    def put(self, data):
        path = self.recording / "annotations/joint_attention.json"
        path.write_text(json.dumps({"recording_id": self.recording.name, "data": data}))

    def test_explicit_times_override_numeric_event_id(self):
        self.put({"999999": {"start_time": 0.1, "end_time": 0.2, "start_frame": 3, "end_frame": 6}})
        timeline = AnnotationTimeline(self.recording, 30, 1000000)
        self.assertEqual(timeline.events[0]["start_us"], 100000)
        self.assertTrue(timeline.report()["temporal_mapping_complete"])

    def test_short_event_is_preserved_even_without_active_graph_sample(self):
        self.put({"event": {"start_time": 0.05, "end_time": 0.08}})
        timeline = AnnotationTimeline(self.recording, 30, 1000000)
        timeline.advance(0)
        row = timeline.advance(200000)
        self.assertEqual(row["started_event_ids"], ["joint_attention:event"])
        self.assertEqual(row["completed_event_ids"], ["joint_attention:event"])
        self.assertEqual(row["active_event_ids"], [])
        self.assertEqual(row["interval_overlap_event_ids"], ["joint_attention:event"])

    def test_frame_time_conflicts_and_null_labels_are_not_silently_aligned(self):
        self.put({"a": {"start_time": 0, "end_time": 0.2, "start_frame": 30}, "withheld": None})
        report = AnnotationTimeline(self.recording, 30, 1000000).report()
        self.assertFalse(report["temporal_mapping_complete"])
        self.assertEqual(report["untimed_source_entries"], 1)
        self.assertEqual(report["issues"][0]["reason"], "start_frame_time_mismatch")

    def test_benchmark_targets_remain_separate_from_encoder_inputs(self):
        fixture = test_sensor_sync.ExportIntegrationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        folder = fixture.recording / "annotations"
        folder.mkdir()
        for filename in KINDS.values():
            (folder / filename).write_text(json.dumps({"recording_id": fixture.recording.name,
                "data": {"target": {"start_time": 0.1, "end_time": 0.2,
                                    "transcript_10s": "future target text"}}}))
        manifest = export_recording(fixture.recording)
        self.assertTrue(manifest["all_modalities_aligned"])
        with gzip.open(fixture.recording / "belief_inputs/steps.jsonl.gz", "rt") as handle:
            self.assertNotIn("future target text", handle.read())
        self.assertTrue((folder / "timed_events.json").exists())
        self.assertEqual(manifest["benchmark_annotation_timing"]["timed_events"], 3)


if __name__ == "__main__":
    unittest.main()
