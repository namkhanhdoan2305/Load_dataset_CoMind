"""Synthetic graph-stream checks. These do not validate GPU model predictions."""

import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import sgg_pipeline as pipeline
from sgg_models import match_detections


class FakeVideo:
    def __init__(self, path, tracking_fps):
        self.samples, self.fps, self.frames = 81, 30, 243

    def decode(self, start, end, folder, max_side):
        return [{"sample_index": i, "source_frame_index": i * 3, "time_s": i / 10,
                 "source_width": 1408, "source_height": 1408} for i in range(start, end + 1)]

    def close(self):
        pass


class FakePerception:
    def __init__(self):
        self.calls = 0

    def chunk(self, folder, times, view, carry, next_id, stride):
        self.calls += 1
        rows = []
        for sample in times:
            if sample["sample_index"] % stride == 0:
                rows.append(dict(sample, view=view,
                                 nodes=[{"id": f"{view}:1"}, {"id": f"{view}:2"}],
                                 edges=[{"subject": f"{view}:1", "object": f"{view}:2",
                                         "relation": "on", "score": 0.8}]))
        return rows, [{"track_id": 1}], 3


class GraphStreamTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = copy.deepcopy(pipeline.DEFAULT_SETTINGS)

    def extract(self, rid, view, model, deadline=float("inf")):
        folder = self.root / "recordings" / rid / "graphs" / view
        return pipeline.extract_view("unused.mp4", folder, self.settings, "source-sha256",
                                     view, model, deadline, reader_class=FakeVideo)

    def test_two_sessions_two_views_have_exact_5fps_and_timestamp_links(self):
        model = FakePerception()
        for rid in ("session_one", "session_two"):
            for view in ("leader", "helper"):
                status = self.extract(rid, view, model)
                self.assertTrue(status["complete"])
                self.assertEqual(status["graph_count"], 41)
                with gzip.open(self.root / f"recordings/{rid}/graphs/{view}/graphs.jsonl.gz", "rt") as f:
                    rows = [json.loads(line) for line in f]
                self.assertEqual([r["source_frame_index"] for r in rows], list(range(0, 241, 6)))
                self.assertEqual([r["time_s"] for r in rows], [i / 5 for i in range(41)])
                self.assertTrue(all(r["recording_id"] == rid for r in rows))
            self.assertEqual(pipeline.link_graph_views(self.root, rid), 41)
        # Complete runs are verified and skipped without invoking perception.
        calls = model.calls
        self.extract("session_one", "leader", model)
        self.assertEqual(model.calls, calls)

    def test_uncommitted_append_recovers_without_duplicate_times(self):
        folder = self.root / "graphs"
        progress = pipeline.recover_stream(folder, "identity")
        row = {"time_s": 0.0, "nodes": [{"id": "a"}], "edges": []}
        progress = pipeline.commit_chunk(folder, progress, [row], [], 2)
        committed = progress["committed_bytes"]
        with (folder / "graphs.jsonl.gz").open("ab") as f:
            f.write(b"interrupted gzip append")
        recovered = pipeline.recover_stream(folder, "identity")
        self.assertEqual((folder / "graphs.jsonl.gz").stat().st_size, committed)
        updated = pipeline.commit_chunk(folder, recovered,
                                         [row, dict(row, time_s=0.2)], [], 2)
        self.assertEqual(updated["graph_count"], 2)

    def test_corrupt_committed_data_and_settings_change_are_rejected(self):
        folder = self.root / "graphs"
        state = pipeline.recover_stream(folder, "identity")
        pipeline.commit_chunk(folder, state, [{"time_s": 0, "nodes": [], "edges": []}], [], 1)
        with self.assertRaisesRegex(ValueError, "configuration/source changed"):
            pipeline.recover_stream(folder, "different_identity")
        with (folder / "graphs.jsonl.gz").open("r+b") as f:
            f.write(b"bad")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            pipeline.recover_stream(folder, "identity")

    def test_expired_budget_returns_incomplete_without_running_model(self):
        model = FakePerception()
        status = self.extract("session_one", "leader", model, deadline=0)
        self.assertFalse(status["complete"])
        self.assertEqual(model.calls, 0)

    def test_pairing_rejects_different_timestamps(self):
        self.extract("session_one", "leader", FakePerception())
        folder = self.root / "recordings/session_one/graphs/helper"
        folder.mkdir()
        with gzip.open(folder / "graphs.jsonl.gz", "wt") as f:
            f.write(json.dumps({"time_s": 0.1, "sample_index": 0}) + "\n")
        with self.assertRaisesRegex(ValueError, "timestamps disagree"):
            pipeline.link_graph_views(self.root, "session_one")

    def test_timeline_includes_speech_only_after_source_word_end_time(self):
        for view in ("leader", "helper"):
            self.extract("session_one", view, FakePerception())
        folder = self.root / "recordings/session_one/transcripts"
        folder.mkdir()
        (folder / "leader_trimmed_sync_transcript.json").write_text(json.dumps({
            "segments": [{"words": [{"word": "drawer", "start": 0.1, "end": 0.3}]}]}))
        pipeline.link_graph_views(self.root, "session_one")
        with gzip.open(folder.parent / "graphs/timeline.jsonl.gz", "rt") as f:
            rows = [json.loads(line) for line in f]
        self.assertEqual(rows[1]["completed_transcript_words"], [])
        self.assertEqual(rows[2]["completed_transcript_words"][0]["word"], "drawer")
        self.assertEqual(rows[2]["completed_transcript_words"][0]["audio_stream"], "leader")

    def test_sensor_clock_is_reported_unresolved_without_guessing_zero_offset(self):
        folder = self.root / "recordings/session_one/leader_gaze"
        folder.mkdir(parents=True)
        (folder / "general_eye_gaze.csv").write_text("tracking_timestamp_us,yaw\n998505868,0.1\n")
        pipeline.write_sync_report(self.root, "session_one")
        report = json.loads((folder.parent / "sync-status.json").read_text())
        self.assertFalse(report["all_modalities_aligned"])
        self.assertEqual(report["sensor_clocks"]["leader_gaze"]["mapping_to_mp4"], "unresolved")

    def test_tracking_match_is_one_to_one_for_same_class_objects(self):
        tracks = [{"track_id": 1, "label": "bowl", "box": [0, 0, 10, 10]}]
        detections = [{"label": "bowl", "box": [0, 0, 10, 10]},
                      {"label": "bowl", "box": [1, 1, 10, 10]}]
        assigned, next_id = match_detections(detections, tracks, 2, maximum=20)
        self.assertEqual({a["track_id"] for a in assigned}, {1, 2})
        self.assertEqual(next_id, 3)


if __name__ == "__main__":
    unittest.main()
