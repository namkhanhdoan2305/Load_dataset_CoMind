"""Synthetic graph-stream checks. These do not validate GPU model predictions."""

import copy
from contextlib import ExitStack, redirect_stdout
import gzip
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import sgg_pipeline as pipeline


class FakeVideo:
    def __init__(self, path, tracking_fps):
        self.fps, self.frames = 30, 243
        self.tracking_fps = tracking_fps
        self.samples = (self.frames - 1) * tracking_fps // self.fps + 1

    def decode(self, start, end, folder, max_side):
        return [{"sample_index": i, "source_frame_index": i * self.fps // self.tracking_fps,
                 "time_s": i / self.tracking_fps,
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

    def test_unmounted_drive_stops_before_model_or_network(self):
        with mock.patch.object(pipeline.os.path, "ismount", return_value=False), \
                mock.patch.object(pipeline.download, "make_plan") as plan:
            with self.assertRaises(SystemExit):
                pipeline.main([])
        plan.assert_not_called()

    def test_full_batch_reaches_last_recording_after_one_fails(self):
        ids = pipeline.select_sessions(55, 42)
        failed, unresolved = ids[4], ids[6]
        paths = [pipeline.download.GROUPS[role][0] for role in ("leader_video", "helper_video")]
        plan = [{"url_id": rid, "dest_subpath": f"recordings/{rid}", "raw": b"{}",
                 "hash_algorithm": "sha256", "files": [{"path": k, "size": 100, "hash": k} for k in paths]}
                for rid in ids]
        budget_stop = None
        def extract(video, folder, settings, digest, view, model, deadline):
            rid = folder.parents[1].name
            if rid == failed:
                raise ValueError("recording decode error")
            return {"complete": rid != budget_stop}
        def export(recording):
            return {"all_modalities_aligned": recording.name != unresolved,
                    "timed_model_inputs_aligned": recording.name != unresolved}
        target = self.root / "output"
        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(mock.patch.dict("sys.modules", {"onnxruntime": SimpleNamespace(
                get_available_providers=lambda: ["CUDAExecutionProvider"])}))
            stack.enter_context(mock.patch.object(pipeline.download, "make_plan", return_value=plan))
            stack.enter_context(mock.patch.object(pipeline.download, "download_verified"))
            stack.enter_context(mock.patch.object(pipeline.source, "free_space_bytes", return_value=2**40))
            stack.enter_context(mock.patch("reactpp_models.Perception", return_value=SimpleNamespace(info={})))
            stack.enter_context(mock.patch("sensor_sync.prepare_clock", return_value={}))
            stack.enter_context(mock.patch.object(pipeline, "extract_view", side_effect=extract))
            stack.enter_context(mock.patch.object(pipeline, "link_graph_views"))
            stack.enter_context(mock.patch("belief_data.export_recording", side_effect=export))
            args = ["--sessions", "55", "--target", str(target), "--work-dir", str(self.root / "work")]
            budget_stop = ids[8]
            pipeline.main(args)
            saved = json.loads((target / "sgg-status.json").read_text())
            self.assertEqual(saved["state"], "budget_reached")
            self.assertEqual(saved["pending"], ids[8:])
            self.assertEqual(saved["completed"], [rid for rid in ids[:8] if rid != failed])
            budget_stop = None
            pipeline.main(args)
        status = json.loads((target / "sgg-status.json").read_text())
        self.assertEqual(status["recordings"], ids)
        self.assertEqual(status["completed"], [rid for rid in ids if rid != failed])
        self.assertEqual(status["state"], "complete_with_issues")
        self.assertEqual(status["pending"], [])
        self.assertEqual(status["failed"][failed]["error"], "recording decode error")
        self.assertEqual(status["failed"][failed]["stage"], "graph")
        self.assertIn(unresolved, status["needs_review"])

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

    def test_missing_graph_sample_leaves_previous_chunk_available_for_resume(self):
        model = FakePerception()
        def drop_sample(*args):
            rows, carry, next_id = original(*args)
            return [row for row in rows if row["sample_index"] != 25], carry, next_id
        original = model.chunk
        with mock.patch.object(model, "chunk", side_effect=drop_sample):
            with self.assertRaisesRegex(ValueError, "chunk 2: expected graph samples"):
                self.extract("session_one", "leader", model)
        folder = self.root / "recordings/session_one/graphs/leader"
        status = json.loads((folder / "progress.json").read_text())
        self.assertFalse(status["complete"])
        self.assertEqual(status["graph_count"], 21)
        model = FakePerception()
        resumed = self.extract("session_one", "leader", model)
        self.assertEqual(model.calls, 1)
        self.assertEqual(resumed["graph_count"], resumed["expected_graph_count"])
        self.assertEqual(resumed["graph_count"], 41)

    def test_saved_pair_with_a_shared_missing_step_is_rejected(self):
        rid = "session_one"
        for view in ("leader", "helper"):
            self.extract(rid, view, FakePerception())
            path = self.root / "recordings" / rid / "graphs" / view / "graphs.jsonl.gz"
            with gzip.open(path, "rt") as f:
                rows = f.readlines()
            with gzip.open(path, "wt") as f:
                f.writelines(rows[:1] + rows[2:])
        with self.assertRaisesRegex(ValueError, "Missing graph grid step 1"):
            pipeline.link_graph_views(self.root, rid)

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
            "segments": [{"words": [{"word": "drawer", "start": 0.1, "end": 0.3},
                                      {"word": "untimed", "start": None, "end": None}]}]}))
        pipeline.write_sync_report(self.root, "session_one")
        pipeline.link_graph_views(self.root, "session_one")
        with gzip.open(folder.parent / "graphs/timeline.jsonl.gz", "rt") as f:
            rows = [json.loads(line) for line in f]
        self.assertEqual(rows[1]["completed_transcript_words"], [])
        self.assertEqual(rows[2]["completed_transcript_words"][0]["word"], "drawer")
        self.assertEqual(rows[2]["completed_transcript_words"][0]["audio_stream"], "leader")
        report = json.loads((folder.parent / "sync-status.json").read_text())
        self.assertEqual(report["transcript_words"]["untimed_words"], 1)

    def test_sensor_clock_is_reported_unresolved_without_guessing_zero_offset(self):
        folder = self.root / "recordings/session_one/leader_gaze"
        folder.mkdir(parents=True)
        (folder / "general_eye_gaze.csv").write_text("tracking_timestamp_us,yaw\n998505868,0.1\n")
        pipeline.write_sync_report(self.root, "session_one")
        report = json.loads((folder.parent / "sync-status.json").read_text())
        self.assertFalse(report["all_modalities_aligned"])
        self.assertEqual(report["sensor_clocks"]["leader_gaze"]["mapping_to_mp4"], "unresolved")



if __name__ == "__main__":
    unittest.main()
