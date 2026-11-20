"""Causal text events, stable sample pairing and supervised-label requirements."""

import gzip
import json
from pathlib import Path
import tempfile
import unittest

import belief_data
import comind_download as source


class BeliefDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.recording = Path(self.temp.name) / "session_one"
        for view in ("leader", "helper"):
            folder = self.recording / "graphs" / view
            folder.mkdir(parents=True)
            graph = folder / "graphs.jsonl.gz"
            with gzip.open(graph, "wt") as handle:
                for i in range(4):
                    handle.write(json.dumps({"time_s": i / 5, "nodes": [], "edges": []}) + "\n")
            (folder / "progress.json").write_text(json.dumps({
                "complete": True, "sha256": source.file_digest(graph)}))
        transcripts = self.recording / "transcripts"
        transcripts.mkdir()
        (transcripts / "leader_trimmed_sync_transcript.json").write_text(json.dumps({
            "segments": [{"start": 0.1, "end": 0.3, "text": "in the drawer", "words": []}]}))

    def labels(self, timestamp=0.4):
        folder = self.recording / "labels"
        folder.mkdir(exist_ok=True)
        (folder / "belief_labels.jsonl").write_text(json.dumps({
            "time_s": timestamp, "person_id": "leader", "object_id": "helper:1",
            "candidate_location_ids": ["A", "B", "C"], "allowed_location_ids": ["B", "C"],
            "label_source": "independent human annotation"}) + "\n")

    def test_speech_segment_emits_once_after_completion_and_context_keeps_reference(self):
        manifest = belief_data.export_recording(self.recording)
        with gzip.open(self.recording / "belief_inputs/steps.jsonl.gz", "rt") as handle:
            rows = [json.loads(line) for line in handle]
        self.assertEqual([len(row["new_speech_events"]) for row in rows], [0, 0, 1, 0])
        self.assertEqual(rows[2]["recent_speech_event_ids"], rows[3]["recent_speech_event_ids"])
        self.assertIsNone(rows[2]["gaze_features"])
        self.assertEqual(manifest["label_count"], 0)
        self.assertFalse((self.recording / "labels/belief_labels.jsonl").exists())

    def test_supervised_loader_requires_real_labels(self):
        belief_data.export_recording(self.recording)
        with self.assertRaisesRegex(ValueError, "Supply verified belief labels"):
            next(belief_data.iter_training_steps(self.recording))

    def test_loader_yields_all_steps_but_labels_only_at_labeled_time(self):
        self.labels()
        belief_data.export_recording(self.recording)
        rows = list(belief_data.iter_training_steps(self.recording, require_sensor_alignment=False))
        self.assertEqual([len(labels) for row, labels in rows], [0, 0, 1, 0])
        self.assertEqual(rows[2][1][0]["allowed_location_ids"], ["B", "C"])
        self.assertNotIn("allowed_location_ids", rows[2][0])

    def test_off_grid_label_rejected_before_first_training_step(self):
        self.labels(timestamp=0.35)
        belief_data.export_recording(self.recording)
        with self.assertRaisesRegex(ValueError, "graph grid"):
            next(belief_data.iter_training_steps(self.recording, require_sensor_alignment=False))

    def test_speech_ids_are_stable_on_reexport(self):
        a = belief_data.speech_events(self.recording)
        b = belief_data.speech_events(self.recording)
        self.assertEqual(a, b)

    def test_multimodal_training_rejects_unresolved_sensor_clock_by_default(self):
        self.labels()
        belief_data.export_recording(self.recording)
        with self.assertRaisesRegex(ValueError, "clock alignment is unresolved"):
            next(belief_data.iter_training_steps(self.recording))

    def test_export_stamps_same_clock_and_integer_microseconds(self):
        belief_data.export_recording(self.recording)
        with gzip.open(self.recording / "belief_inputs/steps.jsonl.gz", "rt") as handle:
            rows = [json.loads(line) for line in handle]
        self.assertEqual([row["time_us"] for row in rows], [0, 200000, 400000, 600000])
        self.assertEqual(rows[2]["clock_id"], rows[2]["new_speech_events"][0]["clock_id"])
        self.assertEqual(rows[2]["new_speech_events"][0]["available_us"], 300000)

    def test_stride_aggregates_intervening_speech_without_overwriting_base(self):
        base = belief_data.export_recording(self.recording)
        coarse = belief_data.export_recording(self.recording, step_stride=3)
        folder = self.recording / "belief_inputs_stride_3"
        with gzip.open(folder / "steps.jsonl.gz", "rt") as handle:
            rows = [json.loads(line) for line in handle]
        self.assertEqual([r["time_us"] for r in rows], [0, 600000])
        self.assertEqual([len(r["new_speech_events"]) for r in rows], [0, 1])
        self.assertEqual(rows[1]["new_speech_events"][0]["available_us"], 300000)
        self.assertEqual(rows[1]["source_step_index"], 3)
        self.assertEqual(base["source_graph_sha256"], coarse["source_graph_sha256"])
        self.assertEqual(source.file_digest(self.recording / "belief_inputs/steps.jsonl.gz"), base["steps_sha256"])
        audit = json.loads((folder / "speech_alignment.json").read_text())
        self.assertEqual(audit["emissions"][0]["step_delay_us"], 300000)
        self.assertEqual(coarse["step_period_us"], 600000)

    def test_context_expiry_does_not_drop_new_events_on_coarse_steps(self):
        belief_data.export_recording(self.recording, context_seconds=0, step_stride=3)
        with gzip.open(self.recording / "belief_inputs_stride_3/steps.jsonl.gz", "rt") as handle:
            rows = [json.loads(line) for line in handle]
        self.assertEqual(len(rows[1]["new_speech_events"]), 1)
        self.assertEqual(rows[1]["recent_speech_event_ids"], [])

    def test_tail_speech_is_reported_without_future_emission(self):
        path = self.recording / "transcripts/leader_trimmed_sync_transcript.json"
        path.write_text(json.dumps({"segments": [{"start": 0.3, "end": 0.7, "text": "future"}]}))
        manifest = belief_data.export_recording(self.recording)
        self.assertEqual(manifest["speech_events_emitted"], 0)
        self.assertEqual(manifest["speech_events_unemitted"], 1)
        audit = json.loads((self.recording / "belief_inputs/speech_alignment.json").read_text())
        self.assertEqual(audit["unemitted_events"][0]["available_us"], 700000)
        self.assertFalse(audit["physical_audio_video_alignment_verified"])

    def test_missing_segment_timestamps_fail_instead_of_losing_text(self):
        path = self.recording / "transcripts/leader_trimmed_sync_transcript.json"
        path.write_text(json.dumps({"segments": [{"text": "untimed text"}]}))
        with self.assertRaisesRegex(ValueError, "Missing speech segment timestamps"):
            belief_data.export_recording(self.recording)

    def test_word_stream_is_available_before_long_segment_completes(self):
        path = self.recording / "transcripts/leader_trimmed_sync_transcript.json"
        path.write_text(json.dumps({"segments": [{"start": 0.1, "end": 10, "text": "left drawer",
            "words": [{"start": 0.1, "end": 0.15, "word": "left"},
                      {"start": 0.35, "end": 0.45, "word": "drawer"},
                      {"start": 0.35, "end": 0.45, "word": "now"}]}]}))
        manifest = belief_data.export_recording(self.recording)
        with gzip.open(self.recording / "belief_inputs/steps.jsonl.gz", "rt") as handle:
            rows = [json.loads(line) for line in handle]
        self.assertEqual([[w["text"] for w in row["new_speech_words"]] for row in rows],
                         [[], ["left"], [], ["drawer", "now"]])
        self.assertTrue(all(not row["new_speech_events"] for row in rows))
        self.assertEqual(manifest["speech_words_emitted"], 3)

    def test_wrong_recording_transcript_is_rejected_by_source_hash(self):
        (self.recording / "sensor-clock.json").write_text(json.dumps({
            "source_transcript_sha256": {"leader": "wrong-source-hash"}}))
        with self.assertRaisesRegex(ValueError, "Transcript missing or hash differs"):
            belief_data.speech_events(self.recording)

    def test_invalid_stride_is_rejected(self):
        for stride in (0, -1, 1.5, True):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                belief_data.export_recording(self.recording, step_stride=stride)

    def test_coarse_labels_must_match_selected_grid(self):
        self.labels(timestamp=0.4)
        belief_data.export_recording(self.recording, step_stride=3)
        with self.assertRaisesRegex(ValueError, "graph grid"):
            next(belief_data.iter_training_steps(self.recording, require_sensor_alignment=False, step_stride=3))

    def test_nonuniform_base_grid_fails_for_stride_comparisons(self):
        for view in ("leader", "helper"):
            folder = self.recording / "graphs" / view
            path = folder / "graphs.jsonl.gz"
            with gzip.open(path, "wt") as handle:
                for timestamp in (0, 0.2, 0.5):
                    handle.write(json.dumps({"time_s": timestamp}) + "\n")
            (folder / "progress.json").write_text(json.dumps({
                "complete": True, "sha256": source.file_digest(path)}))
        with self.assertRaisesRegex(ValueError, "uniform"):
            belief_data.export_recording(self.recording, step_stride=3)


if __name__ == "__main__":
    unittest.main()
