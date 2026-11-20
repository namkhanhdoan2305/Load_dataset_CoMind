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
        rows = list(belief_data.iter_training_steps(self.recording))
        self.assertEqual([len(labels) for row, labels in rows], [0, 0, 1, 0])
        self.assertEqual(rows[2][1][0]["allowed_location_ids"], ["B", "C"])
        self.assertNotIn("allowed_location_ids", rows[2][0])

    def test_off_grid_label_rejected_before_first_training_step(self):
        self.labels(timestamp=0.35)
        belief_data.export_recording(self.recording)
        with self.assertRaisesRegex(ValueError, "graph grid"):
            next(belief_data.iter_training_steps(self.recording))

    def test_speech_ids_are_stable_on_reexport(self):
        a = belief_data.speech_events(self.recording)
        b = belief_data.speech_events(self.recording)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
