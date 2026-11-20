"""Regression tests for excess MPS variants and per-recording grouping."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import comind_download as source
import comind_to_hf as downloader
import load_comind_drive as preset


class ExactSelectionTests(unittest.TestCase):
    def setUp(self):
        self.rid = source.TRAIN_IDS[0]
        self.wanted = [p for paths in downloader.GROUPS.values() for p in paths]
        self.extra = [
            "mp4s/leader_helper_trimmed_sync.mp4", "mp4s/leader.mp4",
            "mps_leader_gapfill_vrs/eye_gaze/general_eye_gaze.csv",
            "mps_leader_gapfill_vrs/hand_tracking/hand_tracking_results.csv",
            "mps_helper_trimmed_vrs/hand_tracking/hand_tracking_results.csv",
            "mps_leader_trimmed_vrs/eye_gaze/summary.json",
            "mps_leader_trimmed_vrs/eye_gaze/personalized_eye_gaze.csv",
            "mps_leader_trimmed_vrs/hand_tracking/summary.json",
            "mps_leader_trimmed_vrs/slam/closed_loop_trajectory.csv",
            "mps_leader_trimmed_vrs/slam/online_calibration.jsonl",
            "transcripts/leader_trimmed_sync_transcript.ass",
            "transcripts/leader_trimmed_sync_transcripts_squared.ass",
            "transcripts/leader_trimmed_sync_transcript_with_profanity.json",
            "trimmed_vrs/leader_trimmed.vrs", "gopro_front/video.mp4", "scan/scene.pcd",
        ]

    def manifest(self, paths, rid):
        return {"recording_id": rid, "url_id": rid,
                "dest_subpath": "annotations" if rid == "annotations" else f"recordings/{rid}",
                "hash_algorithm": "sha256", "raw": b"{}",
                "files": [{"path": p, "size": 10, "hash": "x"} for p in paths],
                "total_size_bytes": 10 * len(paths)}

    def test_exactly_six_payload_files_and_three_common_annotations(self):
        recording = self.manifest(self.wanted + self.extra, self.rid)
        common = self.manifest(downloader.ANNOTATIONS + ["unrelated.json"], "annotations")
        def fetch(base_url, rid, timeout, **kwargs):
            return copy.deepcopy(common if rid == "annotations" else recording)
        with mock.patch.object(source, "fetch_manifest", side_effect=fetch):
            plan = downloader.make_plan([self.rid])
        self.assertEqual([e["path"] for e in plan[1]["files"]], self.wanted)
        self.assertEqual(len(plan[1]["files"]), 6)
        self.assertEqual([e["path"] for e in plan[0]["files"]], downloader.ANNOTATIONS)

    def test_gapfill_does_not_satisfy_missing_trimmed_hands(self):
        hands = downloader.GROUPS["leader_hands"][0]
        recording = self.manifest([p for p in self.wanted if p != hands] + self.extra, self.rid)
        common = self.manifest(downloader.ANNOTATIONS, "annotations")
        with mock.patch.object(source, "fetch_manifest", side_effect=[recording, common]):
            plan = downloader.make_plan([self.rid], allow_missing=True)
        self.assertIn("leader_hands", plan[1]["missing_modalities"])
        self.assertFalse(any("gapfill" in e["path"] for e in plan[1]["files"]))


class GroupedLayoutTests(unittest.TestCase):
    def test_reorganization_recovers_after_copy_committed_but_old_file_remained(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old, dest = root / "old.mp4", root / "new.mp4"
            old.write_bytes(b"video")
            dest.write_bytes(b"video")
            entry = {"size": 5, "hash": hashlib.sha256(b"video").hexdigest()}
            downloader.adopt_existing(old, dest, entry, "sha256")
            self.assertFalse(old.exists())
            self.assertEqual(dest.read_bytes(), b"video")

    def test_grouped_download_has_six_groups_and_separate_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rid = source.TRAIN_IDS[0]
            paths = [p for values in downloader.GROUPS.values() for p in values]
            def manifest(url_id, entries):
                return {"url_id": url_id, "dest_subpath": f"recordings/{url_id}",
                        "raw": b"{}", "hash_algorithm": "sha256",
                        "files": [{"path": p, "size": 1, "hash": None} for p in entries]}
            annotations = manifest("annotations", downloader.ANNOTATIONS)
            annotations["dest_subpath"] = "annotations"
            recording = manifest(rid, paths)
            def fake_download(man, entry, dest, base_url):
                dest.parent.mkdir(parents=True, exist_ok=True)
                if man["url_id"] == "annotations":
                    dest.write_text(json.dumps({"data": {rid: {"event": {"value": "correct"}}}}))
                else:
                    dest.write_text(entry["path"])
            with mock.patch.object(downloader, "download_verified", side_effect=fake_download):
                result = downloader.save_to_directory([annotations, recording], root, grouped=True)
            folder = root / "recordings" / rid
            self.assertEqual({p.name for p in folder.iterdir()}, set(downloader.GROUPS) | {"annotations"})
            self.assertEqual((folder / "leader_video/leader_trimmed_sync.mp4").read_text(), paths[0])
            self.assertEqual((folder / "helper_video/helper_trimmed_sync.mp4").read_text(), paths[1])
            self.assertTrue((root / "metadata" / rid / "source-healthcheck.json").is_file())
            self.assertEqual(result["state"], "complete")

    def test_full_and_partial_payloads_move_without_redownloading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for suffix, contents in [("", b"whole video"), (".part", b"whole")]:
                previous = root / "old" / ("file" + suffix)
                previous.parent.mkdir(exist_ok=True)
                previous.write_bytes(contents)
                destination = root / "group" / "video.mp4"
                entry = {"size": len(b"whole video"),
                         "hash": hashlib.sha256(b"whole video").hexdigest()}
                downloader.adopt_existing(root / "old/file", destination, entry, "sha256")
                self.assertEqual(Path(str(destination) + suffix).read_bytes(), contents)
                self.assertFalse(previous.exists())

    def test_each_recording_gets_its_own_annotations_including_null_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "annotations").mkdir()
            for filename in downloader.ANNOTATIONS:
                (root / "annotations" / filename).write_text(json.dumps({
                    "dataset_version": "1.0", "data": {
                        "recording_a": {"000141": {"label": "a"}, "000465": None},
                        "recording_b": {"000141": {"label": "b"}}}}))
            downloader.split_annotations([{"url_id": "annotations"}, {"url_id": "recording_a"}], root)
            annotation = json.loads((root / "recordings/recording_a/annotations/joint_attention.json").read_text())
            self.assertEqual(annotation["recording_id"], "recording_a")
            self.assertEqual(annotation["data"], {"000141": {"label": "a"}, "000465": None})

    def test_audit_lists_only_extras_and_does_not_change_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rid = "recording_a"
            path = downloader.GROUPS["leader_video"][0]
            man = {"url_id": rid, "dest_subpath": f"recordings/{rid}",
                   "files": [{"path": path, "size": 3}]}
            wanted = downloader.grouped_destination(root, man, man["files"][0])
            wanted.parent.mkdir(parents=True)
            wanted.write_bytes(b"abc")
            extra = wanted.parent / "unwanted.ass"
            extra.write_text("subtitle")
            extras = preset.audit_existing([man], root)
            self.assertEqual([e["path"] for e in extras], [str(extra)])
            self.assertTrue(extra.exists())


if __name__ == "__main__":
    unittest.main()
