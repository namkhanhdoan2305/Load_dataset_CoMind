"""Metadata origin/scale checks, causal sensor sampling and belief export integration."""

import csv
import gzip
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import belief_data
import comind_source as source
from sensor_sync import (PROFILE, RECORD, RangeReader, SensorAligner, PastSampler,
                         gaze_values, hand_values, parse_moov, prepare_clock, verify_anchors, vrs_rgb_bounds)
from time_contract import clock_id


ORIGIN = 1000000000000


def atom(kind, body):
    return struct.pack(">I4s", len(body) + 8, kind) + body


def moov(description, timing=((301, 1000),)):
    mdhd = bytes(12) + struct.pack(">II", 30000, 301000) + bytes(4)
    stbl = atom(b"stsz", bytes(8) + struct.pack(">I", 301))
    stbl += atom(b"stts", bytes(4) + struct.pack(">I", len(timing)) + b"".join(struct.pack(">II", *row) for row in timing))
    media = atom(b"hdlr", bytes(8) + b"vide" + bytes(12)) + atom(b"mdhd", mdhd)
    media += atom(b"minf", atom(b"stbl", stbl))
    result = atom(b"trak", atom(b"mdia", media))
    if description is not None:
        desc = atom(b"desc", atom(b"data", struct.pack(">II", 1, 0) + description.encode()))
        result += atom(b"udta", atom(b"meta", bytes(4) + atom(b"ilst", desc)))
    return result


def valid_hand(timestamp, confidence=0.9):
    row = {"tracking_timestamp_us": str(timestamp)}
    for side in ("left", "right"):
        row[f"{side}_tracking_confidence"] = str(confidence)
        for i in range(21):
            for axis in "xyz":
                row[f"t{axis}_{side}_landmark_{i}_device"] = str(0.01 * (i + 1))
    return row


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


class MetadataTests(unittest.TestCase):
    def test_scalar_description_origin_is_independent_of_first_gaze_row(self):
        info = parse_moov(moov(str(ORIGIN)))
        self.assertEqual(info["origin_device_ns"], ORIGIN)
        self.assertEqual(info["fps"], 30)
        self.assertEqual(info["last_frame_time_ns"], 10000000000)
        self.assertEqual(verify_anchors(info, ORIGIN, ORIGIN + 10000000000)["first_anchor_error_ns"], 0)

    def test_wrong_unit_origin_or_duration_is_rejected(self):
        info = parse_moov(moov(str(ORIGIN)))
        for start, end in ((ORIGIN // 1000, ORIGIN + 10000000000),
                           (ORIGIN, ORIGIN + 11000000000)):
            with self.assertRaisesRegex(ValueError, "anchors disagree"):
                verify_anchors(info, start, end)

    def test_missing_description_and_vfr_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "description"):
            parse_moov(moov(None))
        with self.assertRaisesRegex(ValueError, "CFR"):
            parse_moov(moov(str(ORIGIN), timing=((300, 1000), (1, 1001))))

    def test_per_frame_vector_validates_length_and_order(self):
        with self.assertRaisesRegex(ValueError, "length/order"):
            parse_moov(moov(json.dumps([ORIGIN, ORIGIN + 100])))
        values = [ORIGIN + round(i * 1e9 / 30) for i in range(301)]
        self.assertEqual(parse_moov(moov(json.dumps(values)))["frame_device_ns"], values)
        values[100] = values[0]
        with self.assertRaisesRegex(ValueError, "length/order"):
            parse_moov(moov(json.dumps(values)))

    def test_vrs_start_and_end_bounds_for_front_and_end_index(self):
        class MemoryReader:
            def __init__(self, data):
                self.data, self.size = data, len(data)
            def read(self, offset, count):
                return self.data[offset:offset + count]
            buffered = read
        for front_index in (True, False):
            user_start = 160 if front_index else 116
            records = b""
            previous = 0
            for stream, timestamp in ((214, 1000.0), (1202, 1000.1), (214, 1010.0), (1202, 1010.01)):
                records += RECORD.pack(40, previous, stream, 2, timestamp, 1, 3, 0, 0) + bytes(8)
                previous = 40
            index_offset = 116 if front_index else user_start + len(records)
            header = struct.pack("<IIQIIqqqQQQII", 1769171286, 1699901039, 1, 80, 32,
                                 index_offset, 80, user_start if front_index else 0, 0, 0, 0, 1685221219, 844321366)
            description = RECORD.pack(36, 0, 1, 2, 0, 1, 3, 0, 0) + bytes(4)
            data = header + description + bytes(user_start - 116) + records
            if not front_index:
                data += bytes(64)
            self.assertEqual(vrs_rgb_bounds(MemoryReader(data), ORIGIN, 10000000000),
                             (ORIGIN, ORIGIN + 10000000000))

    def test_http_without_range_support_is_rejected_before_payload_read(self):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        with mock.patch("sensor_sync.urllib.request.urlopen", return_value=response), \
                mock.patch("sensor_sync.retry", side_effect=lambda operation: operation()):
            with self.assertRaisesRegex(ValueError, "exact metadata Range"):
                RangeReader("https://example.invalid/file", 1000).read(0, 80)
        response.read.assert_not_called()


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def test_latest_past_sample_has_bounded_age_and_no_future_leak(self):
        path = self.folder / "sample.csv"
        write_csv(path, [{"tracking_timestamp_us": t, "x": 1} for t in (100, 200, 300)])
        sampler = PastSampler(path, 50)
        self.addCleanup(sampler.close)
        self.assertEqual(sampler.sample(99000), (None, "before_first_sample"))
        self.assertEqual(sampler.sample(250000)[0][0], 200000)
        self.assertEqual(sampler.sample(299000), (None, "stale_sample"))
        self.assertEqual(sampler.sample(300000)[0][0], 300000)

    def test_nonmonotonic_and_mixed_unit_rows_fail(self):
        for times in ((100, 100), (100, 90), ("0.1", 100)):
            path = self.folder / "sample.csv"
            write_csv(path, [{"tracking_timestamp_us": t} for t in times])
            with self.assertRaises(ValueError):
                sampler = PastSampler(path, 50)
                try:
                    sampler.sample(1000000)
                finally:
                    sampler.close()

    def test_invalid_pose_is_masked_instead_of_used_as_zero_landmarks(self):
        value = hand_values(valid_hand(1, confidence=-1), 0.5)
        self.assertFalse(value["valid"])
        self.assertIsNone(value["hands"]["left"]["landmarks_device_m"])
        self.assertTrue(hand_values(valid_hand(1), 0.5)["valid"])

    def test_nonfinite_gaze_values_are_masked_and_json_serializable(self):
        value = gaze_values({"left_yaw_rads_cpf": "NaN", "right_yaw_rads_cpf": "0",
                             "pitch_rads_cpf": "0", "depth_m": "1", "tracking_timestamp_us": "1"})
        self.assertFalse(value["valid"])
        json.dumps(value, allow_nan=False)


class ExportIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.recording = Path(self.temp.name) / "session_one"
        self.recording.mkdir()
        info = parse_moov(moov(str(ORIGIN)))
        self.mapping = {"profile": PROFILE, "status": "verified_metadata_anchors",
                        "recording_id": self.recording.name, "clock_id": clock_id(self.recording.name),
                        "device_clock_id": "device", "video": info, "source_hashes": {"mp4": "video-sha"},
                        "anchor_checks": verify_anchors(info, ORIGIN, ORIGIN + 10000000000)}
        (self.recording / "sensor-clock.json").write_text(json.dumps(self.mapping))
        for view in ("leader", "helper"):
            folder = self.recording / "graphs" / view
            folder.mkdir(parents=True)
            path = folder / "graphs.jsonl.gz"
            with gzip.open(path, "wt") as handle:
                for i in range(4):
                    handle.write(json.dumps({"time_s": i / 5, "time_us": i * 200000,
                                             "source_pts_us": i * 200000, "source_frame_index": i * 6,
                                             "source_video_sha256": "video-sha", "pts_verified": True,
                                             "nodes": [], "edges": []}) + "\n")
            (folder / "progress.json").write_text(json.dumps({"complete": True, "sha256": source.file_digest(path)}))
            transcripts = self.recording / "transcripts"
            transcripts.mkdir(exist_ok=True)
            (transcripts / f"{view}_trimmed_sync_transcript.json").write_text('{"segments": []}')
        self.gaze_path = self.recording / "leader_gaze/general_eye_gaze.csv"
        write_csv(self.gaze_path, [{"tracking_timestamp_us": ORIGIN // 1000 + delta,
                                    "left_yaw_rads_cpf": 0, "right_yaw_rads_cpf": 0,
                                    "pitch_rads_cpf": 0, "depth_m": 1} for delta in (100, 100100, 200100, 300100, 400100, 500100)])
        self.hand_path = self.recording / "leader_hands/hand_tracking_results.csv"
        write_csv(self.hand_path, [valid_hand(ORIGIN // 1000 + delta,
                                            confidence=-1 if delta == 400000 else 0.9)
                                   for delta in (0, 200000, 400000, 600000)])
        self.mapping["source_sensor_sha256"] = {"gaze": source.file_digest(self.gaze_path),
                                               "hands": source.file_digest(self.hand_path)}
        self.mapping["source_transcript_sha256"] = {
            role: source.file_digest(self.recording / "transcripts" / f"{role}_trimmed_sync_transcript.json")
            for role in ("leader", "helper")}
        (self.recording / "sensor-clock.json").write_text(json.dumps(self.mapping))

    def rows(self, stride=1):
        folder = belief_data.input_folder(self.recording, stride)
        with gzip.open(folder / "steps.jsonl.gz", "rt") as handle:
            return [json.loads(line) for line in handle]

    def test_export_attaches_cues_with_same_step_time_and_raw_clock_provenance(self):
        manifest = belief_data.export_recording(self.recording)
        rows = self.rows()
        self.assertTrue(manifest["timed_model_inputs_aligned"])
        self.assertFalse(manifest["all_modalities_aligned"])  # Fixture has no benchmark annotation files.
        self.assertIsNone(rows[0]["gaze_features"])
        gaze = rows[1]["gaze_features"]
        self.assertEqual(gaze["aligned_time_us"], rows[1]["time_us"])
        self.assertEqual(gaze["source_timestamp_us"], ORIGIN // 1000 + 100100)
        self.assertEqual(gaze["age_ns"], 99900000)
        self.assertFalse(rows[2]["hand_features"]["valid"])
        self.assertIsNone(rows[2]["hand_features"]["hands"]["left"]["landmarks_device_m"])

    def test_changing_stride_does_not_change_sensor_at_shared_times(self):
        belief_data.export_recording(self.recording)
        base = self.rows()
        belief_data.export_recording(self.recording, step_stride=2)
        coarse = self.rows(2)
        self.assertEqual(base[2]["gaze_features"], coarse[1]["gaze_features"])
        self.assertEqual(base[2]["hand_features"], coarse[1]["hand_features"])

    def test_missing_source_keeps_null_and_alignment_incomplete(self):
        self.hand_path.unlink()
        manifest = belief_data.export_recording(self.recording)
        self.assertFalse(manifest["all_modalities_aligned"])
        self.assertTrue(all(row["hand_features"] is None for row in self.rows()))

    def test_unresolved_sensor_clock_retains_graphs_raw_sources_and_audits(self):
        # A real failure in the full CoMind batch has a matching start and a 9.6s tail mismatch.
        info = self.mapping["video"]
        raw = {"files": [
            {"path": "mp4s/leader_trimmed_sync.mp4", "size": 100, "hash": "video-sha"},
            {"path": "trimmed_vrs/leader_trimmed.vrs", "size": 100, "hash": "vrs-sha"},
            {"path": "mps_leader_trimmed_vrs/eye_gaze/general_eye_gaze.csv", "size": 100,
             "hash": self.mapping["source_sensor_sha256"]["gaze"]},
            {"path": "mps_leader_trimmed_vrs/hand_tracking/hand_tracking_results.csv", "size": 100,
             "hash": self.mapping["source_sensor_sha256"]["hands"]},
        ]}
        for role, digest in self.mapping["source_transcript_sha256"].items():
            raw["files"].append({"path": f"transcripts/{role}_trimmed_sync_transcript.json",
                                 "size": 100, "hash": digest})
        man = {"raw": json.dumps(raw), "hash_algorithm": "sha256"}
        with mock.patch("sensor_sync.read_mp4_clock", return_value=info), \
                mock.patch("sensor_sync.vrs_rgb_bounds", return_value=(ORIGIN, ORIGIN + 19_622_731_713)):
            with self.assertRaisesRegex(ValueError, "anchors disagree"):
                prepare_clock(self.recording, man)
            result = prepare_clock(self.recording, man, allow_unresolved=True)
        self.assertEqual(result["status"], "unresolved")
        self.assertIn("anchors disagree", result["error"])
        manifest = belief_data.export_recording(self.recording)
        self.assertFalse(manifest["timed_model_inputs_aligned"])
        self.assertFalse(manifest["all_modalities_aligned"])
        self.assertTrue(manifest["all_sensor_sources_present"])
        self.assertTrue(self.gaze_path.is_file() and self.hand_path.is_file())
        self.assertEqual(manifest["steps"], 4)
        self.assertTrue(all(row["gaze_features"] is None and row["hand_features"] is None for row in self.rows()))
        speech = json.loads((belief_data.input_folder(self.recording) / "speech_alignment.json").read_text())
        self.assertTrue(speech["source_hashes_verified"] and speech["all_speech_times_within_video"])

    def test_wrong_video_hash_rejects_sensor_join(self):
        self.mapping["source_hashes"]["mp4"] = "other-video"
        (self.recording / "sensor-clock.json").write_text(json.dumps(self.mapping))
        with self.assertRaisesRegex(ValueError, "source hash differs"):
            belief_data.export_recording(self.recording)

    def test_changed_or_wrong_recording_csv_is_rejected(self):
        with self.gaze_path.open("a") as handle:
            handle.write("changed source CSV\n")
        with self.assertRaisesRegex(ValueError, "CSV hash does not match"):
            belief_data.export_recording(self.recording)

    def test_default_training_loader_accepts_verified_clock_with_independent_labels(self):
        folder = self.recording / "labels"
        folder.mkdir()
        (folder / "belief_labels.jsonl").write_text(json.dumps({
            "time_s": 0.4, "person_id": "leader", "object_id": "helper:1",
            "candidate_location_ids": ["A", "B"], "allowed_location_ids": ["B"],
            "label_source": "synthetic independent fixture"}) + "\n")
        belief_data.export_recording(self.recording)
        self.assertEqual(len(list(belief_data.iter_training_steps(self.recording))), 4)


if __name__ == "__main__":
    unittest.main()
