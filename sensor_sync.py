"""Verify CoMind MP4↔Aria clock anchors, then causally sample MPS gaze/hands.

Reads only bounded MP4/VRS metadata ranges. Never infers video zero from a CSV.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import re
import struct
import urllib.request

import comind_source as source
from comind_data import retry, write_json_atomic
from time_contract import clock_id


PROFILE = "comind_mp4_description_vrs_rgb_v1"
JOIN_POLICY = {"method": "latest_at_or_before_reference", "gaze_max_age_us": 150000,
               "hand_max_age_us": 50000, "hand_min_confidence": 0.5}
RECORD = struct.Struct("<IIiIdHBBI")


class RangeReader:
    def __init__(self, url, size, max_bytes=32 * 2**20):
        self.url, self.size, self.max_bytes = url, size, max_bytes
        self.bytes_read = 0
        self.cache_start, self.cache = 0, b""

    def read(self, offset, count):
        if offset < 0 or count < 0 or offset + count > self.size:
            raise ValueError("Metadata range is outside source file")
        if self.bytes_read + count > self.max_bytes:
            raise ValueError("Metadata read budget exceeded; refusing full VRS/video download")
        def request():
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={offset}-{offset + count - 1}"})
            with urllib.request.urlopen(req, timeout=45) as response:
                expected = f"bytes {offset}-{offset + count - 1}/{self.size}"
                if response.status != 206 or response.headers.get("Content-Range") != expected:
                    raise ValueError("Server did not honor exact metadata Range")
                data = response.read(count)
                if len(data) != count:
                    raise ValueError("Truncated metadata range")
                return data
        data = retry(request)
        self.bytes_read += len(data)
        return data

    def buffered(self, offset, count):
        if not self.cache_start <= offset or offset + count > self.cache_start + len(self.cache):
            self.cache_start = offset
            self.cache = self.read(offset, min(max(65536, count), self.size - offset))
        start = offset - self.cache_start
        return self.cache[start:start + count]


def boxes(data):
    pos = 0
    while pos < len(data):
        if pos + 8 > len(data):
            raise ValueError("Truncated MP4 metadata box")
        size, kind = struct.unpack_from(">I4s", data, pos)
        header = 8
        if size == 1:
            size = struct.unpack_from(">Q", data, pos + 8)[0]
            header = 16
        if size == 0:
            size = len(data) - pos
        if size < header or pos + size > len(data):
            raise ValueError("Invalid MP4 metadata box size")
        yield kind, data[pos + header:pos + size]
        pos += size


def parse_moov(data):
    descriptions, video = [], None
    for kind, body in boxes(data):
        if kind == b"udta":
            for tag, content in boxes(body):
                if tag != b"meta":
                    continue
                for subtag, subcontent in boxes(content[4:]):
                    if subtag != b"ilst":
                        continue
                    for key, value in boxes(subcontent):
                        if key == b"desc":
                            for field, payload in boxes(value):
                                if field == b"data":
                                    descriptions.append(payload[8:].decode("utf-8").strip())
        if kind == b"trak":
            track = dict(boxes(body))
            media = dict(boxes(track[b"mdia"]))
            if media[b"hdlr"][8:12] != b"vide":
                continue
            if video is not None:
                raise ValueError("Expected a single camera video track")
            mdhd = media[b"mdhd"]
            timescale = struct.unpack_from(">I", mdhd, 20 if mdhd[0] == 1 else 12)[0]
            table = dict(boxes(dict(boxes(media[b"minf"]))[b"stbl"]))
            count = struct.unpack_from(">I", table[b"stsz"], 8)[0]
            entries = struct.unpack_from(">I", table[b"stts"], 4)[0]
            timing = [struct.unpack_from(">II", table[b"stts"], 8 + 8 * i) for i in range(entries)]
            deltas = {delta for _, delta in timing}
            if not count or not timescale or len(deltas) != 1 or sum(n for n, _ in timing) != count:
                raise ValueError("Clock extraction requires a nonempty, verified CFR MP4 track")
            delta = next(iter(deltas))
            if not delta:
                raise ValueError("Invalid MP4 sample duration")
            video = {"frame_count": count, "timescale": timescale, "frame_duration_ticks": delta,
                     "fps": timescale / delta,
                     "last_frame_time_ns": round((count - 1) * delta * 1e9 / timescale)}
    if video is None or len(set(descriptions)) != 1:
        raise ValueError("Missing or conflicting MP4 description clock metadata")
    text = descriptions[0]
    values = [int(text)] if re.fullmatch(r"[0-9]+", text) else json.loads(text)
    if not isinstance(values, list) or not values or any(type(t) is not int or t <= 0 for t in values):
        raise ValueError("MP4 description is not an Aria timestamp/vector in nanoseconds")
    if len(values) not in (1, video["frame_count"]) or any(a > b for a, b in zip(values, values[1:])):
        raise ValueError("MP4 timestamp vector length/order disagrees with its frames")
    video.update(origin_device_ns=values[0],
                 mapping_kind="scalar_origin_ns" if len(values) == 1 else "per_frame_device_ns")
    if len(values) > 1:
        video["frame_device_ns"] = values
    return video


def read_mp4_clock(reader):
    offset = 0
    for _ in range(128):
        size, kind = struct.unpack(">I4s", reader.read(offset, 8))
        header = 8
        if size == 1:
            size = struct.unpack(">Q", reader.read(offset + 8, 8))[0]
            header = 16
        if size == 0:
            size = reader.size - offset
        if size < header or offset + size > reader.size:
            raise ValueError("Invalid MP4 root box")
        if kind == b"moov":
            return parse_moov(reader.read(offset + header, size - header))
        offset += size
        if offset >= reader.size:
            break
    raise ValueError("MP4 moov clock metadata missing")


def vrs_rgb_bounds(reader, origin_ns, video_end_ns):
    header = struct.unpack("<IIQIIqqqQQQII", reader.read(0, 80))
    file_header_size, record_size, index_offset, description_offset, first_user = header[3:8]
    if file_header_size != 80 or record_size != RECORD.size or not 0 < description_offset < reader.size:
        raise ValueError("Unsupported VRS header; refusing guessed device origin")
    position = first_user or (description_offset + struct.unpack("<I", reader.buffered(description_offset, 4))[0])
    user_start = position
    first = None
    for _ in range(2048):
        values = RECORD.unpack(reader.buffered(position, record_size))
        size, _, stream, _, timestamp, instance, kind, _, _ = values
        if size < record_size or position + size > reader.size:
            raise ValueError("Invalid VRS forward record boundary")
        if stream == 214 and instance == 1 and kind == 3:
            first = round(timestamp * 1e9)
            break
        position += size
    if first is None:
        raise ValueError("Cannot verify first VRS RGB anchor within bounded metadata scan")
    end = index_offset if index_offset > user_start else reader.size
    tail_start = max(file_header_size, end - 131072)
    tail = reader.read(tail_start, end - tail_start)
    candidates = []
    for offset in range(len(tail) - RECORD.size + 1):
        values = RECORD.unpack_from(tail, offset)
        size, prev, stream, version, timestamp, instance, kind, comp, _ = values
        if (size == len(tail) - offset and size >= record_size and 0 < prev < 64 * 2**20
                and 0 < stream < 100000 and 0 < instance < 100 and kind == 3 and comp <= 3
                and 0 < version < 100 and math.isfinite(timestamp)
                and origin_ns / 1e9 <= timestamp <= (origin_ns + video_end_ns) / 1e9 + 60):
            candidates.append((tail_start + offset, values))
    if len(candidates) != 1:
        raise ValueError("Cannot identify an unambiguous last VRS record")
    position, values = candidates[0]
    last = None
    for _ in range(2048):
        size, previous, stream, _, timestamp, instance, kind, _, _ = values
        if stream == 214 and instance == 1 and kind == 3:
            last = round(timestamp * 1e9)
            break
        if previous < record_size or position - previous < user_start:
            raise ValueError("Invalid VRS backward record boundary")
        next_position = position - previous
        values = RECORD.unpack(reader.buffered(next_position, record_size))
        if values[0] != previous:
            raise ValueError("Broken VRS previous-record chain")
        position = next_position
    if last is None:
        raise ValueError("Cannot verify last VRS RGB anchor within metadata scan")
    return first, last


def verify_anchors(info, first, last):
    first_delta = first - info["origin_device_ns"]
    expected_last = (info["frame_device_ns"][-1] if info["mapping_kind"] == "per_frame_device_ns"
                     else info["origin_device_ns"] + info["last_frame_time_ns"])
    last_delta = last - expected_last
    tolerance = max(1000000, round(1e9 / info["fps"] * 1.5))
    if abs(first_delta) > 1000 or abs(last_delta) > tolerance:
        raise ValueError(f"MP4/VRS clock anchors disagree: start={first_delta}ns, end={last_delta}ns")
    return {"first_rgb_device_ns": first, "last_rgb_device_ns": last,
            "first_anchor_error_ns": first_delta, "last_anchor_error_ns": last_delta,
            "first_tolerance_ns": 1000, "end_tolerance_ns": tolerance}


def prepare_clock(recording, manifest, base_url=source.DEFAULT_BASE_URL):
    """Independent of model inference: can retrofit clock metadata to existing graphs."""
    recording = Path(recording)
    rid = recording.name
    raw = json.loads(manifest["raw"])
    entries = {row["path"]: row for row in raw["files"]}
    paths = {"mp4": "mp4s/leader_trimmed_sync.mp4", "vrs": "trimmed_vrs/leader_trimmed.vrs"}
    selected = {key: entries[path] for key, path in paths.items()}
    if manifest["hash_algorithm"] != "sha256":
        raise ValueError("Sensor clock profile requires SHA256 source manifests")
    sensor_paths = {"gaze": "mps_leader_trimmed_vrs/eye_gaze/general_eye_gaze.csv",
                    "hands": "mps_leader_trimmed_vrs/hand_tracking/hand_tracking_results.csv"}
    sensor_hashes = {key: entries[path]["hash"] for key, path in sensor_paths.items() if path in entries}
    transcript_hashes = {role: entries[path]["hash"] for role in ("leader", "helper")
                         if (path := f"transcripts/{role}_trimmed_sync_transcript.json") in entries}
    path = recording / "sensor-clock.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if (saved.get("profile") == PROFILE and saved.get("recording_id") == rid
                and saved.get("source_hashes") == {k: row["hash"] for k, row in selected.items()}
                and saved.get("source_sensor_sha256") == sensor_hashes
                and saved.get("status") == "verified_metadata_anchors"
                and saved.get("clock_id") == clock_id(rid)):
            proof = saved["anchor_checks"]
            verify_anchors(saved["video"], proof["first_rgb_device_ns"], proof["last_rgb_device_ns"])
            if saved.get("source_transcript_sha256") != transcript_hashes:
                saved["source_transcript_sha256"] = transcript_hashes
                write_json_atomic(path, saved)
            return saved
    readers = {key: RangeReader(source.build_url(base_url, rid, paths[key]), row["size"],
                                max_bytes=32 * 2**20 if key == "mp4" else 4 * 2**20)
               for key, row in selected.items()}
    info = read_mp4_clock(readers["mp4"])
    first, last = vrs_rgb_bounds(readers["vrs"], info["origin_device_ns"], info["last_frame_time_ns"])
    proof = verify_anchors(info, first, last)
    mapping = {"schema_version": 1, "profile": PROFILE, "recording_id": rid,
               "status": "verified_metadata_anchors", "clock_id": clock_id(rid),
               "device_clock_id": f"comind:{rid}:leader_device_time", "video": info,
               "source_paths": paths, "source_hashes": {k: row["hash"] for k, row in selected.items()},
               "source_sensor_sha256": sensor_hashes,
               "source_transcript_sha256": transcript_hashes,
               "hash_algorithm": manifest["hash_algorithm"], "anchor_checks": proof,
               "metadata_bytes_read": {k: r.bytes_read for k, r in readers.items()},
               "physical_audio_video_alignment_verified": False,
               "basis": "MP4 description nanoseconds; verified RGB first/last VRS record anchors. "
                        "Scalar map uses 1e9 device nanoseconds per MP4 second; CFR resampling tolerance is reported."}
    recording.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, mapping)
    print(f"{rid}: sensor clock verified, origin={info['origin_device_ns']}ns, "
          f"end anchor error={proof['last_anchor_error_ns'] / 1e6:.3f}ms", flush=True)
    return mapping


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


class PastSampler:
    def __init__(self, path, max_age_us):
        self.handle = Path(path).open(newline="")
        self.reader = csv.DictReader(self.handle)
        if "tracking_timestamp_us" not in (self.reader.fieldnames or []):
            self.handle.close()
            raise ValueError(f"Sensor CSV has no tracking_timestamp_us: {path}")
        self.max_age_ns = max_age_us * 1000
        self.previous_timestamp, self.last, self.last_reference = -1, None, -1
        try:
            self.next = self._advance()
        except BaseException:
            self.handle.close()
            raise

    def _advance(self):
        row = next(self.reader, None)
        if row is None:
            return None
        text = row["tracking_timestamp_us"]
        if not re.fullmatch(r"[0-9]+", text):
            raise ValueError("Sensor tracking_timestamp_us must be integer microseconds")
        timestamp = int(text) * 1000
        if timestamp <= self.previous_timestamp:
            raise ValueError("Sensor timestamps must be strictly increasing")
        self.previous_timestamp = timestamp
        return timestamp, row

    def sample(self, reference_ns):
        if reference_ns < self.last_reference:
            raise ValueError("Sensor reference timestamps went backwards")
        self.last_reference = reference_ns
        while self.next is not None and self.next[0] <= reference_ns:
            self.last = self.next
            self.next = self._advance()
        if self.last is None:
            return None, "before_first_sample"
        if reference_ns - self.last[0] > self.max_age_ns:
            return None, "stale_sample"
        return self.last, "matched"

    def close(self):
        self.handle.close()


def gaze_values(row):
    values = {key: (value if key == "session_uid" else number(value)) for key, value in row.items()
              if key != "tracking_timestamp_us"}
    valid = all(values.get(k) is not None for k in
                ("left_yaw_rads_cpf", "right_yaw_rads_cpf", "pitch_rads_cpf", "depth_m"))
    return {"valid": bool(valid and values["depth_m"] > 0), "coordinate_frame": "CPF",
            "angle_unit": "radians", "length_unit": "meters", "values": values}


def hand_values(row, min_confidence):
    hands = {}
    for side in ("left", "right"):
        confidence = number(row.get(f"{side}_tracking_confidence"))
        points = [[number(row.get(f"t{axis}_{side}_landmark_{i}_device")) for axis in "xyz"] for i in range(21)]
        finite = all(x is not None for point in points for x in point)
        valid = confidence is not None and confidence >= min_confidence and finite and any(x != 0 for point in points for x in point)
        hands[side] = {"valid": bool(valid), "tracking_confidence": confidence,
                       "landmarks_device_m": points if valid else None,
                       "wrist_pose": {k: number(row.get(k)) for k in row if f"_{side}_device_wrist" in k} if valid else None,
                       "surface_normals": {k: number(row.get(k)) for k in row
                                           if f"_{side}_palm_device" in k or f"_{side}_wrist_device" in k} if valid else None}
    return {"valid": any(h["valid"] for h in hands.values()), "coordinate_frame": "leader_device",
            "length_unit": "meters", "hands": hands}


class SensorAligner:
    def __init__(self, recording):
        self.recording = Path(recording)
        path = self.recording / "sensor-clock.json"
        self.mapping = json.loads(path.read_text()) if path.exists() else None
        self.samplers, self.stats, self.source_hashes = {}, {}, {}
        if self.mapping is None:
            return
        if (self.mapping.get("status") != "verified_metadata_anchors" or self.mapping.get("profile") != PROFILE
                or self.mapping.get("recording_id") != self.recording.name
                or self.mapping.get("clock_id") != clock_id(self.recording.name)
                or not isinstance(self.mapping.get("source_sensor_sha256"), dict)):
            raise ValueError("Unverified or wrong-recording sensor clock mapping")
        proof = self.mapping["anchor_checks"]
        verify_anchors(self.mapping["video"], proof["first_rgb_device_ns"], proof["last_rgb_device_ns"])
        for name, group, file, age in [
                ("gaze", "leader_gaze", "general_eye_gaze.csv", JOIN_POLICY["gaze_max_age_us"]),
                ("hands", "leader_hands", "hand_tracking_results.csv", JOIN_POLICY["hand_max_age_us"])]:
            csv_path = self.recording / group / file
            self.stats[name] = {"present": csv_path.exists(), "matched": 0, "valid": 0,
                                "missing": 0, "max_age_ns": None, "missing_reasons": {}}
            if csv_path.exists():
                self.source_hashes[name] = source.file_digest(csv_path)
                if self.source_hashes[name] != self.mapping["source_sensor_sha256"].get(name):
                    self.close()
                    raise ValueError(f"Sensor CSV hash does not match this recording's source manifest: {name}")
                try:
                    self.samplers[name] = PastSampler(csv_path, age)
                except BaseException:
                    self.close()
                    raise

    def attach(self, human_graph, step_time_us):
        if self.mapping is None:
            return None, None
        if human_graph.get("source_video_sha256") != self.mapping["source_hashes"]["mp4"]:
            raise ValueError("Sensor clock source hash differs from the leader graph's MP4")
        video = self.mapping["video"]
        pts_us = human_graph.get("source_pts_us", human_graph["time_us"])
        cutoff = video["origin_device_ns"] + pts_us * 1000
        if video["mapping_kind"] == "per_frame_device_ns":
            cutoff = min(cutoff, video["frame_device_ns"][human_graph["source_frame_index"]])
        features = {}
        for name in ("gaze", "hands"):
            sampler = self.samplers.get(name)
            pair, reason = sampler.sample(cutoff) if sampler else (None, "source_modality_missing")
            if pair is None:
                self.stats[name]["missing"] += 1
                counts = self.stats[name]["missing_reasons"]
                counts[reason] = counts.get(reason, 0) + 1
                features[name] = None
                continue
            timestamp_ns, row = pair
            value = gaze_values(row) if name == "gaze" else hand_values(row, JOIN_POLICY["hand_min_confidence"])
            age = cutoff - timestamp_ns
            value.update(aligned_time_us=step_time_us, clock_id=self.mapping["clock_id"],
                         source_timestamp_us=timestamp_ns // 1000,
                         source_clock_id=self.mapping["device_clock_id"],
                         reference_device_time_ns=cutoff, reference_video_time_us=pts_us,
                         age_ns=age, join_status=reason)
            features[name] = value
            self.stats[name]["matched"] += 1
            self.stats[name]["valid"] += int(value["valid"])
            self.stats[name]["max_age_ns"] = max(age, self.stats[name]["max_age_ns"] or 0)
        return features["gaze"], features["hands"]

    def report(self):
        return {"clock_mapping_verified": self.mapping is not None,
                "all_sensor_sources_present": len(self.samplers) == 2,
                "join_policy": JOIN_POLICY, "modalities": self.stats, "source_csv_sha256": self.source_hashes,
                "clock_mapping_sha256": source.file_digest(self.recording / "sensor-clock.json") if self.mapping else None,
                "note": "Missing/stale samples stay null; invalid hands are masked, no forward fill of valid pose. "
                        "Clock alignment is distinct from tracking validity or physical AV verification."}

    def close(self):
        for sampler in self.samplers.values():
            sampler.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", help="Recording directory on Drive; can retrofit without rerunning SGG")
    parser.add_argument("--prepare-only", action="store_true", help="Verify clock metadata without requiring completed graphs")
    args = parser.parse_args()
    recording = Path(args.recording)
    manifest = source.fetch_manifest(source.DEFAULT_BASE_URL, recording.name, 60)
    prepare_clock(recording, manifest)
    if args.prepare_only:
        print("Sensor clock prepared; aligned gaze/hand features are exported when paired graphs are complete")
        return
    from sgg_pipeline import link_graph_views, write_sync_report
    target = recording.parent.parent
    write_sync_report(target, recording.name)
    link_graph_views(target, recording.name)
    from belief_data import export_recording
    print(json.dumps(export_recording(recording), indent=2))


if __name__ == "__main__":
    main()
