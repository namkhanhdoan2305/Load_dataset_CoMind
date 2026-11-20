"""Export chronological graph/speech inputs and load independently supplied belief labels."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import comind_download as source
from comind_to_hf import write_json_atomic


def speech_events(recording):
    events = []
    for stream in ("leader", "helper"):
        path = Path(recording) / "transcripts" / f"{stream}_trimmed_sync_transcript.json"
        if not path.exists():
            continue
        for index, segment in enumerate(json.loads(path.read_text()).get("segments", [])):
            if "start" not in segment or "end" not in segment:
                continue
            start, end = float(segment["start"]), float(segment["end"])
            if not math.isfinite(start) or not math.isfinite(end) or end < start:
                raise ValueError(f"Invalid speech segment times in {path}")
            event_id = hashlib.sha256(
                f"{Path(recording).name}|{stream}|{index}|{start}|{end}".encode()).hexdigest()[:24]
            events.append({"event_id": event_id, "audio_stream": stream,
                           "segment_index": index, "start_s": start, "available_s": end,
                           "text": segment.get("text", ""),
                           "speaker_ids": sorted({str(w["speaker"]) for w in segment.get("words", [])
                                                  if "speaker" in w}),
                           "type": "completed_transcript_segment"})
    return sorted(events, key=lambda event: (event["available_s"], event["event_id"]))


def export_recording(recording, context_seconds=10):
    recording = Path(recording)
    if not math.isfinite(context_seconds) or context_seconds < 0:
        raise ValueError("context_seconds must be finite and nonnegative")
    streams = {}
    for view in ("leader", "helper"):
        path = recording / "graphs" / view / "graphs.jsonl.gz"
        progress = json.loads((path.parent / "progress.json").read_text())
        if not progress["complete"] or source.file_digest(path) != progress["sha256"]:
            raise ValueError(f"Complete, verified graphs required before export: {view}")
        streams[view] = path
    folder = recording / "belief_inputs"
    folder.mkdir(parents=True, exist_ok=True)
    events = speech_events(recording)
    write_json_atomic(folder / "speech_events.json", events)
    sync_path = recording / "sync-status.json"
    sync = json.loads(sync_path.read_text()) if sync_path.exists() else {}
    temporary = folder / "steps.jsonl.gz.tmp"
    count, next_event, previous_time = 0, 0, -1
    recent = []
    with gzip.open(streams["leader"], "rt") as leader, \
            gzip.open(streams["helper"], "rt") as helper, gzip.open(temporary, "wt") as output:
        while True:
            a, b = leader.readline(), helper.readline()
            if not a and not b:
                break
            if not a or not b:
                raise ValueError("Paired streams have unequal lengths")
            human, robot = json.loads(a), json.loads(b)
            timestamp = robot["time_s"]
            if abs(human["time_s"] - timestamp) > 0.001 or timestamp <= previous_time:
                raise ValueError("Paired graph timestamps are invalid")
            new_events = []
            while next_event < len(events) and events[next_event]["available_s"] <= timestamp:
                new_events.append(events[next_event])
                next_event += 1
            recent = [event for event in recent + new_events
                      if timestamp - event["available_s"] <= context_seconds]
            row = {"schema_version": 1, "recording_id": recording.name,
                   "step_index": count, "time_s": timestamp,
                   "sequence_start": count == 0,
                   "robot_observation_graph": robot, "human_observation_graph": human,
                   "new_speech_events": new_events,
                   "recent_speech_event_ids": [event["event_id"] for event in recent],
                   "gaze_features": None, "hand_features": None,
                   "sensor_alignment": "unresolved",
                   "source_time_interval": {"previous_s": previous_time, "current_s": timestamp}}
            output.write(json.dumps(row, allow_nan=False) + "\n")
            previous_time, count = timestamp, count + 1
    temporary.replace(folder / "steps.jsonl.gz")
    labels = recording / "labels" / "belief_labels.jsonl"
    label_count = 0
    if labels.exists():
        with labels.open() as handle:
            label_count = sum(bool(line.strip()) for line in handle)
    manifest = {"recording_id": recording.name, "steps": count, "speech_events": len(events),
                "context_seconds": context_seconds,
                "sensor_alignment": sync.get("all_modalities_aligned", False),
                "cross_view_entity_matching": "unresolved",
                "labels_path": "../labels/belief_labels.jsonl", "label_count": label_count,
                "steps_sha256": source.file_digest(folder / "steps.jsonl.gz"),
                "speech_events_sha256": source.file_digest(folder / "speech_events.json"),
                "speech_policy": "Emit completed segment once; context lookup does not create new evidence",
                "belief_memory_policy": "Maintained by the future encoder; no automatic time decay in data",
                "warning": "Observed human-view graph is not a human-belief label. "
                           "Audio streams may capture the same utterance; sources are not independent evidence."}
    write_json_atomic(folder / "manifest.json", manifest)
    return manifest


def iter_training_steps(recording):
    """Yield every time step for recurrent state updates, with zero or more labeled queries."""
    recording = Path(recording)
    path = recording / "labels" / "belief_labels.jsonl"
    if not path.exists():
        raise ValueError(f"Supply verified belief labels at {path} before supervised training")
    labels = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        label = json.loads(line)
        candidates, allowed = label["candidate_location_ids"], label["allowed_location_ids"]
        timestamp = float(label["time_s"])
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Belief label time must be finite and nonnegative")
        if not candidates or len(set(candidates)) != len(candidates) or not allowed or not set(allowed) <= set(candidates):
            raise ValueError("Belief labels require candidate locations and a nonempty allowed subset")
        if not label.get("person_id") or not label.get("object_id") or not label.get("label_source"):
            raise ValueError("Belief labels require person_id, object_id and independent label_source")
        labels.setdefault(round(timestamp * 1e6), []).append(label)
    if not labels:
        raise ValueError("No supervised belief labels supplied")
    inputs = recording / "belief_inputs" / "steps.jsonl.gz"
    manifest = json.loads((inputs.parent / "manifest.json").read_text())
    if source.file_digest(inputs) != manifest["steps_sha256"]:
        raise ValueError("Training input hash changed")
    with gzip.open(inputs, "rt") as handle:
        times = {round(json.loads(line)["time_s"] * 1e6) for line in handle}
    if set(labels) - times:
        raise ValueError("Some label timestamps do not match the exported graph grid")
    with gzip.open(inputs, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            timestamp = round(row["time_s"] * 1e6)
            yield row, labels.get(timestamp, [])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", help="Path to one recording directory")
    args = parser.parse_args()
    print(json.dumps(export_recording(args.recording), indent=2))


if __name__ == "__main__":
    main()
