"""Export chronological graph/speech inputs and load independently supplied belief labels."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import comind_download as source
from comind_to_hf import write_json_atomic
from time_contract import clock_id, graph_time_us, microseconds


def speech_events(recording):
    events = []
    for stream in ("leader", "helper"):
        path = Path(recording) / "transcripts" / f"{stream}_trimmed_sync_transcript.json"
        if not path.exists():
            continue
        for index, segment in enumerate(json.loads(path.read_text()).get("segments", [])):
            if "start" not in segment or "end" not in segment:
                raise ValueError(f"Missing speech segment timestamps in {path}: segment {index}")
            start, end = float(segment["start"]), float(segment["end"])
            if not math.isfinite(start) or not math.isfinite(end) or end < start:
                raise ValueError(f"Invalid speech segment times in {path}")
            event_id = hashlib.sha256(
                f"{Path(recording).name}|{stream}|{index}|{start}|{end}".encode()).hexdigest()[:24]
            events.append({"event_id": event_id, "audio_stream": stream,
                           "segment_index": index, "start_s": start, "available_s": end,
                           "start_us": microseconds(start), "available_us": microseconds(end),
                           "clock_id": clock_id(Path(recording).name),
                           "clock_alignment": "as_published_trimmed_sync_transcript",
                           "text": segment.get("text", ""),
                           "speaker_ids": sorted({str(w["speaker"]) for w in segment.get("words", [])
                                                  if "speaker" in w}),
                           "type": "completed_transcript_segment"})
    return sorted(events, key=lambda event: (event["available_us"], event["event_id"]))


def input_folder(recording, step_stride=1):
    if isinstance(step_stride, bool) or not isinstance(step_stride, int) or step_stride < 1:
        raise ValueError("step_stride must be a positive integer")
    return Path(recording) / ("belief_inputs" if step_stride == 1 else f"belief_inputs_stride_{step_stride}")


def export_recording(recording, context_seconds=10, step_stride=1):
    recording = Path(recording)
    folder = input_folder(recording, step_stride)
    if not math.isfinite(context_seconds) or context_seconds < 0:
        raise ValueError("context_seconds must be finite and nonnegative")
    streams = {}
    for view in ("leader", "helper"):
        path = recording / "graphs" / view / "graphs.jsonl.gz"
        progress = json.loads((path.parent / "progress.json").read_text())
        if not progress["complete"] or source.file_digest(path) != progress["sha256"]:
            raise ValueError(f"Complete, verified graphs required before export: {view}")
        streams[view] = path
    folder.mkdir(parents=True, exist_ok=True)
    events = speech_events(recording)
    write_json_atomic(folder / "speech_events.json", events)
    sync_path = recording / "sync-status.json"
    sync = json.loads(sync_path.read_text()) if sync_path.exists() else {}
    temporary = folder / "steps.jsonl.gz.tmp"
    count, source_count, next_event, previous_time = 0, 0, 0, -1
    previous_us = -1
    first_source_us = None
    base_period_us = None
    graph_pts_verified = True
    recent = []
    emissions = []
    context_us = microseconds(context_seconds)
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
            human_us = graph_time_us(human, recording.name)
            robot_us = graph_time_us(robot, recording.name)
            graph_pts_verified = graph_pts_verified and human.get("pts_verified", False) and robot.get("pts_verified", False)
            source_delta = abs(human.get("source_pts_us", human_us) - robot.get("source_pts_us", robot_us))
            if human_us != robot_us or source_delta > 1000 or robot_us <= previous_us:
                raise ValueError("Paired graph timestamps are invalid")
            if previous_us >= 0:
                delta = robot_us - previous_us
                if base_period_us is None:
                    base_period_us = delta
                elif delta != base_period_us:
                    raise ValueError("Base graph grid must be uniform for step-stride comparisons")
            else:
                first_source_us = robot_us
            previous_us = robot_us
            source_index = source_count
            source_count += 1
            if source_index % step_stride:
                continue
            new_events = []
            while next_event < len(events) and events[next_event]["available_us"] <= robot_us:
                event = events[next_event]
                new_events.append(event)
                emissions.append({"event_id": event["event_id"], "audio_stream": event["audio_stream"],
                                  "start_us": event["start_us"], "available_us": event["available_us"],
                                  "first_emitted_step": count, "first_emitted_us": robot_us,
                                  "step_delay_us": robot_us - event["available_us"],
                                  "segment_duration_us": event["available_us"] - event["start_us"]})
                next_event += 1
            recent = [event for event in recent + new_events
                      if robot_us - event["available_us"] <= context_us]
            row = {"schema_version": 1, "recording_id": recording.name,
                   "step_index": count, "source_step_index": source_index, "time_s": timestamp,
                   "time_us": robot_us, "clock_id": clock_id(recording.name),
                   "graph_pair_delta_us": source_delta,
                   "sequence_start": count == 0,
                   "robot_observation_graph": robot, "human_observation_graph": human,
                   "new_speech_events": new_events,
                   "recent_speech_event_ids": [event["event_id"] for event in recent],
                   "gaze_features": None, "hand_features": None,
                   "sensor_alignment": "unresolved",
                   "source_time_interval": {"previous_s": previous_time, "current_s": timestamp},
                   "speech_interval_policy": "All completed segments since previous retained step; emit once"}
            output.write(json.dumps(row, allow_nan=False) + "\n")
            previous_time, count = timestamp, count + 1
    if not count:
        raise ValueError("Cannot export an empty graph sequence")
    temporary.replace(folder / "steps.jsonl.gz")
    speech_audit = {
        "recording_id": recording.name, "clock_id": clock_id(recording.name),
        "clock_alignment": "as_published_trimmed_sync_transcript",
        "physical_audio_video_alignment_verified": False,
        "note": "Timestamp checks do not verify ASR accuracy, speaker identity or physical AV offset.",
        "streams": {stream: {"present": (recording / "transcripts" / f"{stream}_trimmed_sync_transcript.json").exists(),
                             "segment_count": sum(e["audio_stream"] == stream for e in events)}
                    for stream in ("leader", "helper")},
        "first_base_graph_us": first_source_us,
        "last_base_graph_us": previous_us, "last_retained_graph_us": microseconds(previous_time),
        "emitted_event_count": next_event, "total_event_count": len(events),
        "max_step_delay_us": max((e["step_delay_us"] for e in emissions), default=None),
        "emissions": emissions,
        "unemitted_events": [{"event_id": e["event_id"], "available_us": e["available_us"],
                              "reason": "segment_ends_after_last_retained_graph"} for e in events[next_event:]]}
    write_json_atomic(folder / "speech_alignment.json", speech_audit)
    labels = recording / "labels" / "belief_labels.jsonl"
    label_count = 0
    if labels.exists():
        with labels.open() as handle:
            label_count = sum(bool(line.strip()) for line in handle)
    manifest = {"recording_id": recording.name, "steps": count, "speech_events": len(events),
                "context_seconds": context_seconds,
                "sensor_alignment": sync.get("all_modalities_aligned", False),
                "all_modalities_aligned": sync.get("all_modalities_aligned", False),
                "clock_id": clock_id(recording.name), "timestamp_unit": "microseconds",
                "decoder_pts_verified": bool(graph_pts_verified),
                "base_graph_steps": source_count, "step_stride": step_stride,
                "base_period_us": base_period_us,
                "step_period_us": None if base_period_us is None else base_period_us * step_stride,
                "source_graph_sha256": {view: source.file_digest(path) for view, path in streams.items()},
                "task_progress_alignment": "not_implemented; source annotation event IDs preserved separately",
                "cross_view_entity_matching": "unresolved",
                "labels_path": "../labels/belief_labels.jsonl", "label_count": label_count,
                "steps_sha256": source.file_digest(folder / "steps.jsonl.gz"),
                "speech_events_sha256": source.file_digest(folder / "speech_events.json"),
                "speech_alignment_path": "speech_alignment.json", "speech_events_emitted": next_event,
                "speech_events_unemitted": len(events) - next_event,
                "speech_policy": "Emit completed segment once; context lookup does not create new evidence",
                "belief_memory_policy": "Maintained by the future encoder; no automatic time decay in data",
                "warning": "Observed human-view graph is not a human-belief label. "
                           "Audio streams may capture the same utterance; sources are not independent evidence."}
    write_json_atomic(folder / "manifest.json", manifest)
    return manifest


def iter_training_steps(recording, require_sensor_alignment=True, step_stride=1):
    """Yield every time step for recurrent state updates, with zero or more labeled queries."""
    recording = Path(recording)
    inputs = input_folder(recording, step_stride) / "steps.jsonl.gz"
    manifest = json.loads((inputs.parent / "manifest.json").read_text())
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
    if require_sensor_alignment and not manifest.get("all_modalities_aligned", False):
        raise ValueError("Multimodal clock alignment is unresolved. For an explicitly graph/speech-only "
                         "experiment use require_sensor_alignment=False")
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
    parser.add_argument("--step-stride", type=int, default=1,
                        help="Keep every Nth base graph; aggregate intervening speech events without loss")
    args = parser.parse_args()
    print(json.dumps(export_recording(args.recording, step_stride=args.step_stride), indent=2))


if __name__ == "__main__":
    main()
