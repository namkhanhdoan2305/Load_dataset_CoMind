"""Normalize benchmark labels by explicit source times; keep targets outside inputs."""

import json
from pathlib import Path

import comind_source as source
from comind_data import write_json_atomic
from time_contract import clock_id, microseconds


KINDS = {"joint_attention": "joint_attention.json", "interaction_anticipation": "interaction_anticipation.json",
         "handover": "handover.json"}


class AnnotationTimeline:
    def __init__(self, recording, source_fps=None, video_duration_us=None):
        self.recording = Path(recording)
        self.events, self.untimed, self.missing, self.issues, self.hashes = [], [], [], [], {}
        self.previous_us, self.start_cursor, self.end_cursor = -1, 0, 0
        self.active = set()
        for kind, filename in KINDS.items():
            path = self.recording / "annotations" / filename
            if not path.exists():
                self.missing.append(kind)
                continue
            self.hashes[kind] = source.file_digest(path)
            payload = json.loads(path.read_text())
            if payload.get("recording_id") != self.recording.name:
                raise ValueError(f"Wrong recording in benchmark annotation {path}")
            if not payload.get("recording_present_in_source", True):
                self.missing.append(kind)
            for event_id, label in payload.get("data", {}).items():
                identity = f"{kind}:{event_id}"
                if label is None:
                    self.untimed.append({"event_id": identity, "reason": "source_label_withheld_or_null"})
                    continue
                if not isinstance(label, dict) or "start_time" not in label or "end_time" not in label:
                    self.untimed.append({"event_id": identity, "reason": "missing_explicit_source_times"})
                    continue
                start, end = microseconds(label["start_time"]), microseconds(label["end_time"])
                if end < start:
                    raise ValueError(f"Annotation interval reversed: {identity}")
                if video_duration_us is not None and end > video_duration_us + 1000:
                    self.issues.append({"event_id": identity, "reason": "outside_video_duration"})
                for prefix, time_us in (("start", start), ("end", end)):
                    frame = label.get(f"{prefix}_frame")
                    if source_fps is not None and frame is not None:
                        if type(frame) is not int or frame < 0:
                            raise ValueError(f"Invalid annotation frame: {identity}")
                        if abs(microseconds(frame / source_fps) - time_us) > 1000:
                            self.issues.append({"event_id": identity, "reason": f"{prefix}_frame_time_mismatch"})
                self.events.append({"event_id": identity, "annotation_type": kind, "source_event_id": event_id,
                                    "start_us": start, "end_us": end, "clock_id": clock_id(self.recording.name),
                                    "source_file": filename, "label": label})
        self.events.sort(key=lambda e: (e["start_us"], e["event_id"]))
        self.ends = sorted(self.events, key=lambda e: (e["end_us"], e["event_id"]))
        self.source_fps = source_fps

    def advance(self, time_us):
        if time_us <= self.previous_us:
            raise ValueError("Annotation query times must increase")
        overlap, started, completed = set(self.active), [], []
        while self.start_cursor < len(self.events) and self.events[self.start_cursor]["start_us"] <= time_us:
            event = self.events[self.start_cursor]
            started.append(event["event_id"])
            self.active.add(event["event_id"])
            overlap.add(event["event_id"])
            self.start_cursor += 1
        while self.end_cursor < len(self.ends) and self.ends[self.end_cursor]["end_us"] <= time_us:
            event = self.ends[self.end_cursor]
            completed.append(event["event_id"])
            self.active.discard(event["event_id"])
            self.end_cursor += 1
        row = {"recording_id": self.recording.name, "clock_id": clock_id(self.recording.name),
               "time_us": time_us, "previous_time_us": self.previous_us,
               "started_event_ids": started, "completed_event_ids": completed,
               "active_event_ids": sorted(self.active), "interval_overlap_event_ids": sorted(overlap),
               "usage": "supervision_or_evaluation_only; never inject target labels/transcript_10s into model inputs"}
        self.previous_us = time_us
        return row

    def save(self):
        if not self.hashes:
            return
        write_json_atomic(self.recording / "annotations/timed_events.json", {
            "recording_id": self.recording.name, "clock_id": clock_id(self.recording.name),
            "events": self.events, "untimed_source_entries": self.untimed,
            "source_sha256": self.hashes,
            "usage": "Benchmark targets, not task progress or human-belief labels"})

    def report(self):
        return {"temporal_mapping_complete": not (self.missing or self.untimed or self.issues) and bool(self.hashes),
                "timed_events": len(self.events), "untimed_source_entries": len(self.untimed),
                "missing_categories": self.missing, "issues": self.issues,
                "frame_time_crosscheck_available": self.source_fps is not None,
                "events_starting_after_last_step": len(self.events) - self.start_cursor,
                "events_ending_after_last_step": len(self.ends) - self.end_cursor,
                "source_sha256": self.hashes, "task_progress_generated": False,
                "belief_labels_generated": False}
