"""Explicit per-recording MP4 clock, stored as integer microseconds."""

import math


def clock_id(recording_id):
    return f"comind:{recording_id}:synced_mp4"


def microseconds(seconds):
    seconds = float(seconds)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Timestamp must be finite and nonnegative")
    return round(seconds * 1_000_000)


def graph_time_us(row, recording_id):
    expected_clock = clock_id(recording_id)
    if row.get("recording_id", recording_id) != recording_id:
        raise ValueError("Graph belongs to another recording")
    if row.get("clock_id", expected_clock) != expected_clock:
        raise ValueError("Graph belongs to another recording clock")
    derived = microseconds(row["time_s"])
    recorded = row.get("time_us", derived)
    if not isinstance(recorded, int) or isinstance(recorded, bool) or recorded != derived:
        raise ValueError("Graph time_us and time_s disagree")
    return recorded
