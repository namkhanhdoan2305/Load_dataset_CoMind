"""Extract resumable 5-FPS REACT++/YOLO graph streams on Colab."""

import argparse
import csv
import errno
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

os.environ.setdefault("TQDM_DISABLE", "1")

import comind_source as source
import comind_data as download
from comind_data import select_sessions
from reactpp_models import SETTINGS
from time_contract import clock_id, graph_time_us, microseconds


DEFAULT_SETTINGS = SETTINGS


def fingerprint(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def write_sync_report(target, rid):
    """Report source clocks explicitly; never guess sensor offset from its first row."""
    folder = Path(target) / "recordings" / rid
    mapping_path = folder / "sensor-clock.json"
    mapping = json.loads(mapping_path.read_text()) if mapping_path.exists() else {}
    verified = mapping.get("status") == "verified_metadata_anchors"
    clocks = {}
    for group, name in [("leader_gaze", "general_eye_gaze.csv"),
                        ("leader_hands", "hand_tracking_results.csv")]:
        path = folder / group / name
        if path.exists():
            with path.open(newline="") as handle:
                first = next(csv.DictReader(handle), {})
            clocks[group] = {"path": str(path.relative_to(target)),
                             "timestamp_column": "tracking_timestamp_us",
                             "clock_id": f"comind:{rid}:leader_device_time",
                             "first_source_timestamp_us": first.get("tracking_timestamp_us"),
                             "mapping_to_mp4": "verified_metadata_anchors" if verified else "unresolved"}
        else:
            clocks[group] = {"status": "missing_source_modality"}
    report = {"recording_id": rid, "video_clock": "published synced MP4 relative seconds",
              "clock_id": clock_id(rid), "timestamp_unit": "microseconds",
              "graph_pairing": {"status": "not_checked"},
              "graph_grid_seconds": 0.2,
              "object_ids": "view-local; cross-view entity matching remains unresolved",
              "transcript_clock": "source start/end seconds in trimmed_sync transcript",
              "annotations": "explicit source start/end times normalized at export; IDs retained; benchmark targets kept separate",
              "sensor_clocks": clocks, "all_modalities_aligned": False}
    report["sensor_clock_mapping"] = {"path": "sensor-clock.json", "verified": verified}
    download.write_json_atomic(folder / "sync-status.json", report)


def link_graph_views(target, rid):
    """Link the two graph streams by their MP4 timestamp, without fabricating sensor joins."""
    folder = Path(target) / "recordings" / rid
    temporary = folder / "graphs" / "timeline.jsonl.gz.tmp"
    count = 0
    max_delta_us = 0
    pts_verified = True
    mapping_path = folder / "sensor-clock.json"
    mapping_verified = (mapping_path.exists() and
                        json.loads(mapping_path.read_text()).get("status") == "verified_metadata_anchors")
    words = []
    untimed_words = 0
    for stream in ("leader", "helper"):
        transcript = folder / "transcripts" / f"{stream}_trimmed_sync_transcript.json"
        if not transcript.exists():
            continue
        for segment_index, segment in enumerate(json.loads(transcript.read_text()).get("segments", [])):
            for word_index, word in enumerate(segment.get("words", [])):
                if word.get("start") is not None and word.get("end") is not None:
                    start_us, end_us = microseconds(word["start"]), microseconds(word["end"])
                    if end_us < start_us:
                        raise ValueError(f"Invalid word timestamps in {transcript}")
                    words.append(dict(word, audio_stream=stream, start_us=start_us, available_us=end_us,
                                      segment_index=segment_index, word_index=word_index,
                                      clock_id=clock_id(rid)))
                else:
                    untimed_words += 1
    words.sort(key=lambda word: word["available_us"])
    next_word = 0
    with gzip.open(folder / "graphs/leader/graphs.jsonl.gz", "rt") as leader, \
            gzip.open(folder / "graphs/helper/graphs.jsonl.gz", "rt") as helper, \
            gzip.open(temporary, "wt") as output:
        while True:
            left, right = leader.readline(), helper.readline()
            if not left and not right:
                break
            if not left or not right:
                raise ValueError("Paired graph streams have different lengths; inspect source synchronization")
            a, b = json.loads(left), json.loads(right)
            a_us, b_us = graph_time_us(a, rid), graph_time_us(b, rid)
            delta_us = abs(a.get("source_pts_us", a_us) - b.get("source_pts_us", b_us))
            max_delta_us = max(max_delta_us, delta_us)
            pts_verified = pts_verified and a.get("pts_verified", False) and b.get("pts_verified", False)
            if a_us != b_us or delta_us > 1000:
                raise ValueError("Paired graph timestamps disagree; keep videos for inspection")
            expected_us = round(count * 1_000_000 / DEFAULT_SETTINGS["graph_fps"])
            if a_us != expected_us:
                raise ValueError(f"Missing graph grid step {count}: expected {expected_us}us, got {a_us}us")
            elapsed_words = []
            while next_word < len(words) and words[next_word]["available_us"] <= a_us:
                elapsed_words.append(words[next_word])
                next_word += 1
            output.write(json.dumps({"recording_id": rid, "time_s": a["time_s"],
                                     "time_us": a_us, "clock_id": clock_id(rid),
                                     "pair_delta_us": delta_us,
                                     "leader_row": count, "helper_row": count,
                                     "leader_sample_index": a["sample_index"],
                                     "helper_sample_index": b["sample_index"],
                                     "completed_transcript_words": elapsed_words,
                                     "sensor_alignment": "metadata_clock_mapped" if mapping_verified else "unresolved"}) + "\n")
            count += 1
    for view in ("leader", "helper"):
        view_folder = folder / "graphs" / view
        progress = json.loads((view_folder / "progress.json").read_text())
        expected = verify_complete_graph_count(progress, DEFAULT_SETTINGS, view_folder)
        if not progress["complete"] or count != expected:
            raise ValueError(f"Incomplete paired graph coverage for {view}: {count}/{expected} rows")
    os.replace(temporary, folder / "graphs/timeline.jsonl.gz")
    sync_path = folder / "sync-status.json"
    if sync_path.exists():
        sync = json.loads(sync_path.read_text())
        sync["graph_pairing"] = {"status": "paired", "pairs": count,
                                 "max_delta_us": max_delta_us, "tolerance_us": 1000,
                                 "decoder_pts_verified": bool(pts_verified)}
        sync["transcript_words"] = {"timed_words": len(words), "untimed_words": untimed_words,
                                    "emitted_words": next_word, "unemitted_timed_words": len(words) - next_word,
                                    "clock_alignment": "as_published_trimmed_sync_transcript",
                                    "physical_audio_video_alignment_verified": False}
        download.write_json_atomic(sync_path, sync)
    return count


def verify_rows(rows, previous_time=-1):
    for row in rows:
        timestamp = row["time_s"]
        if not math.isfinite(timestamp) or timestamp <= previous_time:
            raise ValueError("Graph timestamps must be finite and strictly increasing")
        previous_time = timestamp
        ids = {node["id"] for node in row["nodes"]}
        if len(ids) != len(row["nodes"]):
            raise ValueError("Duplicate node IDs")
        for edge in row["edges"]:
            if edge["subject"] not in ids or edge["object"] not in ids:
                raise ValueError("Relation references a missing node")
    return previous_time


def verify_complete_graph_count(progress, settings, folder):
    samples = math.floor((progress["source_frame_count"] - 1) * settings["tracking_fps"]
                         / progress["source_fps"]) + 1
    stride = settings["tracking_fps"] // settings["graph_fps"]
    expected = (samples - 1) // stride + 1
    if progress["graph_count"] != expected:
        raise ValueError(f"Incomplete graph coverage in {folder}: {progress['graph_count']}/{expected} rows")
    return expected


def recover_stream(folder, identity):
    """Verify committed bytes; discard only an interrupted, uncommitted append."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / "progress.json"
    graph = folder / "graphs.jsonl.gz"
    if checkpoint.exists():
        progress = json.loads(checkpoint.read_text())
        if progress["identity"] != identity:
            raise ValueError(f"Graph configuration/source changed: use a new output folder ({folder})")
    else:
        progress = {"identity": identity, "committed_bytes": 0,
                    "sha256": hashlib.sha256(b"").hexdigest(), "next_chunk": 0,
                    "next_id": 1, "carry": [], "last_time_s": -1,
                    "graph_count": 0, "node_count": 0, "edge_count": 0,
                    "complete": False}
    committed = progress["committed_bytes"]
    if not graph.exists():
        if committed:
            raise ValueError(f"Missing committed graph stream: {graph}")
        graph.touch()
    if graph.stat().st_size < committed:
        raise ValueError(f"Truncated committed graph stream: {graph}")
    digest = hashlib.sha256()
    with graph.open("rb") as handle:
        remaining = committed
        while remaining:
            block = handle.read(min(1024 * 1024, remaining))
            digest.update(block)
            remaining -= len(block)
    if digest.hexdigest() != progress["sha256"]:
        raise ValueError(f"Committed graph hash mismatch: {graph}")
    if graph.stat().st_size > committed:
        with graph.open("r+b") as handle:
            handle.truncate(committed)
    progress["_hash_state"] = digest
    return progress


def commit_chunk(folder, progress, rows, carry, next_id):
    rows = [row for row in rows if row["time_s"] > progress["last_time_s"]]
    last_time = verify_rows(rows, progress["last_time_s"])
    body = "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows).encode()
    graph = Path(folder) / "graphs.jsonl.gz"
    compressed = gzip.compress(body, mtime=0)
    with graph.open("ab") as handle:
        handle.write(compressed)
        handle.flush()
        os.fsync(handle.fileno())
    with graph.open("rb") as handle:
        handle.seek(progress["committed_bytes"])
        if handle.read() != compressed:
            raise RuntimeError(f"Graph append verification failed: {graph}")
    digest = progress["_hash_state"].copy()
    digest.update(compressed)
    updated = dict(progress)
    updated.update(committed_bytes=graph.stat().st_size,
                   sha256=digest.hexdigest(), _hash_state=digest,
                   next_chunk=progress["next_chunk"] + 1,
                   next_id=next_id, carry=carry, last_time_s=last_time,
                   graph_count=progress["graph_count"] + len(rows),
                   node_count=progress["node_count"] + sum(len(r["nodes"]) for r in rows),
                   edge_count=progress["edge_count"] + sum(len(r["edges"]) for r in rows))
    download.write_json_atomic(Path(folder) / "progress.json",
                               {k: v for k, v in updated.items() if not k.startswith("_")})
    return updated


class VideoReader:
    def __init__(self, path, tracking_fps):
        import cv2
        self.cv2 = cv2
        self.capture = cv2.VideoCapture(str(path))
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not self.capture.isOpened() or not math.isfinite(self.fps) or self.fps < tracking_fps:
            raise ValueError(f"Invalid video metadata or tracking FPS exceeds source FPS: {path}")
        if self.frames <= 0:
            raise ValueError(f"Empty video: {path}")
        self.tracking_fps = tracking_fps
        self.samples = math.floor((self.frames - 1) * tracking_fps / self.fps) + 1

    def decode(self, start, end, folder, max_side):
        times = []
        cursor = math.floor(start * self.fps / self.tracking_fps + 1e-9)
        self.capture.set(self.cv2.CAP_PROP_POS_FRAMES, cursor)
        for local_index, sample_index in enumerate(range(start, end + 1)):
            source_index = math.floor(sample_index * self.fps / self.tracking_fps + 1e-9)
            while cursor < source_index:
                if not self.capture.grab():
                    raise RuntimeError(f"Failed to skip source frame {cursor}")
                cursor += 1
            ok, image = self.capture.read()
            if not ok:
                raise RuntimeError(f"Failed to decode source frame {source_index}")
            cursor += 1
            pts_s = self.capture.get(self.cv2.CAP_PROP_POS_MSEC) / 1000
            expected_s = source_index / self.fps
            if not math.isfinite(pts_s) or pts_s < 0 or abs(pts_s - expected_s) > 0.001:
                raise RuntimeError(
                    f"Decoder timestamp disagrees with CFR timeline at frame {source_index}: "
                    f"PTS={pts_s}s, frame/fps={expected_s}s. Keep video for timestamp inspection")
            original_h, original_w = image.shape[:2]
            scale = min(1, max_side / max(original_h, original_w))
            if scale < 1:
                image = self.cv2.resize(image, (round(original_w * scale), round(original_h * scale)))
            if not self.cv2.imwrite(str(Path(folder) / f"{local_index:06d}.jpg"), image):
                raise RuntimeError("Failed to save a temporary frame")
            grid_time_us = round(sample_index * 1_000_000 / self.tracking_fps)
            if microseconds(pts_s) > grid_time_us + 1000:
                raise RuntimeError("Decoded frame would expose a future image on the requested graph grid")
            times.append({"sample_index": sample_index, "source_frame_index": source_index,
                          "time_s": grid_time_us / 1_000_000, "time_us": grid_time_us,
                          "source_pts_us": microseconds(pts_s), "pts_verified": True,
                          "source_width": original_w, "source_height": original_h})
        return times

    def close(self):
        self.capture.release()


def extract_view(video, folder, settings, video_hash, view, model, deadline,
                 reader_class=VideoReader):
    identity = fingerprint({"settings": settings, "video_hash": video_hash, "view": view})
    progress = recover_stream(folder, identity)
    if progress["complete"]:
        verify_complete_graph_count(progress, settings, folder)
        return progress
    reader = reader_class(video, settings["tracking_fps"])
    step = settings["tracking_fps"] * settings["chunk_seconds"]
    chunks = max(1, math.ceil((reader.samples - 1) / step))
    print(f"{view}: source {reader.frames} frames @ {reader.fps:.3f} FPS, "
          f"duration approximately {reader.frames / reader.fps:.1f}s, "
          f"{chunks} chunks, resuming chunk {progress['next_chunk'] + 1}", flush=True)
    stride = settings["tracking_fps"] // settings["graph_fps"]
    try:
        committed_chunks = progress["next_chunk"]
        if committed_chunks:
            last_sample = min(committed_chunks * step, reader.samples - 1)
            if committed_chunks > chunks or progress["graph_count"] != last_sample // stride + 1:
                raise ValueError(f"Incomplete graph coverage in saved checkpoint: {folder}")
        while progress["next_chunk"] < chunks:
            if time.monotonic() >= deadline:
                return progress
            index = progress["next_chunk"]
            start, end = index * step, min((index + 1) * step, reader.samples - 1)
            chunk_started = time.monotonic()
            with tempfile.TemporaryDirectory(prefix="comind-frames-") as temporary:
                print(f"{view}: chunk {index + 1}/{chunks} decoding source frames", flush=True)
                times = reader.decode(start, end, temporary, settings["max_side"])
                print(f"{view}: chunk {index + 1}/{chunks} decoded {len(times)} samples "
                      f"in {time.monotonic() - chunk_started:.1f}s", flush=True)
                rows, carry, next_id = model.chunk(
                    temporary, times, view, progress["carry"], progress["next_id"], stride)
                if index:
                    rows = [row for row in rows if row["sample_index"] != start]
                expected = [i for i in range(start + bool(index), end + 1) if i % stride == 0]
                actual = [row["sample_index"] for row in rows]
                if actual != expected:
                    raise ValueError(f"{view} chunk {index + 1}: expected graph samples {expected}, got {actual}")
                for row in rows:
                    sample = next(t for t in times if t["sample_index"] == row["sample_index"])
                    row.update(recording_clock="synced_mp4_relative_seconds",
                               recording_id=Path(folder).parents[1].name,
                               time_s=sample["time_s"], time_us=microseconds(sample["time_s"]),
                               source_pts_us=sample.get("source_pts_us", microseconds(sample["time_s"])),
                               clock_id=clock_id(Path(folder).parents[1].name),
                               pts_verified=sample.get("pts_verified", False),
                               sensor_alignment="external_mapping; see recording sensor-clock.json",
                               source_video_sha256=video_hash,
                               source_width=sample["source_width"], source_height=sample["source_height"])
                progress = commit_chunk(folder, progress, rows, carry, next_id)
            print(f"{view}: chunk {index + 1}/{chunks}, "
                  f"{progress['graph_count']} graphs, {progress['node_count']} total nodes, "
                  f"{progress['edge_count']} total edges, last t={progress['last_time_s']:.2f}s, "
                  f"chunk took {time.monotonic() - chunk_started:.1f}s", flush=True)
        if progress["node_count"] == 0 or progress["edge_count"] == 0:
            raise RuntimeError(f"No usable object/relation output in {folder}; keep source video for inspection")
        progress.update(source_fps=reader.fps, source_frame_count=reader.frames,
                        timestamp_basis="Canonical session grid; source_pts_us from decoder, CFR validated")
        progress["expected_graph_count"] = verify_complete_graph_count(progress, settings, folder)
        progress["complete"] = True
        if source.file_digest(Path(folder) / "graphs.jsonl.gz") != progress["sha256"]:
            raise RuntimeError(f"Final graph stream hash verification failed: {folder}")
        download.write_json_atomic(Path(folder) / "progress.json",
                                   {k: v for k, v in progress.items() if not k.startswith("_")})
        return progress
    finally:
        reader.close()


def resolved_settings(target):
    path = Path(target) / "sgg-settings.json"
    if path.exists():
        settings = json.loads(path.read_text())
        if settings != DEFAULT_SETTINGS:
            raise ValueError("Existing SGG settings differ (possibly old DINO/SAM2 graphs). "
                             "Use a separate target such as /content/drive/MyDrive/CoMind_REACTpp")
        return settings
    settings = json.loads(json.dumps(DEFAULT_SETTINGS))
    download.write_json_atomic(path, settings)
    return settings


def write_batch_status(path, status):
    resolved = set(status["completed"]) | set(status["failed"])
    status["pending"] = [rid for rid in status["recordings"] if rid not in resolved]
    download.write_json_atomic(path, status)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", default="/content/drive/MyDrive/CoMind_REACTpp")
    parser.add_argument("--source-dir", default="/content/drive/MyDrive/CoMind",
                        help="Read-only reuse of previously downloaded videos; never moves/deletes those files")
    parser.add_argument("--work-dir", default="/content/comind-sgg-work")
    parser.add_argument("--max-hours", type=float, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-videos", action="store_true", help="Keep temporary downloaded videos after verification")
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_hours) or args.max_hours <= 0:
        parser.error("--max-hours must be positive and finite")
    try:
        ids = select_sessions(args.sessions, args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    target, work = Path(args.target).resolve(), Path(args.work_dir).resolve()
    if not args.dry_run:
        if target.is_relative_to(Path("/content/drive")) and not os.path.ismount("/content/drive"):
            parser.error("Mount Google Drive first")
        if work == target or work.is_relative_to(target) or target.is_relative_to(work):
            parser.error("Use separate temporary work and persistent target directories")
        try:
            import onnxruntime as ort
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                parser.error("Choose a T4 runtime and install the GPU ONNX Runtime in requirements.txt")
        except ImportError:
            parser.error("Install requirements.txt in the Colab GPU runtime first")
    if args.dry_run:
        download.make_plan(ids, allow_missing=True, show_files=True)
        print("Settings: REACT++ YOLO12m ONNX + bbox tracker, graph/tracking 5 FPS, 4-second chunks")
        return
    deadline = time.monotonic() + args.max_hours * 3600
    target.mkdir(parents=True, exist_ok=True)
    source.write_split_files(str(target))
    work.mkdir(parents=True, exist_ok=True)
    status = {"recordings": ids, "source_split": "train", "seed": args.seed,
              "completed": [], "failed": {}, "needs_review": {}, "state": "running", "current": None}
    report = target / "sgg-status.json"
    write_batch_status(report, status)
    source_root = Path(args.source_dir).resolve()
    try:
        plan = download.make_plan(ids, allow_missing=True, failures=status["failed"])
        if time.monotonic() >= deadline:
            status["state"] = "budget_reached"
            write_batch_status(report, status)
            return
        settings = resolved_settings(target)
        from reactpp_models import Perception
        model = Perception(settings)  # Check models/runtime before any large payload transfer.
        download.write_json_atomic(target / "reactpp-model-info.json", model.info)
        video_paths = {"leader": download.GROUPS["leader_video"][0],
                       "helper": download.GROUPS["helper_video"][0]}
        for man in plan:
            rid = man["url_id"]
            is_annotation = rid == "annotations"
            status["current"] = {"recording_id": rid, "view": None, "stage": "source_files"}
            write_batch_status(report, status)
            try:
                entries = {e["path"]: e for e in man["files"]}
                metadata_folder = target / ("annotations" if is_annotation else f"metadata/{man['url_id']}")
                metadata_folder.mkdir(parents=True, exist_ok=True)
                (metadata_folder / "source-healthcheck.json").write_bytes(man["raw"])
                download.write_json_atomic(metadata_folder / "selection.json", {
                    "files": man["files"], "missing_modalities": man.get("missing_modalities", []),
                    "missing_files": man.get("missing_files", [])})
                for entry in man["files"]:
                    if entry["path"] in video_paths.values():
                        continue
                    if time.monotonic() >= deadline:
                        status["state"] = "budget_reached"
                        write_batch_status(report, status)
                        return
                    dest = download.grouped_destination(target, man, entry)
                    download.adopt_existing(target / man["dest_subpath"] / entry["path"],
                                            dest, entry, man["hash_algorithm"])
                    download.download_verified(man, entry, dest, source.DEFAULT_BASE_URL)
                if is_annotation:
                    download.split_annotations(plan, target)
                    continue
                if not set(video_paths.values()).issubset(entries):
                    raise ValueError(f"Missing required paired videos: {man['url_id']}")
                status["current"]["stage"] = "sensor_clock"
                write_batch_status(report, status)
                from sensor_sync import prepare_clock
                clock = prepare_clock(target / "recordings" / rid, man, allow_unresolved=True)
                write_sync_report(target, rid)
                for view, path in video_paths.items():
                    if time.monotonic() >= deadline:
                        status["state"] = "budget_reached"
                        write_batch_status(report, status)
                        return
                    status["current"] = {"recording_id": rid, "view": view, "stage": "graph"}
                    write_batch_status(report, status)
                    entry = entries[path]
                    output = target / "recordings" / man["url_id"] / "graphs" / view
                    identity = fingerprint({"settings": settings, "video_hash": entry["hash"], "view": view})
                    progress = recover_stream(output, identity)
                    if progress["complete"]:
                        verify_complete_graph_count(progress, settings, output)
                        print(f"VERIFIED existing graph {output}", flush=True)
                        continue
                    existing = download.grouped_destination(target, man, entry)
                    old = target / man["dest_subpath"] / path
                    prior_grouped = download.grouped_destination(source_root, man, entry)
                    prior_old = source_root / man["dest_subpath"] / path
                    video = work / man["url_id"] / path
                    persistent = next((p for p in (existing, old, prior_grouped, prior_old) if p.is_file()
                                       and p.stat().st_size == entry["size"]
                                       and source.file_digest(p, man["hash_algorithm"]) == entry["hash"]), None)
                    if persistent is not None:
                        video = persistent
                    else:
                        free = source.free_space_bytes(work)
                        if free is not None and free < entry["size"] + 2 * 2**30:
                            raise OSError(errno.ENOSPC, "Insufficient temporary disk", str(work))
                        download.download_verified(man, entry, video, source.DEFAULT_BASE_URL)
                    progress = extract_view(video, output, settings, entry["hash"], view, model, deadline)
                    if not progress["complete"]:
                        status["state"] = "budget_reached"
                        write_batch_status(report, status)
                        return
                status["current"] = {"recording_id": rid, "view": None, "stage": "export"}
                write_batch_status(report, status)
                link_graph_views(target, rid)
                from belief_data import export_recording
                exported = export_recording(target / "recordings" / rid)
                if not exported["all_modalities_aligned"]:
                    status["needs_review"][rid] = {
                        "timed_model_inputs_aligned": exported["timed_model_inputs_aligned"],
                        "sensor_clock_error": clock.get("error"),
                        "missing_files": man.get("missing_files", []),
                    }
                # Both view streams and their timestamp pairing passed checks before cleanup.
                for view, path in video_paths.items():
                    candidate = work / man["url_id"] / path
                    if candidate.is_file() and not args.keep_videos:
                        candidate.unlink()
                status["completed"].append(rid)
            except Exception as exc:
                if (is_annotation or isinstance(exc, (MemoryError, PermissionError))
                        or isinstance(exc, OSError) and exc.errno == errno.ENOSPC):
                    raise
                status["failed"][rid] = {
                    "stage": status["current"]["stage"], "view": status["current"]["view"],
                    "error_type": type(exc).__name__, "error": str(exc),
                }
                print(f"FAILED {rid} ({status['current']['stage']}): {exc}", flush=True)
            write_batch_status(report, status)
        issues = status["failed"] or status["needs_review"] or len(status["completed"]) != len(ids)
        status.update(state="complete_with_issues" if issues else "complete", current=None)
        write_batch_status(report, status)
    except BaseException as exc:
        status.update(state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error_type=type(exc).__name__, error=str(exc))
        write_batch_status(report, status)
        raise
    print(f"Completed {len(status['completed'])}/{len(ids)} recordings. "
          f"Needs review: {len(status['needs_review'])}. Failed: {len(status['failed'])}. "
          f"See {report}", flush=True)


if __name__ == "__main__":
    main()
