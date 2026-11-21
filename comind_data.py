"""Select exactly the required CoMind files and verify local downloads.

Shared by the REACT++ pipeline and CPU synchronization/audit tools.
"""

import fnmatch
import json
import os
from pathlib import Path
import random
import shutil
import time

import comind_source as source

GROUPS = {
    "leader_video": ["mp4s/leader_trimmed_sync.mp4"],
    "helper_video": ["mp4s/helper_trimmed_sync.mp4"],
    "leader_gaze": ["mps_leader_trimmed_vrs/eye_gaze/general_eye_gaze.csv"],
    "leader_hands": ["mps_leader_trimmed_vrs/hand_tracking/hand_tracking_results.csv"],
    "transcripts": ["transcripts/leader_trimmed_sync_transcript.json",
                    "transcripts/helper_trimmed_sync_transcript.json"],
}

ANNOTATIONS = ["dataset_joint_attention_consolidated.json",
               "dataset_scoia_consolidated.json", "dataset_handover_consolidated.json"]


def select_sessions(count=30, seed=42):
    if not 1 <= count <= len(source.TRAIN_IDS):
        raise ValueError(f"Choose between 1 and {len(source.TRAIN_IDS)} train sessions")
    ids = sorted(source.TRAIN_IDS)
    random.Random(seed).shuffle(ids)
    return ids[:count]


def retry(operation, attempts=4):
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:
            if attempt == attempts - 1:
                raise
            delay = min(2 ** (attempt + 1), 20)
            # Avoid printing HTTP request details that could contain credentials.
            print(f"{type(exc).__name__}; retrying in {delay}s", flush=True)
            time.sleep(delay)


def make_plan(ids, base_url=source.DEFAULT_BASE_URL, allow_missing=False, show_files=False, *, failures=None):
    """Read manifests and report recording-specific failures when a batch requests it."""
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Choose a nonempty list of unique recording IDs")
    patterns = [p for group in GROUPS.values() for p in group]
    plan = []
    for index, rid in enumerate(ids, 1):
        if not source.UUID_RE.fullmatch(rid):
            raise ValueError(f"Invalid recording ID: {rid!r}")
        try:
            man = retry(lambda: source.fetch_manifest(base_url, rid, 60))
            paths = [entry["path"] for entry in man["files"]]
            missing_files = [pattern for pats in GROUPS.values() for pattern in pats
                             if not any(fnmatch.fnmatchcase(path, pattern) for path in paths)]
            missing = [name for name, pats in GROUPS.items() if any(p in missing_files for p in pats)]
            if missing:
                print(f"MISSING {rid}: {', '.join(missing)}", flush=True)
                if not allow_missing:
                    raise ValueError("Missing modalities. Inspect the manifest or set "
                                     "allow_missing=True explicitly")
            man["missing_modalities"] = missing
            man["missing_files"] = missing_files
            source.filter_manifest_files([man], patterns, [])
            if not man["files"]:
                raise ValueError(f"No selected files for {rid}")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            if failures is None:
                raise
            failures[rid] = {"stage": "manifest", "error_type": type(exc).__name__, "error": str(exc)}
            print(f"FAILED {rid} manifest: {exc}", flush=True)
            continue
        plan.append(man)
        print(f"[{index}/{len(ids)}] {rid}: "
              f"{man['total_size_bytes'] / 1e9:.3f} GB, "
              f"{len(man['files'])} files", flush=True)
        if show_files:
            for entry in man["files"]:
                print(f"  {entry['path']} ({entry['size'] / 1e6:.3f} MB)")
    annotations = retry(lambda: source.fetch_manifest(
        base_url, "annotations", 60, dest_subpath="annotations"))
    available_annotations = {entry["path"] for entry in annotations["files"]}
    if not set(ANNOTATIONS).issubset(available_annotations):
        raise ValueError("Required consolidated annotation files are missing from source")
    source.filter_manifest_files([annotations], ANNOTATIONS, [])
    if show_files:
        for entry in annotations["files"]:
            print(f"  annotations/{entry['path']} ({entry['size'] / 1e6:.3f} MB)")
    plan.insert(0, annotations)
    total = sum(man["total_size_bytes"] for man in plan)
    biggest = max(e["size"] for man in plan for e in man["files"])
    print(f"TOTAL {total / 1e9:.3f} GB ({total / 2**30:.3f} GiB)")
    print(f"Largest temporary file: {biggest / 1e9:.3f} GB")
    return plan


def download_verified(man, entry, dest, base_url):
    part = Path(str(dest) + ".part")
    if (not dest.exists() and part.exists() and part.stat().st_size == entry["size"]
            and entry["hash"] is not None
            and source.file_digest(part, man["hash_algorithm"]) == entry["hash"]):
        os.replace(part, dest)
    url = source.build_url(base_url, man["url_id"], entry["path"])
    last_report = time.monotonic()

    def report_progress(delta, absolute):
        nonlocal last_report
        now = time.monotonic()
        if now - last_report >= 20:
            print(f"  {dest.name}: {absolute / 1e9:.3f}/{entry['size'] / 1e9:.3f} GB",
                  flush=True)
            last_report = now

    for attempt in range(2):
        source.download_file(url, str(dest), entry["size"],
                             report_progress, retries=4, timeout=60)
        digest = source.file_digest(dest, man["hash_algorithm"])
        if entry["hash"] is None or digest == entry["hash"]:
            return
        dest.unlink()
        if attempt == 1:
            raise RuntimeError(f"Source hash mismatch: {entry['path']}")


def write_json_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def grouped_destination(target, man, entry):
    if man["url_id"] == "annotations":
        return Path(target) / "annotations" / entry["path"]
    group = next(name for name, paths in GROUPS.items() if entry["path"] in paths)
    return Path(target) / "recordings" / man["url_id"] / group / Path(entry["path"]).name


def adopt_existing(old, dest, entry, algorithm):
    """Reorganize only a selected payload, checking the copy before removing old bytes."""
    if old == dest:
        return
    for suffix in ("", ".part"):
        previous = Path(str(old) + suffix)
        destination = Path(str(dest) + suffix)
        if not previous.is_file():
            continue
        size = previous.stat().st_size
        if size > entry["size"] or (not suffix and size != entry["size"]):
            continue
        digest = source.file_digest(previous, algorithm)
        if not suffix and entry["hash"] and digest != entry["hash"]:
            continue
        if destination.exists():
            if destination.stat().st_size == size and source.file_digest(destination, algorithm) == digest:
                previous.unlink()
                continue
            if suffix:
                continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        copying = Path(str(destination) + ".migrating")
        shutil.copyfile(previous, copying)
        if copying.stat().st_size != size or source.file_digest(copying, algorithm) != digest:
            raise RuntimeError(f"Reorganization verification failed: {previous}")
        os.replace(copying, destination)
        previous.unlink()
        print(f"REORGANIZED {previous} -> {destination}", flush=True)


def split_annotations(plan, target):
    """Retain official event IDs and values, including null test annotations."""
    names = {"dataset_joint_attention_consolidated.json": "joint_attention.json",
             "dataset_scoia_consolidated.json": "interaction_anticipation.json",
             "dataset_handover_consolidated.json": "handover.json"}
    for original, filename in names.items():
        payload = json.loads((Path(target) / "annotations" / original).read_text())
        if not isinstance(payload.get("data"), dict):
            raise ValueError(f"Unsupported annotation structure: {original}")
        for man in plan:
            rid = man["url_id"]
            if rid == "annotations":
                continue
            data = {"recording_id": rid, "source_file": original,
                    "dataset_version": payload.get("dataset_version"),
                    "file_timestamp": payload.get("file_timestamp"),
                    "recording_present_in_source": rid in payload["data"],
                    "data": payload["data"].get(rid, {})}
            write_json_atomic(Path(target) / "recordings" / rid / "annotations" / filename, data)
