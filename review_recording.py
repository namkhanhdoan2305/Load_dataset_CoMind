"""Read-only verification of one completed recording on Colab/Drive; no GPU or download."""

import argparse
import gzip
import json
from pathlib import Path

import comind_download as source
from comind_to_hf import GROUPS, ANNOTATIONS, grouped_destination
from time_contract import clock_id, graph_time_us


def review(recording, work_dir="/content/comind-sgg-work", source_dir="/content/drive/MyDrive/CoMind"):
    recording = Path(recording).resolve()
    root, rid = recording.parent.parent, recording.name
    errors, files, graph_hashes = [], {}, {}
    manifest_path = recording / "belief_inputs/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    raw_path = root / "metadata" / rid / "source-healthcheck.json"
    raw = json.loads(raw_path.read_text())
    expected = {entry["path"]: entry for entry in raw["files"]}
    for view in ("leader", "helper"):
        path = recording / "graphs" / view / "graphs.jsonl.gz"
        progress = json.loads((path.parent / "progress.json").read_text())
        digest = source.file_digest(path)
        if not progress["complete"] or digest != progress["sha256"] or digest != manifest["source_graph_sha256"][view]:
            errors.append(f"Incomplete/corrupt graph: {view}")
        graph_hashes[view] = digest
    for group, names in GROUPS.items():
        for name in names:
            entry = expected.get(name)
            if entry is None:
                files[name] = {"status": "missing_in_source"}
                errors.append(f"Missing source file: {name}")
                continue
            primary = grouped_destination(root, {"url_id": rid}, entry)
            candidates = [primary]
            if group.endswith("video"):
                candidates.extend([Path(work_dir) / rid / name,
                                   grouped_destination(Path(source_dir), {"url_id": rid}, entry),
                                   Path(source_dir) / rid / name])
            existing = next((p for p in candidates if p.is_file()), None)
            if existing is not None:
                valid = existing.stat().st_size == entry["size"] and source.file_digest(existing) == entry["hash"]
                files[name] = {"status": "source_hash_verified" if valid else "hash_mismatch", "path": str(existing)}
                if not valid:
                    errors.append(f"Source file hash mismatch: {name}")
            elif group.endswith("video"):
                view = "leader" if group == "leader_video" else "helper"
                with gzip.open(recording / "graphs" / view / "graphs.jsonl.gz", "rt") as handle:
                    matches = all(json.loads(line).get("source_video_sha256") == entry["hash"] for line in handle)
                files[name] = {"status": "graph_provenance_verified_raw_not_retained" if matches else "video_provenance_mismatch"}
                if not matches:
                    errors.append(f"Video graph provenance mismatch: {view}")
            else:
                files[name] = {"status": "missing_local_file"}
                errors.append(f"Missing local file: {name}")
    common = json.loads((root / "annotations/source-healthcheck.json").read_text())
    common_entries = {entry["path"]: entry for entry in common["files"]}
    for name in ANNOTATIONS:
        path = root / "annotations" / name
        valid = path.is_file() and source.file_digest(path) == common_entries[name]["hash"]
        files[f"annotations/{name}"] = {"status": "source_hash_verified" if valid else "missing_or_corrupt"}
        if not valid:
            errors.append(f"Annotation source missing/corrupt: {name}")
        else:
            names = {"dataset_joint_attention_consolidated.json": ("joint_attention", "joint_attention.json"),
                     "dataset_scoia_consolidated.json": ("interaction_anticipation", "interaction_anticipation.json"),
                     "dataset_handover_consolidated.json": ("handover", "handover.json")}
            kind, filtered_name = names[name]
            filtered_path = recording / "annotations" / filtered_name
            filtered = json.loads(filtered_path.read_text())
            original = json.loads(path.read_text())
            if (filtered.get("recording_id") != rid or filtered.get("data") != original["data"].get(rid, {})
                    or source.file_digest(filtered_path) != manifest["benchmark_annotation_timing"]["source_sha256"].get(kind)):
                errors.append(f"Wrong recording or changed filtered annotation: {filtered_name}")
    steps = recording / "belief_inputs/steps.jsonl.gz"
    if source.file_digest(steps) != manifest["steps_sha256"]:
        errors.append("Step export hash mismatch")
    seen, word_seen, previous, count = set(), set(), -1, 0
    with gzip.open(steps, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            timestamp = graph_time_us(row, rid)
            if timestamp <= previous:
                errors.append("Nonmonotonic step timestamps")
            for key in ("robot_observation_graph", "human_observation_graph"):
                if graph_time_us(row[key], rid) != timestamp:
                    errors.append("View/step clock mismatch")
            for key, emitted in (("new_speech_events", seen), ("new_speech_words", word_seen)):
                for event in row.get(key, []):
                    if event["available_us"] > timestamp or event["event_id"] in emitted or event["clock_id"] != clock_id(rid):
                        errors.append("Future/duplicate/wrong-clock speech event")
                    emitted.add(event["event_id"])
            for key in ("gaze_features", "hand_features"):
                feature = row.get(key)
                if feature and (feature["aligned_time_us"] != timestamp or feature["age_ns"] < 0
                                or feature["source_timestamp_us"] * 1000 + feature["age_ns"] != feature["reference_device_time_ns"]
                                or feature["clock_id"] != clock_id(rid)):
                    errors.append("Sensor timestamp/provenance mismatch")
            previous, count = timestamp, count + 1
    if count != manifest["steps"]:
        errors.append("Step count differs from manifest")
    for filename, hash_key in (("speech_events.json", "speech_events_sha256"), ("speech_words.json", "speech_words_sha256"),
                               ("benchmark_labels.jsonl.gz", "benchmark_labels_sha256"),
                               ("speech_alignment.json", "speech_alignment_sha256"),
                               ("sensor_alignment.json", "sensor_alignment_sha256")):
        if hash_key not in manifest or source.file_digest(steps.parent / filename) != manifest[hash_key]:
            errors.append(f"Missing/corrupt derived file: {filename}")
    if not manifest.get("all_modalities_aligned", False):
        errors.append("Six-group timestamp mapping is not complete; inspect manifest flags/reports")
    return {"recording_id": rid, "status": "metadata_and_file_checks_passed" if not errors else "needs_attention",
            "errors": sorted(set(errors)), "source_files": files, "steps_checked": count,
            "segments_emitted": len(seen), "words_emitted": len(word_seen),
            "tail_segments_not_emitted": manifest.get("speech_events_unemitted"),
            "tail_words_not_emitted": manifest.get("speech_words_unemitted"),
            "timed_model_inputs_aligned": manifest.get("timed_model_inputs_aligned", False),
            "benchmark_annotation_timing": manifest.get("benchmark_annotation_timing"),
            "physical_audio_video_alignment_verified": False,
            "note": "Video raw_not_retained relies on verified graph provenance, not a fresh check of deleted bytes. "
                    "Tail speech after the final graph remains in raw/event files; no backward timestamp reassignment."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording")
    args = parser.parse_args()
    try:
        result = review(args.recording)
    except (OSError, ValueError, KeyError) as exc:
        result = {"status": "needs_attention", "errors": [f"{type(exc).__name__}: {exc}"],
                  "hint": "Finish both graph streams, update code and run sensor_sync.py before reviewing this export"}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if not result["errors"] else 1)


if __name__ == "__main__":
    main()
