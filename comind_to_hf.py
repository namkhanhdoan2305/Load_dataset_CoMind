"""Download selected CoMind modalities to mounted Drive or Hugging Face.

Uses comind_download.py for manifest parsing, resumable downloads and hashes.
HF authentication uses a separate notebook login or an existing HF_TOKEN.
"""

import argparse
import fnmatch
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time

import comind_download as source

# Configure before importing huggingface_hub or hf_xet.
os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")


GROUPS = {
    "leader_video": ["mp4s/leader_trimmed_sync.mp4"],
    "helper_video": ["mp4s/helper_trimmed_sync.mp4"],
    "leader_gaze": ["mps_leader_*/eye_gaze/*"],
    "leader_hands": ["mps_leader_*/hand_tracking/*"],
    "transcripts": ["transcripts/*"],
}
CALIBRATION = ["mps_leader_*/slam/online_calibration.jsonl"]


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


def make_plan(ids, base_url=source.DEFAULT_BASE_URL, calibration=False,
              allow_missing=False):
    """Read small manifests only. Never silently omit a failed recording."""
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Choose a nonempty list of unique recording IDs")
    patterns = [p for group in GROUPS.values() for p in group]
    if calibration:
        patterns += CALIBRATION
    plan = []
    for index, rid in enumerate(ids, 1):
        if not source.UUID_RE.fullmatch(rid):
            raise ValueError(f"Invalid recording ID: {rid!r}")
        man = retry(lambda: source.fetch_manifest(base_url, rid, 60))
        paths = [entry["path"] for entry in man["files"]]
        missing = [name for name, pats in GROUPS.items()
                   if not any(fnmatch.fnmatchcase(path, p)
                              for path in paths for p in pats)]
        if calibration and not any(fnmatch.fnmatchcase(path, p)
                                   for path in paths for p in CALIBRATION):
            missing.append("leader_calibration")
        if missing:
            print(f"MISSING {rid}: {', '.join(missing)}", flush=True)
            if not allow_missing:
                raise ValueError("Missing modalities. Inspect the manifest or set "
                                 "ALLOW_MISSING_MODALITIES=True explicitly")
        man["missing_modalities"] = missing
        source.filter_manifest_files([man], patterns, [])
        if not man["files"]:
            raise ValueError(f"No selected files for {rid}")
        plan.append(man)
        print(f"[{index}/{len(ids)}] {rid}: "
              f"{man['total_size_bytes'] / 1e9:.3f} GB, "
              f"{len(man['files'])} files", flush=True)
    annotations = retry(lambda: source.fetch_manifest(
        base_url, "annotations", 60, dest_subpath="annotations"))
    plan.insert(0, annotations)
    total = sum(man["total_size_bytes"] for man in plan)
    biggest = max(e["size"] for man in plan for e in man["files"])
    print(f"TOTAL {total / 1e9:.3f} GB ({total / 2**30:.3f} GiB)")
    print(f"Largest temporary file: {biggest / 1e9:.3f} GB")
    return plan


def remote_file(api, repo_id, path, revision=None):
    matches = retry(lambda: api.get_paths_info(
        repo_id=repo_id, paths=[path], repo_type="dataset", revision=revision))
    return next((item for item in matches if item.path == path
                 and hasattr(item, "size")), None)


def matches_remote(item, size, sha256=None, git_sha1=None):
    if item is None or item.size != size:
        return False
    lfs = getattr(item, "lfs", None)
    if lfs is not None:
        digest = lfs.get("sha256") if isinstance(lfs, dict) else lfs.sha256
        return sha256 is not None and digest == sha256
    return git_sha1 is not None and item.blob_id == git_sha1


def upload_bytes(api, repo_id, path, payload):
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    sha256 = hashlib.sha256(payload).hexdigest()
    git_sha1 = hashlib.sha1(
        f"blob {len(payload)}\0".encode() + payload).hexdigest()
    existing = remote_file(api, repo_id, path)
    if matches_remote(existing, len(payload), sha256, git_sha1):
        return
    commit = retry(lambda: api.upload_file(
        repo_id=repo_id, repo_type="dataset", path_in_repo=path,
        path_or_fileobj=io.BytesIO(payload), commit_message=f"Add {path}"))
    stored = remote_file(api, repo_id, path, revision=commit.oid)
    if not matches_remote(stored, len(payload), sha256, git_sha1):
        raise RuntimeError(f"Hub hash verification failed: {path}")


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


def save_to_directory(plan, target, base_url=source.DEFAULT_BASE_URL, max_hours=None):
    """Download into mounted Drive or another persistent directory, without HF."""
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    source.write_split_files(str(target))
    started = time.monotonic()
    status = {"state": "running", "verified_files": 0, "verified_bytes": 0,
              "completed_groups": [], "current_file": None,
              "total_files": sum(len(m["files"]) for m in plan),
              "total_bytes": sum(e["size"] for m in plan for e in m["files"]),
              "recordings": [m["url_id"] for m in plan if m["url_id"] != "annotations"]}
    report = target / "download-status.json"
    write_json_atomic(report, status)
    try:
        for man in plan:
            folder = target / man["dest_subpath"]
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "source-healthcheck.json").write_bytes(man["raw"])
            selected = {"source": base_url, "url_id": man["url_id"],
                        "hash_algorithm": man["hash_algorithm"], "files": man["files"],
                        "missing_modalities": man.get("missing_modalities", [])}
            write_json_atomic(folder / "selection.json", selected)
            for entry in man["files"]:
                if max_hours is not None and time.monotonic() - started >= max_hours * 3600:
                    status["state"] = "budget_reached"
                    write_json_atomic(report, status)
                    print("Time budget reached; rerun the same command to continue.")
                    return status
                dest = folder / entry["path"]
                status["current_file"] = str(dest.relative_to(target))
                write_json_atomic(report, status)
                print(f"DOWNLOAD / VERIFY {dest} ({entry['size'] / 1e9:.3f} GB)", flush=True)
                download_verified(man, entry, dest, base_url)
                status["verified_files"] += 1
                status["verified_bytes"] += entry["size"]
                status["elapsed_seconds"] = round(time.monotonic() - started, 1)
                write_json_atomic(report, status)
                print(f"VERIFIED {status['verified_files']}/{status['total_files']} files "
                      f"({status['verified_bytes'] / 1e9:.3f} GB)", flush=True)
            status["completed_groups"].append(man["url_id"])
            write_json_atomic(report, status)
    except BaseException as exc:
        status["state"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        status["error_type"] = type(exc).__name__
        write_json_atomic(report, status)
        raise
    status.update(state="complete", current_file=None)
    write_json_atomic(report, status)
    print(f"Completed {target}")
    return status


def transfer_file(api, repo_id, man, entry, staging, base_url):
    remote_path = f"{man['dest_subpath']}/{entry['path']}"
    key = hashlib.sha256(remote_path.encode()).hexdigest()
    dest = staging / key
    part = Path(str(dest) + ".part")
    expected_sha256 = entry["hash"] if man["hash_algorithm"] == "sha256" else None
    existing = remote_file(api, repo_id, remote_path)
    if matches_remote(existing, entry["size"], expected_sha256):
        dest.unlink(missing_ok=True)
        part.unlink(missing_ok=True)
        print(f"SKIP {remote_path}", flush=True)
        return

    # One unique temporary payload per source path, with resumable .part beside it.
    present = dest.stat().st_size if dest.exists() else 0
    partial = part.stat().st_size if part.exists() else 0
    needed = max(0, entry["size"] - max(present, partial)) + 2 * 2**30
    if shutil.disk_usage(staging).free < needed:
        raise RuntimeError(f"Insufficient temporary disk for {remote_path}")

    print(f"DOWNLOAD {remote_path} ({entry['size'] / 1e9:.3f} GB)", flush=True)
    download_verified(man, entry, dest, base_url)

    sha256 = source.file_digest(dest)
    git_hash = hashlib.sha1(f"blob {entry['size']}\0".encode())
    with dest.open("rb") as handle:
        for chunk in iter(lambda: handle.read(source.READ_CHUNK), b""):
            git_hash.update(chunk)
    # Recheck small non-LFS files with their git object ID before uploading.
    if not matches_remote(existing, entry["size"], sha256, git_hash.hexdigest()):
        print(f"UPLOAD {remote_path}", flush=True)
        commit = retry(lambda: api.upload_file(
            repo_id=repo_id, repo_type="dataset", path_in_repo=remote_path,
            path_or_fileobj=str(dest), commit_message=f"Add {remote_path}"))
        stored = remote_file(api, repo_id, remote_path, revision=commit.oid)
        if not matches_remote(stored, entry["size"], sha256, git_hash.hexdigest()):
            raise RuntimeError(f"Hub hash verification failed; kept {dest}")
    dest.unlink()
    print(f"VERIFIED {remote_path}; temporary file removed", flush=True)


def transfer(plan, repo_id, token=None, staging="/content/comind-transfer",
             private=True, base_url=source.DEFAULT_BASE_URL):
    from huggingface_hub import HfApi

    if "/" not in repo_id or repo_id.startswith("YOUR_"):
        raise ValueError("Set REPO_ID to your Hugging Face namespace/repository")
    api = HfApi(token=token)
    api.whoami()
    retry(lambda: api.create_repo(repo_id=repo_id, repo_type="dataset",
                                 private=private, exist_ok=True))
    info = retry(lambda: api.repo_info(repo_id=repo_id, repo_type="dataset"))
    if info.private != private:
        raise ValueError("Existing repository visibility differs from PRIVATE_REPO. "
                         "Choose another repository or match its visibility")
    staging = Path(staging)
    staging.mkdir(parents=True, exist_ok=True)

    card = """---
license: cc-by-4.0
tags:
- comind
- human-robot-interaction
---
# Selected CoMind modalities

Source: https://comind.ethz.ch/ — CoMind authors, ETH Zurich and collaborators,
CoMind: Understanding Collaborative Human Activity from Multiple Minds and Views,
ECCV 2026. Original license: https://creativecommons.org/licenses/by/4.0/

This is a selected mirror. Selected payload files are byte-for-byte copies.
Original source manifests are retained as provenance and list additional files
outside this selection. `selection.json` in each directory lists mirrored files.
Official train/test ID lists are retained under `split/`; the selection metadata
identifies the recordings actually present. Leader/helper are the dataset's
human roles. Helper video is a robot-view proxy with partial observations.
Official annotations cover joint attention, interaction anticipation and
handover. Dense human belief and recipe-step labels require further work.
"""
    # Preserve an existing user-written dataset card.
    if remote_file(api, repo_id, "README.md") is None:
        upload_bytes(api, repo_id, "README.md", card)
    for name, ids in (("train", source.TRAIN_IDS), ("test", source.TEST_IDS)):
        upload_bytes(api, repo_id, f"split/{name}.txt", "\n".join(ids) + "\n")

    for man in plan:
        prefix = man["dest_subpath"]
        upload_bytes(api, repo_id, f"{prefix}/source-healthcheck.json", man["raw"])
        selected = {"source": base_url, "url_id": man["url_id"],
                    "hash_algorithm": man["hash_algorithm"], "files": man["files"]}
        upload_bytes(api, repo_id, f"{prefix}/selection.json",
                     json.dumps(selected, indent=2))
        for entry in man["files"]:
            transfer_file(api, repo_id, man, entry, staging, base_url)
    print(f"Completed https://huggingface.co/datasets/{repo_id}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--repo-id", help="HF namespace/dataset repository")
    destination.add_argument("--drive-dir", help="Persistent target, e.g. /content/drive/MyDrive/CoMind")
    parser.add_argument("--split", choices=["train", "test", "all"], default="train")
    parser.add_argument("--limit", type=int, help="Use the first N selected recording IDs")
    parser.add_argument("--ids-file", help="Optional text file of recording UUIDs")
    parser.add_argument("--dry-run", action="store_true", help="Read manifests and report sizes only")
    parser.add_argument("--public", action="store_true", help="Create a public HF repository")
    parser.add_argument("--leader-calibration", action="store_true", help="Also select leader online calibration")
    parser.add_argument("--allow-missing-modalities", action="store_true")
    parser.add_argument("--staging", default="/content/comind-transfer")
    parser.add_argument("--base-url", default=source.DEFAULT_BASE_URL)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not args.dry_run and not (args.repo_id or args.drive_dir):
        parser.error("Choose --repo-id or --drive-dir")
    if args.public and not args.repo_id:
        parser.error("--public requires --repo-id")
    if not args.dry_run and args.repo_id:
        from huggingface_hub import get_token
        if not get_token():
            parser.error("Log in to HF first as described in COLAB_HF_GUIDE.md")
    splits = {"train": source.TRAIN_IDS, "test": source.TEST_IDS,
              "all": source.TRAIN_IDS + source.TEST_IDS}
    ids = source.read_id_list(args.ids_file) if args.ids_file else splits[args.split]
    if args.limit is not None:
        ids = ids[:args.limit]
    plan = make_plan(ids, base_url=args.base_url,
                     calibration=args.leader_calibration,
                     allow_missing=args.allow_missing_modalities)
    if not args.dry_run and args.drive_dir:
        save_to_directory(plan, args.drive_dir, base_url=args.base_url)
    elif not args.dry_run:
        transfer(plan, repo_id=args.repo_id, staging=args.staging,
                 private=not args.public, base_url=args.base_url)


if __name__ == "__main__":
    main()
