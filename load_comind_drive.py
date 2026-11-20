"""Load a repeatable 30-session CoMind subset into mounted Google Drive."""

import argparse
import json
import math
import os
from pathlib import Path
import random

import comind_download as source
import comind_to_hf as downloader


def select_sessions(count=30, seed=42):
    if not 1 <= count <= len(source.TRAIN_IDS):
        raise ValueError(f"Choose between 1 and {len(source.TRAIN_IDS)} train sessions")
    ids = sorted(source.TRAIN_IDS)
    random.Random(seed).shuffle(ids)
    return ids[:count]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", default="/content/drive/MyDrive/CoMind")
    parser.add_argument("--dry-run", action="store_true", help="Read metadata and report size only")
    parser.add_argument("--max-hours", type=float, default=5,
                        help="Stop before the next file after this many hours; rerun to continue")
    parser.add_argument("--leader-calibration", action="store_true")
    parser.add_argument("--strict-modalities", action="store_true",
                        help="Stop if a session lacks any requested modality")
    args = parser.parse_args(argv)
    try:
        ids = select_sessions(args.sessions, args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    if not math.isfinite(args.max_hours) or args.max_hours <= 0:
        parser.error("--max-hours must be finite and positive")
    target = Path(args.target).expanduser().resolve()
    drive_root = Path("/content/drive")
    if not args.dry_run and target.is_relative_to(drive_root):
        if not os.path.ismount(drive_root) or not (drive_root / "MyDrive").is_dir():
            parser.error("Mount Drive first: from google.colab import drive; "
                         "drive.mount('/content/drive')")
    print(f"Selected {len(ids)} sessions from official TRAIN, seed={args.seed}.")
    print("Same seed preserves the first 30 when expanding to --sessions 50.")
    plan = downloader.make_plan(ids, calibration=args.leader_calibration,
                                allow_missing=not args.strict_modalities)
    for man in plan:
        missing_videos = {"leader_video", "helper_video"}.intersection(
            man.get("missing_modalities", []))
        if missing_videos:
            raise ValueError(f"{man['url_id']} lacks required videos: {sorted(missing_videos)}")
    missing = {man["url_id"]: man["missing_modalities"] for man in plan
               if man.get("missing_modalities")}
    print(f"Sessions missing a requested modality: {len(missing)}")
    if args.dry_run:
        print("Metadata only; no payload files downloaded.")
        return

    target.mkdir(parents=True, exist_ok=True)
    subset = target / "subsets" / f"sessions-{args.sessions}-seed-{args.seed}.json"
    description = {"official_source_split": "train", "seed": args.seed,
                   "recordings": ids, "leader_calibration": args.leader_calibration,
                   "missing_modalities": missing}
    if subset.exists() and json.loads(subset.read_text())["recordings"] != ids:
        parser.error(f"Recording IDs changed; inspect {subset} before overwriting")
    downloader.write_json_atomic(subset, description)
    downloader.save_to_directory(plan, target, max_hours=args.max_hours)


if __name__ == "__main__":
    main()
