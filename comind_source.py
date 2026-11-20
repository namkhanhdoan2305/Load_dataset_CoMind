"""CoMind source manifests, official splits and resumable verified-download primitives.

Adapted from CVG Group, ETH Zurich, CoMind downloader version 20260626-01.
Source: https://comind.ethz.ch/ . Dataset attribution: CC BY 4.0.
Only the library primitives used by the current pipeline are retained.
"""

import fnmatch
import hashlib
import json
import os
import re
import shutil
import socket
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
DEFAULT_BASE_URL = "https://comind.ethz.ch/dataset"
MANIFEST_NAME = "healthcheck.json"
RECORDINGS_SUBDIR = "recordings"
USER_AGENT = "comind-v1-download/1.0 (+https://comind.ethz.ch)"
DOWNLOAD_CHUNK = 4 * 1024 * 1024
READ_CHUNK = 8 * 1024 * 1024

UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

TRAIN_IDS = [
    "01d63ce5-f734-4647-93c8-ea4af07b8c7f",
    "0542e1ea-8c26-4f86-9ff1-b66adacb7639",
    "0ea30478-5d19-410b-9b96-ee8f4fb4dc15",
    "0eb5f8d2-5ed2-4a8d-927c-cadda0dfae9e",
    "13175f01-e72c-42d5-87e8-737633ecfe27",
    "1326e688-9e84-4d1a-aef7-a588ca2cc47d",
    "14aaebe8-0a7f-43f0-b7f3-78ac6127df75",
    "16b9e645-0f3b-4caf-bbdd-f89d4e3387b3",
    "199d1b52-2f0b-44bd-806c-d8c61e851433",
    "1df816f6-d987-4c00-bc32-22ab6f08aa18",
    "222861ac-b5d7-4f92-b682-d6c639ede7eb",
    "2fc0aa53-9070-4c86-81c2-41450253c74d",
    "3c254eb1-5a00-4e73-b72f-1fc1f1cd8167",
    "3ef7356e-3b75-4eab-add9-363b0cdc2b99",
    "43276420-701f-4731-b9ab-bebc7fd14994",
    "451bf296-ed93-4f29-9ced-81f3a4f8c0cf",
    "5533b1d7-838f-4b1d-9b54-a5bf6e8e3128",
    "571b195e-95dc-4686-a355-0b26a6ece873",
    "585b52a0-a019-44a0-b70f-531244971bcf",
    "5f8dc440-e2df-41a7-b144-14de3877b8fb",
    "62531cc1-b19e-452c-96b4-d1e60f077e3f",
    "63a789f1-32bf-4cfc-815b-0f4e93dcbfb3",
    "6473136b-6037-46be-a8b0-fb86b5997376",
    "665e4989-28c4-4d1a-84fe-a425ea193ebe",
    "68b97376-0dce-4dc3-93ae-01f3b3977b57",
    "69fed21a-8fb3-4757-a728-8b0dc3dce828",
    "73e68fb0-de34-4982-b0a3-6c5ae70442ba",
    "806c599c-ebb7-4bd8-9372-84582bdf0a4c",
    "833eac29-25e8-476d-b81c-f311cac0fd28",
    "845323e6-ca6d-4eeb-87f0-590b5ce08aa5",
    "8ee6c694-38be-4a90-ad5a-75ef4449e972",
    "968a90d3-2fba-49ca-a34c-ccfe6939b3c7",
    "9db1ea2e-9677-41f3-8bb6-3dc7d61bd962",
    "9fcdca26-9acd-4dec-a940-bb425080bc3a",
    "a20c8ae1-147d-4996-8cf8-69c599a710f5",
    "a268a350-c55f-4f75-832b-52750ca30bce",
    "a5081f80-f7cb-475e-9c8d-3b9567d69a02",
    "a9e48896-8f2f-431b-9ad9-ad4c6b37b58e",
    "ae5b5499-2913-4e10-85f0-1fec63a3791f",
    "b1446007-e7d8-4463-83cd-5a6c187ef027",
    "b4f68f2b-2460-48e4-bd6d-863a25a796f9",
    "bb657948-9db6-4f64-807b-8fd36d6bb00f",
    "c5fdc20b-d865-43d8-a5e8-52af272ac17e",
    "c807cf0e-c32e-4c57-9215-603047e8f471",
    "c8776f57-6f73-41a9-974e-1890cc2bb7c1",
    "cdd2114f-4bef-401e-803f-a8da2327bb37",
    "d053e943-5dd5-4833-a0c8-813de6c93a24",
    "d8d72aca-ecd2-485f-a725-f391bcc4ef87",
    "d90d7d88-24a0-48f1-8406-bae0d5ceac27",
    "da2e013e-e431-4788-9dd7-7bc436e9376b",
    "e1868637-8cfa-41d2-b6e1-8b2754383dbe",
    "e30c47bf-c150-4a1b-aa04-76a5c3a95c54",
    "eb784196-2ece-4c95-b430-91a20b985df2",
    "f867da74-9841-4261-8ae9-3193029e0568",
    "f87737d4-bf15-4242-8082-e6cb055867c4",
]

TEST_IDS = [
    "f0b49334-565a-4d30-a010-15d92e511c9a",
    "d1c6ae3a-3f7b-4193-9923-2e029d9513fd",
    "7b04b3ef-43ed-4c46-ab3d-0d3baa25352e",
    "58dd70af-0c19-486e-9bdc-24b5c22b49e6",
    "63f25fa8-4573-48da-84e4-4cd5ea1c2701",
    "f6968635-3d7c-4784-8b45-1354f7a1d68f",
    "0b9598ec-b2a0-4211-86c3-d77d2c82cb8f",
    "d7f489e1-227a-4de7-aa07-7b7d37afd139",
    "4621cc65-f05d-46fb-aea2-7245d513e959",
    "21c13149-ca54-45dc-94a1-bd74a1c8a27e",
    "4a93a8f7-305e-489b-b4e7-c31253507ade",
    "b19c41f5-719d-4a93-92e2-0555cad8a607",
    "87f944b2-ff9d-47fe-bf59-6c374f139356",
    "313483e1-e51a-44f6-8e73-3d80d56fe08e",
    "8f8b5bf7-a8ab-41f2-a912-d42139e603e1",
    "d8fcc3dd-e9bb-42e2-b8fd-b2ec5c5e54f1",
    "482e5b50-8c69-4d03-b6d8-e4932a981711",
    "c4944bc8-994d-4ed2-9d99-9812c3275bf0",
    "11bd710a-d8f3-4e3c-88e4-d97d7bc8b3f5",
    "b1ade277-a777-43cf-92a6-ede187252a5c",
    "9a667685-1cdd-4097-9138-9138faf388b2",
    "4fbe9724-b960-4d6c-a016-6b545557b7de",
    "f794039a-f9c7-49f1-82f4-39f93c561523",
    "39097cb5-de7a-49d4-8217-b2c30e19a3ae",
    "e5a6754a-5149-4cc7-821a-bb967d8857bf",
]


def build_url(base_url, recording_id, relpath):
    """Join base / recording / relpath, URL-quoting each path segment."""
    quoted = "/".join(urllib.parse.quote(seg) for seg in relpath.split("/"))
    return f"{base_url.rstrip('/')}/{recording_id}/{quoted}"


def write_split_files(target):
    """Write the baked-in train/test ID lists to <target>/split/{train,test}.txt.

    Returns the list of paths written.  These come straight from the constants
    compiled into this script -- nothing is read from disk to produce them.
    """
    split_dir = os.path.join(target, "split")
    os.makedirs(split_dir, exist_ok=True)
    written = []
    for name, ids in (("train.txt", TRAIN_IDS), ("test.txt", TEST_IDS)):
        path = os.path.join(split_dir, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(ids) + "\n")
        written.append(path)
    return written


def is_non_trimmed_vrs(relpath):
    """True for VRS files that are not the trimmed VRS (excluded everywhere).

    Only files under ``trimmed_vrs/`` are kept; raw / eye-calibration / gap-fill
    VRS files (e.g. ``eye_calibration_helper.vrs``) are never downloaded.
    """
    return relpath.lower().endswith(".vrs") and not relpath.startswith("trimmed_vrs/")


def filter_manifest_files(manifests, includes, excludes):
    """Filter recording paths before listing parts or calculating sizes."""
    kept = []
    for man in manifests:
        man["files"] = [
            e for e in man["files"]
            if (not includes or any(fnmatch.fnmatchcase(e["path"], p)
                                    for p in includes))
            and not any(fnmatch.fnmatchcase(e["path"], p) for p in excludes)
        ]
        man["total_size_bytes"] = sum(e["size"] for e in man["files"])
        if man["files"]:
            kept.append(man)
    return kept


def fetch_manifest(base_url, url_id, timeout, dest_subpath=None, label=None):
    """Fetch and parse a healthcheck.json. Returns a dict.

    *url_id* is the path segment under *base_url* (a recording UUID, or the
    literal ``meshes``).  *dest_subpath* is where the files are written relative
    to the download target (defaults to ``recordings/<url_id>``); *label* is the
    human-readable name shown in logs (defaults to *url_id*).
    """
    url = build_url(base_url, url_id, MANIFEST_NAME)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    data = json.loads(raw.decode("utf-8"))
    files = data.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("manifest contains no file list")
    algo = data.get("hash_algorithm", "sha256")
    if algo not in hashlib.algorithms_available:
        raise ValueError(f"unsupported hash algorithm '{algo}'")
    clean = []
    for entry in files:
        path = entry["path"]
        size = int(entry["size"])
        digest = entry.get("hash")
        if path.startswith("/") or ".." in path.split("/"):
            raise ValueError(f"unsafe path in manifest: {path!r}")
        if is_non_trimmed_vrs(path):
            continue  # raw / non-trimmed VRS are excluded everywhere
        clean.append({"path": path, "size": size, "hash": digest})
    if not clean:
        raise ValueError("manifest has no downloadable files after exclusions")
    return {
        "recording_id": label or url_id,
        "url_id": url_id,
        "dest_subpath": dest_subpath or os.path.join(RECORDINGS_SUBDIR, url_id),
        "hash_algorithm": algo,
        "files": clean,
        "total_size_bytes": sum(e["size"] for e in clean),
        "raw": raw,
    }


def free_space_bytes(path):
    """Free bytes on the filesystem holding *path* (nearest existing parent).

    Returns None if it cannot be determined (e.g. permissions, unusual mount).
    Never requires elevated privileges.
    """
    try:
        probe = os.path.abspath(path)
        while not os.path.exists(probe):
            parent = os.path.dirname(probe)
            if parent == probe:
                return None
            probe = parent
        return shutil.disk_usage(probe).free
    except (OSError, ValueError):
        return None


class DownloadError(Exception):
    pass


def file_digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(READ_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url, dest, expected_size, on_progress, retries, timeout):
    """Download *url* to *dest* with resume + retries.

    ``on_progress(delta, file_done_abs)`` is called as bytes arrive.  Returns
    'exists' if already complete, otherwise 'downloaded'.  Raises DownloadError
    on permanent failure.
    """
    part = dest + ".part"
    if os.path.isfile(dest) and os.path.getsize(dest) == expected_size:
        return "exists"
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    resume = os.path.getsize(part) if os.path.isfile(part) else 0
    if resume > expected_size:
        os.remove(part)
        resume = 0

    attempt = 0
    while True:
        attempt += 1
        try:
            headers = {"User-Agent": USER_AGENT}
            if resume:
                headers["Range"] = f"bytes={resume}-"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                ranged = resp.status == 206
                if resume and not ranged:
                    resume = 0  # server ignored the range -> restart cleanly
                mode = "ab" if resume else "wb"
                on_progress(0, resume)
                written = resume
                with open(part, mode) as fh:
                    while True:
                        chunk = resp.read(DOWNLOAD_CHUNK)
                        if not chunk:
                            break
                        fh.write(chunk)
                        written += len(chunk)
                        on_progress(len(chunk), written)
            actual = os.path.getsize(part)
            if actual != expected_size:
                # Could be a truncated transfer; retry/resume.
                raise DownloadError(
                    f"size mismatch (got {actual}, expected {expected_size})")
            os.replace(part, dest)
            return "downloaded"
        except (urllib.error.HTTPError, urllib.error.URLError, socket.timeout,
                ssl.SSLError, ConnectionError, DownloadError, OSError) as exc:
            if attempt > retries:
                raise DownloadError(f"{type(exc).__name__}: {exc}")
            resume = os.path.getsize(part) if os.path.isfile(part) else 0
            wait = min(2 ** attempt, 30)
            # Drop below the live block (if any) to print the warning.
            sys.stdout.write("\n")
            print(f"    ! {os.path.basename(dest)}: {exc}; retry "
                  f"{attempt}/{retries} in {wait}s", file=sys.stderr)
            time.sleep(wait)
