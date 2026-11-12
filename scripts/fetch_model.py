#!/usr/bin/env python3
"""Fetch a CosyVoice3 checkpoint with parallel range requests.

Single-connection downloads from the public mirrors are slow from mainland
China: measured ~0.3-1.7 MB/s per connection for hf-mirror, ~6.6 MB/s for
ModelScope. Splitting each large file into 16 MiB ranges over eight
connections is several times faster than one connection on both, and on
hf-mirror it turns a multi-hour download into minutes. Completed files are
skipped, sizes and sha256 are verified, and failed ranges are retried, so
re-running after a failure is safe.

Both sources are supported and expose the same file layout:

    python scripts/fetch_model.py --dest /data/models/Fun-CosyVoice3-0.5B
    python scripts/fetch_model.py --source modelscope --dest ...
    python scripts/fetch_model.py --source hf --base https://hf-mirror.com ...

`--source auto` (the default) tries ModelScope first and falls back to
hf-mirror.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request

CHUNK = 16 * 1024 * 1024
WORKERS = 8
UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
# Training-only or optional artifacts that the runtime never loads.
DEFAULT_EXCLUDES = {
    "llm.rl.pt",
    "flow.decoder.estimator.fp32.onnx",
    "speech_tokenizer_v3.batch.onnx",
}
DEFAULT_BASE = {
    "hf": "https://hf-mirror.com",
    "modelscope": "https://www.modelscope.cn",
}
DEFAULT_REVISION = {"hf": "main", "modelscope": "master"}


def log(msg: str) -> None:
    print(msg, flush=True)


def request(url: str, headers: dict[str, str] | None = None) -> urllib.request.Request:
    merged = {"User-Agent": UA}
    if headers:
        merged.update(headers)
    return urllib.request.Request(url, headers=merged)


def fetch_range(url: str, dest: str, start: int, end: int, retries: int = 6) -> None:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = request(url, {"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                if resp.status != 206:
                    raise RuntimeError(f"unexpected status {resp.status}")
                data = resp.read()
            if len(data) != end - start + 1:
                raise RuntimeError(f"short read {len(data)} != {end - start + 1}")
            with open(dest, "r+b") as f:
                f.seek(start)
                f.write(data)
            return
        except Exception as exc:  # noqa: BLE001 - retried below
            last = exc
            time.sleep(min(30, 2 * (attempt + 1)))
    raise RuntimeError(f"range {start}-{end} failed: {last}")


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def list_hf(base: str, repo: str, revision: str) -> list[dict]:
    url = f"{base}/api/models/{repo}?blobs=true"
    with urllib.request.urlopen(request(url), timeout=60) as resp:
        info = json.load(resp)
    return [
        {
            "path": e["rfilename"],
            "size": int(e["size"]),
            "sha256": ((e.get("lfs") or {}).get("sha256") or "").lower(),
        }
        for e in info["siblings"]
    ]


def list_modelscope(base: str, repo: str, revision: str) -> list[dict]:
    url = (
        f"{base}/api/v1/models/{repo}/repo/files"
        f"?Revision={urllib.parse.quote(revision)}&Recursive=true"
    )
    with urllib.request.urlopen(request(url), timeout=60) as resp:
        info = json.load(resp)
    return [
        {
            "path": f["Path"],
            "size": int(f["Size"]),
            "sha256": (f.get("Sha256") or "").lower(),
        }
        for f in info.get("Data", {}).get("Files", [])
        if f.get("Type") == "blob"
    ]


LISTERS = {"hf": list_hf, "modelscope": list_modelscope}


def file_url(source: str, base: str, repo: str, revision: str, rel: str) -> str:
    quoted = urllib.parse.quote(rel, safe="/")
    if source == "modelscope":
        return f"{base}/models/{repo}/resolve/{revision}/{quoted}"
    return f"{base}/{repo}/resolve/{revision}/{quoted}"


def download(entry: dict, url: str, dest_root: str, workers: int) -> None:
    rel = entry["path"]
    size = entry["size"]
    target = os.path.join(dest_root, rel)
    os.makedirs(os.path.dirname(target) or dest_root, exist_ok=True)
    if os.path.exists(target) and os.path.getsize(target) == size:
        log(f"skip  {rel} ({size} bytes)")
        return

    tmp = target + ".part"
    started = time.perf_counter()
    if size <= CHUNK:
        data = b""
        last: Exception | None = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request(url), timeout=180) as resp:
                    data = resp.read()
                break
            except Exception as exc:  # noqa: BLE001 - retried below
                last = exc
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"{rel}: download failed: {last}")
        with open(tmp, "wb") as f:
            f.write(data)
    else:
        with open(tmp, "wb") as f:
            f.truncate(size)
        ranges = [
            (start, min(start + CHUNK - 1, size - 1))
            for start in range(0, size, CHUNK)
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(fetch_range, url, tmp, a, b) for a, b in ranges
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    got = os.path.getsize(tmp)
    if got != size:
        raise RuntimeError(f"{rel}: size {got} != {size}")
    expected = entry.get("sha256") or ""
    if expected:
        digest = sha256(tmp)
        if digest != expected:
            os.remove(tmp)
            raise RuntimeError(f"{rel}: sha256 mismatch {digest} != {expected}")
    os.replace(tmp, target)
    elapsed = time.perf_counter() - started
    log(
        f"done  {rel} ({size / 2**20:.1f} MiB in {elapsed:.1f}s, "
        f"{size / 2**20 / max(elapsed, 1e-9):.2f} MiB/s)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", default="FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
    )
    parser.add_argument("--dest", required=True)
    parser.add_argument(
        "--source",
        choices=["auto", "hf", "modelscope"],
        default="auto",
        help="checkpoint host; auto tries ModelScope, then hf-mirror",
    )
    parser.add_argument(
        "--base",
        default=None,
        help="override the endpoint (default: hf-mirror.com or modelscope.cn)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="file name to skip; repeatable (defaults also skip training-only files)",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="file name to keep even if it is in the default excludes "
        "(e.g. flow.decoder.estimator.fp32.onnx for load_trt)",
    )
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    excludes = (DEFAULT_EXCLUDES | set(args.exclude)) - set(args.include)
    candidates = ["modelscope", "hf"] if args.source == "auto" else [args.source]
    entries: list[dict] = []
    chosen: tuple[str, str, str] | None = None
    for source in candidates:
        base = args.base or DEFAULT_BASE[source]
        revision = DEFAULT_REVISION[source]
        try:
            listed = LISTERS[source](base, args.repo, revision)
        except Exception as exc:  # noqa: BLE001 - try the next source
            log(f"{source}: listing failed: {exc}")
            continue
        if listed:
            chosen = (source, base, revision)
            entries = listed
            break
    if chosen is None:
        log("could not list the checkpoint from any source")
        return 1

    source, base, revision = chosen
    entries = [
        e for e in entries if os.path.basename(e["path"]) not in excludes
    ]
    total = sum(e["size"] for e in entries)
    log(
        f"{source}: {len(entries)} files, {total / 2**30:.2f} GiB -> {args.dest}"
    )
    if args.list_only:
        for e in entries:
            log(f"  {e['path']}  {e['size'] / 2**20:.1f} MiB")
        return 0

    failures = []
    for entry in entries:
        url = file_url(source, base, args.repo, revision, entry["path"])
        try:
            download(entry, url, args.dest, max(1, args.workers))
        except Exception as exc:  # noqa: BLE001 - report and continue
            log(f"FAIL  {entry['path']}: {exc}")
            failures.append(entry["path"])
    if failures:
        log(f"FAILED ({len(failures)}): {failures}")
        return 1
    log("MODEL_FETCH_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
