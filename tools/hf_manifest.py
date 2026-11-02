#!/usr/bin/env python3
"""Resolve an immutable Hugging Face revision and save its complete file tree."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from datetime import datetime, timezone

import httpx
from huggingface_hub import HfApi, set_client_factory
from huggingface_hub.hf_api import RepoFile


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", choices=("dataset", "model"), required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-stdin", action="store_true")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--direct", action="store_true", help="Ignore proxy environment variables")
    args = parser.parse_args()

    token = sys.stdin.readline().strip() if args.token_stdin else None
    set_client_factory(lambda: httpx.Client(timeout=args.timeout, trust_env=not args.direct))
    api = HfApi(endpoint=args.endpoint, token=token or None)
    info = api.repo_info(
        args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        token=token or None,
    )
    commit = info.sha
    files = []
    for item in api.list_repo_tree(
        args.repo_id,
        repo_type=args.repo_type,
        revision=commit,
        recursive=True,
        expand=True,
        token=token or None,
    ):
        if not isinstance(item, RepoFile):
            continue
        lfs = item.lfs
        files.append(
            {
                "path": item.path,
                "size": item.size,
                "lfs_oid": getattr(lfs, "sha256", None) if lfs else None,
                "lfs_size": getattr(lfs, "size", None) if lfs else None,
                "xet_hash": item.xet_hash,
                "blob_oid": item.blob_id,
            }
        )
    files.sort(key=lambda value: value["path"])
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_id": args.repo_id,
        "repo_type": args.repo_type,
        "requested_revision": args.revision,
        "revision": commit,
        "endpoint_host": args.endpoint.split("//", 1)[-1].split("/", 1)[0],
        "remote_files": len(files),
        "remote_bytes": sum(value["size"] for value in files),
        "files": files,
    }
    atomic_json(args.output, payload)
    print(json.dumps({key: payload[key] for key in ("repo_id", "revision", "remote_files", "remote_bytes")}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"error": type(error).__name__}), file=sys.stderr)
        raise SystemExit(1)
