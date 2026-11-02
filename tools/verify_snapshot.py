#!/usr/bin/env python3
"""Verify a local Hugging Face snapshot against a saved remote manifest."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def is_lfs_pointer(path: Path) -> bool:
    if path.stat().st_size > 1024:
        return False
    return path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--hash", choices=("none", "lfs", "all"), default="lfs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    issues, verified, records = [], 0, []
    expected_paths = {item["path"] for item in manifest.get("files", [])}
    for item in manifest.get("files", []):
        relative = item["path"]
        path = args.target / relative
        sidecars = [Path(f"{path}.aria2"), Path(f"{path}.incomplete")]
        if any(value.exists() for value in sidecars):
            issues.append({"path": relative, "error": "partial_sidecar"})
            continue
        if not path.is_file():
            issues.append({"path": relative, "error": "missing"})
            continue
        if path.stat().st_size != int(item["size"]):
            issues.append({"path": relative, "error": "size_mismatch"})
            continue
        if is_lfs_pointer(path):
            issues.append({"path": relative, "error": "lfs_pointer"})
            continue
        should_hash = args.hash == "all" or (args.hash == "lfs" and item.get("lfs_oid"))
        actual = sha256(path) if should_hash else None
        if item.get("lfs_oid") and actual and actual != item["lfs_oid"]:
            issues.append({"path": relative, "error": "lfs_sha256_mismatch"})
            continue
        records.append({"path": relative, "size": path.stat().st_size, "sha256": actual})
        verified += path.stat().st_size
    unexpected_partials = []
    if args.target.exists():
        for path in args.target.rglob("*"):
            if path.is_file() and (path.name.endswith(".aria2") or path.name.endswith(".incomplete")):
                base = str(path.relative_to(args.target)).removesuffix(".aria2").removesuffix(".incomplete")
                if base not in expected_paths:
                    unexpected_partials.append(str(path.relative_to(args.target)))
    issues.extend({"path": value, "error": "unexpected_partial"} for value in unexpected_partials)
    result = {
        "schema_version": 1,
        "verified": not issues,
        "repo_id": manifest.get("repo_id"),
        "revision": manifest.get("revision"),
        "verified_bytes": verified,
        "verified_files": len(records),
        "remote_bytes": manifest.get("remote_bytes"),
        "remote_files": len(manifest.get("files", [])),
        "hash_mode": args.hash,
        "issues": issues,
        "files": records,
    }
    atomic_json(args.output, result)
    print(json.dumps({"verified": result["verified"], "verified_files": len(records), "issues": len(issues)}))
    return 0 if result["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
