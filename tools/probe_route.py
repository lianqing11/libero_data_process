#!/usr/bin/env python3
"""Probe one immutable Hugging Face route without persisting resolved URLs."""

import argparse
import json
import sys
import time
from urllib.parse import quote, urlsplit

import requests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", choices=("dataset", "model"), required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--route-label", required=True)
    parser.add_argument("--bytes", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--token-stdin", action="store_true")
    parser.add_argument("--direct", action="store_true", help="Ignore proxy environment variables")
    args = parser.parse_args()
    token = sys.stdin.readline().strip() if args.token_stdin else None
    prefix = "datasets/" if args.repo_type == "dataset" else ""
    url = f"{args.endpoint.rstrip('/')}/{prefix}{args.repo_id}/resolve/{quote(args.revision, safe='')}/{quote(args.path, safe='/')}"
    session = requests.Session()
    session.trust_env = not args.direct
    headers = {"Range": f"bytes=0-{max(0, args.bytes - 1)}"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    started = time.monotonic()
    response = session.get(url, headers=headers, allow_redirects=False, timeout=30, stream=True)
    first = time.monotonic() - started
    if response.status_code not in (301, 302, 303, 307, 308):
        response.raise_for_status()
        download = response
    else:
        resolved = response.headers.get("Location")
        if not resolved:
            raise RuntimeError("redirect_missing_location")
        download = session.get(
            resolved,
            headers={"Range": f"bytes=0-{max(0, args.bytes - 1)}"},
            timeout=60,
            stream=True,
        )
        download.raise_for_status()
    received = 0
    payload_started = time.monotonic()
    for chunk in download.iter_content(1024 * 1024):
        received += len(chunk)
        if received >= args.bytes:
            break
    elapsed = max(time.monotonic() - payload_started, 1e-9)
    print(
        json.dumps(
            {
                "route": args.route_label,
                "endpoint_host": urlsplit(args.endpoint).hostname,
                "http_status": download.status_code,
                "x_repo_commit": response.headers.get("x-repo-commit"),
                "x_linked_size": response.headers.get("x-linked-size"),
                "first_response_seconds": round(first, 4),
                "received_bytes": received,
                "throughput_bps": round(received / elapsed, 2),
            }
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"error": type(error).__name__}), file=sys.stderr)
        raise SystemExit(1)
