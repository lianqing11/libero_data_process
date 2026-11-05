from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Tuple


_LOCAL_MODEL_ROOTS = (
    *[Path(p).expanduser() for p in os.environ.get("VLA_MODEL_ROOTS", "").split(os.pathsep) if p],
)


def _is_offline_mode() -> bool:
    flags = (
        os.environ.get("HF_HUB_OFFLINE"),
        os.environ.get("TRANSFORMERS_OFFLINE"),
    )
    return any(str(flag).lower() in {"1", "true", "yes"} for flag in flags if flag is not None)


def _has_required_files(path: Path, required_files: Iterable[str]) -> bool:
    return all((path / filename).exists() for filename in required_files)


def resolve_pretrained_path(
    pretrained_name_or_path: str,
    required_files: Iterable[str] = (),
) -> Tuple[str, bool]:
    """
    Resolve a HF repo id to a local mirrored directory when available.

    Returns `(resolved_path, is_local)`.
    """
    path = Path(pretrained_name_or_path).expanduser()
    if path.exists():
        return str(path), True

    repo_parts = str(pretrained_name_or_path).split("/")
    for root in _LOCAL_MODEL_ROOTS:
        candidate = root.joinpath(*repo_parts)
        if candidate.exists() and _has_required_files(candidate, required_files):
            logging.info("Resolved pretrained asset %s -> %s", pretrained_name_or_path, candidate)
            return str(candidate), True

    cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / (
        "models--" + str(pretrained_name_or_path).replace("/", "--")
    )
    snapshot_root = cache_dir / "snapshots"
    if snapshot_root.exists():
        snapshot_dirs = sorted(
            (p for p in snapshot_root.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for candidate in snapshot_dirs:
            if _has_required_files(candidate, required_files):
                logging.info("Resolved pretrained asset %s -> %s", pretrained_name_or_path, candidate)
                return str(candidate), True

    return pretrained_name_or_path, False


def resolve_pretrained_load(
    pretrained_name_or_path: str,
    required_files: Iterable[str] = (),
    **kwargs,
) -> Tuple[str, dict]:
    """
    Resolve path and attach `local_files_only=True` whenever a local copy exists.
    """
    resolved_path, is_local = resolve_pretrained_path(
        pretrained_name_or_path,
        required_files=required_files,
    )
    load_kwargs = dict(kwargs)
    if is_local or _is_offline_mode():
        load_kwargs.setdefault("local_files_only", True)
    return resolved_path, load_kwargs
