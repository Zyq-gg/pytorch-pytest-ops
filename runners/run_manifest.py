#!/usr/bin/env python3
"""Create and validate semantic run manifests for checkpointed test work dirs."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


MANIFEST_VERSION = 1
MANIFEST_NAME = ".run_manifest.json"
ENV_PREFIXES = (
    "AMDGPU_",
    "BUILD_",
    "CUDA_",
    "HIP_",
    "HSA_",
    "MIOPEN_",
    "MKL_",
    "NCCL_",
    "OMP_",
    "PYTORCH_",
    "ROCM_",
    "TORCH",
    "USE_",
)
ENV_NAMES = {
    "C_INCLUDE_PATH",
    "CPLUS_INCLUDE_PATH",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "MAX_JOBS",
    "PATH",
    "PYTHONPATH",
}


def json_fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def selected_environment(env: dict[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in env.items()
        if key in ENV_NAMES or key.startswith(ENV_PREFIXES)
    }


def git_source_identity(pytorch_root: Path) -> dict[str, Any]:
    identity: dict[str, Any] = {"root": str(pytorch_root.resolve())}
    try:
        head = subprocess.run(
            ["git", "-C", str(pytorch_root), "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "-C",
                str(pytorch_root),
                "status",
                "--porcelain=v1",
                "--untracked-files=no",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=30,
            check=True,
        ).stdout
        tracked_diff = subprocess.run(
            ["git", "-C", str(pytorch_root), "diff", "--no-ext-diff", "--binary", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=True,
        ).stdout
        identity.update(
            {
                "git_head": head,
                "tracked_worktree_dirty": bool(status.strip()),
                "tracked_status_sha256": json_fingerprint(status),
                "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
            }
        )
    except (OSError, subprocess.SubprocessError):
        identity["git_head"] = None
        identity["tracked_worktree_dirty"] = None
        identity["tracked_status_sha256"] = None
        identity["tracked_diff_sha256"] = None
    return identity


def build_run_manifest(
    *,
    runner: str,
    pytorch_root: str | Path,
    planned_items: list[str],
    options: dict[str, Any],
    env: dict[str, str],
) -> dict[str, Any]:
    selected_env = selected_environment(env)
    details = {
        "runner": runner,
        "source": git_source_identity(Path(pytorch_root)),
        "plan": {
            "count": len(planned_items),
            "sha256": json_fingerprint(planned_items),
        },
        "environment": {
            "keys": sorted(selected_env),
            "sha256": json_fingerprint(selected_env),
        },
        "options": options,
    }
    return {
        "version": MANIFEST_VERSION,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fingerprint": json_fingerprint(details),
        "details": details,
    }


def without_ignored_paths(value: dict[str, Any], ignored_paths: set[str]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(value))
    for dotted in ignored_paths:
        parts = dotted.split(".")
        current: Any = normalized
        for part in parts[:-1]:
            if not isinstance(current, dict) or part not in current:
                current = None
                break
            current = current[part]
        if isinstance(current, dict):
            current.pop(parts[-1], None)
    return normalized


def changed_sections(
    previous: dict[str, Any],
    current: dict[str, Any],
    *,
    ignored_paths: set[str],
) -> list[str]:
    old_details = without_ignored_paths(previous.get("details", {}), ignored_paths)
    new_details = without_ignored_paths(current.get("details", {}), ignored_paths)
    keys = sorted(set(old_details) | set(new_details))
    return [key for key in keys if old_details.get(key) != new_details.get(key)]


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def ensure_run_manifest(
    work_dir: str | Path,
    manifest: dict[str, Any],
    *,
    fresh: bool,
    checkpoint_exists: bool,
    ignored_paths: set[str] | None = None,
    saved_plan_path: str | Path | None = None,
) -> Path:
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    path = work / MANIFEST_NAME
    ignored = ignored_paths or set()

    if fresh or not path.is_file():
        if checkpoint_exists and not fresh:
            saved_plan = Path(saved_plan_path) if saved_plan_path is not None else None
            if saved_plan is not None and saved_plan.is_file():
                saved_items = [
                    line.strip()
                    for line in saved_plan.read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.strip().startswith("#")
                ]
                current_plan = manifest.get("details", {}).get("plan", {})
                if (
                    current_plan.get("count") != len(saved_items)
                    or current_plan.get("sha256") != json_fingerprint(saved_items)
                ):
                    raise RuntimeError(
                        "legacy checkpoint plan does not match the newly generated plan; "
                        f"use the original source/filters or a new work-dir "
                        f"(saved plan: {saved_plan})"
                    )
            manifest["legacy_checkpoint_adopted_without_verification"] = True
            print(
                "WARNING: existing checkpoint has no run manifest; "
                "adopting the current configuration without historical verification.",
                flush=True,
            )
        write_manifest(path, manifest)
        print(f"Run manifest: {path}", flush=True)
        return path

    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid existing run manifest: {path}: {exc}") from exc

    changed = changed_sections(previous, manifest, ignored_paths=ignored)
    if changed:
        raise RuntimeError(
            "run manifest mismatch in "
            + ", ".join(changed)
            + f"; use the original semantic configuration or a new work-dir "
            f"(manifest: {path})"
        )
    print(f"Run manifest verified: {path}", flush=True)
    return path
