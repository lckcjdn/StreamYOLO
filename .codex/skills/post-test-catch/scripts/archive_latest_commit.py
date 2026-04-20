#!/usr/bin/env python3
"""Archive files from a git commit into tmp/catch_<timestamp>."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {stderr}")
    return result.stdout.strip()


def resolve_repo_root(explicit_root: str | None) -> Path:
    if explicit_root:
        return Path(explicit_root).resolve()
    root = run_git(["rev-parse", "--show-toplevel"], Path.cwd())
    return Path(root).resolve()


def parse_changed_paths(name_status_text: str) -> tuple[list[str], list[str]]:
    copyable: list[str] = []
    deleted: list[str] = []
    for raw_line in name_status_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("\t")
        status = parts[0]
        code = status[0]
        path = parts[-1]
        if code == "D":
            deleted.append(path)
            continue
        copyable.append(path)
    return copyable, deleted


def copy_path(repo_root: Path, relative_path: str, output_dir: Path) -> bool:
    source = repo_root / relative_path
    destination = output_dir / relative_path
    if not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        shutil.copy2(source, destination)
    return True


def build_output_dir(repo_root: Path, output_root: str, timestamp: str, folder_name: str | None) -> Path:
    root = repo_root / output_root
    name = folder_name or f"catch_{timestamp}"
    output_dir = root / name
    if output_dir.exists():
        raise FileExistsError(f"Archive directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Archive files from a git commit into tmp/catch_<timestamp>.")
    parser.add_argument("--repo-root", help="Repository root. Defaults to the current git toplevel.")
    parser.add_argument("--commit", default="HEAD", help="Commit to archive. Defaults to HEAD.")
    parser.add_argument("--output-root", default="tmp", help="Archive parent directory relative to repo root.")
    parser.add_argument("--timestamp", help="Timestamp override in YYYYMMDD_HHMMSS format.")
    parser.add_argument("--folder-name", help="Override the archive folder name.")
    args = parser.parse_args()

    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    repo_root = resolve_repo_root(args.repo_root)
    commit_sha = run_git(["rev-parse", args.commit], repo_root)
    commit_subject = run_git(["show", "-s", "--format=%s", commit_sha], repo_root)
    commit_time = run_git(["show", "-s", "--format=%cI", commit_sha], repo_root)
    changed_text = run_git(["diff-tree", "--root", "--no-commit-id", "--name-status", "-r", commit_sha], repo_root)
    copyable_paths, deleted_paths = parse_changed_paths(changed_text)
    output_dir = build_output_dir(repo_root, args.output_root, timestamp, args.folder_name)

    copied_paths: list[str] = []
    missing_paths: list[str] = []
    for relative_path in copyable_paths:
        if copy_path(repo_root, relative_path, output_dir):
            copied_paths.append(relative_path)
        else:
            missing_paths.append(relative_path)

    manifest = {
        "repo_root": str(repo_root),
        "commit": {
            "sha": commit_sha,
            "subject": commit_subject,
            "committed_at": commit_time,
        },
        "archive_dir": str(output_dir),
        "copied_paths": copied_paths,
        "copied_files": copied_paths,
        "deleted_files": deleted_paths,
        "missing_files": missing_paths,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Archived commit: {commit_sha}")
    print(f"Subject: {commit_subject}")
    print(f"Archive: {output_dir}")
    print(f"Copied files: {len(copied_paths)}")
    print(f"Deleted files: {len(deleted_paths)}")
    print(f"Missing files: {len(missing_paths)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
