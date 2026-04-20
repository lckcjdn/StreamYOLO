#!/usr/bin/env python3
"""Sync archived catch metadata into a local Obsidian vault."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent


def log(message: str) -> None:
    print(f"[sync-catch-to-obsidian] {message}")


def fail(message: str) -> int:
    print(f"[sync-catch-to-obsidian] ERROR: {message}", file=sys.stderr)
    return 1


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


def resolve_repo_root(explicit_repo_root: str | None) -> Path:
    if explicit_repo_root:
        return Path(explicit_repo_root).resolve()
    root = run_git(["rev-parse", "--show-toplevel"], Path.cwd())
    return Path(root).resolve()


def load_json_file(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return data


def resolve_archive_dir(repo_root: Path, archive_dir_arg: str) -> Path:
    candidate = Path(archive_dir_arg)
    if not candidate.is_absolute():
        candidate = (repo_root / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"archive_dir does not exist: {candidate}")
    return candidate


def read_git_head(repo_root: Path) -> dict[str, str]:
    sha = run_git(["rev-parse", "HEAD"], repo_root)
    subject = run_git(["show", "-s", "--format=%s", "HEAD"], repo_root)
    committed_at = run_git(["show", "-s", "--format=%cI", "HEAD"], repo_root)
    return {
        "sha": sha,
        "short_sha": sha[:8],
        "subject": subject,
        "committed_at": committed_at,
    }


def read_manifest(archive_dir: Path) -> dict[str, Any]:
    return load_json_file(archive_dir / "manifest.json", "manifest")


def get_copied_paths(manifest: dict[str, Any]) -> list[str]:
    for key in ("copied_paths", "copied_files", "included_files"):
        value = manifest.get(key)
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return list(value)
    files = manifest.get("files")
    if isinstance(files, list):
        if all(isinstance(item, str) for item in files):
            return list(files)
        extracted = []
        for item in files:
            if isinstance(item, dict) and isinstance(item.get("path"), str):
                extracted.append(item["path"])
        if extracted:
            return extracted
    raise ValueError("manifest.json is missing copied_paths/copied_files/included_files/files[path].")


def get_manifest_created_at(manifest: dict[str, Any]) -> str | None:
    value = manifest.get("created_at") or manifest.get("reorganized_at")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def read_commit_info_from_sha(repo_root: Path, sha: str) -> dict[str, str]:
    return {
        "sha": sha,
        "short_sha": sha[:8],
        "subject": run_git(["show", "-s", "--format=%s", sha], repo_root),
        "committed_at": run_git(["show", "-s", "--format=%cI", sha], repo_root),
    }


def read_manifest_commit_info(repo_root: Path, manifest: dict[str, Any]) -> dict[str, str] | None:
    commit_value = manifest.get("commit")
    if isinstance(commit_value, dict):
        sha = commit_value.get("sha")
        if isinstance(sha, str) and sha.strip():
            sha = sha.strip()
            subject = commit_value.get("subject")
            committed_at = commit_value.get("committed_at")
            if isinstance(subject, str) and subject.strip() and isinstance(committed_at, str) and committed_at.strip():
                return {
                    "sha": sha,
                    "short_sha": sha[:8],
                    "subject": subject.strip(),
                    "committed_at": committed_at.strip(),
                }
            return read_commit_info_from_sha(repo_root, sha)
    if isinstance(commit_value, str) and commit_value.strip():
        return read_commit_info_from_sha(repo_root, commit_value.strip())

    source_commits = manifest.get("source_commits")
    if isinstance(source_commits, list) and source_commits:
        latest = source_commits[-1]
        if isinstance(latest, dict):
            sha = latest.get("sha")
            if isinstance(sha, str) and sha.strip():
                sha = sha.strip()
                subject = latest.get("subject")
                committed_at = latest.get("committed_at")
                if isinstance(subject, str) and subject.strip() and isinstance(committed_at, str) and committed_at.strip():
                    return {
                        "sha": sha,
                        "short_sha": sha[:8],
                        "subject": subject.strip(),
                        "committed_at": committed_at.strip(),
                    }
                return read_commit_info_from_sha(repo_root, sha)
    return None


def build_catch_id(project: str, archive_dir: Path, commit_info: dict[str, str]) -> str:
    return f"{project}:{commit_info['short_sha']}:{archive_dir.name}"


def validate_config(config: dict[str, Any]) -> tuple[Path, str, dict[str, str], Path | None]:
    vault_path_raw = config.get("vault_path")
    catch_root = config.get("catch_root")
    projects = config.get("projects")
    index_note_raw = config.get("index_note")

    if not isinstance(vault_path_raw, str) or not vault_path_raw.strip():
        raise ValueError("obsidian_config.json is missing 'vault_path'.")
    if not isinstance(catch_root, str) or not catch_root.strip():
        raise ValueError("obsidian_config.json is missing 'catch_root'.")
    if not isinstance(projects, dict) or not projects:
        raise ValueError("obsidian_config.json is missing 'projects'.")
    if index_note_raw is not None and (
        not isinstance(index_note_raw, str) or not index_note_raw.strip()
    ):
        raise ValueError("obsidian_config.json field 'index_note' must be a non-empty string when provided.")

    normalized_projects: dict[str, str] = {}
    for key, value in projects.items():
        if not isinstance(key, str) or not key.strip() or not isinstance(value, str) or not value.strip():
            raise ValueError("obsidian_config.json projects must map non-empty strings to non-empty strings.")
        normalized_projects[key.strip()] = value.strip()

    vault_path = Path(vault_path_raw).resolve()
    if not vault_path.exists():
        raise FileNotFoundError(f"vault_path does not exist: {vault_path}")

    index_note = None
    if isinstance(index_note_raw, str) and index_note_raw.strip():
        index_note = (vault_path / Path(index_note_raw)).resolve()
    return vault_path, catch_root.strip(), normalized_projects, index_note


def slugify_title(title: str, max_length: int = 48) -> str:
    normalized = re.sub(r"\s+", "_", title.strip())
    normalized = re.sub(r'[<>:"/\\\\|?*]', "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("._ ")
    if not normalized:
        normalized = "catch"
    return normalized[:max_length]


def format_yaml_list(items: list[str], indent: int = 0) -> list[str]:
    prefix = " " * indent
    return [f"{prefix}- {item}" for item in items]


def build_note_content(
    *,
    project: str,
    catch_id: str,
    title: str,
    created_at: str,
    status: str,
    commit_info: dict[str, str],
    archive_dir: Path,
    handoff_body: str,
    next_todo: str,
    copied_paths: list[str],
) -> str:
    lines = [
        "---",
        f"project: {project}",
        f"catch_id: {catch_id}",
        f"commit_sha: {commit_info['sha']}",
        f"commit_subject: {json.dumps(commit_info['subject'], ensure_ascii=False)}",
        f"archive_dir: {json.dumps(str(archive_dir), ensure_ascii=False)}",
        f"created_at: {created_at}",
        f"status: {status}",
        "tags:",
        "  - catch",
        f"  - {project}",
        "---",
        "",
        f"# {title}",
        "",
        "## 本次转移内容",
        handoff_body.strip() or "(empty)",
        "",
        "## 下一步待做",
        next_todo.strip() or "(empty)",
        "",
        "## 提交信息",
        f"- commit sha: {commit_info['sha']}",
        f"- commit short sha: {commit_info['short_sha']}",
        f"- commit subject: {commit_info['subject']}",
        f"- committed at: {commit_info['committed_at']}",
        "",
        "## 归档位置",
        f"- {archive_dir}",
        "",
        "## 归档文件列表",
    ]
    if copied_paths:
        lines.extend(format_yaml_list(copied_paths))
    else:
        lines.append("- (none)")
    lines.append("")
    return "\n".join(lines)


def find_existing_note_by_catch_id(project_dir: Path, catch_id: str) -> Path | None:
    if not project_dir.exists():
        return None
    target = f"catch_id: {catch_id}"
    for note_path in sorted(project_dir.rglob("*.md")):
        try:
            content = note_path.read_text(encoding="utf-8")
        except OSError:
            continue
        if target in content:
            return note_path
    return None


def find_existing_note_by_archive_dir(project_dir: Path, archive_dir: Path) -> Path | None:
    if not project_dir.exists():
        return None
    target = f'archive_dir: {json.dumps(str(archive_dir), ensure_ascii=False)}'
    for note_path in sorted(project_dir.rglob("*.md")):
        try:
            content = note_path.read_text(encoding="utf-8")
        except OSError:
            continue
        if target in content:
            return note_path
    return None


def ensure_index_note(index_note_path: Path) -> str:
    if not index_note_path.exists():
        index_note_path.parent.mkdir(parents=True, exist_ok=True)
        default_content = "# Catch转移进度\n\n"
        index_note_path.write_text(default_content, encoding="utf-8")
        return default_content
    return index_note_path.read_text(encoding="utf-8")


def ensure_project_heading(lines: list[str], project: str) -> tuple[list[str], int]:
    heading = f"## {project}"
    for idx, line in enumerate(lines):
        if line.strip() == heading:
            return lines, idx
    if lines and lines[-1].strip():
        lines.append("")
    lines.extend([heading, ""])
    return lines, len(lines) - 2


def read_note_frontmatter(note_path: Path) -> dict[str, str]:
    try:
        lines = note_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    if not lines or lines[0].strip() != "---":
        return {}

    frontmatter: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        frontmatter[key.strip()] = value.strip().strip('"')
    return frontmatter


def collect_project_index_lines(catch_root_dir: Path, project_dir: Path) -> list[str]:
    entries: list[tuple[str, str, str]] = []
    if not project_dir.exists():
        return []

    for note_path in sorted(project_dir.rglob("*.md")):
        frontmatter = read_note_frontmatter(note_path)
        created_at = frontmatter.get("created_at", "")
        status = frontmatter.get("status", "open").strip().lower()
        checked = "x" if status == "completed" else " "
        link_target = note_path.relative_to(catch_root_dir).with_suffix("").as_posix()
        entries.append((created_at, link_target, f"- [{checked}] [[{link_target}]]"))

    entries.sort(key=lambda item: (item[0], item[1]))
    return [line for _, _, line in entries]


def update_index_note(index_note_path: Path, project: str, catch_root_dir: Path, project_dir: Path) -> bool:
    content = ensure_index_note(index_note_path)
    lines = content.splitlines()
    lines, heading_idx = ensure_project_heading(lines, project)
    heading = f"## {project}"
    section_end = len(lines)
    for idx in range(heading_idx + 1, len(lines)):
        if lines[idx].startswith("## ") and lines[idx].strip() != heading:
            section_end = idx
            break

    replacement_section = collect_project_index_lines(catch_root_dir, project_dir)
    before = lines[: heading_idx + 1]
    after = lines[section_end:]
    while after and not after[0].strip():
        after = after[1:]

    new_lines = before[:]
    if replacement_section:
        new_lines.extend(replacement_section)
    if after:
        new_lines.append("")
        new_lines.extend(after)

    new_content = "\n".join(new_lines) + "\n"
    if new_content == content:
        return False
    index_note_path.write_text(new_content, encoding="utf-8")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync an archived catch into a local Obsidian vault.")
    parser.add_argument("--project", required=True, help="Project section, e.g. YOLO or TransVOD.")
    parser.add_argument("--handoff-title", required=True, help="Note title shown in Obsidian.")
    parser.add_argument("--handoff-body", required=True, help="Transfer handoff note.")
    parser.add_argument("--next-todo", required=True, help="Next todo to record.")
    parser.add_argument("--archive-dir", required=True, help="Archive directory that contains manifest.json.")
    parser.add_argument(
        "--status",
        default="open",
        choices=["open", "completed"],
        help="Frontmatter status written into the catch note.",
    )
    parser.add_argument(
        "--update-index",
        action="store_true",
        help="Also rewrite the configured Catch index note. Disabled by default.",
    )
    parser.add_argument("--repo-root", help="Repository root. Defaults to current git toplevel.")
    parser.add_argument(
        "--config-path",
        default=str(SKILL_DIR / "obsidian_config.json"),
        help="Obsidian config JSON path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        repo_root = resolve_repo_root(args.repo_root)
        config_path = Path(args.config_path).resolve()
        archive_dir = resolve_archive_dir(repo_root, args.archive_dir)

        log(f"Reading config: {config_path}")
        config = load_json_file(config_path, "config")
        vault_path, catch_root, projects, index_note_path = validate_config(config)

        if args.project not in projects:
            supported = ", ".join(sorted(projects.keys()))
            raise ValueError(f"project '{args.project}' is not configured. Supported projects: {supported}")

        log(f"Reading manifest: {archive_dir / 'manifest.json'}")
        manifest = read_manifest(archive_dir)
        commit_info = read_manifest_commit_info(repo_root, manifest)
        if commit_info is None:
            log("Manifest commit metadata missing, falling back to git HEAD")
            commit_info = read_git_head(repo_root)
        log(f"Resolved commit: {commit_info['sha']} {commit_info['subject']}")
        copied_paths = get_copied_paths(manifest)

        catch_id = build_catch_id(args.project, archive_dir, commit_info)
        created_at = get_manifest_created_at(manifest) or datetime.now().astimezone().isoformat()
        note_date = commit_info["committed_at"][:10]
        note_slug = slugify_title(args.handoff_title)
        note_filename = f"{note_date}_{commit_info['short_sha']}_{note_slug}.md"

        catch_root_dir = vault_path / catch_root
        project_dir = catch_root_dir / projects[args.project]
        project_dir.mkdir(parents=True, exist_ok=True)

        note_path = find_existing_note_by_archive_dir(project_dir, archive_dir)
        if note_path is None:
            note_path = find_existing_note_by_catch_id(project_dir, catch_id)
        if note_path is None:
            note_path = project_dir / note_filename

        note_content = build_note_content(
            project=args.project,
            catch_id=catch_id,
            title=args.handoff_title,
            created_at=created_at,
            status=args.status,
            commit_info=commit_info,
            archive_dir=archive_dir,
            handoff_body=args.handoff_body,
            next_todo=args.next_todo,
            copied_paths=copied_paths,
        )
        note_path.write_text(note_content, encoding="utf-8")

        index_updated: bool | None = None
        if args.update_index:
            if index_note_path is None:
                raise ValueError(
                    "obsidian_config.json is missing 'index_note', so --update-index cannot be used."
                )
            index_updated = update_index_note(
                index_note_path, args.project, catch_root_dir, project_dir
            )

        print(f"commit sha: {commit_info['sha']}")
        print(f"archive dir: {archive_dir}")
        print(f"obsidian note path: {note_path}")
        if index_updated is None:
            print("index note updated: skipped")
        else:
            print(f"index note updated: {index_updated}")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        return fail(str(exc))


if __name__ == "__main__":
    sys.exit(main())
