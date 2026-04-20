---
name: post-test-catch
description: Finalize a completed and fully tested change by committing the intended git files, archiving files from the latest commit into tmp/catch_YYYYMMDD_HHMMSS/, syncing the catch record into local Obsidian, and reporting the commit SHA plus archive path. Use when the user asks to "走 catch 流程", "catch 一下", "提交并归档", "收口提交", or otherwise wants the post-test wrap-up workflow after implementation and validation are complete.
---

# Post Test Catch

## Overview

Use this skill only after the implementation work is complete and the relevant validation has already succeeded or been explicitly accepted by the user. The goal is to finish the change cleanly and leave a reproducible archive of exactly what was committed.

## Workflow

1. Confirm the change is ready to close.
   - Make sure the requested feature or fix is already implemented.
   - Make sure the relevant test, training smoke test, evaluation command, or other validation step has finished.
   - Record the exact validation command(s) and the result you will report back.

2. Inspect the git state before touching commits.
   - Review `git status --short`.
   - If unrelated user changes exist, do not revert them. Stage only the files that belong to the completed task.
   - If the worktree is already clean and `HEAD` is clearly the just-finished change, reuse that commit instead of creating a no-op commit.

3. Guarantee that `tmp/` stays ignored.
   - Check `.gitignore` for `tmp/`.
   - If `tmp/` is missing, add it before archiving.
   - If `tmp/` is already ignored, leave `.gitignore` unchanged.

4. Create the commit for the finished change.
   - Stage only the intended files.
   - Use a short imperative commit subject.
   - After committing, capture the final commit SHA and subject from `HEAD`.

5. Archive the committed files.
   - Run the bundled script from the repository root:
     - `py -3 .codex/skills/post-test-catch/scripts/archive_latest_commit.py`
   - If `py` or `python` is not usable on the machine, switch to a concrete Python interpreter path that exists in the project environment.
   - The script copies files from the latest commit into `tmp/catch_<timestamp>/` and writes `manifest.json` with commit metadata and copied paths.
   - Never archive before the commit exists. Archive the exact commit that represents the finished work.

6. Sync the archived catch into Obsidian.
   - After archiving, run:
     - `py -3 .codex/skills/post-test-catch/scripts/sync_catch_to_obsidian.py --project <StreamYOLO|YOLO|TransVOD> --handoff-title "<title>" --handoff-body "<transfer note>" --next-todo "<next todo>" --archive-dir "<tmp/catch_...>"`
   - Keep local vault metadata in:
     - `.codex/skills/post-test-catch/obsidian_config.json`
   - The Python script owns config loading, git inspection, manifest parsing, deterministic markdown note generation, and optional index-note rewriting.
   - By default the script only writes or updates the project catch note. It does **not** touch `Catch转移进度.md` unless you explicitly pass `--update-index`.
   - The script reads:
     - `.codex/skills/post-test-catch/obsidian_config.json`
     - `<archive_dir>/manifest.json`
     - `git HEAD`
   - The script writes or updates one markdown catch note under the configured Obsidian project folder.
   - If `--update-index` is explicitly provided, the script also rewrites the configured index note and avoids duplicate entries for the same `catch_id` or note link.

7. Report the closure details.
   - Include the validation command(s) you ran.
   - Include the commit SHA and subject.
   - Include the archive directory path.
   - Include whether Obsidian sync succeeded and which note path it wrote.
   - Mention if deleted files were present, because deletions are recorded in `manifest.json` but cannot be copied as files.

## Quick Invocation

Treat short user requests such as `走 catch 流程`, `catch 一下`, `提交并归档`, or `收口提交` as a request to run this full workflow after the implementation and validation are done.

## Bundled Script

`scripts/archive_latest_commit.py` archives the files from one git commit, defaults to `HEAD`, preserves repository-relative paths under `tmp/catch_<timestamp>/`, and writes a machine-readable `manifest.json` for later lookup.

`scripts/sync_catch_to_obsidian.py` reads the archive manifest plus local Obsidian config, derives the default `catch_id` as `project:short_sha`, and writes a markdown handoff note under the configured project folder. Pass `--update-index` only when you intentionally want to rewrite the configured catch index note.
