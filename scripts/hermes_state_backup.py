#!/usr/bin/env python3
"""Small, explicit and verifiable snapshots of Hermes' durable user state.

The runner is ephemeral. This helper copies only state that Hermes uses to
reconstruct conversations, memories, profiles, schedules and task boards. It
never archives the installed checkout, environments, caches, logs or sockets.
SQLite files are captured with the online-backup API so committed WAL frames
are included without copying live -wal/-shm files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

FORMAT_VERSION = 2
ROOT_FILES = ("config.yaml", ".env", "SOUL.md", "AGENTS.md", "profile.yaml", "auth.json")
DURABLE_DIRS = (
    "plugin-data",
    "memories",
    "sessions",
    "pending_messages",
    "skills",
    "cron",
    "kanban",
)
SKIP_DIRS = {
    ".git", "__pycache__", "cache", "caches", "hermes-agent", "installs",
    "logs", "scratch", "tmp", "workspaces", "venv", "venvs",
}
SKIP_FILE_SUFFIXES = (
    ".lock", ".sock", ".tmp", ".swp", ".db", ".sqlite", ".sqlite3",
    "-wal", "-shm", "-journal",
)
COUNT_TABLES = (
    "sessions", "messages", "gateway_routing", "session_admissions",
    "admissions", "executions", "tasks",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(value: str) -> Path:
    raw = PurePosixPath(value)
    if raw.is_absolute() or not raw.parts or ".." in raw.parts or "\\" in value:
        raise ValueError(f"unsafe relative path: {value!r}")
    return Path(*raw.parts)


def _is_regular_file(path: Path) -> bool:
    try:
        return not path.is_symlink() and stat.S_ISREG(path.lstat().st_mode)
    except FileNotFoundError:
        return False


def _copy_one(source: Path, destination: Path) -> bool:
    if not _is_regular_file(source):
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if destination.name == ".env":
        destination.chmod(0o600)
    return True


def _copy_tree(source: Path, destination: Path) -> int:
    """Copy regular files from one approved durable tree, never symlinks/sockets."""
    if source.is_symlink() or not source.is_dir():
        return 0
    copied = 0
    destination.mkdir(parents=True, exist_ok=True)
    for current, dirs, files in os.walk(source, topdown=True, followlinks=False):
        current_path = Path(current)
        dirs[:] = [
            name for name in dirs
            if name.lower() not in SKIP_DIRS and not (current_path / name).is_symlink()
        ]
        relative_dir = current_path.relative_to(source)
        for name in files:
            source_file = current_path / name
            if name.lower().endswith(SKIP_FILE_SUFFIXES) or source_file.is_symlink():
                continue
            try:
                mode = source_file.lstat().st_mode
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(mode):
                continue
            rel = relative_dir / name
            if _copy_one(source_file, destination / rel):
                copied += 1
    return copied


def _copy_durable_layout(source_home: Path, output_home: Path) -> int:
    """Copy only documented durable files/trees from a Hermes home."""
    copied = 0
    output_home.mkdir(parents=True, exist_ok=True)

    for name in ROOT_FILES:
        if _copy_one(source_home / name, output_home / name):
            copied += 1

    for name in DURABLE_DIRS:
        copied += _copy_tree(source_home / name, output_home / name)

    profiles = source_home / "profiles"
    if profiles.is_dir() and not profiles.is_symlink():
        for profile in sorted(profiles.iterdir()):
            if profile.is_symlink() or not profile.is_dir():
                continue
            target = output_home / "profiles" / profile.name
            target.mkdir(parents=True, exist_ok=True)
            for name in ROOT_FILES:
                if _copy_one(profile / name, target / name):
                    copied += 1
            for name in DURABLE_DIRS:
                copied += _copy_tree(profile / name, target / name)

    return copied


def _database_candidates(home: Path) -> list[Path]:
    candidates: set[Path] = {
        home / "state.db",
        home / "kanban.db",
        home / "cron" / "executions.db",
    }
    profiles = home / "profiles"
    profile_homes = [p for p in profiles.iterdir() if p.is_dir() and not p.is_symlink()] if profiles.is_dir() else []
    for profile in profile_homes:
        candidates.update({
            profile / "state.db",
            profile / "kanban.db",
            profile / "cron" / "executions.db",
        })

    # Multi-board Kanban DBs and additional SQLite stores are scoped to these
    # durable subtrees only; never scan caches, installs or the whole home.
    durable_roots = [home / "cron", home / "kanban"]
    for profile in profile_homes:
        durable_roots.extend((profile / "cron", profile / "kanban"))
    for root in durable_roots:
        if root.is_dir() and not root.is_symlink():
            for path in root.rglob("*"):
                if path.is_file() and not path.is_symlink() and path.name.lower().endswith(
                    (".db", ".sqlite", ".sqlite3")
                ):
                    candidates.add(path)

    return sorted(
        (p for p in candidates if p.exists()),
        key=lambda p: str(p.relative_to(home)),
    )


def _database_info(path: Path) -> dict[str, Any]:
    uri = path.resolve().as_uri() + "?mode=ro"
    counts: dict[str, int] = {}
    with sqlite3.connect(uri, uri=True, timeout=60) as db:
        check = db.execute("PRAGMA quick_check").fetchone()
        if not check or check[0] != "ok":
            raise RuntimeError(f"SQLite quick_check failed for {path}: {check!r}")
        existing = {
            row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        for table in COUNT_TABLES:
            if table in existing:
                # Table names are selected from the fixed allowlist above.
                counts[table] = int(db.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
    return counts


def _snapshot_databases(home: Path, output_home: Path) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for source in _database_candidates(home):
        if source.is_symlink() or not _is_regular_file(source):
            raise RuntimeError(f"refusing a symlink or non-regular SQLite path: {source}")
        relative = source.relative_to(home)
        target = output_home / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.unlink(missing_ok=True)
        try:
            source_uri = source.resolve().as_uri() + "?mode=ro"
            with sqlite3.connect(source_uri, uri=True, timeout=60) as src:
                with sqlite3.connect(target, timeout=60) as dst:
                    src.backup(dst, pages=512, sleep=0.05)
                    check = dst.execute("PRAGMA quick_check").fetchone()
                    if not check or check[0] != "ok":
                        raise RuntimeError(f"snapshot quick_check failed: {check!r}")
                    existing = {
                        row[0] for row in dst.execute(
                            "SELECT name FROM sqlite_master WHERE type='table'"
                        ).fetchall()
                    }
                    counts = {}
                    for table in COUNT_TABLES:
                        if table in existing:
                            counts[table] = int(dst.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
        except Exception:
            target.unlink(missing_ok=True)
            raise
        target.chmod(0o600)
        snapshots.append({
            "path": relative.as_posix(),
            "bytes": target.stat().st_size,
            "sha256": _sha256(target),
            "quick_check": "ok",
            "row_counts": counts,
        })
        human_counts = ", ".join(f"{k}={v}" for k, v in counts.items()) or "no tracked tables"
        print(f"SQLite snapshot: {relative} ({target.stat().st_size / (1024 * 1024):.2f} MiB; {human_counts})")
    return snapshots


def snapshot_home(home: Path, work: Path) -> dict[str, Any]:
    home = home.resolve()
    work = work.resolve()
    output_home = work / "hermes"
    shutil.rmtree(work, ignore_errors=True)
    output_home.mkdir(parents=True, exist_ok=True)

    copied = _copy_durable_layout(home, output_home)
    databases = _snapshot_databases(home, output_home)

    if not (output_home / "config.yaml").is_file():
        raise RuntimeError(f"refusing to snapshot without {home / 'config.yaml'}")

    database_paths = {item["path"] for item in databases}
    files: list[dict[str, Any]] = []
    for path in sorted(output_home.rglob("*")):
        if not _is_regular_file(path):
            continue
        relative = path.relative_to(output_home).as_posix()
        if relative in database_paths:
            continue
        files.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })

    manifest = {
        "format_version": FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_home": str(home),
        "copied_regular_files": copied,
        "durable_files": files,
        "database_files": databases,
        "database_bytes": sum(item["bytes"] for item in databases),
        "durable_file_bytes": sum(item["bytes"] for item in files),
    }
    (work / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Snapshot ready: {len(files)} durable files "
        f"({manifest['durable_file_bytes'] / (1024 * 1024):.2f} MiB), "
        f"{len(databases)} SQLite DB(s) "
        f"({manifest['database_bytes'] / (1024 * 1024):.2f} MiB)."
    )
    for item in sorted(files, key=lambda entry: entry["bytes"], reverse=True)[:10]:
        if item["bytes"] >= 1024 * 1024:
            print(f"Large retained state file: {item['path']} ({item['bytes'] / (1024 * 1024):.2f} MiB)")
    return manifest


def _validate_manifest(source_home: Path, manifest: dict[str, Any]) -> None:
    version = manifest.get("format_version", 1)
    if version not in (1, FORMAT_VERSION):
        raise RuntimeError(f"unsupported Hermes backup format: {version!r}")

    for item in manifest.get("durable_files", []):
        relative = _safe_relative(item["path"])
        path = source_home / relative
        if not _is_regular_file(path):
            raise RuntimeError(f"backup file is missing or unsafe: {relative}")
        if path.stat().st_size != item.get("bytes"):
            raise RuntimeError(f"backup file size mismatch: {relative}")
        expected = item.get("sha256")
        if expected and _sha256(path) != expected:
            raise RuntimeError(f"backup file checksum mismatch: {relative}")

    for item in manifest.get("database_files", []):
        relative = _safe_relative(item["path"])
        path = source_home / relative
        if not _is_regular_file(path):
            raise RuntimeError(f"SQLite backup is missing or unsafe: {relative}")
        if path.stat().st_size != item.get("bytes"):
            raise RuntimeError(f"SQLite backup size mismatch: {relative}")
        expected = item.get("sha256")
        if expected and _sha256(path) != expected:
            raise RuntimeError(f"SQLite backup checksum mismatch: {relative}")
        counts = _database_info(path)
        expected_counts = item.get("row_counts") or {}
        if counts != expected_counts and expected_counts:
            raise RuntimeError(
                f"SQLite row-count mismatch for {relative}: expected {expected_counts}, got {counts}"
            )


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"cannot read Hermes backup manifest {path}: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("database_files", []), list):
        raise RuntimeError("invalid Hermes backup manifest")
    return value


def _restore_legacy_model_settings(
    source_home: Path, home: Path, manifest_out: Path,
) -> None:
    """Seed a clean runtime from a pre-v2 archive without reviving old user data.

    Format-v1 archives were created before the operator asked for a fresh data
    baseline. They may contain old conversations, databases and memories, so
    only the three original model/configuration files are eligible for import.
    The first format-v2 snapshot, taken by the new runner, becomes the first
    generation that is allowed to restore durable session data.
    """
    allowed = (
        Path("config.yaml"),
        Path(".env"),
        Path("plugin-data/custom-model-manager/models.json"),
    )
    copied: list[dict[str, Any]] = []

    def copy_model_files(source_root: Path, destination_root: Path) -> None:
        for relative in allowed:
            source = source_root / relative
            destination = destination_root / relative
            if _copy_one(source, destination):
                copied.append({
                    "path": destination.relative_to(home).as_posix(),
                    "bytes": destination.stat().st_size,
                    "sha256": _sha256(destination),
                })

    copy_model_files(source_home, home)
    profiles = source_home / "profiles"
    if profiles.is_dir() and not profiles.is_symlink():
        for profile in sorted(profiles.iterdir()):
            if profile.is_symlink() or not profile.is_dir():
                continue
            copy_model_files(profile, home / "profiles" / profile.name)

    if not (home / "config.yaml").is_file():
        raise RuntimeError("legacy backup does not contain config.yaml; refusing a blank model configuration")

    # Write a normalized v2 validation record for precisely what was imported.
    # It deliberately has no database/session entries from the legacy archive.
    normalized = {
        "format_version": FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "legacy_seed": "model-settings-only",
        "durable_files": copied,
        "database_files": [],
        "database_bytes": 0,
        "durable_file_bytes": sum(item["bytes"] for item in copied),
    }
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(
        json.dumps(normalized, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    validate_home(home, normalized)
    print(
        "Legacy format-v1 baseline imported model settings only; "
        "old sessions, SQLite databases, memories, queues and other runtime data were not restored."
    )


def restore_home(source_home: Path, manifest_path: Path, home: Path, manifest_out: Path) -> None:
    source_home = source_home.resolve()
    home = home.resolve()
    manifest = _load_manifest(manifest_path)
    version = manifest.get("format_version", 1)

    if version == 1:
        _restore_legacy_model_settings(source_home, home, manifest_out)
        return
    if version != FORMAT_VERSION:
        raise RuntimeError(f"unsupported Hermes backup format: {version!r}")

    _validate_manifest(source_home, manifest)

    _copy_durable_layout(source_home, home)
    for item in manifest.get("database_files", []):
        relative = _safe_relative(item["path"])
        source = source_home / relative
        destination = home / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        destination.chmod(0o600)

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, manifest_out)
    validate_home(home, manifest)
    print(
        f"Restored and verified Hermes state: {len(manifest.get('database_files', []))} SQLite DB(s), "
        f"{len(manifest.get('durable_files', []))} checksummed durable file(s)."
    )


def validate_home(home: Path, manifest: dict[str, Any]) -> None:
    home = home.resolve()
    for item in manifest.get("durable_files", []):
        relative = _safe_relative(item["path"])
        path = home / relative
        if not _is_regular_file(path) or path.stat().st_size != item.get("bytes"):
            raise RuntimeError(f"restored durable file is missing or changed: {relative}")
        expected = item.get("sha256")
        if expected and _sha256(path) != expected:
            raise RuntimeError(f"restored durable file checksum mismatch: {relative}")

    for item in manifest.get("database_files", []):
        relative = _safe_relative(item["path"])
        path = home / relative
        if not _is_regular_file(path):
            raise RuntimeError(f"restored SQLite database is missing: {relative}")
        if path.stat().st_size != item.get("bytes"):
            raise RuntimeError(f"restored SQLite database size mismatch: {relative}")
        expected = item.get("sha256")
        if expected and _sha256(path) != expected:
            raise RuntimeError(f"restored SQLite database checksum mismatch: {relative}")
        counts = _database_info(path)
        expected_counts = item.get("row_counts") or {}
        if expected_counts and counts != expected_counts:
            raise RuntimeError(f"restored SQLite row-count mismatch for {relative}: {counts}")
        summary = ", ".join(f"{k}={v}" for k, v in counts.items()) or "no tracked tables"
        print(f"Restored SQLite OK: {relative} ({summary})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    snapshot_parser = sub.add_parser("snapshot")
    snapshot_parser.add_argument("--home", type=Path, required=True)
    snapshot_parser.add_argument("--work", type=Path, required=True)

    restore_parser = sub.add_parser("restore")
    restore_parser.add_argument("--source", type=Path, required=True)
    restore_parser.add_argument("--manifest", type=Path, required=True)
    restore_parser.add_argument("--home", type=Path, required=True)
    restore_parser.add_argument("--manifest-out", type=Path, required=True)

    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--home", type=Path, required=True)
    validate_parser.add_argument("--manifest", type=Path, required=True)

    args = parser.parse_args()
    try:
        if args.command == "snapshot":
            snapshot_home(args.home, args.work)
        elif args.command == "restore":
            restore_home(args.source, args.manifest, args.home, args.manifest_out)
        elif args.command == "validate":
            validate_home(args.home, _load_manifest(args.manifest))
        return 0
    except Exception as exc:
        print(f"ERROR: Hermes durable-state backup: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
