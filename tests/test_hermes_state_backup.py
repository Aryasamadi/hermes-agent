from __future__ import annotations

import sqlite3
from pathlib import Path

from scripts.hermes_state_backup import restore_home, snapshot_home, validate_home


def _make_state_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.executescript(
            """
            CREATE TABLE sessions (id TEXT PRIMARY KEY);
            CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, content TEXT);
            CREATE TABLE gateway_routing (key TEXT PRIMARY KEY, value TEXT);
            INSERT INTO sessions VALUES ('session-1');
            INSERT INTO messages VALUES (1, 'session-1', 'new conversation');
            INSERT INTO gateway_routing VALUES ('telegram:123', 'session-1');
            """
        )


def test_snapshot_and_restore_only_durable_hermes_state(tmp_path: Path) -> None:
    source = tmp_path / "source-home"
    source.mkdir()
    (source / "config.yaml").write_text("model: test\n", encoding="utf-8")
    (source / ".env").write_text("API_KEY=test-secret\n", encoding="utf-8")
    (source / "memories").mkdir()
    (source / "memories" / "MEMORY.md").write_text("Important new memory\n", encoding="utf-8")
    (source / "memories" / "USER.md").write_text("User profile\n", encoding="utf-8")
    (source / "sessions").mkdir()
    (source / "sessions" / "sessions.json").write_text('{"telegram:123": "session-1"}', encoding="utf-8")
    (source / "pending_messages").mkdir()
    (source / "pending_messages" / "pending-1.json").write_text('{"session_id":"session-1"}', encoding="utf-8")
    (source / "cron").mkdir()
    (source / "cron" / "jobs.json").write_text('{"jobs":[]}', encoding="utf-8")
    _make_state_db(source / "state.db")
    _make_state_db(source / "profiles" / "work" / "state.db")

    # These are deliberately not durable Hermes state and must not enter the backup.
    (source / "cache").mkdir()
    (source / "cache" / "huge-cache.bin").write_bytes(b"x" * 4096)
    (source / "hermes-agent").mkdir()
    (source / "hermes-agent" / "source.py").write_text("ephemeral checkout", encoding="utf-8")

    work = tmp_path / "snapshot"
    manifest = snapshot_home(source, work)
    assert {item["path"] for item in manifest["database_files"]} == {
        "state.db", "profiles/work/state.db"
    }
    assert any(item["path"] == "memories/MEMORY.md" for item in manifest["durable_files"])
    assert not any("huge-cache" in item["path"] for item in manifest["durable_files"])

    restored = tmp_path / "restored-home"
    manifest_out = tmp_path / "restored-manifest.json"
    restore_home(work / "hermes", work / "manifest.json", restored, manifest_out)
    validate_home(restored, manifest)

    assert (restored / "memories" / "MEMORY.md").read_text(encoding="utf-8") == "Important new memory\n"
    assert (restored / "sessions" / "sessions.json").is_file()
    assert (restored / "pending_messages" / "pending-1.json").is_file()
    assert (restored / "cron" / "jobs.json").is_file()
    assert not (restored / "cache").exists()
    assert not (restored / "hermes-agent").exists()

    with sqlite3.connect(restored / "state.db") as db:
        assert db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1


def test_legacy_format_v1_restore_is_filtered_to_durable_paths(tmp_path: Path) -> None:
    source = tmp_path / "legacy-home"
    source.mkdir()
    (source / "config.yaml").write_text("model: old-backup\n", encoding="utf-8")
    (source / "sessions").mkdir()
    (source / "sessions" / "sessions.json").write_text("{}", encoding="utf-8")
    _make_state_db(source / "state.db")
    (source / "cache").mkdir()
    (source / "cache" / "old.bin").write_bytes(b"do not restore")

    manifest_path = tmp_path / "legacy-manifest.json"
    import json
    manifest_path.write_text(json.dumps({
        "format_version": 1,
        "database_files": [{"path": "state.db", "bytes": (source / "state.db").stat().st_size}],
    }), encoding="utf-8")

    restored = tmp_path / "legacy-restored"
    restore_home(source, manifest_path, restored, tmp_path / "legacy-restored-manifest.json")

    assert (restored / "state.db").is_file()
    assert (restored / "sessions" / "sessions.json").is_file()
    assert not (restored / "cache").exists()
