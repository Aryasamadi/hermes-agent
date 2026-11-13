from __future__ import annotations

import json
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


def test_legacy_format_v1_restores_model_settings_only_and_starts_clean(tmp_path: Path) -> None:
    source = tmp_path / "legacy-home"
    source.mkdir()
    (source / "config.yaml").write_text("model: old-backup\n", encoding="utf-8")
    (source / ".env").write_text("API_KEY=old-model-key\n", encoding="utf-8")
    (source / "plugin-data" / "custom-model-manager").mkdir(parents=True)
    (source / "plugin-data" / "custom-model-manager" / "models.json").write_text(
        '{"models":[{"name":"saved-model"}]}', encoding="utf-8"
    )
    (source / "plugin-data" / "unrelated").mkdir(parents=True)
    (source / "plugin-data" / "unrelated" / "state.json").write_text(
        '{"old":"plugin state"}', encoding="utf-8"
    )

    # A v1 archive may contain these old records; none may be restored.
    (source / "sessions").mkdir()
    (source / "sessions" / "sessions.json").write_text('{"old":"session"}', encoding="utf-8")
    (source / "sessions" / "old-session.jsonl").write_text('{"role":"user","content":"old"}\n', encoding="utf-8")
    (source / "memories").mkdir()
    (source / "memories" / "MEMORY.md").write_text("old memory\n", encoding="utf-8")
    (source / "pending_messages").mkdir()
    (source / "pending_messages" / "pending-old.json").write_text('{"old":true}', encoding="utf-8")
    (source / "cron").mkdir()
    (source / "cron" / "jobs.json").write_text('{"jobs":["old"]}', encoding="utf-8")
    _make_state_db(source / "state.db")
    _make_state_db(source / "profiles" / "work" / "state.db")
    (source / "profiles" / "work" / "plugin-data" / "custom-model-manager").mkdir(parents=True)
    (source / "profiles" / "work" / "config.yaml").write_text(
        "model: profile-model\n", encoding="utf-8"
    )
    (source / "profiles" / "work" / "plugin-data" / "custom-model-manager" / "models.json").write_text(
        '{"models":[{"name":"profile-saved-model"}]}', encoding="utf-8"
    )
    (source / "cache").mkdir()
    (source / "cache" / "old.bin").write_bytes(b"do not restore")

    manifest_path = tmp_path / "legacy-manifest.json"
    manifest_path.write_text(json.dumps({
        "format_version": 1,
        "database_files": [
            {"path": "state.db", "bytes": (source / "state.db").stat().st_size},
            {"path": "profiles/work/state.db", "bytes": (source / "profiles/work/state.db").stat().st_size},
        ],
    }), encoding="utf-8")

    restored = tmp_path / "legacy-restored"
    manifest_out = tmp_path / "legacy-restored-manifest.json"
    restore_home(source, manifest_path, restored, manifest_out)

    # Keep model/provider setup only.
    assert (restored / "config.yaml").read_text(encoding="utf-8") == "model: old-backup\n"
    assert (restored / ".env").read_text(encoding="utf-8") == "API_KEY=old-model-key\n"
    assert (restored / "plugin-data" / "custom-model-manager" / "models.json").is_file()
    assert (restored / "profiles" / "work" / "config.yaml").is_file()
    assert (restored / "profiles" / "work" / "plugin-data" / "custom-model-manager" / "models.json").is_file()

    # No old database, transcript, memory, pending work, cron jobs or unrelated plugin data.
    assert not (restored / "state.db").exists()
    assert not (restored / "profiles" / "work" / "state.db").exists()
    assert not (restored / "sessions").exists()
    assert not (restored / "memories").exists()
    assert not (restored / "pending_messages").exists()
    assert not (restored / "cron").exists()
    assert not (restored / "plugin-data" / "unrelated").exists()
    assert not (restored / "cache").exists()

    normalized = json.loads(manifest_out.read_text(encoding="utf-8"))
    assert normalized["format_version"] == 2
    assert normalized["database_files"] == []
    assert {item["path"] for item in normalized["durable_files"]} == {
        "config.yaml",
        ".env",
        "plugin-data/custom-model-manager/models.json",
        "profiles/work/config.yaml",
        "profiles/work/plugin-data/custom-model-manager/models.json",
    }
