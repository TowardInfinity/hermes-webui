"""Regression for #8249.

Opening a Desktop/CLI session in WebUI must keep the provider that session was
created on. ``sessions.model`` is a bare id. If import drops the provider, chat
start attaches the profile's active provider and bills the wrong account.
"""
import json
import sqlite3

from api.agent_sessions import (
    _model_provider_from_agent_row,
    read_importable_agent_session_rows,
)


def test_model_provider_prefers_selected_provider_over_billing_identity():
    row = {
        "model_config": json.dumps({"model": "grok-4.7", "provider": "xai-oauth"}),
        "billing_provider": "opencode-go",
    }
    assert _model_provider_from_agent_row(row) == "xai-oauth"


def test_model_provider_falls_back_to_billing_identity():
    assert _model_provider_from_agent_row({"billing_provider": "xai-oauth"}) == "xai-oauth"
    assert _model_provider_from_agent_row({"model_config": json.dumps({"provider": "default"})}) is None
    assert _model_provider_from_agent_row({}) is None


def _desktop_db(path, *, model_config, billing_provider):
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            title TEXT,
            model TEXT,
            message_count INTEGER,
            started_at REAL,
            source TEXT,
            session_source TEXT,
            model_config TEXT,
            billing_provider TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO sessions (
            id, title, model, message_count, started_at, source, session_source,
            model_config, billing_provider
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "desktop-1",
            "Desktop session",
            "grok-4.7",
            1,
            1000.0,
            "desktop",
            "desktop",
            model_config,
            billing_provider,
        ),
    )
    conn.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, timestamp REAL)"
    )
    conn.execute(
        "INSERT INTO messages (session_id, role, timestamp) VALUES (?, ?, ?)",
        ("desktop-1", "user", 1001.0),
    )
    conn.commit()
    conn.close()


def test_importable_desktop_row_publishes_selected_provider_not_raw_config(tmp_path):
    db = tmp_path / "state.db"
    _desktop_db(
        db,
        model_config=json.dumps({"model": "grok-4.7", "provider": "xai-oauth"}),
        billing_provider="xai-oauth",
    )
    rows = read_importable_agent_session_rows(db, exclude_sources=None)
    match = [row for row in rows if row["id"] == "desktop-1"]
    assert len(match) == 1
    row = match[0]
    assert row["model"] == "grok-4.7"
    assert row["model_provider"] == "xai-oauth"
    assert "model_config" not in row
    assert "billing_provider" not in row


def test_importable_row_without_provider_does_not_invent_one(tmp_path):
    db = tmp_path / "state.db"
    _desktop_db(db, model_config=None, billing_provider=None)
    rows = read_importable_agent_session_rows(db, exclude_sources=None)
    row = next(item for item in rows if item["id"] == "desktop-1")
    assert row["model"] == "grok-4.7"
    assert "model_provider" not in row


def test_import_cli_session_preserves_model_provider():
    from api.models import SESSION_DIR, Session, import_cli_session

    sid = "import_provider_8249"
    path = SESSION_DIR / f"{sid}.json"
    path.unlink(missing_ok=True)
    try:
        session = import_cli_session(
            sid,
            "Imported",
            [{"role": "user", "content": "hi", "timestamp": 1.0}],
            model="grok-4.7",
            model_provider="xai-oauth",
            created_at=1.0,
            updated_at=2.0,
        )
        assert session.model_provider == "xai-oauth"
        loaded = Session.load(sid)
        assert loaded.model == "grok-4.7"
        assert loaded.model_provider == "xai-oauth"
    finally:
        path.unlink(missing_ok=True)
