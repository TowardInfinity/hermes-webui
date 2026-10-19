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


def test_reimport_backfill_skips_a_changed_model():
    from api.agent_sessions import model_provider_backfill

    assert model_provider_backfill("grok-4.7", None, "grok-4.7", "xai-oauth") == "xai-oauth"
    assert model_provider_backfill("deepseek-v4.1-flash", None, "grok-4.7", "xai-oauth") is None
    assert model_provider_backfill("grok-4.7", "opencode-go", "grok-4.7", "xai-oauth") is None
    assert model_provider_backfill("unknown", None, "grok-4.7", "xai-oauth") == "xai-oauth"
    assert model_provider_backfill("grok-4.7", None, "grok-4.7", "custom") == "custom"
    assert model_provider_backfill("grok-4.7", None, "grok-4.7", "auto") is None


def test_claim_row_reads_provider_without_the_recent_list(tmp_path):
    from api.agent_sessions import read_state_db_claim_row

    db = tmp_path / "state.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT,
            title TEXT,
            model TEXT,
            cwd TEXT,
            started_at REAL,
            ended_at REAL,
            model_config TEXT,
            billing_provider TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "old-desktop",
            "desktop",
            "Older chat",
            "grok-4.7",
            "/tmp",
            10.0,
            11.0,
            json.dumps({"provider": "xai-oauth"}),
            "xai-oauth",
        ),
    )
    conn.commit()
    conn.close()

    row = read_state_db_claim_row(db, "old-desktop")
    assert row["model"] == "grok-4.7"
    assert row["model_provider"] == "xai-oauth"
    assert "model_config" not in row
    assert "billing_provider" not in row


def test_bare_billing_buckets_are_not_imported_as_providers(tmp_path):
    from api.agent_sessions import _model_provider_from_agent_row, read_state_db_claim_row

    assert _model_provider_from_agent_row({"billing_provider": "custom"}) is None
    assert _model_provider_from_agent_row({"billing_provider": "auto"}) is None
    assert _model_provider_from_agent_row({"model_config": json.dumps({"provider": "custom"})}) == "custom"
    assert _model_provider_from_agent_row({"billing_provider": "xai-oauth"}) == "xai-oauth"
    assert _model_provider_from_agent_row({
        "model_config": json.dumps({"provider": "custom:my-relay"}),
        "billing_provider": "custom",
    }) == "custom:my-relay"
    assert _model_provider_from_agent_row({
        "model_config": json.dumps({
            "provider": "xai-oauth",
            "gateway_runtime": {"provider": "opencode-go"},
        }),
        "billing_provider": "custom",
    }) == "opencode-go"
    assert _model_provider_from_agent_row({
        "model_config": json.dumps({"gateway_runtime": {"provider": "custom", "base_url": "http://relay"}}),
    }) is None
    assert _model_provider_from_agent_row({
        "model_config": json.dumps({"gateway_runtime": {"provider": "custom:my-relay"}}),
    }) == "custom:my-relay"

    db = tmp_path / "state.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT,
            title TEXT,
            model TEXT,
            billing_provider TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO sessions VALUES (?, ?, ?, ?, ?)",
        ("bucket", "cli", "Bucket", "grok-4.7", "custom"),
    )
    conn.commit()
    conn.close()
    row = read_state_db_claim_row(db, "bucket")
    assert row["model"] == "grok-4.7"
    assert "model_provider" not in row


def test_claim_row_survives_a_db_without_provider_columns(tmp_path):
    from api.agent_sessions import read_state_db_claim_row

    db = tmp_path / "state.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT,
            title TEXT,
            model TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO sessions VALUES (?, ?, ?, ?)",
        ("plain", "cli", "Plain", "grok-4.7"),
    )
    conn.commit()
    conn.close()

    row = read_state_db_claim_row(db, "plain")
    assert row["source"] == "cli"
    assert row["model"] == "grok-4.7"
    assert "model_provider" not in row


# ── Routability gate ─────────────────────────────────────────────────────────
# Import may keep a Desktop/CLI provider only when the owning profile can route
# it. A provider the profile no longer has must stay unset, so the first send
# uses the profile provider exactly as before #8249 instead of failing.

import io
import sys
import types

import pytest

_GATE_PROFILE = "p8249gate"
_GATE_CONFIG = """\
model:
  provider: custom:available
  default: grok-4.7
custom_providers:
  - name: available
    base_url: http://127.0.0.1:18888/v1
    api_key: sk-available
  - name: relay-b
    base_url: http://127.0.0.1:18999/v1
    api_key: sk-relay-b
providers:
  keyed-builtin:
    api_key: sk-keyed
  disabled-builtin:
    api_key: sk-disabled
    enabled: false
"""


@pytest.fixture
def gate_profile(tmp_path, monkeypatch):
    """A named profile whose home holds only ``_GATE_CONFIG``."""
    import api.profiles as profiles

    home = tmp_path / "profile-home"
    home.mkdir()
    (home / "config.yaml").write_text(_GATE_CONFIG, encoding="utf-8")
    real = profiles.get_hermes_home_for_profile
    monkeypatch.setattr(
        profiles,
        "get_hermes_home_for_profile",
        lambda name: home if name == _GATE_PROFILE else real(name),
    )
    return home


@pytest.fixture
def stub_resolver(monkeypatch):
    """Stand-in for the Agent resolver: only ``openrouter`` has a credential."""
    calls = []

    def resolve_runtime_provider(*, requested=None, **_kwargs):
        calls.append(requested)
        if requested == "openrouter":
            return {"provider": "openrouter", "api_key": "sk-or", "base_url": "https://openrouter.ai/api/v1"}
        raise RuntimeError(f"No usable credentials found for provider {requested!r}")

    module = types.ModuleType("hermes_cli.runtime_provider")
    module.resolve_runtime_provider = resolve_runtime_provider
    monkeypatch.setitem(sys.modules, "hermes_cli.runtime_provider", module)
    return calls


def test_gate_keeps_only_providers_the_owning_profile_can_route(gate_profile, stub_resolver):
    from api.routes import _routable_imported_model_provider as gate

    # Named custom relays: kept only when this profile's config owns them.
    assert gate("custom:relay-b", _GATE_PROFILE) == "custom:relay-b"
    assert gate("custom:removed-relay", _GATE_PROFILE) is None
    # The profile's own provider is routed the same way an unpinned session is.
    assert gate("custom:available", _GATE_PROFILE) == "custom:available"
    # A bare `custom` names the profile's own endpoint. This profile is on a
    # named relay, so the Agent would resolve it to an ambient fallback.
    assert gate("custom", _GATE_PROFILE) is None
    # Built-ins: a providers: entry, or a credential from the Agent resolver.
    assert gate("keyed-builtin", _GATE_PROFILE) == "keyed-builtin"
    assert gate("openrouter", _GATE_PROFILE) == "openrouter"
    assert gate("xai-oauth", _GATE_PROFILE) is None
    # A disabled providers: entry is not a route; the resolver decides.
    assert gate("disabled-builtin", _GATE_PROFILE) is None
    assert gate(None, _GATE_PROFILE) is None
    assert gate("default", _GATE_PROFILE) is None
    # Config answered every custom and providers: case without the resolver.
    assert stub_resolver == ["openrouter", "xai-oauth", "disabled-builtin"]


def test_gate_fails_closed_when_the_resolver_is_unavailable(gate_profile, monkeypatch):
    from api.routes import _routable_imported_model_provider as gate

    monkeypatch.setitem(sys.modules, "hermes_cli.runtime_provider", None)
    assert gate("openrouter", _GATE_PROFILE) is None
    # Config-only answers do not need the Agent.
    assert gate("custom:relay-b", _GATE_PROFILE) == "custom:relay-b"


def _installed_agent_resolver():
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider
    except Exception:
        return None
    return resolve_runtime_provider


@pytest.mark.skipif(_installed_agent_resolver() is None, reason="installed hermes-agent resolver not importable")
def test_gate_uses_the_installed_agent_resolver(gate_profile, monkeypatch):
    from api.routes import _routable_imported_model_provider as gate

    for name in ("OPENROUTER_API_KEY", "XAI_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    (gate_profile / ".env").write_text("OPENROUTER_API_KEY=sk-or-gate-test\n", encoding="utf-8")

    assert gate("openrouter", _GATE_PROFILE) == "openrouter"
    # No xAI login and no DeepSeek key in this profile.
    assert gate("xai-oauth", _GATE_PROFILE) is None
    assert gate("deepseek", _GATE_PROFILE) is None
    assert gate("custom:removed-relay", _GATE_PROFILE) is None


class _ImportHandler:
    def __init__(self):
        self.status = None
        self.headers = {}
        self.wfile = io.BytesIO()

    def send_response(self, status):
        self.status = status

    def send_header(self, key, value):
        self.headers[key] = value

    def end_headers(self):
        pass


def _desktop_meta(provider):
    return {
        "session_id": "unused",
        "title": "Desktop session",
        "model": "grok-4.7",
        "model_provider": provider,
        "profile": _GATE_PROFILE,
        "source_tag": "desktop",
        "raw_source": "desktop",
        "session_source": "desktop",
        "created_at": 10.0,
        "updated_at": 11.0,
    }


def _stub_cli_store(monkeypatch, routes, provider):
    msgs = [{"role": "user", "content": "hi", "timestamp": 10.0}]
    monkeypatch.setattr(routes, "_resolve_cli_import_metadata", lambda *a, **k: _desktop_meta(provider))
    monkeypatch.setattr(routes, "get_cli_session_messages", lambda *a, **k: list(msgs))
    monkeypatch.setattr(routes, "_session_visible_to_active_profile", lambda *a, **k: True)
    captured = {}
    monkeypatch.setattr(routes, "j", lambda h, payload, status=200, *a, **k: captured.update(payload) or True)
    monkeypatch.setattr(routes, "bad", lambda h, msg, status=400: captured.update(error=(msg, status)) or True)
    return captured


@pytest.mark.parametrize(
    ("desktop_provider", "expected"),
    [
        ("custom:relay-b", "custom:relay-b"),
        ("custom:removed-relay", None),
        ("xai-oauth", None),
    ],
)
def test_first_import_pins_only_a_routable_provider(gate_profile, stub_resolver, monkeypatch, desktop_provider, expected):
    import api.routes as routes
    from api.models import SESSION_DIR, Session

    sid = "import_gate_8249"
    (SESSION_DIR / f"{sid}.json").unlink(missing_ok=True)
    captured = _stub_cli_store(monkeypatch, routes, desktop_provider)
    try:
        routes._handle_session_import_cli(_ImportHandler(), {"session_id": sid})
        assert "error" not in captured, captured
        loaded = Session.load(sid)
        assert loaded.model == "grok-4.7"
        assert loaded.model_provider == expected
    finally:
        (SESSION_DIR / f"{sid}.json").unlink(missing_ok=True)


@pytest.mark.parametrize(
    ("desktop_provider", "expected"),
    [
        ("custom:relay-b", "custom:relay-b"),
        ("custom:removed-relay", None),
        ("xai-oauth", None),
    ],
)
def test_reimport_backfills_only_a_routable_provider(gate_profile, stub_resolver, monkeypatch, desktop_provider, expected):
    import api.routes as routes
    from api.models import SESSION_DIR, Session, import_cli_session

    sid = "reimport_gate_8249"
    (SESSION_DIR / f"{sid}.json").unlink(missing_ok=True)
    # A sidecar imported before #8249: same model, no provider.
    import_cli_session(
        sid,
        "Desktop session",
        [{"role": "user", "content": "hi", "timestamp": 10.0}],
        model="grok-4.7",
        profile=_GATE_PROFILE,
        created_at=10.0,
        updated_at=11.0,
    )
    captured = _stub_cli_store(monkeypatch, routes, desktop_provider)
    try:
        routes._handle_session_import_cli(_ImportHandler(), {"session_id": sid})
        assert "error" not in captured, captured
        assert Session.load(sid).model_provider == expected
    finally:
        (SESSION_DIR / f"{sid}.json").unlink(missing_ok=True)


@pytest.mark.parametrize(
    ("desktop_provider", "expected"),
    [
        ("custom:relay-b", "custom:relay-b"),
        ("custom:removed-relay", None),
        ("xai-oauth", None),
    ],
)
def test_saved_link_claim_pins_only_a_routable_provider(gate_profile, stub_resolver, monkeypatch, desktop_provider, expected):
    import api.routes as routes

    sid = "claim_gate_8249"
    monkeypatch.setattr(routes, "get_cli_session_messages", lambda *a, **k: [{"role": "user", "content": "hi"}])
    monkeypatch.setattr(routes, "_session_index_marks_was_webui", lambda *a, **k: False)
    session, reason = routes._claim_or_synthesize_cli_session(sid, _desktop_meta(desktop_provider))
    assert reason == "materialized"
    assert session.model == "grok-4.7"
    assert session.model_provider == expected
