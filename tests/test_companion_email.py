"""Owner-scope tests for the companion email endpoints.

The security crux is account selection: every endpoint must resolve the token's
real owner and gate on _assert_owns_account(account_id, owner) BEFORE reading
creds or opening a connection, and the accounts list must never leak imap/smtp
passwords. IMAP/SMTP I/O itself isn't unit-testable here, so we drive the
ownership gate via a fake routes.email_helpers module.
"""

import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if "core.database" not in sys.modules:
    _db = types.ModuleType("core.database")
    _db.SessionLocal = MagicMock()
    _db.EmailAccount = MagicMock()
    sys.modules["core.database"] = _db

import companion.routes as companion_routes
from companion.routes import setup_companion_routes


# --- account list harness (real-ish core.database stub) --------------------

class _Predicate:
    def __init__(self, check):
        self._check = check

    def __call__(self, row):
        return self._check(row)

    def __or__(self, other):
        return _Predicate(lambda r: self(r) or other(r))


class _Column:
    def __init__(self, name):
        self.name = name

    def __eq__(self, value):
        return _Predicate(lambda r: getattr(r, self.name) == value)


class _EmailAccount:
    owner = _Column("owner")


class _Query:
    def __init__(self, rows):
        self._rows = list(rows)

    def filter(self, *ps):
        self._rows = [r for r in self._rows if all(p(r) for p in ps)]
        return self

    def all(self):
        return list(self._rows)


class _DB:
    def __init__(self, rows):
        self._rows = rows
        self.closed = False

    def query(self, model):
        assert model is _EmailAccount
        return _Query(self._rows)

    def close(self):
        self.closed = True


def _acct(id, owner, *, name="Acct"):
    return SimpleNamespace(
        id=id, owner=owner, name=name, from_address=f"{name}@x.test",
        enabled=True, is_default=False,
        imap_password="IMAP-SECRET", smtp_password="SMTP-SECRET",
    )


def _route(path, method):
    for r in setup_companion_routes().routes:
        if getattr(r, "path", "") == path and method in getattr(r, "methods", set()):
            return r.endpoint
    raise AssertionError(f"{method} {path} not found")


def _request(*, owner, scopes=("companion",)):
    return SimpleNamespace(state=SimpleNamespace(
        api_token=True, api_token_owner=owner, api_token_scopes=list(scopes), current_user="api",
    ))


@pytest.fixture
def accounts_db(monkeypatch):
    def _install(rows):
        db = _DB(rows)
        m = sys.modules["core.database"]
        monkeypatch.setattr(m, "SessionLocal", lambda: db)
        monkeypatch.setattr(m, "EmailAccount", _EmailAccount)
        monkeypatch.setattr(companion_routes, "get_current_user", lambda request: "api")
        return db
    return _install


def test_accounts_scoped_to_owner_and_shared(accounts_db):
    accounts_db([_acct("a", "alice"), _acct("s", None), _acct("b", "bob")])
    res = _route("/api/companion/email/accounts", "GET")(_request(owner="alice"))
    assert {a["id"] for a in res["items"]} == {"a", "s"}


def test_accounts_never_leak_passwords(accounts_db):
    accounts_db([_acct("a", "alice")])
    res = _route("/api/companion/email/accounts", "GET")(_request(owner="alice"))
    blob = repr(res)
    assert "IMAP-SECRET" not in blob and "SMTP-SECRET" not in blob
    assert set(res["items"][0]) == {"id", "name", "from_address", "enabled", "is_default"}


def test_accounts_requires_scope(accounts_db):
    accounts_db([_acct("a", "alice")])
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/email/accounts", "GET")(_request(owner="alice", scopes=("chat",)))
    assert exc.value.status_code == 403


# --- ownership gate on send (fake email_helpers) ---------------------------

@pytest.fixture
def fake_helpers(monkeypatch):
    """Install a controllable fake routes.email_helpers for the send/list gate."""
    mod = types.ModuleType("routes.email_helpers")
    calls = {"asserted": []}

    def _assert_owns_account(account_id, owner):
        calls["asserted"].append((account_id, owner))
        if account_id == "foreign":
            raise HTTPException(404, "Account not found")

    def _get_email_config(account_id=None, owner=""):
        return {"smtp_host": "smtp.x.test", "smtp_user": "u", "smtp_password": "p",
                "smtp_port": 465, "smtp_security": "ssl", "from_address": "me@x.test"}

    sent = {}

    def _send_smtp_message(cfg, from_addr, recipients, message, timeout=30):
        sent["from"] = from_addr
        sent["recipients"] = recipients

    mod._assert_owns_account = _assert_owns_account
    mod._get_email_config = _get_email_config
    mod._send_smtp_message = _send_smtp_message
    mod._imap = MagicMock()
    mod._decode_header = lambda v: v
    mod._extract_text = lambda m: ""
    monkeypatch.setitem(sys.modules, "routes.email_helpers", mod)
    monkeypatch.setattr(companion_routes, "get_current_user", lambda request: "api")
    return calls, sent


def test_send_gates_on_ownership_then_sends(fake_helpers):
    calls, sent = fake_helpers
    res = _route("/api/companion/email/send", "POST")(
        _request(owner="alice"), account_id="a", to="x@y.test", subject="hi", body="yo",
    )
    assert res["status"] == "sent"
    assert calls["asserted"] == [("a", "alice")]      # gate ran with the resolved owner
    assert sent["recipients"] == ["x@y.test"]


def test_send_cross_owner_account_is_404_and_does_not_send(fake_helpers):
    calls, sent = fake_helpers
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/email/send", "POST")(
            _request(owner="alice"), account_id="foreign", to="x@y.test", subject="", body="",
        )
    assert exc.value.status_code == 404
    assert sent == {}  # never reached SMTP


def test_send_requires_valid_recipient(fake_helpers):
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/email/send", "POST")(
            _request(owner="alice"), account_id="a", to="not-an-email", subject="", body="",
        )
    assert exc.value.status_code == 400


def test_send_requires_scope(fake_helpers):
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/email/send", "POST")(
            _request(owner="alice", scopes=("chat",)), account_id="a", to="x@y.test", subject="", body="",
        )
    assert exc.value.status_code == 403


def test_send_requires_resolved_owner(fake_helpers):
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/email/send", "POST")(
            _request(owner=None), account_id="a", to="x@y.test", subject="", body="",
        )
    assert exc.value.status_code == 403


def test_messages_gates_on_ownership(fake_helpers):
    # Cross-owner account → 404 from the gate, before any IMAP use.
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/email/messages", "GET")(_request(owner="alice"), account_id="foreign")
    assert exc.value.status_code == 404


def test_messages_null_owner_is_403(fake_helpers):
    # A null/empty owner makes _assert_owns_account a no-op — it must be refused
    # BEFORE any account access, not allowed to reach an arbitrary mailbox.
    calls, _ = fake_helpers
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/email/messages", "GET")(_request(owner=None), account_id="a")
    assert exc.value.status_code == 403
    assert calls["asserted"] == []  # never even called the (no-op) ownership check


def test_read_message_null_owner_is_403(fake_helpers):
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/email/message/{uid}", "GET")(_request(owner=None), uid="1", account_id="a")
    assert exc.value.status_code == 403
