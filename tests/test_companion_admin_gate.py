"""Tests for the companion ADMIN gate (companion/routes.py).

Admin-only mobile features (terminal/vault/mcp/cookbook/contacts) must be
reachable ONLY when a triple lock holds, fail-closed:
  1. the `companion_admin_enabled` server setting is on,
  2. the caller carries the `companion` scope, and
  3. the resolved owner is a server admin.

Same direct-helper style as test_companion_readonly.py: drive the pure-ish
predicate + the raising gate against mock request state, so the lock can't
silently regress into exposing RCE / secret export to a paired phone.
"""

import os
import sys
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NOTE: deliberately NO core.database stub here. The admin gate touches only
# src.settings + request.app.state.auth_manager — never the DB — so importing
# companion.routes doesn't require it. Installing a minimal core.database stub
# (as some sibling companion tests do) pollutes shared sys.modules and breaks
# the gallery/document gate tests in test_null_owner_gates.py when collected
# together; we avoid contributing to that.
from companion.routes import companion_admin_available, require_companion_admin


class _AuthManager:
    def __init__(self, admins):
        self._admins = set(admins)

    def is_admin(self, user):
        return user in self._admins


def _request(*, owner, scopes, admins, api_token=True, cookie_user=None, has_auth_manager=True):
    """Build a mock request. Bearer callers run as pseudo-user 'api' with an
    owner stamped on state; cookie callers pass cookie_user as current_user."""
    auth_manager = _AuthManager(admins) if has_auth_manager else None
    app = SimpleNamespace(state=SimpleNamespace(auth_manager=auth_manager))
    if api_token:
        state = SimpleNamespace(
            api_token=True,
            api_token_owner=owner,
            api_token_scopes=scopes,
            current_user="api",
        )
    else:
        state = SimpleNamespace(api_token=False, current_user=cookie_user)
    return SimpleNamespace(state=state, app=app)


@pytest.fixture
def setting(monkeypatch):
    """Control the companion_admin_enabled flag the gate reads lazily."""
    import src.settings as settings_mod

    def _set(on):
        monkeypatch.setattr(
            settings_mod,
            "get_setting",
            lambda key, default=None: on if key == "companion_admin_enabled" else default,
        )

    return _set


# --- all locks satisfied ----------------------------------------------------

def test_available_and_returns_owner_when_all_locks_pass(setting):
    setting(True)
    req = _request(owner="alice", scopes=["companion"], admins=["alice"])
    assert companion_admin_available(req) is True
    assert require_companion_admin(req) == "alice"


def test_cookie_admin_session_passes(setting):
    # A logged-in admin in a browser (no bearer token) also passes: cookie
    # sessions implicitly have companion scope, owner = current_user.
    setting(True)
    req = _request(owner=None, scopes=[], admins=["alice"], api_token=False, cookie_user="alice")
    assert companion_admin_available(req) is True
    assert require_companion_admin(req) == "alice"


# --- each lock, individually, must fail closed ------------------------------

def test_blocked_when_setting_off(setting):
    setting(False)
    req = _request(owner="alice", scopes=["companion"], admins=["alice"])
    assert companion_admin_available(req) is False
    with pytest.raises(HTTPException) as exc:
        require_companion_admin(req)
    assert exc.value.status_code == 403


def test_blocked_when_chat_scope_only(setting):
    # A plain chat token must never reach admin surface, even owned by an admin.
    setting(True)
    req = _request(owner="alice", scopes=["chat"], admins=["alice"])
    assert companion_admin_available(req) is False
    with pytest.raises(HTTPException) as exc:
        require_companion_admin(req)
    assert exc.value.status_code == 403


def test_blocked_when_owner_not_admin(setting):
    setting(True)
    req = _request(owner="bob", scopes=["companion"], admins=["alice"])
    assert companion_admin_available(req) is False
    with pytest.raises(HTTPException):
        require_companion_admin(req)


def test_blocked_when_owner_unresolved(setting):
    setting(True)
    req = _request(owner=None, scopes=["companion"], admins=["alice"])
    assert companion_admin_available(req) is False
    with pytest.raises(HTTPException):
        require_companion_admin(req)


def test_blocked_when_no_auth_manager(setting):
    setting(True)
    req = _request(owner="alice", scopes=["companion"], admins=["alice"], has_auth_manager=False)
    assert companion_admin_available(req) is False
    with pytest.raises(HTTPException):
        require_companion_admin(req)


def test_generic_403_does_not_disclose_which_lock(setting):
    # Two different failures (setting off vs owner-not-admin) raise the SAME
    # message, so a caller can't probe which lock stopped them. The gate reads
    # the setting at call time, so flip it right before each call.
    msgs = set()

    setting(False)  # fails on lock 1 (setting off)
    try:
        require_companion_admin(_request(owner="alice", scopes=["companion"], admins=["alice"]))
    except HTTPException as e:
        msgs.add(e.detail)

    setting(True)  # passes lock 1, fails on lock 3 (owner not admin)
    try:
        require_companion_admin(_request(owner="bob", scopes=["companion"], admins=["alice"]))
    except HTTPException as e:
        msgs.add(e.detail)

    assert len(msgs) == 1
