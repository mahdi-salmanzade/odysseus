"""Owner-scope tests for the companion personal-assistant endpoints.

GET returns only the caller's own assistant (per-owner singleton); PATCH
updates/creates the caller's assistant, refuses synthetic/non-human owners, and
never mutates another owner's row. Both require the companion scope.
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
    _db.CrewMember = MagicMock()
    sys.modules["core.database"] = _db

import companion.routes as companion_routes
from companion.routes import setup_companion_routes


class _Predicate:
    def __init__(self, check):
        self._check = check

    def __call__(self, row):
        return self._check(row)


class _Column:
    def __init__(self, name):
        self.name = name

    def __eq__(self, value):
        return _Predicate(lambda r: getattr(r, self.name) == value)


class _CrewMember:
    owner = _Column("owner")
    is_default_assistant = _Column("is_default_assistant")

    def __init__(self, **kw):
        # sensible defaults so _assistant_dict works on a freshly-created row
        self.id = None
        self.name = None
        self.user_name = None
        self.personality = None
        self.model = None
        self.greeting = None
        self.timezone = None
        self.avatar = None
        self.is_active = True
        self.is_default_assistant = False
        self.owner = None
        for k, v in kw.items():
            setattr(self, k, v)


class _Query:
    def __init__(self, rows):
        self._rows = list(rows)

    def filter(self, *ps):
        self._rows = [r for r in self._rows if all(p(r) for p in ps)]
        return self

    def first(self):
        return self._rows[0] if self._rows else None


class _DB:
    def __init__(self, rows):
        self._rows = rows
        self.added = []
        self.committed = False
        self.closed = False

    def query(self, model):
        assert model is _CrewMember
        return _Query(self._rows)

    def add(self, obj):
        self.added.append(obj)
        self._rows.append(obj)

    def commit(self):
        self.committed = True

    def refresh(self, obj):
        pass

    def close(self):
        self.closed = True


def _crew(owner, *, name="A", default=True):
    return _CrewMember(id="c1", owner=owner, name=name, is_default_assistant=default, is_active=True)


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
def db_with(monkeypatch):
    def _install(rows):
        db = _DB(list(rows))
        m = sys.modules["core.database"]
        monkeypatch.setattr(m, "SessionLocal", lambda: db)
        monkeypatch.setattr(m, "CrewMember", _CrewMember)
        monkeypatch.setattr(companion_routes, "get_current_user", lambda request: "api")
        return db
    return _install


# --- GET -------------------------------------------------------------------

def test_get_returns_own_assistant(db_with):
    db_with([_crew("alice", name="Ally")])
    res = _route("/api/companion/assistant", "GET")(_request(owner="alice"))
    assert res["assistant"]["name"] == "Ally"


def test_get_does_not_return_other_owners(db_with):
    db_with([_crew("bob", name="Bobby")])
    res = _route("/api/companion/assistant", "GET")(_request(owner="alice"))
    assert res["assistant"] is None


def test_get_none_when_unset(db_with):
    db_with([])
    res = _route("/api/companion/assistant", "GET")(_request(owner="alice"))
    assert res["assistant"] is None


def test_get_requires_scope(db_with):
    db_with([_crew("alice")])
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/assistant", "GET")(_request(owner="alice", scopes=("chat",)))
    assert exc.value.status_code == 403


# --- PATCH -----------------------------------------------------------------

def test_patch_updates_own_assistant(db_with):
    db = db_with([_crew("alice", name="Ally")])
    res = _route("/api/companion/assistant", "PATCH")(
        _request(owner="alice"), name="Renamed", personality="kind", user_name="Boss",
        greeting=None, model=None, timezone=None,
    )
    assert res["assistant"]["name"] == "Renamed"
    assert res["assistant"]["personality"] == "kind"
    assert db.committed is True and db.added == []  # updated, not created


def test_patch_creates_when_absent(db_with):
    db = db_with([])
    res = _route("/api/companion/assistant", "PATCH")(
        _request(owner="alice"), name="New", personality=None, user_name=None,
        greeting=None, model=None, timezone=None,
    )
    assert res["assistant"]["name"] == "New"
    assert len(db.added) == 1 and db.added[0].owner == "alice"
    assert db.added[0].is_default_assistant is True


def test_patch_refuses_synthetic_owner(db_with):
    db_with([])
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/assistant", "PATCH")(
            _request(owner="api"), name="x", personality=None, user_name=None,
            greeting=None, model=None, timezone=None,
        )
    assert exc.value.status_code == 400


def test_patch_refuses_unresolved_owner(db_with):
    db_with([])
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/assistant", "PATCH")(
            _request(owner=None), name="x", personality=None, user_name=None,
            greeting=None, model=None, timezone=None,
        )
    assert exc.value.status_code == 400


def test_patch_requires_scope(db_with):
    db_with([_crew("alice")])
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/assistant", "PATCH")(
            _request(owner="alice", scopes=("chat",)), name="x", personality=None,
            user_name=None, greeting=None, model=None, timezone=None,
        )
    assert exc.value.status_code == 403
