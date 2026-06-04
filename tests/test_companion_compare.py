"""Owner-scope tests for the companion model-compare endpoints.

history/record/delete must be strictly owner-scoped: a bearer token for owner A
never sees or deletes owner B's comparisons; record stamps the resolved owner;
delete is strict (cross-owner or legacy null-owner shared → 404, never 403);
all three require the companion scope. Same fake-query harness style as
test_companion_documents.py.
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
    _db.Comparison = MagicMock()
    sys.modules["core.database"] = _db

import companion.routes as companion_routes
from companion.routes import setup_companion_routes


class _Predicate:
    def __init__(self, check):
        self._check = check

    def __call__(self, row):
        return self._check(row)

    def __or__(self, other):
        return _Predicate(lambda row: self(row) or other(row))


class _Column:
    def __init__(self, name):
        self.name = name

    def __eq__(self, value):
        return _Predicate(lambda row: getattr(row, self.name) == value)


class _Comparison:
    # Class-level columns used to build query filters; instances (built in the
    # record route) shadow these with real values via __init__.
    id = _Column("id")
    owner = _Column("owner")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _Query:
    def __init__(self, rows):
        self._rows = list(rows)

    def filter(self, *predicates):
        self._rows = [r for r in self._rows if all(p(r) for p in predicates)]
        return self

    def all(self):
        return list(self._rows)

    def first(self):
        return self._rows[0] if self._rows else None


class _DB:
    def __init__(self, rows):
        self._rows = rows
        self.added = []
        self.deleted = []
        self.committed = False
        self.closed = False

    def query(self, model):
        assert model is _Comparison
        return _Query(self._rows)

    def add(self, obj):
        self.added.append(obj)

    def delete(self, obj):
        self.deleted.append(obj)

    def commit(self):
        self.committed = True

    def close(self):
        self.closed = True


def _comp(id, owner, *, prompt="p", model_a="a", model_b="b", winner=None, is_blind=False):
    return SimpleNamespace(
        id=id, owner=owner, prompt=prompt, model_a=model_a, model_b=model_b,
        winner=winner, is_blind=is_blind, voted_at=None, created_at=None,
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
def db_with(monkeypatch):
    def _install(rows):
        db = _DB(rows)
        db_mod = sys.modules["core.database"]
        monkeypatch.setattr(db_mod, "SessionLocal", lambda: db)
        monkeypatch.setattr(db_mod, "Comparison", _Comparison)
        monkeypatch.setattr(companion_routes, "get_current_user", lambda request: "api")
        return db
    return _install


# --- history ---------------------------------------------------------------

def test_history_scopes_to_owner_and_shared(db_with):
    db_with([_comp(1, "alice"), _comp(2, None), _comp(3, "bob")])
    res = _route("/api/companion/compare/history", "GET")(_request(owner="alice"))
    assert {c["id"] for c in res["items"]} == {1, 2}


def test_history_requires_companion_scope(db_with):
    db_with([_comp(1, "alice")])
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/compare/history", "GET")(_request(owner="alice", scopes=("chat",)))
    assert exc.value.status_code == 403


# --- record ----------------------------------------------------------------

def test_record_stamps_owner_and_persists(db_with):
    db = db_with([])
    res = _route("/api/companion/compare/record", "POST")(
        _request(owner="alice"), prompt="which?", model_a="m-a", model_b="m-b",
        winner="a", is_blind="false",
    )
    assert res["status"] == "ok" and res["id"]
    assert len(db.added) == 1 and db.added[0].owner == "alice"
    assert db.added[0].winner == "a" and db.committed is True


def test_record_rejects_bad_winner(db_with):
    db_with([])
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/compare/record", "POST")(
            _request(owner="alice"), prompt="p", model_a="a", model_b="b",
            winner="left", is_blind="false",
        )
    assert exc.value.status_code == 400


def test_record_requires_resolved_owner(db_with):
    db_with([])
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/compare/record", "POST")(
            _request(owner=None), prompt="p", model_a="a", model_b="b",
            winner="a", is_blind="false",
        )
    assert exc.value.status_code == 403


def test_record_requires_companion_scope(db_with):
    db_with([])
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/compare/record", "POST")(
            _request(owner="alice", scopes=("chat",)), prompt="p", model_a="a",
            model_b="b", winner="a", is_blind="false",
        )
    assert exc.value.status_code == 403


# --- delete ----------------------------------------------------------------

def test_delete_own_comparison(db_with):
    db = db_with([_comp(1, "alice")])
    res = _route("/api/companion/compare/{comp_id}", "DELETE")(_request(owner="alice"), comp_id=1)
    assert res["status"] == "deleted" and len(db.deleted) == 1


def test_delete_cross_owner_is_404(db_with):
    db = db_with([_comp(1, "bob")])
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/compare/{comp_id}", "DELETE")(_request(owner="alice"), comp_id=1)
    assert exc.value.status_code == 404
    assert db.deleted == []


def test_delete_shared_null_owner_is_404(db_with):
    # Strict: a null-owner shared row is readable but NOT deletable by a non-owner.
    db = db_with([_comp(1, None)])
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/compare/{comp_id}", "DELETE")(_request(owner="alice"), comp_id=1)
    assert exc.value.status_code == 404
    assert db.deleted == []


def test_delete_missing_is_404(db_with):
    db_with([])
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/compare/{comp_id}", "DELETE")(_request(owner="alice"), comp_id=9)
    assert exc.value.status_code == 404


def test_delete_requires_companion_scope(db_with):
    db_with([_comp(1, "alice")])
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/compare/{comp_id}", "DELETE")(_request(owner="alice", scopes=("chat",)), comp_id=1)
    assert exc.value.status_code == 403
