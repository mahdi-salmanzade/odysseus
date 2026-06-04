"""Owner-scope tests for the companion documents bridge endpoints.

Same direct-route style as test_companion_readonly.py: drive the real route
handlers against a tiny fake-query harness so the owner-scoping rule can't
regress. A bearer token for owner A must never see owner B's documents; legacy
null-owner rows are shared; a plain `chat` token (no companion scope) is
refused; archived/inactive docs are excluded from the list.
"""

import os
import sys
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The handlers import core.database lazily; provide a stub module whose
# SessionLocal/Document we monkeypatch per-test. (Local to this file; we never
# install a partial real-name stub that would pollute sibling tests.)
import types as _types
from unittest.mock import MagicMock

if "core.database" not in sys.modules:
    _db = _types.ModuleType("core.database")
    _db.SessionLocal = MagicMock()
    _db.Document = MagicMock()
    sys.modules["core.database"] = _db

import companion.routes as companion_routes
from companion.routes import setup_companion_routes


# --- fake query harness (supports ==, |, filter chains, all(), first()) -----

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


class _Document:
    id = _Column("id")
    is_active = _Column("is_active")
    archived = _Column("archived")
    owner = _Column("owner")


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
        self.closed = False

    def query(self, model):
        assert model is _Document
        return _Query(self._rows)

    def close(self):
        self.closed = True


def _doc(id, title, owner, *, is_active=True, archived=False, content="body", language="text"):
    return SimpleNamespace(
        id=id, title=title, owner=owner, is_active=is_active, archived=archived,
        current_content=content, language=language, updated_at=None,
    )


def _route(path, method="GET"):
    for r in setup_companion_routes().routes:
        if getattr(r, "path", "") == path and method in getattr(r, "methods", set()):
            return r.endpoint
    raise AssertionError(f"{method} {path} not found")


def _request(*, owner, scopes=("companion",), api_token=True):
    return SimpleNamespace(state=SimpleNamespace(
        api_token=api_token, api_token_owner=owner,
        api_token_scopes=list(scopes), current_user="api",
    ))


@pytest.fixture
def db_with(monkeypatch):
    def _install(rows):
        db = _DB(rows)
        db_mod = sys.modules["core.database"]
        monkeypatch.setattr(db_mod, "SessionLocal", lambda: db)
        monkeypatch.setattr(db_mod, "Document", _Document)
        monkeypatch.setattr(companion_routes, "get_current_user", lambda request: "api")
        return db
    return _install


# --- list ------------------------------------------------------------------

def test_list_scopes_to_owner_and_shared(db_with):
    db = db_with([
        _doc(1, "alice-doc", "alice"),
        _doc(2, "shared-doc", None),
        _doc(3, "bob-doc", "bob"),
    ])
    res = _route("/api/companion/documents")(_request(owner="alice"))
    assert {d["title"] for d in res["items"]} == {"alice-doc", "shared-doc"}
    assert db.closed is True


def test_list_excludes_archived_and_inactive(db_with):
    db_with([
        _doc(1, "live", "alice"),
        _doc(2, "archived", "alice", archived=True),
        _doc(3, "inactive", "alice", is_active=False),
    ])
    res = _route("/api/companion/documents")(_request(owner="alice"))
    assert {d["title"] for d in res["items"]} == {"live"}


def test_list_snippet_is_truncated_not_full_body(db_with):
    db_with([_doc(1, "big", "alice", content="x" * 5000)])
    res = _route("/api/companion/documents")(_request(owner="alice"))
    assert len(res["items"][0]["snippet"]) == 200
    assert "content" not in res["items"][0]  # list never carries the full body


def test_list_requires_companion_scope(db_with):
    db_with([_doc(1, "alice-doc", "alice")])
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/documents")(_request(owner="alice", scopes=("chat",)))
    assert exc.value.status_code == 403


# --- detail ----------------------------------------------------------------

def test_detail_returns_owned_doc_full_body(db_with):
    db_with([_doc(1, "alice-doc", "alice", content="the full body")])
    res = _route("/api/companion/documents/{doc_id}")(_request(owner="alice"), doc_id=1)
    assert res["content"] == "the full body"


def test_detail_shared_null_owner_visible(db_with):
    db_with([_doc(1, "shared", None, content="shared body")])
    res = _route("/api/companion/documents/{doc_id}")(_request(owner="alice"), doc_id=1)
    assert res["title"] == "shared"


def test_detail_cross_owner_is_404_not_403(db_with):
    db_with([_doc(1, "bob-doc", "bob")])
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/documents/{doc_id}")(_request(owner="alice"), doc_id=1)
    assert exc.value.status_code == 404  # never confirm existence to a non-owner


def test_detail_missing_is_404(db_with):
    db_with([])
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/documents/{doc_id}")(_request(owner="alice"), doc_id=99)
    assert exc.value.status_code == 404


def test_detail_requires_companion_scope(db_with):
    db_with([_doc(1, "alice-doc", "alice")])
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/documents/{doc_id}")(_request(owner="alice", scopes=("chat",)), doc_id=1)
    assert exc.value.status_code == 403
