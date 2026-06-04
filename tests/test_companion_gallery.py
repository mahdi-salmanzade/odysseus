"""Owner-scope tests for the companion gallery endpoints.

The list must be owner-scoped (own + legacy null-owner shared), and the image
byte-serving endpoint must refuse a missing OR cross-owner image with a 404
BEFORE any file access — so a caller can't read another owner's image by id.
Both require the companion scope. File streaming itself isn't unit-tested.
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
    _db.GalleryImage = MagicMock()
    sys.modules["core.database"] = _db

import companion.routes as companion_routes
from companion.routes import setup_companion_routes


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


class _GalleryImage:
    id = _Column("id")
    is_active = _Column("is_active")
    owner = _Column("owner")


class _Query:
    def __init__(self, rows):
        self._rows = list(rows)

    def filter(self, *ps):
        self._rows = [r for r in self._rows if all(p(r) for p in ps)]
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
        assert model is _GalleryImage
        return _Query(self._rows)

    def close(self):
        self.closed = True


def _img(id, owner, *, is_active=True, filename=None):
    return SimpleNamespace(
        id=id, owner=owner, is_active=is_active,
        filename=filename or f"{id}.png", prompt="p", model="m", favorite=False,
        width=10, height=10, taken_at=None, created_at=None,
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
        m = sys.modules["core.database"]
        monkeypatch.setattr(m, "SessionLocal", lambda: db)
        monkeypatch.setattr(m, "GalleryImage", _GalleryImage)
        monkeypatch.setattr(companion_routes, "get_current_user", lambda request: "api")
        return db
    return _install


# --- list ------------------------------------------------------------------

def test_gallery_list_scoped_to_owner_and_shared(db_with):
    db_with([_img(1, "alice"), _img(2, None), _img(3, "bob")])
    res = _route("/api/companion/gallery", "GET")(_request(owner="alice"))
    assert {i["id"] for i in res["items"]} == {1, 2}


def test_gallery_list_excludes_inactive(db_with):
    db_with([_img(1, "alice"), _img(2, "alice", is_active=False)])
    res = _route("/api/companion/gallery", "GET")(_request(owner="alice"))
    assert {i["id"] for i in res["items"]} == {1}


def test_gallery_list_url_points_at_companion_endpoint(db_with):
    db_with([_img(1, "alice")])
    res = _route("/api/companion/gallery", "GET")(_request(owner="alice"))
    assert res["items"][0]["image_url"] == "/api/companion/gallery/image/1"


def test_gallery_list_requires_scope(db_with):
    db_with([_img(1, "alice")])
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/gallery", "GET")(_request(owner="alice", scopes=("chat",)))
    assert exc.value.status_code == 403


# --- image bytes (ownership gate before file access) -----------------------

def test_image_cross_owner_is_404_before_file(db_with, monkeypatch):
    db_with([_img(1, "bob")])
    # If the gate fails open we'd reach the filesystem; make that explode so a
    # leak would be loud rather than silently serving the file.
    monkeypatch.setattr(os.path, "isfile", lambda p: (_ for _ in ()).throw(AssertionError("reached FS")))
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/gallery/image/{image_id}", "GET")(_request(owner="alice"), image_id=1)
    assert exc.value.status_code == 404


def test_image_missing_is_404(db_with):
    db_with([])
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/gallery/image/{image_id}", "GET")(_request(owner="alice"), image_id=9)
    assert exc.value.status_code == 404


def test_image_requires_scope(db_with):
    db_with([_img(1, "alice")])
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/gallery/image/{image_id}", "GET")(
            _request(owner="alice", scopes=("chat",)), image_id=1)
    assert exc.value.status_code == 403


def test_image_owned_but_file_absent_is_404(db_with, monkeypatch):
    # Owner passes the gate; with no file on disk it's a clean 404 (not a crash).
    db_with([_img(1, "alice", filename="nope.png")])
    monkeypatch.setattr(os.path, "isfile", lambda p: False)
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/gallery/image/{image_id}", "GET")(_request(owner="alice"), image_id=1)
    assert exc.value.status_code == 404


def test_image_filename_cannot_escape_image_dir(db_with, monkeypatch):
    # A malicious stored filename must be collapsed to a basename under the
    # image dir — never resolve to an arbitrary path via '../' traversal.
    db_with([_img(1, "alice", filename="../../../../etc/passwd")])
    seen = {}
    monkeypatch.setattr(os.path, "isfile", lambda p: seen.setdefault("path", p) and False)
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/gallery/image/{image_id}", "GET")(_request(owner="alice"), image_id=1)
    assert exc.value.status_code == 404
    checked = seen["path"]
    assert ".." not in checked
    assert checked == os.path.join("data", "generated_images", "passwd")
