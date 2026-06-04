"""Owner-scope tests for the companion calendar endpoints.

Calendars are owner-scoped directly; events are scoped THROUGH their calendar's
owner. A bearer token for owner A must never read A-foreign events, nor create
into / delete from a calendar it doesn't own. All endpoints require the
companion scope. Fake-query harness extended with `.in_()` for the event scope.
"""

import os
import sys
import types
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if "core.database" not in sys.modules:
    _db = types.ModuleType("core.database")
    _db.SessionLocal = MagicMock()
    _db.CalendarCal = MagicMock()
    _db.CalendarEvent = MagicMock()
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

    def in_(self, values):
        vs = set(values)
        return _Predicate(lambda r: getattr(r, self.name) in vs)


class _CalendarCal:
    id = _Column("id")
    owner = _Column("owner")


class _CalendarEvent:
    uid = _Column("uid")
    calendar_id = _Column("calendar_id")

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
    def __init__(self, cals, events):
        self._cals = cals
        self._events = events
        self.added = []
        self.deleted = []
        self.committed = False
        self.closed = False

    def query(self, model):
        if model is _CalendarCal:
            return _Query(self._cals)
        if model is _CalendarEvent:
            return _Query(self._events)
        raise AssertionError("unexpected model")

    def add(self, obj):
        self.added.append(obj)

    def delete(self, obj):
        self.deleted.append(obj)

    def commit(self):
        self.committed = True

    def close(self):
        self.closed = True


def _cal(id, owner, name="Cal"):
    return SimpleNamespace(id=id, owner=owner, name=name, color="#fff", source="local")


def _ev(uid, calendar_id, *, summary="ev", status="confirmed",
        dtstart=datetime(2026, 6, 4, 10), dtend=datetime(2026, 6, 4, 11)):
    return SimpleNamespace(
        uid=uid, calendar_id=calendar_id, summary=summary, description="", location="",
        dtstart=dtstart, dtend=dtend, all_day=False, rrule="", status=status,
        importance="normal", event_type=None, color=None,
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
    def _install(cals, events=()):
        db = _DB(list(cals), list(events))
        m = sys.modules["core.database"]
        monkeypatch.setattr(m, "SessionLocal", lambda: db)
        monkeypatch.setattr(m, "CalendarCal", _CalendarCal)
        monkeypatch.setattr(m, "CalendarEvent", _CalendarEvent)
        monkeypatch.setattr(companion_routes, "get_current_user", lambda request: "api")
        return db
    return _install


# --- calendars -------------------------------------------------------------

def test_calendars_scoped_to_owner_and_shared(db_with):
    db_with([_cal("a", "alice"), _cal("s", None), _cal("b", "bob")])
    res = _route("/api/companion/calendars", "GET")(_request(owner="alice"))
    assert {c["id"] for c in res["items"]} == {"a", "s"}


def test_calendars_requires_scope(db_with):
    db_with([_cal("a", "alice")])
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/calendars", "GET")(_request(owner="alice", scopes=("chat",)))
    assert exc.value.status_code == 403


# --- events ----------------------------------------------------------------

def test_events_scoped_through_calendar_owner(db_with):
    db_with(
        [_cal("a", "alice"), _cal("b", "bob")],
        [_ev("e1", "a"), _ev("e2", "b")],  # e2 belongs to bob's calendar
    )
    res = _route("/api/companion/events", "GET")(_request(owner="alice"))
    assert {e["uid"] for e in res["items"]} == {"e1"}


def test_events_excludes_cancelled(db_with):
    db_with([_cal("a", "alice")], [_ev("e1", "a"), _ev("e2", "a", status="cancelled")])
    res = _route("/api/companion/events", "GET")(_request(owner="alice"))
    assert {e["uid"] for e in res["items"]} == {"e1"}


def test_events_window_overlap(db_with):
    db_with([_cal("a", "alice")], [
        _ev("inside", "a", dtstart=datetime(2026, 6, 4, 10), dtend=datetime(2026, 6, 4, 11)),
        _ev("before", "a", dtstart=datetime(2026, 6, 1, 10), dtend=datetime(2026, 6, 1, 11)),
        _ev("after", "a", dtstart=datetime(2026, 6, 9, 10), dtend=datetime(2026, 6, 9, 11)),
    ])
    res = _route("/api/companion/events", "GET")(
        _request(owner="alice"), start="2026-06-03T00:00:00", end="2026-06-05T00:00:00",
    )
    assert {e["uid"] for e in res["items"]} == {"inside"}


def test_events_requires_scope(db_with):
    db_with([_cal("a", "alice")], [_ev("e1", "a")])
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/events", "GET")(_request(owner="alice", scopes=("chat",)))
    assert exc.value.status_code == 403


# --- create event ----------------------------------------------------------

def test_create_event_into_owned_calendar(db_with):
    db = db_with([_cal("a", "alice")])
    res = _route("/api/companion/events", "POST")(
        _request(owner="alice"), calendar_id="a", summary="hi",
        dtstart="2026-06-04T10:00:00", dtend="2026-06-04T11:00:00",
        description="", location="", all_day="false",
    )
    assert res["status"] == "ok" and res["uid"]
    assert len(db.added) == 1 and db.added[0].calendar_id == "a"


def test_create_event_into_cross_owner_calendar_is_404(db_with):
    db = db_with([_cal("b", "bob")])
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/events", "POST")(
            _request(owner="alice"), calendar_id="b", summary="hi",
            dtstart="2026-06-04T10:00:00", dtend="2026-06-04T11:00:00",
            description="", location="", all_day="false",
        )
    assert exc.value.status_code == 404
    assert db.added == []


def test_create_event_bad_datetime_is_400(db_with):
    db_with([_cal("a", "alice")])
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/events", "POST")(
            _request(owner="alice"), calendar_id="a", summary="hi",
            dtstart="not-a-date", dtend="also-bad",
            description="", location="", all_day="false",
        )
    assert exc.value.status_code == 400


def test_create_event_requires_scope(db_with):
    db_with([_cal("a", "alice")])
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/events", "POST")(
            _request(owner="alice", scopes=("chat",)), calendar_id="a", summary="hi",
            dtstart="2026-06-04T10:00:00", dtend="2026-06-04T11:00:00",
            description="", location="", all_day="false",
        )
    assert exc.value.status_code == 403


# --- delete event ----------------------------------------------------------

def test_delete_owned_event(db_with):
    db = db_with([_cal("a", "alice")], [_ev("e1", "a")])
    res = _route("/api/companion/events/{uid}", "DELETE")(_request(owner="alice"), uid="e1")
    assert res["status"] == "deleted" and len(db.deleted) == 1


def test_delete_event_in_cross_owner_calendar_is_404(db_with):
    db = db_with([_cal("b", "bob")], [_ev("e2", "b")])
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/events/{uid}", "DELETE")(_request(owner="alice"), uid="e2")
    assert exc.value.status_code == 404
    assert db.deleted == []


def test_delete_missing_event_is_404(db_with):
    db_with([_cal("a", "alice")], [])
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/events/{uid}", "DELETE")(_request(owner="alice"), uid="nope")
    assert exc.value.status_code == 404
