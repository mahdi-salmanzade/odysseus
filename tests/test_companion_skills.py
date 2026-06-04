"""Owner-scope tests for the companion skills endpoints.

SkillsManager.load(owner) scopes to that owner's skills, so the security
requirement is that the endpoints pass the token's RESOLVED real owner (not the
sandboxed "api") — a cross-owner skill name then isn't in the list and markdown
404s. Both endpoints require the companion scope.

IMPORTANT: we do NOT install global module stubs at import time (that poisons
collection for the real skills/services tests). Instead we monkeypatch the real
SkillsManager class inside a fixture, which pytest reverts after each test.
"""

import os
import sys
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NOTE: no core.database stub here. The skills handler imports
# `from core.constants import DATA_DIR`, which pulls the full `core` package
# init chain (core/__init__ -> session_manager -> needs core.database.Session).
# A minimal stub would starve that, so we let the real core.database load (it
# imports cleanly under conftest, like the other route-level tests). We also
# don't stub services.memory.skills — we monkeypatch its SkillsManager per-test.
import companion.routes as companion_routes
from companion.routes import setup_companion_routes


_SKILLS_BY_OWNER = {}
_MD_BY_OWNER = {}


class _FakeSkillsManager:
    last_load_owner = None

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load(self, owner=None):
        _FakeSkillsManager.last_load_owner = owner
        return list(_SKILLS_BY_OWNER.get(owner, []))

    def read_skill_md(self, name, owner=None):
        return _MD_BY_OWNER.get((owner, name))


def _route(path, method="GET"):
    for r in setup_companion_routes().routes:
        if getattr(r, "path", "") == path and method in getattr(r, "methods", set()):
            return r.endpoint
    raise AssertionError(f"{method} {path} not found")


def _request(*, owner, scopes=("companion",)):
    return SimpleNamespace(state=SimpleNamespace(
        api_token=True, api_token_owner=owner, api_token_scopes=list(scopes), current_user="api",
    ))


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    _SKILLS_BY_OWNER.clear()
    _MD_BY_OWNER.clear()
    _FakeSkillsManager.last_load_owner = None
    # Swap the real SkillsManager for our fake — reverted automatically.
    import services.memory.skills as _skmod
    monkeypatch.setattr(_skmod, "SkillsManager", _FakeSkillsManager)
    monkeypatch.setattr(companion_routes, "get_current_user", lambda request: "api")


# --- list ------------------------------------------------------------------

def test_list_passes_resolved_owner_and_returns_their_skills():
    _SKILLS_BY_OWNER["alice"] = [{"name": "s1", "description": "d", "category": "c"}]
    _SKILLS_BY_OWNER["bob"] = [{"name": "secret", "description": "x", "category": "c"}]
    res = _route("/api/companion/skills")(_request(owner="alice"))
    assert [s["name"] for s in res["items"]] == ["s1"]
    assert _FakeSkillsManager.last_load_owner == "alice"  # never the "api" pseudo-user


def test_list_requires_scope():
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/skills")(_request(owner="alice", scopes=("chat",)))
    assert exc.value.status_code == 403


# --- markdown --------------------------------------------------------------

def test_markdown_returns_owned_skill_source():
    _SKILLS_BY_OWNER["alice"] = [{"name": "s1"}]
    _MD_BY_OWNER[("alice", "s1")] = "# Skill One"
    res = _route("/api/companion/skills/{name}/markdown")(_request(owner="alice"), name="s1")
    assert res == {"name": "s1", "markdown": "# Skill One"}


def test_markdown_cross_owner_skill_is_404():
    _SKILLS_BY_OWNER["alice"] = [{"name": "s1"}]
    _SKILLS_BY_OWNER["bob"] = [{"name": "secret"}]
    _MD_BY_OWNER[("bob", "secret")] = "# Bob secret"
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/skills/{name}/markdown")(_request(owner="alice"), name="secret")
    assert exc.value.status_code == 404


def test_markdown_unknown_skill_is_404():
    _SKILLS_BY_OWNER["alice"] = []
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/skills/{name}/markdown")(_request(owner="alice"), name="nope")
    assert exc.value.status_code == 404


def test_markdown_requires_scope():
    with pytest.raises(HTTPException) as exc:
        _route("/api/companion/skills/{name}/markdown")(_request(owner="alice", scopes=("chat",)), name="s1")
    assert exc.value.status_code == 403
