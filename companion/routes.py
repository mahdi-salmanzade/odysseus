"""Companion bridge — /api/companion/*.

A thin, additive layer so a LAN client (e.g. a phone) can discover what a server
offers and pair to it, without duplicating any LLM logic.

Auth is enforced globally by AuthMiddleware (app.py), so reaching a handler here
means the caller is authenticated by either a cookie session or a Bearer `ody_`
API token. The read endpoints (ping/info/models) accept either; the pairing
endpoints are admin-cookie only.

Pairing CSRF posture: minting happens ONLY on POST. The session cookie is
SameSite=Lax (routes/auth_routes.py), which a browser does not send on a
cross-site POST, so an admin's cookie can't be used by a malicious page to mint
a token -- the same protection the existing POST /api/tokens relies on. Minting
on a GET would be unsafe (Lax cookies ride top-level GET navigations), so GET
/pair only renders a form.
"""

import html
import json as _json
import time
import uuid

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse

from src.auth_helpers import get_current_user

from companion import pairing as _pairing


def token_owner(request: Request) -> str | None:
    """The real owner to attribute a request to, for read-scoping.

    Cookie sessions resolve to the logged-in username via get_current_user.
    Bearer-token callers come through as the sandboxed pseudo-user "api"; their
    real owner is stamped on request.state.api_token_owner by the auth
    middleware. Returns None when no owner can be resolved.
    """
    if getattr(request.state, "api_token", False):
        return getattr(request.state, "api_token_owner", None)
    return get_current_user(request)


def owner_can_see(row_owner, owner) -> bool:
    """Owner-scope rule for read endpoints.

    A caller sees a row when it is their own, or when it is a legacy null-owner
    ("shared") row. A caller must NEVER see another owner's row. Mirrors the
    `owner_filter` rule used elsewhere, expressed as a pure predicate so it can
    be tested directly and used as a defensive in-Python check alongside the
    SQL filter.
    """
    return row_owner is None or row_owner == owner


def has_companion_scope(request: Request) -> bool:
    """Whether the caller may read the companion DATA views (notes/tasks/memory).

    A cookie session (the logged-in user) always may. A bearer token must carry
    the explicit ``companion`` scope: a plain ``chat`` token cannot read your
    private notes or memory. This keeps these reads strictly NARROWER than chat,
    per review. Pure + testable.
    """
    if not getattr(request.state, "api_token", False):
        return True
    scopes = getattr(request.state, "api_token_scopes", None) or []
    return "companion" in scopes


def companion_admin_available(request: Request) -> bool:
    """Whether ADMIN-only companion features are reachable for this caller.

    A pure-ish predicate (no raise) the status endpoint uses to tell a paired
    phone whether to even show admin tabs (terminal/vault/mcp/cookbook/contacts).
    True only when ALL hold — the same triple lock require_companion_admin
    enforces:
      1. an admin flipped on the ``companion_admin_enabled`` server setting,
      2. the caller carries the ``companion`` scope (a plain ``chat`` token never
         reaches admin surface), and
      3. the caller's real owner is a server admin.
    Fail-closed: any missing piece (no auth_manager, unknown owner) → False.
    """
    from src.settings import get_setting

    if not get_setting("companion_admin_enabled", False):
        return False
    if not has_companion_scope(request):
        return False
    owner = token_owner(request)
    if not owner:
        return False
    auth_manager = getattr(request.app.state, "auth_manager", None)
    if auth_manager is None:
        return False
    try:
        return bool(auth_manager.is_admin(owner))
    except Exception:
        return False


def require_companion_admin(request: Request) -> str:
    """Gate for ADMIN-only companion endpoints. Returns the owner, or raises 403.

    This is the ONLY sanctioned way to expose an admin-privileged server
    capability (shell exec, vault export, MCP/cookbook admin, contacts) to a
    paired phone. The stock routes hard-block the bearer pseudo-user ``api`` by
    design (``current_user == "api"`` → 403, "RCE-after-signup"); we do NOT
    bypass that loosely. Instead we re-establish privilege from the token's real
    OWNER, behind an explicit, off-by-default admin opt-in:

      1. ``companion_admin_enabled`` must be on (an admin set it deliberately),
      2. the token must carry the ``companion`` scope (never a plain ``chat`` token),
      3. the resolved owner must be a server admin (``auth_manager.is_admin``).

    Fail-closed and non-disclosive: every failure raises the same generic 403 so
    a caller can't probe which lock stopped them. Never call a stock admin route's
    own ``_require_admin`` from here — that checks ``current_user`` (always ``api``
    for a bearer caller) and would always 403.
    """
    if not companion_admin_available(request):
        raise HTTPException(403, "Companion admin access is not enabled")
    return token_owner(request)


def require_companion_scope(request: Request) -> None:
    """Raise 403 unless the caller may touch the companion data views. The
    write handlers gate on this exactly like the reads gate on
    has_companion_scope, so a plain ``chat`` token can neither read nor mutate
    notes/memory."""
    if not has_companion_scope(request):
        raise HTTPException(403, "This token is not allowed to access companion data.")


def writer_owner(request: Request) -> str | None:
    """Owner to stamp on a new/mutated row.

    Cookie sessions and single-user mode resolve to a username or None (None =
    legacy shared row, the long-standing behaviour). A BEARER token, however,
    must have a resolvable owner: a null-owner token would otherwise fall
    through to mutating shared/null-owner rows it doesn't own, so we refuse it
    (401) rather than widen its scope. Mirrors the reasoning in the desktop
    note/memory routes.
    """
    owner = token_owner(request)
    if owner is None and getattr(request.state, "api_token", False):
        raise HTTPException(401, "Token owner could not be resolved.")
    return owner


# Categories the mobile composer offers; anything else coerces to "fact". Mirrors
# the server allowlist in src/request_models.py so companion-created memories are
# indistinguishable from desktop ones.
_MEMORY_CATEGORIES = {"fact", "identity", "preference", "contact", "project", "goal", "task"}


def _serialize_note(n) -> dict:
    """The exact shape GET /notes returns, so a created/updated note round-trips
    into the mobile list without a refetch."""
    try:
        items = _json.loads(n.items) if n.items else None
    except (ValueError, TypeError):
        items = None
    return {
        "id": n.id, "title": n.title, "content": n.content,
        "items": items, "pinned": bool(n.pinned),
    }


def mint_pairing_token(owner: str, invalidate=None) -> tuple[str, str]:
    """Mint a pairing token AND invalidate the auth middleware's in-memory token
    cache, so the new token is accepted on the very next request without a server
    restart. Returns (token_id, raw_token); the raw token is shown once.

    `invalidate` is the app's request.app.state.invalidate_token_cache callable
    (passed in so this stays a pure, testable unit).
    """
    token_id, raw_token = _pairing.mint_token(owner)
    if callable(invalidate):
        invalidate()
    return token_id, raw_token


def setup_companion_routes() -> APIRouter:
    router = APIRouter(prefix="/api/companion", tags=["companion"])

    @router.get("/ping")
    def ping(request: Request):
        """Cheap, auth-validated health check. A 200 with ok=true confirms the
        host/port and credential are valid; middleware returns 401 otherwise."""
        from core.constants import APP_VERSION
        return {
            "ok": True,
            "name": "odysseus",
            "version": APP_VERSION,
            "auth": "token" if getattr(request.state, "api_token", False) else "session",
        }

    @router.get("/info")
    def info(request: Request):
        """Server identity + coarse capability flags. `owner` is the caller's own
        identity (the token's owner for bearer callers)."""
        from core.constants import APP_VERSION
        return {
            "name": "odysseus",
            "version": APP_VERSION,
            "owner": token_owner(request),
            "capabilities": {"chat": True, "streaming": True},
        }

    @router.get("/models")
    def models(request: Request):
        """LLM model endpoints the CALLER can use.

        The stock /api/models route scopes to get_current_user, which for a
        bearer token is the sandboxed pseudo-user "api" (owns nothing). Here we
        scope to the token's real owner instead, plus legacy null-owner shared
        rows -- the same rule as owner_filter. Read-only; never returns api_key
        material.
        """
        import json as _json

        from core.database import SessionLocal, ModelEndpoint
        from src.endpoint_resolver import build_chat_url

        owner = token_owner(request)
        out = []
        db = SessionLocal()
        try:
            q = db.query(ModelEndpoint).filter(
                ModelEndpoint.is_enabled == True,  # noqa: E712
                (ModelEndpoint.model_type == "llm") | (ModelEndpoint.model_type == None),  # noqa: E711
            )
            if owner:
                q = q.filter((ModelEndpoint.owner == owner) | (ModelEndpoint.owner == None))  # noqa: E711
            for ep in q.all():
                if not owner_can_see(ep.owner, owner):
                    continue
                try:
                    model_ids = _json.loads(ep.cached_models) if ep.cached_models else []
                except (ValueError, TypeError):
                    model_ids = []
                try:
                    hidden = set(_json.loads(ep.hidden_models)) if ep.hidden_models else set()
                except (ValueError, TypeError):
                    hidden = set()
                model_ids = [m for m in model_ids if m not in hidden]
                try:
                    chat_url = build_chat_url(ep.base_url)
                except Exception:
                    chat_url = ep.base_url
                out.append({
                    "endpoint_id": ep.id,
                    "name": ep.name,
                    "endpoint_url": chat_url,
                    "models": model_ids,
                    "supports_tools": ep.supports_tools,
                })
        finally:
            db.close()
        return {"endpoints": out}

    @router.get("/admin/status")
    def admin_status(request: Request):
        """Coarse booleans telling a paired phone whether ADMIN-only companion
        features (terminal/vault/mcp/cookbook/contacts) are reachable, so it can
        show or hide those tabs. Returns ONLY booleans — never secrets or admin
        internals. `enabled` = the server opt-in; `is_admin` = the token owner is
        a server admin; `available` = both, i.e. the gate would let them through."""
        from src.settings import get_setting

        owner = token_owner(request)
        auth_manager = getattr(request.app.state, "auth_manager", None)
        is_admin = False
        if owner and auth_manager is not None:
            try:
                is_admin = bool(auth_manager.is_admin(owner))
            except Exception:
                is_admin = False
        return {
            "enabled": bool(get_setting("companion_admin_enabled", False)),
            "is_admin": is_admin,
            "available": companion_admin_available(request),
        }

    @router.get("/pair")
    def pair_page(request: Request):
        """Admin-only pairing page. Renders a form that POSTs to mint a code.

        A GET never mints a credential: SameSite=Lax session cookies ride
        top-level GET navigations, so minting on GET would be triggerable by a
        link or <img> (CSRF). The actual mint is the POST handler below.
        """
        require_admin(request)
        page = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pair a device</title>
<style>
  body{font-family:-apple-system,system-ui,sans-serif;max-width:520px;margin:48px auto;padding:0 20px;color:#e8e8e8;background:#16161a}
  .card{background:#1f1f25;border:1px solid #2c2c35;border-radius:14px;padding:28px;text-align:center}
  button{background:#7c9cff;color:#0e0e12;border:none;border-radius:10px;padding:12px 20px;font-size:15px;font-weight:600;cursor:pointer}
</style></head>
<body><div class="card">
  <h2>Pair a device</h2>
  <p>Generate a one-time pairing code (a chat-scoped API token) for a LAN client.</p>
  <form method="POST" action="/api/companion/pair">
    <button type="submit">Generate pairing code</button>
  </form>
  <p style="color:#8a8a96;font-size:12px;margin-top:18px">Admin only. Each code mints a new token, shown once. Manage or revoke under Settings &rarr; API tokens.</p>
</div></body></html>"""
        return HTMLResponse(page)

    @router.post("/pair")
    def pair_create(request: Request):
        """Mint a pairing code. Admin-cookie only; CSRF-safe because the
        SameSite=Lax session cookie is not sent on a cross-site POST (same
        protection as POST /api/tokens). Minting invalidates the token cache so
        the code works immediately, no restart. `?format=json` returns the
        payload for an in-app pairing screen."""
        require_admin(request)
        owner = get_current_user(request)
        invalidate = getattr(request.app.state, "invalidate_token_cache", None)
        token_id, raw_token = mint_pairing_token(owner, invalidate)

        hosts = _pairing.lan_ip_candidates()
        host = hosts[0] if hosts else "127.0.0.1"
        port = request.url.port or _pairing.default_port()
        payload = _pairing.pairing_payload(host, port, raw_token)
        qr = _pairing.pairing_qr_png_data_uri(payload)
        qr_ok = bool(qr and qr.startswith("data:image/png;base64,"))

        if (request.query_params.get("format") or "").lower() == "json":
            return {
                "host": host,
                "port": port,
                "token": raw_token,
                "token_id": token_id,
                "hosts": hosts,
                "payload": payload,
                "qr": qr if qr_ok else None,
            }

        import json as _json
        payload_json = _json.dumps(payload, separators=(",", ":"))
        # Only ever emit a known PNG data-URI into the src; every other value is
        # html.escaped.
        qr_block = (
            f'<img src="{html.escape(qr)}" alt="Pairing QR" width="260" height="260">'
            if qr_ok else "<p><em>QR rendering unavailable -- enter the details manually.</em></p>"
        )
        page = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pairing code</title>
<style>
  body{{font-family:-apple-system,system-ui,sans-serif;max-width:520px;margin:40px auto;padding:0 20px;color:#e8e8e8;background:#16161a}}
  .card{{background:#1f1f25;border:1px solid #2c2c35;border-radius:14px;padding:24px;text-align:center}}
  code{{background:#0e0e12;padding:2px 6px;border-radius:6px;word-break:break-all}}
  .row{{text-align:left;margin:10px 0;font-size:14px;color:#bdbdc7}}
  .warn{{color:#e0a85e;font-size:13px;margin-top:18px}}
</style></head>
<body><div class="card">
  <h2>Pairing code</h2>
  {qr_block}
  <div class="row"><strong>Host:</strong> <code>{html.escape(host)}</code></div>
  <div class="row"><strong>Port:</strong> <code>{html.escape(str(port))}</code></div>
  <div class="row"><strong>Token:</strong> <code>{html.escape(raw_token)}</code></div>
  <div class="row"><strong>Payload:</strong> <code>{html.escape(payload_json)}</code></div>
  <p class="warn">Shown once. This grants chat access to your Odysseus; revoke it
  in Settings &rarr; API tokens (id <code>{html.escape(token_id)}</code>). The
  device must be on the same network, and the server must bind to your LAN.</p>
</div></body></html>"""
        return HTMLResponse(page)

    @router.get("/notes")
    def notes(request: Request):
        """List the caller's own notes. Requires the companion scope."""
        if not has_companion_scope(request):
            raise HTTPException(403, "This token is not allowed to read notes.")
        import json as _json
        from core.database import SessionLocal, Note

        owner = token_owner(request)
        out = []
        db = SessionLocal()
        try:
            q = db.query(Note).filter(Note.archived == False)  # noqa: E712
            if owner:
                q = q.filter((Note.owner == owner) | (Note.owner == None))  # noqa: E711
            for n in q.all():
                if not owner_can_see(n.owner, owner):
                    continue
                try:
                    items = _json.loads(n.items) if n.items else None
                except (ValueError, TypeError):
                    items = None
                out.append({
                    "id": n.id, "title": n.title, "content": n.content,
                    "items": items, "pinned": bool(n.pinned),
                })
        finally:
            db.close()
        return {"items": out}

    @router.get("/tasks")
    def tasks(request: Request):
        """The caller's own scheduled tasks (read-only). Requires the companion scope."""
        if not has_companion_scope(request):
            raise HTTPException(403, "This token is not allowed to read tasks.")
        from core.database import SessionLocal, ScheduledTask

        owner = token_owner(request)
        out = []
        db = SessionLocal()
        try:
            q = db.query(ScheduledTask)
            if owner:
                q = q.filter((ScheduledTask.owner == owner) | (ScheduledTask.owner == None))  # noqa: E711
            for t in q.all():
                if not owner_can_see(t.owner, owner):
                    continue
                out.append({
                    "id": t.id, "name": t.name, "schedule": t.schedule,
                    "enabled": t.status == "active",
                    "last_run": t.last_run.isoformat() + "Z" if t.last_run else None,
                })
        finally:
            db.close()
        return {"items": out}

    @router.get("/memory")
    def memory(request: Request):
        """List the caller's own long-term memories. Requires the companion scope."""
        if not has_companion_scope(request):
            raise HTTPException(403, "This token is not allowed to read memory.")
        from core.database import SessionLocal, Memory

        owner = token_owner(request)
        out = []
        db = SessionLocal()
        try:
            q = db.query(Memory)
            if owner:
                q = q.filter((Memory.owner == owner) | (Memory.owner == None))  # noqa: E711
            for m in q.all():
                if not owner_can_see(m.owner, owner):
                    continue
                out.append({"id": m.id, "text": m.text, "category": m.category})
        finally:
            db.close()
        return {"items": out}

    @router.get("/documents")
    def documents(request: Request):
        """List the caller's own documents (RAG library), newest first.

        Owner-scoped exactly like the stock /api/documents/library, but resolved
        to the token's real owner (plus legacy null-owner shared rows) instead of
        the sandboxed "api" user. Read-only summary — returns a short content
        snippet, never the full body (that's the per-doc GET below). Requires the
        companion scope."""
        if not has_companion_scope(request):
            raise HTTPException(403, "This token is not allowed to read documents.")
        from core.database import SessionLocal, Document

        owner = token_owner(request)
        out = []
        db = SessionLocal()
        try:
            q = db.query(Document).filter(Document.is_active == True)  # noqa: E712
            # Exclude archived (NULL = legacy rows = not archived).
            q = q.filter((Document.archived == False) | (Document.archived == None))  # noqa: E711,E712
            if owner:
                q = q.filter((Document.owner == owner) | (Document.owner == None))  # noqa: E711
            for d in q.all():
                if not owner_can_see(d.owner, owner):
                    continue
                content = d.current_content or ""
                out.append({
                    "id": d.id,
                    "title": d.title,
                    "language": d.language,
                    "snippet": content[:200],
                    "updated_at": getattr(d, "updated_at", None) and str(d.updated_at),
                })
        finally:
            db.close()
        # Newest first when a timestamp is available; stable otherwise.
        out.sort(key=lambda r: r.get("updated_at") or "", reverse=True)
        return {"items": out}

    @router.get("/documents/{doc_id}")
    def document_detail(request: Request, doc_id: str):
        """Full body of one of the caller's documents. 404 (never 403) for a
        missing OR cross-owner doc, so existence is never confirmed to a
        non-owner. Requires the companion scope."""
        if not has_companion_scope(request):
            raise HTTPException(403, "This token is not allowed to read documents.")
        from core.database import SessionLocal, Document

        owner = token_owner(request)
        db = SessionLocal()
        try:
            doc = db.query(Document).filter(Document.id == doc_id).first()
            if not doc or not owner_can_see(doc.owner, owner):
                raise HTTPException(404, "Document not found")
            return {
                "id": doc.id,
                "title": doc.title,
                "language": doc.language,
                "content": doc.current_content or "",
                "archived": bool(doc.archived),
                "updated_at": getattr(doc, "updated_at", None) and str(doc.updated_at),
            }
        finally:
            db.close()

    # ---- Model compare (history + verdict record) -------------------------
    # The phone runs the two model streams itself via the EXISTING owner-scoped
    # /api/session + /api/chat_stream — so it never touches the stock
    # /api/compare/start, whose endpoint-key lookup is not owner-scoped. These
    # endpoints only persist and list the caller's own comparison verdicts.

    @router.get("/compare/history")
    def compare_history(request: Request):
        """The caller's own past model comparisons, newest first. Companion scope."""
        if not has_companion_scope(request):
            raise HTTPException(403, "This token is not allowed to read comparisons.")
        from core.database import SessionLocal, Comparison

        owner = token_owner(request)
        out = []
        db = SessionLocal()
        try:
            q = db.query(Comparison)
            if owner:
                q = q.filter((Comparison.owner == owner) | (Comparison.owner == None))  # noqa: E711
            for c in q.all():
                if not owner_can_see(c.owner, owner):
                    continue
                out.append({
                    "id": c.id,
                    "prompt": (c.prompt or "")[:100],
                    "model_a": c.model_a,
                    "model_b": c.model_b,
                    "winner": c.winner,
                    "is_blind": bool(c.is_blind),
                    "voted_at": c.voted_at.isoformat() if c.voted_at else None,
                    "created_at": c.created_at.isoformat() if getattr(c, "created_at", None) else None,
                })
        finally:
            db.close()
        out.sort(key=lambda r: r.get("created_at") or "", reverse=True)
        return {"items": out}

    @router.post("/compare/record")
    def compare_record(
        request: Request,
        prompt: str = Form(...),
        model_a: str = Form(...),
        model_b: str = Form(...),
        winner: str = Form(...),        # "a", "b", or "tie"
        is_blind: str = Form("false"),
    ):
        """Persist a comparison verdict owned by the caller. Companion scope."""
        if not has_companion_scope(request):
            raise HTTPException(403, "This token is not allowed to record comparisons.")
        if winner not in ("a", "b", "tie"):
            raise HTTPException(400, "winner must be 'a', 'b', or 'tie'")
        import uuid as _uuid
        from datetime import datetime as _dt
        from core.database import SessionLocal, Comparison

        owner = token_owner(request)
        if not owner:
            raise HTTPException(403, "Could not resolve an owner for this token.")
        comp_id = str(_uuid.uuid4())
        db = SessionLocal()
        try:
            comp = Comparison(
                id=comp_id,
                prompt=(prompt or "")[:500],
                model_a=model_a,
                model_b=model_b,
                endpoint_a="",
                endpoint_b="",
                winner=winner,
                is_blind=str(is_blind).lower() == "true",
                voted_at=_dt.utcnow(),
                owner=owner,
            )
            db.add(comp)
            db.commit()
        finally:
            db.close()
        return {"id": comp_id, "status": "ok"}

    @router.delete("/compare/{comp_id}")
    def compare_delete(request: Request, comp_id: str):
        """Delete one of the caller's comparisons. Strict ownership: missing OR
        cross-owner (incl. legacy null-owner shared) → 404, never confirming
        existence to a non-owner. Companion scope."""
        if not has_companion_scope(request):
            raise HTTPException(403, "This token is not allowed to delete comparisons.")
        from core.database import SessionLocal, Comparison

        owner = token_owner(request)
        db = SessionLocal()
        try:
            comp = db.query(Comparison).filter(Comparison.id == comp_id).first()
            if not comp or comp.owner != owner:
                raise HTTPException(404, "Comparison not found")
            db.delete(comp)
            db.commit()
            return {"status": "deleted"}
        finally:
            db.close()

    # ---- Calendar ---------------------------------------------------------
    # Owner-scoped view + light editing of the caller's calendars/events. Events
    # are scoped THROUGH their calendar's owner (the stock route joins on
    # CalendarCal.owner); we resolve the caller's calendar ids first, then scope
    # events to that set. v1 does NOT expand RRULEs — recurring events surface at
    # their base date only. Companion scope required.

    def _cal_iso(dt):
        try:
            return dt.isoformat() if dt is not None else None
        except Exception:
            return None

    def _cal_parse_dt(value):
        from datetime import datetime as _dt
        if not value:
            return None
        try:
            return _dt.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None

    @router.get("/calendars")
    def calendars(request: Request):
        """List the caller's own calendars. Companion scope."""
        if not has_companion_scope(request):
            raise HTTPException(403, "This token is not allowed to read calendars.")
        from core.database import SessionLocal, CalendarCal

        owner = token_owner(request)
        out = []
        db = SessionLocal()
        try:
            q = db.query(CalendarCal)
            if owner:
                q = q.filter((CalendarCal.owner == owner) | (CalendarCal.owner == None))  # noqa: E711
            for c in q.all():
                if not owner_can_see(c.owner, owner):
                    continue
                out.append({"id": c.id, "name": c.name, "color": c.color, "source": c.source})
        finally:
            db.close()
        return {"items": out}

    @router.get("/events")
    def events(request: Request, start: str = "", end: str = ""):
        """The caller's events overlapping [start, end] (ISO). Scoped to the
        caller's calendars. Non-recurring overlap only (no RRULE expansion in
        v1). Companion scope."""
        if not has_companion_scope(request):
            raise HTTPException(403, "This token is not allowed to read events.")
        from core.database import SessionLocal, CalendarCal, CalendarEvent

        owner = token_owner(request)
        start_dt = _cal_parse_dt(start)
        end_dt = _cal_parse_dt(end)
        out = []
        db = SessionLocal()
        try:
            cq = db.query(CalendarCal)
            if owner:
                cq = cq.filter((CalendarCal.owner == owner) | (CalendarCal.owner == None))  # noqa: E711
            cal_ids = [c.id for c in cq.all() if owner_can_see(c.owner, owner)]
            if cal_ids:
                for e in db.query(CalendarEvent).filter(CalendarEvent.calendar_id.in_(cal_ids)).all():
                    if e.status == "cancelled":
                        continue
                    # Window overlap (when no/unparseable range given → return all).
                    if start_dt and e.dtend is not None and e.dtend <= start_dt:
                        continue
                    if end_dt and e.dtstart is not None and e.dtstart >= end_dt:
                        continue
                    out.append({
                        "uid": e.uid,
                        "calendar_id": e.calendar_id,
                        "summary": e.summary,
                        "description": e.description,
                        "location": e.location,
                        "dtstart": _cal_iso(e.dtstart),
                        "dtend": _cal_iso(e.dtend),
                        "all_day": bool(e.all_day),
                        "rrule": e.rrule or "",
                        "status": e.status,
                        "importance": e.importance,
                        "event_type": e.event_type,
                        "color": e.color,
                    })
        finally:
            db.close()
        out.sort(key=lambda r: r.get("dtstart") or "")
        return {"items": out}

    def _owned_calendar(db, cal_id, owner):
        from core.database import CalendarCal
        cal = db.query(CalendarCal).filter(CalendarCal.id == cal_id).first()
        # Strict: the calendar must be the caller's own (not missing, not a
        # legacy null-owner shared row) before we let them write into it.
        if not cal or cal.owner != owner:
            raise HTTPException(404, "Calendar not found")
        return cal

    @router.post("/events")
    def create_event(
        request: Request,
        calendar_id: str = Form(...),
        summary: str = Form(...),
        dtstart: str = Form(...),
        dtend: str = Form(...),
        description: str = Form(""),
        location: str = Form(""),
        all_day: str = Form("false"),
    ):
        """Create an event in one of the caller's OWN calendars. Companion scope."""
        if not has_companion_scope(request):
            raise HTTPException(403, "This token is not allowed to create events.")
        import uuid as _uuid
        from core.database import SessionLocal, CalendarEvent

        owner = token_owner(request)
        if not owner:
            raise HTTPException(403, "Could not resolve an owner for this token.")
        start_dt = _cal_parse_dt(dtstart)
        end_dt = _cal_parse_dt(dtend)
        if start_dt is None or end_dt is None:
            raise HTTPException(400, "dtstart and dtend must be ISO datetimes")
        db = SessionLocal()
        try:
            _owned_calendar(db, calendar_id, owner)
            uid = str(_uuid.uuid4())
            ev = CalendarEvent(
                uid=uid,
                calendar_id=calendar_id,
                summary=summary,
                description=description or "",
                location=location or "",
                dtstart=start_dt,
                dtend=end_dt,
                all_day=str(all_day).lower() == "true",
                status="confirmed",
            )
            db.add(ev)
            db.commit()
            return {"uid": uid, "status": "ok"}
        finally:
            db.close()

    @router.delete("/events/{uid}")
    def delete_event(request: Request, uid: str):
        """Delete one of the caller's events. 404 (not 403) when the event is
        missing or lives in a calendar the caller doesn't own. Companion scope."""
        if not has_companion_scope(request):
            raise HTTPException(403, "This token is not allowed to delete events.")
        from core.database import SessionLocal, CalendarEvent

        owner = token_owner(request)
        db = SessionLocal()
        try:
            ev = db.query(CalendarEvent).filter(CalendarEvent.uid == uid).first()
            if not ev:
                raise HTTPException(404, "Event not found")
            # Ownership is via the event's calendar — reuse the strict gate.
            _owned_calendar(db, ev.calendar_id, owner)
            db.delete(ev)
            db.commit()
            return {"status": "deleted"}
        finally:
            db.close()

    # ---- Email ------------------------------------------------------------
    # Owner-scoped email for the phone. Account selection is the security crux:
    # every endpoint resolves the token's real owner and calls
    # email_helpers._assert_owns_account(account_id, owner) BEFORE touching creds
    # or opening IMAP/SMTP — so a caller can only ever act on their OWN mailbox
    # (cross-owner account_id → 404). Reuses the vetted module-level helpers
    # rather than reimplementing transport. Companion scope required. Never
    # returns imap/smtp passwords.

    @router.get("/email/accounts")
    def email_accounts(request: Request):
        """The caller's own email accounts (no secrets). Companion scope."""
        if not has_companion_scope(request):
            raise HTTPException(403, "This token is not allowed to read email.")
        from core.database import SessionLocal, EmailAccount

        owner = token_owner(request)
        out = []
        db = SessionLocal()
        try:
            q = db.query(EmailAccount)
            if owner:
                q = q.filter((EmailAccount.owner == owner) | (EmailAccount.owner == None))  # noqa: E711
            for a in q.all():
                if not owner_can_see(a.owner, owner):
                    continue
                out.append({
                    "id": a.id,
                    "name": a.name,
                    "from_address": a.from_address,
                    "enabled": bool(a.enabled),
                    "is_default": bool(a.is_default),
                })
        finally:
            db.close()
        return {"items": out}

    @router.get("/email/messages")
    def email_messages(request: Request, account_id: str, folder: str = "INBOX", limit: int = 30):
        """List recent message headers from one of the caller's mailboxes.
        Owner-asserted before any IMAP I/O. Companion scope."""
        if not has_companion_scope(request):
            raise HTTPException(403, "This token is not allowed to read email.")
        import email as _email
        from routes.email_helpers import _assert_owns_account, _imap, _decode_header

        owner = token_owner(request)
        _assert_owns_account(account_id, owner)  # 404 on cross-owner — the gate
        limit = max(1, min(int(limit or 30), 100))
        out = []
        try:
            with _imap(account_id, owner=owner) as conn:
                conn.select(folder, readonly=True)
                typ, data = conn.uid("search", None, "ALL")
                uids = (data[0].split() if data and data[0] else [])[-limit:]
                for u in reversed(uids):
                    typ, md = conn.uid("fetch", u, "(BODY.PEEK[HEADER.FIELDS (SUBJECT FROM DATE)])")
                    if not md or not md[0]:
                        continue
                    msg = _email.message_from_bytes(md[0][1])
                    out.append({
                        "uid": u.decode() if isinstance(u, bytes) else str(u),
                        "subject": _decode_header(msg.get("Subject")),
                        "from": _decode_header(msg.get("From")),
                        "date": msg.get("Date"),
                    })
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(502, f"Could not reach the mailbox: {e}")
        return {"items": out, "folder": folder}

    @router.get("/email/message/{uid}")
    def email_message(request: Request, uid: str, account_id: str, folder: str = "INBOX"):
        """Read one message's text body. Owner-asserted. Companion scope."""
        if not has_companion_scope(request):
            raise HTTPException(403, "This token is not allowed to read email.")
        import email as _email
        from routes.email_helpers import _assert_owns_account, _imap, _decode_header, _extract_text

        owner = token_owner(request)
        _assert_owns_account(account_id, owner)
        try:
            with _imap(account_id, owner=owner) as conn:
                conn.select(folder, readonly=True)
                typ, md = conn.uid("fetch", uid.encode() if isinstance(uid, str) else uid, "(RFC822)")
                if not md or not md[0]:
                    raise HTTPException(404, "Message not found")
                msg = _email.message_from_bytes(md[0][1])
                return {
                    "uid": uid,
                    "subject": _decode_header(msg.get("Subject")),
                    "from": _decode_header(msg.get("From")),
                    "to": _decode_header(msg.get("To")),
                    "date": msg.get("Date"),
                    "body": _extract_text(msg),
                }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(502, f"Could not read the message: {e}")

    @router.post("/email/send")
    def email_send(
        request: Request,
        account_id: str = Form(...),
        to: str = Form(...),
        subject: str = Form(""),
        body: str = Form(""),
    ):
        """Send a plain-text email from one of the caller's OWN accounts.
        Owner-asserted before creds are read. Companion scope."""
        if not has_companion_scope(request):
            raise HTTPException(403, "This token is not allowed to send email.")
        from email.mime.text import MIMEText
        from email.utils import parseaddr
        from routes.email_helpers import _assert_owns_account, _get_email_config, _send_smtp_message

        owner = token_owner(request)
        if not owner:
            raise HTTPException(403, "Could not resolve an owner for this token.")
        _assert_owns_account(account_id, owner)

        recipients = [r.strip() for r in to.replace(";", ",").split(",") if r.strip()]
        if not recipients or not all("@" in parseaddr(r)[1] for r in recipients):
            raise HTTPException(400, "Provide at least one valid recipient address.")

        cfg = _get_email_config(account_id, owner=owner)
        from_addr = cfg.get("from_address") or cfg.get("smtp_user") or ""
        if not cfg.get("smtp_host") or not from_addr:
            raise HTTPException(400, "This account has no SMTP configuration.")
        msg = MIMEText(body or "", _charset="utf-8")
        msg["Subject"] = subject or ""
        msg["From"] = from_addr
        msg["To"] = ", ".join(recipients)
        try:
            _send_smtp_message(cfg, from_addr, recipients, msg.as_string())
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(502, f"Send failed: {e}")
        return {"status": "sent", "to": recipients}

    # ---- Writes -----------------------------------------------------------
    # The reads above are the established pattern; these add the phone's
    # create/delete/toggle affordances. Each write requires the companion scope
    # and a resolvable owner, stamps that owner on new rows, and enforces strict
    # ownership on mutate/delete (404 — never confirm a row's existence to a
    # non-owner), exactly like the desktop note/memory routes. Writes hit the
    # same tables the matching GET reads, so the mobile list stays consistent.

    def _owned_note(db, note_id: str, owner):
        from core.database import Note
        note = db.query(Note).filter(Note.id == note_id).first()
        if not note or note.owner != owner:
            raise HTTPException(404, "Note not found")
        return note

    @router.post("/notes")
    def add_note(
        request: Request,
        title: str = Form(""),
        content: str = Form(None),
        items: str = Form(None),
        pinned: bool = Form(False),
    ):
        """Create a note or checklist. `items`, when given, is a JSON array of
        {text, done} objects (a checklist); otherwise it's a plain text note."""
        require_companion_scope(request)
        owner = writer_owner(request)

        checklist_json = None
        note_type = "note"
        if items:
            try:
                parsed = _json.loads(items)
            except (ValueError, TypeError):
                raise HTTPException(400, "items must be a JSON array")
            if not isinstance(parsed, list):
                raise HTTPException(400, "items must be a JSON array")
            norm = [
                {"text": str(it.get("text", "")), "done": bool(it.get("done", False))}
                for it in parsed if isinstance(it, dict)
            ]
            checklist_json = _json.dumps(norm)
            note_type = "checklist"

        if not (title or "").strip() and not (content or "") and not checklist_json:
            raise HTTPException(400, "empty note")

        from core.database import SessionLocal, Note
        db = SessionLocal()
        try:
            note = Note(
                id=str(uuid.uuid4()), owner=owner, title=title or "", content=content,
                items=checklist_json, note_type=note_type, pinned=bool(pinned), source="mobile",
            )
            db.add(note)
            db.commit()
            db.refresh(note)
            return _serialize_note(note)
        finally:
            db.close()

    @router.delete("/notes/{note_id}")
    def delete_note(request: Request, note_id: str):
        """Delete one of the caller's notes."""
        require_companion_scope(request)
        owner = writer_owner(request)
        from core.database import SessionLocal
        db = SessionLocal()
        try:
            db.delete(_owned_note(db, note_id, owner))
            db.commit()
            return {"ok": True}
        finally:
            db.close()

    @router.post("/notes/{note_id}/pin")
    def toggle_note_pin(request: Request, note_id: str):
        """Flip a note's pinned flag."""
        require_companion_scope(request)
        owner = writer_owner(request)
        from core.database import SessionLocal
        db = SessionLocal()
        try:
            note = _owned_note(db, note_id, owner)
            note.pinned = not note.pinned
            db.commit()
            return {"ok": True, "pinned": note.pinned}
        finally:
            db.close()

    @router.post("/notes/{note_id}/items/{index}/toggle")
    def toggle_note_item(request: Request, note_id: str, index: int):
        """Toggle the done state of one checklist item by index."""
        require_companion_scope(request)
        owner = writer_owner(request)
        from core.database import SessionLocal
        from sqlalchemy.orm.attributes import flag_modified
        db = SessionLocal()
        try:
            note = _owned_note(db, note_id, owner)
            try:
                checklist = _json.loads(note.items) if note.items else None
            except (ValueError, TypeError):
                checklist = None
            if not isinstance(checklist, list):
                raise HTTPException(400, "Note has no checklist items")
            if index < 0 or index >= len(checklist):
                raise HTTPException(400, f"Item index {index} out of range")
            checklist[index]["done"] = not checklist[index].get("done", False)
            note.items = _json.dumps(checklist)
            flag_modified(note, "items")
            db.commit()
            return {"ok": True, "items": checklist}
        finally:
            db.close()

    @router.post("/memory")
    def add_memory(request: Request, text: str = Form(...), category: str = Form("fact")):
        """Create a memory owned by the caller. Writes the same `memories` table
        GET /memory reads, so it appears in the mobile list immediately."""
        require_companion_scope(request)
        owner = writer_owner(request)
        text = (text or "").strip()
        if not text:
            raise HTTPException(400, "empty memory")
        cat = category if category in _MEMORY_CATEGORIES else "fact"

        from core.database import SessionLocal, Memory
        db = SessionLocal()
        try:
            row = Memory(
                id=str(uuid.uuid4()), text=text, category=cat,
                source="mobile", owner=owner, timestamp=int(time.time()),
            )
            db.add(row)
            db.commit()
            db.refresh(row)
            return {"id": row.id, "text": row.text, "category": row.category}
        finally:
            db.close()

    @router.delete("/memory/{memory_id}")
    def delete_memory(request: Request, memory_id: str):
        """Delete one of the caller's memories."""
        require_companion_scope(request)
        owner = writer_owner(request)
        from core.database import SessionLocal, Memory
        db = SessionLocal()
        try:
            row = db.query(Memory).filter(Memory.id == memory_id).first()
            if not row or row.owner != owner:
                raise HTTPException(404, "Memory not found")
            db.delete(row)
            db.commit()
            return {"ok": True}
        finally:
            db.close()

    return router
