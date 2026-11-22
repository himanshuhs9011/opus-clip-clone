"""ClipForge API + React UI.

    python -m uvicorn app:app --port 8000      ->  http://127.0.0.1:8000
"""
from __future__ import annotations

import html
import io
import json
import os
import re
import shutil
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from opusclip import pipeline
from opusclip.captions import list_styles
from opusclip.config import settings, update_env
from opusclip.edit import ASPECTS, DEFAULT_EDIT, FILTERS, LAYOUTS
from opusclip import autopilot, providers, publish, wizard
from opusclip.llm import LLM, OLLAMA_PICKS, health as llm_health, ollama_models, pull_state, start_pull
from opusclip.render import encoder_name, preview_plan
from opusclip.text import download

app = FastAPI(title="ClipForge", version="2.1.0")

DIST = settings.root / "frontend" / "dist"
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".flv", ".ts", ".wmv"}
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3-turbo", "large-v3"]

app.mount("/media", StaticFiles(directory=str(settings.output_dir)), name="media")


def _project_or_404(pid: str):
    p = pipeline.get(pid)
    if not p:
        raise HTTPException(404, "project not found")
    return p


def _plain(x):
    """numpy scalars/arrays -> plain Python so the JSON encoder never chokes."""
    return json.loads(json.dumps(x, default=lambda o: o.item() if hasattr(o, "item") else
                                 (o.tolist() if hasattr(o, "tolist") else str(o))))


def _check_cid(cid: str) -> None:
    if not pipeline.CID_RE.match(cid or ""):
        raise HTTPException(400, "bad clip id")


# --------------------------------------------------------------------------
# meta
# --------------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "ffmpeg": bool(shutil.which("ffmpeg")),
        "yt_dlp": bool(shutil.which("yt-dlp")),
        "encoder": encoder_name(),
        "llm": llm_health(),
        "whisper_model": settings.whisper_model,
        "whisper_models": WHISPER_MODELS,
    }


@app.get("/api/presets")
def presets() -> dict:
    return {
        "captions": list_styles(),
        "aspects": {k: list(v) for k, v in ASPECTS.items()},
        "layouts": list(LAYOUTS),
        "filters": list(FILTERS),
        "lengths": {k: list(v) for k, v in pipeline.LENGTHS.items()},
        "default_edit": DEFAULT_EDIT,
        "default_options": {**pipeline.DEFAULT_OPTIONS, "provider": settings.llm_provider,
                            "whisper_model": settings.whisper_model},
    }


@app.get("/api/settings")
def get_settings() -> dict:
    key = settings.llm_api_key
    return {
        "llm_provider": providers.active(),
        "ollama_host": settings.ollama_host,
        "ollama_model": settings.ollama_model,
        "ollama_models": ollama_models(),
        "llm_base_url": settings.llm_base_url,
        "llm_model": settings.llm_model,
        "openrouter_key_set": bool(key),
        "openrouter_key_hint": f"…{key[-4:]}" if len(key) > 8 else "",
        "whisper_model": settings.whisper_model,
        "whisper_models": WHISPER_MODELS,
        "encoder": encoder_name(),
    }


@app.put("/api/settings")
def put_settings(body: dict = Body(...)) -> dict:
    allowed = {"llm_provider", "ollama_host", "ollama_model", "llm_model", "llm_api_key",
               "whisper_model", "llm_base_url"}
    vals = {k: str(v).strip() for k, v in body.items() if k in allowed and v is not None}
    if "llm_provider" in vals and vals["llm_provider"] not in ("ollama", "openrouter", "openai", "none"):
        raise HTTPException(400, "unknown provider")
    if "llm_api_key" in vals and not vals["llm_api_key"]:
        vals.pop("llm_api_key")   # empty field = keep the stored key
    if "whisper_model" in vals and vals["whisper_model"] not in WHISPER_MODELS:
        raise HTTPException(400, "unknown whisper model")
    if "llm_provider" in vals:
        providers.set_active(vals.pop("llm_provider").replace("openai", "custom"))
    update_env(vals)
    return get_settings()


# --------------------------------------------------------------------------
# providers: the user's own API keys, model allow-lists, local model downloads
# --------------------------------------------------------------------------
@app.get("/api/providers")
def get_providers() -> dict:
    return {**providers.public(), "ollama_models": ollama_models(), "ollama_picks": OLLAMA_PICKS,
            "pull": pull_state}


@app.put("/api/providers/active")
def set_active_provider(body: dict = Body(...)) -> dict:
    try:
        providers.set_active(str(body.get("id", "")))
    except KeyError:
        raise HTTPException(400, "unknown provider")
    return get_providers()


@app.put("/api/providers/{pid}")
def put_provider(pid: str, body: dict = Body(...)) -> dict:
    try:
        providers.update(pid, body)
    except KeyError:
        raise HTTPException(404, "unknown provider")
    return get_providers()


SKIP_MODELS = ("embed", "safety", "guard", "rerank", "whisper", "tts", "image", "vision-only", "parse")


@app.get("/api/providers/{pid}/models")
def provider_models(pid: str) -> dict:
    """Models the provider offers, for the allow-list picker."""
    import httpx

    if pid == "ollama":
        return {"models": [{"id": m, "free": True} for m in ollama_models()]}
    cfg = providers.get(pid)
    if not cfg or not cfg["base_url"]:
        raise HTTPException(400, "add the server address first")
    headers = {"Authorization": f"Bearer {cfg['key']}"} if cfg["key"] else {}
    try:
        r = httpx.get(f"{cfg['base_url']}/models", headers=headers, timeout=20)
        r.raise_for_status()
        data = r.json().get("data", [])
    except Exception as e:
        raise HTTPException(502, f"could not list models: {str(e)[:150]}")
    out = []
    for m in data:
        mid = str(m.get("id", "")).removeprefix("models/")
        if not mid or any(x in mid for x in SKIP_MODELS):
            continue
        price = m.get("pricing") or {}
        free = mid.endswith(":free") or (bool(price) and all(str(v) in ("0", "0.0") for v in price.values()))
        out.append({"id": mid, "free": free if pid == "openrouter" else None,
                    "context": m.get("context_length")})
    out.sort(key=lambda x: (not x["free"], x["id"]) if pid == "openrouter" else (0, x["id"]))
    return {"models": out}


@app.post("/api/providers/{pid}/test")
def test_provider(pid: str, body: dict = Body(default={})) -> dict:
    import time

    llm = LLM(pid, body.get("model") or None)
    if body.get("model"):
        llm.models = [body["model"]]          # test exactly the model asked for
    ok, detail = llm.verify()
    if ok:
        try:
            t = time.time()
            out = llm.json_completion(
                "Reply with JSON only.",
                'Return {"ok": true, "line": "<one short Hinglish line: a gamer wins a 1v4 clutch>"}',
                max_tokens=200)
            ok = bool(out.get("ok"))
            detail = f"{llm.used_model or llm.model} answered in {time.time() - t:.1f}s"
            if out.get("line"):
                detail += f': "{str(out["line"])[:80]}"'
        except Exception as e:
            ok, detail = False, str(e)[:220]
    return {"ok": ok, "detail": detail}


@app.post("/api/ollama/pull")
def ollama_pull(body: dict = Body(...)) -> dict:
    model = str(body.get("model", "")).strip()
    if not re.fullmatch(r"[\w.\-/:]+", model):
        raise HTTPException(400, "bad model name")
    if not start_pull(model, providers.get("ollama")["base_url"]):
        raise HTTPException(409, "a download is already running")
    return pull_state


@app.get("/api/ollama/pull")
def ollama_pull_state() -> dict:
    return pull_state


@app.post("/api/settings/test")
def test_llm(body: dict = Body(default={})) -> dict:
    llm = LLM(body.get("provider") or settings.llm_provider, body.get("model") or None)
    ok, detail = llm.verify()
    if ok and body.get("deep"):
        try:
            out = llm.json_completion("Reply with JSON.", 'Return {"ok": true}', max_tokens=50)
            ok, detail = bool(out.get("ok")), "model answered"
        except Exception as e:
            ok, detail = False, str(e)[:200]
    return {"ok": ok, "detail": detail, "label": llm.label}


# --------------------------------------------------------------------------
# projects
# --------------------------------------------------------------------------
@app.get("/api/projects")
def projects() -> dict:
    return {"projects": pipeline.list_projects()}


@app.get("/api/projects/{pid}")
def project(pid: str, transcript: bool = True) -> dict:
    return _plain(_project_or_404(pid).to_dict(include_transcript=transcript))


@app.delete("/api/projects/{pid}")
def delete_project(pid: str) -> dict:
    if not pipeline.valid_pid(pid):
        raise HTTPException(400, "bad project id")
    try:
        pipeline.delete(pid)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {"deleted": pid}


def _options(raw: str) -> dict:
    try:
        o = json.loads(raw or "{}")
        return o if isinstance(o, dict) else {}
    except json.JSONDecodeError:
        raise HTTPException(400, "options must be JSON")


def _save_upload(file: UploadFile, dst: Path) -> None:
    with dst.open("wb") as fh:
        shutil.copyfileobj(file.file, fh, length=4 * 1024 * 1024)


@app.post("/api/projects")
async def create_project(
    file: UploadFile | None = File(None),
    url: str = Form(""),
    options: str = Form("{}"),
) -> dict:
    return await _create(file, url, _options(options))


async def _create(file: UploadFile | None, url: str, opts: dict) -> dict:
    opts["length"] = "ai"     # clip length is decided by the AI per clip, not by a setting
    if file is not None and file.filename:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in VIDEO_EXT:
            raise HTTPException(400, f"unsupported file type: {suffix}")
        dst = settings.upload_dir / f"{uuid.uuid4().hex[:10]}{suffix}"
        await run_in_threadpool(_save_upload, file, dst)
        p = pipeline.new_project(dst, name=Path(file.filename).stem[:80], options=opts)
        pipeline.start(p, dst)
        return {"id": p.id, "status": p.status}

    url = url.strip()
    if url:
        if not re.match(r"^https?://", url, re.I):
            raise HTTPException(400, "links must start with http:// or https://")
        if os.getenv("DOWNLOAD_MODE", "smart").lower() != "full":
            # smart: audio first, then only the video pieces around the chosen clips
            p = pipeline.new_project(Path("remote.mp4"), name=url[:80], options=opts)
            p.remote = {"url": url}
            p.stage, p.message = "download", "Reading the link"
            pipeline.save(p)
            pipeline.start(p, None)
            return {"id": p.id, "status": p.status}

        placeholder = settings.upload_dir / f"{uuid.uuid4().hex[:10]}.mp4"
        p = pipeline.new_project(placeholder, name=url[:80], options=opts)
        p.stage, p.message = "download", "Downloading video"
        pipeline.save(p)

        def job() -> None:
            try:
                src = download(url, placeholder)
            except Exception as e:
                for f in placeholder.parent.glob(placeholder.stem + ".*"):  # partial yt-dlp files
                    try:
                        f.unlink()
                    except OSError:
                        pass
                p.status, p.stage = "error", "error"
                p.error = p.message = f"download failed: {str(e)[:400]}"
                pipeline.save(p)
                return
            pipeline.run(p, src)

        pipeline._job_pool.submit(job)
        return {"id": p.id, "status": p.status}

    raise HTTPException(400, "provide a file or a link")


@app.post("/api/projects/{pid}/rerun")
def rerun(pid: str, body: dict = Body(default={})) -> dict:
    _project_or_404(pid)
    try:
        p = pipeline.rerun(pid, {**(body.get("options") or {}), "length": "ai"})
    except FileNotFoundError as e:
        raise HTTPException(400, str(e))
    return {"id": pid, "status": p.status if p else "error"}


@app.post("/api/projects/{pid}/finish")
def finish(pid: str, body: dict = Body(default={})) -> dict:
    """Make (render) the clips the user picked; ids=null makes every clip found."""
    _project_or_404(pid)
    ids = body.get("ids")
    if ids is not None and (not isinstance(ids, list) or not all(isinstance(x, str) for x in ids)):
        raise HTTPException(400, "ids must be a list of clip ids")
    try:
        p = pipeline.finish(pid, ids)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {"id": pid, "status": p.status if p else "error"}


# --------------------------------------------------------------------------
# Clip Agent: describe the clips you want in plain words (opusclip/wizard.py)
# --------------------------------------------------------------------------
_ai_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="agent")   # never blocks clip jobs


def _read_request(text: str, use_ai: bool = True) -> dict:
    req = wizard.parse_request(text)
    if use_ai and wizard.needs_ai_focus(req):
        llm = LLM(providers.active())
        if llm.provider != "none" and llm.available:
            # the plan card must not hang on a cold local model: give the AI 25 s at most
            fut = _ai_pool.submit(wizard.ai_focus, text, llm)
            try:
                f = fut.result(timeout=25)
            except Exception:
                f = None
            if f:
                req["focus"] = f
                req["focus_by"] = "ai"
    return req


@app.post("/api/agent/plan")
def agent_plan(body: dict = Body(...)) -> dict:
    text = str(body.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "say what clips you want")
    req = _read_request(text, use_ai=bool(body.get("ai", True)))
    return {"request": req, "plan": wizard.plan(req, has_source=bool(body.get("has_file")))}


@app.post("/api/agent/start")
async def agent_start(
    file: UploadFile | None = File(None),
    text: str = Form(""),
    request: str = Form(""),
    options: str = Form("{}"),
) -> dict:
    """Start a find -> pick -> make project from an approved plan."""
    req = _options(request) if request else _read_request(text, use_ai=False)
    if not req:
        req = wizard.parse_request(text)
    url = str(req.get("url") or "")
    if (file is None or not file.filename) and not url:
        raise HTTPException(400, "send the video: paste a link or attach a file")
    opts = wizard.to_options(req, _options(options))
    return await _create(file, url, opts)


@app.post("/api/agent/followup")
def agent_followup(body: dict = Body(...)) -> dict:
    """A message about a project already found: "make 3", "make all", or a new ask
    ("only the funny ones", "shorter, under 20 seconds") that re-finds clips."""
    pid = str(body.get("pid") or "")
    p = _project_or_404(pid)
    text = str(body.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "say what to change")
    req = _read_request(text)
    if req.get("make"):
        cands = sorted([c for c in p.clips if (c.render or {}).get("status") == "candidate"],
                       key=lambda c: -(c.score.total if c.score else 0))
        ids = [c.id for c in cands] if req["make"] == "all" else [c.id for c in cands[: int(req["make"])]]
        try:
            pipeline.finish(pid, ids)
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        return {"action": "make", "ids": ids}
    # a new ask on the same video: keep what the user said before unless they changed it
    prev = dict(p.options or {})
    new = wizard.to_options(req, prev)
    for k in ("max_clip", "min_clip", "range"):
        if req.get(k) is None and prev.get(k) is not None:
            new[k] = prev[k]
    if not req.get("count") and prev.get("clips"):
        new["clips"] = prev["clips"]
    if req.get("format") == "short" and not re.search(r"\bshorts?\b|\breels?\b|\btiktoks?\b", text, re.I):
        new["format"], new["long_len"] = prev.get("format") or "short", prev.get("long_len")
    if not req.get("focus") and not re.search(r"\b(any|all|everything|best)\b", text, re.I):
        new["focus"] = prev.get("focus") or ""
    try:
        pipeline.rerun(pid, new)
    except FileNotFoundError as e:
        raise HTTPException(400, str(e))
    merged = {**req, "focus": new.get("focus"), "max_clip": new.get("max_clip"),
              "min_clip": new.get("min_clip"), "range": new.get("range"),
              "count": new.get("clips") or None}
    return {"action": "refind", "request": merged, "plan": wizard.plan(merged, has_source=True)}


# --------------------------------------------------------------------------
# Autopilot: watch channels, clip new videos, post on a schedule (opusclip/autopilot.py)
# --------------------------------------------------------------------------
@app.on_event("startup")
def _start_autopilot() -> None:
    autopilot.start()


@app.get("/api/autopilot")
def autopilot_state() -> dict:
    return autopilot.public()


@app.put("/api/autopilot")
def autopilot_update(body: dict = Body(...)) -> dict:
    try:
        return autopilot.update(body)
    except (TypeError, ValueError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/autopilot/sources")
def autopilot_add_source(body: dict = Body(...)) -> dict:
    try:
        autopilot.add_source(str(body.get("url") or ""), fmt=str(body.get("format") or "both"),
                             focus=str(body.get("focus") or ""), backfill=int(body.get("backfill", 1)),
                             name=str(body.get("name") or ""))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return autopilot.public()


@app.delete("/api/autopilot/sources/{sid}")
def autopilot_remove_source(sid: str) -> dict:
    autopilot.remove_source(sid)
    return autopilot.public()


@app.post("/api/autopilot/check")
def autopilot_check_now() -> dict:
    """Look for new videos now (in the background) instead of waiting for the next check."""
    def job() -> None:
        try:
            autopilot.follow_jobs()
            n = autopilot.check_sources()
            autopilot.log("check now: " + ("started a new video" if n else "nothing new (or a video is still being clipped)"))
        except Exception as e:
            autopilot.log(f"check failed: {str(e)[:200]}")
    _ai_pool.submit(job)
    return autopilot.public()


@app.post("/api/autopilot/post-now")
def autopilot_post_now() -> dict:
    """Post the next waiting clips now (up to today's caps)."""
    def job() -> None:
        n = autopilot.post_round(force=False)
        autopilot.log(f"post now: {n} posted")
    _ai_pool.submit(job)
    return autopilot.public()


@app.post("/api/autopilot/queue/{item_id}/{action}")
def autopilot_queue_action(item_id: str, action: str) -> dict:
    if action not in ("approve", "skip", "retry"):
        raise HTTPException(400, "unknown action")
    if not autopilot.set_item(item_id, action):
        raise HTTPException(404, "queue item not found")
    return autopilot.public()


_yt_states: dict[str, float] = {}


def _yt_redirect(request: Request) -> str:
    return str(request.base_url).rstrip("/") + "/api/autopilot/youtube/callback"


@app.put("/api/autopilot/youtube")
def autopilot_youtube(request: Request, body: dict = Body(...)) -> dict:
    """Save the Google OAuth client and return the sign-in link."""
    cid = str(body.get("client_id") or "").strip()
    secret = str(body.get("client_secret") or "").strip()
    with autopilot._lock:
        acc = autopilot.load()["accounts"].setdefault("youtube", {})
        if cid:
            acc["client_id"] = cid
        if secret:
            acc["client_secret"] = secret
        ready = bool(acc.get("client_id") and acc.get("client_secret"))
    autopilot.save()
    if not ready:
        raise HTTPException(400, "add the OAuth client ID and secret first")
    state = uuid.uuid4().hex
    _yt_states[state] = time.time()
    return {"auth_url": publish.youtube_auth_url(acc["client_id"], _yt_redirect(request), state),
            "redirect_uri": _yt_redirect(request)}


@app.get("/api/autopilot/youtube/callback", include_in_schema=False)
def autopilot_youtube_callback(request: Request, code: str = "", state: str = "", error: str = ""):
    if error or not code:
        return HTMLResponse(f"<p>Google sign-in was cancelled ({html.escape(error or 'no code')}). "
                            "<a href='/#/autopilot'>Back to Autopilot</a></p>")
    if state not in _yt_states or time.time() - _yt_states.pop(state) > 900:
        return HTMLResponse("<p>This sign-in link expired. <a href='/#/autopilot'>Try again</a></p>", 400)
    acc = autopilot.load()["accounts"]["youtube"]
    try:
        tok = publish.youtube_exchange(acc["client_id"], acc["client_secret"], code, _yt_redirect(request))
        ch = publish.youtube_channel(tok["access_token"])
    except publish.PublishError as e:
        return HTMLResponse(f"<p>Could not connect YouTube: {html.escape(str(e))}. "
                            "<a href='/#/autopilot'>Back</a></p>", 400)
    with autopilot._lock:
        acc.update(refresh_token=tok["refresh_token"], channel_id=ch["id"], channel_title=ch["title"])
    autopilot.save()
    autopilot.log(f"YouTube connected: {ch['title']}")
    return HTMLResponse("<script>location.replace('/#/autopilot')</script>")


@app.put("/api/autopilot/meta")
def autopilot_meta(body: dict = Body(...)) -> dict:
    """Connect Instagram + Facebook: app ID, app secret and a user token from Graph API Explorer."""
    app_id = str(body.get("app_id") or "").strip()
    secret = str(body.get("app_secret") or "").strip()
    token = str(body.get("user_token") or "").strip()
    with autopilot._lock:
        acc = autopilot.load()["accounts"].setdefault("meta", {})
        app_id, secret = app_id or acc.get("app_id", ""), secret or acc.get("app_secret", "")
    if not (app_id and secret and token):
        raise HTTPException(400, "app ID, app secret and an access token are all needed")
    try:
        res = publish.meta_connect(app_id, secret, token)
    except publish.PublishError as e:
        raise HTTPException(400, str(e))
    pages = res["pages"]
    pick = next((p for p in pages if p["ig_id"]), pages[0])
    with autopilot._lock:
        acc.update(app_id=app_id, app_secret=secret, pages=pages, page_id=pick["id"], page_name=pick["name"],
                   token=pick["token"], ig_id=pick["ig_id"], ig_username=pick["ig_username"])
    autopilot.save()
    autopilot.log(f"Meta connected: Page {pick['name']}" + (f", Instagram @{pick['ig_username']}" if pick["ig_username"] else ""))
    return autopilot.public()


@app.put("/api/autopilot/meta/page")
def autopilot_meta_page(body: dict = Body(...)) -> dict:
    with autopilot._lock:
        acc = autopilot.load()["accounts"].get("meta") or {}
        page = next((p for p in acc.get("pages") or [] if p["id"] == str(body.get("page_id"))), None)
        if not page:
            raise HTTPException(404, "page not found")
        acc.update(page_id=page["id"], page_name=page["name"], token=page["token"],
                   ig_id=page["ig_id"], ig_username=page["ig_username"])
    autopilot.save()
    return autopilot.public()


@app.delete("/api/autopilot/accounts/{name}")
def autopilot_disconnect(name: str) -> dict:
    if name not in ("youtube", "meta"):
        raise HTTPException(404, "unknown account")
    with autopilot._lock:
        autopilot.load()["accounts"][name] = {}
    autopilot.save()
    autopilot.log(f"{name} disconnected")
    return autopilot.public()


@app.put("/api/projects/{pid}/clips/{cid}")
def update_clip(pid: str, cid: str, body: dict = Body(...)) -> dict:
    _check_cid(cid)
    _project_or_404(pid)
    clip = pipeline.update_clip(pid, cid, edit=body.get("edit"), title=body.get("title"),
                                hashtags=body.get("hashtags"))
    if not clip:
        raise HTTPException(404, "clip not found")
    if body.get("render"):
        clip = pipeline.queue_render(pid, cid)
    return clip.to_dict()


@app.post("/api/projects/{pid}/clips/{cid}/plan")
def clip_plan(pid: str, cid: str, body: dict = Body(default={})) -> dict:
    """What the renderer will do with this edit - drives the live preview in the editor."""
    _check_cid(cid)
    p = _project_or_404(pid)
    clip = pipeline.find_clip(p, cid)
    if not clip:
        raise HTTPException(404, "clip not found")
    plan = preview_plan(clip, p.sentences, p.width, p.height, body.get("edit"))
    preview = p.preview
    offset = 0.0
    if not preview:
        src = pipeline.source_path(p)
        preview = src.name if src else None
    if not preview and p.segments:
        # link project: the downloaded piece that overlaps this clip the most
        a, b = plan["a"], plan["b"]
        seg = max(p.segments, key=lambda s: min(b, s["t1"]) - max(a, s["t0"]))
        if min(b, seg["t1"]) - max(a, seg["t0"]) > 0:
            preview, offset = seg["file"], float(seg["t0"])
            plan["partial"] = seg["t0"] > a + 0.05 or seg["t1"] < b - 0.05
    plan["video"] = f"/media/{pid}/{preview}" if preview else None
    plan["video_offset"] = offset
    plan["source"] = {"w": p.width, "h": p.height, "duration": p.duration}
    return _plain(plan)


@app.post("/api/projects/{pid}/clips/{cid}/render")
def render_clip(pid: str, cid: str) -> dict:
    _check_cid(cid)
    _project_or_404(pid)
    clip = pipeline.queue_render(pid, cid)
    if not clip:
        raise HTTPException(404, "clip not found")
    return clip.to_dict()


@app.post("/api/projects/{pid}/apply-all")
def apply_all(pid: str, body: dict = Body(...)) -> dict:
    _project_or_404(pid)
    p = pipeline.apply_to_all(pid, body.get("edit") or {}, render=bool(body.get("render", True)))
    return {"ok": True, "clips": len(p.clips) if p else 0}


@app.get("/api/projects/{pid}/zip")
def zip_project(pid: str):
    p = _project_or_404(pid)
    d = pipeline.project_dir(pid)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for c in sorted(p.clips, key=lambda c: c.rank or 99):
            if not c.file or not (d / c.file).exists():
                continue
            slug = re.sub(r"[^\w-]+", "-", c.title.lower()).strip("-")[:40] or c.id
            base = f"{c.rank:02d}_{slug}"
            z.write(d / c.file, f"{base}.mp4")
            sc = c.score.total if c.score else "?"
            pk = c.package or {}
            chapters = "\n".join(f"{int(t // 60)}:{int(t % 60):02d} {label}" for t, label in pk.get("chapters") or [])
            z.writestr(f"{base}.txt",
                       f"YouTube title: {pk.get('yt_title') or c.title}\n"
                       f"Thumbnail text: {pk.get('thumbnail_text') or ''}\n\n"
                       f"{pk.get('description') or c.summary}\n\n"
                       + (f"Chapters:\n{chapters}\n\n" if chapters else "")
                       + f"{' '.join(c.hashtags)}\n\nVirality score: {sc}/100\n{c.why}\n")
    buf.seek(0)
    # HTTP headers are latin-1: a Hindi / emoji project name crashed the download. Send an
    # ASCII fallback plus the real name as RFC 5987 filename*.
    from urllib.parse import quote

    ascii_name = re.sub(r"[^A-Za-z0-9_-]+", "_", p.name).strip("_")[:40] or pid
    full = re.sub(r"[^\w-]+", "_", p.name)[:40] or pid
    disp = f"attachment; filename=\"{ascii_name}_clips.zip\"; filename*=UTF-8''{quote(full + '_clips.zip')}"
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition": disp})


# --------------------------------------------------------------------------
# React app (frontend/dist). In development run `npm run dev` in frontend/ instead.
# --------------------------------------------------------------------------
if (DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(DIST / "assets")), name="assets")


@app.get("/{path:path}", include_in_schema=False)
def spa(path: str):
    if path.startswith(("api/", "media/")):
        raise HTTPException(404)
    f = (DIST / path).resolve()
    if path and f.is_file() and DIST.resolve() in f.parents:
        return FileResponse(f)
    index = DIST / "index.html"
    if index.exists():
        return FileResponse(index)
    return HTMLResponse("<h2>UI not built</h2><p>Run <code>cd frontend &amp;&amp; npm install &amp;&amp; "
                        "npm run build</code>, then reload.</p>", status_code=503)
