"""ClipForge API + React UI.

    python -m uvicorn app:app --port 8000      ->  http://127.0.0.1:8000
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import uuid
import zipfile
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from opusclip import pipeline
from opusclip.captions import list_styles
from opusclip.config import settings, update_env
from opusclip.edit import ASPECTS, DEFAULT_EDIT, FILTERS, LAYOUTS
from opusclip import providers
from opusclip.llm import LLM, OLLAMA_PICKS, health as llm_health, ollama_models, pull_state, start_pull
from opusclip.render import encoder_name, preview_plan
from opusclip.text import download

app = FastAPI(title="ClipForge", version="2.0.0")

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
    opts = _options(options)
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
            z.writestr(f"{base}.txt", f"{c.title}\n\n{' '.join(c.hashtags)}\n\n"
                                      f"Virality score: {sc}/100\n{c.why}\n")
    buf.seek(0)
    name = re.sub(r"[^\w-]+", "_", p.name)[:40] or pid
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition": f'attachment; filename="{name}_clips.zip"'})


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
