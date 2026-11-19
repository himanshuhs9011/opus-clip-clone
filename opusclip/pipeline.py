from __future__ import annotations

import copy
import json
import re
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

from .audio import AudioFeatures, extract_wav
from .config import settings
from .edit import merge_edit
from .highlights import select
from .llm import LLM, unload_ollama
from .models import Clip, Project
from .probe import probe, run as run_cmd
from .reframe import covers, track
from .speakers import diarize
from .render import encoder_name, render_clip
from . import remote
from .text import build_sentences, download, transcribe

PID_RE = re.compile(r"^[0-9a-f]{8}$")
CID_RE = re.compile(r"^clip_\d{2,3}$")

# "ai" = the AI picks each clip's length from the content (only platform limits apply).
# The fixed ranges stay for the command line only.
LENGTHS = {"ai": (8.0, 90.0), "short": (12.0, 30.0), "medium": (30.0, 60.0), "long": (60.0, 90.0)}

# quality bar: clips scoring below it are dropped instead of padding the clip count
QUALITY = {"any": 0, "good": 65, "viral": 75}

DEFAULT_OPTIONS: dict = {
    "clips": 6,
    "length": "ai",
    "quality": "good",
    "topic": "",
    "provider": None,          # None = settings.llm_provider
    "llm_model": None,
    "whisper_model": None,
    "language": "",
    "preset": "bold-pop",
    "captions": True,
    "aspect": "9:16",
    "layout": "auto",
    "zoom": True,
    "title": True,
    "emoji": True,
    "progress": False,
    "silence": False,
    "fillers": False,
}

_live: dict[str, Project] = {}
_locks: dict[str, threading.RLock] = {}
_registry_lock = threading.Lock()
_asr_lock = threading.Lock()   # one transcription/LLM pass at a time (8 GB VRAM)
_render_pool = ThreadPoolExecutor(max_workers=max(1, settings.render_workers), thread_name_prefix="render")
_job_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="job")


def normalize_options(opts: dict | None) -> dict:
    o = {**DEFAULT_OPTIONS, **{k: v for k, v in (opts or {}).items() if k in DEFAULT_OPTIONS}}
    o["clips"] = max(1, min(20, int(o["clips"] or 6)))
    if o.get("quality") not in QUALITY:
        o["quality"] = "good"
    if o["length"] not in LENGTHS:
        o["length"] = "ai"   # includes the old "auto" setting
    o["provider"] = o["provider"] or settings.llm_provider
    o["whisper_model"] = o["whisper_model"] or settings.whisper_model
    return o


def edit_from_options(o: dict) -> dict:
    return merge_edit(None, {
        "aspect": o["aspect"],
        "layout": o["layout"],
        "captions": {"enabled": bool(o["captions"]), "preset": o["preset"], "emoji": bool(o["emoji"])},
        "title": {"enabled": bool(o["title"])},
        "progress": {"enabled": bool(o["progress"])},
        "zoom": {"enabled": bool(o["zoom"])},
        "cuts": {"silence": bool(o["silence"]), "fillers": bool(o["fillers"]), "removed": []},
    })


# --------------------------------------------------------------------------
# storage
# --------------------------------------------------------------------------
def valid_pid(pid: str) -> bool:
    return bool(PID_RE.match(pid or ""))


def project_dir(pid: str) -> Path:
    if not valid_pid(pid):
        raise ValueError("bad project id")
    return settings.output_dir / pid


def lock(pid: str) -> threading.RLock:
    with _registry_lock:
        return _locks.setdefault(pid, threading.RLock())


def save(p: Project) -> None:
    d = project_dir(p.id)
    d.mkdir(parents=True, exist_ok=True)
    with lock(p.id):
        data = json.dumps(p.to_dict(), ensure_ascii=False,
                          default=lambda o: o.item() if hasattr(o, "item") else str(o))
        tmp = d / "project.json.tmp"
        tmp.write_text(data, encoding="utf-8")
        for _ in range(5):
            try:
                tmp.replace(d / "project.json")
                break
            except PermissionError:  # a reader has it open on Windows
                time.sleep(0.05)


def get(pid: str) -> Project | None:
    """The live in-memory project (shared by the pipeline thread and the API)."""
    if not valid_pid(pid):
        return None
    with _registry_lock:
        if pid in _live:
            return _live[pid]
    path = project_dir(pid) / "project.json"
    if not path.exists():
        return None
    try:
        p = Project.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None
    # nothing is running for a project we had to load from disk -> it was interrupted
    dirty = False
    if p.status in ("running", "queued"):
        p.status, p.stage = "error", "error"
        p.error = p.message = "Interrupted (server restarted) - press Re-run"
        dirty = True
    for c in p.clips:
        if (c.render or {}).get("status") in ("queued", "rendering"):
            c.render = {"status": "error", "error": "Interrupted - render again"}
            dirty = True
    with _registry_lock:
        p = _live.setdefault(pid, p)
    if dirty:
        save(p)
    return p


def forget(pid: str) -> None:
    with _registry_lock:
        _live.pop(pid, None)


def list_projects() -> list[dict]:
    out = []
    if not settings.output_dir.exists():
        return out
    for d in settings.output_dir.iterdir():
        if not d.is_dir() or not valid_pid(d.name):
            continue
        with _registry_lock:
            live = _live.get(d.name)
        if live is not None:
            j = live.to_dict(include_transcript=False)
        else:
            f = d / "project.json"
            if not f.exists():
                continue
            try:
                j = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            if j.get("status") in ("running", "queued"):
                j["status"] = "error"
        clips = j.get("clips", [])
        best = max((c.get("score") or {}).get("total", 0) for c in clips) if clips else 0
        thumb_clip = next((c for c in clips if c.get("thumb")), None)
        out.append({
            "id": j.get("id", d.name),
            "name": j.get("name", d.name),
            "status": j.get("status"),
            "stage": j.get("stage"),
            "progress": j.get("progress", 0),
            "created": j.get("created", 0),
            "duration": j.get("duration", 0),
            "clips": len(clips),
            "best": best,
            "thumb": f"{thumb_clip['thumb']}?v={thumb_clip.get('version', 0)}" if thumb_clip else None,
        })
    out.sort(key=lambda x: x.get("created", 0), reverse=True)
    return out


def new_project(source: Path, name: str | None = None, options: dict | None = None) -> Project:
    pid = uuid.uuid4().hex[:8]
    p = Project(id=pid, source=str(source), name=name or source.stem, status="queued")
    p.options = normalize_options(options)
    with _registry_lock:
        _live[pid] = p
    save(p)
    return p


def source_path(p: Project) -> Path | None:
    return next(iter(project_dir(p.id).glob("source.*")), None)


# --------------------------------------------------------------------------
# link projects: audio first, video only where the clips are
# --------------------------------------------------------------------------
_dl_locks: dict[str, threading.Lock] = {}
_index_cache: dict[str, dict] = {}


def _ingest_remote(p: Project, d: Path, setp, log) -> None:
    url = p.remote["url"]
    setp("download", 0.01, "Reading the link")
    info = remote.probe_url(url)
    p.remote = info
    if p.name == url[:80]:
        p.name = info["title"][:80]
    p.duration = info["duration"]
    p.width, p.height, p.fps = info["video"]["width"], info["video"]["height"], info["video"]["fps"]
    p.has_audio = True
    full_mb = (info["video"]["bytes"] + info["audio"]["bytes"]) / 1e6
    log(f"link: {p.width}x{p.height}, {p.duration:.0f}s - fetching audio only "
        f"({info['audio']['bytes'] / 1e6:.0f} MB instead of {full_mb:.0f} MB)")
    audio = next(iter(d.glob("audio_src.*")), None)
    if audio is None:
        setp("download", 0.02, "Downloading audio only")
        audio = remote.fetch_audio(info, d)
    p.remote["audio_file"] = audio.name
    save(p)
    wav = d / "audio.wav"
    if not wav.exists():
        setp("audio", 0.04, "Extracting audio")
        extract_wav(audio, wav)


def media_for(p: Project, a: float, b: float, log=None) -> tuple[Path, float]:
    """A local file that covers [a, b] and the absolute time of its 0.
    Uploads use the full source; links download just this piece (once) and reuse it."""
    local = source_path(p)
    if local is not None:
        return local, 0.0
    if not p.remote:
        raise RuntimeError("source file missing - re-upload the video")
    d = project_dir(p.id)
    with _registry_lock:
        lk = _dl_locks.setdefault(p.id, threading.Lock())
    with lk:
        for s in p.segments:
            f = d / s["file"]
            if s["t0"] <= a + 0.05 and s["t1"] >= b - 0.05 and f.exists():
                return f, float(s["t0"])
        audio = d / p.remote.get("audio_file", "")
        if not audio.is_file():
            raise RuntimeError("audio for this link is missing - press Re-run")
        pad = 8.0     # extra room so small trims in the editor need no new download
        lo, hi = max(0.0, a - pad), min(p.duration or b + pad, b + pad)
        dst = d / f"seg_{int(lo):05d}_{int(hi):05d}.mp4"
        for attempt in range(2):
            try:
                if attempt or time.time() - p.remote.get("fetched", 0) > remote.URL_TTL:
                    fresh = remote.probe_url(p.remote["url"])
                    p.remote.update({k: fresh[k] for k in ("video", "fetched")})
                    _index_cache.pop(p.id, None)
                idx = _index_cache.get(p.id) or remote.index(p.remote)
                _index_cache[p.id] = idx
                t0, t1 = remote.fetch_range(p.remote, idx, lo, hi, audio, dst)
                break
            except remote.RemoteError:
                if attempt:
                    raise
        seg = {"file": dst.name, "t0": round(t0, 3), "t1": round(t1, 3),
               "bytes": dst.stat().st_size}
        p.segments.append(seg)
        save(p)
        if log:
            log(f"downloaded {t1 - t0:.0f}s of video ({seg['bytes'] / 1e6:.1f} MB) for {a:.0f}-{b:.0f}s")
        return dst, float(t0)


# --------------------------------------------------------------------------
# preview proxy (what the browser editor plays)
# --------------------------------------------------------------------------
def _make_preview(src: Path, meta: dict, dst_dir: Path) -> str | None:
    # Chrome / Edge play these directly - no copy needed (saves disk space and time)
    playable = (meta.get("video_codec") in ("h264", "av1", "vp9")
                and src.suffix.lower() in (".mp4", ".m4v", ".mov", ".webm")
                and meta.get("audio_codec") in (None, "aac", "mp3", "opus")
                and meta.get("height", 0) <= 1440)
    if playable:
        return src.name
    out = dst_dir / "preview.mp4"
    # small on purpose: it is only for the editor preview (~1.5 Mbps at 720p)
    cap = ["-maxrate", "1500k", "-bufsize", "3000k"]
    enc = (["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "30", "-b:v", "0", *cap]
           if "NVENC" in encoder_name()
           else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "30", *cap])
    try:
        run_cmd(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
                 "-vf", "scale=-2:'min(720,ih)'", *enc, "-g", "30", "-pix_fmt", "yuv420p",
                 "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(out)])
        return out.name
    except Exception:
        return None


# --------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------
def run(p: Project, source: Path, log=None) -> Project:
    log = log or (lambda m: None)
    t_start = time.time()
    o = p.options = normalize_options(p.options)
    timings: dict[str, float] = {}

    def setp(stage: str, pct: float, msg: str = "") -> None:
        p.stage = stage
        p.progress = max(0.0, min(1.0, pct))
        if msg:
            p.message = msg
        save(p)

    def mark(name: str, t0: float) -> None:
        timings[name] = round(time.time() - t0, 2)
        p.timings = dict(timings)

    try:
        p.status = "running"
        p.error = None
        setp("probe", 0.01, "Reading video")
        t0 = time.time()
        d = project_dir(p.id)
        d.mkdir(parents=True, exist_ok=True)
        wav = d / "audio.wav"
        preview_box: dict = {}
        prev_th = None

        if p.remote and source_path(p) is None:
            # link: audio only now, video pieces later (only around the chosen clips)
            try:
                _ingest_remote(p, d, setp, log)
            except Exception as e:
                log(f"smart download not possible ({str(e)[:160]}) - downloading the whole video")
                setp("download", 0.02, "Downloading video")
                source = download(p.remote["url"], settings.upload_dir / f"{p.id}.mp4")
                p.remote = {}
        if not p.remote:
            src = Path(source)
            local = d / f"source{src.suffix.lower()}"
            if src.resolve() != local.resolve():
                if src.resolve().parent == settings.upload_dir.resolve():
                    shutil.move(str(src), str(local))
                else:
                    shutil.copy2(src, local)
            source = local
            p.source = str(source)
            meta = probe(source)
            p.duration, p.width, p.height, p.fps = meta["duration"], meta["width"], meta["height"], meta["fps"]
            p.has_audio = bool(meta.get("has_audio"))
            if not p.has_audio:
                raise RuntimeError("this video has no audio track - there is nothing to transcribe")
            log(f"source {p.width}x{p.height} @ {p.fps:.2f}fps, {p.duration:.1f}s, {meta.get('video_codec')}")
            prev_th = threading.Thread(target=lambda: preview_box.setdefault(
                "name", _make_preview(source, meta, d)), daemon=True)
            prev_th.start()
            setp("audio", 0.03, "Extracting audio")
            if not wav.exists():
                extract_wav(source, wav)
        mark("ingest", t0)

        llm = LLM(o["provider"], o.get("llm_model"))
        with _asr_lock:
            t0 = time.time()
            cached = (p.sentences and p.settings.get("whisper_model") == o["whisper_model"]
                      and p.settings.get("language_req", "") == (o["language"] or ""))
            if cached:
                setp("transcribe", 0.45, "Using cached transcript")
            else:
                setp("transcribe", 0.05, "Transcribing (Whisper)")
                if llm.provider == "ollama":
                    freed = unload_ollama()
                    if freed:
                        log(f"freed VRAM held by Ollama ({', '.join(freed)})")

                def asr_progress(frac: float, msg: str) -> None:
                    setp("transcribe", 0.05 + 0.40 * frac, msg)

                asr = transcribe(wav, model_size=o["whisper_model"], device=settings.whisper_device,
                                 compute=settings.whisper_compute, language=o["language"] or None,
                                 progress=asr_progress, log=log)
                p.language = asr["language"]
                p.sentences = build_sentences(asr["segments"])
                p.settings["whisper_model"] = o["whisper_model"]
                p.settings["language_req"] = o["language"] or ""
                save(p)
                log(f"transcript: {len(p.sentences)} sentences, language={p.language}")
            mark("transcribe", t0)
            if not p.sentences:
                raise RuntimeError("no speech found in this video")

            if not p.speakers or not cached:
                t0 = time.time()
                setp("transcribe", 0.46, "Identifying speakers")
                try:
                    p.speakers = diarize(wav, p.sentences, log=log)
                except Exception as e:     # never fatal: captions just won't be speaker-coloured
                    log(f"speaker detection skipped: {e}")
                    p.speakers = {"count": 1, "share": [1.0], "method": "failed"}
                save(p)
                mark("speakers", t0)

            t0 = time.time()
            setp("analyse", 0.48, "Finding highlights")
            audio = AudioFeatures.from_wav(wav)

            def sel_log(m: str) -> None:
                log(m)
                p.message = m
                save(p)

            lo, hi = LENGTHS[o["length"]]
            if p.duration < lo * 1.5:      # short source: allow shorter clips
                lo = max(5.0, min(lo, p.duration * 0.5))
                hi = max(lo + 5, min(hi, p.duration))
            context = (p.remote or {}).get("title") or p.name
            clips, model = select(p.sentences, audio, topic=o["topic"], k=o["clips"], context=context,
                                  weights=settings.scoring_weights(), log=sel_log,
                                  min_len=lo, max_len=hi, llm=llm,
                                  min_score=QUALITY[o["quality"]])
            if llm.provider == "ollama":
                unload_ollama()   # give the GPU back to the renderer
            mark("select", t0)
        if not clips:
            raise RuntimeError(f"no clip candidates found ({p.duration:.0f}s source, {len(p.sentences)} sentences)")

        base_edit = edit_from_options(o)
        for c in clips:
            c.edit = copy.deepcopy(base_edit)
            c.render = {"status": "queued", "progress": 0.0}
        p.clips = clips
        p.settings["selector"] = model
        p.settings["quality"] = {"bar": QUALITY[o["quality"]], "level": o["quality"],
                                 "kept": len(clips), "asked": o["clips"]}
        p.settings["encoder"] = encoder_name()
        save(p)
        log(f"selected {len(clips)} clips via {model}")
        try:   # the 16 kHz copy is only needed for analysis; rebuilt automatically on Re-run
            wav.unlink(missing_ok=True)
        except OSError:
            pass

        # ------------------------------------------------ track + render (parallel)
        t0 = time.time()
        setp("render", 0.60, f"Rendering {len(clips)} clips on {encoder_name()}")
        futures = [_render_pool.submit(_track_and_render, p, c, log) for c in clips]
        wait(futures)
        mark("render", t0)
        if prev_th is not None:
            prev_th.join(timeout=600)
            p.preview = preview_box.get("name")

        ok = len([c for c in p.clips if c.file])
        p.status = "done" if ok else "error"
        p.stage = p.status
        p.progress = 1.0
        timings["total"] = round(time.time() - t_start, 2)
        p.timings = timings
        p.message = f"{ok} clips in {timings['total']:.0f}s"
        if not ok:
            p.error = next((c.render.get("error") for c in p.clips if c.render.get("error")), "render failed")
        save(p)
        log(p.message)
        return p
    except Exception as e:
        p.status = "error"
        p.stage = "error"
        p.error = str(e)[:600]
        p.message = str(e)[:300]
        save(p)
        log("ERROR: " + str(e))
        log(traceback.format_exc(limit=5))
        return p


def _update_render_progress(p: Project) -> None:
    if p.status != "running":
        return
    fr = [float((c.render or {}).get("progress") or (1.0 if c.file else 0.0)) for c in p.clips]
    if fr:
        p.progress = 0.60 + 0.395 * (sum(fr) / len(fr))
        done = sum(1 for c in p.clips if (c.render or {}).get("status") == "done")
        p.message = f"Rendered {done}/{len(p.clips)} clips"


def _track_and_render(p: Project, clip: Clip, log=None) -> None:
    log = log or (lambda m: None)
    try:
        clip.render = {"status": "rendering", "progress": 0.0}
        save(p)
        edit = merge_edit(clip.edit, None)
        from .edit import bounds

        a, b = bounds(clip, edit)
        pad = 2.0
        t0, t1 = max(0.0, a - pad), min(p.duration or b + pad, b + pad)
        if p.remote and source_path(p) is None:
            clip.render["message"] = "downloading this part"
            save(p)
        src, offset = media_for(p, t0, t1, log=log)
        clip.render.pop("message", None)
        layout = edit["layout"]
        if layout in ("auto", "track", "split", "gaming") and not covers(clip.track, a, b):
            clip.render["message"] = "finding speakers"
            save(p)
            words = [w for s in p.sentences if s.end >= t0 and s.start <= t1 for w in s.words]
            clip.track = track(src, t0, t1, p.width, p.height, words=words, log=log, offset=offset)
            clip.render.pop("message", None)
            save(p)

        last = [0.0]

        def prog(frac: float) -> None:
            if frac - last[0] < 0.04 and frac < 1.0:
                return
            last[0] = frac
            clip.render["progress"] = round(frac, 3)
            _update_render_progress(p)
            save(p)

        out = render_clip(src, p.width, p.height, p.has_audio, clip, p.sentences,
                          project_dir(p.id) / "clips", log=log, on_progress=prog, offset=offset)
        clip.file = f"clips/{out['file']}"
        clip.thumb = f"clips/{out['thumb']}" if out.get("thumb") else None
        clip.version = out["version"]
        clip.error = None
        clip.render = {"status": "done", "progress": 1.0, "seconds": out["seconds"],
                       "encoder": out["encoder"], "duration": out["duration"],
                       "layout": out["layout"], "bytes": out["bytes"], "at": time.time()}
        log(f"{clip.id}: rendered {out['duration']:.1f}s in {out['seconds']:.1f}s ({out['encoder']})")
    except Exception as e:
        clip.error = str(e)[:500]
        clip.render = {"status": "error", "error": str(e)[:500]}
        log(f"{clip.id} failed: {e}")
    finally:
        again = clip.render.pop("again", False) if isinstance(clip.render, dict) else False
        _update_render_progress(p)
        save(p)
        if again:
            queue_render(p.id, clip.id)


# --------------------------------------------------------------------------
# API helpers
# --------------------------------------------------------------------------
def start(p: Project, source: Path) -> None:
    _job_pool.submit(run, p, source)


def find_clip(p: Project, cid: str) -> Clip | None:
    return next((c for c in p.clips if c.id == cid), None)


def update_clip(pid: str, cid: str, edit: dict | None = None, title: str | None = None,
                hashtags: list[str] | None = None) -> Clip | None:
    p = get(pid)
    if not p:
        return None
    clip = find_clip(p, cid)
    if not clip:
        return None
    with lock(pid):
        if edit is not None:
            clip.edit = merge_edit(None, edit)
        if title is not None:
            clip.title = title.strip()[:120]
        if hashtags is not None:
            clip.hashtags = [("#" + h.strip().lstrip("#")).replace(" ", "") for h in hashtags if h.strip()][:10]
    save(p)
    return clip


def queue_render(pid: str, cid: str) -> Clip | None:
    p = get(pid)
    if not p:
        return None
    clip = find_clip(p, cid)
    if not clip:
        return None
    with lock(pid):
        st = (clip.render or {}).get("status")
        if st in ("queued", "rendering"):
            clip.render["again"] = True      # render once more with the newest edit
            save(p)
            return clip
        clip.render = {"status": "queued", "progress": 0.0}
    save(p)
    _render_pool.submit(_track_and_render, p, clip)
    return clip


def apply_to_all(pid: str, patch: dict, render: bool = True) -> Project | None:
    """Apply a style (captions/layout/effects) to every clip, keeping each clip's own trims/cuts."""
    p = get(pid)
    if not p:
        return None
    style = {k: v for k, v in patch.items() if k not in ("start", "end", "word_edits")}
    if "cuts" in style:
        style["cuts"] = {k: v for k, v in style["cuts"].items() if k != "removed"}
    if "title" in style:
        style["title"] = {k: v for k, v in style["title"].items() if k != "text"}
    with lock(pid):
        for c in p.clips:
            c.edit = merge_edit(c.edit, style)
    save(p)
    if render:
        for c in p.clips:
            queue_render(pid, c.id)
    return p


def rerun(pid: str, options: dict | None) -> Project | None:
    p = get(pid)
    if not p:
        return None
    src = source_path(p)
    if src is None and not p.remote:
        raise FileNotFoundError("original file missing - re-upload")
    with lock(pid):
        if p.status in ("running", "queued"):
            return p
        p.options = normalize_options({**p.options, **(options or {})})
        p.clips = []
        p.progress = 0.0
        p.status = "queued"
        p.error = None
        p.stage = "queued"
        p.message = "Re-running"
    save(p)
    for f in (project_dir(pid) / "clips").glob("*"):
        try:
            f.unlink()
        except OSError:
            pass
    start(p, src)
    return p


def delete(pid: str) -> bool:
    p = get(pid)
    if p and p.status in ("running", "queued"):
        raise RuntimeError("project is still processing")
    forget(pid)
    d = project_dir(pid)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
        return True
    return False
