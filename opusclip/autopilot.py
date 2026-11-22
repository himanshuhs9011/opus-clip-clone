"""Autopilot: watch channels, clip every new video, post the clips on a daily schedule.

Runs inside the ClipForge server (a background thread), so it works while the app is open:

1. Check: every `check_minutes`, list the newest videos of each source channel (YouTube
   channel / tab, Kick, Twitch, any page yt-dlp can list). Videos that are still live or
   shorter than 3 minutes are skipped; a video is handled once.
2. Clip: a new video becomes a normal ClipForge project (shorts, long videos or both, with
   the source's focus), one at a time.
3. Queue: every finished clip gets one queue entry per platform it suits:
   shorts -> YouTube Shorts, Instagram Reels, Facebook Reels, the phone folder;
   long videos -> YouTube. With `review` on, entries wait for an OK first.
4. Post: at each posting time the best waiting clips go out, within each platform's daily cap.

Only channels the user adds are clipped. Post clips you have the rights to: your own
videos, or channels that allow it (a clipping program). Uploading other people's videos
gets strikes and Content ID claims.

State lives in autopilot.json in the ClipForge data folder (tokens included, like
providers.json); it never leaves this PC except to the platforms themselves.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

from .config import DATA

PATH = Path(DATA) / "autopilot.json"
PHONE_DIR = Path(DATA) / "autopilot_phone"
PLATFORMS = ("youtube", "instagram", "facebook", "phone")
MIN_SOURCE_SECONDS = 180

DEFAULT: dict = {
    "enabled": False,
    "review": False,             # True: every clip waits for an OK before it is posted
    "check_minutes": 60,
    "post_times": ["11:00", "15:00", "19:00"],
    # per day: YouTube allows ~6 API uploads a day on the default quota
    "per_day": {"youtube_short": 3, "youtube_long": 1, "instagram": 3, "facebook": 3, "phone": 3},
    "youtube_privacy": "public",  # public | unlisted | private
    "credit": True,               # add "Source: <channel>" to captions
    "min_score": 70,              # clips below this are not posted
    "sources": [],
    "accounts": {"youtube": {}, "meta": {}},
    "jobs": [],                   # projects Autopilot started: {pid, source, url, title, status}
    "queue": [],                  # {id, pid, cid, kind, platform, status, score, title, ...}
    "posted_days": {},            # "YYYY-MM-DD" -> {platform key: count}
    "slots_done": [],             # "YYYY-MM-DD HH:MM" posting slots already handled
    "last_check": 0,
    "log": [],
}

_lock = threading.RLock()
_state: dict | None = None
_thread: threading.Thread | None = None
_wake = threading.Event()


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------
def load() -> dict:
    global _state
    with _lock:
        if _state is None:
            data = {}
            if PATH.exists():
                try:
                    data = json.loads(PATH.read_text(encoding="utf-8"))
                except Exception:
                    data = {}
            st = copy.deepcopy(DEFAULT)
            for k, v in data.items():
                if isinstance(v, dict) and isinstance(st.get(k), dict):
                    st[k].update(v)
                else:
                    st[k] = v
            _state = st
        return _state


def save() -> None:
    with _lock:
        st = load()
        st["log"] = st["log"][-300:]
        st["queue"] = st["queue"][-2000:]
        st["jobs"] = st["jobs"][-500:]
        st["slots_done"] = st["slots_done"][-200:]
        tmp = PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(PATH)


def log(msg: str) -> None:
    with _lock:
        st = load()
        st["log"].append([round(time.time()), msg[:400]])
    save()


def public() -> dict:
    """State for the UI: tokens and secrets never leave the server."""
    with _lock:
        st = copy.deepcopy(load())
    yt = st["accounts"].get("youtube") or {}
    meta = st["accounts"].get("meta") or {}
    st["accounts"] = {
        "youtube": {"client_id": yt.get("client_id", ""), "has_secret": bool(yt.get("client_secret")),
                    "connected": bool(yt.get("refresh_token")), "channel": yt.get("channel_title", "")},
        "meta": {"app_id": meta.get("app_id", ""), "has_secret": bool(meta.get("app_secret")),
                 "connected": bool(meta.get("token")), "page_id": meta.get("page_id", ""),
                 "page_name": meta.get("page_name", ""), "ig_username": meta.get("ig_username", ""),
                 "pages": [{k: p.get(k, "") for k in ("id", "name", "ig_username")}
                           for p in meta.get("pages") or []]},
    }
    today = dt.date.today().isoformat()
    st["today"] = st["posted_days"].get(today, {})
    st["phone_dir"] = str(PHONE_DIR)
    st["next_post"] = next_slot_label(st)
    st["running"] = bool(_thread and _thread.is_alive())
    st["log"] = st["log"][-80:]
    return st


def update(values: dict) -> dict:
    with _lock:
        st = load()
        for k in ("enabled", "review", "credit"):
            if k in values:
                st[k] = bool(values[k])
        if "check_minutes" in values:
            st["check_minutes"] = int(max(15, min(1440, int(values["check_minutes"]))))
        if "min_score" in values:
            st["min_score"] = int(max(0, min(100, int(values["min_score"]))))
        if values.get("youtube_privacy") in ("public", "unlisted", "private"):
            st["youtube_privacy"] = values["youtube_privacy"]
        if isinstance(values.get("post_times"), list):
            times = []
            for t in values["post_times"]:
                try:
                    h, m = str(t).strip().split(":")
                    times.append(f"{int(h):02d}:{int(m):02d}")
                except ValueError:
                    continue
            st["post_times"] = sorted(set(x for x in times if x <= "23:59"))[:12] or st["post_times"]
        if isinstance(values.get("per_day"), dict):
            for k, v in values["per_day"].items():
                if k in st["per_day"]:
                    st["per_day"][k] = int(max(0, min(50, int(v))))
    save()
    if values.get("enabled"):
        start()
        _wake.set()
    return public()


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------
def _tabs(url: str) -> list[str]:
    """A bare YouTube channel lists its tabs, not videos: look at uploads and past streams."""
    u = url.rstrip("/")
    if "youtube.com/" in u and any(x in u for x in ("/@", "/channel/", "/c/", "/user/")) \
            and not u.endswith(("/videos", "/streams", "/shorts", "/live")):
        return [u + "/videos", u + "/streams"]
    return [u]


def list_videos(url: str, limit: int = 6) -> list[dict]:
    """Newest finished videos of a channel page: [{id, url, title, duration}] newest first."""
    from .probe import run
    from .remote import _ytdlp_base

    out: list[dict] = []
    for tab in _tabs(url):
        cmd = [c for c in _ytdlp_base() if c != "--no-playlist"]
        try:
            raw = run(cmd + ["--flat-playlist", "--playlist-items", f"1:{limit}", "-J", tab], timeout=180)
            data = json.loads(raw)
        except Exception:
            continue
        entries = data.get("entries") if isinstance(data, dict) else None
        if entries is None and isinstance(data, dict) and data.get("id"):
            entries = [data]
        for pos, e in enumerate(entries or []):
            if not isinstance(e, dict) or not e.get("id"):
                continue
            if e.get("live_status") in ("is_live", "is_upcoming", "post_live") or e.get("is_live"):
                continue          # still streaming / not processed yet: next check
            dur = float(e.get("duration") or 0)
            if dur and dur < MIN_SOURCE_SECONDS:
                continue          # a short of their own: nothing to cut
            link = e.get("webpage_url") or e.get("url") or ""
            if link and not link.startswith("http"):
                link = f"https://www.youtube.com/watch?v={e['id']}"
            chan = (data.get("channel") or data.get("uploader") or e.get("channel") or e.get("uploader") or ""
                    ) if isinstance(data, dict) else ""
            out.append({"id": str(e["id"]), "url": link, "title": str(e.get("title") or "")[:120],
                        "duration": dur, "pos": pos, "channel": str(chan)[:60]})   # pos 0 = newest
    seen, uniq = set(), []
    for v in out:
        if v["id"] not in seen:
            seen.add(v["id"])
            uniq.append(v)
    return uniq


def add_source(url: str, fmt: str = "both", focus: str = "", backfill: int = 1,
               name: str = "") -> dict:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        raise ValueError("a channel link must start with https://")
    vids = list_videos(url)
    if not vids:
        raise ValueError("no finished videos found on that page - paste a channel or its Videos / "
                         "Live tab link")
    backfill = max(0, min(5, int(backfill)))
    src = {"id": uuid.uuid4().hex[:8], "url": url,
           "name": name.strip()[:60] or vids[0].get("channel") or vids[0]["title"][:40],
           "format": fmt if fmt in ("short", "long", "both") else "both", "focus": focus.strip()[:300],
           # older ones are history, not news: only the newest `backfill` get clipped
           "seen": [v["id"] for v in sorted(vids, key=lambda x: x.get("pos", 0))[backfill:]],
           "added": round(time.time()), "enabled": True}
    with _lock:
        load()["sources"].append(src)
    save()
    log(f"watching {src['name']} ({url}); {backfill} latest video(s) will be clipped")
    _wake.set()
    return src


def remove_source(sid: str) -> None:
    with _lock:
        st = load()
        st["sources"] = [s for s in st["sources"] if s["id"] != sid]
    save()


def check_sources() -> int:
    """Start a project for the next new video (one at a time). Returns how many started."""
    from . import pipeline

    with _lock:
        st = load()
        st["last_check"] = round(time.time())
        busy = any(j["status"] in ("running", "queued") for j in st["jobs"])
        sources = [s for s in st["sources"] if s.get("enabled", True)]
    save()
    if busy:
        return 0
    for src in sources:
        try:
            vids = list_videos(src["url"])
        except Exception as e:
            log(f"{src['name']}: could not list videos ({str(e)[:120]})")
            continue
        new = [v for v in vids if v["id"] not in set(src.get("seen") or [])]
        if not new:
            continue
        v = max(new, key=lambda x: x.get("pos", 0))   # oldest unseen first, so nothing is skipped
        opts = {"format": src.get("format", "both"), "focus": src.get("focus", ""), "clips": 0,
                "pick": False, "quality": "good"}
        p = pipeline.new_project(Path("remote.mp4"), name=v["title"][:80] or v["url"][:80], options=opts)
        p.remote = {"url": v["url"]}
        p.stage, p.message = "download", "Autopilot: reading the link"
        pipeline.save(p)
        pipeline.start(p, None)
        with _lock:
            src["seen"] = (list(src.get("seen") or []) + [v["id"]])[-300:]
            load()["jobs"].append({"pid": p.id, "source": src["id"], "source_name": src["name"],
                                   "url": v["url"], "title": v["title"], "status": "running",
                                   "started": round(time.time())})
        save()
        log(f"new video from {src['name']}: {v['title'][:80]} -> project {p.id}")
        return 1
    return 0


# --------------------------------------------------------------------------
# queue
# --------------------------------------------------------------------------
def _accounts_ready() -> dict:
    acc = load()["accounts"]
    yt, meta = acc.get("youtube") or {}, acc.get("meta") or {}
    return {"youtube": bool(yt.get("refresh_token")),
            "instagram": bool(meta.get("token") and meta.get("ig_id")),
            "facebook": bool(meta.get("token") and meta.get("page_id")),
            "phone": True}


def follow_jobs() -> None:
    """Finished Autopilot projects: queue their clips."""
    from . import pipeline

    with _lock:
        jobs = [j for j in load()["jobs"] if j["status"] in ("running", "queued")]
    for j in jobs:
        p = pipeline.get(j["pid"])
        if p is None:
            j["status"] = "error"
            continue
        if p.status in ("running", "queued"):
            continue
        j["status"] = p.status
        if p.status != "done":
            log(f"project {p.id} ({j['title'][:60]}) failed: {(p.error or '')[:160]}")
            continue
        enqueue(p, j)
    save()


def enqueue(p, job: dict) -> int:
    ready = _accounts_ready()
    st = load()
    n = 0
    for c in p.clips:
        if not c.file or (c.score and c.score.total < st["min_score"]):
            continue
        targets = ["youtube"] if c.kind == "long" else ["youtube", "instagram", "facebook", "phone"]
        for plat in targets:
            if not ready.get(plat):
                continue
            st["queue"].append({
                "id": uuid.uuid4().hex[:10], "pid": p.id, "cid": c.id, "kind": c.kind, "platform": plat,
                "status": "review" if st["review"] else "waiting", "score": c.score.total if c.score else 0,
                "title": (c.package or {}).get("yt_title") or c.title, "source": job.get("source_name", ""),
                "source_url": job.get("url", ""), "added": round(time.time()), "url": "", "error": ""})
            n += 1
    log(f"project {p.id}: {n} posts queued from {len([c for c in p.clips if c.file])} clips")
    return n


def set_item(item_id: str, action: str) -> dict | None:
    with _lock:
        for it in load()["queue"]:
            if it["id"] == item_id:
                if action == "approve" and it["status"] == "review":
                    it["status"] = "waiting"
                elif action == "skip" and it["status"] in ("review", "waiting", "failed"):
                    it["status"] = "skipped"
                elif action == "retry" and it["status"] == "failed":
                    it["status"], it["error"] = "waiting", ""
                save()
                return it
    return None


# --------------------------------------------------------------------------
# posting
# --------------------------------------------------------------------------
def caption_for(c, platform: str, credit: str, source_url: str) -> dict:
    """Title / text / tags for one platform from the clip's ready-to-post package."""
    pk = c.package or {}
    title = (pk.get("yt_title") or c.title or "Highlight").strip()
    desc = (pk.get("description") or c.summary or "").strip()
    tags = [t if t.startswith("#") else f"#{t}" for t in (c.hashtags or [])][:8]
    chapters = ""
    if c.kind == "long" and len(pk.get("chapters") or []) >= 3:
        chapters = "\n".join(f"{int(t // 60)}:{int(t % 60):02d} {label}" for t, label in pk["chapters"])
    credit_line = f"Source: {credit} {source_url}".strip() if credit else ""
    if platform == "youtube":
        if c.kind != "long" and "#shorts" not in " ".join(tags).lower():
            tags = ["#Shorts"] + tags
        body = "\n\n".join(x for x in (desc, chapters, credit_line, " ".join(tags)) if x)
        return {"title": title[:100], "text": body, "tags": tags}
    body = "\n\n".join(x for x in (title, desc, credit_line, " ".join(tags[:10])) if x)
    return {"title": title, "text": body[:2200], "tags": tags}


def post_item(it: dict) -> None:
    """Post one queue entry now (raises on failure)."""
    from . import pipeline, publish

    p = pipeline.get(it["pid"])
    c = pipeline.find_clip(p, it["cid"]) if p else None
    if not c or not c.file:
        raise publish.PublishError("the clip file is gone")
    file = pipeline.project_dir(p.id) / c.file
    st = load()
    cap = caption_for(c, it["platform"], it.get("source", "") if st["credit"] else "",
                      it.get("source_url", "") if st["credit"] else "")
    acc = st["accounts"]
    if it["platform"] == "youtube":
        res = publish.youtube_upload(acc["youtube"], file, cap["title"], cap["text"], cap["tags"],
                                     privacy=st["youtube_privacy"],
                                     category="20" if p.settings.get("content") == "gaming" else "22")
        if c.kind != "long":
            res["url"] = f"https://youtube.com/shorts/{res['id']}"
    elif it["platform"] == "instagram":
        res = publish.instagram_reel(acc["meta"], file, cap["text"])
    elif it["platform"] == "facebook":
        res = publish.facebook_reel(acc["meta"], file, cap["text"])
    else:   # phone folder: Snapchat Spotlight, Moj, Josh
        day = PHONE_DIR / dt.date.today().isoformat()
        day.mkdir(parents=True, exist_ok=True)
        stem = f"{int(time.time())}_{c.id}"
        shutil.copy2(file, day / f"{stem}.mp4")
        (day / f"{stem}.txt").write_text(cap["text"], encoding="utf-8")
        res = {"id": stem, "url": str(day / f"{stem}.mp4")}
    it.update(status="posted", url=res.get("url", ""), posted=round(time.time()), error="")


def _key(it: dict) -> str:
    if it["platform"] == "youtube":
        return "youtube_long" if it["kind"] == "long" else "youtube_short"
    return it["platform"]


def post_round(force: bool = False) -> int:
    """One posting slot: for each platform, post its share of today's cap."""
    st = load()
    today = dt.date.today().isoformat()
    done = st["posted_days"].setdefault(today, {})
    slots = max(1, len(st["post_times"]))
    sent = 0
    waiting = sorted([it for it in st["queue"] if it["status"] == "waiting"],
                     key=lambda it: (-it["added"] // 86400, -it["score"]))
    for key, cap in st["per_day"].items():
        share = cap if force else max(1, -(-cap // slots))          # ceil(cap / slots)
        room = min(share, cap - done.get(key, 0))
        for it in [x for x in waiting if _key(x) == key][: max(0, room)]:
            try:
                post_item(it)
                done[key] = done.get(key, 0) + 1
                sent += 1
                log(f"posted to {it['platform']}: {it['title'][:70]} {it['url']}")
            except Exception as e:
                it["status"], it["error"] = "failed", str(e)[:300]
                log(f"{it['platform']} post failed: {str(e)[:200]}")
            save()
    return sent


def next_slot_label(st: dict) -> str:
    now = dt.datetime.now()
    for t in st["post_times"]:
        if f"{now.date().isoformat()} {t}" not in st["slots_done"] and t >= now.strftime("%H:%M"):
            return f"today {t}"
    return f"tomorrow {st['post_times'][0]}" if st["post_times"] else ""


def due_slot(st: dict) -> str | None:
    """The posting time that has come and not been handled yet today (within 3 hours)."""
    now = dt.datetime.now()
    for t in st["post_times"]:
        h, m = (int(x) for x in t.split(":"))
        at = now.replace(hour=h, minute=m, second=0, microsecond=0)
        key = f"{now.date().isoformat()} {t}"
        if at <= now <= at + dt.timedelta(hours=3) and key not in st["slots_done"]:
            return key
    return None


# --------------------------------------------------------------------------
# loop
# --------------------------------------------------------------------------
def tick() -> None:
    st = load()
    if not st["enabled"]:
        return
    follow_jobs()
    if time.time() - st.get("last_check", 0) >= st["check_minutes"] * 60:
        check_sources()
    slot = due_slot(st)
    if slot:
        with _lock:
            st["slots_done"].append(slot)
        save()
        n = post_round()
        log(f"posting slot {slot.split()[1]}: {n} posted")


def _loop() -> None:
    while True:
        try:
            tick()
        except Exception:
            log("autopilot error: " + traceback.format_exc(limit=3)[-300:])
        _wake.wait(60)
        _wake.clear()


def start() -> None:
    """Start the background loop once (the server calls this at startup)."""
    global _thread
    with _lock:
        if _thread and _thread.is_alive():
            return
        _thread = threading.Thread(target=_loop, name="autopilot", daemon=True)
        _thread.start()
