"""Smart ingest for links: never download the whole video.

1. Audio only (about 1 MB per minute) -> transcript, speakers, clip selection.
2. For each chosen clip, only the video bytes around it. YouTube's DASH streams start with a
   segment index (`sidx`) that maps time to byte offsets, so we read the index, fetch just the
   fragments we need with HTTP range requests, and mux them with the matching slice of the
   audio we already have.

A 23-minute 1080p video is about 515 MB in full; with six clips this fetches roughly 22 MB of
audio plus ~10-15 MB of video per clip. Sites without a usable index fall back to a normal
download.
"""
from __future__ import annotations

import json
import shutil
import struct
import time
from pathlib import Path

import httpx

from .probe import probe, run

VIDEO_FMT = ("bv*[height<=1080][vcodec^=avc1][protocol=https]/"
             "bv*[height<=1080][protocol=https]")
AUDIO_FMT = "ba[ext=m4a][protocol=https]/ba[protocol=https]/ba"
CHUNK = 8 * 1024 * 1024        # stay under YouTube's per-request throttle
URL_TTL = 3 * 3600             # signed stream URLs expire after ~6 h


class RemoteError(RuntimeError):
    pass


def _ytdlp_base() -> list[str]:
    cmd = ["yt-dlp", "--no-playlist", "--no-warnings"]
    if shutil.which("deno") is None and shutil.which("node"):
        cmd += ["--js-runtimes", "node"]   # YouTube needs a JS runtime for its stream URLs
    return cmd


def probe_url(url: str) -> dict:
    """Title, duration and the direct stream URLs, without downloading anything."""
    out = run(_ytdlp_base() + ["-J", "-f", f"({VIDEO_FMT})+({AUDIO_FMT})", url], timeout=180)
    d = json.loads(out)
    req = d.get("requested_formats") or []
    v = next((f for f in req if f.get("vcodec") not in (None, "none")), None)
    a = next((f for f in req if f.get("acodec") not in (None, "none") and f.get("vcodec") in (None, "none")), None)
    if not v or not a:
        raise RemoteError("no separate video/audio streams for this link")
    return {
        "url": url,
        "id": d.get("id"),
        "title": d.get("title") or url,
        # who made it: titles and banners should name the streamer, not guess from speech
        "channel": d.get("channel") or d.get("uploader") or "",
        "description": (d.get("description") or "")[:600],
        "duration": float(d.get("duration") or 0),
        "fetched": time.time(),
        "video": {"url": v["url"], "format": v.get("format_id"), "width": v.get("width") or 0,
                  "height": v.get("height") or 0, "fps": v.get("fps") or 30,
                  "codec": v.get("vcodec"), "bytes": v.get("filesize") or v.get("filesize_approx") or 0,
                  "headers": v.get("http_headers") or {}},
        "audio": {"format": a.get("format_id"),
                  "bytes": a.get("filesize") or a.get("filesize_approx") or 0},
    }


def fetch_audio(info: dict, dst_dir: Path) -> Path:
    """Audio-only download (yt-dlp's chunked downloader is fast)."""
    template = str(dst_dir / "audio_src.%(ext)s")
    run(_ytdlp_base() + ["-f", info["audio"]["format"] or AUDIO_FMT, "--no-part", "-o", template,
                         info["url"]], timeout=3600)
    files = sorted(dst_dir.glob("audio_src.*"), key=lambda p: p.stat().st_size, reverse=True)
    if not files:
        raise RemoteError("audio download produced no file")
    return files[0]


# --------------------------------------------------------------------------
# MP4 box parsing (just enough for ftyp/moov/sidx)
# --------------------------------------------------------------------------
def _boxes(buf: bytes, start: int = 0):
    i = start
    while i + 8 <= len(buf):
        size, kind = struct.unpack(">I4s", buf[i : i + 8])
        hdr = 8
        if size == 1:
            if i + 16 > len(buf):
                return
            size = struct.unpack(">Q", buf[i + 8 : i + 16])[0]
            hdr = 16
        elif size == 0:
            size = len(buf) - i
        yield kind.decode("latin-1"), i, size, hdr
        if size < 8:
            return
        i += size


def _parse_sidx(buf: bytes, off: int, hdr: int) -> dict:
    p = off + hdr
    version = buf[p]
    p += 4  # version + flags
    _ref_id, timescale = struct.unpack(">II", buf[p : p + 8])
    p += 8
    if version == 0:
        ept, first = struct.unpack(">II", buf[p : p + 8])
        p += 8
    else:
        ept, first = struct.unpack(">QQ", buf[p : p + 16])
        p += 16
    p += 2  # reserved
    count = struct.unpack(">H", buf[p : p + 2])[0]
    p += 2
    refs = []
    for _ in range(count):
        a, dur, _sap = struct.unpack(">III", buf[p : p + 12])
        refs.append((a & 0x7FFFFFFF, dur))
        p += 12
    return {"timescale": timescale, "ept": ept, "first": first, "refs": refs}


class _Http:
    def __init__(self, url: str, headers: dict) -> None:
        self.url = url
        self.c = httpx.Client(timeout=60, follow_redirects=True,
                              headers={k: v for k, v in headers.items() if k.lower() != "accept-encoding"})

    def get(self, a: int, b: int) -> bytes:
        """Bytes [a, b] inclusive, in throttle-safe chunks."""
        out = bytearray()
        pos = a
        while pos <= b:
            end = min(b, pos + CHUNK - 1)
            for attempt in range(3):
                r = self.c.get(self.url, headers={"Range": f"bytes={pos}-{end}"})
                if r.status_code in (200, 206):
                    break
                if r.status_code == 403:
                    raise RemoteError("stream URL expired (403)")
                time.sleep(1 + attempt)
            else:
                raise RemoteError(f"range request failed: HTTP {r.status_code}")
            out += r.content
            pos = end + 1
        return bytes(out)

    def close(self) -> None:
        self.c.close()


def index(info: dict) -> dict:
    """Init segment + fragment table (start time, byte range) of the video stream."""
    v = info["video"]
    http = _Http(v["url"], v.get("headers") or {})
    try:
        head = http.get(0, 256 * 1024 - 1)
        moov_end = sidx = None
        for kind, off, size, hdr in _boxes(head):
            if kind == "moov":
                moov_end = off + size
            elif kind == "sidx":
                if off + size > len(head):
                    head += http.get(len(head), off + size - 1)
                sidx = (off, size, _parse_sidx(head, off, hdr))
                break
        if moov_end is None or sidx is None:
            raise RemoteError("stream has no segment index")
        off, size, sx = sidx
        pos = off + size + sx["first"]
        t = sx["ept"]
        frags = []
        for nbytes, dur in sx["refs"]:
            frags.append((t / sx["timescale"], (t + dur) / sx["timescale"], pos, pos + nbytes - 1))
            pos += nbytes
            t += dur
        return {"init": head[:moov_end], "frags": frags}
    finally:
        http.close()


def fetch_range(info: dict, idx: dict, a: float, b: float, audio_file: Path, dst: Path) -> tuple[float, float]:
    """Download only the fragments covering [a, b] and mux them with the local audio.
    Returns (t0, t1): the absolute source times that dst's 0 and end correspond to."""
    frags = idx["frags"]
    first = max([i for i, f in enumerate(frags) if f[0] <= a] or [0])
    last = min([i for i, f in enumerate(frags) if f[1] >= b] or [len(frags) - 1])
    t0, t1 = frags[first][0], frags[last][1]
    v = info["video"]
    http = _Http(v["url"], v.get("headers") or {})
    try:
        body = http.get(frags[first][2], frags[last][3])
    finally:
        http.close()
    raw = dst.with_suffix(".frag.mp4")
    raw.write_bytes(idx["init"] + body)
    try:
        run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(raw), "-ss", f"{t0:.3f}", "-t", f"{t1 - t0:.3f}", "-i", str(audio_file),
             "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
             "-movflags", "+faststart", str(dst)])
    finally:
        raw.unlink(missing_ok=True)
    meta = probe(dst)
    return t0, t0 + float(meta["duration"] or (t1 - t0))
