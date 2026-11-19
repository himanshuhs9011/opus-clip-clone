from __future__ import annotations

import json
import subprocess
from pathlib import Path


class FFmpegError(RuntimeError):
    pass


# Windows: give every helper process (ffmpeg, ffprobe, yt-dlp, ollama) a hidden console.
# Without this a console window flashes up for each call when the server has no console,
# and a detached process's own children each open a blank window.
NO_WINDOW: dict = {"creationflags": subprocess.CREATE_NO_WINDOW} if hasattr(subprocess, "CREATE_NO_WINDOW") else {}


def run(cmd: list[str], cwd: Path | None = None, timeout: int | None = None) -> str:
    p = subprocess.run(
        cmd,
        **NO_WINDOW,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if p.returncode != 0:
        tail = (p.stderr or "").strip().splitlines()[-25:]
        raise FFmpegError("command failed: " + " ".join(cmd[:6]) + "\n" + "\n".join(tail))
    return p.stdout


def probe(path: str | Path) -> dict:
    """Return normalised video metadata via ffprobe."""
    out = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )
    data = json.loads(out or "{}")
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not v:
        raise FFmpegError(f"no video stream in {path}")

    def _f(x, default=0.0):
        try:
            return float(x)
        except (TypeError, ValueError):
            return default

    num, _, den = (v.get("avg_frame_rate") or "0/1").partition("/")
    fps = _f(num) / (_f(den) or 1.0)
    if fps <= 0:
        num, _, den = (v.get("r_frame_rate") or "0/1").partition("/")
        fps = _f(num) / (_f(den) or 1.0)

    duration = _f(data.get("format", {}).get("duration")) or _f(v.get("duration"))
    if duration <= 0 and a:
        duration = _f(a.get("duration"))

    rotation = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(_f(sd["rotation"]))
    if not rotation:
        rotation = int(_f(v.get("tags", {}).get("rotate", 0)))

    width, height = int(v.get("width") or 0), int(v.get("height") or 0)
    if abs(rotation) % 180 == 90:
        width, height = height, width

    return {
        "duration": duration,
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "has_audio": a is not None,
        "video_codec": v.get("codec_name"),
        "audio_codec": (a or {}).get("codec_name"),
    }
