"""One-pass ffmpeg render: crop/track -> jump cuts -> auto-zoom -> colour -> captions -> emoji.

Speed: NVENC (GPU) encode when available, CUDA decode, a single filtergraph per clip and
several clips in parallel (see pipeline.py). Falls back to x264 / CPU decode automatically.
"""
from __future__ import annotations

import functools
import re
import subprocess
import threading
import time
from pathlib import Path

from .captions import build_ass, get_preset, output_scale, speaker_palette
from .config import settings
from .edit import (ASPECTS, bounds, emoji_events, keep_segments, merge_edit, output_words,
                   remap_clamped, total_length, words_in, zoom_events)
from .models import Clip, Sentence
from .probe import NO_WINDOW, probe, run
from .reframe import blank_for, crop_box, expr, path_for, split_crop, split_for, stream_geometry


class RenderError(RuntimeError):
    pass


COLOR_FILTERS = {
    "none": "eq=contrast=1.03:saturation=1.05",
    "vivid": "eq=contrast=1.08:saturation=1.35",
    "warm": "colorbalance=rs=0.08:gs=0.02:bs=-0.08:rm=0.05:bm=-0.05,eq=saturation=1.1",
    "cool": "colorbalance=rs=-0.06:bs=0.08:rm=-0.04:bm=0.06",
    "cinematic": "eq=contrast=1.12:saturation=0.85,colorbalance=rs=0.04:bs=-0.03:rh=-0.03:bh=0.05",
    "bw": "hue=s=0,eq=contrast=1.12",
    "vintage": "curves=preset=vintage",
}

_flags = {"hwaccel": True}
_flags_lock = threading.Lock()


# --------------------------------------------------------------------------
# encoder detection
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _nvenc_ok() -> bool:
    if settings.encoder == "x264":
        return False
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
             "color=black:s=256x256:d=0.2", "-c:v", "h264_nvenc", "-f", "null", "-"],
            capture_output=True, timeout=30, **NO_WINDOW,
        )
        return r.returncode == 0
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def _graph_option() -> str:
    """How to pass a filtergraph file: `-/filter_complex` (FFmpeg 7+; 9 removed the old name)
    or `-filter_complex_script` (FFmpeg 6 and older)."""
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-version"], capture_output=True, text=True,
                           timeout=15, **NO_WINDOW)
        m = re.search(r"ffmpeg version (?:n)?(\d+)", r.stdout or "")
        if m and int(m.group(1)) >= 7:
            return "-/filter_complex"
        if m:
            return "-filter_complex_script"
    except Exception:
        pass
    return "-/filter_complex"   # git builds report a date, not a number: they are new


def encoder_name() -> str:
    return "NVENC (GPU)" if _nvenc_ok() else "x264 (CPU)"


def _video_args(gpu: bool) -> list[str]:
    if gpu:
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr",
                "-cq", str(max(15, settings.crf + 1)), "-b:v", "0", "-maxrate", "25M",
                "-bufsize", "50M", "-profile:v", "high", "-pix_fmt", "yuv420p", "-g", "60"]
    return ["-c:v", "libx264", "-preset", settings.preset, "-crf", str(settings.crf),
            "-profile:v", "high", "-pix_fmt", "yuv420p"]


# --------------------------------------------------------------------------
# emoji -> PNG (libass can't draw colour emoji)
# --------------------------------------------------------------------------
_emoji_lock = threading.Lock()


def emoji_png(ch: str, size: int) -> Path | None:
    d = settings.asset_dir / "emoji"
    d.mkdir(parents=True, exist_ok=True)
    name = "-".join(f"{ord(c):x}" for c in ch) + f"_{size}.png"
    path = d / name
    if path.exists():
        return path
    with _emoji_lock:
        if path.exists():
            return path
        try:
            from PIL import Image, ImageDraw, ImageFont

            font_path = next((p for p in (
                Path("C:/Windows/Fonts/seguiemj.ttf"),
                Path("/System/Library/Fonts/Apple Color Emoji.ttc"),
                Path("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"),
            ) if p.exists()), None)
            if font_path is None:
                return None
            big = 160
            font = ImageFont.truetype(str(font_path), int(big * 0.8))
            im = Image.new("RGBA", (big, big), (0, 0, 0, 0))
            ImageDraw.Draw(im).text((big / 2, big / 2), ch, font=font, embedded_color=True, anchor="mm")
            bbox = im.getbbox()
            if not bbox:
                return None
            im = im.crop(bbox)
            scale = size / max(im.size)
            im = im.resize((max(2, int(im.width * scale)) // 2 * 2, max(2, int(im.height * scale)) // 2 * 2),
                           Image.LANCZOS)
            im.save(path)
            return path
        except Exception:
            return None


# --------------------------------------------------------------------------
def plan(clip: Clip, sentences: list[Sentence], src_w: int, src_h: int) -> dict:
    """Everything the render (and the preview) needs, computed from the clip's edit."""
    edit = merge_edit(clip.edit, None)
    a, b = bounds(clip, edit)
    words = words_in(sentences, a, b)
    segs = keep_segments(words, a, b, edit["cuts"])
    dur = total_length(segs)
    owords = output_words(words, a, segs, edit)
    ow, oh = ASPECTS[edit["aspect"]]
    trk = clip.track or {}
    voices = {w["sp"] for w in owords if w.get("sp", -1) >= 0}
    split = split_for(trk, a, b) if oh > ow * 1.1 else []
    layout = edit["layout"]
    scfg = edit.get("stream") or {}
    vertical = oh > ow * 1.1
    if layout == "auto" and trk.get("stream_cam") and vertical and scfg.get("auto", True):
        layout = "gaming"             # a webcam overlay on gameplay for most of the clip
    if layout == "auto":
        cov = trk.get("coverage")
        if cov is not None and cov < 0.35:
            layout = "fit"
        elif split and len(voices) >= 2:
            layout = "split"          # two people on one camera, both talking -> show both
        else:
            layout = "track"          # follow whoever is speaking
    if layout == "split" and not split:
        layout = "track"
    if layout != "split":
        split = []
    stream = None            # gaming layout geometry
    stream_ranges: list[list[float]] = []
    if layout == "gaming":
        cam = trk.get("stream_cam") or trk.get("main_face")
        if cam and vertical:
            stream = stream_geometry(src_w, src_h, ow, oh, cam, scfg)
            stream_ranges = [[0.0, round(b - a, 3)]]
        else:
            layout = "fit"   # no face-cam found (or not a tall format): show the whole frame
    keys = path_for(trk, a, b) if layout in ("track", "split") else None
    # Auto: scenes with only a small face-cam (gameplay, streams) get the gaming layout;
    # scenes with nobody on screen (screen share, replays) are shown whole over a blur
    fit: list[list[float]] = []
    if edit["layout"] == "auto" and layout in ("track", "split"):
        for item in blank_for(trk, a, b):
            if item.get("cam") and scfg.get("auto", True) and vertical:
                stream_ranges.append([item["t0"], item["t1"]])
                stream = stream or stream_geometry(src_w, src_h, ow, oh, item["cam"], scfg)
            else:
                fit.append([item["t0"], item["t1"]])
    pos = float(edit["captions"].get("position", 0.7))
    split_cover = sum(sc["t1"] - sc["t0"] for sc in split) / max(0.1, b - a)
    stream_cover = sum(e - s for s, e in stream_ranges) / max(0.1, b - a)
    seam = None                   # where the two panels meet (fraction of height)
    if split_cover > 0.5:
        seam = 0.5
    elif stream and stream_cover > 0.5:
        seam = round(stream["cam_h"] / oh if stream["cam_pos"] == "top" else 1 - stream["cam_h"] / oh, 3)

    # title: a context banner on the seam (gaming / split), or a short hook at the top
    tcfg = edit.get("title") or {}
    mode = tcfg.get("mode") or "auto"
    if mode == "auto":
        mode = "banner" if seam is not None else "hook"
    banner_text = (tcfg.get("text") or clip.banner or clip.title or "").strip()
    title = {"mode": mode, "enabled": bool(tcfg.get("enabled")) and bool(banner_text),
             "text": banner_text if mode == "banner" else (tcfg.get("text") or clip.title or "").strip(),
             "y": seam if (mode == "banner" and seam is not None) else (0.16 if mode == "banner" else None),
             "seconds": float(tcfg.get("seconds") or 3.5)}

    if abs(pos - 0.70) < 1e-6 and seam is not None:   # default caption position
        if title["enabled"] and mode == "banner":
            # the banner owns the seam: captions go into the gameplay / lower panel
            lower_is_bottom = not (stream and stream_cover > 0.5 and stream["cam_pos"] == "bottom")
            edit["captions"]["position"] = 0.74 if lower_is_bottom else 0.26
        else:
            edit["captions"]["position"] = seam
    return {"edit": edit, "a": a, "b": b, "segs": segs, "duration": dur, "words": owords,
            "w": ow, "h": oh, "layout": layout, "keys": keys, "split": split, "fit": fit,
            "stream": stream, "stream_ranges": stream_ranges, "voices": sorted(voices),
            "title": title}


def _split_size(pl: dict, src_w: int, src_h: int) -> tuple[int, int]:
    fh = max((sc[h]["h"] for sc in pl["split"] for h in ("top", "bottom")), default=0.25)
    return split_crop(src_w, src_h, pl["w"], pl["h"] // 2, fh)


def preview_plan(clip: Clip, sentences: list[Sentence], src_w: int, src_h: int,
                 edit: dict | None = None) -> dict:
    """The render plan in a browser-friendly shape - the editor preview draws from this."""
    from .captions import cue_timing, group_words

    c = Clip.from_dict({**clip.to_dict(), "edit": edit if edit is not None else clip.edit})
    pl = plan(c, sentences, src_w, src_h)
    e = pl["edit"]
    cap = e["captions"]
    st = get_preset(cap.get("preset"))
    per_cue = int(cap.get("words") or st["words"])
    cues = group_words(pl["words"], per_cue)
    timing = cue_timing(cues, pl["duration"])
    cue_out = [[{**w, "as": round(s, 3), "ae": round(t, 3)} for w, (s, t) in zip(cue, row)]
               for cue, row in zip(cues, timing)]
    zk: list[list[float]] = [[0.0, 1.0]]
    zcfg = e.get("zoom") or {}
    if pl["layout"] not in ("fit", "gaming") and zcfg.get("enabled") and float(zcfg.get("strength") or 0) > 0:
        for t, lvl in zoom_events(pl["words"], pl["duration"], float(zcfg["strength"]))[1:]:
            zk.append([t, zk[-1][1]])
            zk.append([round(t + 0.22, 3), lvl])
    cw, ch, axis = crop_box(src_w, src_h, pl["w"], pl["h"])
    emo = emoji_events(pl["words"]) if cap.get("enabled") and cap.get("emoji") else []
    voices = pl["voices"]
    palette = (speaker_palette(st, max(voices) + 1, cap.get("speakers"))
               if voices and len(voices) >= 2 and cap.get("speaker_colors", True) else None)
    sw, sh = _split_size(pl, src_w, src_h) if pl["split"] else (0, 0)
    return {
        "split": pl["split"],
        "fit_ranges": pl["fit"],
        "stream": pl["stream"],
        "stream_ranges": pl["stream_ranges"],
        "split_crop": {"w": sw, "h": sh},
        "voices": voices,
        "speaker_colors": palette,
        "speaker_map": (clip.track or {}).get("speaker_map", {}),
        "edit": e,
        "a": pl["a"], "b": pl["b"], "duration": round(pl["duration"], 3),
        "segs": [[round(s, 3), round(t, 3)] for s, t in pl["segs"]],
        "words": pl["words"],
        "cues": cue_out,
        "zoom": zk,
        "emoji": emo,
        "layout": pl["layout"],
        "keys": pl["keys"] if (pl["keys"] and (clip.track or {}).get("axis", "x") == axis) else None,
        "tracked": pl["keys"] is not None or pl["layout"] != "track",
        "crop": {"w": cw, "h": ch, "axis": axis},
        "out": {"w": pl["w"], "h": pl["h"]},
        "scale": output_scale(pl["w"], pl["h"]),
        "preset": st,
        "upper": st["upper"] if cap.get("uppercase") is None else bool(cap["uppercase"]),
        "per_cue": per_cue,
        "title": pl["title"]["text"],
        "title_info": pl["title"],
        "banner": clip.banner,
    }


def _select_expr(segs: list[tuple[float, float]]) -> str:
    return "+".join(f"between(t,{s:.3f},{e:.3f})" for s, e in segs)


def _range_expr(ranges: list[tuple[float, float]]) -> str:
    return "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in ranges) or "0"


def _scene_expr(split: list[dict], half: str, axis: str) -> str:
    """Per-scene piecewise expression for one half of the split screen (clip-relative t)."""
    out = "0.5"
    for sc in reversed(split):
        e = expr(sc[half][axis], "t")
        out = f"if(between(t,{sc['t0'] - 0.05:.3f},{sc['t1'] + 0.05:.3f}),{e},{out})"
    return out


def build_graph(pl: dict, src_w: int, src_h: int, has_audio: bool, ass_name: str | None,
                emojis: list[tuple[Path, dict]], track_axis: str) -> str:
    ow, oh, fps = pl["w"], pl["h"], settings.fps
    edit = pl["edit"]
    segs = pl["segs"]
    full = len(segs) == 1 and segs[0][0] <= 0.001 and abs(segs[0][1] - (pl["b"] - pl["a"])) < 0.01
    select = f"select='{_select_expr(segs)}',setpts=N/FRAME_RATE/TB" if not full else ""
    head = f"setpts=PTS-STARTPTS,fps={fps}"
    chain: list[str] = []

    cw, ch, axis = crop_box(src_w, src_h, ow, oh)
    layout = pl["layout"]
    if layout in ("track", "center", "split") and axis != "none":
        keys = pl["keys"] if (layout != "center" and pl["keys"] and track_axis == axis) else [[0.0, 0.5]]
        c = expr(keys, "t")
        if axis == "x":
            chain.append(f"crop=w={cw}:h={ch}:x='clip(({c})*iw-ow/2,0,iw-ow)':y=0")
        else:
            chain.append(f"crop=w={cw}:h={ch}:x=0:y='clip(({c})*ih-oh/2,0,ih-oh)'")
    if select:
        chain.append(select)

    graph = []
    split = pl.get("split") or []
    fit_ranges = pl.get("fit") or []
    stream = pl.get("stream")
    stream_ranges = pl.get("stream_ranges") or []

    def stream_branch(src: str, out: str) -> None:
        """Gaming layout: face-cam panel stacked with the gameplay panel."""
        cx, cy, cw2, ch2 = stream["cam_crop"]
        gx, gy, gw, gh = stream["game_crop"]
        cam_h, game_h = stream["cam_h"], stream["game_h"]
        graph.append(f"{src}{select + ',' if select else ''}split=2[g1][g2]")
        graph.append(f"[g1]crop={cw2}:{ch2}:{cx}:{cy},scale={ow}:{cam_h}:flags=lanczos,setsar=1[gcam]")
        if stream["game"] == "fit":
            bw, bh = max(2, ow // 8 // 2 * 2), max(2, game_h // 8 // 2 * 2)
            graph.append("[g2]split=2[g3][g4]")
            graph.append(f"[g3]scale={bw}:{bh}:force_original_aspect_ratio=increase,crop={bw}:{bh},"
                         f"boxblur=6:2,scale={ow}:{game_h}:flags=bilinear,eq=brightness=-0.07[ggbg]")
            graph.append(f"[g4]scale={ow}:{game_h}:force_original_aspect_ratio=decrease:flags=lanczos[ggfg]")
            graph.append("[ggbg][ggfg]overlay=(W-w)/2:(H-h)/2,setsar=1[ggame]")
        else:
            graph.append(f"[g2]crop={gw}:{gh}:{gx}:{gy},scale={ow}:{game_h}:flags=lanczos,setsar=1[ggame]")
        top_first = stream["cam_pos"] == "top"
        seam = (cam_h if top_first else game_h) - 2
        pair = "[gcam][ggame]" if top_first else "[ggame][gcam]"
        graph.append(f"{pair}vstack=2,drawbox=x=0:y={seam}:w={ow}:h=4:color=black@0.85:t=fill{out}")

    if layout == "gaming" and stream:
        stream_branch(f"[0:v]{head},", "[v0]")
        stream_ranges = []   # the whole clip is already the gaming layout
        split, fit_ranges = [], []
    src_label = "[0:v]"
    branches = 1 + bool(split) + bool(fit_ranges) + bool(stream_ranges)
    if layout == "gaming" and stream:
        branches = 0
    elif branches > 1:
        labels = ("[vm]" + ("[vs]" if split else "") + ("[vf]" if fit_ranges else "")
                  + ("[vg]" if stream_ranges else ""))
        graph.append(f"[0:v]{head},split={branches}{labels}")
        src_label = "[vm]"
    else:
        chain.insert(0, head)
    if stream_ranges:
        stream_branch("[vg]", "[vstream]")
    if fit_ranges:
        # scenes with nobody on screen: the whole frame over a blurred copy of itself
        bw, bh = max(2, ow // 8 // 2 * 2), max(2, oh // 8 // 2 * 2)
        graph.append(f"[vf]{select + ',' if select else ''}split=2[fa][fb]")
        graph.append(f"[fa]scale={bw}:{bh}:force_original_aspect_ratio=increase,crop={bw}:{bh},"
                     f"boxblur=6:2,scale={ow}:{oh}:flags=bilinear,eq=brightness=-0.07:saturation=1.1[fbg]")
        graph.append(f"[fb]scale={ow}:{oh}:force_original_aspect_ratio=decrease:flags=lanczos[ffg]")
        graph.append("[fbg][ffg]overlay=(W-w)/2:(H-h)/2,setsar=1[vfit]")
    if split:
        sw, sh = _split_size(pl, src_w, src_h)
        half = oh // 2
        parts = []
        for name in ("top", "bottom"):
            x = _scene_expr(split, name, "x")
            y = _scene_expr(split, name, "y")
            parts.append(f"crop=w={sw}:h={sh}:x='clip(({x})*iw-ow/2,0,iw-ow)':"
                         f"y='clip(({y})*ih-oh*0.42,0,ih-oh)',scale={ow}:{half}:flags=lanczos,setsar=1")
        graph.append("[vs]split=2[sa][sb]")
        graph.append(f"[sa]{parts[0]}[st]")
        graph.append(f"[sb]{parts[1]}[sbt]")
        graph.append(f"[st][sbt]vstack=2,drawbox=x=0:y={half - 2}:w={ow}:h=4:color=black@0.85:t=fill"
                     + (f",{select}" if select else "") + "[vsp]")

    if layout == "gaming" and stream:
        pass   # [v0] already built by stream_branch
    elif layout == "fit" and axis != "none":
        bw, bh = max(2, ow // 8 // 2 * 2), max(2, oh // 8 // 2 * 2)
        graph.append(f"{src_label}{','.join(chain)},split=2[va][vb]")
        graph.append(f"[va]scale={bw}:{bh}:force_original_aspect_ratio=increase,crop={bw}:{bh},"
                     f"boxblur=6:2,scale={ow}:{oh}:flags=bilinear,eq=brightness=-0.07:saturation=1.1[bg]")
        graph.append(f"[vb]scale={ow}:{oh}:force_original_aspect_ratio=decrease:flags=lanczos[fg]")
        graph.append("[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[v0]")
    else:
        zcfg = edit.get("zoom") or {}
        if zcfg.get("enabled") and float(zcfg.get("strength") or 0) > 0:
            ev = zoom_events(pl["words"], pl["duration"], float(zcfg["strength"]))
            zk: list[list[float]] = [[0.0, 1.0]]
            for t, lvl in ev[1:]:
                zk.append([t, zk[-1][1]])
                zk.append([t + 0.22, lvl])
            z = expr(zk, "(on/%d)" % fps)
            chain.append(f"zoompan=z='{z}':x='iw/2-iw/zoom/2':y='(ih-ih/zoom)*0.42':d=1:s={ow}x{oh}:fps={fps}")
        else:
            chain.append(f"scale={ow}:{oh}:flags=lanczos")
        chain.append("setsar=1")
        graph.append(f"{src_label}{','.join(chain)}[v0]")

    cur = "v0"
    if split:
        ranges = [(remap_clamped(sc["t0"], segs), remap_clamped(sc["t1"], segs)) for sc in split]
        graph.append(f"[v0][vsp]overlay=0:0:enable='{_range_expr(ranges)}'[vo]")
        cur = "vo"
    if fit_ranges:
        ranges = [(remap_clamped(s, segs), remap_clamped(e, segs)) for s, e in fit_ranges]
        graph.append(f"[{cur}][vfit]overlay=0:0:enable='{_range_expr(ranges)}'[vof]")
        cur = "vof"
    if stream_ranges:
        ranges = [(remap_clamped(s, segs), remap_clamped(e, segs)) for s, e in stream_ranges]
        graph.append(f"[{cur}][vstream]overlay=0:0:enable='{_range_expr(ranges)}'[vog]")
        cur = "vog"

    post = [COLOR_FILTERS.get(edit.get("filter") or "none", COLOR_FILTERS["none"])]
    if ass_name:
        post.append(f"subtitles={ass_name}")
    graph.append(f"[{cur}]{','.join(post)}[v1]")
    cur = "v1"
    for i, (_, ev) in enumerate(emojis, start=1):
        nxt = f"v{i + 1}"
        graph.append(f"[{cur}][{i}:v]overlay=x='(W-w)/2':y='{ev['y']}-h/2':"
                     f"enable='between(t,{ev['t']:.3f},{ev['end']:.3f})'[{nxt}]")
        cur = nxt
    graph.append(f"[{cur}]format=yuv420p[v]")

    if has_audio:
        a = ["asetpts=PTS-STARTPTS"]
        if not full:
            a.append(f"aselect='{_select_expr(segs)}',asetpts=N/SR/TB")
        if (edit.get("audio") or {}).get("normalize", True):
            a.append("loudnorm=I=-14:TP=-1.5:LRA=11")
        a.append("aresample=48000")
        graph.append(f"[0:a]{','.join(a)}[a]")
    return ";\n".join(graph)


def render_clip(src: str | Path, src_w: int, src_h: int, has_audio: bool, clip: Clip,
                sentences: list[Sentence], out_dir: Path, log=None, on_progress=None,
                offset: float = 0.0) -> dict:
    """`offset`: absolute source time at the file's 0 (link projects render from pieces)."""
    t_start = time.time()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(src).resolve()
    pl = plan(clip, sentences, src_w, src_h)
    edit = pl["edit"]
    ver = clip.version + 1
    stem = f"{clip.id}_v{ver}"
    video_path = out_dir / f"{stem}.mp4"
    thumb_path = out_dir / f"{stem}.jpg"
    ass_path = out_dir / f"{stem}.ass"
    fg_path = out_dir / f"{stem}.graph.txt"

    ass_path.write_text(build_ass(pl["words"], pl["duration"], pl["w"], pl["h"], edit,
                                  title=pl["title"]), encoding="utf-8")

    emojis: list[tuple[Path, dict]] = []
    cap = edit["captions"]
    if cap.get("enabled") and cap.get("emoji"):
        st = get_preset(cap.get("preset"))
        fs = st["size"] * output_scale(pl["w"], pl["h"]) * float(cap.get("size") or 1.0)
        size = int(fs * 1.5) // 2 * 2
        y = int(pl["h"] * float(cap.get("position", 0.7)) - fs * 1.45)
        for ev in emoji_events(pl["words"]):
            png = emoji_png(ev["emoji"], size)
            if png:
                emojis.append((png, {**ev, "y": y, "end": min(ev["end"], pl["duration"])}))

    dur = pl["duration"]
    track_axis = (clip.track or {}).get("axis", "x")

    def attempt(gpu: bool, hwaccel: bool) -> tuple[int, list[str]]:
        graph = build_graph(pl, src_w, src_h, has_audio, ass_path.name, emojis, track_axis)
        fg_path.write_text(graph, encoding="utf-8")
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostdin"]
        if hwaccel:
            cmd += ["-hwaccel", "cuda"]
        cmd += ["-ss", f"{max(0.0, pl['a'] - offset):.3f}", "-t", f"{pl['b'] - pl['a']:.3f}", "-i", str(src)]
        for png, _ in emojis:
            cmd += ["-loop", "1", "-framerate", str(settings.fps), "-t", f"{dur:.3f}", "-i", str(png)]
        cmd += [_graph_option(), str(fg_path), "-map", "[v]"]
        if has_audio:
            cmd += ["-map", "[a]", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]
        cmd += _video_args(gpu)
        cmd += ["-t", f"{dur:.3f}", "-movflags", "+faststart", "-progress", "pipe:1", "-nostats",
                str(video_path)]
        proc = subprocess.Popen(cmd, cwd=str(out_dir), stdout=subprocess.PIPE, **NO_WINDOW,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace", bufsize=1)
        tail: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            if line.startswith("out_time_us="):
                if on_progress:
                    try:
                        on_progress(min(1.0, int(line.split("=", 1)[1]) / 1e6 / max(0.1, dur)))
                    except ValueError:
                        pass
                continue
            if "=" in line and " " not in line:
                continue  # other -progress keys
            tail = (tail + [line])[-30:]
        code = proc.wait()
        ok = code == 0 and video_path.exists() and video_path.stat().st_size > 10_000
        return (0 if ok else (code or 1)), tail

    gpu = _nvenc_ok()
    with _flags_lock:
        hw = gpu and _flags["hwaccel"]
    tries = [(gpu, hw)]
    if hw:
        tries.append((gpu, False))
    if gpu:
        tries.append((False, False))
    tail: list[str] = []
    used = None
    for g, h in tries:
        code, tail = attempt(g, h)
        if code == 0:
            used = (g, h)
            break
        if h:
            with _flags_lock:
                _flags["hwaccel"] = False
        if log:
            log(f"render retry ({'nvenc' if g else 'x264'}{' +cuda' if h else ''} failed): "
                + " | ".join(tail[-3:])[:300])
    fg_path.unlink(missing_ok=True)
    if used is None:
        raise RenderError("ffmpeg failed:\n" + "\n".join(tail[-15:]))

    try:
        run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss",
             f"{min(max(0.5, dur * 0.3), max(0.0, dur - 0.2)):.3f}", "-i", str(video_path),
             "-frames:v", "1", "-vf", "scale=360:-2", "-q:v", "4", str(thumb_path)])
    except Exception as e:
        if log:
            log(f"thumbnail failed: {e}")

    # drop older versions of this clip (best effort: a browser may still hold the file open)
    for old in out_dir.glob(f"{clip.id}_v*.*"):
        if old.stem != stem and old.suffix in (".mp4", ".jpg", ".ass"):
            try:
                old.unlink()
            except OSError:
                pass

    meta = probe(video_path)
    return {
        "file": video_path.name,
        "thumb": thumb_path.name if thumb_path.exists() else None,
        "version": ver,
        "duration": meta["duration"],
        "bytes": video_path.stat().st_size,
        "encoder": ("nvenc" if used[0] else "x264") + ("+cuda" if used[1] else ""),
        "seconds": round(time.time() - t_start, 2),
        "layout": pl["layout"],
    }
