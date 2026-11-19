"""Animated captions as an .ass subtitle track (burned in by libass).

Presets are served to the React editor (GET /api/presets) and drawn there with the same
geometry, so the live preview matches the render.
"""
from __future__ import annotations

import math

PRESETS: dict[str, dict] = {
    "bold-pop": dict(label="Bold Pop", font="Arial Black", size=80, bold=True, upper=True, words=3,
                     color="#FFFFFF", active="#FFD400", keyword="#4ADE80", outline="#000000",
                     outline_w=7, shadow=3, anim="pop", box=None, glow=0),
    "beast": dict(label="Beast", font="Impact", size=110, bold=False, upper=True, words=2,
                  color="#FFFFFF", active="#22FF66", keyword="#FFE600", outline="#000000",
                  outline_w=9, shadow=4, anim="bounce", box=None, glow=0),
    "box": dict(label="Highlight Box", font="Arial Black", size=74, bold=True, upper=True, words=3,
                color="#FFFFFF", active="#FFFFFF", keyword="#FDE047", outline="#000000",
                outline_w=6, shadow=2, anim="box", box="#7C3AED", glow=0),
    "karaoke": dict(label="Karaoke", font="Arial", size=74, bold=True, upper=False, words=5,
                    color="#FFFFFF", active="#38BDF8", keyword=None, outline="#000000",
                    outline_w=6, shadow=2, anim="karaoke", box=None, glow=0),
    "neon": dict(label="Neon", font="Impact", size=92, bold=False, upper=True, words=3,
                 color="#FFF0FB", active="#FF3DF2", keyword="#7DF9FF", outline="#3B0A3B",
                 outline_w=5, shadow=0, anim="pop", box=None, glow=3),
    "clean": dict(label="Clean", font="Segoe UI Black", size=68, bold=False, upper=False, words=4,
                  color="#FFFFFF", active="#22D3EE", keyword="#FACC15", outline="#000000",
                  outline_w=5, shadow=2, anim="none", box=None, glow=0),
    "minimal": dict(label="Minimal", font="Bahnschrift", size=62, bold=True, upper=False, words=6,
                    color="#FFFFFF", active=None, keyword=None, outline="#000000",
                    outline_w=3, shadow=2, anim="fade", box=None, glow=0),
    "comic": dict(label="Comic", font="Comic Sans MS", size=80, bold=True, upper=True, words=3,
                  color="#FFE600", active="#FF4D4D", keyword="#FFFFFF", outline="#000000",
                  outline_w=7, shadow=3, anim="bounce", box=None, glow=0),
}
# names used by the previous version of the app
ALIASES = {"karaoke-green": "karaoke"}


def get_preset(name: str | None) -> dict:
    name = ALIASES.get(name or "", name or "")
    return PRESETS.get(name, PRESETS["bold-pop"])


def list_styles() -> list[dict]:
    return [{"id": k, **v} for k, v in PRESETS.items()]


SPEAKER_CANDIDATES = ["#FFFFFF", "#7DD3FC", "#F9A8D4", "#86EFAC", "#FDBA74", "#C4B5FD",
                      "#FDE68A", "#5EEAD4", "#FCA5A5"]


def _lab(hex_color: str) -> tuple[float, float, float]:
    def lin(c: float) -> float:
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(v) for v in _rgb(hex_color))
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116

    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def _dist(a: str, b: str) -> float:
    la, lb = _lab(a), _lab(b)
    return sum((p - q) ** 2 for p, q in zip(la, lb)) ** 0.5


def speaker_palette(preset: dict, n: int, overrides: dict | None = None) -> list[str]:
    """One caption colour per speaker: speaker 1 keeps the template colour, the others get
    the light colours that contrast most with the template's highlight/keyword colours and
    with each other. Mirrored nowhere - the editor reads it from the plan."""
    overrides = overrides or {}
    reserved = [c for c in (preset.get("active"), preset.get("keyword")) if c]
    out: list[str] = []
    for k in range(max(1, n)):
        if overrides.get(str(k)):
            out.append(overrides[str(k)])
            continue
        if k == 0:
            out.append(preset["color"])
            continue
        pool = [c for c in SPEAKER_CANDIDATES if c.upper() not in (x.upper() for x in out)]
        # first colour (in preference order) that clearly differs from the highlight/keyword
        # colours and from the speakers already assigned; fall back to the most distinct one
        pick = next((c for c in pool
                     if min((_dist(c, x) for x in reserved), default=99) >= 55
                     and min(_dist(c, x) for x in out) >= 25), None)
        out.append(pick or max(pool, key=lambda c: min(_dist(c, x) for x in reserved + out)))
    return out


def output_scale(w: int, h: int) -> float:
    return math.sqrt((w * h) / (1080 * 1920))


# --------------------------------------------------------------------------
def _rgb(hex_color: str | None) -> tuple[int, int, int]:
    c = (hex_color or "#FFFFFF").lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    try:
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    except ValueError:
        return 255, 255, 255


def style_color(hex_color: str | None, alpha: int = 0) -> str:  # &HAABBGGRR
    r, g, b = _rgb(hex_color)
    return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"


def tag_color(hex_color: str | None) -> str:  # &HBBGGRR&
    r, g, b = _rgb(hex_color)
    return f"&H{b:02X}{g:02X}{r:02X}&"


def _ts(t: float) -> str:
    t = max(0.0, float(t))
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _esc(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ")


def group_words(words: list[dict], max_words: int) -> list[list[dict]]:
    """Mirrored in frontend/src/lib/timeline.ts (groupWords)."""
    cues: list[list[dict]] = []
    cur: list[dict] = []
    max_chars = max_words * 7 + 4
    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        gap = (nxt["s"] - w["e"]) if nxt else 0.0
        ended = w["w"].strip()[-1:] in ".!?,;:"
        too_long = (cur[-1]["e"] - cur[0]["s"]) >= 2.8
        chars = sum(len(x["w"]) + 1 for x in cur) + (len(nxt["w"]) if nxt else 0)
        new_speaker = nxt is not None and nxt.get("sp", -1) != w.get("sp", -1)
        if len(cur) >= max_words or gap > 0.5 or ended or too_long or chars > max_chars or new_speaker:
            cues.append(cur)
            cur = []
    if cur:
        cues.append(cur)
    return cues


def cue_timing(cues: list[list[dict]], duration: float) -> list[list[tuple[float, float]]]:
    """Start/end of each word's 'active' window. Mirrored in timeline.ts (cueTiming)."""
    out = []
    prev_end = 0.0
    for ci, cue in enumerate(cues):
        nxt_start = cues[ci + 1][0]["s"] if ci + 1 < len(cues) else duration
        row = []
        for i, w in enumerate(cue):
            start = max(w["s"], prev_end)
            if i + 1 < len(cue):
                end = cue[i + 1]["s"]
            else:
                end = min(w["e"] + 0.35, nxt_start)
            if end <= start:
                end = start + 0.05
            row.append((start, end))
            prev_end = end
        out.append(row)
    return out


# --------------------------------------------------------------------------
def build_ass(owords: list[dict], duration: float, width: int, height: int, edit: dict,
              title: str | None = None) -> str:
    cap = edit.get("captions") or {}
    st = dict(get_preset(cap.get("preset")))
    sc = output_scale(width, height) * float(cap.get("size") or 1.0)
    upper = st["upper"] if cap.get("uppercase") is None else bool(cap["uppercase"])
    per_cue = int(cap.get("words") or st["words"])
    use_kw = bool(cap.get("keywords", True)) and st.get("keyword")
    fs = int(round(st["size"] * sc))
    ow = max(1, int(round(st["outline_w"] * sc)))
    shadow = int(round(st["shadow"] * sc))
    x = width // 2
    y = int(height * float(cap.get("position", 0.70)))
    margin = int(70 * output_scale(width, height))
    bold = -1 if st.get("bold") else 0

    tsc = output_scale(width, height)
    title_fs = int(round(54 * tsc))
    styles = [
        f"Style: Cap,{st['font']},{fs},{style_color(st['color'])},{style_color(st['color'])},"
        f"{style_color(st['outline'])},&H64000000,{bold},0,0,0,100,100,1,0,1,{ow},{shadow},5,"
        f"{margin},{margin},0,1",
        f"Style: Box,{st['font']},{fs},{style_color(st['active'] or st['color'])},"
        f"{style_color(st['color'])},{style_color(st.get('box') or '#7C3AED')},&H00000000,{bold},"
        f"0,0,0,100,100,1,0,3,{max(6, int(14 * sc))},0,5,{margin},{margin},0,1",
        f"Style: Kar,{st['font']},{fs},{style_color(st['active'] or st['color'])},"
        f"{style_color(st['color'])},{style_color(st['outline'])},&H64000000,{bold},0,0,0,100,100,"
        f"1,0,1,{ow},{shadow},5,{margin},{margin},0,1",
        f"Style: Title,Arial Black,{title_fs},&H00101010,&H00101010,&H00FFFFFF,&H00FFFFFF,-1,0,0,0,"
        f"100,100,0,0,3,{int(18 * tsc)},0,8,{int(110 * tsc)},{int(110 * tsc)},0,1",
        "Style: Bar,Arial,10,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1",
        f"Style: Banner,Arial Black,{int(round(58 * tsc))},&H00101010,&H00101010,&H00FFFFFF,&H00FFFFFF,-1,0,0,0,"
        f"100,100,0,0,3,{int(20 * tsc)},0,5,{int(170 * tsc)},{int(170 * tsc)},0,1",
    ]
    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {height}\nWrapStyle: 0\nScaledBorderAndShadow: yes\n"
        "YCbCr Matrix: TV.709\n\n[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        + "\n".join(styles)
        + "\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    ev: list[str] = []

    # ---------------------------------------------------------------- title
    tcfg = edit.get("title") or {}
    tinfo = title if isinstance(title, dict) else {
        "mode": "hook", "enabled": bool(tcfg.get("enabled")), "y": None,
        "text": (tcfg.get("text") or title or "").strip(), "seconds": float(tcfg.get("seconds") or 3.5)}
    ttext = (tinfo.get("text") or "").strip()
    if tinfo.get("enabled") and ttext:
        if tinfo.get("mode") == "banner":
            # context banner: stays for the whole clip, centred on the panel seam
            by = int(height * float(tinfo.get("y") or 0.16))
            ev.append(
                f"Dialogue: 5,{_ts(0)},{_ts(duration)},Banner,,0,0,0,,"
                f"{{\\an5\\pos({x},{by})\\fad(150,0)\\fscx92\\fscy92\\t(0,180,\\fscx100\\fscy100)}}"
                f"{_esc(ttext)}"
            )
        else:
            secs = min(float(tinfo.get("seconds") or 3.5), duration)
            ty = int(height * (0.11 if height > width else 0.07))
            ev.append(
                f"Dialogue: 5,{_ts(0)},{_ts(secs)},Title,,0,0,0,,"
                f"{{\\an8\\pos({x},{ty})\\fad(120,250)\\fscx88\\fscy88\\t(0,160,\\fscx100\\fscy100)}}"
                f"{_esc(ttext)}"
            )

    # -------------------------------------------------------------- progress
    pcfg = edit.get("progress") or {}
    if pcfg.get("enabled"):
        bh = max(6, int(12 * tsc))
        by = 0 if pcfg.get("position", "top") == "top" else height - bh
        dur_ms = int(duration * 1000)
        ev.append(
            f"Dialogue: 6,{_ts(0)},{_ts(duration)},Bar,,0,0,0,,"
            f"{{\\an7\\pos(0,{by})\\bord0\\shad0\\1c{tag_color(pcfg.get('color') or '#FFD400')}"
            f"\\p1\\clip(0,{by},0,{by + bh})\\t(0,{dur_ms},\\clip(0,{by},{width},{by + bh}))}}"
            f"m 0 0 l {width} 0 {width} {bh} 0 {bh}{{\\p0}}"
        )

    if not cap.get("enabled", True) or not owords:
        return header + "\n".join(ev) + "\n"

    # -------------------------------------------------------------- captions
    def word_text(w: dict) -> str:
        t = _esc(w["w"].strip())
        return t.upper() if upper else t

    glow = f"\\blur{st['glow']}" if st.get("glow") else ""
    base_c = tag_color(st["color"])
    act_c = tag_color(st["active"] or st["color"])
    kw_c = tag_color(st.get("keyword") or st["color"])
    anim = st["anim"]
    cues = group_words(owords, per_cue)
    timing = cue_timing(cues, duration)
    speakers = sorted({w.get("sp", -1) for w in owords if w.get("sp", -1) >= 0})
    palette = (speaker_palette(st, max(speakers) + 1, cap.get("speakers"))
               if cap.get("speaker_colors", True) and len(speakers) >= 2 else None)
    default_base_c = base_c

    for cue, row in zip(cues, timing):
        c0 = row[0][0]
        c1 = row[-1][1]
        sp = cue[0].get("sp", -1)
        base_c = tag_color(palette[sp]) if palette and 0 <= sp < len(palette) else default_base_c
        lead_c = f"\\c{base_c}" if base_c != default_base_c else ""
        if anim in ("karaoke", "fade") or not st.get("active"):
            if anim == "karaoke":
                parts = []
                for i, w in enumerate(cue):
                    nxt = cue[i + 1]["s"] if i + 1 < len(cue) else w["e"]
                    cs = max(1, int(round((max(nxt, w["s"] + 0.05) - w["s"]) * 100)))
                    parts.append(f"{{\\kf{cs}}}{word_text(w)}")
                lead = max(0, int(round((cue[0]["s"] - c0) * 100)))
                text = (f"{{\\k{lead}}}" if lead else "") + " ".join(parts)
                unsung = f"\\2c{base_c}" if lead_c else ""   # speaker colour before the sweep
                ev.append(f"Dialogue: 1,{_ts(c0)},{_ts(c1)},Kar,,0,0,0,,"
                          f"{{\\an5\\pos({x},{y}){glow}{unsung}\\fad(60,0)}}{text}")
            else:
                parts = [
                    f"{{\\c{kw_c}}}{word_text(w)}{{\\c{base_c}}}" if use_kw and w["k"] else word_text(w)
                    for w in cue
                ]
                ev.append(f"Dialogue: 1,{_ts(c0)},{_ts(c1)},Cap,,0,0,0,,"
                          f"{{\\an5\\pos({x},{y}){glow}{lead_c}\\fad(110,90)}}{' '.join(parts)}")
            continue

        for i, (w, (s, e)) in enumerate(zip(cue, row)):
            parts = []
            for j, cw in enumerate(cue):
                t = word_text(cw)
                if j == i:
                    if anim == "pop":
                        parts.append(f"{{\\c{act_c}\\fscx110\\fscy110}}{t}{{\\c{base_c}\\fscx100\\fscy100}}")
                    else:
                        parts.append(f"{{\\c{act_c}}}{t}{{\\c{base_c}}}")
                elif use_kw and cw["k"]:
                    parts.append(f"{{\\c{kw_c}}}{t}{{\\c{base_c}}}")
                else:
                    parts.append(t)
            enter = ""
            if i == 0:
                if anim == "pop" or anim == "box":
                    enter = "\\fscx80\\fscy80\\t(0,110,\\fscx100\\fscy100)"
                elif anim == "bounce":
                    enter = f"\\move({x},{y + int(46 * sc)},{x},{y},0,130)\\fad(70,0)"
            pos = "" if "\\move" in enter else f"\\pos({x},{y})"
            ev.append(f"Dialogue: 2,{_ts(s)},{_ts(e)},Cap,,0,0,0,,"
                      f"{{\\an5{pos}{enter}{glow}{lead_c}}}{' '.join(parts)}")
            if anim == "box":
                bparts = []
                for j, cw in enumerate(cue):
                    t = word_text(cw)
                    bparts.append(f"{{\\alpha&H00&}}{t}{{\\alpha&HFF&}}" if j == i else t)
                ev.append(f"Dialogue: 1,{_ts(s)},{_ts(e)},Box,,0,0,0,,"
                          f"{{\\an5\\pos({x},{y}){enter}\\alpha&HFF&}}{' '.join(bparts)}")

    return header + "\n".join(ev) + "\n"
