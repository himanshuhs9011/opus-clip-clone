"""Clip Agent: describe the clips you want in plain words.

"Clip this stream into Shorts, only the rage moments, under 30 seconds, from 1:20:00"
becomes project options, a plan the user approves, then a find -> pick -> make flow:

1. find: transcript, then every moment worth a clip (nothing is rendered yet);
2. pick: the user keeps the clips they want (the best few are pre-selected);
3. make: only the picked clips are reframed, captioned and rendered.

Reading the request is rule-based first, so it works offline and with "Rules only".
When an AI is available it is asked only for the part rules cannot read reliably: what
kind of moment the user wants. The rules follow how people phrase these requests:

* a length is the longest a clip may be ("under 60 seconds" is 60, "30 second clips" is
  30, "2 minute clips" is 120); "Shorts" / "Reels" with no number sets no length;
* a minimum only when one is said ("at least a minute", "over 30 seconds");
* a count only when a number of clips is said ("about 10 clips", "top 5");
* a part of the video only when one is named ("from 1:20:00", "the last hour",
  "between 10:00 and 25:00", "skip the first 10 minutes");
* a focus only for a kind of moment the user named ("only the rage moments", "every time
  someone gets voted out", "the biggest laughs"). "Viral", "best", "good hooks", a
  platform, a count or a length is not a kind of moment.
"""
from __future__ import annotations

import re

URL = re.compile(r"https?://[^\s<>\"']+", re.I)
NUM_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "ninety": 90, "half": 0.5,
}
_N = r"(\d+(?:\.\d+)?|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty|fifty|sixty|ninety|half)"
_SEC = r"(?:s|sec|secs|second|seconds|sekund|सेकंड|सेकेंड)(?![\w])"
_MIN = r"(?:m|min|mins|minute|minutes|minat|मिनट)(?![\w])"
_HOUR = r"(?:h|hr|hrs|hour|hours)\b"
_TS = r"(\d{1,2}:\d{2}(?::\d{2})?|\d+(?:\.\d+)?\s*(?:h|hr|hour|hours|m|min|mins|minute|minutes|s|sec|seconds)\b(?:\s*\d+\s*(?:m|min|s|sec)\b)*)"

GAMING = re.compile(r"\b(twitch|kick|gta|gameplay|gaming|game|gamer|facecam|face ?cam|"
                    r"valorant|fortnite|minecraft|bgmi|pubg|roblox|warzone|apex|cod)\b", re.I)
TALK = re.compile(r"\b(podcast|interview|episode|talk|lecture|sermon|panel|webinar|keynote)\b", re.I)
# "make all", "make 3", "ok render the top 5" - picking from clips already found. The whole
# message must be that: "make 5 clips about pricing" is a new request.
MAKE = re.compile(r"^\s*(?:ok(?:ay)?[, ]+|yes[, ]+|great[, ]+)?(?:make|render|export|finish|do)\s+"
                  r"(all|them|those|these|it|the (?:top|best|first)\s+\d+|\d+|top\s+\d+)"
                  r"(?:\s+(?:of them|clips?|ones?|please))*\s*[.!]*\s*$", re.I)
# a number followed by a time unit is a length, not a count ("30 second clips")
_NOT_TIME = r"(?!(?:s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hr|hour|hours)\b)"
GENERIC = {
    "viral", "best", "good", "great", "top", "engaging", "interesting", "hooks", "hook", "shorts",
    "short", "reels", "reel", "tiktok", "tiktoks", "youtube", "clips", "clip", "highlights",
    "highlight", "moments", "moment", "parts", "part", "captions", "caption", "video", "videos",
    "vertical", "this", "the", "a", "an", "me", "my", "some", "into", "with", "and", "of", "from",
    "please", "make", "turn", "cut", "give", "find", "get", "create", "all", "most", "for", "it",
    "stream", "podcast", "episode", "interview", "vod", "link", "here", "is", "to", "in", "out",
}
FOCUS_PATTERNS = [
    r"(?:सिर्फ|केवल|\bsirf|\bkewal)\s+(?P<f>[^.,;!?।]+)",
    r"\bonly\s+(?:the\s+)?(?P<f>[^.,;!?]+)",
    r"\bjust\s+the\s+(?P<f>[^.,;!?]+)",
    r"\bevery\s+time\s+(?P<f>[^.,;!?]+)",
    r"\b(?:whenever|each time)\s+(?P<f>[^.,;!?]+)",
    r"\b(?:moments?|clips?|parts?|bits?)\s+(?:where|when|in which|about|of|with)\s+(?P<f>[^.,;!?]+)",
    r"\b(?:about|on the topic of|related to|focus(?:ed)? on)\s+(?P<f>[^.,;!?]+)",
    r"\bthe\s+(?P<f>(?:biggest|funniest|craziest|best|scariest|wildest|saddest|angriest|loudest)\s+[^.,;!?]+)",
]
# phrases that end a focus: the rest of the request (length, count, range, style)
FOCUS_STOP = re.compile(
    r"\b(under|over|at least|at most|less than|more than|longer than|shorter than|between|from \d|after \d|"
    r"before \d|skip|(?:(?:in )?the )?(?:first|last)\s+(?:\d+\s*|an?\s+|half an?\s+)?(?:s|sec|seconds?|m|mins?|minutes?|h|hours?)\b|"
    r"with captions|without captions|no captions|"
    r"in \d|\d+\s*(?:s|sec|second|seconds|m|min|minute|minutes)\b|\d+\s+(?:clips|shorts|reels)|"
    r"as (?:shorts|reels)|for (?:shorts|reels|tiktok|youtube)|vertical|square|landscape|facecam|face cam)", re.I)


def _num(tok: str) -> float:
    tok = tok.lower()
    return float(NUM_WORDS[tok]) if tok in NUM_WORDS else float(tok)


def to_seconds(ts: str) -> float | None:
    """'1:20:00' -> 4800, '10:30' -> 630, '1h 20m' -> 4800, '90 s' -> 90, '2 hours' -> 7200."""
    ts = (ts or "").strip().lower()
    if not ts:
        return None
    if re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?", ts):
        parts = [int(x) for x in ts.split(":")]
        total = 0
        for p in parts:
            total = total * 60 + p
        return float(total)
    total, found = 0.0, False
    for n, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(h|hr|hrs|hours?|m|mins?|minutes?|s|secs?|seconds?)\b", ts):
        found = True
        mult = 3600 if unit.startswith("h") else 60 if unit.startswith("m") else 1
        total += float(n) * mult
    return total if found else None


def parse_request(text: str) -> dict:
    """Plain-words request -> structured request (see the module doc for the rules)."""
    raw = (text or "").strip()
    low = raw.lower()
    req: dict = {"text": raw[:2000], "url": None, "focus": "", "max_clip": None, "min_clip": None,
                 "count": None, "range": None, "content": "auto", "stream_template": None,
                 "aspect": None, "captions": None, "preset": None, "quality": None, "make": None,
                 "format": "short", "long_len": None}
    m = URL.search(raw)
    if m:
        req["url"] = m.group(0).rstrip(").,]")
    body = URL.sub(" ", low)

    # ---- "make 3" / "make all" (picking from clips already found)
    mk = MAKE.search(body)
    if mk:
        word = mk.group(1)
        n = re.search(r"\d+", word)
        req["make"] = "all" if word in ("all", "them", "those", "these", "it") else int(n.group(0)) if n else "all"

    # ---- length: the longest a clip may be, and an optional minimum
    def secs(n: str, unit: str) -> float:
        v = _num(n)
        return v * 60 if re.match(_MIN, unit) else v

    mx = re.search(rf"\b(?:under|below|less than|shorter than|at most|max(?:imum)?|up to|no (?:longer|more) than)\s+(?:about\s+)?{_N}\s*({_SEC}|{_MIN})", body)
    if mx:
        req["max_clip"] = secs(mx.group(1), mx.group(2))
    elif re.search(r"\bunder (?:a|one) minute\b", body):
        req["max_clip"] = 60.0
    else:   # Hindi / Hinglish order: "45 सेकंड से कम", "45 second tak", "1 minute se kam"
        hm = re.search(rf"{_N}\s*({_SEC}|{_MIN})\s*(?:से कम|तक|se kam|tak|ke andar|के अंदर)", body)
        if hm:
            req["max_clip"] = secs(hm.group(1), hm.group(2))
    mn = re.search(rf"\b(?:at least|over|more than|longer than|min(?:imum)?|no shorter than)\s+(?:about\s+)?{_N}\s*({_SEC}|{_MIN})", body)
    if mn:
        req["min_clip"] = secs(mn.group(1), mn.group(2))
    elif re.search(r"\bat least (?:a|one) minute\b", body):
        req["min_clip"] = 60.0
    rng = re.search(rf"\b{_N}\s*(?:-|to|–)\s*{_N}\s*({_SEC}|{_MIN})\s*(?:long\s+)?(?:clips?|shorts?|reels?|videos?|each)?", body)
    if rng and not mx and not mn:
        req["min_clip"], req["max_clip"] = secs(rng.group(1), rng.group(3)), secs(rng.group(2), rng.group(3))
    elif not mx:
        fixed = re.search(rf"\b{_N}[\s-]*({_SEC}|{_MIN})\s*(?:long\b|(?:long\s+)?(?:\w+\s+)?(?:clips?|shorts?|reels?|videos?|tiktoks?|each|max|वीडियो|क्लिप)(?![\w]))", body)
        if fixed:
            req["max_clip"] = secs(fixed.group(1), fixed.group(2))
    # ---- long format: standalone YouTube videos (episodes) instead of / as well as shorts
    long_words = re.search(
        r"\blong[- ]?(?:form(?:at)?)?\s+(?:videos?|clips?|episodes?|format|versions?|uploads?)\b|\blong[- ]form(?:at)?\b|"
        r"\bfull (?:episodes?|videos?)\b|\bcompilations?\b|\bbest[- ]of\b|\bhighlights? videos?\b|\bsupercut\b|"
        r"\blamb[ia]\s+(?:video|clip)s?\b|\bmain channel\b|लंबी|लम्बी|लॉन्ग|पूरा एपिसोड", body)
    both_words = re.search(r"\b(?:shorts?|short clips?|short form)\b.{0,30}\b(?:and|aur|plus|&|bhi)\b.{0,30}"
                           r"\b(?:long|lamb[ia]|full|episodes?)\b|\b(?:long|lamb[ia])\b.{0,40}\b(?:and|aur|plus|&)\b.{0,20}"
                           r"\b(?:shorts?|short clips?|reels?)\b|\bboth\b|\bdono\b|दोनों", body)
    long_len = None
    for k in ("max_clip", "min_clip"):
        if req[k] is not None and req[k] > 180:      # "10 minute videos" can't be a short
            long_len = long_len or req[k]
            req[k] = None
    if (long_len or long_words) and re.search(r"\b(?:shorts?|reels?)\b|शॉर्ट्स|दोनों", body):
        both_words = both_words or True       # "10 minute videos and shorts"
    if long_len or long_words or both_words:
        req["format"] = "both" if both_words else "long"
        if long_len is None:
            ln = re.search(rf"\b{_N}[\s-]*({_MIN})\s*(?:long\s+)?(?:videos?|episodes?|clips?|versions?|long|each)?", body)
            if ln:
                v = secs(ln.group(1), ln.group(2))
                if v > 180:
                    long_len = v
        if long_len:
            req["long_len"] = float(max(180, min(1800, round(long_len))))
    for k in ("max_clip", "min_clip"):
        if req[k] is not None:
            req[k] = float(max(10 if k == "max_clip" else 15, min(180, round(req[k]))))
    if req["min_clip"] and req["max_clip"] and req["min_clip"] > req["max_clip"]:
        req["min_clip"] = req["max_clip"]

    # ---- how many clips
    cnt = re.search(r"\b(?:about|around|roughly|~)?\s*(\d{1,3})\s+" + _NOT_TIME
                    + r"(?:\w+\s+){0,2}?(?:clips|shorts|reels|tiktoks|highlights|moments|videos)\b", body) \
        or re.search(r"\btop\s+(\d{1,3})\b", body)
    if cnt:
        n = int(cnt.group(1))
        if 1 <= n <= 100:
            req["count"] = n
    else:
        w = re.search(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|twelve|fifteen|twenty)\s+(?:\w+\s+)?(?:clips|shorts|reels|highlights|moments)\b", body)
        if w:
            req["count"] = int(_num(w.group(1)))

    # ---- which part of the video
    frm = to = last = None
    bt = re.search(rf"\bbetween\s+{_TS}\s+and\s+{_TS}", body)
    if bt:
        frm, to = to_seconds(bt.group(1)), to_seconds(bt.group(2))
    else:
        f = re.search(rf"\b(?:from|after|starting at|start at|starting from)\s+{_TS}", body)
        t = re.search(rf"\b(?:to|until|till|up to|before)\s+{_TS}", body)
        frm = to_seconds(f.group(1)) if f else None
        if t and re.search(r"(?:^|[^:\d])\d+(?:\.\d+)?\s*$", body[: t.start()]) and ":" not in t.group(1):
            t = None                 # "1 to 2 minute clips" is a length, not "up to 2:00"
        to = to_seconds(t.group(1)) if t and (not f or t.start() > f.start()) else None
    sk = re.search(rf"\bskip (?:the )?first\s+{_N}\s*({_MIN}|{_HOUR}|{_SEC})", body)
    if sk:
        v = _num(sk.group(1))
        frm = v * (3600 if re.match(_HOUR, sk.group(2)) else 60 if re.match(_MIN, sk.group(2)) else 1)
    rest = body[: sk.start()] + " " * (sk.end() - sk.start()) + body[sk.end():] if sk else body
    fl = re.search(rf"\b(?:in )?the (first|last)\s+(?:{_N}\s*)?({_MIN}|{_HOUR}|{_SEC})", rest)
    if fl:
        v = _num(fl.group(2)) if fl.group(2) else 1.0
        u = fl.group(3)
        v *= 3600 if re.match(_HOUR, u) else 60 if re.match(_MIN, u) else 1
        if fl.group(1) == "first":
            to = v
        else:
            last = v
    if frm is not None or to is not None or last is not None:
        req["range"] = {"from": frm, "to": to, "last": last}

    # ---- kind of video, layout, captions, aspect, quality
    if GAMING.search(body):
        req["content"] = "gaming"
    elif TALK.search(body):
        req["content"] = "talk"
    if re.search(r"face ?cam (?:on )?(?:the )?top|cam on top|webcam on top", body):
        req["stream_template"] = "stack"
    elif re.search(r"\b(?:big|large) face|reaction layout", body):
        req["stream_template"] = "reaction"
    elif re.search(r"\b(?:round|circle|bubble) (?:face ?cam|cam)", body):
        req["stream_template"] = "bubble"
    elif re.search(r"\bgameplay only|no face ?cam\b", body):
        req["stream_template"] = "gameplay"
    if re.search(r"\b(?:no|without) captions?\b|\bcaptions? off\b", body):
        req["captions"] = False
    elif re.search(r"\bcaptions?\b|\bsubtitles?\b", body):
        req["captions"] = True
    from .captions import PRESETS

    for pid, cfg in PRESETS.items():
        if re.search(rf"\b{re.escape(str(cfg.get('label', pid)).lower())}\b|\b{re.escape(pid)}\s+(?:captions?|style)", body):
            req["preset"] = pid
            break
    if re.search(r"\bsquare\b|\b1:1\b", body):
        req["aspect"] = "1:1"
    elif re.search(r"\b16:9\b|\blandscape\b|\bhorizontal\b|\bwidescreen\b", body):
        req["aspect"] = "16:9"
    elif re.search(r"\b4:5\b|\binstagram feed\b", body):
        req["aspect"] = "4:5"
    elif re.search(r"\b9:16\b|\bvertical\b|\bshorts?\b|\breels?\b|\btiktoks?\b", body):
        req["aspect"] = "9:16"
    if re.search(r"\b(?:only|just) (?:the )?viral\b|\bviral only\b|\bonly (?:the )?(?:very )?best\b", body):
        req["quality"] = "viral"
    elif re.search(r"\bevery (?:single )?moment\b|\ball (?:the )?moments\b|\bas many as\b", body):
        req["quality"] = "any"

    req["focus"] = rule_focus(body)
    return req


def rule_focus(body: str) -> str:
    """The kind of moment the user named, in their words; '' for 'the best moments'."""
    for pat in FOCUS_PATTERNS:
        m = re.search(pat, body, re.I)
        if not m:
            continue
        f = m.group("f")
        stop = FOCUS_STOP.search(f)
        if stop:
            f = f[: stop.start()]
        f = re.sub(r"\s+", " ", f).strip(" -:")
        f = re.sub(r"\s+(?:from|in|of|during)\s+(?:this|the|my)\s+(?:podcast|stream|video|vod|episode|interview|show)\b.*$",
                   "", f).strip()
        f = re.sub(r"\b(?:clips?|moments?|parts?|bits?|ones?|please|from this|in this|of this)\s*$", "", f).strip()
        # "only the parts about X" -> "X"
        f = re.sub(r"^(?:the\s+)?(?:parts?|moments?|bits?|clips?|times?|sections?)\s+"
                   r"(?:about|where|when|with|of|in which|that)\s+", "", f).strip()
        words = [w for w in re.findall(r"[\w']+", f)]
        if words and any(w not in GENERIC for w in words):
            return f[:300]
    return ""


FOCUS_SYSTEM = """You read a request for short clips cut from a long video and say what KIND of
moment the user wants, in their own words. Return JSON {"focus": "..."}.
- Only a kind the user named: "every time someone gets voted out", "product close-ups",
  "the biggest laughs", "moments about pricing". Never a kind you assumed.
- "Viral", "best", "engaging", "good hooks", a platform (Shorts, TikTok, Reels), a count, a
  length or a part of the video is not a kind: for those return {"focus": ""}.
- Keep every "only" or "skip" rule they gave. At most 300 characters."""


def ai_focus(text: str, llm) -> str | None:
    """Ask the AI for the focus when the rules found none; None when it cannot answer."""
    if llm is None or not getattr(llm, "available", False):
        return None
    try:
        data = llm.json_completion(FOCUS_SYSTEM, f"REQUEST: {URL.sub('<link>', text)[:1500]}",
                                   schema={"type": "object", "properties": {"focus": {"type": "string"}},
                                           "required": ["focus"]},
                                   max_tokens=200, temperature=0.0)
    except Exception:
        return None
    f = str(data.get("focus") or "").strip().strip('"')[:300]
    words = re.findall(r"[\w']+", f.lower())
    return f if words and any(w not in GENERIC for w in words) else ""


def needs_ai_focus(req: dict) -> bool:
    """Leftover words that may name a kind of moment the rules did not catch."""
    if req.get("focus") or req.get("make"):
        return False
    body = URL.sub(" ", (req.get("text") or "").lower())
    words = [w for w in re.findall(r"[a-z']+", body) if w not in GENERIC and len(w) > 3]
    return len(words) >= 2


# --------------------------------------------------------------------------
def to_options(req: dict, base: dict | None = None) -> dict:
    """Request -> pipeline options (on top of ``base``, e.g. the user's saved defaults)."""
    o = dict(base or {})
    o["focus"] = req.get("focus") or ""
    o["max_clip"] = req.get("max_clip")
    o["min_clip"] = req.get("min_clip")
    o["range"] = req.get("range")
    o["clips"] = int(min(30, req["count"])) if req.get("count") else 0
    o["pick"] = True
    o["format"] = req.get("format") or "short"
    o["long_len"] = req.get("long_len")
    if req.get("content") in ("talk", "gaming"):
        o["content"] = req["content"]
    for k in ("stream_template", "aspect", "captions", "preset", "quality"):
        if req.get(k) is not None:
            o[k] = req[k]
    return o


def _fmt(sec: float | None) -> str:
    if sec is None:
        return ""
    sec = int(round(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def plan(req: dict, has_source: bool) -> dict:
    """What will happen, in plain words, for the user to approve."""
    kind = {"gaming": "stream", "talk": "video"}.get(req.get("content"), "video")
    bits = []
    if req.get("count"):
        bits.append(f"about {req['count']} clips")
    else:
        bits.append("every moment worth a clip")
    if req.get("max_clip") and req.get("min_clip"):
        bits.append(f"{int(req['min_clip'])}-{int(req['max_clip'])} s each")
    elif req.get("max_clip"):
        bits.append(f"up to {int(req['max_clip'])} s each")
    elif req.get("min_clip"):
        bits.append(f"at least {int(req['min_clip'])} s each")
    else:
        bits.append("each as long as its moment")
    r = req.get("range") or {}
    if r.get("last"):
        bits.append(f"from the last {_fmt(r['last'])}")
    elif r.get("from") is not None and r.get("to") is not None:
        bits.append(f"between {_fmt(r['from'])} and {_fmt(r['to'])}")
    elif r.get("from") is not None:
        bits.append(f"from {_fmt(r['from'])} on")
    elif r.get("to") is not None:
        bits.append(f"up to {_fmt(r['to'])}")
    focus = req.get("focus") or ""
    fmt = req.get("format") or "short"
    long_min = round((req.get("long_len") or 600) / 60)
    if fmt == "long":
        bits[0] = f"about {req['count']} long videos" if req.get("count") else "the best long videos"
        bits[1] = f"about {long_min} min each, each with a beginning and a payoff"
    summary = f"I'll find {', '.join(bits)} in your {kind}"
    if fmt == "both":
        summary += f", plus long YouTube videos of about {long_min} min"
    summary += f", looking only for: {focus}." if focus else (
        "." if fmt == "long" else ", picked for how well they work as Shorts.")
    look = []
    if req.get("content") == "gaming" or req.get("stream_template"):
        look.append("face-cam panel + gameplay" if req.get("stream_template") != "gameplay" else "gameplay only")
    look.append({"1:1": "square", "16:9": "landscape", "4:5": "4:5"}.get(req.get("aspect") or "", "vertical 9:16"))
    look.append("no captions" if req.get("captions") is False else "animated captions")
    steps = [
        {"title": "Get the audio and the transcript", "detail": "links fetch audio only; video is downloaded later, just around the clips you make"},
        {"title": "Understand the whole video", "detail": "where people get excited or laugh, what happens part by part, which setups pay off later"},
        {"title": "Find the moments worth a clip" + (f": {focus}" if focus else ""),
         "detail": "each clip starts on a line that stands alone and ends after its payoff"},
        {"title": "Write a YouTube title, thumbnail text and description for each", "detail": "using what happens in the whole video"},
        {"title": "You pick which clips to make", "detail": "the best ones are pre-selected"},
    ]
    if fmt != "long":
        steps.append({"title": "Cut shorts to " + ", ".join(look), "detail": "framed on whoever is talking, then rendered on your GPU"})
    if fmt != "short":
        steps.append({"title": "Cut long videos in 16:9", "detail": "opens on a teaser of its best moment, dead stretches cut out"})
    questions = []
    if not has_source and not req.get("url"):
        questions.append("Send the video: paste a link, or attach a file with the paperclip.")
    return {"summary": summary, "steps": steps, "questions": questions,
            "ready": not questions, "settings": {k: req.get(k) for k in (
                "focus", "max_clip", "min_clip", "count", "range", "content", "stream_template",
                "aspect", "captions", "preset", "quality", "format", "long_len")}}


def resolve_range(r: dict | None, duration: float) -> tuple[float, float] | None:
    """{"from", "to", "last"} -> absolute (start, end) inside the video, or None for all of it."""
    if not r:
        return None
    a = r.get("from")
    b = r.get("to")
    if r.get("last"):
        a, b = max(0.0, duration - float(r["last"])), duration
    a = max(0.0, float(a or 0.0))
    b = min(duration, float(b)) if b is not None else duration
    if b - a < 20:            # a typo must not leave nothing to clip
        return None
    return a, b
