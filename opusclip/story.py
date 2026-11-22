"""Whole-video understanding: where people get excited, what happens, what pays off.

Clip finders that read a long video one window at a time miss what a human editor
knows: who these people are, what the running storyline is, which moment pays off a
setup from an hour ago, and where the room actually lights up. This module builds that
context once per video and every later step (shorts, long episodes, titles) uses it.

1. Engagement map (any video, per second, 0..1): loudness spikes against the video's own
   level, sudden jumps, bursts of fast talk, exclamations, reaction phrases ("no way",
   "are you serious", "bhai kya"), laughter, and fast back-and-forth between speakers.
2. Story log: the video in ~5-minute parts. For each: what happens (names, game events,
   claims), tags (funny, clutch, argument, reveal, shock...), a 1-10 clip-worthy rating,
   and whether it sets up or pays off something. Then one brief for the whole video:
   who, what, and the storylines that run through it.
3. Long episodes: standalone 5-20 minute YouTube videos built from runs of parts with a
   setup and a payoff, edges moved onto natural pauses, dead stretches cut out.
4. Packaging: a YouTube title, 2-4 words of thumbnail text and a description for every
   clip, written with the whole-video brief in mind.

Everything has a rules-only fallback, so it works without an AI too.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .audio import AudioFeatures
from .boundaries import is_punctuated, mid_sentence, opening_issue
from .llm import LLM, LLMError
from .models import Sentence

STORY_VERSION = 3
PART_SECONDS = 300.0          # one story-log part
PARTS_PER_CALL = 8            # parts logged per AI call (the answer has to fit as well)

REACTIONS = re.compile(
    r"\b(no way|are you (serious|kidding)|what the|oh my god|omg|oh no|wait what|bro what|holy|"
    r"let'?s go+|that'?s (crazy|insane|wild)|i can'?t believe|shut up|stop it|no no no|"
    r"bhai kya|kya baat|kya bol|pagal|arre yaar|oh bhai|mat kar|chal be|bas kar|"
    r"भाई क्या|क्या बात|पागल|अरे यार)\b", re.I)
LAUGH = re.compile(r"\b(ha(ha)+|he(he)+|lol|lmao|lmfao|rofl)\b|\[(laughter|laughs|laughing)\]|\((laughs|laughter)\)|हाहा", re.I)


# --------------------------------------------------------------------------
# 1. engagement
# --------------------------------------------------------------------------
def engagement_map(audio: AudioFeatures, sentences: list[Sentence], duration: float) -> np.ndarray:
    """Per-second engagement 0..1 for any kind of video."""
    from .gaming import hype_map

    n = int(duration) + 1
    base = hype_map(audio, sentences, duration)[:n]
    react = np.zeros(n)
    turns = np.zeros(n)
    last_sp = None
    for s in sentences:
        i0, i1 = max(0, int(s.start)), min(n - 1, int(s.end))
        hits = len(REACTIONS.findall(s.text)) + 1.5 * len(LAUGH.findall(s.text))
        if hits:
            react[i0 : i1 + 1] += hits / max(1, i1 - i0 + 1) * 3
        for w in s.words:
            sp = getattr(w, "sp", -1)
            if sp is not None and sp >= 0:
                if last_sp is not None and sp != last_sp and 0 <= int(w.start) < n:
                    turns[int(w.start)] += 1
                last_sp = sp

    def z(x: np.ndarray) -> np.ndarray:
        sd = float(np.std(x)) or 1.0
        return np.clip((x - float(np.mean(x))) / sd, 0, 4) / 4

    k10 = np.ones(min(10, n)) / min(10, n)
    turns = np.convolve(turns, k10, mode="same")
    score = 0.62 * base + 0.24 * z(react) + 0.14 * z(turns)
    k5 = np.ones(min(5, n)) / min(5, n)
    score = np.convolve(score, k5, mode="same")
    top = float(np.percentile(score, 99)) or 1.0
    return np.clip(score / top, 0, 1).astype(np.float32)


def levels(emap: np.ndarray) -> tuple[float, float]:
    """This video's own bar for [hot] and [HYPE] (85th / 96th percentile)."""
    if emap is None or not len(emap):
        return 1.1, 1.1
    return float(np.percentile(emap, 85)), float(np.percentile(emap, 96))


def mark(emap: np.ndarray | None, t0: float, t1: float, lv: tuple[float, float]) -> str:
    if emap is None or not len(emap):
        return ""
    seg = emap[max(0, int(t0)) : max(int(t0) + 1, min(len(emap), int(t1) + 1))]
    if not seg.size:
        return ""
    peak = float(seg.max())
    return "[HYPE] " if peak >= lv[1] else "[hot] " if peak >= lv[0] else ""


def peak_rank(emap: np.ndarray | None, t0: float, t1: float) -> float:
    """How the clip's strongest second ranks in the whole video (0 = flat, 1 = the peak)."""
    if emap is None or not len(emap):
        return 0.5
    seg = emap[max(0, int(t0)) : max(int(t0) + 1, min(len(emap), int(t1) + 1))]
    if not seg.size:
        return 0.5
    return float(np.mean(emap <= float(seg.max())))


# --------------------------------------------------------------------------
# 2. story log
# --------------------------------------------------------------------------
def _parts(sentences: list[Sentence], duration: float, part: float = PART_SECONDS) -> list[dict]:
    """~5-minute parts, each boundary moved onto the longest pause within 40 s of it."""
    if not sentences:
        return []
    bounds = [0.0]
    t = part
    gaps = [(sentences[i + 1].start, sentences[i + 1].start - sentences[i].end) for i in range(len(sentences) - 1)]
    while t < duration - part * 0.4:
        near = [g for g in gaps if abs(g[0] - t) <= 40]
        bounds.append(max(near, key=lambda g: g[1])[0] if near else t)
        t = bounds[-1] + part
    bounds.append(duration)
    out = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if b - a < 5:
            continue
        out.append({"id": len(out), "t0": round(a, 2), "t1": round(b, 2)})
    return out


def _part_lines(sentences: list[Sentence], a: float, b: float, emap, lv, budget: int) -> str:
    lines = []
    for s in sentences:
        if s.end < a or s.start >= b:
            continue
        lines.append(f"{s.start:.0f} {mark(emap, s.start, s.end, lv)}| {s.text.strip()[:180]}")
    text = "\n".join(lines)
    if len(text) <= budget:
        return text or "(no speech: gameplay / music)"
    hot = [ln for ln in lines if "[" in ln.split("|")[0]]
    calm = [ln for ln in lines if "[" not in ln.split("|")[0]]
    room = max(0, budget - sum(len(x) + 1 for x in hot))
    step = max(1, int(np.ceil(sum(len(x) + 1 for x in calm) / max(1, room))))
    keep = set(hot) | set(calm[::step])
    return "\n".join(ln for ln in lines if ln in keep)[:budget]


PARTS_SYSTEM = """You are logging a long video (stream, podcast, show) for a video editor, part by part.
Lines are "seconds [mark] | speech"; [HYPE] and [hot] mark the loudest, most intense seconds.
For EACH part return:
- what: 1-2 concrete sentences: who does or says what (names, game events, claims, numbers).
  Not "they talk" - say WHAT happens. Gameplay with little speech: describe what the
  reactions show (a win, a death, a scare, a chase).
- tags: from funny, clutch, fail, rage, argument, story, reveal, shock, emotional, advice,
  hype, wholesome, drama, chill
- rating 1-10: how clip-worthy (10 = would go viral, 1 = dead air, setup, ads, chat reading)
- setup: what this part starts that may pay off later ("" if nothing)
- payoff: what earlier setup this part pays off ("" if nothing)
{lang}
Return STRICT JSON only: {{"parts":[{{"id":0,"what":"","tags":[""],"rating":0,"setup":"","payoff":""}}]}}"""

PARTS_SCHEMA = {
    "type": "object",
    "properties": {"parts": {"type": "array", "items": {"type": "object", "properties": {
        "id": {"type": "integer"}, "what": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "rating": {"type": "number"}, "setup": {"type": "string"}, "payoff": {"type": "string"}},
        "required": ["id", "what", "rating"]}}},
    "required": ["parts"],
}

BRIEF_SYSTEM = """You get the part-by-part log of one long video. Write the brief an editor keeps
next to them while cutting it. Return STRICT JSON only:
{"who":"streamer / hosts / guests and who is who","what":"game, show or topic in one line",
"summary":"3 sentences: what happens over the whole video",
"storylines":["a running story or joke, and where it pays off (in seconds)"],
"best":"the single moment people will talk about, with its time in seconds"}
{lang}"""


def _lang_line(language: str) -> str:
    lang = (language or "en")[:2]
    if lang == "hi":
        return "Write in Hinglish (Hindi in Latin letters). Never use Chinese or unrelated scripts."
    if lang == "en":
        return "Write in English."
    return f"Write in the video's language ({lang}). Never use unrelated scripts."


def _rules_part(sentences: list[Sentence], p: dict, emap: np.ndarray, lv) -> dict:
    inside = [s for s in sentences if s.start >= p["t0"] and s.start < p["t1"]]
    seg = emap[int(p["t0"]) : max(int(p["t0"]) + 1, int(p["t1"]))] if len(emap) else np.zeros(1)
    best = max(inside, key=lambda s: float(emap[min(len(emap) - 1, int(s.start))]) if len(emap) else 0,
               default=None)
    pk = float(seg.max()) if seg.size else 0.0
    rating = 1 + 9 * float(np.mean(emap <= pk)) if len(emap) else 5
    return {"what": (best.text.strip()[:160] if best else "(no speech)"), "tags": [],
            "rating": round(rating, 1), "setup": "", "payoff": "", "by": "rules"}


def build_story(sentences: list[Sentence], emap: np.ndarray, duration: float, llm: LLM | None,
                context: str = "", language: str = "", log=None) -> dict:
    """The story log + brief for one video (see the module doc)."""
    parts = _parts(sentences, duration)
    lv = levels(emap)
    for p in parts:
        seg = emap[int(p["t0"]) : max(int(p["t0"]) + 1, int(p["t1"]))] if len(emap) else np.zeros(1)
        p["energy"] = round(float(seg.mean()), 3) if seg.size else 0.0
        p["peak"] = round(float(seg.max()), 3) if seg.size else 0.0
        p["peak_t"] = round(p["t0"] + float(np.argmax(seg)), 1) if seg.size else p["t0"]
        p.update(_rules_part(sentences, p, emap, lv))
    # rules rating: rank among this video's own parts (every 5 min hold some loud second,
    # so an absolute bar would rate them all 10)
    if parts:
        vals = np.array([0.6 * p["energy"] + 0.4 * p["peak"] for p in parts])
        for p, v in zip(parts, vals):
            p["rating"] = round(1 + 9 * float(np.mean(vals <= v)), 1) if len(parts) > 1 else 6.0
    use_llm = llm is not None and llm.available and len(parts) > 0
    brief: dict = {}
    if use_llm:
        per = max(2500, min(9000, (llm.context_chars - 3500) // 3))
        system = PARTS_SYSTEM.format(lang=_lang_line(language))
        texts = {p["id"]: _part_lines(sentences, p["t0"], p["t1"], emap, lv, per) for p in parts}
        batches, cur, size = [], [], 0
        budget = max(per, llm.context_chars - 4000)
        for p in parts:
            ln = len(texts[p["id"]]) + 60
            # the ANSWER must fit too: a 4-hour stream in one call came back cut off mid-JSON,
            # so at most PARTS_PER_CALL parts go into one request
            if cur and (size + ln > budget or len(cur) >= PARTS_PER_CALL):
                batches.append(cur)
                cur, size = [], 0
            cur.append(p)
            size += ln
        if cur:
            batches.append(cur)
        done = [0]

        def run(batch: list[dict], retry: bool = True) -> None:
            blocks = [f"PART {p['id']} ({p['t0']:.0f}-{p['t1']:.0f}s)\n{texts[p['id']]}" for p in batch]
            user = ((f"VIDEO TITLE: {context}\n\n" if context else "") + "\n\n".join(blocks)
                    + f"\n\nLog every part ({', '.join(str(p['id']) for p in batch)}).")
            try:
                data = llm.json_completion(system, user, schema=PARTS_SCHEMA,
                                           max_tokens=min(8000, 500 + 260 * len(batch)), temperature=0.2)
            except LLMError as e:
                if len(batch) > 1:          # retry as two smaller calls instead of losing them all
                    if log:
                        log(f"story log: a batch of {len(batch)} failed ({str(e)[:80]}), retrying in halves")
                    half = len(batch) // 2
                    run(batch[:half])
                    run(batch[half:])
                elif log:
                    log(f"story log: part {batch[0]['id']} failed ({str(e)[:80]}), using the energy map")
                return
            byid = {p["id"]: p for p in batch}
            for item in data.get("parts") or []:
                try:
                    p = byid[int(item.get("id"))]
                except (KeyError, TypeError, ValueError):
                    continue
                try:
                    r = float(item.get("rating"))
                    r = r / 10 if r > 10 else r
                except (TypeError, ValueError):
                    r = p["rating"]
                # blend the AI's read with how loud the room actually got
                p.update({"what": str(item.get("what") or p["what"])[:300],
                          "tags": [str(t).lower()[:16] for t in (item.get("tags") or [])][:4],
                          "rating": round(0.75 * max(1.0, min(10.0, r)) + 0.25 * p["rating"], 1),
                          "setup": str(item.get("setup") or "")[:200],
                          "payoff": str(item.get("payoff") or "")[:200], "by": "ai"})
            missing = [p for p in batch if p.get("by") != "ai"]
            if missing and retry:
                # the model answered but skipped some parts: ask once more for just those
                if log:
                    log(f"story log: asking again about {len(missing)} skipped part(s)")
                done[0] += len(batch) - len(missing)
                run(missing, retry=False)
                return
            done[0] += len(batch)
            if log:
                log(f"story log: {done[0]}/{len(parts)} parts read")

        if log:
            log(f"understanding the whole video: {len(parts)} parts in {len(batches)} AI calls")
        if llm.provider == "ollama":
            for b in batches:
                run(b)
        else:
            with ThreadPoolExecutor(max_workers=3) as pool:
                list(pool.map(run, batches))
        log_text = "\n".join(f"{p['t0']:.0f}-{p['t1']:.0f}s r{p['rating']} [{','.join(p.get('tags') or [])}] "
                             f"{p['what']}" + (f" SETUP:{p['setup']}" if p.get("setup") else "")
                             + (f" PAYOFF:{p['payoff']}" if p.get("payoff") else "") for p in parts)
        try:
            brief = llm.json_completion(BRIEF_SYSTEM.replace("{lang}", _lang_line(language)),
                                        (f"VIDEO TITLE: {context}\n\n" if context else "") + log_text[: llm.context_chars - 3000],
                                        max_tokens=1200, temperature=0.2)
        except LLMError as e:
            if log:
                log(f"story brief skipped ({str(e)[:100]})")
            brief = {}
    if not brief:
        top = sorted(parts, key=lambda p: -p["rating"])[:3]
        brief = {"who": "", "what": context, "summary": "", "storylines": [],
                 "best": "; ".join(f"{p['peak_t']:.0f}s: {p['what'][:80]}" for p in top)}
    brief = {k: (v if isinstance(v, (str, list)) else str(v)) for k, v in brief.items()}
    read = sum(1 for p in parts if p.get("by") == "ai")
    if log and use_llm:
        log(f"story log: the AI read {read} of {len(parts)} parts")
    # "ai" only when the AI really read (nearly) all of it, so a failed run is redone next time
    by = "ai" if use_llm and read >= 0.8 * len(parts) else ("partial" if read else "rules")
    return {"version": STORY_VERSION, "by": by, "ai_parts": read, "parts": parts, "brief": brief}


def brief_text(story: dict | None, limit: int = 1400) -> str:
    """The brief as a few lines for prompts."""
    if not story:
        return ""
    b = story.get("brief") or {}
    lines = [f"WHO: {b.get('who')}" if b.get("who") else "", f"WHAT: {b.get('what')}" if b.get("what") else "",
             f"SUMMARY: {b.get('summary')}" if b.get("summary") else ""]
    sl = b.get("storylines") or []
    if isinstance(sl, list) and sl:
        lines.append("STORYLINES: " + " | ".join(str(x)[:160] for x in sl[:5]))
    return "\n".join(x for x in lines if x)[:limit]


def parts_near(story: dict | None, a: float, b: float, pad: float = 120.0) -> str:
    """What happens around [a, b], from the story log (for a window or a clip)."""
    if not story:
        return ""
    out = []
    for p in story.get("parts") or []:
        if p["t1"] >= a - pad and p["t0"] <= b + pad:
            out.append(f"{p['t0']:.0f}-{p['t1']:.0f}s (rating {p['rating']}): {p['what']}"
                       + (f" [pays off: {p['payoff']}]" if p.get("payoff") else ""))
    return "\n".join(out[:8])


def part_rating(story: dict | None, t: float) -> float | None:
    for p in (story or {}).get("parts") or []:
        if p["t0"] <= t < p["t1"]:
            return float(p["rating"])
    return None


# --------------------------------------------------------------------------
# 3. long episodes
# --------------------------------------------------------------------------
LONG_SYSTEM = """You turn a long stream / podcast into standalone YouTube videos (episodes) of
{lo:.0f}-{hi:.0f} seconds, aiming at about {target:.0f}s. You get the video's brief and its
part-by-part log (time range, rating 1-10, tags, what happens, setups and payoffs).
An episode is a run of consecutive parts that works on its own: it opens on something that
grabs (a claim, a problem, the start of a fight or a story), builds, and ends on a payoff
(the win, the reveal, the punchline, the answer). Prefer high-rated runs; never end before the
payoff; skip ads, chat reading, setup-only stretches and intros. Episodes must not overlap.
For each episode write, in {lang}:
- yt_title: a YouTube title (max 70 chars) that makes people click: name the streamer / guest
  and the game or topic, the stakes or the shocking thing, curiosity - but only promise what
  the episode delivers
- thumbnail_text: 2-4 words for the thumbnail, the emotional core ("HE QUIT LIVE", "1V5 CLUTCH")
- description: 2-3 sentences: who, what happens, why watch
- chapters: 3-6 chapters, each the part id it starts at and a short label
- rating 0-100 and why (one sentence)
Return STRICT JSON only:
{{"episodes":[{{"from_part":0,"to_part":0,"yt_title":"","thumbnail_text":"","description":"",
"chapters":[{{"part":0,"label":""}}],"rating":0,"why":""}}]}}"""

LONG_SCHEMA = {
    "type": "object",
    "properties": {"episodes": {"type": "array", "items": {"type": "object", "properties": {
        "from_part": {"type": "integer"}, "to_part": {"type": "integer"},
        "yt_title": {"type": "string"}, "thumbnail_text": {"type": "string"},
        "description": {"type": "string"},
        "chapters": {"type": "array", "items": {"type": "object", "properties": {
            "part": {"type": "integer"}, "label": {"type": "string"}}}},
        "rating": {"type": "number"}, "why": {"type": "string"}},
        "required": ["from_part", "to_part", "yt_title", "thumbnail_text", "description"]}}},
    "required": ["episodes"],
}


def _edge(sentences: list[Sentence], emap: np.ndarray, t: float, side: str, punctuated: bool,
          search: float = 75.0, not_before: float | None = None) -> float:
    """Move an episode edge onto a natural pause: a sentence that starts fresh after a
    silence (start), or a finished sentence followed by a silence and a calm (end)."""
    best, best_v = t, -1e9
    n = len(sentences)
    for i, s in enumerate(sentences):
        if side == "start":
            if abs(s.start - t) > search or mid_sentence(sentences, i, punctuated) or opening_issue(s.text):
                continue
            gap = s.start - sentences[i - 1].end if i > 0 else 3.0
            v = 2.0 * min(gap, 4.0) - abs(s.start - t) / 30.0
            if v > best_v:
                best, best_v = s.start - 0.05, v
        else:
            if abs(s.end - t) > search or (not_before is not None and s.end < not_before):
                continue
            if punctuated and not s.text.rstrip().endswith((".", "!", "?", "…", "।")):
                continue
            gap = sentences[i + 1].start - s.end if i + 1 < n else 3.0
            after = emap[min(len(emap) - 1, int(s.end) + 1) : min(len(emap), int(s.end) + 6)] if len(emap) else np.zeros(1)
            calm = 1.0 - (float(after.mean()) if after.size else 0.0)
            v = 2.0 * min(gap, 4.0) + calm - abs(s.end - t) / 30.0
            if v > best_v:
                best, best_v = s.end + 0.15, v
    return max(0.0, best)


def dead_stretches(sentences: list[Sentence], emap: np.ndarray, a: float, b: float,
                   min_len: float = 25.0) -> list[list[float]]:
    """Quiet, wordless stretches inside an episode (AFK, loading, waiting) to cut out."""
    if not len(emap):
        return []
    talk = np.zeros(len(emap), dtype=bool)
    for s in sentences:
        if s.end >= a and s.start <= b:
            talk[max(0, int(s.start)) : min(len(emap), int(s.end) + 1)] = True
    low = float(np.percentile(emap, 30))
    out, run0 = [], None
    for t in range(int(a) + 5, int(b) - 5):
        dead = not talk[t] and emap[t] <= low
        if dead and run0 is None:
            run0 = t
        if (not dead or t == int(b) - 6) and run0 is not None:
            if t - run0 >= min_len:
                out.append([float(run0 + 2), float(t - 2)])
            run0 = None
    total = sum(e - s for s, e in out)
    while out and total > 0.35 * (b - a):
        out.pop(max(range(len(out)), key=lambda i: out[i][1] - out[i][0]))
        total = sum(e - s for s, e in out)
    return out


def long_select(sentences: list[Sentence], emap: np.ndarray, story: dict, llm: LLM | None, k: int,
                lo: float = 360.0, hi: float = 900.0, target: float = 600.0, context: str = "",
                language: str = "", focus: str = "", log=None) -> list[dict]:
    """Standalone long episodes (picks shaped like highlights.llm_select's, kind=long)."""
    parts = story.get("parts") or []
    if not parts or not sentences:
        return []
    total = parts[-1]["t1"] - parts[0]["t0"]
    if total < lo * 0.9:
        return []
    punctuated = is_punctuated(sentences)
    raw: list[dict] = []
    if llm is not None and llm.available:
        rows = "\n".join(f"{p['id']} | {p['t0']:.0f}-{p['t1']:.0f}s | rating {p['rating']} | "
                         f"{','.join(p.get('tags') or [])} | {p['what']}"
                         + (f" | SETUP: {p['setup']}" if p.get("setup") else "")
                         + (f" | PAYOFF: {p['payoff']}" if p.get("payoff") else "") for p in parts)
        user = ((f"VIDEO TITLE: {context}\n" if context else "") + brief_text(story) + "\n\n"
                + (f"THE USER ASKED ONLY FOR: {focus}\n\n" if focus else "")
                + "PARTS (id | time | rating | tags | what happens):\n" + rows[: llm.context_chars - 4000]
                + f"\n\nFind the best {k} episodes.")
        if log:
            log(f"AI planning {k} long episodes from the story log")
        try:
            data = llm.json_completion(LONG_SYSTEM.format(lo=lo, hi=hi, target=target, lang=_lang_line(language)),
                                       user, schema=LONG_SCHEMA, max_tokens=3500, temperature=0.3)
            raw = [x for x in (data.get("episodes") or []) if isinstance(x, dict)]
        except LLMError as e:
            if log:
                log(f"long episodes: AI failed ({str(e)[:100]}), using the energy map")
    ids = {p["id"]: i for i, p in enumerate(parts)}

    def fit(i: int, j: int) -> tuple[int, int]:
        """Grow / shrink a run of parts to the length limits, towards the better neighbours."""
        dur = lambda a, b: parts[b]["t1"] - parts[a]["t0"]  # noqa: E731
        while dur(i, j) > hi and j > i:
            if parts[i]["rating"] < parts[j]["rating"]:
                i += 1
            else:
                j -= 1
        while dur(i, j) < lo and (i > 0 or j < len(parts) - 1):
            left = parts[i - 1]["rating"] if i > 0 else -1
            right = parts[j + 1]["rating"] if j < len(parts) - 1 else -1
            if right >= left:
                j += 1
            else:
                i -= 1
        return i, j

    eps: list[dict] = []
    for x in raw:
        try:
            i, j = ids[int(x["from_part"])], ids[int(x["to_part"])]
        except (KeyError, TypeError, ValueError):
            continue
        i, j = fit(min(i, j), max(i, j))
        eps.append({"i": i, "j": j, "x": x})
    if not eps:
        # rules: the runs of parts with the highest ratings and peaks
        span = max(1, int(round(target / PART_SECONDS)))
        cands = []
        for i in range(len(parts)):
            i2, j2 = fit(i, min(len(parts) - 1, i + span - 1))
            run = parts[i2 : j2 + 1]
            v = float(np.mean([p["rating"] for p in run])) + 2 * max(p["peak"] for p in run)
            cands.append((v, i2, j2))
        for v, i, j in sorted(cands, reverse=True):
            eps.append({"i": i, "j": j, "x": {}})
    picked: list[dict] = []
    for e in eps:
        if any(not (e["j"] < q["i"] or e["i"] > q["j"]) for q in picked):
            continue
        picked.append(e)
        if len(picked) >= k:
            break

    out = []
    for e in picked:
        run = parts[e["i"] : e["j"] + 1]
        last_peak = max(run, key=lambda p: p["peak"])["peak_t"]
        a = _edge(sentences, emap, run[0]["t0"], "start", punctuated)
        b = _edge(sentences, emap, run[-1]["t1"], "end", punctuated, not_before=last_peak + 5)
        if b - a < lo * 0.8:
            b = min(run[-1]["t1"], a + target)
        words = [w for s in sentences if s.end >= a and s.start <= b for w in s.words if a <= w.start < b]
        text = " ".join(w.text for w in words)
        x = e["x"]
        rating = x.get("rating")
        try:
            rating = float(rating)
            rating = rating * 10 if rating <= 10 else rating
        except (TypeError, ValueError):
            rating = float(np.mean([p["rating"] for p in run])) * 10
        chapters = []
        for c in x.get("chapters") or []:
            try:
                p = parts[ids[int(c.get("part"))]]
            except (KeyError, TypeError, ValueError, AttributeError):
                continue
            if a - 1 <= p["t0"] <= b:
                chapters.append([round(max(0.0, p["t0"] - a), 1), str(c.get("label") or "")[:60]])
        if not chapters:
            chapters = [[round(max(0.0, p["t0"] - a), 1), p["what"][:50]] for p in run]
        chapters[0][0] = 0.0
        best = max(run, key=lambda p: p["rating"])
        title = str(x.get("yt_title") or best["what"][:70]).strip()
        out.append({
            "start_sid": -1, "end_sid": -1, "start": round(a, 3), "end": round(b, 3), "words": words,
            "text": text, "title": title[:100], "banner": "", "hook": "",
            "summary": str(x.get("description") or " ".join(p["what"] for p in run[:3]))[:500],
            "why": str(x.get("why") or f"{len(run)} parts, best rated {best['rating']}")[:300],
            "hashtags": [], "scores": {}, "risks": [], "long": True, "rating": max(0.0, min(100.0, rating)),
            "skips": dead_stretches(sentences, emap, a, b),
            "package": {"yt_title": title[:100],
                        "thumbnail_text": str(x.get("thumbnail_text") or "").strip()[:40] or _thumb_words(best["what"]),
                        "description": str(x.get("description") or "").strip()[:800],
                        "chapters": chapters[:8]},
        })
    if log:
        mins = ", ".join("%.1f min" % ((x["end"] - x["start"]) / 60) for x in out)
        log(f"long episodes: {len(out)} ({mins})")
    return out


# --------------------------------------------------------------------------
# 4. packaging
# --------------------------------------------------------------------------
PACK_SYSTEM = """You write the text that goes with each clip when it is posted on YouTube
(Shorts or long videos), knowing the whole video it came from. For every clip, in {lang}:
- yt_title: max 70 chars. Make people click: name the streamer / guest / game when known,
  the stakes or the shocking thing said, curiosity. Only promise what the clip delivers.
- thumbnail_text: 2-4 words, the emotional core, in CAPS ("HE SAID WHAT", "LAST SECOND WIN")
- description: 1-2 sentences: who, what happens, why watch
Return STRICT JSON only: {{"clips":[{{"id":0,"yt_title":"","thumbnail_text":"","description":""}}]}}"""

PACK_SCHEMA = {
    "type": "object",
    "properties": {"clips": {"type": "array", "items": {"type": "object", "properties": {
        "id": {"type": "integer"}, "yt_title": {"type": "string"},
        "thumbnail_text": {"type": "string"}, "description": {"type": "string"}},
        "required": ["id", "yt_title", "thumbnail_text", "description"]}}},
    "required": ["clips"],
}

_THUMB_STOP = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "is", "was", "it", "this", "that",
               "i", "you", "he", "she", "they", "we", "my", "your", "so", "but", "just", "for", "with"}


def _thumb_words(text: str) -> str:
    # split on spaces, not \w: Devanagari vowel signs are not \w and would break words apart
    words = [w.strip(".,!?;:\"'()[]…।-") for w in (text or "").split()]
    words = [w for w in words if w and w.lower() not in _THUMB_STOP]
    return " ".join(words[:3]).upper()[:40]


def package_clips(clips, story: dict | None, llm: LLM | None, context: str = "", language: str = "",
                  log=None) -> None:
    """Fill clip.package (yt_title, thumbnail_text, description) for clips without one."""
    todo = [c for c in clips if not (c.package or {}).get("yt_title")]
    for c in todo:      # rules first, so every clip has something even if the AI fails
        c.package = {**(c.package or {}), "yt_title": (c.banner or c.title)[:100],
                     "thumbnail_text": _thumb_words(c.title), "description": (c.summary or "")[:300]}
    if not todo or llm is None or not llm.available:
        return
    system = PACK_SYSTEM.format(lang=_lang_line(language))
    head = (f"VIDEO TITLE: {context}\n" if context else "") + brief_text(story) + "\n\n"
    for bi in range(0, len(todo), 8):
        group = todo[bi : bi + 8]
        blocks = []
        for i, c in enumerate(group):
            near = parts_near(story, c.start, c.end, pad=0)[:400]
            blocks.append(f"[{i}] {c.end - c.start:.0f}s at {c.start:.0f}s | working title: {c.title}\n"
                          f"what happens around it: {near}\nclip: {(c.summary or '')[:500]}")
        try:
            data = llm.json_completion(system, head + "CLIPS:\n" + "\n\n".join(blocks), schema=PACK_SCHEMA,
                                       max_tokens=2000, temperature=0.4)
        except LLMError as e:
            if log:
                log(f"titles / thumbnails: AI skipped ({str(e)[:100]})")
            return
        for item in data.get("clips") or []:
            try:
                c = group[int(item.get("id"))]
            except (IndexError, TypeError, ValueError):
                continue
            t = str(item.get("yt_title") or "").strip().strip('"')
            if t:
                c.package = {**c.package, "yt_title": t[:100],
                             "thumbnail_text": str(item.get("thumbnail_text") or c.package["thumbnail_text"]).strip()[:40],
                             "description": str(item.get("description") or c.package["description"]).strip()[:600]}
    if log:
        log(f"wrote YouTube titles and thumbnail text for {len(todo)} clips")
