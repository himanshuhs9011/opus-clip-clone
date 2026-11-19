"""Gaming agent: highlight finding for game streams.

Talk-based selection fails on streams: a 100-minute stream may hold only ~5,000 spoken words,
and the best moments are shouts, laughs and sudden action rather than sentences. This agent
works in three steps:

1. Hype map: score every second from the sound (loudness spikes against the stream's own
   level, sudden jumps) and the speech (bursts of fast talk, exclamations in English/Hindi).
2. Review: the top peaks become candidates. The LLM reads ~60 s of transcript around each,
   marked with hype levels, and decides whether it is a real highlight, what kind
   (funny, clutch, fail, rage, reaction, story, hype), and the exact start/end: setup, payoff,
   reaction. It also writes the title and the context banner.
3. The normal viral judge (highlights.viral_judge) then compares the finalists.

Without an LLM the peaks are still used, with titles from the transcript.
"""
from __future__ import annotations

import re

import numpy as np

from .audio import AudioFeatures
from .llm import LLM, LLMError
from .models import Sentence, Word

GAMING_TITLE = re.compile(
    r"gameplay|gaming|game|live\b|stream|let'?s play|\bgta\b|bgmi|pubg|free ?fire|minecraft|valorant|"
    r"fortnite|cod\b|warzone|apex|fragpunk|roblox|ps5|xbox|\bfps\b|speedrun|rank(ed)?\b|clutch|facecam",
    re.I,
)
HYPE_WORDS = {
    # english
    "wow", "what", "no", "nooo", "yes", "lets", "let's", "go", "omg", "bro", "dude", "insane", "crazy",
    "wait", "run", "help", "dead", "died", "kill", "killed", "clutch", "haha", "hahaha", "lol", "damn",
    "oh", "god", "holy", "why", "how", "nice", "gg", "bruh", "shoot", "behind",
    # hindi / hinglish (latin and devanagari)
    "bhai", "arey", "arre", "abe", "kya", "yaar", "chal", "bhaag", "maar", "mar", "gaya", "gya", "nahi",
    "nahin", "haan", "oye", "wah", "baap", "bachao", "dekh", "dekho",
    "भाई", "अरे", "अबे", "क्या", "यार", "चल", "भाग", "मार", "गया", "नहीं", "हां", "ओए", "वाह", "बाप",
    "बचाओ", "देख", "देखो", "ओ", "भागो",
}
TYPES = ["funny", "clutch", "fail", "rage", "reaction", "story", "hype"]


def is_gaming(title: str, sentences: list[Sentence], duration: float) -> bool:
    """Game stream? Title words, or long video with sparse speech."""
    if GAMING_TITLE.search(title or ""):
        return True
    words = sum(len(s.words) for s in sentences)
    return duration > 20 * 60 and words / max(1.0, duration / 60) < 70


def _norm(w: str) -> str:
    return re.sub(r"[^\w']", "", w.lower())


def hype_map(audio: AudioFeatures, sentences: list[Sentence], duration: float) -> np.ndarray:
    """Hype score per second, 0..1."""
    n = int(duration) + 1
    per = max(1, int(round(1.0 / audio.hop)))
    rms = audio.rms[: n * per]
    if rms.size < n * per:
        rms = np.pad(rms, (0, n * per - rms.size))
    sec = rms.reshape(n, per)
    loud = sec.max(axis=1)                        # loudest moment of each second
    base = np.convolve(loud, np.ones(120) / 120, mode="same")   # stream's own level, 2 min
    spike = np.clip(loud - base, 0, None)
    jump = np.clip(np.diff(loud, prepend=loud[0]), 0, None)

    talk = np.zeros(n)
    excl = np.zeros(n)
    for s in sentences:
        for w in s.words:
            i = int(w.start)
            if 0 <= i < n:
                talk[i] += 1
                t = _norm(w.text)
                if t in HYPE_WORDS or w.text.strip().endswith(("!", "?")):
                    excl[i] += 1
    burst = np.convolve(talk, np.ones(4), mode="same")
    excl = np.convolve(excl, np.ones(4), mode="same")

    def z(x: np.ndarray) -> np.ndarray:
        sd = float(np.std(x)) or 1.0
        return np.clip((x - float(np.mean(x))) / sd, 0, 4) / 4

    score = 0.38 * z(spike) + 0.17 * z(jump) + 0.25 * z(burst) + 0.20 * z(excl)
    return np.convolve(score, np.ones(5) / 5, mode="same")


def pick_peaks(score: np.ndarray, k: int, min_gap: int = 45, skip_edges: int = 30) -> list[int]:
    order = np.argsort(score)[::-1]
    out: list[int] = []
    for i in order:
        i = int(i)
        if i < skip_edges or i > len(score) - skip_edges:     # intros / goodbyes
            continue
        if all(abs(i - j) >= min_gap for j in out):
            out.append(i)
        if len(out) >= k:
            break
    return sorted(out)


def _words_between(sentences: list[Sentence], a: float, b: float) -> list[Word]:
    return [w for s in sentences if s.end >= a and s.start <= b for w in s.words if a <= w.start <= b]


def _context(sentences: list[Sentence], score: np.ndarray, a: float, b: float) -> str:
    lines = []
    for s in sentences:
        if s.end < a or s.start > b:
            continue
        i = min(len(score) - 1, max(0, int(s.start)))
        level = score[i]
        mark = " [HYPE]" if level > 0.45 else (" [hot]" if level > 0.3 else "")
        lines.append(f"{s.start:7.1f}{mark} | {s.text.strip()}")
    gaps = "(long stretches without speech are gameplay)" if len(lines) < (b - a) / 6 else ""
    return "\n".join(lines) + (f"\n{gaps}" if gaps else "")


LANG = {"hi": "Hinglish (Hindi written in Latin letters, the way Indian gaming creators write)",
        "en": "English"}

AGENT_SYSTEM = """You are a gaming highlight editor for YouTube Shorts, Reels and TikTok.
You review candidate moments from a long game stream. Each candidate has a timestamped
transcript around a sound/speech spike; [HYPE] and [hot] mark loud or intense seconds.

For EACH candidate decide:
- keep: true only if a viewer would watch it: a funny moment, a clutch play, a fail,
  rage, a big reaction, a good story, or pure hype. Reject chatter, ads, setup talk, reading
  donations/super chats, and intros or goodbyes.
- type: one of funny, clutch, fail, rage, reaction, story, hype
- start, end (seconds): begin on the setup (what leads up to it), end right after the
  reaction. 12-60 s. Never start mid-sentence if you can avoid it.
- what: one sentence, what happens
- title: max 8 words
- banner: on-screen line, max 12 words: who + what happens + the game, ONE key word in CAPS
- scores 0-100: hook, retention, emotion, value, quotability
- story: 2-4 English words naming the event (e.g. "copyright strike", "first car chase").
  Candidates about the SAME event get the SAME story label; keep only the best one of them.
Write title and banner in {lang}. Never use Chinese or other unrelated scripts.
Return STRICT JSON only:
{{"clips":[{{"id":0,"keep":true,"type":"funny","start":0.0,"end":0.0,"what":"","title":"",
"banner":"","story":"","hashtags":["#tag"],"scores":{{"hook":0,"retention":0,"emotion":0,"value":0,"quotability":0}}}}]}}"""

AGENT_SCHEMA = {
    "type": "object",
    "properties": {"clips": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "id": {"type": "integer"}, "keep": {"type": "boolean"},
            "type": {"type": "string", "enum": TYPES},
            "start": {"type": "number"}, "end": {"type": "number"},
            "what": {"type": "string"}, "title": {"type": "string"}, "banner": {"type": "string"},
            "story": {"type": "string"},
            "hashtags": {"type": "array", "items": {"type": "string"}},
            "scores": {"type": "object", "properties": {k: {"type": "integer"} for k in
                       ("hook", "retention", "emotion", "value", "quotability")}},
        },
        "required": ["id", "keep", "type", "start", "end", "what", "title", "banner", "story", "scores"]}}},
    "required": ["clips"],
}


def _snap(sentences: list[Sentence], a: float, b: float, lo: float, hi: float) -> tuple[float, float]:
    """Nudge cut points off mid-word: start before a word, end after one."""
    words = [w for s in sentences for w in s.words if lo - 5 <= w.start <= hi + 5]
    for w in words:
        if w.start <= a < w.end:
            a = w.start - 0.15
            break
    for w in words:
        if w.start < b <= w.end:
            b = w.end + 0.25
            break
    return max(lo, a), min(hi, b)


OVERVIEW_SYSTEM = """You are the producer of a gaming channel. You get the transcript of a whole
live stream (timestamps in seconds; [HYPE]/[hot] mark loud, intense seconds). Understand it:
which game(s), who is streaming, what happens over the stream, and where the moments worth a
Short are, including ones that are good because of what is said (a story, a roast, a reveal),
not only loud ones. Ignore intros, ads, donation reading and goodbyes.
Write in {lang}. Never use Chinese or other unrelated scripts.
Return STRICT JSON only:
{{"game":"","streamer":"","summary":"two sentences","moments":[{{"t":0,"why":""}}]}}"""

OVERVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "game": {"type": "string"}, "streamer": {"type": "string"}, "summary": {"type": "string"},
        "moments": {"type": "array", "items": {"type": "object", "properties": {
            "t": {"type": "number"}, "why": {"type": "string"}}, "required": ["t", "why"]}},
    },
    "required": ["game", "streamer", "summary", "moments"],
}


def _overview_text(sentences: list[Sentence], score: np.ndarray, budget: int) -> str:
    lines = []
    for s in sentences:
        i = min(len(score) - 1, max(0, int(s.start)))
        mark = " [HYPE]" if score[i] > 0.45 else (" [hot]" if score[i] > 0.3 else "")
        lines.append(f"{s.start:.0f}{mark} | {s.text.strip()[:160]}")
    text = "\n".join(lines)
    if len(text) <= budget:
        return text
    # too long for the model: keep every hot line, thin out the calm ones evenly
    keep = [ln for ln in lines if "[" in ln.split("|")[0]]
    calm = [ln for ln in lines if "[" not in ln.split("|")[0]]
    room = max(0, budget - sum(len(x) + 1 for x in keep))
    step = max(1, int(np.ceil(sum(len(x) + 1 for x in calm) / max(1, room))))
    chosen = set(keep) | set(calm[::step])
    return "\n".join(ln for ln in lines if ln in chosen)[:budget]


def overview(sentences: list[Sentence], score: np.ndarray, llm: LLM, context: str, lang: str,
             log=None) -> dict:
    """Step 0: read the whole stream once (game, streamer, storyline, the AI's own moments)."""
    budget = max(5000, llm.context_chars - 3000)
    user = ((f"STREAM TITLE: {context}\n\n" if context else "")
            + "TRANSCRIPT:\n" + _overview_text(sentences, score, budget)
            + "\n\nList up to 12 moments.")
    if log:
        log("gaming agent: reading the whole stream")
    try:
        data = llm.json_completion(OVERVIEW_SYSTEM.format(lang=lang), user, schema=OVERVIEW_SCHEMA,
                                   max_tokens=2000, temperature=0.3)
    except LLMError as e:
        if log:
            log(f"gaming agent: overview skipped ({str(e)[:120]})")
        return {}
    if log and data.get("summary"):
        log(f"gaming agent: {str(data.get('game') or '?')[:40]} - {str(data['summary'])[:160]}")
    return data


def _fallback_banner(r: dict, title: str, ov: dict) -> str:
    what = str(r.get("what") or "").strip()
    base = title or what
    who = str(ov.get("streamer") or "").strip()
    if base and who and who.lower() not in base.lower():
        base = f"{who}: {base}"
    words = base.split()
    return " ".join(words[:12])


STOP = {"the", "and", "his", "her", "with", "for", "from", "that", "this", "into", "after", "while",
        "live", "stream", "streamer", "game", "gameplay", "bhai", "hai", "kar", "raha", "gaya"}


def _story_words(p: dict) -> set[str]:
    text = f"{p.get('story', '')} {p.get('banner', '')} {p.get('summary', '')}".lower()
    return {w for w in re.findall(r"[a-z]{4,}", text) if w not in STOP}


def same_story(p: dict, q: dict, common: set[str]) -> bool:
    """Two picks about one event (the agent's label, or most of the same words)."""
    if p.get("story") and p.get("story") == q.get("story"):
        return True
    a, b = _story_words(p) - common, _story_words(q) - common
    if len(a) < 3 or len(b) < 3:
        return False
    return len(a & b) / len(a | b) > 0.4


def common_words(picks: list[dict]) -> set[str]:
    """Words most picks share (streamer, game name): no signal for telling events apart."""
    from collections import Counter

    cnt = Counter(w for p in picks for w in _story_words(p))
    return {w for w, n in cnt.items() if n >= max(3, len(picks) * 0.4)}


def agent_select(sentences: list[Sentence], audio: AudioFeatures, duration: float, k: int,
                 llm: LLM | None, context: str = "", language: str = "", log=None,
                 min_len: float = 12.0, max_len: float = 60.0) -> list[dict]:
    """Returns picks shaped like highlights.llm_select output (plus 'gaming': True)."""
    score = hype_map(audio, sentences, duration)
    peaks = pick_peaks(score, max(k * 4, 12))
    if log:
        log(f"gaming agent: {len(peaks)} hype peaks found in {duration / 60:.0f} min")
    use_llm = llm is not None and llm.available
    lang = LANG.get((language or "en")[:2], f"the transcript's language ({language})")

    ov: dict = overview(sentences, score, llm, context, lang, log) if use_llm else {}
    times = [(float(t), "peak") for t in peaks]
    for m in (ov.get("moments") or [])[:12]:
        try:
            t = float(m.get("t"))
        except (TypeError, ValueError, AttributeError):
            continue
        if 30 <= t <= duration - 20 and all(abs(t - x) >= 40 for x, _ in times):
            times.append((t, "story"))
    if log and len(times) > len(peaks):
        log(f"gaming agent: +{len(times) - len(peaks)} moments the AI found in the talk")
    times.sort()

    cands = []
    for i, (t, src) in enumerate(times):
        lo, hi = max(0.0, t - 40.0), min(duration, t + 25.0)
        if src == "story":                       # talk moments: the payoff is after the line
            lo, hi = max(0.0, t - 25.0), min(duration, t + 40.0)
        ti = min(len(score) - 1, int(t))
        cands.append({"id": i, "peak": t, "lo": lo, "hi": hi, "src": src,
                      "a": max(lo, t - 18.0), "b": min(hi, t + 12.0),
                      "hype": float(max(score[ti], 0.3 if src == "story" else 0.0))})

    reviewed: dict[int, dict] = {}
    if use_llm:
        system = AGENT_SYSTEM.format(lang=lang)
        stream = ""
        if ov:
            stream = (f"STREAM: game: {ov.get('game', '')}; streamer: {ov.get('streamer', '')}\n"
                      f"WHAT HAPPENS: {ov.get('summary', '')}\n"
                      "Use the game and streamer names in banners when it helps.\n\n")
        batch = 4 if llm.provider == "ollama" else 10

        def review(group: list[dict], note: str = "") -> bool:
            blocks = [f"CANDIDATE {c['id']} ({'talk moment' if c['src'] == 'story' else 'sound peak'} "
                      f"at {c['peak']:.0f}s, window {c['lo']:.0f}-{c['hi']:.0f}s)\n"
                      + _context(sentences, score, c["lo"], c["hi"]) for c in group]
            ids = ", ".join(str(c["id"]) for c in group)
            user = ((f"STREAM TITLE: {context}\n" if context else "") + stream + "\n\n".join(blocks)
                    + f"\n\nReview every candidate ({ids}): one entry per id, even when keep is false.{note}")
            try:
                data = llm.json_completion(system, user, schema=AGENT_SCHEMA, max_tokens=3000,
                                           temperature=0.3)
            except LLMError as e:
                if log:
                    log(f"gaming agent: review failed ({str(e)[:120]})")
                return False
            for item in data.get("clips", []) or []:
                try:
                    i = int(item["id"])
                except (KeyError, TypeError, ValueError):
                    continue
                if any(c["id"] == i for c in group):
                    reviewed[i] = item
            return True

        failed = 0
        for bi in range(0, len(cands), batch):
            chunk = cands[bi : bi + batch]
            if log:
                log(f"gaming agent reviewing candidates {bi + 1}-{bi + len(chunk)} of {len(cands)}")
            if not review(chunk):
                failed += 1
                if failed >= 2 and not reviewed:
                    break                          # the model is down, not just one bad batch
        missing = [c for c in cands if c["id"] not in reviewed]
        if reviewed and missing:
            if log:
                log(f"gaming agent: asking again about {len(missing)} skipped candidates")
            for bi in range(0, len(missing), batch):
                review(missing[bi : bi + batch], " Earlier you skipped these.")

    picks = []
    for c in cands:
        r = reviewed.get(c["id"])
        if r is not None and not r.get("keep", True):
            continue
        if r is None and reviewed:
            continue                     # the agent never vouched for it
        a, b = c["a"], c["b"]
        if r is not None:
            try:
                ra, rb = float(r.get("start", a)), float(r.get("end", b))
                if c["lo"] - 5 <= ra < rb <= c["hi"] + 5:
                    a, b = ra, rb
            except (TypeError, ValueError):
                pass
        if b - a < min_len:
            pad = (min_len - (b - a)) / 2
            a, b = max(c["lo"], a - pad), min(c["hi"], b + pad)
        if b - a > max_len:
            b = a + max_len
        a, b = _snap(sentences, a, b, c["lo"], c["hi"])
        words = _words_between(sentences, a, b)
        text = " ".join(w.text for w in words) or "(gameplay)"
        from .highlights import _clean_tags, _normalize_scores, make_title
        title = str((r or {}).get("title") or (make_title(text) if words else "Gameplay moment")).strip()
        banner = str((r or {}).get("banner") or "").strip()
        if r is not None and not banner:
            banner = _fallback_banner(r, title, ov)
        picks.append({
            "start_sid": -1, "end_sid": -1, "start": round(a, 3), "end": round(b, 3), "words": words,
            "text": text, "title": title[:80],
            "banner": banner[:110],
            "hook": text[:160], "summary": str((r or {}).get("what") or "").strip()[:300],
            "why": f"{(r or {}).get('type', 'hype')} moment, hype {c['hype']:.2f}",
            "hashtags": _clean_tags((r or {}).get("hashtags")),
            "scores": _normalize_scores((r or {}).get("scores") or {}),
            "risks": [], "gaming": True, "hype": c["hype"], "kind": (r or {}).get("type", "hype"),
            "story": str((r or {}).get("story") or "").strip().lower(),
        })
    if log:
        log(f"gaming agent kept {len(picks)} of {len(cands)} candidates")
    return picks
