from __future__ import annotations

import re

import numpy as np
from dataclasses import dataclass

from .audio import AudioFeatures
from .boundaries import (can_stretch, focus_hits, focus_terms, is_banter, is_punctuated,
                         mid_sentence, opening_issue, padded_edges, review, stretch_to)
from .config import settings
from .edit import hashtags_for
from .llm import LLM, LLMError
from .models import Clip, Score, Sentence
from .score import grade, score_span
from .text import sentence_span, tokens


@dataclass
class Candidate:
    start_sid: int
    end_sid: int
    start: float
    end: float
    score: int = 0

    @property
    def duration(self) -> float:
        return self.end - self.start


CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")

def build_candidates(sentences: list[Sentence], min_len: float, max_len: float) -> list[Candidate]:
    """Sentence-aligned windows across a few target durations inside [min_len, max_len]."""
    out: dict[tuple[int, int], Candidate] = {}
    n = len(sentences)
    if n == 0:
        return []
    span = sentences[-1].end - sentences[0].start
    lo, hi = min_len, max(min_len + 1, max_len)
    # many natural lengths; the scorer (not a setting) decides which one a moment deserves
    targets = [t for t in (12, 18, 25, 32, 40, 50, 62, 75, 88) if lo <= t <= hi]
    targets = targets or [round((lo + hi) / 2)]
    targets = sorted({min(t, max(span, lo)) for t in targets})
    punctuated = is_punctuated(sentences)

    for i in range(n):
        if i > 0 and len(tokens(sentences[i].text)) < 3:
            continue
        if mid_sentence(sentences, i, punctuated):
            continue          # a window never opens in the middle of a sentence
        for target in targets:
            j = i
            while j + 1 < n and (sentences[j].end - sentences[i].start) < target:
                j += 1
            dur = sentences[j].end - sentences[i].start
            if dur < min_len or dur > max_len:
                continue
            if not sentences[j].text.rstrip().endswith((".", "!", "?", "…")) and j < n - 1:
                continue
            out.setdefault((i, j), Candidate(i, j, sentences[i].start, sentences[j].end))
    return list(out.values())


def _iou(a: Candidate, b: Candidate) -> float:
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return inter / union if union > 0 else 0.0


def prefilter(cands: list[Candidate], sentences: list[Sentence], audio: AudioFeatures,
              keep: int = 45) -> list[Candidate]:
    for c in cands:
        txt = " ".join(s.text for s in sentences[c.start_sid : c.end_sid + 1])
        _, _, words = sentence_span(sentences, c.start_sid, c.end_sid)
        c.score = score_span(txt, words, c.duration, c.start, c.end, audio).total
    cands.sort(key=lambda c: c.score, reverse=True)
    kept: list[Candidate] = []
    for c in cands:
        if all(_iou(c, k) < 0.68 for k in kept):
            kept.append(c)
        if len(kept) >= keep:
            break
    return kept


# --------------------------------------------------------------------------
# LLM ranking
# --------------------------------------------------------------------------
def _transcript_block(sentences: list[Sentence], emap=None) -> str:
    """sentence_id|start|[mark] text - [HYPE] / [hot] where this video's energy peaks."""
    from .story import levels, mark

    lv = levels(emap) if emap is not None else (1.1, 1.1)
    return "\n".join(f"{s.id}|{s.start:6.1f}|{mark(emap, s.start, s.end, lv)}{s.text.strip()}"
                     for s in sentences)


def _windows(sentences: list[Sentence], max_chars: int, overlap: float) -> list[list[Sentence]]:
    """Split the transcript into LLM-sized windows that overlap by about one clip length."""
    out: list[list[Sentence]] = []
    i = 0
    n = len(sentences)
    while i < n:
        size = 0
        j = i
        while j < n and size + len(sentences[j].text) + 14 <= max_chars:
            size += len(sentences[j].text) + 14
            j += 1
        j = max(j, i + 1)
        out.append(sentences[i:j])
        if j >= n:
            break
        back = j - 1
        while back > i + 1 and sentences[j - 1].end - sentences[back].start < overlap:
            back -= 1
        i = max(i + 1, back)
    return out


SYSTEM = """You are the clip curator inside a video repurposing engine (like Opus Clip).
You read a timestamped transcript and choose the strongest standalone short-form clips.

Rules:
- A clip must be self-contained: a viewer with zero context understands and cares.
- Prefer a complete mini-story or argument: setup, tension or claim, specific detail, payoff.
- The FIRST sentence must work as a hook (question, bold claim, number, conflict, curiosity gap).
  Never start on a filler (um, like, well), a bare agreement (yeah, right, exactly), a
  conjunction that continues an earlier point ("But...", "And then...", "That's why..."),
  a back-reference ("as I said", "that point") or a pronoun for someone not yet named.
  If the moment needs the sentence before it, start on that sentence.
- END on the last sentence that lands the point. Never stop mid-thought, never on an
  unanswered question, never keep the opening words of the next topic.
- YOU decide each clip's length from the content, for TikTok / Reels / Shorts. The kind
  of moment sets the length:
    punchline, reaction, one-liner, hot take ........ 12-25 s
    tip, answer, insight, quick rant ................ 20-45 s
    story, argument, step-by-step that keeps tension  45-90 s
    a full story that holds tension all the way ....... up to 180 s (rare)
  Start on the hook (or the minimum setup it needs), end right after the payoff. Never pad
  with filler or the next topic to make it longer; never chop the idea to make it shorter.
  Lengths should differ clip to clip, because moments differ.
- Prefer emotional, specific, contrarian or highly actionable moments.
- Skip intros, sponsor reads, "like and subscribe", wrap-ups and goodbyes, and passages where
  the speech says the picture is broken ("we'll be right back", technical problems).
- Do not pick overlapping clips. Maximise variety across topics.

The "banner" is shown on screen for the whole clip so a scrolling viewer instantly knows
what it is about. Name the game, show or topic from the video title, say what happens,
make it intriguing. Examples: "Streamer finds a secret room in Minecraft and PANICS",
"Pro player explains why console aim is UNFAIR in Fragpunk".

Score each clip 0-100 on:
  hook        - does the first 3 seconds stop the scroll
  retention   - pacing, no rambling, keeps tension
  emotion     - energy, stakes, relatability
  value       - insight, specificity, usefulness
  quotability - memorable, shareable, loopable

Return STRICT JSON only:
{"clips":[{"start_sid":<int>,"end_sid":<int>,"title":"<=8 words, same language as transcript",
"hook":"the literal opening words","summary":"1 sentence","why":"why it works, 1 sentence",
"banner":"on-screen context line, max 12 words: who + what happens + the game/topic, curiosity, ONE key word in CAPS",
"hashtags":["#tag"],
"scores":{"hook":0-100,"retention":0-100,"emotion":0-100,"value":0-100,"quotability":0-100},
"risks":["short risk"]}]}"""

_SCORE_KEYS = ("hook", "retention", "emotion", "value", "quotability")
SCHEMA = {
    "type": "object",
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_sid": {"type": "integer"},
                    "end_sid": {"type": "integer"},
                    "title": {"type": "string"},
                    "banner": {"type": "string"},
                    "hook": {"type": "string"},
                    "summary": {"type": "string"},
                    "why": {"type": "string"},
                    "hashtags": {"type": "array", "items": {"type": "string"}},
                    "scores": {
                        "type": "object",
                        "properties": {k: {"type": "integer"} for k in _SCORE_KEYS},
                        "required": list(_SCORE_KEYS),
                    },
                    "risks": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["start_sid", "end_sid", "title", "banner", "hook", "why", "hashtags", "scores"],
            },
        }
    },
    "required": ["clips"],
}


def _scale_for(values) -> float:
    """Models answer on 0-1, 0-10 or 0-100 depending on mood. Infer the scale from the
    whole answer, not one number: a harsh 8/100 must not turn into 80/100."""
    nums = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not nums:
        return 1.0
    mx = max(nums)
    if mx <= 1.5:
        return 100.0
    if mx <= 10.0:
        return 10.0
    return 1.0


def _raw_scores(raw: dict) -> dict:
    vals: dict[str, float] = {}
    for k in _SCORE_KEYS:
        v = raw.get(k, raw.get(k.title()))
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            vals[k] = float(v)
    return vals


def _normalize_scores(raw: dict, scale: float | None = None) -> dict:
    """Sub-scores to 0-100. Pass ``scale`` (from _scale_for over the whole answer) when
    known; without it the scale is guessed from this clip's own numbers."""
    vals = _raw_scores(raw)
    if not vals:
        return {}
    s = _scale_for(vals.values()) if scale is None else scale
    return {k: max(0.0, min(100.0, v * s)) for k, v in vals.items()}


def _clean_tags(tags) -> list[str]:
    out: list[str] = []
    for t in tags or []:
        t = "#" + str(t).strip().lstrip("#").replace(" ", "")
        if len(t) > 1 and t.lower() not in (x.lower() for x in out):
            out.append(t)
    return out[:6]


def _language_rule(language: str) -> str:
    lang = (language or "en")[:2]
    if lang == "hi":
        return ("Write title, banner and hashtags in Hinglish (Hindi in Latin letters, as Indian "
                "creators write). Never use Chinese or other unrelated scripts.")
    if lang == "en":
        return "Write title, banner and hashtags in English."
    return (f"Write title, banner and hashtags in the transcript's language ({lang}). "
            "Never use Chinese or other unrelated scripts unless that is the transcript's language.")


def _fatal(e: Exception) -> bool:
    msg = str(e)
    return any(x in msg for x in ("rejected", "not running", "not pulled", "disabled"))


def llm_select(sentences: list[Sentence], cands: list[Candidate], llm: LLM, topic: str,
               k: int, log, min_len: float, max_len: float, context: str = "",
               language: str = "", focus: str = "", target: float | None = None,
               emap=None, story: dict | None = None) -> list[dict]:
    """Ask the LLM window by window, then turn its sentence ranges into picks.

    ``focus``: the kind of moment the user asked for; relevance to it beats general virality.
    ``target``: the length the user asked for (the longest a clip may be); the AI aims at it."""
    from .story import brief_text, parts_near

    brief = brief_text(story)
    room = 2500 + (len(brief) + 900 if story else 0)
    windows = _windows(sentences, max(4000, llm.context_chars - room), overlap=max_len)
    total = max(1.0, sentences[-1].end - sentences[0].start)
    n = len(sentences)
    picks: list[dict] = []
    failures = 0
    for wi, win in enumerate(windows, 1):
        span = win[-1].end - win[0].start
        per = max(2, round(k * span / total) + 1) if len(windows) > 1 else k + 2
        lo, hi = win[0].id, win[-1].id
        hints = "\n".join(
            [f"- sentences {c.start_sid}-{c.end_sid} ({c.duration:.0f}s, heuristic {c.score})"
             for c in cands if lo <= c.start_sid and c.end_sid <= hi][:12]
        )
        user = (
            (f"NICHE / TOPIC: {topic}\n\n" if topic else "")
            + (f"VIDEO TITLE (use it for the game / topic / who is speaking): {context}\n" if context else "")
            + (f"Clip length: the user asked for clips up to {target:.0f} s. Aim close to it "
               f"(never longer); {min_len:.0f} s is the shortest.\n" if target else
               f"Clip length: your call per clip (hard limits {min_len:.0f}-{max_len:.0f} s).\n")
            + (f"THE USER ASKED ONLY FOR: {focus}\nPick only clips that match this request; "
               "matching it matters more than general virality. If nothing in this part matches, "
               "return {\"clips\": []}.\n" if focus else "")
            + _language_rule(language) + "\n\n"
            + (f"WHOLE VIDEO (for context - who these people are, what is going on):\n{brief}\n\n"
               if brief else "")
            + (f"WHAT HAPPENS AROUND THIS PART:\n{parts_near(story, win[0].start, win[-1].end, pad=60)[:900]}\n\n"
               if story else "")
            + "TRANSCRIPT (sentence_id|start_seconds|text). [HYPE] / [hot] mark where the energy in "
              "the room peaks (loud, fast, reactions, laughter): the moments people react to are "
              "usually there, but a clip must still start on its setup and end after the payoff.\n"
            + _transcript_block(win, emap)
            + (f"\n\nHigh-scoring windows from a cheap pre-filter (hints, not limits):\n{hints}"
               if hints else "")
            + f"\n\nPick the best {per} clips (fewer only if this part truly has fewer moments "
            "that stand alone). start_sid/end_sid must be sentence ids above."
        )
        if log:
            log(f"AI ({llm.label}) reading part {wi}/{len(windows)}")
        try:
            data = llm.json_completion(SYSTEM, user, schema=SCHEMA, max_tokens=4000, temperature=0.15)
        except LLMError as e:
            failures += 1
            if log:
                log(f"LLM failed on part {wi}: {str(e)[:160]}")
            if failures >= 2 or len(windows) == 1 or _fatal(e):
                break
            continue
        items = [x for x in (data.get("clips", []) or []) if isinstance(x, dict)]
        scale = _scale_for([v for x in items for v in _raw_scores(x.get("scores") or {}).values()])
        for item in items:
            try:
                a = int(item["start_sid"])
                b = int(item["end_sid"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (0 <= a < n and 0 <= b < n):
                continue
            if b < a:
                a, b = b, a
            start, end, words = sentence_span(sentences, a, b)
            # keep the AI's own length. Only the platform limits apply: longer than a short
            # can be is trimmed; a little short is kept here and lengthened by the boundary
            # review while the speech runs on; much shorter is no clip.
            while (end - start) > max_len + 3 and b > a:
                b -= 1
                start, end, words = sentence_span(sentences, a, b)
            if not can_stretch(end - start, min_len) or end - start > max_len + 3:
                continue
            picks.append({
                "start_sid": a, "end_sid": b, "start": start, "end": end, "words": words,
                "text": " ".join(s.text for s in sentences[a : b + 1]),
                "title": str(item.get("title") or "").strip().strip('"')[:80],
                "banner": str(item.get("banner") or "").strip().strip('"')[:110],
                "hook": str(item.get("hook") or "").strip()[:200],
                "summary": str(item.get("summary") or "").strip()[:400],
                "why": str(item.get("why") or "").strip()[:400],
                "hashtags": _clean_tags(item.get("hashtags")),
                "scores": _normalize_scores(item.get("scores") or {}, scale),
                "risks": [str(r) for r in (item.get("risks") or [])][:3],
            })
    return picks


# --------------------------------------------------------------------------
# rule-based fallback
# --------------------------------------------------------------------------
def _clean_sentence(s: str) -> str:
    s = s.strip()
    for lead in ("so ", "and ", "but ", "like ", "you know ", "okay ", "well "):
        if s.lower().startswith(lead):
            s = s[len(lead):]
    return s.strip()


STOP_END = {
    "to", "of", "and", "but", "the", "a", "an", "in", "on", "for", "with", "that",
    "this", "is", "was", "are", "were", "it", "my", "your", "at", "as", "by", "from",
    "or", "if", "so", "because", "when", "what", "than", "then", "about", "into",
}


def make_title(text: str) -> str:
    """Extractive fallback title (used when no LLM is available)."""
    sents = [x.strip() for x in text.replace("!", ".").replace("?", ".").split(".") if x.strip()]
    if not sents:
        return "Clip"
    strongest = ["never", "always", "nobody", "secret", "mistake", "money", "best", "worst",
                 "everyone", "why", "how", "stop", "truth"]
    short = [s for s in sents if 3 <= len(tokens(s)) <= 9]
    pool = short or [s for s in sents if len(tokens(s)) >= 3] or sents[:1]
    pool = [s for s in pool if any(t not in STOP_END for t in tokens(s))] or sents[:1]

    def rank(s: str) -> float:
        toks = tokens(s)
        r = 0.0
        if any(ch.isdigit() for ch in s):
            r += 3
        r += 2 * sum(1 for w in toks if w in strongest)
        if "you" in toks or "your" in toks:
            r += 1.5
        if "?" in s:
            r += 1.0
        r -= 0.35 * pool.index(s)
        return r

    best = max(pool, key=rank)
    words = _clean_sentence(best).rstrip(" .,!?").split()[:9]
    while len(words) > 2 and words[-1].lower().strip(".,!?") in STOP_END:
        words.pop()
    out = " ".join(words).strip(" ,.-")
    return (out[:1].upper() + out[1:])[:60] if out else "Clip"


def heuristic_select(cands: list[Candidate], sentences: list[Sentence], k: int) -> list[dict]:
    picks: list[dict] = []
    for c in cands:
        if any(min(c.end, p["end"]) - max(c.start, p["start"])
               > 0.15 * min(c.duration, p["end"] - p["start"]) for p in picks):
            continue
        start, end, words = sentence_span(sentences, c.start_sid, c.end_sid)
        text = " ".join(s.text for s in sentences[c.start_sid : c.end_sid + 1])
        picks.append({
            "start_sid": c.start_sid, "end_sid": c.end_sid, "start": start, "end": end,
            "words": words, "text": text, "title": make_title(text),
            "hook": _clean_sentence(sentences[c.start_sid].text)[:160],
            "summary": "", "why": "", "hashtags": hashtags_for(text), "scores": {}, "risks": [],
        })
        if len(picks) >= k:
            break
    return picks


# --------------------------------------------------------------------------
# virality: hook tightening + a harsh comparative AI judge
# --------------------------------------------------------------------------
def _tighten_hook(p: dict, sc: Score, sentences: list[Sentence], audio: AudioFeatures,
                  weights: dict | None, min_len: float,
                  punctuated: bool = True) -> tuple[dict, Score]:
    """If the first sentence is a weak opener, try starting 1-2 sentences later.
    Keep the variant whose first seconds hook harder without losing overall score."""
    a, b = p["start_sid"], p["end_sid"]
    best_p, best_sc = p, sc
    for a2 in (a + 1, a + 2):
        if a2 > b:
            break
        s, e, words = sentence_span(sentences, a2, b)
        if e - s < min_len:
            break
        if (opening_issue(sentences[a2].text) or is_banter(sentences[a2].text)
                or mid_sentence(sentences, a2, punctuated)):
            continue          # a later start that needs the lines we would cut is no hook
        text = " ".join(x.text for x in sentences[a2 : b + 1])
        sc2 = score_span(text, words, e - s, s, e, audio, weights=weights,
                         llm_subscores=p.get("scores") or None)
        if sc2.hook >= best_sc.hook + 10 and sc2.total >= best_sc.total:
            best_sc = sc2
            best_p = {**p, "start_sid": a2, "start": s, "words": words, "text": text,
                      "hook": _clean_sentence(sentences[a2].text)[:160], "trimmed": a2 - a}
    if best_p is not p:
        best_sc.reasons = (["Start tightened to the strongest hook"] + best_sc.reasons)[:4]
    return best_p, best_sc


JUDGE_SYSTEM = """You are a ruthless short-form video editor. You know what actually gets views on
TikTok, YouTube Shorts and Instagram Reels:
- the first line stops the scroll (surprise, conflict, bold claim, question, number)
- one clear idea that needs zero context
- emotional stakes, a twist, or a payoff worth waiting for
- a line people want to quote, rewatch, argue with or send to a friend
Most clips are NOT viral. Be harsh and compare the clips against each other: only the
genuinely strong ones get high numbers, filler and rambling get low numbers.
Return STRICT JSON only:
{"clips":[{"id":<int>,"viral":0-100,"verdict":"post"|"skip","reason":"one short sentence"}]}"""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"clips": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "integer"}, "viral": {"type": "integer"},
                       "verdict": {"type": "string", "enum": ["post", "skip"]},
                       "reason": {"type": "string"}},
        "required": ["id", "viral", "verdict", "reason"]}}},
    "required": ["clips"],
}


LABEL = re.compile(r"\s*[\(\[]\s*[A-Za-z]{1,12}\s*[:=]\s*\d{1,3}\s*[\)\]]")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def romanize(clips: list[Clip], llm: LLM | None, log=None) -> None:
    """Hindi videos: titles and banners in Devanagari become Hinglish (Latin letters), the way
    Indian creators write on Shorts. One AI call for all of them; untouched if it fails."""
    if llm is None or not llm.available:
        return
    slots = [(c, f) for c in clips for f in ("title", "banner") if DEVANAGARI.search(getattr(c, f) or "")]
    if not slots:
        return
    lines = [getattr(c, f) for c, f in slots]
    user = ("Rewrite each line in Hinglish: Hindi words in Latin letters, English words stay "
            "English, same meaning, same CAPS emphasis. Return JSON {\"lines\": [...]} in the same "
            "order.\n\n" + "\n".join(f"{i}. {t}" for i, t in enumerate(lines)))
    try:
        data = llm.json_completion("You transliterate Hindi to Hinglish. Reply with JSON only.", user,
                                   schema={"type": "object", "properties": {"lines": {
                                       "type": "array", "items": {"type": "string"}}}, "required": ["lines"]},
                                   max_tokens=1200, temperature=0.1)
    except LLMError as e:
        if log:
            log(f"Hinglish rewrite skipped: {str(e)[:100]}")
        return
    out = data.get("lines") or []
    fixed = 0
    for (c, f), new in zip(slots, out):
        new = str(new or "").strip()
        if new and not DEVANAGARI.search(new) and not CJK.search(new):
            setattr(c, f, new[:110])
            fixed += 1
    if log and fixed:
        log(f"rewrote {fixed} Devanagari titles/banners in Hinglish")


def auto_intro(p: dict, audio: AudioFeatures, length: float = 3.2) -> list[float]:
    """Hook teaser for a clip: the loudest ~3 s after its opening, cut on word edges."""
    a, b = float(p["start"]), float(p["end"])
    words = sorted(p.get("words") or [], key=lambda w: w.start)
    if b - a < 12 or not words:
        return []
    per = max(1, int(round(1.0 / audio.hop)))
    best, best_t = -1.0, None
    t = a + 2.0
    while t + length <= b - 0.5:
        i0, i1 = int(t / audio.hop), int((t + length) / audio.hop)
        seg = audio.rms[i0:i1]
        if seg.size >= per:
            v = float(np.mean(seg)) + 0.5 * float(np.max(seg))
            if v > best:
                best, best_t = v, t
        t += 0.5
    if best_t is None:
        return []
    starts = [w for w in words if best_t - 1.0 <= w.start <= best_t + 0.6]
    h0 = (starts[0].start - 0.05) if starts else best_t
    ends = [w for w in words if h0 + 1.8 <= w.end <= h0 + 4.2]
    h1 = (ends[-1].end + 0.12) if ends else h0 + length
    return [round(max(a, h0), 3), round(min(b, h1), 3)]


def viral_judge(shortlist: list[tuple[dict, Score]], llm: LLM, topic: str, log=None,
                focus: str = "") -> None:
    """Second AI pass: rank the shortlist side by side and blend that into each score."""
    blocks = []
    per = int(max(200, min(900, (llm.context_chars - 3000) / max(1, len(shortlist)))))
    for i, (p, _) in enumerate(shortlist):
        text = p["text"].strip()
        if len(text) > per:
            text = text[:per] + " ..."
        what = f"WHAT HAPPENS: {p['summary']} | " if p.get("gaming") and p.get("summary") else ""
        blocks.append(f"[{i}] ({p['end'] - p['start']:.0f}s) {what}{text}")
    user = ((f"Channel niche: {topic}\n\n" if topic else "")
            + (f"The user asked only for: {focus}\nA clip that does not match that gets a low "
               "number and verdict skip, however good it is otherwise.\n\n" if focus else "")
            + "Candidate clips (id, length, transcript):\n\n" + "\n\n".join(blocks)
            + "\n\nJudge every clip.")
    if log:
        log(f"AI viral judge comparing {len(shortlist)} candidates")
    try:
        data = llm.json_completion(JUDGE_SYSTEM, user, schema=JUDGE_SCHEMA, max_tokens=2500,
                                   temperature=0.0)
    except LLMError as e:
        if log:
            log(f"viral judge skipped: {str(e)[:120]}")
        return
    verdicts = {}
    for item in data.get("clips", []) or []:
        try:
            verdicts[int(item["id"])] = item
        except (KeyError, TypeError, ValueError):
            continue
    # one scale for the whole answer: judging per number turned a harsh 8/100 into 80/100
    scale = _scale_for([v.get("viral") for v in verdicts.values()])
    nums: dict[int, float] = {}
    for i, _ in enumerate(shortlist):
        raw = (verdicts.get(i) or {}).get("viral")
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            nums[i] = max(0.0, min(100.0, float(raw) * scale))
    # Consistency: the judge decides the ORDER. Its overall mood (harsh on one run, generous
    # on the next) moved every score by the same amount and decided how many clips passed
    # the quality bar - the same stream gave 19 clips once and 10 the next time. Half of the
    # gap between its average and the measured average is taken out, so a really weak batch
    # still scores lower, but a mood swing no longer halves the clip count.
    shift = 0.0
    if len(nums) >= 3:
        mean_v = sum(nums.values()) / len(nums)
        mean_s = sum(shortlist[i][1].total for i in nums) / len(nums)
        shift = 0.5 * (mean_s - mean_v)
        if log and abs(shift) >= 3:
            log(f"viral judge: averaged {mean_v:.0f} vs {mean_s:.0f} measured, evened out by {shift:+.0f}")
    for i, (p, sc) in enumerate(shortlist):
        v = verdicts.get(i)
        if not v or i not in nums:
            continue                      # no number: leave this clip's score alone
        viral = nums[i]
        total = 0.6 * sc.total + 0.4 * max(0.0, min(100.0, viral + shift))
        reason = str(v.get("reason") or "").strip()[:160]
        if str(v.get("verdict")).lower() == "skip":
            total -= 6
            if reason:
                sc.risks = ([f"AI judge: {reason}"] + sc.risks)[:4]
        elif reason:
            sc.reasons = ([f"AI judge: {reason}"] + sc.reasons)[:4]
        sc.total = int(round(max(0.0, min(100.0, total))))
        sc.features["ai_viral"] = int(round(viral))
        sc.features["ai_verdict"] = str(v.get("verdict"))
        sc.source = sc.source + "+judge"


# --------------------------------------------------------------------------
def select(
    sentences: list[Sentence],
    audio: AudioFeatures,
    topic: str = "",
    k: int | None = None,
    weights: dict | None = None,
    log=None,
    min_len: float | None = None,
    max_len: float | None = None,
    llm: LLM | None = None,
    min_score: int = 0,
    context: str = "",
    gaming: bool = False,
    language: str = "",
    duration: float = 0.0,
    auto_count: bool = False,
    focus: str = "",
    target: float | None = None,
    span: tuple[float, float] | None = None,
    emap=None,
    story: dict | None = None,
) -> tuple[list[Clip], str]:
    """Pick, review and rank clips.

    ``auto_count``: k is a target, not a cap - a video with more strong moments keeps up to
    half as many again (clips that clear the quality bar by a clear margin), the way a human
    editor clips every moment a video deserves rather than a fixed number.
    ``focus``: only this kind of moment (the user's words). ``target``: the clip length the
    user asked for. ``span``: (start, end) of the part of the video to search."""
    k = k or settings.max_clips
    k_max = min(20, max(k, round(k * 1.5))) if auto_count else k
    min_len = settings.min_clip if min_len is None else min_len
    max_len = settings.max_clip if max_len is None else max_len
    if not sentences:
        return [], "rules"

    all_cands = build_candidates(sentences, min_len=min_len, max_len=max_len)
    if log:
        log(f"{len(all_cands)} candidate windows")
    cands = prefilter(all_cands, sentences, audio, keep=max(45, k * 6))  # also sorts all_cands

    picks: list[dict] = []
    model = "rules"
    if gaming:
        from .gaming import agent_select

        picks = agent_select(sentences, audio, duration or sentences[-1].end, k, llm, context=context,
                             language=language, log=log, min_len=max(10.0, min_len * 0.7),
                             max_len=min(max(60.0, target or 0.0), max_len), focus=focus, span=span)
        model = f"gaming agent ({llm.label})" if llm is not None and llm.available else "gaming agent (rules)"
        talky = sum(len(s.words) for s in sentences) / max(1.0, (duration or sentences[-1].end) / 60) >= 25
        if llm is not None and llm.available and talky and len(picks) < k * 2:
            # the streamer talks a lot: stories, rants and reveals make clips too, not only
            # the loud moments. Read the talk the normal way and pool both.
            if log:
                log(f"gaming agent found {len(picks)}; also reading the talk for story clips")
            talk = llm_select(sentences, cands, llm, topic, k, log, min_len, max_len, context=context,
                              language=language, focus=focus, target=target, emap=emap, story=story)
            picks = picks + [t for t in talk if not any(
                min(t["end"], q["end"]) - max(t["start"], q["start"]) > 0.3 * (t["end"] - t["start"])
                for q in picks)]
    elif llm is not None and llm.available:
        picks = llm_select(sentences, cands, llm, topic, k, log, min_len, max_len, context=context,
                           language=language, focus=focus, target=target, emap=emap, story=story)
        if picks:
            model = llm.label

    punctuated = is_punctuated(sentences)
    terms = focus_terms(focus) if focus else []

    def refined(items: list[dict]) -> list[dict]:
        """Boundary review: each talk pick's start, end and so its length follow its content."""
        out = []
        for p in items:
            if p.get("gaming") or p.get("start_sid", -1) < 0:
                out.append(p)
                continue
            rv = review(sentences, p["start_sid"], p["end_sid"], min_len, max_len, punctuated)
            if target:
                a2, b2 = stretch_to(sentences, rv.a, rv.b, target, punctuated)
                if (a2, b2) != (rv.a, rv.b):
                    rv.a, rv.b = a2, b2
                    rv.notes.append(f"Lengthened towards the {target:.0f}s you asked for")
            s, e, words = sentence_span(sentences, rv.a, rv.b)
            if e - s < min_len or e - s > max_len + 3:
                continue                   # could not grow into a full moment
            if (rv.a, rv.b) != (p["start_sid"], p["end_sid"]):
                hook = p.get("hook") if rv.a == p["start_sid"] else _clean_sentence(sentences[rv.a].text)[:160]
                p = {**p, "start_sid": rv.a, "end_sid": rv.b, "start": s, "end": e, "words": words,
                     "text": " ".join(x.text for x in sentences[rv.a : rv.b + 1]), "hook": hook}
            p["edge_notes"], p["edge_risk"] = rv.notes, rv.risk
            out.append(p)
        return out

    def scored(items: list[dict]) -> list[tuple[dict, Score]]:
        out = []
        for p in refined(items):
            sc = score_span(p["text"], p["words"], p["end"] - p["start"], p["start"], p["end"],
                            audio, weights=weights, llm_subscores=p.get("scores") or None)
            if p.get("edge_notes"):
                sc.reasons = ([p["edge_notes"][0]] + sc.reasons)[:4]
            if p.get("edge_risk"):
                sc.risks = ([p["edge_risk"]] + [r for r in sc.risks if "pronoun" not in r])[:4]
            if emap is not None:
                # where the room lights up: a clip holding one of this video's real peaks
                # (laughs, shouts, a shocked silence-then-reaction) beats a flat one
                from .story import part_rating, peak_rank

                pr = peak_rank(emap, p["start"], p["end"])
                sc.features["energy_peak"] = round(pr, 2)
                bonus = 0
                if not p.get("gaming"):        # streams already score their hype below
                    bonus = 10 if pr >= 0.97 else 6 if pr >= 0.92 else 2 if pr >= 0.8 else -4 if pr < 0.5 else 0
                pr_part = part_rating(story, (p["start"] + p["end"]) / 2)
                if pr_part is not None:
                    # the story log is built once and cached, so this is the same on every
                    # run: it anchors the ranking where the per-run AI review wobbles (two
                    # runs on one 4-hour stream agreed on only 4 of 14 shorts without it)
                    bonus += int(round((pr_part - 5.5) * (2.0 if p.get("gaming") else 1.2)))
                    sc.features["part_rating"] = pr_part
                sc.total = int(max(0, min(100, sc.total + bonus)))
                if bonus >= 6:
                    sc.reasons = ["One of the most intense moments of the video"] + sc.reasons[:3]
                elif bonus <= -4:
                    sc.risks = ["Low energy compared to the rest of the video"] + sc.risks[:3]
            if terms:
                # the user asked for one kind of moment: matching it beats general virality
                # (with an AI this is its job; the words are a check that works offline too)
                hits = focus_hits(f"{p['text']} {p.get('title', '')} {p.get('summary', '')}", terms)
                sc.features["focus_hits"] = hits
                if hits:
                    sc.total = int(min(100, sc.total + min(12, 4 + 4 * hits)))
                    sc.reasons = ["Matches what you asked for"] + sc.reasons[:3]
                elif model == "rules":
                    sc.total = int(max(0, sc.total - 15))
                    sc.risks = ["Does not mention what you asked for"] + sc.risks[:3]
            if p.get("gaming"):
                # streams are judged on hype as much as on words: sparse speech is normal there
                bonus = int(round(30 * (float(p.get("hype", 0.3)) - 0.3)))
                sc.total = int(max(0, min(100, sc.total + bonus)))
                if p.get("viral_score"):
                    # the agent's 5x20 rubric (hook, emotion, shareability, clarity, payoff)
                    sc.total = int(round(0.6 * float(p["viral_score"]) + 0.4 * sc.total))
                    sc.features["viral_rubric"] = round(float(p["viral_score"]), 1)
                sc.features["hype"] = round(float(p.get("hype", 0)), 2)
                sc.reasons = ([f"{str(p.get('kind', 'hype')).capitalize()} moment found by the gaming agent"]
                              + sc.reasons)[:4]
            out.append((p, sc))
        out.sort(key=lambda x: x[1].total, reverse=True)
        return out

    def tightened(items: list[tuple[dict, Score]]) -> list[tuple[dict, Score]]:
        out = [(p, sc) if p.get("gaming")
               else _tighten_hook(p, sc, sentences, audio, weights, min_len, punctuated)
               for p, sc in items]
        out.sort(key=lambda x: x[1].total, reverse=True)
        return out

    shortlist: list[tuple[dict, Score]] = []
    cap = max(k_max * 2, k_max + 4)   # judge more candidates than we will keep

    common: set[str] = set()
    if gaming:
        from .gaming import common_words, same_story

        common = common_words(picks)

    def take(items: list[tuple[dict, Score]]) -> None:
        for p, sc in items:
            if len(shortlist) >= cap:
                return
            if any(min(p["end"], q["end"]) - max(p["start"], q["start"])
                   > 0.15 * min(p["end"] - p["start"], q["end"] - q["start"]) for q, _ in shortlist):
                continue
            if p.get("gaming") and any(same_story(p, q, common) for q, _ in shortlist):
                if log:
                    log(f"gaming agent: skipped a repeat of '{p.get('story') or p.get('title')}'")
                continue
            shortlist.append((p, sc))

    take(tightened(scored(picks)))                                    # the AI's picks first
    if not gaming or len(shortlist) < k:                              # then rule-based
        take(tightened(scored(heuristic_select(all_cands, sentences, cap * 2))))

    if llm is not None and llm.available and len(shortlist) >= 2:
        viral_judge(shortlist, llm, topic, log, focus=focus)
    shortlist.sort(key=lambda x: x[1].total, reverse=True)

    passed = [x for x in shortlist if x[1].total >= min_score]
    if log and min_score:
        log(f"quality bar {min_score}: {len(passed)} of {len(shortlist)} candidates pass")
    # the bar trims weak clips but never leaves a long video with one or two: keep at least
    # half of what was asked when near-misses (within 12 points) exist
    floor = min(k, max(2, (k + 1) // 2))
    if len(passed) < floor:
        ids = {id(x) for x in passed}
        extra = [x for x in shortlist if id(x) not in ids and x[1].total >= min_score - 12
                 and not x[1].features.get("housekeeping")]   # never rescue an intro / plug
        added = extra[: floor - len(passed)]
        if added and log:
            log(f"quality bar: kept {len(added)} near-miss clips so you still get {len(passed) + len(added)}")
        passed = sorted(passed + added, key=lambda x: x[1].total, reverse=True)
    chosen = (passed or shortlist[:1])[:k]
    if auto_count and len(passed) > k:
        # more strong moments than the target: keep the ones that clear the bar clearly
        strong = max(min_score + 7, 70)
        bonus = [x for x in passed[k:k_max] if x[1].total >= strong]
        if bonus:
            chosen = chosen + bonus
            if log:
                log(f"clip count: +{len(bonus)} extra clips scoring {strong}+ (this video has more good moments)")

    cjk_ok = (language or "")[:2] in ("zh", "ja", "ko")

    def clean(text: str) -> str:
        """Small models sometimes drift into Chinese; drop such text unless the video is CJK."""
        text = LABEL.sub("", (text or "")).strip()       # leaked tags like "(H:76)"
        return "" if (not cjk_ok and CJK.search(text)) else text

    clips: list[Clip] = []
    for rank, (p, sc) in enumerate(chosen, start=1):
        sc.grade = grade(sc.total)
        title = clean(p.get("title")) or clean(make_title(p["text"])) or "Highlight"
        start, end = p["start"], p["end"]
        if not p.get("gaming") and p.get("start_sid", -1) >= 0:
            # a hair before the first word and after the last: no clipped consonant, the
            # last word lands fully (gaming picks are already padded by their own snap)
            start, end = padded_edges(sentences, p["start_sid"], p["end_sid"])
        clip = Clip(
            id=f"clip_{rank:02d}",
            start=round(start, 3),
            end=round(end, 3),
            title=title,
            hook=p.get("hook") or "",
            summary=clean(p.get("summary")) or " ".join(p["text"].split()[:38]),
            why=p.get("why") or (sc.reasons[0] if sc.reasons else ""),
            hashtags=[h for h in (p.get("hashtags") or hashtags_for(p["text"])) if clean(h)],
            banner=clean(p.get("banner")) or (title if p.get("gaming") else ""),
            intro=list(p.get("intro") or []) or auto_intro(p, audio),
            skips=list(p.get("skips") or []),
            score=sc,
            rank=rank,
        )
        if p.get("risks"):
            clip.score.risks = list(dict.fromkeys(list(sc.risks) + p["risks"]))[:4]
        clips.append(clip)
    if (language or "")[:2] == "hi":
        romanize(clips, llm, log)
    return clips, model
