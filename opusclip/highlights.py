from __future__ import annotations

from dataclasses import dataclass

from .audio import AudioFeatures
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

    for i in range(n):
        if i > 0 and len(tokens(sentences[i].text)) < 3:
            continue
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
def _transcript_block(sentences: list[Sentence]) -> str:
    return "\n".join(f"{s.id}|{s.start:6.1f}|{s.text.strip()}" for s in sentences)


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
- A clip must be self-contained: a viewer with zero context understands it.
- The FIRST sentence must work as a hook (question, bold claim, number, conflict, curiosity gap).
- Prefer a complete idea with a payoff; never cut mid-thought.
- YOU decide each clip's length from the content, for TikTok / Reels / Shorts:
  start on the hook, end right after the payoff or punchline. Never pad with filler,
  setup or the next topic to make it longer; never chop the idea to make it shorter.
  A tight 15-30s clip beats a 60s one that drags; go long (up to 90s) only for a
  story that keeps tension the whole way. Lengths should differ clip to clip.
- Prefer emotional, specific, contrarian or highly actionable moments.
- Skip intros, sponsor reads and "like and subscribe".
- Do not pick overlapping clips. Maximise variety across topics.

Score each clip 0-100 on:
  hook        - does the first 3 seconds stop the scroll
  retention   - pacing, no rambling, keeps tension
  emotion     - energy, stakes, relatability
  value       - insight, specificity, usefulness
  quotability - memorable, shareable, loopable

Return STRICT JSON only:
{"clips":[{"start_sid":<int>,"end_sid":<int>,"title":"<=8 words, same language as transcript",
"hook":"the literal opening words","summary":"1 sentence","why":"why it works, 1 sentence",
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
                "required": ["start_sid", "end_sid", "title", "hook", "why", "hashtags", "scores"],
            },
        }
    },
    "required": ["clips"],
}


def _normalize_scores(raw: dict) -> dict:
    """Models return 0-1, 0-10 or 0-100 depending on mood. Normalise to 0-100."""
    vals: dict[str, float] = {}
    for k in _SCORE_KEYS:
        v = raw.get(k, raw.get(k.title()))
        if isinstance(v, (int, float)):
            vals[k] = float(v)
    if not vals:
        return {}
    mx = max(vals.values())
    if mx <= 1.5:
        vals = {k: v * 100 for k, v in vals.items()}
    elif mx <= 10.0:
        vals = {k: v * 10 for k, v in vals.items()}
    return {k: max(0.0, min(100.0, v)) for k, v in vals.items()}


def _clean_tags(tags) -> list[str]:
    out: list[str] = []
    for t in tags or []:
        t = "#" + str(t).strip().lstrip("#").replace(" ", "")
        if len(t) > 1 and t.lower() not in (x.lower() for x in out):
            out.append(t)
    return out[:6]


def _fatal(e: Exception) -> bool:
    msg = str(e)
    return any(x in msg for x in ("rejected", "not running", "not pulled", "disabled"))


def llm_select(sentences: list[Sentence], cands: list[Candidate], llm: LLM, topic: str,
               k: int, log, min_len: float, max_len: float) -> list[dict]:
    """Ask the LLM window by window, then turn its sentence ranges into picks."""
    windows = _windows(sentences, llm.context_chars - 2500, overlap=max_len)
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
            + f"Clip length: your call per clip (hard limits {min_len:.0f}-{max_len:.0f} s).\n\n"
            + "TRANSCRIPT (sentence_id|start_seconds|text):\n"
            + _transcript_block(win)
            + (f"\n\nHigh-scoring windows from a cheap pre-filter (hints, not limits):\n{hints}"
               if hints else "")
            + f"\n\nPick the best {per} clips. start_sid/end_sid must be sentence ids above."
        )
        if log:
            log(f"AI ({llm.label}) reading part {wi}/{len(windows)}")
        try:
            data = llm.json_completion(SYSTEM, user, schema=SCHEMA, max_tokens=4000, temperature=0.3)
        except LLMError as e:
            failures += 1
            if log:
                log(f"LLM failed on part {wi}: {str(e)[:160]}")
            if failures >= 2 or len(windows) == 1 or _fatal(e):
                break
            continue
        for item in data.get("clips", []) or []:
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
            # keep the AI's own length - no padding with extra sentences. Only enforce the
            # platform limits: too short to be a clip, or longer than a short can be.
            while (end - start) > max_len + 3 and b > a:
                b -= 1
                start, end, words = sentence_span(sentences, a, b)
            if end - start < min_len or end - start > max_len + 3:
                continue
            picks.append({
                "start_sid": a, "end_sid": b, "start": start, "end": end, "words": words,
                "text": " ".join(s.text for s in sentences[a : b + 1]),
                "title": str(item.get("title") or "").strip().strip('"')[:80],
                "hook": str(item.get("hook") or "").strip()[:200],
                "summary": str(item.get("summary") or "").strip()[:400],
                "why": str(item.get("why") or "").strip()[:400],
                "hashtags": _clean_tags(item.get("hashtags")),
                "scores": _normalize_scores(item.get("scores") or {}),
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
                  weights: dict | None, min_len: float) -> tuple[dict, Score]:
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


def viral_judge(shortlist: list[tuple[dict, Score]], llm: LLM, topic: str, log=None) -> None:
    """Second AI pass: rank the shortlist side by side and blend that into each score."""
    blocks = []
    for i, (p, _) in enumerate(shortlist):
        text = p["text"].strip()
        if len(text) > 900:
            text = text[:900] + " ..."
        blocks.append(f"[{i}] ({p['end'] - p['start']:.0f}s) {text}")
    user = ((f"Channel niche: {topic}\n\n" if topic else "")
            + "Candidate clips (id, length, transcript):\n\n" + "\n\n".join(blocks)
            + "\n\nJudge every clip.")
    if log:
        log(f"AI viral judge comparing {len(shortlist)} candidates")
    try:
        data = llm.json_completion(JUDGE_SYSTEM, user, schema=JUDGE_SCHEMA, max_tokens=2500,
                                   temperature=0.2)
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
    for i, (p, sc) in enumerate(shortlist):
        v = verdicts.get(i)
        if not v:
            continue
        raw = v.get("viral")
        viral = float(raw) if isinstance(raw, (int, float)) else 50.0
        viral = viral * 10 if viral <= 10 else viral
        viral = max(0.0, min(100.0, viral))
        total = 0.6 * sc.total + 0.4 * viral
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
) -> tuple[list[Clip], str]:
    k = k or settings.max_clips
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
    if llm is not None and llm.available:
        picks = llm_select(sentences, cands, llm, topic, k, log, min_len, max_len)
        if picks:
            model = llm.label
    def scored(items: list[dict]) -> list[tuple[dict, Score]]:
        out = []
        for p in items:
            sc = score_span(p["text"], p["words"], p["end"] - p["start"], p["start"], p["end"],
                            audio, weights=weights, llm_subscores=p.get("scores") or None)
            out.append((p, sc))
        out.sort(key=lambda x: x[1].total, reverse=True)
        return out

    def tightened(items: list[tuple[dict, Score]]) -> list[tuple[dict, Score]]:
        out = [_tighten_hook(p, sc, sentences, audio, weights, min_len) for p, sc in items]
        out.sort(key=lambda x: x[1].total, reverse=True)
        return out

    shortlist: list[tuple[dict, Score]] = []
    cap = max(k * 2, k + 4)       # judge more candidates than we will keep

    def take(items: list[tuple[dict, Score]]) -> None:
        for p, sc in items:
            if len(shortlist) >= cap:
                return
            if any(min(p["end"], q["end"]) - max(p["start"], q["start"])
                   > 0.15 * min(p["end"] - p["start"], q["end"] - q["start"]) for q, _ in shortlist):
                continue
            shortlist.append((p, sc))

    take(tightened(scored(picks)))                                    # the AI's picks first
    take(tightened(scored(heuristic_select(all_cands, sentences, cap * 2))))   # then rule-based

    if llm is not None and llm.available and len(shortlist) >= 2:
        viral_judge(shortlist, llm, topic, log)
    shortlist.sort(key=lambda x: x[1].total, reverse=True)

    passed = [x for x in shortlist if x[1].total >= min_score]
    if log and min_score:
        log(f"quality bar {min_score}: {len(passed)} of {len(shortlist)} candidates pass")
    chosen = (passed or shortlist[:1])[:k]

    clips: list[Clip] = []
    for rank, (p, sc) in enumerate(chosen, start=1):
        sc.grade = grade(sc.total)
        clip = Clip(
            id=f"clip_{rank:02d}",
            start=round(p["start"], 3),
            end=round(p["end"], 3),
            title=p.get("title") or make_title(p["text"]),
            hook=p.get("hook") or "",
            summary=p.get("summary") or " ".join(p["text"].split()[:38]),
            why=p.get("why") or (sc.reasons[0] if sc.reasons else ""),
            hashtags=p.get("hashtags") or hashtags_for(p["text"]),
            score=sc,
            rank=rank,
        )
        if p.get("risks"):
            clip.score.risks = list(dict.fromkeys(list(sc.risks) + p["risks"]))[:4]
        clips.append(clip)
    return clips, model
