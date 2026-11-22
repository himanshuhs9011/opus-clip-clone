"""Explainable virality scoring.

Every sub-score is a sum of named, auditable signals - no black box. The
weights live in config/scoring.json so the ranking can be tuned per niche.
"""

from __future__ import annotations

import math
import re
from typing import Any

from .audio import AudioFeatures
from .models import Score, Word
from .text import DEFAULT_FILLERS, tokens

DEFAULT_WEIGHTS = {
    "hook": 0.30,
    "retention": 0.22,
    "emotion": 0.18,
    "value": 0.16,
    "quotability": 0.14,
}

QUESTION_WORDS = {
    "what", "why", "how", "when", "where", "who", "which", "do", "does", "did",
    "is", "are", "can", "could", "should", "would", "will", "have", "has",
}
CURIOSITY = [
    "secret", "nobody", "no one", "truth about", "the reason", "here's why",
    "most people", "mistake", "wrong", "actually", "turns out", "what happened",
    "the problem is", "nobody tells", "they don't want", "the real",
]
COMMAND_OPENERS = [
    "stop", "never", "don't", "do not", "quit", "avoid", "forget", "listen",
    "imagine", "look", "watch", "if you", "you need", "you have to",
]
INTENSITY = {
    "amazing", "insane", "crazy", "shocking", "terrible", "awful", "horrible",
    "incredible", "unbelievable", "huge", "massive", "tiny", "worst", "best",
    "love", "hate", "fear", "scared", "angry", "furious", "excited", "obsessed",
    "disaster", "breakthrough", "life-changing", "powerful", "brutal", "wild",
    "fantastic", "perfect", "failed", "failure", "success", "rich", "broke",
}
ABSOLUTES = {"never", "always", "everyone", "nobody", "everything", "nothing", "every", "all"}
HOWTO = [
    "how to", "the way to", "step", "steps", "first", "second", "third", "tip",
    "trick", "strategy", "framework", "system", "method", "process", "recipe",
    "you should", "you need to", "make sure", "here's how",
]
CONTRAST = [
    "but", "however", "instead", "actually", "turns out", "most people think",
    "the truth is", "contrary", "on the other hand", "the problem", "because",
    "which means", "so that", "the key",
]
SIMILE = ["like a", "think of it as", "is basically", "it's like", "imagine"]
CTA = ["comment", "follow", "subscribe", "share", "save this", "try this", "click", "link"]
PRONOUN_START = {
    "he", "she", "it", "they", "this", "that", "these", "those", "his", "her",
    "their", "its", "him", "them",
}
CONJ_START = {"and", "but", "so", "because", "or", "then", "also", "plus"}
MONEY = re.compile(r"[$€£₽]\s?\d|\d+\s?(dollars|bucks|k\b|million|billion|%|percent)", re.I)


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


_PHRASE_RE: dict[tuple, re.Pattern] = {}


def _count_phrases(text: str, phrases: list[str]) -> int:
    """Whole-word phrase hits. Plain substring counting matched "link" in "think",
    "but" in "button" and "tip" in "multiple", which inflated value / quotability."""
    key = tuple(phrases)
    pat = _PHRASE_RE.get(key)
    if pat is None:
        alts = "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True))
        pat = _PHRASE_RE[key] = re.compile(rf"(?<![\w'])(?:{alts})(?![\w'])")
    return len(pat.findall(text))


# Housekeeping that is never a good short: channel plugs, sponsor reads, wrap-ups and
# "we'll be right back" (often over a slate). Penalised so rules-only mode skips them too.
HOUSEKEEPING = re.compile(
    r"\b(subscribe|sub button|like button|hit (that|the) (like|bell)|smash (that|the) like|"
    r"notification bell|link (is )?in (the|my) (description|bio)|use (my )?code|promo code|"
    r"today'?s sponsor|sponsored by|brought to you by|thanks for watching|thank you for watching|"
    r"see you (guys )?(in the )?next (video|time|one)|let you go|before we (go|wrap)|"
    r"thanks? (you )?(so much |very much )?for (coming|having me|joining|tuning in|the time|your time)|"
    r"in today'?s (video|episode|stream)|"
    r"what'?s (good|up) (youtube|guys|everyone)|"
    r"wrap (this|it) up|we'?ll be right back|technical (difficulties|issues)|can you hear me|"
    r"is my (mic|audio)|welcome back to (the|my) (channel|stream|show))\b",
    re.I,
)


def _sentences_from_words(words: list[Word]) -> list[str]:
    out: list[str] = []
    cur: list[str] = []
    for w in words:
        t = w.text.strip()
        cur.append(t)
        if t and t[-1] in ".!?…":
            out.append(" ".join(cur))
            cur = []
    if cur:
        out.append(" ".join(cur))
    return [s for s in out if s.strip()]


def analyse(text: str, words: list[Word], duration: float, t0: float, t1: float,
            audio: AudioFeatures | None) -> dict[str, Any]:
    low = " " + text.lower().strip() + " "
    toks = tokens(text)
    n = max(1, len(toks))
    dur = max(0.5, duration)
    sents = _sentences_from_words(words)
    first = sents[0] if sents else text
    first_toks = tokens(first)[:14]
    last = sents[-1] if sents else text

    filler = DEFAULT_FILLERS["en"] | DEFAULT_FILLERS["ru"]
    filler_hits = sum(1 for t in toks if t in filler)
    numbers = len(re.findall(r"\b\d[\d,.]*\b", text)) + len(MONEY.findall(text))
    questions = text.count("?")
    exclaims = text.count("!")
    intensity = sum(1 for t in toks if t in INTENSITY)
    absolutes = sum(1 for t in toks if t in ABSOLUTES)
    pronouns = sum(1 for t in toks if t in {"i", "we", "you", "your", "my", "our", "me", "us"})
    unique = len(set(toks)) / n
    wps = n / dur

    astats = audio.stats(t0, t1) if audio else {"mean": 0, "std": 0, "silence": 0, "onset": 1, "peak": 0}
    reasons: list[tuple[float, str]] = []
    risks: list[tuple[float, str]] = []

    def hit(bucket, mag, msg):
        bucket.append((mag, msg))

    # ---------------------------------------------------------------- hook
    hook = 40.0
    opens_q = bool(first_toks) and first_toks[0] in QUESTION_WORDS
    if opens_q or "?" in first:
        hook += 20
        hit(reasons, 20, "Opens with a question that creates an open loop")
    if any(ch.isdigit() for ch in " ".join(first_toks[:10])):
        hook += 10
        hit(reasons, 10, "Puts a concrete number in the first seconds")
    if any(t in {"you", "your", "we", "our"} for t in first_toks):
        hook += 10
        hit(reasons, 10, "Speaks directly to the viewer in the opening line")
    if any(first.lower().startswith(c) for c in COMMAND_OPENERS):
        hook += 12
        hit(reasons, 12, "Starts with a bold command / pattern interrupt")
    cur_hits = _count_phrases(low, CURIOSITY)
    if cur_hits:
        hook += min(16, 8 * cur_hits)
        hit(reasons, 8 * cur_hits, "Uses a curiosity gap (secret / most people / the truth)")
    if first_toks and first_toks[0] in filler:
        hook -= 18
        hit(risks, 18, "Opens on a filler word - the hook lands late")
    if first_toks and first_toks[0] in PRONOUN_START and not any(
        w[:1].isupper() for w in first.split()[1:]
    ):
        hook -= 14
        hit(risks, 14, "First line starts on an unresolved pronoun (not self-contained)")
    if len(first_toks) < 4:
        hook -= 8
    if astats["onset"] >= 0.5:
        hook += 8
    elif audio is not None:
        hook -= 12
        hit(risks, 12, "Dead air at the start - viewers scroll in the first 2 seconds")
    house = len(HOUSEKEEPING.findall(text))
    if house:
        hook -= min(30, 15 * house)
        hit(risks, 15 * house, "Channel plug / sponsor / wrap-up talk - not a standalone moment")
    hook = _clamp(hook)

    # ----------------------------------------------------------- retention
    ret = 50.0
    if 2.4 <= wps <= 4.2:
        ret += 22
        hit(reasons, 22, f"Pace is in the sweet spot ({wps:.1f} words/sec)")
    elif 1.9 <= wps < 2.4 or 4.2 < wps <= 5.0:
        ret += 8
    else:
        ret -= 15
        hit(risks, 15, f"Pace is off ({wps:.1f} words/sec - either dragging or rushed)")
    if astats["silence"] <= 0.12:
        ret += 12
    elif astats["silence"] > 0.30:
        ret -= 18
        hit(risks, 18, "Too much silence / dead air inside the clip")
    fr = filler_hits / n
    if fr < 0.02:
        ret += 10
    elif fr > 0.08:
        ret -= 15
        hit(risks, 15, f"Filler-word density is high ({fr * 100:.0f}%)")
    if unique > 0.55:
        ret += 8
    elif unique < 0.40:
        ret -= 12
        hit(risks, 12, "Repetitive wording - low information density")
    avg_sent = n / max(1, len(sents))
    if 6 <= avg_sent <= 16:
        ret += 8
    # short-form completion rates fall with length: tight clips win, long ones must earn it
    if 15 <= dur <= 45:
        ret += 8
        hit(reasons, 4, f"Tight length for Shorts/Reels/TikTok ({dur:.0f}s)")
    elif 45 < dur <= 60:
        ret += 3
    elif dur > 75:
        ret -= 8
        hit(risks, 8, f"Long for a short ({dur:.0f}s) - needs tension the whole way")
    elif dur < 12:
        ret -= 10
        hit(risks, 10, "Clip is very short - not enough payoff")
    ret = _clamp(ret)

    # ------------------------------------------------------------- emotion
    emo = 42.0
    per100 = intensity / n * 100
    if per100 >= 3:
        emo += 20
        hit(reasons, 20, "Dense emotional / high-arousal language")
    elif per100 >= 1.5:
        emo += 12
        hit(reasons, 12, "Punchy emotional language")
    elif per100 >= 0.7:
        emo += 6
    if exclaims:
        emo += 6
    if pronouns / n * 100 >= 6:
        emo += 10
        hit(reasons, 10, "Strong personal address (I / you / we)")
    if 0.18 <= astats["std"] <= 0.40:
        emo += 16
        hit(reasons, 16, "Vocal delivery has real dynamic range")
    elif 0.12 <= astats["std"] < 0.18:
        emo += 8
    elif audio is not None and astats["std"] < 0.08:
        emo -= 10
        hit(risks, 10, "Flat delivery - monotone energy")
    if questions:
        emo += 6
    if absolutes >= 2:
        emo += 6
    emo = _clamp(emo)

    # --------------------------------------------------------------- value
    val = 42.0
    num_per100 = numbers / n * 100
    if num_per100 >= 1.2:
        val += 14
        hit(reasons, 14, f"Uses concrete numbers ({numbers})")
    if _count_phrases(low, HOWTO):
        val += 14
        hit(reasons, 14, "Contains actionable how-to language")
    if _count_phrases(low, CONTRAST):
        val += 12
        hit(reasons, 12, "Has a clear insight / contrast ('most people think… but')")
    named = sum(1 for w in text.split()[1:] if w[:1].isupper())
    if named >= 2:
        val += 8
    if MONEY.search(text):
        val += 6
    if len(re.findall(r"\b(first|second|third|three|two|one)\b", low)) >= 2:
        val += 6
    if house:
        val -= min(24, 12 * house)
    val = _clamp(val)

    # --------------------------------------------------------- quotability
    quo = 45.0
    if first_toks and first_toks[0] not in PRONOUN_START and first_toks[0] not in CONJ_START:
        quo += 12
    elif first_toks and first_toks[0] in PRONOUN_START:
        quo -= 10
    if last.rstrip().endswith((".", "!", "?")):
        quo += 8
    if last.strip().endswith("?") or _count_phrases(last.lower(), CTA):
        quo += 6
        hit(reasons, 6, "Ends with a question / call to action")
    if any(len(tokens(s)) <= 6 for s in sents):
        quo += 8
        hit(reasons, 8, "Contains a short punchy one-liner")
    if len(sents) >= 2 and "but" in " ".join(sents[-2:]).lower():
        quo += 6
    if _count_phrases(low, SIMILE):
        quo += 6
        hit(reasons, 6, "Uses a memorable comparison / metaphor")
    quo = _clamp(quo)

    return {
        "hook": hook,
        "retention": ret,
        "emotion": emo,
        "value": val,
        "quotability": quo,
        "reasons": reasons,
        "risks": risks,
        "features": {
            "words": n,
            "duration": round(dur, 1),
            "words_per_sec": round(wps, 2),
            "sentences": len(sents),
            "filler_ratio": round(fr, 3),
            "silence_ratio": round(astats["silence"], 3),
            "energy_std": round(astats["std"], 3),
            "numbers": numbers,
            "intensity_hits": intensity,
            "unique_ratio": round(unique, 3),
            "questions": questions,
            "housekeeping": house,
        },
    }


def score_span(
    text: str,
    words: list[Word],
    duration: float,
    t0: float,
    t1: float,
    audio: AudioFeatures | None = None,
    weights: dict | None = None,
    llm_subscores: dict | None = None,
) -> Score:
    m = analyse(text, words, duration, t0, t1, audio)
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    keys = ["hook", "retention", "emotion", "value", "quotability"]

    if llm_subscores:
        # blend the model's read of the clip with the measured signals
        for k in keys:
            v = llm_subscores.get(k)
            if isinstance(v, (int, float)):
                m[k] = _clamp(0.55 * m[k] + 0.45 * float(v))

    total = sum(m[k] * w[k] for k in keys)
    total = _clamp(total * 0.94 + 3)  # mild calibration, keeps 0-100 honest

    reasons = [t for _, t in sorted(m["reasons"], reverse=True)[:3]]
    risks = [t for _, t in sorted(m["risks"], reverse=True)[:3]]
    return Score(
        total=int(round(total)),
        hook=int(round(m["hook"])),
        retention=int(round(m["retention"])),
        emotion=int(round(m["emotion"])),
        value=int(round(m["value"])),
        quotability=int(round(m["quotability"])),
        reasons=reasons,
        risks=risks,
        features=m["features"],
        source="llm+heuristic" if llm_subscores else "heuristic",
    )


def grade(total: int) -> str:
    """Letter grade shown next to the 0-100 virality score."""
    for cut, g in ((90, "A+"), (84, "A"), (78, "A-"), (73, "B+"), (68, "B"), (63, "B-"),
                   (58, "C+"), (52, "C"), (45, "C-")):
        if total >= cut:
            return g
    return "D"


def score_label(total: int) -> str:
    if total >= 78:
        return "Excellent"
    if total >= 62:
        return "Good"
    if total >= 45:
        return "Average"
    return "Weak"
