"""Clip edit model: everything the editor can change, and the timeline maths behind it.

The same rules are mirrored in the React preview (frontend/src/lib/timeline.ts) so what
you see in the browser is what ffmpeg renders. Keep the two in sync.
"""
from __future__ import annotations

import copy
import re

from .models import Clip, Sentence, Word

ASPECTS = {"9:16": (1080, 1920), "1:1": (1080, 1080), "4:5": (1080, 1350), "16:9": (1920, 1080)}
LAYOUTS = ("auto", "track", "split", "gaming", "center", "fit")
FILTERS = ("none", "vivid", "warm", "cool", "cinematic", "bw", "vintage")

DEFAULT_EDIT: dict = {
    "start": None,               # absolute seconds in the source, None = clip bounds
    "end": None,
    "aspect": "9:16",
    "layout": "auto",            # auto: follow the speaker, split two-shots, blur-fit if no faces
    "captions": {
        "enabled": True,
        "preset": "bold-pop",
        "position": 0.70,        # vertical centre of the caption block, 0..1 of frame height
        "size": 1.0,
        "uppercase": None,       # None = preset default
        "words": None,           # words per caption, None = preset default
        "keywords": True,        # colour important words
        "emoji": True,
        "speaker_colors": True,  # each speaker gets their own caption colour
        "speakers": {},          # "0" -> "#hex" overrides; empty = automatic contrast palette
    },
    # mode: auto = context banner on the seam for gaming / split layouts, hook title on top otherwise
    "title": {"enabled": True, "text": None, "seconds": 3.5, "mode": "auto"},
    "progress": {"enabled": False, "color": "#FFD400", "position": "top"},
    "zoom": {"enabled": True, "strength": 0.10},
    # gaming / stream layout: face-cam panel + gameplay panel
    "stream": {"auto": True, "template": "stack", "corner": "tl", "cam": "top", "cam_size": 0.48,
               "cam_zoom": 1.0, "game": "fill"},
    "cuts": {"silence": False, "fillers": False, "removed": []},   # removed: word start in ms
    "word_edits": {},            # "start_ms" -> replacement text
    "filter": "none",
    "audio": {"normalize": True},
}

STRICT_FILLERS = {"um", "umm", "uh", "uhh", "uhm", "er", "erm", "ah", "hmm", "mm", "mhm", "eh"}

NUMBER = re.compile(r"\d|\b(one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand|million|billion|percent)\b", re.I)
STRONG = {
    "amazing", "insane", "crazy", "shocking", "terrible", "awful", "incredible", "unbelievable",
    "huge", "massive", "worst", "best", "love", "hate", "fear", "scared", "angry", "excited",
    "disaster", "breakthrough", "powerful", "brutal", "wild", "perfect", "failed", "failure",
    "success", "rich", "broke", "money", "secret", "truth", "mistake", "never", "always",
    "nobody", "everyone", "everything", "nothing", "free", "dead", "die", "kill", "war",
    "win", "won", "lose", "lost", "problem", "dangerous", "illegal", "impossible", "first",
    "only", "biggest", "fastest", "important", "real", "stop", "wrong", "right", "why", "how",
    "dollars", "million", "billion", "fired", "quit", "famous", "viral",
}
STOP = {
    "the", "a", "an", "and", "or", "but", "so", "to", "of", "in", "on", "for", "with", "at",
    "by", "from", "is", "are", "was", "were", "be", "been", "it", "that", "this", "i", "you",
    "he", "she", "we", "they", "me", "my", "your", "our", "their", "just", "like", "really",
    "very", "what", "when", "then", "than", "there", "here", "have", "has", "had", "do", "did",
    "does", "not", "no", "yes", "if", "about", "into", "out", "up", "down", "can", "could",
    "would", "should", "will", "going", "gonna", "know", "think", "thing", "things", "people",
    "because", "actually", "basically", "literally", "right", "okay", "well", "also",
}
EMOJI = {
    "money": "💰", "dollars": "💵", "dollar": "💵", "cash": "💵", "rich": "🤑", "million": "💸",
    "billion": "💸", "broke": "😩", "love": "❤️", "heart": "❤️", "fire": "🔥", "hot": "🔥",
    "crazy": "🤯", "insane": "🤯", "mind": "🧠", "brain": "🧠", "shocking": "😱", "scared": "😱",
    "fear": "😨", "secret": "🤫", "failed": "💀", "failure": "💀", "dead": "💀", "die": "💀",
    "success": "🏆", "win": "🏆", "won": "🏆", "winning": "🏆", "growth": "📈", "grow": "📈",
    "time": "⏰", "hours": "⏰", "days": "📅", "year": "📅", "idea": "💡", "think": "🤔",
    "question": "❓", "why": "🤔", "laugh": "😂", "funny": "😂", "happy": "😄", "sad": "😢",
    "angry": "😡", "hate": "😤", "work": "💼", "business": "💼", "customers": "🤝",
    "phone": "📱", "computer": "💻", "laptop": "💻", "internet": "🌐", "world": "🌍",
    "food": "🍔", "eat": "🍔", "gym": "💪", "strong": "💪", "health": "🩺", "sleep": "😴",
    "energy": "⚡", "fast": "⚡", "power": "⚡", "rocket": "🚀", "launch": "🚀", "start": "🚀",
    "goal": "🎯", "focus": "🎯", "target": "🎯", "book": "📚", "learn": "📚", "school": "🎓",
    "stop": "🛑", "warning": "⚠️", "problem": "⚠️", "mistake": "❌", "wrong": "❌",
    "right": "✅", "yes": "✅", "perfect": "👌", "best": "🥇", "first": "🥇", "car": "🚗",
    "house": "🏠", "home": "🏠", "family": "👨‍👩‍👧", "friend": "🫂", "friends": "🫂",
    "music": "🎵", "game": "🎮", "camera": "📸", "video": "🎬", "eyes": "👀", "look": "👀",
    "secretly": "🤫", "truth": "💯", "honest": "💯", "honestly": "💯", "boom": "💥",
}


def norm(text: str) -> str:
    return re.sub(r"[^\w'%$-]", "", text.lower())


def ms(t: float) -> str:
    return str(int(round(t * 1000)))


def merge_edit(base: dict | None, patch: dict | None) -> dict:
    out = copy.deepcopy(DEFAULT_EDIT)

    def deep(dst: dict, src: dict) -> None:
        for k, v in (src or {}).items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict) and k not in ("word_edits", "speakers"):
                deep(dst[k], v)
            else:
                dst[k] = copy.deepcopy(v)

    deep(out, base or {})
    deep(out, patch or {})
    if out["aspect"] not in ASPECTS:
        out["aspect"] = "9:16"
    if out["layout"] not in LAYOUTS:
        out["layout"] = "auto"
    if out["filter"] not in FILTERS:
        out["filter"] = "none"
    return out


def bounds(clip: Clip, edit: dict) -> tuple[float, float]:
    a = edit.get("start")
    b = edit.get("end")
    a = clip.start if a is None else float(a)
    b = clip.end if b is None else float(b)
    if b - a < 1.0:
        b = a + 1.0
    return a, b


def words_in(sentences: list[Sentence], a: float, b: float) -> list[Word]:
    out = []
    for s in sentences:
        if s.end < a or s.start > b:
            continue
        for w in s.words:
            if w.end > a and w.start < b and w.text.strip():
                out.append(w)
    return out


# --------------------------------------------------------------------------
# cuts -> keep segments (clip-relative seconds)
# --------------------------------------------------------------------------
def keep_segments(words: list[Word], a: float, b: float, cuts: dict) -> list[tuple[float, float]]:
    dur = b - a
    removed_ms = {str(x) for x in (cuts.get("removed") or [])}
    rel = [(w.start - a, w.end - a, w) for w in words]
    drop: list[tuple[float, float]] = []

    kept_idx = []
    for i, (s, e, w) in enumerate(rel):
        gone = ms(w.start) in removed_ms or (cuts.get("fillers") and norm(w.text) in STRICT_FILLERS)
        if gone:
            drop.append((max(0.0, s - 0.02), min(dur, e + 0.02)))
        else:
            kept_idx.append(i)

    if cuts.get("silence") and kept_idx:
        pad = 0.12
        first = rel[kept_idx[0]]
        if first[0] > 0.35:
            drop.append((0.0, first[0] - pad))
        for x, y in zip(kept_idx, kept_idx[1:]):
            gap_a = rel[x][1]
            gap_b = rel[y][0]
            if gap_b - gap_a > 0.55:
                drop.append((gap_a + pad, gap_b - pad))
        last = rel[kept_idx[-1]]
        if dur - last[1] > 0.45:
            drop.append((last[1] + 0.25, dur))

    if not drop:
        return [(0.0, dur)]
    drop.sort()
    merged: list[list[float]] = []
    for s, e in drop:
        if e <= s:
            continue
        if merged and s <= merged[-1][1] + 0.04:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    keep = []
    cur = 0.0
    for s, e in merged:
        if s - cur >= 0.08:
            keep.append((round(cur, 3), round(s, 3)))
        cur = max(cur, e)
    if dur - cur >= 0.08:
        keep.append((round(cur, 3), round(dur, 3)))
    return keep or [(0.0, dur)]


def remap(t: float, segs: list[tuple[float, float]]) -> float | None:
    acc = 0.0
    for s, e in segs:
        if t < s:
            return None
        if t <= e:
            return acc + (t - s)
        acc += e - s
    return None


def remap_clamped(t: float, segs: list[tuple[float, float]]) -> float:
    acc = 0.0
    for s, e in segs:
        if t <= s:
            return acc
        if t <= e:
            return acc + (t - s)
        acc += e - s
    return acc


def total_length(segs: list[tuple[float, float]]) -> float:
    return sum(e - s for s, e in segs)


# --------------------------------------------------------------------------
# words on the output timeline, with keyword / emoji flags
# --------------------------------------------------------------------------
def keyword_score(text: str, prev_text: str = "") -> float:
    t = norm(text)
    if not t or t in STOP:
        return 0.0
    sc = 0.0
    if NUMBER.search(text) or "$" in text or "%" in text:
        sc += 3.0
    if t in STRONG:
        sc += 2.5
    raw = text.strip()
    if raw[:1].isupper() and prev_text and prev_text.strip()[-1:] not in ".!?" and t not in STOP:
        sc += 1.5
    if len(t) >= 8:
        sc += 1.0
    return sc


def output_words(words: list[Word], a: float, segs: list[tuple[float, float]],
                 edit: dict) -> list[dict]:
    """Words mapped to the rendered timeline: {s, e, w, k (keyword), id (source ms)}."""
    edits = edit.get("word_edits") or {}
    out = []
    prev = ""
    for w in words:
        s = remap(w.start - a, segs)
        e_rel = w.end - a
        if s is None:
            # word starts inside a cut: keep it only if its middle survives
            mid = remap((w.start + w.end) / 2 - a, segs)
            if mid is None:
                prev = w.text
                continue
            s = mid
        e = remap_clamped(e_rel, segs)
        if e - s < 0.06:
            e = s + 0.06
        key = ms(w.start)
        text = edits.get(key, w.text)
        if not str(text).strip():
            prev = w.text
            continue
        out.append({"s": round(s, 3), "e": round(e, 3), "w": str(text).strip(),
                    "k": keyword_score(text, prev) >= 2.5, "id": key, "sp": getattr(w, "sp", -1)})
        prev = w.text
    for i in range(len(out) - 1):
        if out[i]["e"] > out[i + 1]["s"]:
            out[i]["e"] = max(out[i]["s"] + 0.04, out[i + 1]["s"])
    return out


def emoji_events(owords: list[dict], max_count: int = 8, min_gap: float = 3.0) -> list[dict]:
    events = []
    last = -99.0
    for w in owords:
        em = EMOJI.get(norm(w["w"]))
        if not em or w["s"] - last < min_gap:
            continue
        events.append({"t": w["s"], "end": w["s"] + 1.6, "emoji": em})
        last = w["s"]
        if len(events) >= max_count:
            break
    return events


def zoom_events(owords: list[dict], duration: float, strength: float) -> list[tuple[float, float]]:
    """Step punch-ins on sentence starts and strong keywords: [(t, zoom_level)]."""
    if strength <= 0 or not owords:
        return [(0.0, 1.0)]
    events = [(0.0, 1.0)]
    level = 1.0
    last = 0.0
    for i, w in enumerate(owords):
        sentence_start = i > 0 and owords[i - 1]["w"].strip()[-1:] in ".!?"
        if not (sentence_start or w["k"]):
            continue
        if w["s"] - last < 2.6 or duration - w["s"] < 1.5:
            continue
        level = 1.0 + strength if level == 1.0 else 1.0
        events.append((round(w["s"], 3), level))
        last = w["s"]
    return events


def hashtags_for(text: str, n: int = 5) -> list[str]:
    counts: dict[str, int] = {}
    for tok in re.findall(r"[A-Za-z][A-Za-z']{3,}", text):
        t = tok.lower().strip("'")
        if t in STOP or len(t) < 4:
            continue
        counts[t] = counts.get(t, 0) + 1 + (2 if t in STRONG else 0)
    top = sorted(counts, key=lambda k: (-counts[k], k))[: max(0, n - 2)]
    return [f"#{t}" for t in top] + ["#shorts", "#viral"]
