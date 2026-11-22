"""Where a clip starts and ends, and so how long it is.

The AI (or the rule-based picker) proposes a sentence range. This module reviews the
edges the way a human editor would, so each clip's length follows its content:

Opening (the first spoken seconds decide whether a viewer stays)
  * skip lead-in banter and lone fragments ("No. No? Ah!", "Yeah.", "Exactly.");
  * a first sentence that continues something unseen ("But the main reason...",
    "That's why...", "As I said...", "He told me...") either gets its setup sentence
    added back, or - when it is a short throwaway - is trimmed away.

Ending (the payoff must land)
  * a clip that ends on a question gets its answer;
  * an unfinished last sentence runs on to its full stop;
  * a payoff line that comes right after ("And that's why...", "Turns out...") is kept;
  * trailing banter after the payoff ("Yeah. Okay.") is cut.

Length
  * a pick that is a little short of the minimum is lengthened while the speech runs on
    (no long pause in between), instead of being thrown away; a much shorter one is dropped.

Edges
  * cut 0.05 s before the first word and 0.12 s after the last one, never into a
    neighbouring word, so no consonant is clipped and the last word fully lands.

The rules are language-light (English plus common Hindi / Hinglish connectives) and pure,
so they can be tested on saved transcripts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import Sentence, Word

LEAD_PAD = 0.05          # before the first word
TAIL_PAD = 0.12          # after the last word (Whisper word ends are often a little early)
NEIGHBOUR_GAP = 0.02     # never start / end closer than this to a word outside the clip

FLOW_GAP = 1.5           # sentences closer than this are one flow of speech
MAX_BACK = 2             # setup sentences that may be added in front
MAX_FORWARD = 3          # payoff sentences that may be added after
STRETCH_SHORT = 0.25     # a pick up to 25 % under the minimum is lengthened, not dropped

BARE = {
    "yeah", "yes", "yep", "yup", "right", "exactly", "totally", "sure", "okay", "ok", "no",
    "nope", "mhm", "uh", "um", "umm", "hmm", "ah", "oh", "wow", "well", "so", "like", "true",
    "haan", "ha", "han", "achha", "accha", "acha", "theek", "thik", "hai", "ji", "nahi", "arre",
    "हाँ", "हां", "जी", "अच्छा", "ठीक", "है", "नहीं", "अरे",
}
DANGLING = {
    "and", "but", "or", "nor", "because", "cause", "plus", "also", "then",
    "aur", "lekin", "magar", "kyunki", "phir", "और", "लेकिन", "मगर", "क्योंकि", "फिर",
}
DANGLING_PHRASES = (
    "so that's why", "so that is why", "that's why", "that is why", "which is why",
    "which means", "so the reason", "so the main reason", "and then",
)
BACKREF_PHRASES = (
    "as i said", "like i said", "as i mentioned", "like i mentioned", "as we said",
    "like we said", "as we discussed", "going back to", "to your point", "that point",
    "the thing we", "before we started", "what you just said", "you just said",
)
PRONOUNS = {"he", "she", "they", "him", "her", "them", "his", "their", "it", "this", "that",
            "these", "those", "वो", "वह", "उसने", "उन्होंने"}
PAYOFF_OPENERS = (
    "and that's why", "that's why", "and that is why", "so that's how", "that's how",
    "turns out", "it turns out", "the point is", "the lesson", "so the lesson", "and that's",
    "which is why", "and the answer", "the answer is", "that's the", "and guess what",
)
SENT_END = (".", "!", "?", "…", "।")
_WORD = re.compile(r"[\w'’]+", re.UNICODE)


def _toks(text: str) -> list[str]:
    return [t.lower().replace("’", "'") for t in _WORD.findall(text or "")]


def _low(text: str) -> str:
    return " ".join(_toks(text))


def is_banter(text: str) -> bool:
    """A lone fragment or reaction with nothing a new viewer can hold on to."""
    toks = _toks(text)
    if not toks:
        return True
    content = [t for t in toks if t not in BARE]
    return len(content) == 0 or (len(toks) <= 2 and len(content) <= 1)


def is_housekeeping(text: str) -> bool:
    """Channel plugs, sponsor reads, greetings and sign-offs (shared with the scorer)."""
    from .score import HOUSEKEEPING

    return bool(HOUSEKEEPING.search(text or ""))


def opening_issue(text: str) -> str | None:
    """Why this sentence cannot open a clip on its own, or None when it can."""
    toks = _toks(text)
    if not toks:
        return "empty"
    low = " ".join(toks)
    if any(low.startswith(p) for p in BACKREF_PHRASES):
        return "back-reference"
    if any(low.startswith(p) for p in DANGLING_PHRASES) or toks[0] in DANGLING:
        return "dangling"
    if toks[0] in PRONOUNS and not any(w[:1].isupper() for w in (text or "").split()[1:6]):
        return "pronoun"
    return None


def _ends(text: str) -> bool:
    return (text or "").rstrip().endswith(SENT_END)


def is_punctuated(sentences: list[Sentence]) -> bool:
    """Whether Whisper punctuated this transcript (it often does not for Hindi). Only then
    does a missing full stop tell us a unit is the middle of a sentence."""
    if not sentences:
        return False
    return sum(1 for s in sentences if _ends(s.text)) / len(sentences) >= 0.5


def mid_sentence(sentences: list[Sentence], i: int, punctuated: bool = True) -> bool:
    """Unit i continues the unit before it (long sentences are split into 20-word units)."""
    return (punctuated and i > 0 and not _ends(sentences[i - 1].text)
            and sentences[i].start - sentences[i - 1].end <= FLOW_GAP)


@dataclass
class Review:
    a: int
    b: int
    notes: list[str] = field(default_factory=list)
    risk: str | None = None


def _dur(sentences: list[Sentence], a: int, b: int) -> float:
    return sentences[b].end - sentences[a].start


def _gap(sentences: list[Sentence], i: int) -> float:
    """Pause between sentence i and i + 1."""
    return sentences[i + 1].start - sentences[i].end


def review(sentences: list[Sentence], a: int, b: int, min_len: float, max_len: float,
           punctuated: bool | None = None) -> Review:
    """Move a sentence range [a, b] onto edges a viewer can follow (see the module doc)."""
    n = len(sentences)
    a, b = max(0, min(a, n - 1)), max(0, min(b, n - 1))
    if b < a:
        a, b = b, a
    if punctuated is None:
        punctuated = is_punctuated(sentences)
    r = Review(a, b)

    # ---- opening: never mid-sentence - go back to where the sentence starts
    back = 0
    while back < 3 and mid_sentence(sentences, r.a, punctuated):
        if _dur(sentences, r.a - 1, r.b) > max_len:
            # no room in front: give up some of the ending rather than start mid-sentence
            b2 = r.b
            while b2 > r.a and _dur(sentences, r.a - 1, b2) > max_len:
                b2 -= 1
            if _dur(sentences, r.a - 1, b2) > max_len or _dur(sentences, r.a - 1, b2) < min_len:
                break
            r.b = b2
        r.a -= 1
        back += 1
    if back:
        r.notes.append("Start moved back to the beginning of the sentence")

    # ---- opening: lead-in banter ("No. No? Ah!") or a channel plug before the real first line
    skipped = 0
    while (r.a < r.b and skipped < 3
           and (is_banter(sentences[r.a].text) or is_housekeeping(sentences[r.a].text))
           and not mid_sentence(sentences, r.a + 1, punctuated)
           and _dur(sentences, r.a + 1, r.b) >= min_len):
        r.a += 1
        skipped += 1
    if skipped:
        r.notes.append("Skipped lead-in banter before the first real line")

    # ---- opening: a first line that continues something the viewer did not hear
    issue = opening_issue(sentences[r.a].text)
    if issue:
        fixed = False
        short = len(_toks(sentences[r.a].text)) <= 6
        nxt = r.a + 1
        if (issue == "dangling" and short and nxt <= r.b
                and opening_issue(sentences[nxt].text) is None
                and not mid_sentence(sentences, nxt, punctuated)
                and _dur(sentences, nxt, r.b) >= min_len):
            r.a = nxt                                   # a throwaway "And so..." line
            r.notes.append("Trimmed a dangling opening line")
            fixed = True
        else:
            a2 = r.a
            for _ in range(MAX_BACK):
                if a2 == 0 or _gap(sentences, a2 - 1) > FLOW_GAP:
                    break
                if _dur(sentences, a2 - 1, r.b) > max_len:
                    break
                if is_banter(sentences[a2 - 1].text):
                    break
                a2 -= 1
                if opening_issue(sentences[a2].text) is None and not mid_sentence(sentences, a2, punctuated):
                    break
            if (a2 < r.a and opening_issue(sentences[a2].text) is None
                    and not mid_sentence(sentences, a2, punctuated)):
                r.a = a2
                r.notes.append("Start moved back to include the setup the first line needs")
                fixed = True
        if not fixed:
            r.risk = {"pronoun": "Opens on a pronoun the viewer has not met yet",
                      "back-reference": "Opens by referring back to something not in the clip",
                      }.get(issue, "Opening line continues an earlier thought")

    # ---- ending: a long question after an answer is the next topic starting - drop it
    last = sentences[r.b].text.rstrip()
    if (last.endswith("?") and r.b > r.a and len(_toks(last)) >= 7
            and _ends(sentences[r.b - 1].text) and not sentences[r.b - 1].text.rstrip().endswith("?")
            and _dur(sentences, r.a, r.b - 1) >= min_len):
        r.b -= 1
        r.notes.append("Dropped the next question that starts a new topic")

    # ---- ending: a real question needs its answer; an unfinished sentence its full stop.
    # (A short rhetorical question - "Isn't that crazy?" - is a fine last line.)
    added = 0
    while added < MAX_FORWARD and r.b + 1 < n and _gap(sentences, r.b) <= FLOW_GAP:
        last = sentences[r.b].text.rstrip()
        nxt_low = _low(sentences[r.b + 1].text)
        open_q = last.endswith("?") and len(_toks(last)) >= 5
        want = (open_q and not is_banter(sentences[r.b + 1].text)) \
            or not _ends(last) \
            or any(nxt_low.startswith(p) for p in PAYOFF_OPENERS)
        if not want or _dur(sentences, r.a, r.b + 1) > max_len:
            break
        reason = ("Ends on the answer, not the question" if last.endswith("?")
                  else "Runs on to finish the last sentence" if not _ends(last)
                  else "Keeps the payoff line that comes right after")
        r.b += 1
        added += 1
        if reason not in r.notes:
            r.notes.append(reason)

    # ---- ending: banter or a sign-off after the payoff ("Yeah. Okay.", "Thanks for coming on")
    trimmed = 0
    while (r.b > r.a and trimmed < 3
           and (is_banter(sentences[r.b].text) or is_housekeeping(sentences[r.b].text))
           and _dur(sentences, r.a, r.b - 1) >= min_len):
        r.b -= 1
        trimmed += 1
    if trimmed:
        r.notes.append("Cut the chatter after the payoff")

    # ---- a little short: lengthen while the speech runs on, forward first
    if _dur(sentences, r.a, r.b) < min_len:
        while _dur(sentences, r.a, r.b) < min_len:
            fwd = (r.b + 1 < n and _gap(sentences, r.b) <= FLOW_GAP
                   and _dur(sentences, r.a, r.b + 1) <= max_len)
            back = (r.a > 0 and _gap(sentences, r.a - 1) <= FLOW_GAP
                    and _dur(sentences, r.a - 1, r.b) <= max_len
                    and opening_issue(sentences[r.a - 1].text) is None
                    and not mid_sentence(sentences, r.a - 1, punctuated))
            if fwd:
                r.b += 1
            elif back:
                r.a -= 1
            else:
                break
        if _dur(sentences, r.a, r.b) >= min_len:
            r.notes.append("Lengthened to a full moment where the speech runs on")
    return r


def can_stretch(duration: float, min_len: float) -> bool:
    """A pick this short is worth lengthening; anything shorter is dropped."""
    return duration >= min_len * (1 - STRETCH_SHORT)


def padded_edges(sentences: list[Sentence], a: int, b: int) -> tuple[float, float]:
    """Cut points for sentences [a, b]: a hair before the first word, after the last."""
    first: Word = sentences[a].words[0] if sentences[a].words else None
    last: Word = sentences[b].words[-1] if sentences[b].words else None
    start = (first.start if first else sentences[a].start) - LEAD_PAD
    end = (last.end if last else sentences[b].end) + TAIL_PAD
    if a > 0:
        prev = sentences[a - 1].words[-1].end if sentences[a - 1].words else sentences[a - 1].end
        start = max(start, min(prev + NEIGHBOUR_GAP, sentences[a].start))
    if b + 1 < len(sentences):
        nxt = sentences[b + 1].words[0].start if sentences[b + 1].words else sentences[b + 1].start
        end = min(end, max(nxt - NEIGHBOUR_GAP, sentences[b].end))
    return max(0.0, start), end


def stretch_to(sentences: list[Sentence], a: int, b: int, target: float,
               punctuated: bool = True) -> tuple[int, int]:
    """The user asked for clips of about ``target`` s (the longest they may be). A pick up to
    20 % short is lengthened towards it while the speech runs on; never past the target."""
    dur = _dur(sentences, a, b)
    if not target or dur >= target or dur < target * 0.8:
        return a, b
    n = len(sentences)
    while b + 1 < n and _gap(sentences, b) <= FLOW_GAP and _dur(sentences, a, b + 1) <= target:
        b += 1
    while (a > 0 and _gap(sentences, a - 1) <= FLOW_GAP and _dur(sentences, a - 1, b) <= target
           and opening_issue(sentences[a - 1].text) is None and not mid_sentence(sentences, a - 1, punctuated)
           and _dur(sentences, a, b) < target * 0.9):
        a -= 1
    return a, b


FOCUS_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "about", "when", "where",
    "someone", "something", "every", "time", "times", "moments", "moment", "clips", "clip", "only",
    "just", "where", "that", "this", "they", "them", "their", "his", "her", "he", "she", "gets", "get",
    "biggest", "best", "most", "really", "very", "talks", "talking", "says", "said",
}


def focus_terms(focus: str) -> list[str]:
    """Content words of the user's request, cut to a 4-letter stem ("laughs" ~ "laughing", "fails" ~ "failed")."""
    out = []
    for t in _toks(focus):
        if t in FOCUS_STOP or len(t) < 3:
            continue
        out.append(t[:4])
    return list(dict.fromkeys(out))


def focus_hits(text: str, terms: list[str]) -> int:
    toks = {t[:4] for t in _toks(text)}
    return sum(1 for t in terms if t in toks)
