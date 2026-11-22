"""Clip edges, length and the fixes that came with them.   python -m pytest tests"""
from __future__ import annotations

import numpy as np

from opusclip.audio import AudioFeatures
from opusclip.boundaries import is_banter, opening_issue, padded_edges, review
from opusclip.gaming import hype_envelope, hype_map
from opusclip.highlights import _normalize_scores, _scale_for, select
from opusclip.models import Sentence, Word
from opusclip.score import _count_phrases, score_span
from opusclip.text import build_sentences


def make(lines: list[tuple[float, str]], wps: float = 3.0) -> list[Sentence]:
    """Sentences from (start, text), words spread evenly at `wps` words per second."""
    out = []
    for i, (t, text) in enumerate(lines):
        words = []
        for j, w in enumerate(text.split()):
            s = t + j / wps
            words.append(Word(start=s, end=s + 0.8 / wps, text=w))
        out.append(Sentence(id=i, start=words[0].start, end=words[-1].end, text=text, words=words))
    return out


def flat_audio(seconds: float) -> AudioFeatures:
    n = int(seconds / 0.05) + 1
    rng = np.random.default_rng(1)
    return AudioFeatures(hop=0.05, rms=rng.uniform(0.3, 0.9, n).astype(np.float32),
                         db=np.full(n, -20.0, np.float32), duration=seconds)


# ------------------------------------------------------------------ opening
def test_opening_issue_detects_context_dependent_starts():
    assert opening_issue("But the main reason is money.") == "dangling"
    assert opening_issue("That's why nobody does it.") == "dangling"
    assert opening_issue("As I said, the market is tiny.") == "back-reference"
    assert opening_issue("He told me to quit.") == "pronoun"
    assert opening_issue("He told Sarah to quit.") is None        # a name follows: fine
    assert opening_issue("Most founders build for a year first.") is None


def test_banter():
    assert is_banter("Yeah.")
    assert is_banter("No. No? Ah!")
    assert not is_banter("Talk to ten customers first.")


def test_dangling_start_gets_its_setup():
    s = make([(0, "The biggest mistake founders make is building alone."),
              (4, "But the real reason is they are scared of rejection and hate cold calls."),
              (10, "Rejection hurts more than any bug report ever could hurt you."),
              (16, "So talk to ten customers before you write a single line of code.")])
    r = review(s, 1, 3, min_len=12, max_len=90)
    assert r.a == 0 and "setup" in r.notes[0]


def test_question_ending_gets_its_answer():
    s = make([(0, "Everybody asks me how I grew the channel to a million subscribers so fast."),
              (6, "So what was the one thing that actually moved the needle for you?"),
              (11, "Posting every single day for a year, no matter what happened.")])
    r = review(s, 0, 1, min_len=8, max_len=90)
    assert r.b == 2


def test_trailing_signoff_and_banter_are_cut():
    s = make([(0, "Ask how they solve the problem today and what it costs them right now."),
              (6, "Then shut up and listen, because that is where the real answer is."),
              (12, "If nobody pays to solve it today, that is your answer, move on."),
              (18, "Yeah."),
              (19, "Thank you for coming on the show.")])
    r = review(s, 0, 4, min_len=12, max_len=90)
    assert r.b == 2


def test_mid_sentence_start_moves_back():
    s = make([(0, "I always tell people that the first thing you should do"),
              (3.4, "is talk to ten customers before writing any code at all."),
              (8, "Most people skip it and regret it a year later when nothing sells.")])
    r = review(s, 1, 2, min_len=8, max_len=90)
    assert r.a == 0


def test_short_pick_is_lengthened_while_speech_runs_on():
    s = make([(0, "Here is the rule."), (2, "Talk to ten customers first."),
              (5, "Then write code."), (7, "That order matters more than anything else you do.")])
    r = review(s, 0, 1, min_len=9, max_len=90)
    assert s[r.b].end - s[r.a].start >= 9


def test_padded_edges_never_cut_into_neighbours():
    s = make([(0, "one two three."), (1.2, "four five six."), (2.4, "seven eight nine.")])
    a, b = padded_edges(s, 1, 1)
    assert s[0].words[-1].end < a < s[1].words[0].start
    assert s[1].words[-1].end < b < s[2].words[0].start


# ------------------------------------------------------------------ bug fixes
def test_phrase_count_is_whole_word():
    assert _count_phrases(" i think so ", ["link"]) == 0          # was 1 ("think")
    assert _count_phrases(" press the button ", ["but"]) == 0
    assert _count_phrases(" link in bio ", ["link"]) == 1


def test_score_scale_uses_whole_answer():
    assert _scale_for([8, 72, 90]) == 1.0                          # 8 stays 8, not 80
    assert _scale_for([7, 9, 3]) == 10.0
    assert _normalize_scores({"hook": 8, "retention": 70}, 1.0)["hook"] == 8


def test_hype_map_short_stream_does_not_crash():
    s = make([(5, "wow no way bro what"), (20, "lets go clutch")])
    m = hype_map(flat_audio(70), s, 70)                           # crashed under 120 s
    assert m.shape == (71,)


def test_hype_envelope_follows_the_moment():
    score = np.zeros(200)
    score[100:104] = 0.9                                           # a short spike
    a, b = hype_envelope(score, 101, 0, 200)
    score2 = np.zeros(200)
    score2[90:130] = 0.9                                           # a long hot stretch
    a2, b2 = hype_envelope(score2, 101, 0, 200)
    assert (b2 - a2) > (b - a)


def test_stub_not_merged_across_long_pause():
    seg = {"words": [Word(0.0, 0.4, "Hello"), Word(0.5, 0.9, "there"), Word(1.0, 1.4, "friend."),
                     Word(6.0, 6.3, "Yeah.")]}
    sents = build_sentences([seg])
    assert sents[0].end < 2.0


def test_housekeeping_is_penalised():
    words = [Word(i * 0.3, i * 0.3 + 0.2, w) for i, w in enumerate(
        "Smash that like button and subscribe, link in the description.".split())]
    bad = score_span(" ".join(w.text for w in words), words, 15, 0, 15)
    assert any("plug" in r for r in bad.risks)


def test_select_rules_only_end_to_end():
    lines = []
    t = 0.0
    for i in range(40):
        text = (f"Point {i} is that most people get pricing wrong because they never ask customers."
                if i % 3 else f"Why do {i} percent of startups fail in the first year?")
        lines.append((t, text))
        t += 5.5
    s = make(lines)
    clips, model = select(s, flat_audio(t + 5), k=3, min_len=12, max_len=90, llm=None,
                          auto_count=True)
    assert model == "rules" and clips
    for c in clips:
        assert 12 - 0.2 <= c.end - c.start <= 90 + 3.5
