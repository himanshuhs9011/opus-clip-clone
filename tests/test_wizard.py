"""Clip Agent: reading plain-words requests.   python -m pytest tests"""
from __future__ import annotations

from opusclip.boundaries import focus_hits, focus_terms
from opusclip.pipeline import normalize_options
from opusclip.wizard import parse_request, plan, resolve_range, to_options, to_seconds


def p(text: str) -> dict:
    return parse_request(text)


def test_length_is_the_longest_a_clip_may_be():
    assert p("clips under 60 seconds")["max_clip"] == 60
    assert p("under a minute")["max_clip"] == 60
    assert p("30 second clips")["max_clip"] == 30
    assert p("make clips 2 minute long")["max_clip"] == 120
    assert p("Turn this into Shorts")["max_clip"] is None          # a platform is not a length
    r = p("15-30 second clips")
    assert (r["min_clip"], r["max_clip"]) == (15, 30)
    assert p("at least 45 seconds each")["min_clip"] == 45


def test_count_only_when_a_number_of_clips_is_said():
    assert p("about 10 clips")["count"] == 10
    assert p("top 5")["count"] == 5
    assert p("30 second clips")["count"] is None                   # a length, not a count
    assert p("viral clips for tiktok")["count"] is None


def test_part_of_the_video():
    assert p("between 10:00 and 25:00")["range"] == {"from": 600.0, "to": 1500.0, "last": None}
    assert p("from 1:20:00")["range"]["from"] == 4800
    assert p("the last hour")["range"]["last"] == 3600
    assert p("in the first 30 minutes")["range"]["to"] == 1800
    r = p("skip the first 10 minutes")["range"]
    assert r["from"] == 600 and r["to"] is None
    assert resolve_range({"from": None, "to": None, "last": 3600}, 7200) == (3600, 7200)
    assert resolve_range({"from": 100, "to": 105, "last": None}, 7200) is None   # too short: whole video
    assert to_seconds("1h 20m") == 4800


def test_focus_is_only_a_kind_of_moment_the_user_named():
    assert p("only the biggest laughs, under a minute")["focus"] == "biggest laughs"
    assert p("every time someone gets voted out between 10:00 and 25:00")["focus"] == "someone gets voted out"
    assert p("only the parts about customers, under 30 seconds")["focus"] == "customers"
    assert p("only the part about the first call")["focus"] == "the first call"
    assert p("best viral moments for shorts")["focus"] == ""


def test_make_is_a_whole_message():
    assert p("make all")["make"] == "all"
    assert p("ok make the top 3")["make"] == 3
    assert p("make 5 clips about pricing")["make"] is None        # a new request


def test_options_round_trip_through_the_pipeline():
    o = normalize_options(to_options(p("only clutch plays, under 45s, 12 clips, from 1:00:00, square")))
    assert o["focus"] == "clutch plays" and o["max_clip"] == 45 and o["clips"] == 12
    assert o["range"]["from"] == 3600 and o["aspect"] == "1:1" and o["pick"] is True
    bad = normalize_options({"max_clip": "x", "range": "nope", "focus": 5})
    assert bad["max_clip"] is None and bad["range"] is None and bad["focus"] == "5"


def test_plan_asks_for_the_video_when_missing():
    pl = plan(p("only the funny parts"), has_source=False)
    assert not pl["ready"] and pl["questions"]
    assert plan(p("https://youtu.be/x only the funny parts"), has_source=False)["ready"]


def test_focus_matching_uses_word_stems():
    terms = focus_terms("the funniest fails")
    assert focus_hits("he failed so badly", terms) == 1
    assert focus_hits("nothing related here", terms) == 0


# Every sample prompt on the website (site/public/index.html, "Prompt guide") must do what
# the site says it does.
SITE = [
    ("Clip this podcast into clips under 45 seconds, only the parts about money", dict(focus="money", max_clip=45)),
    ("Give me the 5 funniest moments, 30 to 60 second clips, karaoke captions", dict(count=5, min_clip=30, max_clip=60, preset="karaoke")),
    ("Make 10 minute YouTube videos from this Kick stream: https://kick.com/x", dict(format="long", long_len=600)),
    ("Make 3 long videos and shorts too from this stream", dict(format="both", count=3)),
    ("Best-of compilation, about 12 minutes, only the scary moments", dict(format="long", long_len=720, focus="scary")),
    ("Make long videos from the last 3 hours of the stream", dict(format="long", range={"from": None, "to": None, "last": 10800.0})),
    ("Make a 15 minute long video of the best story from this podcast", dict(format="long", long_len=900, focus="best story")),
    ("Only the arguments, 1 to 2 minute clips", dict(focus="arguments", min_clip=60, max_clip=120, range=None)),
    ("Only the first 30 minutes, 8 clips", dict(focus="", count=8, range={"from": None, "to": 1800.0, "last": None})),
    ("Is stream ke best moments ki 12 minute ki lambi video aur shorts dono bana do", dict(format="both", long_len=720)),
    ("सिर्फ मज़ेदार पल, 45 सेकंड से कम", dict(focus="मज़ेदार पल", max_clip=45)),
    ("sirf funny moments, 1 minute tak ke clips", dict(focus="funny", max_clip=60)),
    ("इस पॉडकास्ट की 10 मिनट की लंबी वीडियो बना दो", dict(format="long", long_len=600)),
    ("shorter, under 20 seconds", dict(max_clip=20)),
    ("from 10:00 to 20:00", dict(range={"from": 600.0, "to": 1200.0, "last": None})),
    ("Clip this podcast episode into shorts", dict(format="short")),
]


def test_site_prompts():
    for text, want in SITE:
        got = parse_request(text)
        for k, v in want.items():
            assert got[k] == v, (text, k, got[k], v)
