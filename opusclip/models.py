from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Word:
    start: float
    end: float
    text: str
    prob: float = 0.0
    sp: int = -1          # speaker id from diarization, -1 = unknown

    def to_dict(self) -> dict:
        d = {"s": round(self.start, 3), "e": round(self.end, 3), "w": self.text}
        if self.sp >= 0:
            d["sp"] = self.sp
        return d


@dataclass
class Sentence:
    id: int
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.01, self.end - self.start)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "words": [w.to_dict() for w in self.words],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Sentence":
        return cls(
            id=d["id"],
            start=d["start"],
            end=d["end"],
            text=d["text"],
            words=[Word(start=w["s"], end=w["e"], text=w["w"], sp=w.get("sp", -1)) for w in d.get("words", [])],
        )


@dataclass
class Score:
    total: int
    hook: int
    retention: int
    emotion: int
    value: int
    quotability: int
    reasons: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)
    source: str = "heuristic"
    grade: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Score":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


@dataclass
class Clip:
    id: str
    start: float
    end: float
    title: str = ""
    hook: str = ""
    summary: str = ""
    why: str = ""
    hashtags: list[str] = field(default_factory=list)
    banner: str = ""          # on-screen context line (what the clip is about)
    score: Score | None = None
    rank: int = 0
    file: str | None = None
    thumb: str | None = None
    version: int = 0
    track: dict = field(default_factory=dict)      # face path, see reframe.track
    edit: dict = field(default_factory=dict)       # user edits, see edit.DEFAULT_EDIT
    render: dict = field(default_factory=dict)     # {status, progress, error, seconds, encoder}
    error: str | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 2),
            "title": self.title,
            "hook": self.hook,
            "summary": self.summary,
            "why": self.why,
            "hashtags": self.hashtags,
            "banner": self.banner,
            "score": self.score.to_dict() if self.score else None,
            "rank": self.rank,
            "file": self.file,
            "thumb": self.thumb,
            "version": self.version,
            "track": self.track,
            "edit": self.edit,
            "render": self.render,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, c: dict) -> "Clip":
        clip = cls(
            id=c["id"], start=c["start"], end=c["end"], title=c.get("title", ""),
            hook=c.get("hook", ""), summary=c.get("summary", ""), why=c.get("why", ""),
            hashtags=c.get("hashtags") or [], banner=c.get("banner") or "", rank=c.get("rank", 0),
            file=c.get("file"),
            thumb=c.get("thumb"), version=c.get("version", 0), track=c.get("track") or {},
            edit=c.get("edit") or {}, render=c.get("render") or {}, error=c.get("error"),
        )
        if c.get("score"):
            clip.score = Score.from_dict(c["score"])
        return clip


@dataclass
class Project:
    id: str
    source: str
    name: str
    created: float = field(default_factory=time.time)
    status: str = "queued"  # queued | running | done | error
    stage: str = "queued"
    message: str = ""
    progress: float = 0.0
    error: str | None = None
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    has_audio: bool = True
    language: str = ""
    preview: str | None = None     # browser-playable source for the editor
    speakers: dict = field(default_factory=dict)   # diarization summary {count, share}
    remote: dict = field(default_factory=dict)     # link projects: stream info (no full download)
    segments: list = field(default_factory=list)   # downloaded video pieces [{file, t0, t1}]
    sentences: list[Sentence] = field(default_factory=list)
    clips: list[Clip] = field(default_factory=list)
    settings: dict = field(default_factory=dict)
    options: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)

    def to_dict(self, include_transcript: bool = True) -> dict:
        d = {
            "id": self.id,
            "name": self.name,
            "created": self.created,
            "status": self.status,
            "stage": self.stage,
            "message": self.message,
            "progress": round(self.progress, 4),
            "error": self.error,
            "duration": round(self.duration, 3),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "has_audio": self.has_audio,
            "language": self.language,
            "preview": self.preview,
            "speakers": self.speakers,
            "remote": self.remote,
            "segments": self.segments,
            "clips": [c.to_dict() for c in self.clips],
            "settings": self.settings,
            "options": self.options,
            "timings": self.timings,
        }
        if include_transcript:
            d["sentences"] = [s.to_dict() for s in self.sentences]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Project":
        p = cls(id=d["id"], source=d.get("source", ""), name=d.get("name", d["id"]),
                created=d.get("created", 0))
        for k in ("status", "stage", "message", "progress", "error", "duration", "width",
                  "height", "fps", "has_audio", "language", "preview", "speakers", "remote", "segments", "settings", "options",
                  "timings"):
            if k in d and d[k] is not None:
                setattr(p, k, d[k])
        p.sentences = [Sentence.from_dict(s) for s in d.get("sentences", [])]
        p.clips = [Clip.from_dict(c) for c in d.get("clips", [])]
        return p
