"""CLI entry point.

    python run.py input.mp4 --clips 6 --style bold-pop
    python run.py "https://youtube.com/watch?v=..." --clips 3 --provider openrouter
    python run.py input.mp4 --provider none --model small --length short
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from opusclip import pipeline
from opusclip.captions import PRESETS
from opusclip.config import settings


def main() -> int:
    ap = argparse.ArgumentParser(description="Turn long video into short vertical clips.")
    ap.add_argument("input", help="local video file or URL")
    ap.add_argument("--clips", type=int, default=settings.max_clips, help="how many clips")
    ap.add_argument("--length", default="ai", choices=list(pipeline.LENGTHS),
                    help="ai (default) lets the AI pick each clip's length")
    ap.add_argument("--style", default=settings.caption_style, choices=list(PRESETS))
    ap.add_argument("--aspect", default="9:16", choices=["9:16", "1:1", "4:5", "16:9"])
    ap.add_argument("--layout", default="auto", choices=["auto", "track", "center", "fit"])
    ap.add_argument("--model", default=settings.whisper_model,
                    help="whisper size: tiny/base/small/medium/large-v3-turbo/large-v3")
    ap.add_argument("--lang", default="", help="language code, blank = auto-detect")
    ap.add_argument("--provider", default=settings.llm_provider,
                    choices=["ollama", "openrouter", "openai", "none"], help="AI used to pick clips")
    ap.add_argument("--llm-model", default=None, help="override the provider's model")
    ap.add_argument("--topic", default="", help="optional niche hint for the selector")
    ap.add_argument("--no-captions", action="store_true")
    ap.add_argument("--no-zoom", action="store_true")
    ap.add_argument("--remove-silence", action="store_true")
    ap.add_argument("--quality", default="good", choices=list(pipeline.QUALITY),
                    help="drop clips below the bar: any=0, good=65, viral=75")
    args = ap.parse_args()

    src = Path(args.input)
    is_link = str(args.input).startswith(("http://", "https://"))
    if not is_link and not src.exists():
        print(f"file not found: {args.input}")
        return 2

    opts = {
        "clips": args.clips, "length": args.length, "preset": args.style, "aspect": args.aspect,
        "layout": args.layout, "whisper_model": args.model, "language": args.lang,
        "provider": args.provider, "llm_model": args.llm_model, "topic": args.topic,
        "captions": not args.no_captions, "zoom": not args.no_zoom, "silence": args.remove_silence, "quality": args.quality,
    }
    p = pipeline.new_project(Path("remote.mp4") if is_link else src,
                             name=args.input[:80] if is_link else None, options=opts)
    if is_link:   # audio first, then only the video around each clip
        p.remote = {"url": args.input}
    print(f"[..] project {p.id}  ->  {pipeline.project_dir(p.id)}")
    pipeline.run(p, None if is_link else src, log=lambda m: print("[..]", m))

    out = pipeline.project_dir(p.id)
    print("\n=== DONE ===" if p.status == "done" else f"\n=== {p.status.upper()}: {p.error} ===")
    for c in sorted(p.clips, key=lambda c: c.rank):
        s = f"{c.score.total} {c.score.grade}" if c.score else "?"
        print(f"  #{c.rank} {c.start:7.1f}-{c.end:7.1f} ({c.duration:4.1f}s)  score {s:7}  {c.title}")
        if c.file:
            print(f"       {out / c.file}   [{c.render.get('seconds')}s, {c.render.get('encoder')}]")
    print("timings:", p.timings)
    return 0 if p.status == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
