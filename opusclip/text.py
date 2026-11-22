from __future__ import annotations

import re
from pathlib import Path

from .models import Sentence, Word
from .probe import run

SENT_END = ".!?…"
_STRIP = re.compile(r"[^\w\s'’-]", flags=re.UNICODE)

DEFAULT_FILLERS = {
    "en": {
        "um", "umm", "uh", "uhh", "er", "erm", "ah", "hmm", "mm", "mhm",
        "like", "basically", "literally", "actually", "honestly", "right",
        "okay", "ok", "so", "well", "anyway", "anyways", "stuff", "things",
        "kind", "sort", "just", "really", "very",
    },
    "ru": {"эм", "ээ", "ну", "вот", "как", "бы", "типа", "короче", "значит", "это"},
}


def normalize(text: str) -> str:
    return _STRIP.sub("", text.lower().strip())


def tokens(text: str) -> list[str]:
    return [t for t in normalize(text).split() if t]


def _has_video(path: Path) -> bool:
    try:
        from .probe import probe

        return probe(path).get("width", 0) > 0
    except Exception:
        return False


def download(url: str, dst: Path, audio_only: bool = False) -> Path:
    """Fetch a remote video (YouTube / direct URL) with yt-dlp, preferring real video."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    base = dst.with_suffix("")
    template = str(base) + ".%(ext)s"

    fmt = (
        "ba/b"
        if audio_only
        else "bv*[ext=mp4][height<=1080]+ba[ext=m4a]/"
        "bv*[height<=1080]+ba/b[height<=1080]/b"
    )
    run(
        [
            "yt-dlp",
            "-f",
            fmt,
            "--merge-output-format",
            "mp4",
            "--no-playlist",
            "--restrict-filenames",
            "--no-part",
            "-o",
            template,
            url,
        ],
        timeout=3600,
    )

    files = [p for p in base.parent.glob(base.name + ".*") if p.suffix.lower() in
             {".mp4", ".mkv", ".webm", ".mov", ".m4a", ".mp3", ".wav"}]
    files.sort(key=lambda p: p.stat().st_size, reverse=True)
    if not files:
        raise FileNotFoundError(f"download produced no file: {url}")

    if audio_only:
        return files[0]
    good = next((p for p in files if _has_video(p)), None)
    if good is None:
        for p in files:
            p.unlink(missing_ok=True)
        raise RuntimeError(
            "yt-dlp returned audio only - YouTube is blocking video formats for this "
            "video/IP. Download the MP4 manually and upload the file instead."
        )
    for p in files:
        if p != good and p.suffix.lower() != ".mp4":
            p.unlink(missing_ok=True)
    return good


def _add_cuda_dlls() -> None:
    """On Windows, make pip-installed CUDA/cuDNN DLLs (nvidia-* wheels, torch) visible to ctranslate2."""
    import os
    import site

    if os.name != "nt":
        return
    roots = list(site.getsitepackages())
    try:
        roots.append(site.getusersitepackages())
    except Exception:
        pass
    for root in roots:
        for sub in (Path(root) / "nvidia").glob("*/bin"):
            try:
                os.add_dll_directory(str(sub))
            except Exception:
                pass
            os.environ["PATH"] = str(sub) + os.pathsep + os.environ.get("PATH", "")
        torch_lib = Path(root) / "torch" / "lib"
        if torch_lib.is_dir():
            try:
                os.add_dll_directory(str(torch_lib))
            except Exception:
                pass


def _cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def transcribe(
    wav: str | Path,
    model_size: str = "large-v3-turbo",
    device: str = "auto",
    compute: str = "auto",
    language: str | None = None,
    progress=None,
    log=None,
):
    """Whisper (faster-whisper) with word timestamps. GPU batched when possible, CPU fallback."""
    import gc

    import soundfile as sf

    # decode here: PyAV >= 19 broke faster-whisper's own decoder
    audio, sr = sf.read(str(wav), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        raise ValueError(f"expected 16 kHz audio, got {sr}")

    _add_cuda_dlls()
    if device == "auto":
        devices = ["cuda", "cpu"] if _cuda_available() else ["cpu"]
    else:
        devices = [device]

    last_err: Exception | None = None
    for dev in devices:
        try:
            out = _transcribe_on(audio, model_size, dev, compute, language or None, progress, log)
            gc.collect()
            return out
        except Exception as e:  # missing cuDNN, out of VRAM ... -> next device
            last_err = e
            if log:
                log(f"Whisper on {dev} failed: {str(e)[:200]}")
            gc.collect()
    raise RuntimeError(f"transcription failed: {last_err}")


def _transcribe_on(audio, model_size, device, compute, language, progress, log):
    from faster_whisper import WhisperModel

    if compute == "auto":
        compute = "float16" if device == "cuda" else "int8"
    if log:
        log(f"Whisper {model_size} on {device} ({compute})")
    model = WhisperModel(model_size, device=device, compute_type=compute)
    kwargs = dict(
        language=language,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 350, "speech_pad_ms": 120},
        beam_size=5,
        condition_on_previous_text=False,
    )
    seg_iter = info = None
    if device == "cuda":
        try:  # batched inference is 3-6x faster on GPU
            from faster_whisper import BatchedInferencePipeline

            pipe = BatchedInferencePipeline(model=model)
            seg_iter, info = pipe.transcribe(audio, batch_size=12, **kwargs)
        except Exception as e:
            if log:
                log(f"batched whisper unavailable ({str(e)[:80]}), using sequential")
            seg_iter = None
    if seg_iter is None:
        seg_iter, info = model.transcribe(audio, **kwargs)

    raw = []
    total = info.duration or 0.0
    for seg in seg_iter:
        words = []
        for w in seg.words or []:
            t = w.word.strip()
            if not t:
                continue
            words.append(Word(start=max(0.0, w.start), end=max(w.start + 0.02, w.end), text=t,
                              prob=float(getattr(w, "probability", 0.0) or 0.0)))
        if not words:
            continue
        raw.append(
            {
                "start": words[0].start,
                "end": words[-1].end,
                "text": seg.text.strip(),
                "words": words,
            }
        )
        if progress and total:
            pct = min(1.0, max(0.0, seg.end / total))
            progress(pct, f"transcribing… {int(pct * 100)}%")

    # whisper occasionally emits overlapping word timings -> captions would overlap
    flat = [w for seg in raw for w in seg["words"]]
    for i in range(len(flat) - 1):
        if flat[i].end > flat[i + 1].start:
            flat[i].end = max(flat[i].start + 0.05, flat[i + 1].start)
    for w in flat:
        if w.end <= w.start:
            w.end = w.start + 0.08
    for seg in raw:
        seg["start"] = seg["words"][0].start
        seg["end"] = seg["words"][-1].end

    return {
        "language": info.language or "en",
        "duration": total,
        "segments": raw,
    }


def build_sentences(segments: list[dict], max_words: int = 20, max_gap: float = 0.85) -> list[Sentence]:
    """Group whisper segments into sentence-ish units with word timings."""
    units: list[dict] = []
    for seg in segments:
        cur: list[Word] = []
        for i, w in enumerate(seg["words"]):
            cur.append(w)
            txt = w.text.strip()
            nxt_gap = (
                seg["words"][i + 1].start - w.end if i + 1 < len(seg["words"]) else 0.0
            )
            ends = bool(txt) and txt[-1] in SENT_END
            if ends or nxt_gap > max_gap or len(cur) >= max_words:
                units.append(_make_unit(cur))
                cur = []
        if cur:
            units.append(_make_unit(cur))

    # merge stubs into the next/previous unit
    merged: list[dict] = []
    for u in units:
        # a stub only joins the sentence right before it: merging across a long pause made
        # that sentence (and every clip ending on it) run through the dead air
        if (merged and len(u["text"].split()) < 4 and u["end"] - u["start"] < 1.2
                and u["start"] - merged[-1]["end"] <= max_gap):
            prev = merged[-1]
            prev["words"].extend(u["words"])
            prev["end"] = u["end"]
            prev["text"] = (prev["text"] + " " + u["text"]).strip()
        else:
            merged.append(u)

    sentences: list[Sentence] = []
    for i, u in enumerate(merged):
        sentences.append(
            Sentence(id=i, start=u["start"], end=u["end"], text=u["text"], words=u["words"])
        )
    return sentences


def _make_unit(words: list[Word]) -> dict:
    text = " ".join(w.text.strip() for w in words)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text).strip()
    return {"start": words[0].start, "end": words[-1].end, "text": text, "words": list(words)}


def sentence_span(sentences: list[Sentence], start_sid: int, end_sid: int) -> tuple[float, float, list[Word]]:
    start_sid = max(0, min(start_sid, len(sentences) - 1))
    end_sid = max(start_sid, min(end_sid, len(sentences) - 1))
    span = sentences[start_sid:end_sid + 1]
    words = [w for s in span for w in s.words]
    return span[0].start, span[-1].end, words
