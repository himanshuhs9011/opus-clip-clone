from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .probe import run


def extract_wav(src: str | Path, dst: str | Path, sr: int = 16000, mono: bool = True) -> Path:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-vn",
            "-ac",
            "1" if mono else "2",
            "-ar",
            str(sr),
            "-c:a",
            "pcm_s16le",
            str(dst),
        ]
    )
    return dst


@dataclass
class AudioFeatures:
    """Per-50ms loudness envelope, normalised 0..1 (p95 == 1.0)."""

    hop: float
    rms: np.ndarray
    db: np.ndarray
    duration: float

    @classmethod
    def from_wav(cls, wav: str | Path, hop: float = 0.05) -> "AudioFeatures":
        data, sr = sf.read(str(wav), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if data.size == 0:
            return cls(hop=hop, rms=np.zeros(1, dtype=np.float32),
                       db=np.full(1, -90.0, dtype=np.float32), duration=0.0)
        win = max(1, int(sr * hop))
        n = max(1, len(data) // win)
        trimmed = data[: n * win].reshape(n, win)
        rms = np.sqrt(np.mean(trimmed**2, axis=1) + 1e-12).astype(np.float32)
        p95 = float(np.percentile(rms, 95)) or 1e-9
        norm = np.clip(rms / p95, 0.0, 1.5).astype(np.float32)
        db = (20.0 * np.log10(rms + 1e-9)).astype(np.float32)
        return cls(hop=hop, rms=norm, db=db, duration=len(data) / sr)

    # -- slicing ---------------------------------------------------------
    def _slice(self, t0: float, t1: float) -> np.ndarray:
        if self.rms.size == 0:
            return self.rms
        i0 = max(0, int(t0 / self.hop))
        i1 = min(self.rms.size, max(i0 + 1, int(t1 / self.hop)))
        return self.rms[i0:i1]

    def stats(self, t0: float, t1: float) -> dict:
        seg = self._slice(t0, t1)
        if seg.size == 0:
            return {"mean": 0.0, "std": 0.0, "silence": 0.0, "peak": 0.0, "onset": 0.0}
        quiet = float(np.mean(seg < 0.10))
        onset = float(np.mean(seg[: max(1, int(0.6 / self.hop))] >= 0.12))
        return {
            "mean": float(np.mean(seg)),
            "std": float(np.std(seg)),
            "silence": quiet,
            "peak": float(np.percentile(seg, 95)),
            "onset": onset,
        }

    def mean_db(self, t0: float, t1: float) -> float:
        seg = self._slice(t0, t1)
        if seg.size == 0:
            return -90.0
        return float(20.0 * np.log10(float(np.mean(seg)) + 1e-9))


def measure_loudness(src: str | Path) -> dict:
    """Single-pass EBU R128 measurement (used for display only)."""
    try:
        out = run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostats",
                "-i",
                str(src),
                "-af",
                "loudnorm=print_format=json",
                "-f",
                "null",
                "-",
            ]
        )
    except Exception:
        return {}
    start = out.rfind("{")
    if start < 0:
        return {}
    try:
        import json

        return json.loads(out[start:])
    except Exception:
        return {}
