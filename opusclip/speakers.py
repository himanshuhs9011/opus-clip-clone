"""Who speaks when (speaker diarization) from the audio.

Words are grouped into short speech chunks, each chunk gets a voice embedding from the
WeSpeaker ResNet34 model (ONNX, 26 MB, downloaded once into models/), and the embeddings are
clustered. Every word ends up with a speaker id (`Word.sp`), which drives speaker-coloured
captions and the "follow the active speaker" camera in reframe.py.
"""
from __future__ import annotations

import threading
import urllib.request
from pathlib import Path

import numpy as np

from .config import settings
from .models import Sentence

MODEL_NAME = "voxceleb_resnet34_LM.onnx"
MODEL_URL = ("https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM/resolve/main/"
             + MODEL_NAME)
CHUNK_FRAMES = 200          # 2.0 s of 10 ms fbank frames per embedding
_sess = None
_sess_lock = threading.Lock()


def _model() -> Path | None:
    dst = settings.model_dir / MODEL_NAME
    if dst.exists() and dst.stat().st_size > 1_000_000:
        return dst
    try:
        urllib.request.urlretrieve(MODEL_URL, dst)
        return dst if dst.stat().st_size > 1_000_000 else None
    except Exception:
        dst.unlink(missing_ok=True)
        return None


def _session():
    global _sess
    with _sess_lock:
        if _sess is None:
            import onnxruntime as ort

            path = _model()
            if path is None:
                return None
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 8
            _sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        return _sess


_MEL: np.ndarray | None = None


def _mel_banks(n_mels: int = 80, n_fft: int = 512, sr: int = 16000,
               low: float = 20.0, high: float = 8000.0) -> np.ndarray:
    """Kaldi-style triangular mel filters (same maths as torchaudio.compliance.kaldi)."""
    def mel(f):
        return 1127.0 * np.log(1.0 + np.asarray(f) / 700.0)

    n_bins = n_fft // 2
    fft_bin_width = sr / n_fft
    mel_low, mel_high = mel(low), mel(high)
    delta = (mel_high - mel_low) / (n_mels + 1)
    freqs = mel(fft_bin_width * np.arange(n_bins))
    banks = np.zeros((n_mels, n_bins + 1), np.float32)
    for m in range(n_mels):
        left, center, right = mel_low + m * delta, mel_low + (m + 1) * delta, mel_low + (m + 2) * delta
        up = (freqs - left) / (center - left)
        down = (right - freqs) / (right - center)
        banks[m, :n_bins] = np.maximum(0.0, np.minimum(up, down))
    return banks


def _fbank(wave: np.ndarray) -> np.ndarray:
    """80-dim Kaldi log-mel filterbank + CMN, the input WeSpeaker was trained on.
    Pure numpy: matches torchaudio.compliance.kaldi.fbank(dither=0, hamming) without PyTorch."""
    global _MEL
    if _MEL is None:
        _MEL = _mel_banks()
    x = wave.astype(np.float64) * (1 << 15)
    flen, shift, n_fft = 400, 160, 512
    if len(x) < flen:
        x = np.pad(x, (0, flen - len(x)))
    n = 1 + (len(x) - flen) // shift
    idx = np.arange(flen)[None, :] + shift * np.arange(n)[:, None]
    frames = x[idx]
    frames = frames - frames.mean(axis=1, keepdims=True)                  # remove DC
    frames = np.concatenate([frames[:, :1] - 0.97 * frames[:, :1],         # pre-emphasis
                             frames[:, 1:] - 0.97 * frames[:, :-1]], axis=1)
    frames = frames * np.hamming(flen)[None, :]
    spec = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2
    feats = np.log(np.maximum(spec @ _MEL.T.astype(np.float64), np.finfo(np.float32).eps))
    feats = feats - feats.mean(axis=0, keepdims=True)                     # CMN
    return feats.astype(np.float32)


def _fixed(feats: np.ndarray, n: int = CHUNK_FRAMES) -> np.ndarray:
    if len(feats) >= n:
        s = (len(feats) - n) // 2
        return feats[s : s + n]
    reps = int(np.ceil(n / max(1, len(feats))))
    return np.tile(feats, (reps, 1))[:n]


def _chunks(sentences: list[Sentence]) -> list[tuple[float, float, list[tuple[int, int]]]]:
    """Contiguous speech chunks of ~1-3 s; each remembers which (sentence, word) it covers."""
    out = []
    cur: list[tuple[int, int]] = []
    c0 = c1 = 0.0
    for si, s in enumerate(sentences):
        for wi, w in enumerate(s.words):
            gap = w.start - c1 if cur else 0.0
            long_enough = cur and (c1 - c0) >= 1.6
            if cur and (gap > 0.45 or (c1 - c0) >= 3.0 or (long_enough and w.text[:1].isupper())):
                out.append((c0, c1, cur))
                cur = []
            if not cur:
                c0 = w.start
            cur.append((si, wi))
            c1 = w.end
            if w.text.strip()[-1:] in ".?!" and (c1 - c0) >= 1.0:
                out.append((c0, c1, cur))
                cur = []
    if cur:
        out.append((c0, c1, cur))
    return out


def diarize(wav_path: str | Path, sentences: list[Sentence], log=None,
            max_speakers: int = 6, threshold: float = 0.62) -> dict:
    """Label every word with a speaker id. Returns {"count", "seconds", "share"}."""
    import soundfile as sf
    from sklearn.cluster import AgglomerativeClustering

    sess = _session()
    chunks = _chunks(sentences)
    if sess is None or len(chunks) < 2:
        for s in sentences:
            for w in s.words:
                w.sp = 0
        return {"count": 1, "share": [1.0], "method": "none"}

    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    feats = []
    for c0, c1, _ in chunks:
        a = int(max(0.0, c0 - 0.05) * sr)
        b = int(min(len(audio) / sr, c1 + 0.05) * sr)
        seg = audio[a:b]
        if len(seg) < sr * 0.3:
            seg = np.pad(seg, (0, int(sr * 0.3) - len(seg)))
        feats.append(_fixed(_fbank(seg)))
    feats_arr = np.stack(feats).astype(np.float32)
    embs = []
    for i in range(0, len(feats_arr), 64):
        embs.append(sess.run(None, {"feats": feats_arr[i : i + 64]})[0])
    emb = np.concatenate(embs)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    durs = np.array([c1 - c0 for c0, c1, _ in chunks])

    # cluster the reliable (longer) chunks first, then attach short ones to the nearest centroid
    good = durs >= 1.0
    if good.sum() < 2:
        good = np.ones(len(chunks), bool)
    cl = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, metric="cosine",
                                 linkage="average").fit(emb[good])
    labels_good = cl.labels_
    ids = sorted(set(labels_good), key=lambda k: -durs[good][labels_good == k].sum())
    cents = np.stack([emb[good][labels_good == k].mean(0) for k in ids])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True) + 1e-9

    # drop tiny clusters (< 4% of speech) and cap the speaker count
    total = durs[good].sum()
    keep = [i for i, k in enumerate(ids) if durs[good][labels_good == k].sum() >= 0.04 * total][:max_speakers]
    keep = keep or [0]
    cents = cents[keep]
    lab = np.argmax(emb @ cents.T, axis=1)

    # smooth: a lone chunk between two chunks of the same other speaker is usually wrong
    for i in range(1, len(lab) - 1):
        if lab[i - 1] == lab[i + 1] != lab[i] and durs[i] < 1.5:
            lab[i] = lab[i - 1]

    # renumber by first appearance -> speaker 0 talks first
    order: dict[int, int] = {}
    for x in lab:
        order.setdefault(int(x), len(order))
    lab = np.array([order[int(x)] for x in lab])

    for (c0, c1, members), sp in zip(chunks, lab):
        for si, wi in members:
            sentences[si].words[wi].sp = int(sp)
    n = int(lab.max()) + 1
    share = [round(float(durs[lab == k].sum() / max(1e-6, durs.sum())), 3) for k in range(n)]
    if log:
        log(f"speakers: {n} detected ({', '.join(f'S{k + 1} {s:.0%}' for k, s in enumerate(share))})")
    return {"count": n, "share": share, "method": "wespeaker"}


def speaker_at(words: list, t: float, tol: float = 0.35) -> int | None:
    """Speaker talking at absolute time t (words sorted by start)."""
    lo, hi = 0, len(words) - 1
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if words[mid].start <= t:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    for i in (best, (best or 0) + 1):
        if i is None or i >= len(words) or i < 0:
            continue
        w = words[i]
        if w.start - tol <= t <= w.end + tol and getattr(w, "sp", -1) >= 0:
            return w.sp
    return None
