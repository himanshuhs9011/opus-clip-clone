"""Synthetic two-person, multi-camera podcast for testing speaker features (Windows only).

    python tools/make_podcast_fixture.py fixtures/podcast2.mp4

Two different Windows voices take turns. The picture cuts between a wide two-shot and
close-ups like an edited podcast, and each face's mouth moves only while that person talks.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
LINES = [
    ("A", "Welcome back to the show. Today I want to talk about the biggest mistake new founders make."),
    ("B", "Is it spending too much money on ads before they have a real product?"),
    ("A", "Close, but no. The biggest mistake is building for a year without talking to a single customer."),
    ("B", "That is brutal. I did exactly that with my first startup and it failed completely."),
    ("A", "Most people do. So here is the rule. Talk to ten customers before you write one line of code."),
    ("B", "Ten customers. That sounds simple, but almost nobody actually does it."),
    ("A", "Because it is uncomfortable. Rejection hurts more than a bug report."),
    ("B", "So what do you say on that first call? Give me the exact script."),
    ("A", "Ask how they solve the problem today, and what it costs them. Then shut up and listen."),
    ("B", "I love that. Shut up and listen might be the best business advice I have ever heard."),
    ("A", "And if nobody is paying to solve it today, that is your answer. Move on."),
    ("B", "Okay, that one is going on my wall. Thank you for coming on the show."),
]
VOICES = {"A": "Microsoft David Desktop", "B": "Microsoft Zira Desktop"}
W, H, FPS = 1920, 1080, 30


def tts(text: str, voice: str, out: Path) -> None:
    ps = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.SelectVoice('{voice}'); $s.Rate = 1; "
        f"$s.SetOutputToWaveFile('{out}'); $s.Speak(\"{text}\"); $s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)


def load_faces() -> dict[str, np.ndarray]:
    from skimage import data

    a = cv2.imread(str(ROOT / "fixtures" / "face.jpg"))
    b = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    return {"A": a, "B": b}


def mouth_of(img: np.ndarray) -> tuple[int, int, int]:
    det = cv2.FaceDetectorYN.create(str(ROOT / "models" / "face_detection_yunet_2023mar.onnx"),
                                    "", (img.shape[1], img.shape[0]), 0.6)
    _, faces = det.detect(img)
    f = max(faces, key=lambda r: r[2] * r[3])
    rx, ry, lx, ly = f[10], f[11], f[12], f[13]
    return int((rx + lx) / 2), int((ry + ly) / 2), int(abs(lx - rx))


def fit(img: np.ndarray, w: int, h: int) -> tuple[np.ndarray, float, int, int]:
    s = min(w / img.shape[1], h / img.shape[0])
    r = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
    canvas = np.full((h, w, 3), 24, np.uint8)
    ox, oy = (w - r.shape[1]) // 2, (h - r.shape[0]) // 2
    canvas[oy : oy + r.shape[0], ox : ox + r.shape[1]] = r
    return canvas, s, ox, oy


def main(out: str) -> None:
    out_p = Path(out)
    tmp = Path(tempfile.mkdtemp())
    sr = 16000
    pieces, turns, t = [], [], 0.6
    pieces.append(np.zeros(int(sr * 0.6), np.float32))
    for i, (who, text) in enumerate(LINES):
        wav = tmp / f"l{i}.wav"
        tts(text, VOICES[who], wav)
        a, r = sf.read(str(wav), dtype="float32")
        if a.ndim > 1:
            a = a.mean(1)
        if r != sr:
            a = np.interp(np.linspace(0, len(a), int(len(a) * sr / r)), np.arange(len(a)), a).astype(np.float32)
        turns.append((t, t + len(a) / sr, who))
        pieces += [a, np.zeros(int(sr * 0.35), np.float32)]
        t += len(a) / sr + 0.35
    audio = np.concatenate(pieces)
    sf.write(str(tmp / "mix.wav"), audio, sr)
    dur = len(audio) / sr

    # per-speaker loudness envelope at video rate -> mouth opening
    hop = sr // FPS
    env = {k: np.zeros(int(dur * FPS) + 2) for k in "AB"}
    for s0, s1, who in turns:
        for f in range(int(s0 * FPS), int(s1 * FPS)):
            seg = audio[f * hop : (f + 1) * hop]
            env[who][f] = min(1.0, float(np.sqrt(np.mean(seg ** 2))) * 9) if len(seg) else 0

    faces = load_faces()
    # shots: wide 0-35%, close-up A, close-up B, wide to the end
    shots = [(0.0, 0.35, "wide"), (0.35, 0.5, "A"), (0.5, 0.65, "B"), (0.65, 1.01, "wide")]
    layouts = {}
    for name in ("wide", "A", "B"):
        if name == "wide":
            frame = np.full((H, W, 3), 24, np.uint8)
            mouths = {}
            for k, x0 in (("A", 0), ("B", W // 2)):
                c, s, ox, oy = fit(faces[k], W // 2, H)
                frame[:, x0 : x0 + W // 2] = c
                mx, my, mw = mouth_of(faces[k])
                mouths[k] = (x0 + ox + int(mx * s), oy + int(my * s), int(mw * s))
        else:
            frame, s, ox, oy = fit(faces[name], W, H)
            mx, my, mw = mouth_of(faces[name])
            mouths = {name: (ox + int(mx * s), oy + int(my * s), int(mw * s))}
        layouts[name] = (frame, mouths)

    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", str(tmp / "mix.wav"),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(out_p)],
        stdin=subprocess.PIPE,
    )
    n = int(dur * FPS)
    for f in range(n):
        p = f / n
        shot = next(s for s in shots if s[0] <= p < s[1])[2]
        base, mouths = layouts[shot]
        frame = base.copy()
        for k, (mx, my, mw) in mouths.items():
            o = env[k][f]
            cv2.ellipse(frame, (mx, my + int(mw * 0.1)), (int(mw * 0.42), max(1, int(mw * 0.05 + mw * 0.32 * o))),
                        0, 0, 360, (30, 20, 60), -1)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"wrote {out_p} ({dur:.1f}s)")
    for s0, s1, who in turns:
        print(f"  {s0:6.2f}-{s1:6.2f} {who}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "fixtures" / "podcast2.mp4"))
