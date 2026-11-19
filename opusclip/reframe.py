"""Auto-reframe with active-speaker detection.

Per clip we sample frames with ffmpeg and, for every face, record its position, a face
fingerprint (OpenCV SFace) and how much its lips move. Faces are linked over time, grouped
into identities across camera angles, and matched to the voices from diarization
(speakers.py). The camera then cuts to whoever is talking, like an editor switching
cameras on a podcast, and wide two-person shots can be laid out as a split screen.

Everything is stored as normalised paths on the clip, so any aspect ratio can be cropped
later without re-tracking.
"""
from __future__ import annotations

import subprocess
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .config import settings
from .probe import NO_WINDOW

TRACK_VERSION = 5
DET_NAME = "face_detection_yunet_2023mar.onnx"
REC_NAME = "face_recognition_sface_2021dec.onnx"
ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models/"
URLS = {
    DET_NAME: [ZOO + "face_detection_yunet/" + DET_NAME],
    REC_NAME: [ZOO + "face_recognition_sface/" + REC_NAME],
}
SAMPLE_W = 800
ASD_FPS = 6.0           # samples per second; lip motion needs more than plain tracking
MIN_SHOT = 1.0          # seconds a speaker shot must last before we cut away


def _ensure(name: str, min_size: int = 100_000) -> Path | None:
    d = settings.model_dir
    d.mkdir(parents=True, exist_ok=True)
    dst = d / name
    if dst.exists() and dst.stat().st_size > min_size:
        return dst
    for url in URLS[name]:
        try:
            urllib.request.urlretrieve(url, dst)
            if dst.exists() and dst.stat().st_size > min_size:
                return dst
        except Exception:
            continue
    dst.unlink(missing_ok=True)
    return None


def ensure_face_model(model_dir: Path | None = None) -> Path | None:
    return _ensure(DET_NAME)


def _even(x: float) -> int:
    return max(2, int(round(x / 2.0)) * 2)


def crop_box(src_w: int, src_h: int, out_w: int, out_h: int) -> tuple[int, int, str]:
    """Largest out_w:out_h window inside the source, and the axis it can slide along."""
    dst_ar = out_w / out_h
    src_ar = src_w / max(1, src_h)
    if abs(src_ar - dst_ar) < 0.01:
        return _even(src_w), _even(src_h), "none"
    if src_ar > dst_ar:
        return min(_even(src_h * dst_ar), _even(src_w)), _even(src_h), "x"
    return _even(src_w), min(_even(src_w / dst_ar), _even(src_h)), "y"


# --------------------------------------------------------------------------
# detection + recognition
# --------------------------------------------------------------------------
class _Faces:
    def __init__(self) -> None:
        self.det = None
        self.rec = None
        p = _ensure(DET_NAME)
        if p is not None:
            try:
                self.det = cv2.FaceDetectorYN.create(str(p), "", (320, 320), 0.6, 0.3, 50)
            except Exception:
                self.det = None
        r = _ensure(REC_NAME, 1_000_000)
        if r is not None:
            try:
                self.rec = cv2.FaceRecognizerSF.create(str(r), "")
            except Exception:
                self.rec = None
        self.haar = None if self.det is not None else cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect(self, img: np.ndarray) -> list[dict]:
        h, w = img.shape[:2]
        out = []
        if self.det is not None:
            self.det.setInputSize((w, h))
            _, faces = self.det.detect(img)
            for f in faces if faces is not None else []:
                if float(f[14]) < 0.6:
                    continue
                x, y, fw, fh = (float(v) for v in f[:4])
                if fh / h < 0.03:
                    continue   # too small to matter even as a face-cam
                out.append({"cx": (x + fw / 2) / w, "cy": (y + fh / 2) / h, "w": fw / w,
                            "h": fh / h, "area": (fw * fh) / (w * h), "row": f.copy()})
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            for x, y, fw, fh in self.haar.detectMultiScale(gray, 1.1, 6, minSize=(30, 30)):
                out.append({"cx": (x + fw / 2) / w, "cy": (y + fh / 2) / h, "w": fw / w,
                            "h": fh / h, "area": (fw * fh) / (w * h), "row": None})
        return out

    def embed(self, img: np.ndarray, row) -> np.ndarray | None:
        if self.rec is None or row is None or row[2] < 36:
            return None
        try:
            crop = self.rec.alignCrop(img, row)
            e = self.rec.feature(crop).flatten().astype(np.float32)
            return e / (np.linalg.norm(e) + 1e-9)
        except Exception:
            return None


def _mouth(gray: np.ndarray, row) -> np.ndarray | None:
    """Normalised mouth patch, anchored on the mouth corners so head motion cancels out."""
    if row is None:
        return None
    rx, ry, lx, ly = float(row[10]), float(row[11]), float(row[12]), float(row[13])
    mw = float(np.hypot(lx - rx, ly - ry))
    if mw < 6:
        return None
    cx, cy = (rx + lx) / 2, (ry + ly) / 2 + 0.1 * mw
    x0, x1 = int(cx - 0.8 * mw), int(cx + 0.8 * mw)
    y0, y1 = int(cy - 0.5 * mw), int(cy + 0.55 * mw)
    if x0 < 0 or y0 < 0 or x1 > gray.shape[1] or y1 > gray.shape[0]:
        return None
    p = cv2.resize(gray[y0:y1, x0:x1], (24, 14), interpolation=cv2.INTER_AREA).astype(np.float32)
    return (p - p.mean()) / (p.std() + 6.0)


def _frames(src: Path, start: float, end: float, src_w: int, src_h: int, fps: float):
    h = _even(SAMPLE_W * src_h / max(1, src_w))
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-ss", f"{max(0.0, start):.3f}", "-t", f"{max(0.1, end - start):.3f}", "-i", str(src),
        "-an", "-sn", "-vf", f"fps={fps},scale={SAMPLE_W}:{h}:flags=bilinear",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]
    size = SAMPLE_W * h * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=size * 4, **NO_WINDOW)
    i = 0
    try:
        assert proc.stdout is not None
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield i / fps, np.frombuffer(buf, np.uint8).reshape(h, SAMPLE_W, 3)
            i += 1
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.kill()
        proc.wait()


# --------------------------------------------------------------------------
# path smoothing
# --------------------------------------------------------------------------
def _median(a: np.ndarray, k: int = 5) -> np.ndarray:
    if len(a) < 3:
        return a.copy()
    pad = k // 2
    p = np.pad(a, pad, mode="edge")
    return np.array([np.median(p[i : i + k]) for i in range(len(a))])


def _camera(a: np.ndarray, deadzone: float = 0.045, alpha: float = 0.28) -> np.ndarray:
    """Steady camera operator: hold still, move smoothly only when the subject really moves."""
    if len(a) == 0:
        return a
    out = np.empty_like(a)
    cam = a[0]
    for i, t in enumerate(a):
        if abs(t - cam) > deadzone:
            cam += (t - cam) * alpha
        out[i] = cam
    back = out.copy()
    for i in range(len(back) - 2, -1, -1):
        back[i] = back[i] * 0.6 + back[i + 1] * 0.4
    return back


def _simplify(ts: np.ndarray, vs: np.ndarray, tol: float = 0.006) -> list[list[float]]:
    """Ramer-Douglas-Peucker on a 1-D path -> few keyframes, same motion."""
    if len(ts) <= 2:
        return [[round(float(t), 3), round(float(v), 4)] for t, v in zip(ts, vs)]
    keep = np.zeros(len(ts), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(ts) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        seg_t = ts[a + 1 : b]
        lin = vs[a] + (vs[b] - vs[a]) * (seg_t - ts[a]) / max(1e-6, ts[b] - ts[a])
        err = np.abs(vs[a + 1 : b] - lin)
        k = int(np.argmax(err))
        if err[k] > tol:
            m = a + 1 + k
            keep[m] = True
            stack += [(a, m), (m, b)]
    return [[round(float(ts[i]), 3), round(float(vs[i]), 4)] for i in np.where(keep)[0]]


def _shot_keys(ts: np.ndarray, vals: np.ndarray, shots: np.ndarray, smooth: bool = True) -> list[list[float]]:
    """Smooth inside each shot, hard cut between shots."""
    keys: list[list[float]] = []
    for s in np.unique(shots):
        ids = np.where(shots == s)[0]
        v = vals[ids]
        if smooth:
            v = _camera(_median(v))
        seg = _simplify(ts[ids], v)
        if keys and seg:
            keys.append([round(max(keys[-1][0] + 0.001, seg[0][0] - 0.001), 3), keys[-1][1]])
        keys.extend(seg)
    if keys and keys[0][0] > 0:
        keys.insert(0, [0.0, keys[0][1]])
    return keys


def _runs_min(seq: list, min_len: int, breaks: set[int]) -> list:
    """Remove runs shorter than min_len (except at scene breaks) by extending the previous run."""
    out = list(seq)
    i = 0
    n = len(out)
    while i < n:
        j = i
        while j + 1 < n and out[j + 1] == out[i] and (j + 1) not in breaks:
            j += 1
        if i > 0 and (j - i + 1) < min_len and i not in breaks and out[i - 1] is not None:
            for k in range(i, j + 1):
                out[k] = out[i - 1]
        i = j + 1
    return out


# --------------------------------------------------------------------------
def track(src: str | Path, start: float, end: float, src_w: int, src_h: int,
          words: list | None = None, sample_fps: float | None = None, log=None,
          offset: float = 0.0) -> dict:
    """Analyse faces/speakers in [start, end] (absolute seconds). Times in the result are
    relative to `start` (t0). `offset` = absolute time of the file's 0 (downloaded pieces)."""
    from .speakers import speaker_at

    fps = max(ASD_FPS, sample_fps or settings.sample_fps)
    axis = "x" if src_w >= src_h * 0.6 else "y"
    key_i = "cx" if axis == "x" else "cy"
    base = {"v": TRACK_VERSION, "t0": round(start, 3), "t1": round(end, 3), "axis": axis,
            "coverage": 0.0, "scenes": 1, "keys": [[0.0, 0.5]], "faces": 0, "people": 0,
            "speaker_map": {}, "split": [], "switches": 0}
    fx = _Faces()
    words = sorted(words or [], key=lambda w: w.start)

    times: list[float] = []
    dets: list[list[dict]] = []
    cams: list[dict | None] = []     # largest face of any size per sample (face-cam finder)
    scene_of: list[int] = []
    prev_hist = prev_thumb = None
    scene = 0
    tracks: list[dict] = []          # {last_i, last, embs, scene}
    for i, (t, frame) in enumerate(_frames(Path(src), start - offset, end - offset, src_w, src_h, fps)):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
        if prev_hist is not None:
            corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            mad = float(np.mean(np.abs(thumb - prev_thumb)))
            if corr < 0.6 or mad > 26 or (mad > 16 and corr < 0.9):   # camera cut
                scene += 1
        prev_hist, prev_thumb = hist, thumb
        found = fx.detect(frame)
        big = max(found, key=lambda f: f["h"], default=None)
        cams.append({"cx": big["cx"], "cy": big["cy"], "h": big["h"], "w": big["w"]} if big else None)
        faces = [f for f in found if f["h"] >= 0.07]   # tiny ones are insets / audience
        # link to face tracks from the previous sample (same scene)
        live = [tr for tr in tracks if tr["scene"] == scene and tr["last_i"] == i - 1]
        used = set()
        for f in sorted(faces, key=lambda f: -f["area"]):
            best, bd = None, 1e9
            for k, tr in enumerate(live):
                if k in used:
                    continue
                d = np.hypot(f["cx"] - tr["last"]["cx"], f["cy"] - tr["last"]["cy"])
                if d < 0.7 * max(f["w"], tr["last"]["w"]) and d < bd:
                    best, bd = k, d
            mouth = _mouth(gray, f["row"])
            if best is not None:
                used.add(best)
                tr = live[best]
                pm = tr.get("mouth")
                f["motion"] = (float(np.mean(np.abs(mouth - pm)))
                               if mouth is not None and pm is not None else None)
            else:
                tr = {"embs": [], "scene": scene, "n": 0, "k": len(tracks)}
                tracks.append(tr)
                f["motion"] = None
            tr["last_i"], tr["last"], tr["mouth"] = i, f, mouth
            tr["n"] += 1
            f["track"] = tr["k"]
            if len(tr["embs"]) < 3 and tr["n"] % 3 == 1:
                e = fx.embed(frame, f["row"])
                if e is not None:
                    tr["embs"].append(e)
        for f in faces:
            f.pop("row", None)
        times.append(t)
        dets.append(faces)
        scene_of.append(scene)

    if not times:
        if log:
            log("could not read frames for tracking - centre crop")
        return base
    n = len(times)
    ts = np.array(times)
    coverage = sum(1 for d in dets if d) / n
    if coverage == 0:
        if log:
            log("tracking: no faces found")
        return {**base, "scenes": scene + 1}

    # ---------------------------------------------- identities across camera angles
    ident = list(range(len(tracks)))
    with_emb = [k for k, tr in enumerate(tracks) if tr["embs"]]
    if len(with_emb) >= 2:
        from sklearn.cluster import AgglomerativeClustering

        E = np.stack([np.mean(tracks[k]["embs"], 0) for k in with_emb])
        E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
        lab = AgglomerativeClustering(n_clusters=None, distance_threshold=0.62, metric="cosine",
                                      linkage="average").fit(E).labels_
        off = len(tracks)
        for k, l in zip(with_emb, lab):
            ident[k] = off + int(l)
    remap: dict[int, int] = {}
    for k in range(len(tracks)):
        ident[k] = remap.setdefault(ident[k], len(remap))
    for d in dets:
        for f in d:
            f["id"] = ident[f["track"]]
    presence: dict[int, int] = {}
    for d in dets:
        for f in d:
            presence[f["id"]] = presence.get(f["id"], 0) + 1
    people = [p for p, c in presence.items() if c >= max(3, 0.05 * n)]

    # ---------------------------------------------- who is talking at each sample
    sp_at = [speaker_at(words, start + t) if words else None for t in times]
    voices = {s for s in sp_at if s is not None}
    speaker_map: dict[int, int] = {}
    if len(voices) >= 2 and len(people) >= 2:
        # lips of the right person move more while their voice is heard
        stats: dict[tuple[int, int], list[float]] = {}
        base_m: dict[int, list[float]] = {}
        for d, s in zip(dets, sp_at):
            for f in d:
                if f.get("motion") is None:
                    continue
                base_m.setdefault(f["id"], []).append(f["motion"])
                if s is not None:
                    stats.setdefault((s, f["id"]), []).append(f["motion"])
        score = []
        for (s, pid), v in stats.items():
            if len(v) >= 3 and pid in people:
                score.append((float(np.mean(v)) - float(np.mean(base_m[pid])), s, pid))
        for sc, s, pid in sorted(score, reverse=True):
            if sc > 0 and s not in speaker_map and pid not in speaker_map.values():
                speaker_map[s] = pid

    # smoothed lip motion per person for the visual-only fallback
    lip: dict[int, np.ndarray] = {}
    for pid in people:
        arr = np.array([next((f["motion"] for f in d if f["id"] == pid and f.get("motion") is not None), np.nan)
                        for d in dets], dtype=float)
        if np.isfinite(arr).sum() >= 3:
            filled = np.where(np.isfinite(arr), arr, 0.0)
            lip[pid] = np.convolve(filled, np.ones(5) / 5, mode="same")

    target: list[int | None] = []
    prev = None
    for i, d in enumerate(dets):
        vis = {f["id"]: f for f in d}
        choice = None
        s = sp_at[i]
        if s is not None and speaker_map.get(s) in vis:
            choice = speaker_map[s]
        elif len(vis) >= 2 and lip and (s is not None or not words):
            ranked = sorted(((lip[p][i], p) for p in vis if p in lip), reverse=True)
            if len(ranked) >= 2 and ranked[0][0] > 1.35 * max(1e-6, ranked[1][0]):
                choice = ranked[0][1]
        if choice is None:
            if prev in vis:
                choice = prev
            elif vis:
                choice = max(vis.values(), key=lambda f: f["area"])["id"]
        target.append(choice)
        prev = choice if choice is not None else prev

    breaks = {i for i in range(1, n) if scene_of[i] != scene_of[i - 1]}
    target = _runs_min(target, int(round(MIN_SHOT * fps)), breaks)

    # ---------------------------------------------- camera path (cut on speaker change)
    vals = np.full(n, np.nan)
    for i, d in enumerate(dets):
        f = next((f for f in d if f["id"] == target[i]), None)
        if f is not None:
            vals[i] = f[key_i] + (0.12 if axis == "y" else 0.0)
    shots = np.zeros(n, int)
    switches = 0
    for i in range(1, n):
        new = scene_of[i] != scene_of[i - 1] or (target[i] != target[i - 1] and target[i] is not None
                                                   and target[i - 1] is not None)
        if new and scene_of[i] == scene_of[i - 1]:
            switches += 1
        shots[i] = shots[i - 1] + (1 if new else 0)
    for s in np.unique(shots):     # fill gaps inside a shot by holding the last position
        ids = np.where(shots == s)[0]
        v = vals[ids]
        if np.isfinite(v).any():
            first = v[np.isfinite(v)][0]
            last = first
            for k in range(len(v)):
                last = v[k] if np.isfinite(v[k]) else last
                v[k] = last
        else:
            v[:] = 0.5
        vals[ids] = v
    keys = _shot_keys(ts, vals, shots)

    # ---------------------------------------------- split screen for two-person wide shots
    split = []
    for sc in sorted(set(scene_of)):
        ids = [i for i in range(n) if scene_of[i] == sc]
        two = [i for i in ids if len({f["id"] for f in dets[i] if f["h"] >= 0.11} & set(people)) >= 2]
        if len(ids) < 3 or len(two) < 0.6 * len(ids):
            continue
        cnt: dict[int, int] = {}
        for i in two:
            for f in dets[i]:
                if f["id"] in people:
                    cnt[f["id"]] = cnt.get(f["id"], 0) + 1
        pair = sorted(cnt, key=lambda p: -cnt[p])[:2]
        mean_x = {p: np.mean([f["cx"] for i in two for f in dets[i] if f["id"] == p]) for p in pair}
        pair.sort(key=lambda p: mean_x[p])          # left person on top
        halves = []
        for p in pair:
            cx = np.full(len(ids), np.nan)
            cy = np.full(len(ids), np.nan)
            hs = []
            for k, i in enumerate(ids):
                f = next((f for f in dets[i] if f["id"] == p), None)
                if f:
                    cx[k], cy[k] = f["cx"], f["cy"]
                    hs.append(f["h"])
            for arr in (cx, cy):
                good = np.isfinite(arr)
                arr[~good] = np.interp(np.where(~good)[0], np.where(good)[0], arr[good])
            tt = ts[ids]
            kx = _simplify(tt, _camera(_median(cx)))
            ky = _simplify(tt, _camera(_median(cy)))
            halves.append({"id": int(p), "x": kx, "y": ky, "h": round(float(np.median(hs)), 4)})
        t_end = ts[ids[-1]] + 1.0 / fps
        split.append({"t0": round(float(ts[ids[0]]), 3), "t1": round(float(t_end), 3),
                      "top": halves[0], "bottom": halves[1]})

    # scenes with nobody usable on screen (screen share, gameplay, replays) -> show them whole
    blank = []
    for sc in sorted(set(scene_of)):
        ids = [i for i in range(n) if scene_of[i] == sc]
        with_faces = [max(f["h"] for f in dets[i]) for i in ids if dets[i]]
        small = bool(with_faces) and float(np.median(with_faces)) < 0.12   # face-cam inset only
        if len(ids) >= 3 and (len(with_faces) < 0.25 * len(ids) or small):
            seen = [cams[i] for i in ids if cams[i] and cams[i]["h"] < 0.16]
            cam = _cam_box(seen) if len(seen) >= 0.5 * len(ids) else None
            blank.append({"t0": round(float(ts[ids[0]]), 3),
                          "t1": round(float(ts[ids[-1]] + 1.0 / fps), 3), "cam": cam})

    if log:
        log(f"tracking: faces in {coverage:.0%}, {scene + 1} scene(s), {len(people)} people, "
            f"{switches} speaker cuts, {len(split)} two-shot scene(s), {len(blank)} no-face scene(s)"
            + (f", voices->faces {speaker_map}" if speaker_map else ""))
    return {**base, "coverage": round(coverage, 3), "scenes": scene + 1, "keys": keys or [[0.0, 0.5]],
            "faces": int(sum(1 for d in dets if d)), "people": len(people),
            "speaker_map": {str(k): int(v) for k, v in speaker_map.items()},
            "split": split, "blank": blank, "switches": switches,
            "main_face": _cam_box([c for c in cams if c])}


def _cam_box(seen: list[dict]) -> dict | None:
    """Where the face-cam sits: median face position/size (a webcam overlay does not move)."""
    if not seen:
        return None
    return {k: round(float(np.median([c[k] for c in seen])), 4) for k in ("cx", "cy", "h", "w")}


# --------------------------------------------------------------------------
def _rebase(keys: list[list[float]], off: float, dur: float) -> list[list[float]]:
    def at(t: float) -> float:
        if t <= keys[0][0]:
            return keys[0][1]
        for (ta, va), (tb, vb) in zip(keys, keys[1:]):
            if t <= tb:
                return va + (vb - va) * (t - ta) / max(1e-6, tb - ta)
        return keys[-1][1]

    out = [[0.0, round(at(off), 4)]]
    for t, v in keys:
        if off < t < off + dur:
            out.append([round(t - off, 3), v])
    return out


def covers(trk: dict, a: float, b: float) -> bool:
    return bool(trk and trk.get("keys") and trk.get("v") == TRACK_VERSION
                and a >= trk.get("t0", 0.0) - 0.25 and b <= trk.get("t1", 0.0) + 0.25)


def path_for(trk: dict, a: float, b: float) -> list[list[float]] | None:
    """Main camera keyframes re-based to the trimmed range [a, b]; None if not covered."""
    if not trk or not trk.get("keys"):
        return None
    if a < trk.get("t0", 0.0) - 0.25 or b > trk.get("t1", 0.0) + 0.25:
        return None
    return _rebase(trk["keys"], a - trk["t0"], b - a)


def split_for(trk: dict, a: float, b: float) -> list[dict]:
    """Two-shot scenes inside [a, b], re-based to clip-relative time."""
    if not covers(trk, a, b):
        return []
    off = a - trk["t0"]
    out = []
    for s in trk.get("split") or []:
        t0, t1 = max(s["t0"] - off, 0.0), min(s["t1"] - off, b - a)
        if t1 - t0 < 0.5:
            continue
        sc = {"t0": round(t0, 3), "t1": round(t1, 3)}
        for half in ("top", "bottom"):
            h = s[half]
            sc[half] = {"id": h["id"], "h": h["h"], "x": _rebase(h["x"], off, b - a),
                        "y": _rebase(h["y"], off, b - a)}
        out.append(sc)
    return out


def blank_for(trk: dict, a: float, b: float) -> list[list[float]]:
    """No-face scenes inside [a, b], clip-relative."""
    if not covers(trk, a, b):
        return []
    off = a - trk["t0"]
    out = []
    for item in trk.get("blank") or []:
        t0, t1, cam = (item["t0"], item["t1"], item.get("cam")) if isinstance(item, dict) else (*item, None)
        s, e = max(t0 - off, 0.0), min(t1 - off, b - a)
        if e - s >= 0.5:
            out.append({"t0": round(s, 3), "t1": round(e, 3), "cam": cam})
    return out


def stream_geometry(src_w: int, src_h: int, out_w: int, out_h: int, cam: dict, cfg: dict) -> dict:
    """Gaming layout: a face-cam panel (top or bottom) and a gameplay panel, as source crops."""
    frac = min(0.55, max(0.25, float(cfg.get("cam_size") or 0.48)))
    cam_h = _even(out_h * frac)
    game_h = out_h - cam_h
    # face-cam: head and shoulders, face a little above centre
    ar = out_w / cam_h
    zoom = min(2.0, max(0.5, float(cfg.get("cam_zoom") or 1.0)))   # >1 = tighter on the face
    # wide framing (head, shoulders, chair): the face is about a quarter of the panel height
    ch = min(src_h, max(cam["h"] * src_h * 4.2 / zoom, src_h * 0.1))
    cw = ch * ar
    if cw > src_w:
        cw, ch = src_w, src_w / ar
    cw, ch = _even(cw), _even(ch)
    cx = min(max(cam["cx"] * src_w - cw / 2, 0), src_w - cw)
    cy = min(max(cam["cy"] * src_h - ch * 0.38, 0), src_h - ch)   # face in the upper third
    # gameplay: centre crop of the frame (fill) or the whole frame (fit)
    gar = out_w / game_h
    gh = src_h
    gw = gh * gar
    if gw > src_w:
        gw, gh = src_w, src_w / gar
    gw, gh = _even(gw), _even(gh)
    gx, gy = (src_w - gw) / 2, (src_h - gh) / 2
    return {"cam_pos": "bottom" if cfg.get("cam") == "bottom" else "top",
            "cam_h": cam_h, "game_h": game_h, "game": "fit" if cfg.get("game") == "fit" else "fill",
            "cam_crop": [int(cx), int(cy), cw, ch], "game_crop": [int(gx), int(gy), gw, gh]}


def split_crop(src_w: int, src_h: int, out_w: int, half_h: int, face_h: float) -> tuple[int, int]:
    """Crop size for one half of a split screen: frame the face with head-room."""
    ar = out_w / half_h
    ch = min(src_h, max(src_h * 0.42, face_h * src_h * 4.2))   # wider: face ~1/4 of the panel
    cw = ch * ar
    if cw > src_w:
        cw = src_w
        ch = cw / ar
    return _even(cw), _even(ch)


def expr(keys: list[list[float]], var: str = "t") -> str:
    """Balanced piecewise-linear ffmpeg expression (depth log2(n), no parser limits)."""
    if not keys:
        return "0.5"
    if len(keys) == 1:
        return f"{keys[0][1]:.4f}"

    def seg(i: int) -> str:
        (t0, v0), (t1, v1) = keys[i][:2], keys[i + 1][:2]
        dt = max(1e-3, t1 - t0)
        if abs(v1 - v0) < 1e-5:
            return f"{v0:.4f}"
        return f"({v0:.4f}+({v1 - v0:.4f})*({var}-{t0:.3f})/{dt:.3f})"

    def build(lo: int, hi: int) -> str:
        if lo == hi:
            return seg(lo)
        mid = (lo + hi) // 2
        return f"if(lt({var},{keys[mid + 1][0]:.3f}),{build(lo, mid)},{build(mid + 1, hi)})"

    inner = build(0, len(keys) - 2)
    return (f"if(lt({var},{keys[0][0]:.3f}),{keys[0][1]:.4f},"
            f"if(gte({var},{keys[-1][0]:.3f}),{keys[-1][1]:.4f},{inner}))")
