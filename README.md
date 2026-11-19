# ClipForge — a local Opus-Clip-style engine

**Website: [clipforge.raglib-site.workers.dev](https://clipforge.raglib-site.workers.dev)** · **[⬇ Download ClipForge for Windows](https://clipforge.raglib-site.workers.dev/download)**:
run it, click Next a few times, then Finish. It installs everything it needs by itself.

Long video in → ranked, captioned, vertical shorts out. Runs on your own PC (Whisper + Ollama +
FFmpeg/NVENC), with OpenRouter as an optional cloud brain.

```
serve.bat                      # builds the React UI once, starts http://127.0.0.1:8000
python run.py video.mp4 --clips 6
```

## What it does

| Stage | How | File |
|---|---|---|
| Ingest | upload or YouTube link (yt-dlp), ffprobe, 16 kHz audio, browser proxy for the editor | `pipeline.py`, `text.py` |
| Transcribe | faster-whisper on CUDA (batched), word timestamps, CPU fallback, transcript reused on re-run | `text.py` |
| Find highlights | sentence-aligned candidate windows → heuristic prefilter → LLM ranking, window by window so long podcasts are fully covered | `highlights.py` |
| Virality score | 0–100 + letter grade, five sub-scores (Hook, Flow, Value, Emotion, Trend), every signal listed; LLM and measured signals blended 45/55 | `score.py` |
| Speakers | voice fingerprints (WeSpeaker ONNX) clustered into "who talks when"; every word gets a speaker | `speakers.py` |
| Reframe | YuNet faces + SFace face fingerprints link people across camera angles; lip motion matches each voice to a face; the camera cuts to whoever is talking (min 1 s shots), two-person wide shots become a split screen, blur-fit when nobody is on screen | `reframe.py` |
| Edit | trims, jump cuts (silences, um/uh, any word you cut), word fixes, auto-zoom, colour, title, progress bar | `edit.py` |
| Captions | 8 animated templates (pop, bounce, highlight box, karaoke sweep, neon glow…), keyword colours, colour emoji, one colour per speaker (auto-picked to contrast with the template, or your own) | `captions.py` |
| Render | one ffmpeg pass per clip, NVENC + CUDA decode, 3 clips in parallel, x264 fallback | `render.py` |

## The app

* **Home**: paste a link or drop a file and choose clip length (auto, under 30 s, 30–60 s, 60–90 s),
  clip count, aspect, caption template, AI provider, framing and effects.
* **Project**: clips ranked by virality score, each with score ring, grade, sub-score bars,
  the signals behind the score, risks and hashtags. Download one clip or all of them (ZIP with
  title and hashtag `.txt` files).
* **Editor**: three panes.
  * Transcript: click a word to seek, shift-click to select, then cut, set start/end or fix its text.
  * Live preview with a trim timeline.
  * Inspector: captions, effects, layout and title.

  The preview is drawn in the browser from the same plan the renderer uses
  (`POST /api/projects/:id/clips/:cid/plan`), so edits show up instantly and **Export** only
  renders once. *Apply style to all clips* copies the look to every clip.

## AI providers

Set in the app (⚙ Settings) or `.env`:

| `LLM_PROVIDER` | |
|---|---|
| `ollama` (default) | local, free. `qwen2.5:7b` fits 8 GB VRAM. ClipForge starts `ollama serve` if needed and unloads the model before Whisper runs so they share the GPU. |
| `openrouter` | optional. Needs `OPENROUTER_API_KEY`; any model id (`OPENROUTER_MODEL`). |
| `none` | rules only. The explainable scorer picks the clips. |

If the LLM fails or returns too few clips, the rule-based picker tops up the list, so you always
get the clip count you asked for.

## Windows installer

`ClipForge-Setup.exe` is a normal setup wizard: Welcome → folder → options → optional
OpenRouter key → Install → Finish. It installs for the current user only and needs no admin
rights. After copying the app it downloads and sets up everything else by itself:

- a private Python 3.12 (via `uv`) and all packages; PyTorch is not needed
- CUDA libraries for fast Whisper, if an NVIDIA GPU is found
- FFmpeg (with NVENC) and Deno (used by yt-dlp for YouTube)
- Ollama (installed silently) and the `qwen2.5:7b` model, both ticked by default
- checks the OpenRouter key if one was entered, writes the settings and runs a self-test

The app opens in its own window. Projects are saved in `Videos\ClipForge`. Uninstall from
Windows Settings → Apps. A log of the component setup is in the install folder (`install.log`).

Build the installer (needs Inno Setup 6 on the build PC: `winget install JRSoftware.InnoSetup`):

```
powershell -ExecutionPolicy Bypass -File installer\build.ps1
```

Then publish `dist\ClipForge-Setup.exe` as a GitHub Release asset with exactly that file name,
so the download link above always serves the newest version.

## Website

The download site lives in `site/` and runs on Cloudflare Workers:

- `site/public/index.html`: the page
- `site/src/worker.js`: serves `/download`, which streams the newest `ClipForge-Setup.exe`
  from the GitHub release, so the button downloads straight from the site
- `site/wrangler.jsonc`: Cloudflare config

Publish changes with `npx wrangler deploy` from the `site` folder.

## Setup (developers)

Python 3.11–3.13, Node 18+, `ffmpeg` (full build for NVENC) and `yt-dlp` on PATH.

```bash
pip install -r requirements.txt
copy .env.example .env        # then edit
cd frontend && npm install && npm run build
```

UI development with hot reload: run the API (`python -m uvicorn app:app --port 8000`), then
`cd frontend && npm run dev` and open http://localhost:5173.

## CLI

```bash
python run.py podcast.mp4 --clips 6 --length medium --style beast
python run.py "https://youtube.com/watch?v=..." --provider openrouter --aspect 1:1
python run.py talk.mp4 --provider none --model small --remove-silence
```

## Output

```
output/<project-id>/
  source.mp4             the original (lets you re-run and re-edit)
  audio.wav              16 kHz mono
  preview.mp4            720p proxy, only for codecs browsers can't play
  project.json           transcript, scores, face paths, edits
  clips/clip_01_v3.mp4   latest render (version bumps on every export)
  clips/clip_01_v3.jpg   thumbnail
  clips/clip_01_v3.ass   caption track
```

## Links: no full downloads

A pasted link never downloads the whole video:

1. **Audio only.** About 1 MB per minute. Used for the transcript, speakers and clip picking.
2. **Video pieces.** YouTube streams start with an index that maps time to bytes. ClipForge
   reads it and fetches only the fragments around each chosen clip (±8 s, so small trims need
   no new download), using HTTP range requests. Each piece is then muxed with the matching
   slice of the audio.

Example, a 23-minute 1080p video with 2 clips: 22 MB of audio plus 41 MB of video, instead of
515 MB. A trim or re-run that reaches outside a piece fetches just the missing part. Links
from sites without such an index fall back to a normal download. `DOWNLOAD_MODE=full`
restores the old behaviour.

## Podcasts and multi-camera videos

Framing **Auto** (default) does what an editor would:

- **Close-up cameras:** follows each camera cut and stays on the person in the shot.
- **Wide shot with two people, both talking in the clip:** split screen, one person on top and one below, with captions on the seam.
- **Speaker focus:** in a wide shot, the camera cuts to whoever is talking. The voice is matched to the face by lip movement.
- **Game streams (face-cam + gameplay):** when a scene is gameplay with a small webcam, Auto switches to the **Gaming** layout. The face-cam is cropped into its own panel and the gameplay fills the other. In Editor → Layout you choose face-cam on top or bottom, its size and zoom, and gameplay *Fill* (crop the centre) or *Fit* (whole screen over a blur). Pick **Gaming** to use it for the whole clip.
- **Screen share, replays:** when nobody is on screen, that scene shows the whole frame over a blurred background.
- **Webcam detection on streams:** the face-cam is found as a box that stays put for the whole clip, even when it is tiny and the gameplay changes every second. The box edges come from averaging edges over time: the border stays sharp while gameplay blurs out. The face-cam panel is cropped inside that box, so no gameplay leaks in.
- **Context banner:** the AI writes one line per clip saying what it is about, using the video's title for the game or topic (for example "Streamer RANTS Console vs PC Fragpunk"). In gaming and split-screen clips it sits on the seam for the whole clip, and captions move into the gameplay. Change it in Editor → Title.

Captions switch colour when the speaker changes. Turn this off, or pick each speaker's colour,
in Editor → Captions. Test material: `python tools/make_podcast_fixture.py` builds a
two-voice, multi-camera podcast at `fixtures/podcast2.mp4`.

## Tuning

- `config/scoring.json`: sub-score weights.
- `opusclip/score.py`: lexicons and thresholds behind every signal.
- `opusclip/captions.py`: `PRESETS` (font, colours, animation, words per caption).
- `opusclip/edit.py`: filler list, silence threshold, emoji map, zoom rhythm.
- `.env`: `RENDER_WORKERS`, `ENCODER`, `CRF`, `REFRAME_SAMPLE_FPS`.
