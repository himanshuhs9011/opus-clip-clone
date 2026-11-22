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
| Clip edges & length | every pick's start and end are reviewed like an editor would (see below), so each clip is as long as its moment | `boundaries.py` |
| Virality score | 0–100 + letter grade, five sub-scores (Hook, Flow, Value, Emotion, Trend), every signal listed; LLM and measured signals blended 45/55 | `score.py` |
| Speakers | voice fingerprints (WeSpeaker ONNX) clustered into "who talks when"; every word gets a speaker | `speakers.py` |
| Reframe | YuNet faces + SFace face fingerprints link people across camera angles; lip motion matches each voice to a face; the camera cuts to whoever is talking (min 1 s shots), two-person wide shots become a split screen, blur-fit when nobody is on screen | `reframe.py` |
| Edit | trims, jump cuts (silences, um/uh, any word you cut), word fixes, auto-zoom, colour, title, progress bar | `edit.py` |
| Captions | 8 animated templates (pop, bounce, highlight box, karaoke sweep, neon glow…), keyword colours, colour emoji, one colour per speaker (auto-picked to contrast with the template, or your own) | `captions.py` |
| Render | one ffmpeg pass per clip, NVENC + CUDA decode, 3 clips in parallel, x264 fallback | `render.py` |

## The app

* **Home**: paste a link or drop a file and choose clip count (Auto by default), quality bar, aspect,
  caption template, AI provider, framing and effects. Clip length is never a setting: each clip
  is as long as its moment (12–90 s).
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

Open ⚙ Settings → **AI providers**. Every user adds their own API key; keys are saved only in
`providers.json` on their PC (in `Videos\ClipForge`, or the project folder when run from source)
and sent only to that provider. Nothing is built into the app.

| Provider | |
|---|---|
| Ollama (local) | free and private. One-click download of Gemma 4 (`gemma4:e4b`, 6.6 GB, fits 8 GB VRAM, good Hindi). |
| NVIDIA | free key from build.nvidia.com. Gemma 4 31B and Nemotron, with no OpenRouter rate limits. |
| OpenRouter | many models behind one key. "Free models only" is on by default, so a key never spends credits. |
| Google AI Studio, Groq, OpenAI, Custom | any OpenAI-compatible server. |

**Model allow-list:** each provider has an ordered list of allowed models. The first is used,
the rest take over when it is busy (HTTP 429), and nothing outside the list is ever called.
Rules only (no AI) is always available.

## Gaming streams

Long streams (auto-detected, or Content: gaming) go through the **gaming agent**:
1. a per-second hype map from loudness spikes, jumps, bursts of talk and exclamations (English and Hindi);
2. the AI reads the whole stream once (game, streamer, storyline, its own list of moments);
3. it reviews every candidate (keep or not, type, exact start/end, Hinglish title and banner,
   story label), asks again about any it skipped, and drops repeats of the same story;
4. the viral judge ranks the finalists.

If the streamer talks a lot, the normal talk reader runs too, so stories and rants
compete with the loud moments. On cloud providers the review runs three batches at a time.

**How many clips:** Auto by default, about one clip per 3.5 minutes (a 1 h 47 min stream gives 20).
The quality bar removes weak clips, but it never leaves you with fewer than half of what you
asked for while near-misses (within 12 points) exist.

**Viral score (gaming):** the agent scores each clip out of 100 on five parts, 20 points each:
hook strength, emotion, shareability, works without the stream, and payoff. That is blended
60/40 with ClipForge's own score from sound and speech, then the viral judge compares the finalists.

**Cuts:** the agent can cut dull stretches out of a clip (loading screens, menus, silent
driving). Clips start just before the first word and end ~0.35 s after the last one, so no
word is chopped.

**Hook intro:** every clip opens with its strongest 2-4 s (the shout, the punchline), then
a white flash into the full clip. Toggle it per project or per clip.

**Gaming templates** (new project screen, or per clip in the editor): Stacked, Reaction (big face),
50 / 50, Bubble (round face-cam with a white ring over full gameplay), Card (framed face-cam in a corner)
and Gameplay only. The context banner moves with the template.

## Whole-video understanding, long videos, ready-to-post text

Picking clips one transcript window at a time misses what an editor knows. Before any clip
is cut, `opusclip/story.py` builds this context once per video (cached for Re-run):

- **Engagement map:** every second scored from loudness spikes against the video's own level,
  sudden jumps, bursts of fast talk, exclamations, reaction phrases ("no way", "are you
  serious", "bhai kya"), laughter, and fast back-and-forth between speakers. The AI sees
  `[HYPE]` / `[hot]` on those lines, and a clip holding one of the video's top peaks scores higher.
- **Story log** (videos over 6 minutes): the video in ~5-minute parts. The AI writes what
  happens in each (names, game events, claims), tags it (funny, clutch, argument, reveal,
  shock...), rates it 1-10, and notes setups that pay off later. Then a brief of the whole video:
  who, what, the running storylines. Every clip-finding prompt gets the brief and what happens
  around that part, so a 4-hour stream is understood as one story.
- **Format:** *Shorts* (vertical, 12 s to 3 min, each as long as its moment), *Long videos*
  (standalone 16:9 YouTube videos of about 10 min, set with `long_len`) or *Both*. A long video is
  a run of parts with a start that grabs and a payoff; its edges move onto natural pauses, quiet
  wordless stretches (AFK, loading) are cut out, and it opens on a teaser of its best moment.
  About one long video per 40 minutes of source.
- **Ready to post:** every clip gets a YouTube title, 2-4 words of thumbnail text and a
  description (long videos also get chapters), written with the whole-video brief. They show
  under each clip with copy buttons and go into the ZIP's `.txt` files.

Without an AI all of this still runs on the engagement map and transcript, with simpler text.

## Autopilot (clip and post every day by itself)

**Autopilot** in the top bar (`#/autopilot`) runs inside the server while ClipForge is open:

1. **Channels to watch:** paste a YouTube channel (its uploads and past streams are both
   checked), or a Kick / Twitch videos page. Each source has a format (shorts, long videos,
   both) and an optional focus ("only funny moments"). Every `check_minutes` it looks for
   finished videos it has not clipped yet (live streams wait until they end; videos under
   3 minutes are skipped) and clips them one at a time.
2. **Posts:** every finished clip scoring at least `min_score` is queued once per platform:
   shorts to YouTube Shorts, Instagram Reels, Facebook Reels and the phone folder; long videos
   to YouTube. At each posting time the best waiting clips go out, within each platform's
   daily cap. Turn on approval to OK every clip first; failed posts can be retried.
3. **Accounts** use the official APIs, never passwords:
   - YouTube: a Google Cloud project with *YouTube Data API v3* and an OAuth client of type
     *Desktop app*; then *Sign in with Google*. Only a refresh token is stored. A new project
     can upload about 6 videos a day (10,000 quota units, ~1,600 per upload), and uploads stay
     private until the project passes YouTube's API audit.
   - Instagram + Facebook: an Instagram Business / Creator account linked to a Facebook Page,
     a Meta app, and a user token from Graph API Explorer with `pages_show_list,
     pages_read_engagement, pages_manage_posts, instagram_basic, instagram_content_publish,
     business_management`. It is swapped for a Page token that does not expire.
   - Snapchat Spotlight, Moj and Josh have no upload API: their clips are copied with captions
     to `autopilot_phone/<date>/` to post from a phone. TikTok is left out (not in India).
4. Captions come from each clip's package (YouTube title, description, chapters for long
   videos, hashtags, `#Shorts`), with a "Source: <channel>" credit line by default.

Everything (tokens included) is kept in `autopilot.json` in the data folder; it is in
`.gitignore`. Only clip channels you own or have permission for: re-uploading other people's
videos gets copyright strikes and Content ID claims.

## Clip Agent (say what you want)

**Agent** in the top bar (`#/agent`) takes a request in plain words, with a link or an
attached file:

> Clip the last hour of this stream into 5 clips under 45 seconds, only the rage moments, face-cam on top

1. **Plan:** it shows what it understood (count, length, which part of the video, what kind of
   moment, layout) as steps you approve, or change.
2. **Find:** transcript and clip finding run as usual, but nothing is rendered yet.
3. **Pick:** found clips are listed with their score, time range and why they were picked.
   The best few are pre-selected; *Make N clips* or *Make all*.
4. **Make:** only the picked clips are reframed, captioned and rendered.

Keep talking to it about the same video: "make 3", "make all", "only the funny ones",
"shorter, under 20 seconds". A new ask re-finds clips on the same transcript (fast) and keeps
what you said before unless you change it.

How a request is read (`opusclip/wizard.py`, rules first, so it works with Rules only too):

| You say | It means |
|---|---|
| "under 60 seconds", "30 second clips", "2 minute clips" | the longest a clip may be; the picker aims at it and stretches a pick up to 20 % short while the speech runs on |
| "at least a minute", "15-30 second clips" | a minimum (and maximum) |
| "Shorts", "Reels", "TikTok" | vertical, no length |
| "about 10 clips", "top 5" | how many (otherwise Auto) |
| "from 1:20:00", "between 10:00 and 25:00", "the last hour", "skip the first 10 minutes" | only that part is searched |
| "only the biggest laughs", "every time someone gets voted out", "moments about pricing" | the kind of moment: it beats general virality. "Viral", "best" or "good hooks" are not a kind |
| "face-cam on top", "no captions", "karaoke captions", "square" | layout and style |

When the rules find no kind of moment but the request has more words, the AI is asked for it
(25 s at most, so the plan never hangs on a cold model).

API: `POST /api/agent/plan`, `POST /api/agent/start`, `POST /api/agent/followup`,
`POST /api/projects/{id}/finish` (render picked clips). Projects waiting for picks have
status `picking` and clips with render status `candidate`; the Project page shows a
*Make this clip* button on those.

## How a clip's length is decided

There is no length setting. The AI proposes each clip from the transcript, using a length
rubric by kind of moment: a punchline, reaction or hot take is 12–25 s, a tip or answer
20–45 s, a story or argument that keeps tension 45–90 s. Then `opusclip/boundaries.py`
reviews both edges of every pick:

- **Opening:** never mid-sentence (Whisper splits long sentences into 20-word units, so the
  start moves back to the real sentence start). Lead-in banter ("No. No? Ah!", "Yeah.") and
  channel plugs are skipped. A first line that continues something unseen ("But the main
  reason…", "That's why…", "As I said…", "He told me…") gets its setup sentence added back,
  or is trimmed when it is a short throwaway.
- **Ending:** a real question gets its answer; an unfinished sentence runs on to its full stop;
  a payoff line right after ("And that's why…", "Turns out…") is kept; a long question after
  the answer (the next topic starting) and trailing chatter or sign-offs are cut.
- **Too short:** a pick up to 25 % under the minimum is lengthened while the speech runs on,
  instead of being thrown away.
- **Cut points:** 0.05 s before the first word and 0.12 s after the last one, never into a
  neighbouring word.

Game streams without an AI review get their length from the hype curve: the moment starts
on the build-up and ends when the reaction has died down for 4 s, so a quick fail is ~15 s
and a long fight ~45 s.

**How many clips (Auto):** about one per 3.5 minutes is the target. A video with more strong
moments keeps up to half as many again, but only clips scoring 70+ (or 7 over the quality
bar); a weak video keeps at least half the target if near-misses exist. Channel intros and
plugs are never kept just to make up the number.

## Windows installer

`ClipForge-Setup.exe` is a normal setup wizard: Welcome → folder → options → optional
OpenRouter key → Install → Finish. It installs for the current user only and needs no admin
rights. After copying the app it downloads and sets up everything else by itself:

- a private Python 3.12 (via `uv`) and all packages; PyTorch is not needed
- CUDA libraries for fast Whisper, if an NVIDIA GPU is found
- FFmpeg (with NVENC) and Deno (used by yt-dlp for YouTube)
- Ollama (installed silently) and the Gemma 4 `gemma4:e4b` model, both ticked by default
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
python run.py podcast.mp4 --clips 6 --style beast          # length: AI per clip (default)
python run.py podcast.mp4 --length medium                   # force 30-60 s clips
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
- `opusclip/boundaries.py`: the opening / ending rules and cut padding.

Tests: `python -m pytest tests`.
- `opusclip/captions.py`: `PRESETS` (font, colours, animation, words per caption).
- `opusclip/edit.py`: filler list, silence threshold, emoji map, zoom rhythm.
- `.env`: `RENDER_WORKERS`, `ENCODER`, `CRF`, `REFRAME_SAMPLE_FPS`.
