"""Create a synthetic talking-head-style test source (TTS audio + video pattern).

    python tools/make_fixture.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "fixtures"

SCRIPT = """Most people think that making money online is about luck. It is not. I want to tell you exactly what happened to me last year, because it was the strangest twelve months of my life.

In January I was completely broke. I had about four hundred dollars in my bank account and a laptop that overheated every twenty minutes. I was sleeping on my brother's couch, and honestly, I was embarrassed. But then I found one thing that changed everything, and it was not a secret hack or a get rich quick scheme.

Here is the real problem. Everybody wants to start a business, but nobody wants to do the boring part. The boring part is talking to customers. I spent ninety days calling fifty people a week. Fifty. Not five, not ten. Fifty. And by the end of those ninety days I had my first ten thousand dollars.

Let me break down the exact system, because I think this is the part that people miss. First, you pick one single painful problem. Second, you find ten people who already have that problem and you ask them how they solve it today. Third, you build the ugliest possible version of the solution and you charge for it. That is it. There is no fourth step.

Now, the second thing nobody tells you is about energy. I used to work fourteen hour days and get nothing done. Now I work four focused hours and I ship more. The difference was not discipline. The difference was that I stopped doing work that made me feel productive but did not move the needle.

I want to be honest with you, because I think honesty is rare on the internet. I failed twice before this worked. My first product made zero dollars. Zero. I remember refreshing the dashboard every ten minutes for three weeks, waiting for a sale that never came. That failure taught me more than any course I ever bought.

And here is the part that still shocks me. When I finally made money, it did not feel the way I expected. I thought I would feel relief. Instead I felt this weird pressure, like now I had something to lose. That is the emotional side of success that nobody talks about, and I think it is the reason so many people self sabotage right when things start working.

So if you are at the beginning right now, here is what I want you to take away. You are not behind. You are just early. The people you admire spent years being invisible before anyone noticed them. Pick one problem, talk to real humans, charge money, and repeat that loop until it works.

If this was useful, save this video and send it to the one friend who needs to hear it. And tell me in the comments what your first product was, because I read every single comment. Thank you for watching, and I will see you in the next one."""


def tts(text: str, wav: Path) -> Path:
    wav.parent.mkdir(parents=True, exist_ok=True)
    txt = wav.with_suffix(".txt")
    txt.write_text(text, encoding="utf-8")
    ps = f"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SelectVoice('Microsoft David Desktop')
$s.Rate = 1
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(44100,
  [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
  [System.Speech.AudioFormat.AudioChannel]::Mono)
$s.SetOutputToWaveFile('{wav}', $fmt)
$s.Speak([IO.File]::ReadAllText('{txt}'))
$s.Dispose()
"""
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    return wav


def main() -> int:
    OUT.mkdir(exist_ok=True)
    wav = tts(SCRIPT, OUT / "speech.wav")
    video = OUT / "podcast.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30",
            "-i", str(wav),
            "-shortest",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            str(video),
        ],
        check=True,
    )
    print(video)
    return 0


if __name__ == "__main__":
    sys.exit(main())
