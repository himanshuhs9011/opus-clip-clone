"""ClipForge desktop launcher.

Starts the local engine in the background and opens the app in its own window
(Edge WebView2 via pywebview). Falls back to the default browser if that is not available.
Run with pythonw so no console window appears.
"""
from __future__ import annotations

import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

# bundled tools (ffmpeg, deno) and the venv's Scripts folder (yt-dlp.exe) on PATH
os.environ["PATH"] = os.pathsep.join(
    [str(ROOT / "bin"), str(Path(sys.executable).parent), os.environ.get("PATH", "")])
DATA = Path(os.environ.setdefault("CLIPFORGE_DATA", str(Path.home() / "Videos" / "ClipForge")))
DATA.mkdir(parents=True, exist_ok=True)

# pythonw has no console: send output to a log file instead of crashing on print
log = open(DATA / "clipforge.log", "a", encoding="utf-8", buffering=1)
sys.stdout = sys.stderr = log
print(f"\n--- ClipForge start {time.strftime('%Y-%m-%d %H:%M:%S')} ---")


def alive(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/presets", timeout=1.5) as r:
            return r.status == 200
    except Exception:
        return False


def free_port(start: int = 8765) -> int:
    for port in range(start, start + 40):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("no free port")


def open_window(url: str, on_close=None) -> None:
    try:
        import webview

        webview.create_window("ClipForge", url, width=1440, height=920, min_size=(1000, 680))
        webview.start()
        if on_close:
            on_close()
        return
    except Exception as e:  # no WebView2 / pywebview: use the browser
        print(f"app window unavailable ({e}), opening the browser")
    webbrowser.open(url)
    while True:      # keep the engine running while the browser tab is used
        time.sleep(3600)


def main() -> None:
    # already running? just show it
    marker = DATA / "port.txt"
    if marker.exists():
        try:
            port = int(marker.read_text().strip())
            if alive(port):
                open_window(f"http://127.0.0.1:{port}/")
                return
        except ValueError:
            pass

    import uvicorn

    port = free_port()
    marker.write_text(str(port))
    config = uvicorn.Config("app:app", host="127.0.0.1", port=port, log_config=None, access_log=False)
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(150):
        if alive(port):
            break
        time.sleep(0.2)

    def stop() -> None:
        server.should_exit = True
        marker.unlink(missing_ok=True)

    open_window(f"http://127.0.0.1:{port}/", on_close=stop)
    time.sleep(1.0)


if __name__ == "__main__":
    main()
