from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"
# the project's .env is authoritative - a stale shell variable must not override a fresh key
load_dotenv(ENV_PATH, override=True)


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _env(name: str, default: str = "") -> str:
    """Env var with inline `# comments` stripped (python-dotenv keeps them for unquoted values)."""
    v = os.getenv(name)
    if v is None:
        return default
    v = re.sub(r"\s+#.*$", "", v).strip()
    return v if v else default


# Installed app: projects live in a user folder (set by the launcher), not next to the code
DATA = Path(os.environ["CLIPFORGE_DATA"]) if os.getenv("CLIPFORGE_DATA") else ROOT


@dataclass
class Settings:
    root: Path = ROOT
    output_dir: Path = DATA / "output"
    upload_dir: Path = DATA / "uploads"
    model_dir: Path = ROOT / "models"
    config_dir: Path = ROOT / "config"
    asset_dir: Path = ROOT / "assets"

    # ASR
    whisper_model: str = _env("WHISPER_MODEL", "large-v3-turbo")
    whisper_device: str = _env("WHISPER_DEVICE", "auto")
    whisper_compute: str = _env("WHISPER_COMPUTE", "auto")
    whisper_language: str = _env("WHISPER_LANGUAGE", "")

    # LLM provider: ollama (local, default) | openrouter (optional) | openai (any compatible) | none
    llm_provider: str = _env("LLM_PROVIDER", "ollama")
    ollama_host: str = _env("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = _env("OLLAMA_MODEL", "qwen2.5:7b")
    ollama_num_ctx: int = int(_env("OLLAMA_NUM_CTX", "8192"))
    # OpenRouter / OpenAI-compatible
    llm_base_url: str = _env("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    llm_api_key: str = _env("OPENROUTER_API_KEY") or _env("LLM_API_KEY")
    llm_model: str = _env("OPENROUTER_MODEL") or _env("LLM_MODEL", "google/gemini-2.5-flash")
    llm_fallback_models: list[str] = field(
        default_factory=lambda: [
            m.strip() for m in _env("LLM_FALLBACK_MODELS", "").split(",") if m.strip()
        ]
    )
    llm_timeout: float = float(_env("LLM_TIMEOUT", "600"))

    # Output defaults (per-project options override these)
    fps: int = int(_env("OUT_FPS", "30"))
    crf: int = int(_env("CRF", "20"))
    preset: str = _env("X264_PRESET", "veryfast")
    encoder: str = _env("ENCODER", "auto")          # auto | nvenc | x264
    render_workers: int = int(_env("RENDER_WORKERS", "3"))
    max_clips: int = int(_env("MAX_CLIPS", "6"))
    min_clip: float = float(_env("MIN_CLIP", "18"))
    max_clip: float = float(_env("MAX_CLIP", "75"))

    # Reframe
    sample_fps: float = float(_env("REFRAME_SAMPLE_FPS", "4"))
    caption_style: str = _env("CAPTION_STYLE", "bold-pop")

    def ensure_dirs(self) -> None:
        for d in (self.output_dir, self.upload_dir, self.model_dir, self.config_dir, self.asset_dir):
            d.mkdir(parents=True, exist_ok=True)

    def scoring_weights(self) -> dict:
        path = self.config_dir / "scoring.json"
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}


settings = Settings()
settings.ensure_dirs()


# --------------------------------------------------------------------------
# .env editing (used by the Settings screen; the server only listens on localhost)
# --------------------------------------------------------------------------
ENV_KEYS = {
    "llm_provider": "LLM_PROVIDER",
    "ollama_host": "OLLAMA_HOST",
    "ollama_model": "OLLAMA_MODEL",
    "llm_base_url": "LLM_BASE_URL",
    "llm_api_key": "OPENROUTER_API_KEY",
    "llm_model": "OPENROUTER_MODEL",
    "whisper_model": "WHISPER_MODEL",
}


def update_env(values: dict[str, str]) -> None:
    """Persist settings to .env (keeps comments and unrelated lines) and apply them live."""
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.exists() else []
    pending = {}
    for attr, val in values.items():
        key = ENV_KEYS.get(attr)
        if key is None:
            continue
        pending[key] = str(val)
        setattr(settings, attr, val)
        os.environ[key] = str(val)
    out = []
    for line in lines:
        m = re.match(r"\s*([A-Z0-9_]+)\s*=", line)
        if m and m.group(1) in pending:
            out.append(f"{m.group(1)}={pending.pop(m.group(1))}")
        else:
            out.append(line)
    for k, v in pending.items():
        out.append(f"{k}={v}")
    ENV_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")
