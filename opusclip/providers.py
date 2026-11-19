"""AI providers the user can add with their own API key (Settings → Providers).

Everything is stored locally in providers.json inside the ClipForge data folder; keys never
leave the machine except to the provider they belong to. Each provider has an ordered
`models` list: the first is used, the rest are fallbacks, and nothing outside the list is
ever called (that is the model restriction). `free_only` (OpenRouter) refuses paid models so
a key can never run up a bill.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from .config import DATA, settings

PATH = Path(DATA) / "providers.json"
_lock = threading.Lock()

# id -> static facts. All of these speak the OpenAI chat-completions protocol.
CATALOG: dict[str, dict] = {
    "ollama": {
        "name": "Ollama", "kind": "local", "base_url": "http://localhost:11434",
        "about": "Runs on your own GPU. Free and private, but slower and weaker than cloud models.",
        "models": ["gemma4:e4b", "qwen2.5:7b"],
    },
    "nvidia": {
        "name": "NVIDIA", "kind": "cloud", "base_url": "https://integrate.api.nvidia.com/v1",
        "key_url": "https://build.nvidia.com/settings/api-keys", "key_prefix": "nvapi-",
        "about": "Free key from build.nvidia.com. Hosts Gemma 4 31B (best for Hindi) and Nemotron.",
        "models": ["google/gemma-4-31b-it", "nvidia/nemotron-3-ultra-550b-a55b",
                   "nvidia/nemotron-3.5-super-vl-120b-a12b"],
    },
    "openrouter": {
        "name": "OpenRouter", "kind": "cloud", "base_url": "https://openrouter.ai/api/v1",
        "key_url": "https://openrouter.ai/keys", "key_prefix": "sk-or-",
        "about": "Hundreds of models behind one key. ':free' models cost nothing but get rate limited.",
        "models": ["nvidia/nemotron-3-ultra-550b-a55b:free", "nvidia/nemotron-3.5-super-vl:free",
                   "google/gemma-4-31b-it:free"],
    },
    "gemini": {
        "name": "Google AI Studio", "kind": "cloud",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "key_url": "https://aistudio.google.com/apikey", "key_prefix": "AI",
        "about": "Gemini and Gemma from Google, with a free tier.",
        "models": ["gemini-2.5-flash", "gemma-4-31b-it"],
    },
    "groq": {
        "name": "Groq", "kind": "cloud", "base_url": "https://api.groq.com/openai/v1",
        "key_url": "https://console.groq.com/keys", "key_prefix": "gsk_",
        "about": "Very fast inference with a free tier.",
        "models": ["llama-3.3-70b-versatile"],
    },
    "openai": {
        "name": "OpenAI", "kind": "cloud", "base_url": "https://api.openai.com/v1",
        "key_url": "https://platform.openai.com/api-keys", "key_prefix": "sk-",
        "about": "Paid. GPT models.",
        "models": ["gpt-4.1-mini"],
    },
    "custom": {
        "name": "Custom (OpenAI-compatible)", "kind": "cloud", "base_url": "",
        "about": "Any server that speaks the OpenAI API: LM Studio, vLLM, Together, DeepInfra…",
        "models": [],
    },
}
CLOUD = [k for k, v in CATALOG.items() if v["kind"] == "cloud"]
ORDER = list(CATALOG)


def _seed() -> dict:
    """First run: carry over what the old .env settings had."""
    env_model = settings.llm_model or ""
    or_models = list(CATALOG["openrouter"]["models"])
    if env_model and env_model not in or_models and "mimo-v2-flash" not in env_model:
        or_models.insert(0, env_model)
    data = {
        "active": settings.llm_provider if settings.llm_provider in ("ollama", "openrouter", "none") else "ollama",
        "providers": {
            "ollama": {"host": settings.ollama_host, "models": [settings.ollama_model]},
            "openrouter": {"key": settings.llm_api_key, "models": or_models, "free_only": True},
        },
    }
    if settings.llm_provider == "openai" and settings.llm_base_url:
        data["providers"]["custom"] = {"key": settings.llm_api_key, "base_url": settings.llm_base_url,
                                       "models": [env_model] if env_model else []}
        data["active"] = "custom"
    if data["active"] == "openrouter" and not settings.llm_api_key:
        data["active"] = "ollama"
    return data


def load() -> dict:
    with _lock:
        if PATH.exists():
            try:
                data = json.loads(PATH.read_text(encoding="utf-8"))
                data.setdefault("providers", {})
                data.setdefault("active", "ollama")
                return data
            except Exception:
                pass
        data = _seed()
        _write(data)
        return data


def _write(data: dict) -> None:
    PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, PATH)


def save(data: dict) -> None:
    with _lock:
        _write(data)


def get(pid: str) -> dict:
    """Resolved config for one provider: base_url, key, models (ordered, restricted)."""
    pid = "custom" if pid == "openai" else pid
    cat = CATALOG.get(pid)
    if cat is None:
        return {}
    cfg = load()["providers"].get(pid, {})
    models = [m for m in (cfg.get("models") or []) if m] or list(cat["models"])
    if pid == "ollama":
        return {"id": pid, "base_url": (cfg.get("host") or cat["base_url"]).rstrip("/"), "key": "",
                "models": models[:1], "free_only": False}
    if pid == "openrouter" and cfg.get("free_only", True):
        models = [m for m in models if m.endswith(":free")]
    return {"id": pid, "base_url": (cfg.get("base_url") or cat["base_url"]).rstrip("/"),
            "key": cfg.get("key", ""), "models": models, "free_only": bool(cfg.get("free_only", pid == "openrouter"))}


def active() -> str:
    a = load().get("active", "ollama")
    return a if a in CATALOG or a == "none" else "ollama"


def public() -> dict:
    """For the UI: everything except the keys themselves."""
    data = load()
    out = []
    for pid in ORDER:
        cat = CATALOG[pid]
        cfg = data["providers"].get(pid, {})
        key = cfg.get("key", "")
        out.append({
            "id": pid, "name": cat["name"], "kind": cat["kind"], "about": cat["about"],
            "key_url": cat.get("key_url", ""), "key_prefix": cat.get("key_prefix", ""),
            "base_url": cfg.get("base_url") or cfg.get("host") or cat["base_url"],
            "base_url_editable": pid in ("custom", "ollama"),
            "key_set": bool(key), "key_hint": f"…{key[-4:]}" if len(key) > 8 else "",
            "models": [m for m in (cfg.get("models") or []) if m] or list(cat["models"]),
            "suggested": cat["models"],
            "free_only": bool(cfg.get("free_only", pid == "openrouter")),
            "configured": pid == "ollama" or bool(key) or (pid == "custom" and bool(cfg.get("base_url"))),
        })
    return {"active": active(), "providers": out}


def update(pid: str, patch: dict) -> None:
    if pid not in CATALOG:
        raise KeyError(pid)
    data = load()
    cfg = data["providers"].setdefault(pid, {})
    if "key" in patch:
        k = str(patch["key"] or "").strip()
        if k == "__clear__":
            cfg.pop("key", None)
        elif k:
            cfg["key"] = k
    if "base_url" in patch and pid in ("custom", "ollama"):
        cfg["host" if pid == "ollama" else "base_url"] = str(patch["base_url"]).strip().rstrip("/")
    if "models" in patch:
        cfg["models"] = list(dict.fromkeys(str(m).strip() for m in patch["models"] if str(m).strip()))[:8]
    if "free_only" in patch:
        cfg["free_only"] = bool(patch["free_only"])
    if patch.get("activate"):
        data["active"] = pid
    save(data)
    if pid == "ollama":          # keep the legacy settings in step (pipeline unload, health)
        settings.ollama_host = get("ollama")["base_url"]
        settings.ollama_model = get("ollama")["models"][0]


def set_active(pid: str) -> None:
    if pid not in CATALOG and pid != "none":
        raise KeyError(pid)
    data = load()
    data["active"] = pid
    save(data)
