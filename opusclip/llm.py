"""LLM providers for highlight ranking.

* ollama     - local and free (default). Uses /api/chat with a JSON schema, so even 7B
               models return valid JSON.
* openrouter - optional, needs OPENROUTER_API_KEY. Any OpenAI-compatible endpoint works
               the same way (provider "openai" + LLM_BASE_URL + LLM_API_KEY).
* none       - rules-only selection (the heuristic scorer still runs).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import httpx

from .config import settings


class LLMError(RuntimeError):
    pass


PROVIDERS = ("ollama", "openrouter", "openai", "none")


def _strip_json(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S)
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        return s[i : j + 1]
    return s


# --------------------------------------------------------------------------
# Ollama helpers
# --------------------------------------------------------------------------
def ollama_models(host: str | None = None) -> list[str]:
    host = (host or settings.ollama_host).rstrip("/")
    try:
        r = httpx.get(f"{host}/api/tags", timeout=3)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def _ollama_exe() -> str | None:
    exe = shutil.which("ollama")
    if exe:
        return exe
    guess = Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe"
    return str(guess) if guess.exists() else None


def ensure_ollama(host: str | None = None, wait: float = 12.0) -> bool:
    """Start `ollama serve` in the background if it is installed but not running."""
    host = (host or settings.ollama_host).rstrip("/")
    try:
        httpx.get(f"{host}/api/version", timeout=2)
        return True
    except Exception:
        pass
    if "localhost" not in host and "127.0.0.1" not in host:
        return False
    exe = _ollama_exe()
    if not exe:
        return False
    flags = 0
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        # a hidden console (not DETACHED_PROCESS): Ollama's runner processes inherit it,
        # otherwise each one pops up its own blank terminal window
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    try:
        subprocess.Popen([exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=flags)
    except Exception:
        return False
    t0 = time.time()
    while time.time() - t0 < wait:
        try:
            httpx.get(f"{host}/api/version", timeout=1)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def unload_ollama(host: str | None = None) -> list[str]:
    """Free VRAM held by Ollama so Whisper and the LLM never sit on an 8 GB GPU together."""
    host = (host or settings.ollama_host).rstrip("/")
    try:
        loaded = httpx.get(f"{host}/api/ps", timeout=2).json().get("models", [])
        for m in loaded:
            httpx.post(f"{host}/api/generate", json={"model": m["name"], "keep_alive": 0}, timeout=30)
        return [m["name"] for m in loaded]
    except Exception:
        return []


# --------------------------------------------------------------------------
# client
# --------------------------------------------------------------------------
class LLM:
    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = (provider or settings.llm_provider or "none").lower()
        if self.provider not in PROVIDERS:
            self.provider = "none"
        if self.provider == "ollama":
            self.base_url = settings.ollama_host.rstrip("/")
            self.api_key = ""
            self.models = [model or settings.ollama_model]
        else:
            self.base_url = (
                "https://openrouter.ai/api/v1" if self.provider == "openrouter" and
                "openrouter" not in settings.llm_base_url else settings.llm_base_url
            ).rstrip("/")
            self.api_key = settings.llm_api_key
            self.models = [model or settings.llm_model, *settings.llm_fallback_models]
        self.models = list(dict.fromkeys(m for m in self.models if m))

    # ------------------------------------------------------------------
    @property
    def model(self) -> str:
        return self.models[0] if self.models else ""

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}" if self.available else "rules"

    @property
    def available(self) -> bool:
        if self.provider == "none" or not self.models:
            return False
        if self.provider == "ollama":
            return True
        return bool(self.api_key)

    @property
    def context_chars(self) -> int:
        """How much transcript fits in one request."""
        if self.provider == "ollama":
            return max(6000, int(settings.ollama_num_ctx * 2.6))
        return 90_000

    # ------------------------------------------------------------------
    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        if "openrouter" in self.base_url:
            h["HTTP-Referer"] = "http://localhost:8000"
            h["X-Title"] = "ClipForge"
        return h

    def _ollama(self, system: str, user: str, schema: dict | None, max_tokens: int,
                temperature: float) -> str:
        if not ensure_ollama(self.base_url):
            raise LLMError(f"Ollama is not running at {self.base_url} - open the Ollama app")
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "format": schema or "json",
            "options": {"temperature": temperature, "num_ctx": settings.ollama_num_ctx,
                        "num_predict": max_tokens},
        }
        try:
            with httpx.Client(timeout=settings.llm_timeout) as c:
                r = c.post(f"{self.base_url}/api/chat", json=payload)
        except httpx.HTTPError as e:
            raise LLMError(f"Ollama request failed: {e}") from e
        if r.status_code == 404:
            raise LLMError(f"Ollama model '{self.model}' is not pulled - run: ollama pull {self.model}")
        if r.status_code != 200:
            raise LLMError(f"Ollama HTTP {r.status_code}: {r.text[:300]}")
        return (r.json().get("message") or {}).get("content", "")

    def _openai(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        last_err: Exception | None = None
        for candidate in self.models:
            payload = {
                "model": candidate,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            for attempt in range(2):
                try:
                    with httpx.Client(timeout=settings.llm_timeout) as client:
                        r = client.post(f"{self.base_url}/chat/completions",
                                        headers=self._headers(), json=payload)
                except httpx.HTTPError as e:
                    last_err = e
                    time.sleep(1 + attempt)
                    continue
                if r.status_code == 200:
                    choice = (r.json().get("choices") or [{}])[0]
                    text = (choice.get("message") or {}).get("content")
                    if text:
                        return text
                    last_err = LLMError(f"{candidate}: empty reply "
                                        f"(finish_reason={choice.get('finish_reason')})")
                    break
                if r.status_code in (401, 403):
                    raise LLMError(f"API key rejected by {self.base_url} ({r.status_code})")
                if r.status_code in (429, 502, 503):
                    last_err = LLMError(f"{candidate}: HTTP {r.status_code}")
                    time.sleep(2 + attempt * 3)
                    continue
                last_err = LLMError(f"{candidate}: HTTP {r.status_code} {r.text[:200]}")
                break
        raise LLMError(str(last_err))

    def complete(self, system: str, user: str, schema: dict | None = None,
                 max_tokens: int = 4000, temperature: float = 0.3) -> str:
        if not self.available:
            raise LLMError("LLM disabled or not configured")
        if self.provider == "ollama":
            return self._ollama(system, user, schema, max_tokens, temperature)
        return self._openai(system, user, max_tokens, temperature)

    def json_completion(self, system: str, user: str, schema: dict | None = None,
                        max_tokens: int = 4000, temperature: float = 0.3) -> dict:
        last: Exception | None = None
        prompt = user
        for _ in range(2):
            try:
                raw = self.complete(system, prompt, schema=schema, max_tokens=max_tokens,
                                    temperature=temperature)
                return json.loads(_strip_json(raw))
            except LLMError as e:
                if "rejected" in str(e) or "not pulled" in str(e) or "not running" in str(e):
                    raise
                last = e
            except Exception as e:
                last = e
            prompt = user + "\n\nReply with the JSON object only - no prose, no markdown fences."
        raise LLMError(f"invalid JSON from model: {last}")

    # ------------------------------------------------------------------
    def verify(self) -> tuple[bool, str]:
        """Cheap check, cached for a minute."""
        if self.provider == "none":
            return False, "rules only"
        if not self.available:
            return False, "no API key"
        key = (self.provider, self.base_url, self.model, self.api_key[-6:])
        now = time.time()
        hit = _verify_cache.get(key)
        if hit and now - hit[0] < 60:
            return hit[1]
        result: tuple[bool, str]
        try:
            if self.provider == "ollama":
                models = ollama_models(self.base_url)
                if not models and ensure_ollama(self.base_url, wait=8):
                    models = ollama_models(self.base_url)
                if not models and not ensure_ollama(self.base_url, wait=0):
                    result = (False, "Ollama not running")
                elif models and not any(m == self.model or m.split(":")[0] == self.model for m in models):
                    result = (False, f"model not pulled: ollama pull {self.model}")
                else:
                    result = (True, "ok")
            elif "openrouter" in self.base_url:
                r = httpx.get(f"{self.base_url}/key", headers=self._headers(), timeout=8)
                if r.status_code == 200:
                    result = (True, "ok")
                else:
                    try:
                        msg = r.json().get("error", {}).get("message", f"HTTP {r.status_code}")
                    except Exception:
                        msg = f"HTTP {r.status_code}"
                    result = (False, msg)
            else:
                r = httpx.get(f"{self.base_url}/models", headers=self._headers(), timeout=8)
                result = (r.status_code == 200, "ok" if r.status_code == 200 else f"HTTP {r.status_code}")
        except Exception as e:
            result = (False, str(e)[:120])
        _verify_cache[key] = (now, result)
        return result


_verify_cache: dict[tuple, tuple[float, tuple[bool, str]]] = {}


def health() -> dict:
    out = {"provider": settings.llm_provider, "providers": {}}
    for prov in ("ollama", "openrouter"):
        llm = LLM(prov)
        ok, detail = llm.verify()
        out["providers"][prov] = {
            "ok": ok,
            "detail": detail,
            "model": llm.model,
            "configured": llm.available,
        }
    out["ollama_models"] = ollama_models()
    out["openrouter_key_set"] = bool(settings.llm_api_key)
    return out
