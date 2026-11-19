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


# downloads started from Settings > Providers (one at a time)
OLLAMA_PICKS = [
    {"model": "gemma4:e4b", "size": "6.6 GB", "note": "Gemma 4. Best local pick for an 8 GB GPU, good Hindi."},
    {"model": "gemma4:e2b", "size": "4.6 GB", "note": "Gemma 4 small. For 4-6 GB GPUs and laptops."},
    {"model": "gemma4:12b-it-qat", "size": "7.2 GB", "note": "Gemma 4 12B. Stronger; wants 10 GB+ VRAM to be quick."},
    {"model": "qwen2.5:7b", "size": "4.7 GB", "note": "Older default. Fast, weaker at Hindi."},
]
pull_state: dict = {"model": "", "status": "", "completed": 0, "total": 0, "error": "", "done": True}


def start_pull(model: str, host: str | None = None) -> bool:
    import threading

    if not pull_state["done"]:
        return False
    host = (host or settings.ollama_host).rstrip("/")
    pull_state.update(model=model, status="starting", completed=0, total=0, error="", done=False)

    def run() -> None:
        try:
            if not ensure_ollama(host):
                raise LLMError("Ollama is not installed or not running")
            with httpx.stream("POST", f"{host}/api/pull", json={"model": model, "stream": True},
                              timeout=httpx.Timeout(30, read=600)) as r:
                for line in r.iter_lines():
                    if not line:
                        continue
                    msg = json.loads(line)
                    if msg.get("error"):
                        raise LLMError(msg["error"])
                    pull_state["status"] = msg.get("status", "")
                    if msg.get("total"):
                        pull_state["total"] = msg["total"]
                        pull_state["completed"] = msg.get("completed", 0)
            pull_state["status"] = "success"
        except Exception as e:
            err = str(e)
            if "newer version" in err or "412" in err:
                err += " - update Ollama from ollama.com/download"
            pull_state["error"] = err[:300]
        finally:
            pull_state["done"] = True

    threading.Thread(target=run, daemon=True).start()
    return True


# --------------------------------------------------------------------------
# client
# --------------------------------------------------------------------------
class LLM:
    def __init__(self, provider: str | None = None, model: str | None = None):
        from . import providers

        prov = (provider or providers.active() or "none").lower()
        prov = "custom" if prov == "openai" else prov
        if prov not in providers.CATALOG:
            prov = "none"
        self.provider = prov
        self.base_url, self.api_key, self.models = "", "", []
        self.free_only = False
        self.used_model = ""
        if prov == "none":
            return
        cfg = providers.get(prov)
        self.base_url, self.api_key = cfg["base_url"], cfg["key"]
        self.free_only = cfg["free_only"]
        allowed = cfg["models"]
        if prov == "ollama":
            self.models = [model or (allowed[0] if allowed else settings.ollama_model)]
        else:
            # only models the user allowed for this provider are ever called
            first = [model] if model and (model in allowed or not allowed) else []
            if self.free_only:
                first = [m for m in first if m.endswith(":free")]
            self.models = [*first, *allowed]
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
        if self.provider == "custom":
            return bool(self.base_url)
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
        """Tries each allowed model in order. A busy (429) model hands over to the next one
        quickly; only the last model waits out the rate limit."""
        last_err: Exception | None = None
        for ci, candidate in enumerate(self.models):
            is_last = ci == len(self.models) - 1
            payload = {
                "model": candidate,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": temperature,
                # reasoning models (Nemotron, Gemini 2.5...) think before answering and that
                # counts against max_tokens: leave room so the JSON is not cut off
                "max_tokens": min(32000, max_tokens * 2 + 4000),
            }
            tries = 5 if is_last else 2
            for attempt in range(tries):
                try:
                    with httpx.Client(timeout=settings.llm_timeout) as client:
                        r = client.post(f"{self.base_url}/chat/completions",
                                        headers=self._headers(), json=payload)
                except httpx.HTTPError as e:
                    last_err = LLMError(f"{candidate}: {e}")
                    if attempt >= 1:
                        break
                    time.sleep(2)
                    continue
                if r.status_code == 200:
                    try:
                        choice = (r.json().get("choices") or [{}])[0]
                    except ValueError:
                        last_err = LLMError(f"{candidate}: reply was not JSON")
                        break
                    text = (choice.get("message") or {}).get("content")
                    if text:
                        self.used_model = candidate
                        return text
                    last_err = LLMError(f"{candidate}: empty reply "
                                        f"(finish_reason={choice.get('finish_reason')})")
                    break
                if r.status_code in (401, 403):
                    raise LLMError(f"API key rejected by {self.base_url} ({r.status_code})")
                if r.status_code in (429, 500, 502, 503, 504):
                    last_err = LLMError(f"{candidate}: HTTP {r.status_code} (busy or rate limited)")
                    if attempt == tries - 1:
                        break
                    try:
                        wait = float(r.headers.get("retry-after") or 0)
                    except ValueError:
                        wait = 0.0
                    default = (6, 15, 30, 45)[attempt] if is_last else 3
                    time.sleep(min(60.0, wait or default))
                    continue
                if r.status_code == 404:
                    last_err = LLMError(f"{candidate}: model not available on {self.base_url} "
                                        "(retired or renamed) - pick another in Settings > Providers")
                    break
                last_err = LLMError(f"{candidate}: HTTP {r.status_code} {r.text[:200]}")
                break
        raise LLMError(str(last_err or "no model configured"))

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
            return False, ("no model allowed" if not self.models else
                           "no server address" if self.provider == "custom" else "no API key")
        key =(self.provider, self.base_url, self.model, self.api_key[-6:])
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
    from . import providers

    active = providers.active()
    out = {"provider": active, "providers": {}}
    for p in providers.public()["providers"]:
        if not p["configured"] and p["id"] != active:
            continue
        llm = LLM(p["id"])
        ok, detail = llm.verify()
        out["providers"][p["id"]] = {"ok": ok, "detail": detail, "model": llm.model,
                                     "configured": llm.available, "name": p["name"]}
    out["ollama_models"] = ollama_models()
    out["openrouter_key_set"] = bool(providers.get("openrouter").get("key"))
    return out
