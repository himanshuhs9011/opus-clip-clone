import { useEffect, useState } from "react";
import { CheckCircle2, Loader2, X, XCircle } from "lucide-react";
import { api, type AppSettings } from "../api";

const PROVIDERS = [
  { id: "ollama", label: "Ollama", sub: "Local and free. Runs on your GPU." },
  { id: "openrouter", label: "OpenRouter", sub: "Optional cloud models. Needs an API key." },
  { id: "none", label: "Rules only", sub: "No AI. The explainable scorer picks clips." },
];

export function SettingsModal({ onClose, onSaved }: { onClose: () => void; onSaved: () => void }) {
  const [s, setS] = useState<AppSettings | null>(null);
  const [key, setKey] = useState("");
  const [saving, setSaving] = useState(false);
  const [test, setTest] = useState<{ ok: boolean; detail: string } | "busy" | null>(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    api.settings().then(setS).catch((e) => setErr(e.message));
  }, []);

  if (!s) {
    return (
      <Modal onClose={onClose}>
        <div className="muted">{err || "Loading…"}</div>
      </Modal>
    );
  }

  const set = (patch: Partial<AppSettings>) => setS({ ...s, ...patch });

  const save = async () => {
    setSaving(true);
    setErr("");
    try {
      const out = await api.saveSettings({
        llm_provider: s.llm_provider,
        ollama_host: s.ollama_host,
        ollama_model: s.ollama_model,
        llm_model: s.llm_model,
        whisper_model: s.whisper_model,
        ...(key ? { llm_api_key: key } : {}),
      });
      setS(out);
      setKey("");
      onSaved();
      onClose();
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setSaving(false);
    }
  };

  const runTest = async () => {
    setTest("busy");
    try {
      if (key) await api.saveSettings({ llm_api_key: key, llm_model: s.llm_model });
      const model = s.llm_provider === "ollama" ? s.ollama_model : s.llm_model;
      setTest(await api.testLLM(s.llm_provider, model, true));
    } catch (e) {
      setTest({ ok: false, detail: (e as Error).message });
    }
  };

  return (
    <Modal onClose={onClose}>
      <h2>Settings</h2>
      <div className="field-label">AI that picks &amp; ranks clips</div>
      <div className="provider-grid">
        {PROVIDERS.map((p) => (
          <button
            key={p.id}
            className={`provider ${s.llm_provider === p.id ? "sel" : ""}`}
            onClick={() => {
              set({ llm_provider: p.id });
              setTest(null);
            }}
          >
            <b>{p.label}</b>
            <span>{p.sub}</span>
          </button>
        ))}
      </div>

      {s.llm_provider === "ollama" && (
        <div className="form-grid">
          <label>
            <span>Model</span>
            {s.ollama_models.length ? (
              <select value={s.ollama_model} onChange={(e) => set({ ollama_model: e.target.value })}>
                {[...new Set([s.ollama_model, ...s.ollama_models])].map((m) => (
                  <option key={m}>{m}</option>
                ))}
              </select>
            ) : (
              <input value={s.ollama_model} onChange={(e) => set({ ollama_model: e.target.value })} />
            )}
          </label>
          <label>
            <span>Host</span>
            <input value={s.ollama_host} onChange={(e) => set({ ollama_host: e.target.value })} />
          </label>
          <p className="hint">
            qwen2.5:7b fits an 8 GB GPU. ClipForge unloads Ollama before Whisper runs, so both share the card.
            Pull more models with <code>ollama pull llama3.1:8b</code>.
          </p>
        </div>
      )}

      {s.llm_provider === "openrouter" && (
        <div className="form-grid">
          <label className="span2">
            <span>OpenRouter API key {s.openrouter_key_set && <em className="muted">saved {s.openrouter_key_hint}</em>}</span>
            <input
              type="password"
              placeholder={s.openrouter_key_set ? "leave empty to keep the saved key" : "sk-or-v1-…"}
              value={key}
              onChange={(e) => setKey(e.target.value)}
              autoComplete="off"
            />
          </label>
          <label className="span2">
            <span>Model</span>
            <input value={s.llm_model} onChange={(e) => set({ llm_model: e.target.value })} placeholder="google/gemini-2.5-flash" />
          </label>
          <p className="hint">
            Stored in your local <code>.env</code>. Any OpenRouter model id works, e.g. <code>google/gemini-2.5-flash</code>,{" "}
            <code>anthropic/claude-haiku-4.5</code> or a <code>:free</code> model.
          </p>
        </div>
      )}

      {s.llm_provider !== "none" && (
        <div className="test-row">
          <button className="btn ghost" onClick={runTest} disabled={test === "busy"}>
            {test === "busy" ? <Loader2 className="spin" size={15} /> : null} Test connection
          </button>
          {test && test !== "busy" && (
            <span className={test.ok ? "ok-text" : "err-text"}>
              {test.ok ? <CheckCircle2 size={15} /> : <XCircle size={15} />} {test.detail}
            </span>
          )}
        </div>
      )}

      <div className="field-label" style={{ marginTop: 18 }}>Transcription</div>
      <div className="form-grid">
        <label>
          <span>Default Whisper model</span>
          <select value={s.whisper_model} onChange={(e) => set({ whisper_model: e.target.value })}>
            {s.whisper_models.map((m) => (
              <option key={m}>{m}</option>
            ))}
          </select>
        </label>
        <label>
          <span>Encoder</span>
          <input value={s.encoder} disabled />
        </label>
      </div>

      {err && <div className="err-text">{err}</div>}
      <div className="modal-actions">
        <button className="btn ghost" onClick={onClose}>Cancel</button>
        <button className="btn primary" onClick={save} disabled={saving}>
          {saving ? "Saving…" : "Save"}
        </button>
      </div>
    </Modal>
  );
}

export function Modal({ children, onClose, wide = false }: { children: React.ReactNode; onClose: () => void; wide?: boolean }) {
  useEffect(() => {
    const on = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  }, [onClose]);
  return (
    <div className="modal" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className={`modal-card ${wide ? "wide" : ""}`}>
        <button className="modal-x icon-btn" onClick={onClose}>
          <X size={18} />
        </button>
        {children}
      </div>
    </div>
  );
}
