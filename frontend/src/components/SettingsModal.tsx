import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowUp, CheckCircle2, Download, ExternalLink, KeyRound, Loader2, Plus, X, XCircle } from "lucide-react";
import { api, type AppSettings, type ProviderInfo, type ProvidersState } from "../api";

type Tab = "providers" | "general";

export function SettingsModal({ onClose, onSaved }: { onClose: () => void; onSaved: () => void }) {
  const [tab, setTab] = useState<Tab>("providers");
  return (
    <Modal onClose={onClose} wide>
      <h2>Settings</h2>
      <div className="seg tabs">
        <button className={tab === "providers" ? "sel" : ""} onClick={() => setTab("providers")}>AI providers</button>
        <button className={tab === "general" ? "sel" : ""} onClick={() => setTab("general")}>Transcription</button>
      </div>
      {tab === "providers" ? <ProvidersTab onSaved={onSaved} /> : <GeneralTab onClose={onClose} onSaved={onSaved} />}
    </Modal>
  );
}

/* ------------------------------------------------------------------ providers */
function ProvidersTab({ onSaved }: { onSaved: () => void }) {
  const [st, setSt] = useState<ProvidersState | null>(null);
  const [open, setOpen] = useState<string>("");
  const [err, setErr] = useState("");

  const load = useCallback(() => api.providers().then(setSt).catch((e) => setErr(e.message)), []);
  useEffect(() => {
    load();
  }, [load]);

  if (!st) return <div className="muted">{err || "Loading…"}</div>;

  const activate = async (id: string) => {
    setSt(await api.setActiveProvider(id));
    onSaved();
  };
  const configured = st.providers.filter((p) => p.configured);

  return (
    <div className="prov">
      <div className="field-label">AI that finds and ranks clips</div>
      <div className="seg">
        {configured.map((p) => (
          <button key={p.id} className={st.active === p.id ? "sel" : ""} onClick={() => activate(p.id)}>
            {p.name}
          </button>
        ))}
        <button className={st.active === "none" ? "sel" : ""} onClick={() => activate("none")}>Rules only</button>
      </div>
      <p className="hint">
        Add your own API key below. Cloud models are much stronger than a local 7B model, and the free ones cost nothing.
        Keys stay in <code>providers.json</code> on this PC and are only sent to their own provider.
      </p>

      <div className="prov-list">
        {st.providers.map((p) => (
          <ProviderCard
            key={p.id}
            p={p}
            active={st.active === p.id}
            open={open === p.id}
            onToggle={() => setOpen(open === p.id ? "" : p.id)}
            onChange={(next) => {
              setSt(next);
              onSaved();
            }}
            state={st}
          />
        ))}
      </div>
      {err && <div className="err-text">{err}</div>}
    </div>
  );
}

function ProviderCard({
  p,
  active,
  open,
  onToggle,
  onChange,
  state,
}: {
  p: ProviderInfo;
  active: boolean;
  open: boolean;
  onToggle: () => void;
  onChange: (s: ProvidersState) => void;
  state: ProvidersState;
}) {
  const [key, setKey] = useState("");
  const [base, setBase] = useState(p.base_url);
  const [models, setModels] = useState<string[]>(p.models);
  const [freeOnly, setFreeOnly] = useState(p.free_only);
  const [catalog, setCatalog] = useState<{ id: string; free: boolean | null }[] | null>(null);
  const [adding, setAdding] = useState("");
  const [busy, setBusy] = useState(false);
  const [test, setTest] = useState<{ ok: boolean; detail: string } | "busy" | null>(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    setModels(p.models);
    setFreeOnly(p.free_only);
    setBase(p.base_url);
  }, [p.models.join("|"), p.free_only, p.base_url]); // eslint-disable-line react-hooks/exhaustive-deps

  const dirty =
    !!key || base !== p.base_url || freeOnly !== p.free_only || models.join("|") !== p.models.join("|");

  const loadCatalog = async () => {
    try {
      setCatalog((await api.providerModels(p.id)).models);
    } catch (e) {
      setErr((e as Error).message);
    }
  };
  useEffect(() => {
    if (open && !catalog && (p.key_set || p.id === "ollama" || p.id === "nvidia" || p.id === "openrouter")) loadCatalog();
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps

  const save = async (activate = false) => {
    setBusy(true);
    setErr("");
    try {
      const next = await api.saveProvider(p.id, {
        ...(key ? { key } : {}),
        ...(p.base_url_editable ? { base_url: base } : {}),
        models,
        free_only: freeOnly,
        ...(activate ? { activate: true } : {}),
      });
      setKey("");
      onChange(next);
      return next;
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const runTest = async () => {
    setTest("busy");
    if (dirty) await save();
    try {
      setTest(await api.testProvider(p.id));
    } catch (e) {
      setTest({ ok: false, detail: (e as Error).message });
    }
  };

  const add = (m: string) => {
    m = m.trim();
    if (!m || models.includes(m)) return;
    setModels([...models, m]);
    setAdding("");
  };
  const blocked = (m: string) => p.id === "openrouter" && freeOnly && !m.endsWith(":free");
  const status = p.id === "ollama" ? "local" : p.key_set ? `key ${p.key_hint}` : p.id === "custom" && p.configured ? "set" : "no key";

  return (
    <div className={`prov-card ${open ? "open" : ""} ${active ? "active" : ""}`}>
      <button className="prov-head" onClick={onToggle}>
        <span className={`dot ${p.configured ? "ok" : ""}`} />
        <b>{p.name}</b>
        {active && <span className="badge">In use</span>}
        <span className="prov-sub">{p.models[0] || "no model"}</span>
        <span className="prov-status">{status}</span>
      </button>

      {open && (
        <div className="prov-body">
          <p className="hint">{p.about}</p>

          {p.kind === "cloud" && (
            <label className="prov-field">
              <span>
                API key {p.key_set && <em className="muted">saved {p.key_hint}</em>}
                {p.key_url && (
                  <a href={p.key_url} target="_blank" rel="noreferrer" className="link">
                    Get a key <ExternalLink size={12} />
                  </a>
                )}
              </span>
              <div className="key-row">
                <KeyRound size={15} className="muted" />
                <input
                  type="password"
                  autoComplete="off"
                  placeholder={p.key_set ? "leave empty to keep the saved key" : `${p.key_prefix || ""}…`}
                  value={key}
                  onChange={(e) => setKey(e.target.value)}
                />
                {p.key_set && (
                  <button className="chip-btn danger" onClick={async () => onChange(await api.saveProvider(p.id, { key: "__clear__" }))}>
                    Remove
                  </button>
                )}
              </div>
            </label>
          )}

          {p.base_url_editable && (
            <label className="prov-field">
              <span>{p.id === "ollama" ? "Ollama address" : "Server address (…/v1)"}</span>
              <input className="input" value={base} onChange={(e) => setBase(e.target.value)} placeholder="http://localhost:1234/v1" />
            </label>
          )}

          <div className="prov-field">
            <span>
              {p.id === "ollama" ? "Model" : "Allowed models"}
              <em className="muted">
                {p.id === "ollama" ? "the local model ClipForge uses" : "only these are ever called; the first is used, the rest take over when it is busy"}
              </em>
            </span>
            {p.id === "ollama" ? (
              <select className="input" value={models[0] ?? ""} onChange={(e) => setModels([e.target.value])}>
                {[...new Set([...(models[0] ? [models[0]] : []), ...state.ollama_models.filter((m) => !m.includes("embed"))])].map((m) => (
                  <option key={m}>{m}</option>
                ))}
              </select>
            ) : (
              <>
                <ol className="model-list">
                  {models.map((m, i) => (
                    <li key={m} className={blocked(m) ? "blocked" : ""} title={blocked(m) ? "skipped: not a free model" : ""}>
                      <span className="model-n">{i + 1}</span>
                      <code>{m}</code>
                      {blocked(m) && <em>paid, skipped</em>}
                      {i > 0 && (
                        <button className="icon-btn sm" title="Move up" onClick={() => setModels([...models.slice(0, i - 1), m, models[i - 1], ...models.slice(i + 1)])}>
                          <ArrowUp size={14} />
                        </button>
                      )}
                      <button className="icon-btn sm" title="Remove" onClick={() => setModels(models.filter((x) => x !== m))}>
                        <X size={14} />
                      </button>
                    </li>
                  ))}
                  {!models.length && <li className="muted">No model allowed: this provider is switched off.</li>}
                </ol>
                <div className="key-row">
                  <input
                    className="input"
                    list={`models-${p.id}`}
                    placeholder={catalog ? `Add a model (${catalog.length} available)` : "Add a model id"}
                    value={adding}
                    onChange={(e) => setAdding(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && add(adding)}
                  />
                  <datalist id={`models-${p.id}`}>
                    {(catalog ?? [])
                      .filter((m) => !(p.id === "openrouter" && freeOnly && !m.free))
                      .map((m) => (
                        <option key={m.id} value={m.id}>{m.free ? "free" : ""}</option>
                      ))}
                  </datalist>
                  <button className="btn ghost" onClick={() => add(adding)} disabled={!adding.trim()}>
                    <Plus size={15} /> Add
                  </button>
                </div>
                {p.id === "openrouter" && (
                  <label className="check">
                    <input type="checkbox" checked={freeOnly} onChange={(e) => setFreeOnly(e.target.checked)} />
                    Free models only (never spend credits)
                  </label>
                )}
              </>
            )}
          </div>

          {p.id === "ollama" && <OllamaDownloads state={state} onDone={(s) => onChange(s)} />}

          <div className="test-row">
            <button className="btn ghost" onClick={runTest} disabled={test === "busy" || !(p.configured || key || (p.id === "custom" && base))}>
              {test === "busy" ? <Loader2 className="spin" size={15} /> : null} Test
            </button>
            {test && test !== "busy" && (
              <span className={test.ok ? "ok-text" : "err-text"}>
                {test.ok ? <CheckCircle2 size={15} /> : <XCircle size={15} />} {test.detail}
              </span>
            )}
            <span className="grow" />
            <button className="btn ghost" onClick={() => save()} disabled={busy || !dirty}>Save</button>
            {!active && (
              <button className="btn primary" onClick={() => save(true)} disabled={busy || !(p.configured || key)}>
                Save and use
              </button>
            )}
          </div>
          {err && <div className="err-text">{err}</div>}
        </div>
      )}
    </div>
  );
}

function OllamaDownloads({ state, onDone }: { state: ProvidersState; onDone: (s: ProvidersState) => void }) {
  const [pull, setPull] = useState(state.pull);
  const [err, setErr] = useState("");
  const timer = useRef<number>();

  const poll = useCallback(() => {
    window.clearInterval(timer.current);
    timer.current = window.setInterval(async () => {
      const s = await api.pullState();
      setPull(s);
      if (s.done) {
        window.clearInterval(timer.current);
        onDone(await api.providers());
      }
    }, 1000);
  }, [onDone]);

  useEffect(() => {
    if (!state.pull.done) poll();
    return () => window.clearInterval(timer.current);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const start = async (model: string) => {
    setErr("");
    try {
      setPull(await api.pullModel(model));
      poll();
    } catch (e) {
      setErr((e as Error).message);
    }
  };
  const installed = (m: string) => state.ollama_models.some((x) => x === m || x === `${m}:latest`);
  const pct = pull.total ? Math.round((pull.completed / pull.total) * 100) : 0;

  return (
    <div className="prov-field">
      <span>Download a local model</span>
      <div className="pull-list">
        {state.ollama_picks.map((m) => {
          const running = !pull.done && pull.model === m.model;
          return (
            <div key={m.model} className="pull-row">
              <div>
                <code>{m.model}</code> <em className="muted">{m.size}</em>
                <div className="hint">{m.note}</div>
                {running && (
                  <div className="pull-bar" title={pull.status}>
                    <div style={{ width: `${pct}%` }} />
                  </div>
                )}
                {pull.done && pull.model === m.model && pull.error && <div className="err-text">{pull.error}</div>}
              </div>
              {installed(m.model) ? (
                <span className="ok-text"><CheckCircle2 size={15} /> Installed</span>
              ) : running ? (
                <span className="muted">{pct ? `${pct}%` : pull.status}</span>
              ) : (
                <button className="btn ghost" disabled={!pull.done} onClick={() => start(m.model)}>
                  <Download size={15} /> Download
                </button>
              )}
            </div>
          );
        })}
      </div>
      {err && <div className="err-text">{err}</div>}
    </div>
  );
}

/* ------------------------------------------------------------------ general */
function GeneralTab({ onClose, onSaved }: { onClose: () => void; onSaved: () => void }) {
  const [s, setS] = useState<AppSettings | null>(null);
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState("");

  useEffect(() => {
    api.settings().then(setS).catch((e) => setErr(e.message));
  }, []);
  if (!s) return <div className="muted">{err || "Loading…"}</div>;

  const save = async () => {
    setSaving(true);
    try {
      setS(await api.saveSettings({ whisper_model: s.whisper_model }));
      onSaved();
      onClose();
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <>
      <div className="form-grid">
        <label>
          <span>Default Whisper model</span>
          <select value={s.whisper_model} onChange={(e) => setS({ ...s, whisper_model: e.target.value })}>
            {s.whisper_models.map((m) => (
              <option key={m}>{m}</option>
            ))}
          </select>
        </label>
        <label>
          <span>Encoder</span>
          <input value={s.encoder} disabled />
        </label>
        <p className="hint">large-v3-turbo is the best balance on a GPU. Use small on a CPU-only PC.</p>
      </div>
      {err && <div className="err-text">{err}</div>}
      <div className="modal-actions">
        <button className="btn ghost" onClick={onClose}>Cancel</button>
        <button className="btn primary" onClick={save} disabled={saving}>{saving ? "Saving…" : "Save"}</button>
      </div>
    </>
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
