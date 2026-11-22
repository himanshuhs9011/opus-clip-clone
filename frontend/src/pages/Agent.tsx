// Clip Agent: say what clips you want in plain words. A plan to approve, then
// find -> pick -> make. Backend: opusclip/wizard.py and /api/agent/*.
import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTriangle, ArrowUp, Bot, Check, Download, Film, ListChecks, Loader2, Paperclip, Pencil, X,
} from "lucide-react";
import { api, media, type AgentPlan, type AgentRequest, type Clip, type Presets, type Project } from "../api";
import { ScoreBadge } from "../components/Score";
import { fmtTime, go, slug, useInterval } from "../lib/util";

type Msg =
  | { id: string; kind: "user"; text: string; file?: string }
  | { id: string; kind: "plan"; request: AgentRequest; plan: AgentPlan; state: "pending" | "approved" | "changed"; info?: boolean }
  | { id: string; kind: "run"; pid: string }
  | { id: string; kind: "note"; text: string; tone?: "error" };

const STORE = "cf.agent.v1";
const EXAMPLES = [
  "Turn this into Shorts with captions: ",
  "Make 10 minute YouTube videos and shorts from this Kick stream: ",
  "Clip this podcast into clips under 45 seconds, only the parts about money: ",
  "The 5 funniest moments from this stream, face-cam on top: ",
  "Is stream ke best moments ki 12 minute ki lambi video aur shorts dono bana do: ",
  "Every time someone gets voted out, from 10:00 to 40:00: ",
];
const uid = () => Math.random().toString(36).slice(2, 10);

function loadThread(): Msg[] {
  try {
    const t = JSON.parse(localStorage.getItem(STORE) || "[]");
    return Array.isArray(t) ? t.slice(-60) : [];
  } catch {
    return [];
  }
}

/** Upload with progress (big files) - same contract as the Home page. */
function upload(url: string, fd: FormData, onPct: (f: number) => void): Promise<{ id: string }> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);
    xhr.upload.onprogress = (e) => e.lengthComputable && onPct(e.loaded / e.total);
    xhr.onload = () => {
      try {
        const j = JSON.parse(xhr.responseText);
        if (xhr.status >= 300) reject(new Error(j.detail || `HTTP ${xhr.status}`));
        else resolve(j);
      } catch {
        reject(new Error(`HTTP ${xhr.status}`));
      }
    };
    xhr.onerror = () => reject(new Error("network error"));
    xhr.send(fd);
  });
}

export function Agent({ presets }: { presets: Presets | null }) {
  const [thread, setThread] = useState<Msg[]>(loadThread);
  const [text, setText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [pct, setPct] = useState(0);
  const files = useRef<Record<string, File>>({});     // plan id -> attached file (not persisted)
  const fileInput = useRef<HTMLInputElement>(null);
  const box = useRef<HTMLTextAreaElement>(null);
  const end = useRef<HTMLDivElement>(null);

  useEffect(() => {
    try {
      localStorage.setItem(STORE, JSON.stringify(thread.slice(-60)));
    } catch {
      /* storage blocked: the thread just won't survive a reload */
    }
    end.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [thread]);

  const push = (...m: Msg[]) => setThread((t) => [...t, ...m]);
  const note = (t: string, tone?: "error") => push({ id: uid(), kind: "note", text: t, tone });
  const lastRun = [...thread].reverse().find((m): m is Extract<Msg, { kind: "run" }> => m.kind === "run");

  const defaults = () => {
    let saved = {};
    try {
      saved = JSON.parse(localStorage.getItem("cf.options") || "{}");
    } catch {
      /* ignore */
    }
    return { ...(presets?.default_options ?? {}), ...saved };
  };

  const send = async () => {
    const t = text.trim();
    if ((!t && !file) || busy) return;
    const ask = t || "Turn this into Shorts with captions";
    push({ id: uid(), kind: "user", text: ask, file: file?.name });
    setText("");
    setBusy(true);
    const hasLink = /https?:\/\//i.test(ask);
    try {
      if (lastRun && !file && !hasLink) {
        // about the video we already have: "make 3", "only the funny ones", "shorter"
        const r = await api.agentFollowup(lastRun.pid, ask);
        if (r.action === "make") {
          note(r.ids?.length ? `Making ${r.ids.length} clip${r.ids.length === 1 ? "" : "s"}.` : "Those clips are already made.");
        } else if (r.request && r.plan) {
          push({ id: uid(), kind: "plan", request: r.request, plan: r.plan, state: "approved", info: true },
            { id: uid(), kind: "run", pid: lastRun.pid });
        }
      } else {
        const r = await api.agentPlan(ask, !!file);
        const id = uid();
        if (file) files.current[id] = file;
        push({ id, kind: "plan", request: r.request, plan: r.plan, state: "pending" });
        setFile(null);
      }
    } catch (e) {
      note((e as Error).message, "error");
    } finally {
      setBusy(false);
    }
  };

  const approve = async (m: Extract<Msg, { kind: "plan" }>) => {
    setBusy(true);
    const fd = new FormData();
    const f = files.current[m.id];
    if (f) fd.append("file", f);
    fd.append("text", m.request.text);
    fd.append("request", JSON.stringify(m.request));
    fd.append("options", JSON.stringify(defaults()));
    try {
      const r = await upload("/api/agent/start", fd, setPct);
      setThread((t) => t.map((x) => (x.id === m.id ? { ...m, state: "approved" } : x)));
      push({ id: uid(), kind: "run", pid: r.id });
    } catch (e) {
      note((e as Error).message, "error");
    } finally {
      setBusy(false);
      setPct(0);
    }
  };

  const change = (m: Extract<Msg, { kind: "plan" }>) => {
    setThread((t) => t.map((x) => (x.id === m.id ? { ...m, state: "changed" } : x)));
    setText(m.request.text);
    if (files.current[m.id]) setFile(files.current[m.id]);
    box.current?.focus();
  };

  const latestRun: Record<string, string> = {};
  for (const m of thread) if (m.kind === "run") latestRun[m.pid] = m.id;

  return (
    <div className="page agent">
      <div className="agent-head">
        <h1>Tell ClipForge what you want</h1>
        <p className="muted">
          Paste a link or attach a video, and say it in your own words: how many, how long, which part, what kind of moment.
        </p>
        {thread.length > 0 && (
          <button className="link small" onClick={() => setThread([])}>Start a new chat</button>
        )}
      </div>

      <div className="agent-thread">
        {thread.length === 0 && (
          <div className="agent-empty">
            <Bot size={22} className="muted" />
            <div className="muted small">Try one of these, then add your link:</div>
            <div className="agent-examples">
              {EXAMPLES.map((e) => (
                <button key={e} className="chip-btn" onClick={() => { setText(e); box.current?.focus(); }}>{e.replace(/: $/, "")}</button>
              ))}
            </div>
          </div>
        )}
        {thread.map((m) => {
          if (m.kind === "user")
            return (
              <div key={m.id} className="msg user">
                <div className="bubble">
                  {m.text}
                  {m.file && <div className="file-pill"><Film size={13} /> {m.file}</div>}
                </div>
              </div>
            );
          if (m.kind === "note")
            return (
              <div key={m.id} className={`msg bot ${m.tone === "error" ? "is-error" : ""}`}>
                <div className="bubble">{m.tone === "error" && <AlertTriangle size={14} />} {m.text}</div>
              </div>
            );
          if (m.kind === "plan")
            return (
              <PlanCard key={m.id} m={m} hasFile={!!files.current[m.id]} busy={busy} pct={pct}
                onApprove={() => approve(m)} onChange={() => change(m)} />
            );
          return <RunCard key={m.id} pid={m.pid} active={latestRun[m.pid] === m.id} />;
        })}
        <div ref={end} />
      </div>

      <div className="agent-composer card">
        {file && (
          <div className="file-pill">
            <Film size={13} /> {file.name} <span className="muted">({(file.size / 1048576).toFixed(1)} MB)</span>
            <button className="icon-btn" title="Remove" onClick={() => setFile(null)}><X size={13} /></button>
          </div>
        )}
        <div className="composer-row">
          <button className="icon-btn" title="Attach a video" onClick={() => fileInput.current?.click()}>
            <Paperclip size={18} />
          </button>
          <input ref={fileInput} type="file" accept="video/*" hidden onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) setFile(f);
            e.target.value = "";
          }} />
          <textarea
            ref={box}
            rows={1}
            value={text}
            placeholder={lastRun ? "Make 3, make all, or change what to look for…" : "Clip this podcast into 30 second clips: https://…"}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
          />
          <button className="btn primary send" onClick={send} disabled={busy || (!text.trim() && !file)} title="Send">
            {busy ? <Loader2 size={16} className="spin" /> : <ArrowUp size={16} />}
          </button>
        </div>
      </div>
    </div>
  );
}

function PlanCard({ m, hasFile, busy, pct, onApprove, onChange }: {
  m: Extract<Msg, { kind: "plan" }>; hasFile: boolean; busy: boolean; pct: number;
  onApprove: () => void; onChange: () => void;
}) {
  const needsSource = !m.request.url && !hasFile;
  const lost = m.state === "pending" && !m.request.url && !hasFile && m.plan.ready;   // file gone after reload
  return (
    <div className="msg bot">
      <div className="bubble plan-card">
        <div className="plan-head"><ListChecks size={16} /> {m.info ? "Finding again" : "Plan"}</div>
        <p>{m.plan.summary}</p>
        {m.request.focus_by === "ai" && <div className="muted small">The AI read "{m.request.focus}" as the kind of moment you want.</div>}
        <ol className="plan-steps">
          {m.plan.steps.map((s) => (
            <li key={s.title}><b>{s.title}</b><span className="muted small">{s.detail}</span></li>
          ))}
        </ol>
        {(needsSource || lost) && m.state === "pending" && (
          <div className="plan-q">{lost ? "Attach the video again, then send your message." : m.plan.questions[0] ?? "Send the video: paste a link or attach a file."}</div>
        )}
        {m.state === "pending" && !m.info && (
          <div className="row gap8">
            <button className="btn primary" disabled={busy || needsSource} onClick={onApprove}>
              {busy ? <Loader2 size={15} className="spin" /> : <Check size={15} />}
              {busy && pct > 0 && pct < 1 ? `Uploading ${Math.round(pct * 100)}%` : "Looks good, start"}
            </button>
            <button className="btn ghost" disabled={busy} onClick={onChange}><Pencil size={14} /> Change</button>
          </div>
        )}
        {m.state === "approved" && !m.info && <div className="ok-text small"><Check size={14} /> Started</div>}
        {m.state === "changed" && <div className="muted small">Changed - see your new message.</div>}
      </div>
    </div>
  );
}

const STAGE: Record<string, string> = {
  queued: "Waiting for the GPU", download: "Getting the video", probe: "Reading the video", audio: "Getting the audio",
  transcribe: "Transcribing", analyse: "Finding the moments", render: "Making your clips",
};

function RunCard({ pid, active }: { pid: string; active: boolean }) {
  const [p, setP] = useState<Project | null>(null);
  const [err, setErr] = useState("");
  const [picked, setPicked] = useState<Set<string> | null>(null);
  const [sending, setSending] = useState(false);

  const load = useCallback(() => api.project(pid, false).then((x) => { setP(x); setErr(""); }).catch((e) => setErr(e.message)), [pid]);
  useEffect(() => { load(); }, [load]);
  const working = !!p && (p.status === "running" || p.status === "queued"
    || p.clips.some((c) => c.render?.status === "queued" || c.render?.status === "rendering"));
  useInterval(load, active && (working || !p) ? 1200 : null);

  const cands = (p?.clips ?? []).filter((c) => c.render?.status === "candidate")
    .sort((a, b) => (b.score?.total ?? 0) - (a.score?.total ?? 0));
  const made = (p?.clips ?? []).filter((c) => c.render?.status !== "candidate")
    .sort((a, b) => (b.score?.total ?? 0) - (a.score?.total ?? 0));

  // the best few are pre-selected: the count the user asked for, else 3
  useEffect(() => {
    if (!p || picked !== null || !cands.length) return;
    const n = Math.max(1, Math.min(p.options?.clips || 3, cands.length));
    setPicked(new Set(cands.slice(0, n).map((c) => c.id)));
  }, [p, picked, cands]);

  if (err) return <div className="msg bot is-error"><div className="bubble"><AlertTriangle size={14} /> {err}</div></div>;
  if (!p) return <div className="msg bot"><div className="bubble"><Loader2 size={14} className="spin" /></div></div>;
  if (!active)
    return <div className="msg bot"><div className="bubble muted small">Earlier results for {p.name} were replaced by the search below.</div></div>;

  const make = async (ids: string[] | null) => {
    setSending(true);
    try {
      await api.finish(pid, ids);
      setPicked(new Set());
      await load();
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setSending(false);
    }
  };
  const sel = picked ?? new Set<string>();
  const toggle = (id: string) => {
    const s = new Set(sel);
    if (s.has(id)) s.delete(id);
    else s.add(id);
    setPicked(s);
  };
  const finding = p.status === "running" || p.status === "queued";

  return (
    <div className="msg bot wide">
      <div className="bubble run-card">
        <div className="run-head">
          <b className="grow" title={p.name}>{p.name}</b>
          <button className="link small" onClick={() => go(`/p/${pid}`)}>Open project</button>
        </div>

        {finding && (
          <div className="run-progress">
            <div className="row gap8 small"><Loader2 size={14} className="spin" /> {STAGE[p.stage] ?? p.stage} <span className="muted">· {p.message}</span></div>
            <div className="bar"><div style={{ width: `${Math.round(p.progress * 100)}%` }} /></div>
          </div>
        )}
        {p.status === "error" && <div className="err-text small"><AlertTriangle size={14} /> {p.error || p.message}</div>}

        {made.length > 0 && (
          <>
            <div className="run-sub">{made.filter((c) => c.file).length} made</div>
            <div className="made-grid">
              {made.map((c) => <MadeClip key={c.id} pid={pid} c={c} />)}
            </div>
          </>
        )}

        {cands.length > 0 && !finding && (
          <>
            <div className="run-sub">
              {made.length ? `${cands.length} more found. Pick any to make:` : `Found ${cands.length} clip${cands.length === 1 ? "" : "s"}. Pick the ones to make:`}
            </div>
            <div className="pick-list">
              {cands.map((c) => <Candidate key={c.id} p={p} c={c} on={sel.has(c.id)} onToggle={() => toggle(c.id)} />)}
            </div>
            <div className="row gap8 pick-actions">
              <button className="btn primary" disabled={sending || sel.size === 0} onClick={() => make([...sel])}>
                {sending ? <Loader2 size={15} className="spin" /> : <Check size={15} />} Make {sel.size} clip{sel.size === 1 ? "" : "s"}
              </button>
              {cands.length > 1 && (
                <button className="btn ghost" disabled={sending} onClick={() => make(cands.map((c) => c.id))}>Make all {cands.length}</button>
              )}
            </div>
          </>
        )}

        {p.status === "done" && made.some((c) => c.file) && (
          <div className="row gap8">
            <a className="btn ghost" href={`/api/projects/${pid}/zip`}><Download size={15} /> Download all</a>
            <span className="muted small">Say "only the funny ones", "shorter", or "make all" to keep going.</span>
          </div>
        )}
        {!finding && !cands.length && !made.length && p.status !== "error" && (
          <div className="muted small">No clips matched. Try asking for something broader.</div>
        )}
      </div>
    </div>
  );
}

function Candidate({ p, c, on, onToggle }: { p: Project; c: Clip; on: boolean; onToggle: () => void }) {
  const preview = p.preview ? `${media(p.id, p.preview)}#t=${c.start.toFixed(2)},${c.end.toFixed(2)}` : "";
  const why = c.score?.reasons?.[0];
  const risk = c.score?.risks?.[0];
  return (
    <div className={`pick ${on ? "on" : ""}`}>
      <label className="pick-check" title={on ? "Picked" : "Pick"}>
        <input type="checkbox" checked={on} onChange={onToggle} />
        <span>{on ? <Check size={14} /> : null}</span>
      </label>
      <div className="pick-media">
        {preview ? <video src={preview} preload="metadata" controls muted playsInline /> : <Film size={22} className="muted" />}
      </div>
      <div className="pick-body" onClick={onToggle}>
        <div className="row gap8">
          <ScoreBadge score={c.score} />
          <span className="muted small mono">{fmtTime(c.start)}–{fmtTime(c.end)} · {Math.round(c.end - c.start)}s</span>
        </div>
        <div className="pick-title">
          {c.kind === "long" && <span className="kind-badge inline">Long · {Math.round((c.end - c.start) / 60)} min</span>}
          {c.package?.yt_title || c.title}
        </div>
        {c.package?.thumbnail_text && <div className="thumb-text">Thumbnail: {c.package.thumbnail_text}</div>}
        {(c.summary || c.hook) && <div className="muted small pick-text">{c.summary || c.hook}</div>}
        <div className="signals">
          {why && <span className="signal good">+ {why}</span>}
          {risk && <span className="signal bad">− {risk}</span>}
        </div>
      </div>
    </div>
  );
}

function MadeClip({ pid, c }: { pid: string; c: Clip }) {
  const r = c.render || {};
  const src = media(pid, c.file, c.version);
  return (
    <div className={`made ${c.kind === "long" ? "is-long" : ""}`}>
      <div className="clip-player">
        {c.file ? (
          <video key={src} src={src} poster={media(pid, c.thumb, c.version)} controls preload="none" playsInline />
        ) : (
          <div className="clip-ph">
            {r.status === "error" ? <span className="err-text small"><AlertTriangle size={14} /> failed</span>
              : <span className="small muted"><Loader2 size={16} className="spin" /> {r.message || (r.status === "queued" ? "Queued" : `${Math.round((r.progress ?? 0) * 100)}%`)}</span>}
          </div>
        )}
      </div>
      <div className="made-title" title={c.package?.yt_title || c.title}>{c.package?.yt_title || c.title}</div>
      {c.package?.thumbnail_text && <div className="thumb-text">{c.package.thumbnail_text}</div>}
      <div className="row gap8">
        <button className="btn ghost small-btn" onClick={() => go(`/p/${pid}/edit/${c.id}`)}><Pencil size={13} /> Edit</button>
        {c.file && <a className="btn ghost small-btn" href={src} download={`${slug(c.title)}.mp4`}><Download size={13} /></a>}
      </div>
    </div>
  );
}
