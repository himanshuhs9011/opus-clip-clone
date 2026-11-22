import { useCallback, useEffect, useState } from "react";
import {
  AlertTriangle, ArrowLeft, Check, Copy, Download, Loader2, Pencil, RefreshCw, Scissors, Sparkles, Trash2,
} from "lucide-react";
import { api, media, type Clip, type Health, type Options, type Presets, type Project as P } from "../api";
import { OptionsForm } from "../components/OptionsForm";
import { PackageBox } from "../components/PackageBox";
import { ScoreBadge, ScoreRing, SubScores } from "../components/Score";
import { Modal } from "../components/SettingsModal";
import { fmtDuration, fmtTime, go, slug, useInterval } from "../lib/util";

const STEPS: [string, string][] = [
  ["download", "Download"],
  ["transcribe", "Transcribe"],
  ["analyse", "Find highlights"],
  ["render", "Reframe & render"],
  ["done", "Done"],
];
const ORDER = ["queued", "download", "probe", "audio", "transcribe", "analyse", "render", "done"];

export function ProjectPage({ id, presets, health }: { id: string; presets: Presets | null; health: Health | null }) {
  const [p, setP] = useState<P | null>(null);
  const [err, setErr] = useState("");
  const [sort, setSort] = useState<"score" | "time">("score");
  const [rerun, setRerun] = useState<Options | null>(null);

  const load = useCallback(
    () =>
      api
        .project(id, false)
        .then((x) => {
          setP(x);
          setErr("");
        })
        .catch((e) => setErr(e.message)),
    [id],
  );
  useEffect(() => {
    setP(null);
    load();
  }, [load]);

  const busy = !!p && (p.status === "running" || p.status === "queued");
  const rendering = !!p?.clips.some((c) => c.render?.status === "queued" || c.render?.status === "rendering");
  useInterval(load, busy || rendering ? 1200 : null);

  if (err) return <div className="page"><div className="err-text">{err}</div></div>;
  if (!p) return <div className="page center muted"><Loader2 className="spin" /></div>;

  const clips = [...p.clips].sort((a, b) =>
    sort === "score" ? (b.score?.total ?? 0) - (a.score?.total ?? 0) : a.start - b.start,
  );
  const stageIdx = ORDER.indexOf(p.stage);
  const cands = clips.filter((c) => c.render?.status === "candidate");
  const selector = p.settings?.selector;

  return (
    <div className="page project">
      <div className="proj-head">
        <button className="icon-btn" onClick={() => go("/")} title="Back">
          <ArrowLeft size={18} />
        </button>
        <div className="grow">
          <h1 className="proj-title">{p.name}</h1>
          <div className="muted small meta-row">
            <span>{fmtDuration(p.duration)}</span>
            {p.width > 0 && <span>{p.width}×{p.height}</span>}
            {p.language && <span>{p.language.toUpperCase()}</span>}
            {selector && (
              <span>
                <Sparkles size={12} /> {selector}
              </span>
            )}
            {p.settings?.encoder && <span>{p.settings.encoder}</span>}
            {p.timings?.total && <span>processed in {Math.round(p.timings.total)}s</span>}
          </div>
        </div>
        <div className="head-actions">
          {cands.length > 0 && !busy && (
            <button className="btn primary" onClick={async () => { await api.finish(p.id, cands.map((c) => c.id)); load(); }}>
              <Scissors size={15} /> Make {cands.length} found clip{cands.length === 1 ? "" : "s"}
            </button>
          )}
          {(p.status === "done" || p.status === "picking") && clips.some((c) => c.file) && (
            <a className="btn ghost" href={`/api/projects/${p.id}/zip`}>
              <Download size={15} /> All clips
            </a>
          )}
          <button
            className="btn ghost"
            disabled={busy}
            onClick={() => presets && setRerun({ ...presets.default_options, ...p.options })}
          >
            <RefreshCw size={15} /> Re-run
          </button>
          <button
            className="icon-btn danger"
            title="Delete project"
            disabled={busy}
            onClick={async () => {
              if (!confirm("Delete this project?")) return;
              await api.deleteProject(p.id);
              go("/");
            }}
          >
            <Trash2 size={16} />
          </button>
        </div>
      </div>

      {(busy || p.status === "error") && (
        <div className={`card progress-card ${p.status === "error" ? "is-error" : ""}`}>
          <div className="steps">
            {STEPS.map(([k, label]) => {
              const i = ORDER.indexOf(k);
              const state = p.status === "error" ? "" : stageIdx > i ? "done" : stageIdx === i || (k === "transcribe" && ["probe", "audio"].includes(p.stage)) ? "now" : "";
              return (
                <div key={k} className={`step ${state}`}>
                  <span className="step-dot">{state === "done" ? <Check size={12} /> : null}</span>
                  {label}
                </div>
              );
            })}
          </div>
          {p.status !== "error" && (
            <div className="bar">
              <div style={{ width: `${Math.round(p.progress * 100)}%` }} />
            </div>
          )}
          <div className={p.status === "error" ? "err-text" : "muted small"}>
            {p.status === "error" ? <AlertTriangle size={14} /> : <Loader2 size={13} className="spin" />} {p.error || p.message}
          </div>
        </div>
      )}

      {clips.length > 0 && (
        <div className="section-head">
          <h2>
            {clips.length} clips <span className="muted">ranked by virality score</span>
            {p.settings?.quality && p.settings.quality.bar > 0 && p.settings.quality.kept < p.settings.quality.asked && (
              <span className="muted small quality-note">
                {" "}· {p.settings.quality.asked - p.settings.quality.kept} more didn't reach {p.settings.quality.bar}+
              </span>
            )}
          </h2>
          <div className="seg small">
            <button className={sort === "score" ? "sel" : ""} onClick={() => setSort("score")}>Top score</button>
            <button className={sort === "time" ? "sel" : ""} onClick={() => setSort("time")}>Timeline</button>
          </div>
        </div>
      )}

      <div className="clip-list">
        {clips.map((c, i) => (
          <ClipRow key={c.id} p={p} c={c} position={sort === "score" ? i + 1 : c.rank} onChange={load} />
        ))}
      </div>

      {rerun && (
        <Modal onClose={() => setRerun(null)} wide>
          <h2>Re-run with new settings</h2>
          <p className="muted small">The transcript is reused when the Whisper model and language stay the same, so this is quick.</p>
          <OptionsForm value={rerun} onChange={setRerun} presets={presets} health={health} compact />
          <div className="modal-actions">
            <button className="btn ghost" onClick={() => setRerun(null)}>Cancel</button>
            <button
              className="btn primary"
              onClick={async () => {
                await api.rerun(p.id, rerun);
                setRerun(null);
                load();
              }}
            >
              <RefreshCw size={15} /> Re-run
            </button>
          </div>
        </Modal>
      )}
    </div>
  );
}

function ClipRow({ p, c, position, onChange }: { p: P; c: Clip; position: number; onChange: () => void }) {
  const [copied, setCopied] = useState(false);
  const r = c.render || {};
  const working = r.status === "queued" || r.status === "rendering";
  const src = media(p.id, c.file, c.version);
  const sc = c.score;

  const copy = async () => {
    await navigator.clipboard.writeText(`${c.title}\n\n${c.hashtags.join(" ")}`);
    setCopied(true);
    setTimeout(() => setCopied(false), 1400);
  };

  return (
    <div className={`clip-row card ${c.kind === "long" ? "is-long" : ""}`}>
      <div className="clip-player">
        {c.file ? (
          <video key={src} src={src} poster={media(p.id, c.thumb, c.version)} controls preload="none" playsInline />
        ) : (
          <div className="clip-ph">
            {r.status === "error" ? (
              <span className="err-text"><AlertTriangle size={16} /> render failed</span>
            ) : r.status === "candidate" ? (
              <button
                className="btn primary"
                onClick={async () => {
                  await api.finish(p.id, [c.id]);
                  onChange();
                }}
              >
                <Scissors size={15} /> Make this clip
              </button>
            ) : (
              <Loader2 className="spin" />
            )}
          </div>
        )}
        {working && (
          <div className="clip-rendering">
            <Loader2 size={14} className="spin" /> {r.message || (r.status === "queued" ? "Queued" : `Rendering ${Math.round((r.progress ?? 0) * 100)}%`)}
          </div>
        )}
        <span className="rank">#{position}</span>
        {c.kind === "long" && <span className="kind-badge">Long · {Math.round(c.duration / 60)} min</span>}
      </div>

      <div className="clip-info">
        <div className="clip-top">
          {sc && <ScoreRing value={sc.total} size={68} />}
          <div className="grow">
            <div className="row gap8">
              <ScoreBadge score={sc} />
              <span className="muted small">
                {fmtTime(c.start)}–{fmtTime(c.end)} · {(r.duration ?? c.duration).toFixed(0)}s
              </span>
            </div>
            <h3 className="clip-title">{c.title}</h3>
          </div>
        </div>

        {sc && <SubScores score={sc} />}

        {(c.why || c.summary) && <p className="clip-why">{c.why || c.summary}</p>}

        <PackageBox c={c} />

        {sc && (sc.reasons.length > 0 || sc.risks.length > 0) && (
          <div className="signals">
            {sc.reasons.map((x) => (
              <span key={x} className="signal good">+ {x}</span>
            ))}
            {sc.risks.map((x) => (
              <span key={x} className="signal bad">− {x}</span>
            ))}
          </div>
        )}

        {c.hashtags.length > 0 && (
          <div className="tags">
            {c.hashtags.map((h) => (
              <span key={h} className="tag">{h}</span>
            ))}
          </div>
        )}

        <div className="clip-actions">
          <button className="btn primary" onClick={() => go(`/p/${p.id}/edit/${c.id}`)}>
            <Pencil size={15} /> Edit
          </button>
          {c.file && (
            <a className="btn ghost" href={src} download={`${String(position).padStart(2, "0")}_${slug(c.title)}.mp4`}>
              <Download size={15} /> Download
            </a>
          )}
          <button className="btn ghost" onClick={copy}>
            {copied ? <Check size={15} /> : <Copy size={15} />} {copied ? "Copied" : "Copy title + tags"}
          </button>
          {r.status === "error" && (
            <button
              className="btn ghost"
              onClick={async () => {
                await api.renderClip(p.id, c.id);
                onChange();
              }}
            >
              <Scissors size={15} /> Retry render
            </button>
          )}
          {r.seconds !== undefined && r.status === "done" && (
            <span className="muted small render-meta">
              rendered in {r.seconds}s · {r.encoder}
            </span>
          )}
        </div>
        {r.status === "error" && r.error && <div className="err-text small mono">{r.error.slice(0, 300)}</div>}
      </div>
    </div>
  );
}
