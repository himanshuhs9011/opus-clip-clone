import { useEffect, useRef, useState } from "react";
import { Film, Link2, Loader2, Trash2, Upload, Wand2 } from "lucide-react";
import { api, media, type Health, type Options, type Presets, type ProjectSummary } from "../api";
import { OptionsForm } from "../components/OptionsForm";
import { fmtDuration, go, scoreTone, timeAgo, useInterval } from "../lib/util";

const STAGE: Record<string, string> = {
  queued: "Queued", download: "Downloading", probe: "Reading", audio: "Audio", transcribe: "Transcribing",
  analyse: "Finding highlights", render: "Rendering", done: "Done", error: "Failed",
};

export function Home({ presets, health }: { presets: Presets | null; health: Health | null }) {
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [opts, setOpts] = useState<Options | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState("");
  const [drag, setDrag] = useState(false);
  const [busy, setBusy] = useState(false);
  const [uploadPct, setUploadPct] = useState(0);
  const [err, setErr] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const load = () => api.projects().then(setProjects).catch(() => {});
  useEffect(() => {
    load();
  }, []);
  useInterval(load, projects.some((p) => p.status === "running" || p.status === "queued") ? 2000 : null);

  useEffect(() => {
    if (presets && !opts) {
      let saved: Partial<Options> = {};
      try {
        saved = JSON.parse(localStorage.getItem("cf.options") || "{}");
      } catch {
        /* ignore */
      }
      setOpts({ ...presets.default_options, ...saved, topic: "" });
    }
  }, [presets, opts]);

  const updateOpts = (o: Options) => {
    setOpts(o);
    try {
      const { topic: _t, ...rest } = o;
      void _t;
      localStorage.setItem("cf.options", JSON.stringify(rest));
    } catch {
      /* ignore */
    }
  };

  const submit = async () => {
    setErr("");
    if (!file && !url.trim()) {
      setErr("Drop a video or paste a link first.");
      return;
    }
    if (!opts) return;
    setBusy(true);
    const fd = new FormData();
    if (file) fd.append("file", file);
    fd.append("url", url.trim());
    fd.append("options", JSON.stringify(opts));
    try {
      // XHR for upload progress on big files
      const id = await new Promise<string>((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", "/api/projects");
        xhr.upload.onprogress = (e) => e.lengthComputable && setUploadPct(e.loaded / e.total);
        xhr.onload = () => {
          try {
            const j = JSON.parse(xhr.responseText);
            if (xhr.status >= 300) reject(new Error(j.detail || `HTTP ${xhr.status}`));
            else resolve(j.id);
          } catch {
            reject(new Error(`HTTP ${xhr.status}`));
          }
        };
        xhr.onerror = () => reject(new Error("network error"));
        xhr.send(fd);
      });
      go(`/p/${id}`);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
      setUploadPct(0);
    }
  };

  const remove = async (id: string) => {
    if (!confirm("Delete this project and all its clips?")) return;
    try {
      await api.deleteProject(id);
      load();
    } catch (e) {
      alert((e as Error).message);
    }
  };

  return (
    <div className="page home">
      <section className="hero">
        <h1>
          Long video in. <span className="accent-word">Viral shorts out.</span>
        </h1>
        <p className="lead">
          Your best moments, ranked by virality, reframed to vertical and captioned on your own GPU.
        </p>

        <div
          className={`drop ${drag ? "drag" : ""} ${file ? "has-file" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDrag(true);
          }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDrag(false);
            const f = e.dataTransfer.files?.[0];
            if (f) {
              setFile(f);
              setUrl("");
            }
          }}
        >
          <div className="drop-row">
            <Link2 size={18} className="muted" />
            <input
              className="link-input"
              placeholder="Paste a YouTube link…"
              value={url}
              onChange={(e) => {
                setUrl(e.target.value);
                if (e.target.value) setFile(null);
              }}
              onKeyDown={(e) => e.key === "Enter" && submit()}
            />
            <span className="muted small">or</span>
            <button className="btn ghost" type="button" onClick={() => inputRef.current?.click()}>
              <Upload size={15} /> Upload
            </button>
            <input
              ref={inputRef}
              type="file"
              accept="video/*"
              hidden
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) {
                  setFile(f);
                  setUrl("");
                }
              }}
            />
          </div>
          {file && (
            <div className="file-pill">
              <Film size={14} /> {file.name} <span className="muted">({(file.size / 1048576).toFixed(1)} MB)</span>
              <button className="link" onClick={() => setFile(null)}>remove</button>
            </div>
          )}
          {!file && !url && <div className="drop-hint muted small">…or drop a video file anywhere in this box</div>}
        </div>

        {opts && <OptionsForm value={opts} onChange={updateOpts} presets={presets} health={health} />}

        <button className="btn primary big" onClick={submit} disabled={busy}>
          {busy ? <Loader2 className="spin" size={18} /> : <Wand2 size={18} />}
          {busy ? (uploadPct > 0 && uploadPct < 1 ? `Uploading ${Math.round(uploadPct * 100)}%` : "Starting…") : "Get clips"}
        </button>
        {err && <div className="err-text center">{err}</div>}
      </section>

      <section>
        <div className="section-head">
          <h2>Projects</h2>
          <span className="muted small">{projects.length} total</span>
        </div>
        {!projects.length && <div className="empty muted">No projects yet. Your first one will show up here.</div>}
        <div className="proj-grid">
          {projects.map((p) => (
            <div key={p.id} className="proj-card" onClick={() => go(`/p/${p.id}`)}>
              <div className="proj-thumb">
                {p.thumb ? <img src={media(p.id, p.thumb)} alt="" loading="lazy" /> : <Film size={28} className="muted" />}
                {p.best > 0 && <span className={`proj-best tone-${scoreTone(p.best)}`}>{p.best}</span>}
                {(p.status === "running" || p.status === "queued") && (
                  <div className="proj-progress">
                    <div style={{ width: `${Math.round(p.progress * 100)}%` }} />
                  </div>
                )}
              </div>
              <div className="proj-body">
                <div className="proj-name" title={p.name}>{p.name}</div>
                <div className="muted small">
                  {p.status === "running" || p.status === "queued"
                    ? `${STAGE[p.stage] ?? p.stage} · ${Math.round(p.progress * 100)}%`
                    : p.status === "error"
                      ? "Failed"
                      : `${p.clips} clips`}{" "}
                  · {fmtDuration(p.duration)} · {timeAgo(p.created)}
                </div>
              </div>
              <button
                className="icon-btn proj-del"
                title="Delete"
                onClick={(e) => {
                  e.stopPropagation();
                  remove(p.id);
                }}
              >
                <Trash2 size={15} />
              </button>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
