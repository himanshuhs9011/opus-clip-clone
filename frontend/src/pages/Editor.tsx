import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ArrowLeft, CheckCircle2, Download, Loader2, Rocket, Undo2 } from "lucide-react";
import { api, media, type Edit, type Plan, type Presets, type Project, type Word } from "../api";
import { Inspector } from "../components/Inspector";
import { Preview, type PreviewHandle } from "../components/Preview";
import { ScoreBadge } from "../components/Score";
import { Timeline } from "../components/Timeline";
import { Transcript } from "../components/Transcript";
import { clone, go, slug, useDebounced, useInterval } from "../lib/util";

function fullEdit(presets: Presets, partial: Partial<Edit>): Edit {
  const base = clone(presets.default_edit);
  const out = { ...base, ...clone(partial) } as Edit;
  for (const k of ["captions", "title", "progress", "zoom", "cuts", "audio"] as const) {
    (out as unknown as Record<string, unknown>)[k] = { ...(base[k] as object), ...((partial[k] as object) ?? {}) };
  }
  out.word_edits = { ...(partial.word_edits ?? {}) };
  return out;
}

export function Editor({ pid, cid, presets }: { pid: string; cid: string; presets: Presets }) {
  const [project, setProject] = useState<Project | null>(null);
  const [edit, setEdit] = useState<Edit | null>(null);
  const [history, setHistory] = useState<Edit[]>([]);
  const [title, setTitle] = useState("");
  const [tags, setTags] = useState("");
  const [plan, setPlan] = useState<Plan | null>(null);
  const [planErr, setPlanErr] = useState("");
  const [saved, setSaved] = useState<"saved" | "saving" | "dirty">("saved");
  const [exporting, setExporting] = useState(false);
  const [playhead, setPlayhead] = useState(0);
  const previewRef = useRef<PreviewHandle>(null);
  const loaded = useRef(false);

  // ------------------------------------------------------------ load
  useEffect(() => {
    loaded.current = false;
    api.project(pid, true).then((p) => {
      setProject(p);
      const c = p.clips.find((x) => x.id === cid);
      if (!c) return;
      setEdit(fullEdit(presets, c.edit));
      setTitle(c.title);
      setTags(c.hashtags.join(" "));
      setPlayhead(c.start);
      setTimeout(() => (loaded.current = true), 0);
    });
  }, [pid, cid, presets]);

  const clip = project?.clips.find((c) => c.id === cid) ?? null;
  const rendering = clip?.render?.status === "queued" || clip?.render?.status === "rendering";

  useInterval(
    () => api.project(pid, false).then((p) => setProject((old) => (old ? { ...p, sentences: old.sentences } : p))),
    rendering || exporting ? 1000 : null,
  );
  useEffect(() => {
    if (exporting && clip && clip.render?.status && !rendering) setExporting(false);
  }, [exporting, clip, rendering]);

  // ------------------------------------------------------------ plan (live preview)
  const debouncedEdit = useDebounced(edit, 90);
  useEffect(() => {
    if (!debouncedEdit) return;
    const ac = new AbortController();
    api
      .plan(pid, cid, debouncedEdit, ac.signal)
      .then((p) => {
        setPlan(p);
        setPlanErr("");
      })
      .catch((e) => e.name !== "AbortError" && setPlanErr(e.message));
    return () => ac.abort();
  }, [debouncedEdit, pid, cid]);

  // ------------------------------------------------------------ autosave
  const saveKey = useDebounced(JSON.stringify([edit, title, tags]), 700);
  useEffect(() => {
    if (!loaded.current || !edit) return;
    setSaved("saving");
    api
      .updateClip(pid, cid, { edit, title, hashtags: tags.split(/[\s,]+/).filter(Boolean) })
      .then(() => setSaved("saved"))
      .catch(() => setSaved("dirty"));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [saveKey]);

  const change = useCallback((e: Edit) => {
    setEdit((old) => {
      if (old) setHistory((h) => [...h.slice(-49), old]);
      return e;
    });
    setSaved("dirty");
  }, []);

  const undo = () => {
    setHistory((h) => {
      if (!h.length) return h;
      setEdit(h[h.length - 1]);
      return h.slice(0, -1);
    });
  };

  // keyboard: space = play/pause, ctrl+z = undo
  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (e.code === "Space") {
        e.preventDefault();
        previewRef.current?.toggle();
      }
      if ((e.ctrlKey || e.metaKey) && e.key === "z") {
        e.preventDefault();
        undo();
      }
    };
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  });

  // words around the clip (for transcript + timeline)
  const view: [number, number] = useMemo(() => {
    if (!clip || !project) return [0, 1];
    const a = Math.min(clip.start, edit?.start ?? clip.start);
    const b = Math.max(clip.end, edit?.end ?? clip.end);
    return [Math.max(0, a - 20), Math.min(project.duration || b + 20, b + 20)];
  }, [clip, project, edit?.start, edit?.end]);

  const words: Word[] = useMemo(() => {
    if (!project?.sentences) return [];
    const out: Word[] = [];
    for (const s of project.sentences) {
      if (s.end < view[0] || s.start > view[1]) continue;
      for (const w of s.words) if (w.e > view[0] && w.s < view[1]) out.push(w);
    }
    return out;
  }, [project?.sentences, view]);

  if (!project || !edit || !clip) {
    return <div className="page center muted"><Loader2 className="spin" /></div>;
  }

  const exportClip = async () => {
    setExporting(true);
    try {
      const c = await api.updateClip(pid, cid, { edit, title, hashtags: tags.split(/[\s,]+/).filter(Boolean), render: true });
      setProject((p) => (p ? { ...p, clips: p.clips.map((x) => (x.id === cid ? c : x)) } : p));
      setSaved("saved");
    } catch (e) {
      alert((e as Error).message);
      setExporting(false);
    }
  };

  const r = clip.render || {};
  const fileUrl = media(pid, clip.file, clip.version);

  return (
    <div className="editor">
      <div className="ed-bar">
        <button className="icon-btn" onClick={() => go(`/p/${pid}`)} title="Back to clips">
          <ArrowLeft size={18} />
        </button>
        <div className="grow ed-title">
          <ScoreBadge score={clip.score} />
          <b title={title}>{title}</b>
          <span className={`save-state ${saved}`}>{saved === "saved" ? "Saved" : saved === "saving" ? "Saving…" : "Unsaved"}</span>
        </div>
        <button className="icon-btn" onClick={undo} disabled={!history.length} title="Undo (Ctrl+Z)">
          <Undo2 size={17} />
        </button>
        {clip.file && !rendering && (
          <a className="btn ghost" href={fileUrl} download={`${slug(title)}.mp4`}>
            <Download size={15} /> Download
          </a>
        )}
        <button className="btn primary" onClick={exportClip} disabled={rendering}>
          {rendering ? <Loader2 size={16} className="spin" /> : <Rocket size={16} />}
          {rendering
            ? r.status === "queued"
              ? "Queued…"
              : `${r.message ?? "Rendering"} ${Math.round((r.progress ?? 0) * 100)}%`
            : "Export"}
        </button>
      </div>

      <div className="ed-grid">
        <Transcript
          words={words}
          edit={edit}
          a={plan?.a ?? clip.start}
          b={plan?.b ?? clip.end}
          playhead={playhead}
          onEdit={change}
          onSeek={(t) => previewRef.current?.seekSource(t)}
          speakerColors={plan?.speaker_colors}
          speakers={project.speakers?.count ?? 1}
        />

        <div className="ed-center">
          {plan ? (
            <>
              <Preview ref={previewRef} plan={plan} onTime={(_, abs) => setPlayhead(abs)} />
              <Timeline
                plan={plan}
                words={words}
                view={view}
                playhead={playhead}
                onTrim={(s, e) => change({ ...edit, start: Math.round(s * 1000) / 1000, end: Math.round(e * 1000) / 1000 })}
                onSeek={(t) => previewRef.current?.seekSource(t)}
              />
            </>
          ) : (
            <div className="center muted">{planErr || <Loader2 className="spin" />}</div>
          )}
          {r.status === "done" && !rendering && r.at && (
            <div className="export-done">
              <CheckCircle2 size={16} /> Exported {r.duration?.toFixed(1)}s clip in {r.seconds}s ({r.encoder})
              <a href={fileUrl} target="_blank" rel="noreferrer">open</a>
            </div>
          )}
          {r.status === "error" && <div className="err-text small">{r.error}</div>}
        </div>

        <Inspector
          edit={edit}
          onEdit={change}
          presets={presets}
          clip={clip}
          title={title}
          onTitle={(t) => {
            setTitle(t);
            setSaved("dirty");
          }}
          hashtags={tags}
          onHashtags={(t) => {
            setTags(t);
            setSaved("dirty");
          }}
          thumb={media(pid, clip.thumb, clip.version)}
          speakerColors={plan?.speaker_colors ?? null}
          speakerCount={Math.max(plan?.voices.length ? Math.max(...plan.voices) + 1 : 0, 0)}
          onApplyAll={async () => {
            if (!confirm("Apply this clip's captions, effects and layout to every clip in the project and re-render them?")) return;
            await api.applyAll(pid, edit, true);
            go(`/p/${pid}`);
          }}
        />
      </div>
    </div>
  );
}
