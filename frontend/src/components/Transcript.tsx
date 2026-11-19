import { useEffect, useMemo, useRef, useState } from "react";
import { CornerDownLeft, CornerDownRight, Pencil, RotateCcw, Scissors } from "lucide-react";
import type { Edit, Word } from "../api";
import { fmtTime } from "../lib/util";

const msKey = (t: number) => String(Math.round(t * 1000));

interface Props {
  words: Word[];               // absolute-time words around the clip
  edit: Edit;
  a: number;
  b: number;
  playhead: number;            // absolute source time
  onEdit: (e: Edit) => void;
  onSeek: (abs: number) => void;
  speakerColors?: string[] | null;
  speakers?: number;
}

/** Text-based editing: click a word to seek, shift-click to select a range, cut or fix words. */
export function Transcript({ words, edit, a, b, playhead, onEdit, onSeek, speakerColors, speakers = 1 }: Props) {
  const [sel, setSel] = useState<[number, number] | null>(null);
  const [editing, setEditing] = useState<number | null>(null);
  const [draft, setDraft] = useState("");
  const boxRef = useRef<HTMLDivElement>(null);
  const removed = useMemo(() => new Set(edit.cuts.removed.map(String)), [edit.cuts.removed]);

  const activeIdx = useMemo(() => {
    let idx = -1;
    for (let i = 0; i < words.length; i++) if (words[i].s <= playhead + 0.02) idx = i;
    return idx;
  }, [words, playhead]);

  // keep the spoken word in view
  useEffect(() => {
    const el = boxRef.current?.querySelector<HTMLElement>(`[data-i="${activeIdx}"]`);
    const box = boxRef.current;
    if (!el || !box) return;
    const top = el.offsetTop - box.offsetTop;
    if (top < box.scrollTop + 20 || top > box.scrollTop + box.clientHeight - 60) {
      box.scrollTo({ top: top - box.clientHeight / 3, behavior: "smooth" });
    }
  }, [activeIdx]);

  const range = sel ? [Math.min(sel[0], sel[1]), Math.max(sel[0], sel[1])] : null;
  const selWords = range ? words.slice(range[0], range[1] + 1) : [];
  const allRemoved = selWords.length > 0 && selWords.every((w) => removed.has(msKey(w.s)));

  const toggleCut = () => {
    if (!selWords.length) return;
    const next = new Set(removed);
    for (const w of selWords) {
      if (allRemoved) next.delete(msKey(w.s));
      else next.add(msKey(w.s));
    }
    onEdit({ ...edit, cuts: { ...edit.cuts, removed: [...next] } });
  };

  const setStart = () => range && onEdit({ ...edit, start: Math.max(0, words[range[0]].s - 0.05) });
  const setEnd = () => range && onEdit({ ...edit, end: words[range[1]].e + 0.12 });

  const commitText = () => {
    if (editing === null) return;
    const w = words[editing];
    const edits = { ...edit.word_edits };
    if (draft.trim() === w.w || !draft.trim()) delete edits[msKey(w.s)];
    else edits[msKey(w.s)] = draft.trim();
    onEdit({ ...edit, word_edits: edits });
    setEditing(null);
  };

  const nRemoved = edit.cuts.removed.length;

  return (
    <div className="transcript">
      <div className="tr-head">
        <b>Transcript</b>
        <div className="row gap6">
          <label className={`toggle sm ${edit.cuts.silence ? "on" : ""}`}>
            <input type="checkbox" checked={edit.cuts.silence} onChange={(e) => onEdit({ ...edit, cuts: { ...edit.cuts, silence: e.target.checked } })} />
            Silences
          </label>
          <label className={`toggle sm ${edit.cuts.fillers ? "on" : ""}`}>
            <input type="checkbox" checked={edit.cuts.fillers} onChange={(e) => onEdit({ ...edit, cuts: { ...edit.cuts, fillers: e.target.checked } })} />
            Um / uh
          </label>
        </div>
      </div>

      <div className={`tr-tools ${range ? "" : "idle"}`}>
        {range ? (
          <>
            <span className="muted small">
              {selWords.length} word{selWords.length > 1 ? "s" : ""} · {fmtTime(selWords[0].s, true)}
            </span>
            <button className="chip-btn" onClick={setStart} title="Clip starts at the first selected word">
              <CornerDownRight size={13} /> Start
            </button>
            <button className="chip-btn" onClick={setEnd} title="Clip ends after the last selected word">
              <CornerDownLeft size={13} /> End
            </button>
            <button className={`chip-btn ${allRemoved ? "" : "danger"}`} onClick={toggleCut}>
              {allRemoved ? <RotateCcw size={13} /> : <Scissors size={13} />} {allRemoved ? "Restore" : "Cut"}
            </button>
            {selWords.length === 1 && (
              <button
                className="chip-btn"
                onClick={() => {
                  setEditing(range[0]);
                  setDraft(edit.word_edits[msKey(words[range[0]].s)] ?? words[range[0]].w);
                }}
              >
                <Pencil size={13} /> Fix text
              </button>
            )}
          </>
        ) : (
          <span className="muted small">Click a word to jump · shift-click to select a range, then cut or trim</span>
        )}
      </div>

      <div className="tr-body" ref={boxRef}>
        {words.map((w, i) => {
          const inClip = w.e > a && w.s < b;
          const isSel = range && i >= range[0] && i <= range[1];
          const text = edit.word_edits[msKey(w.s)] ?? w.w;
          const cls = [
            "tw",
            inClip ? "in" : "out",
            removed.has(msKey(w.s)) ? "cut" : "",
            isSel ? "sel" : "",
            i === activeIdx ? "now" : "",
            edit.word_edits[msKey(w.s)] ? "fixed" : "",
          ].join(" ");
          if (editing === i) {
            return (
              <input
                key={i}
                className="tw-input"
                autoFocus
                value={draft}
                size={Math.max(3, draft.length)}
                onChange={(e) => setDraft(e.target.value)}
                onBlur={commitText}
                onKeyDown={(e) => {
                  if (e.key === "Enter") commitText();
                  if (e.key === "Escape") setEditing(null);
                }}
              />
            );
          }
          const sp = w.sp ?? -1;
          const turn = speakers >= 2 && sp >= 0 && (i === 0 || (words[i - 1].sp ?? -1) !== sp);
          return (
            <span key={i}>
            {turn && (
              <span className="sp-chip" style={{ color: speakerColors?.[sp] ?? undefined }}>
                Speaker {sp + 1}
              </span>
            )}
            <span
              data-i={i}
              className={cls}
              onClick={(e) => {
                if (e.shiftKey && sel) setSel([sel[0], i]);
                else {
                  setSel([i, i]);
                  onSeek(w.s);
                }
              }}
              onDoubleClick={() => {
                setSel([i, i]);
                setEditing(i);
                setDraft(text);
              }}
            >
              {text}{" "}
            </span>
            </span>
          );
        })}
      </div>
      {nRemoved > 0 && (
        <div className="tr-foot muted small">
          {nRemoved} word{nRemoved > 1 ? "s" : ""} cut ·{" "}
          <button className="link" onClick={() => onEdit({ ...edit, cuts: { ...edit.cuts, removed: [] } })}>
            restore all
          </button>
        </div>
      )}
    </div>
  );
}
