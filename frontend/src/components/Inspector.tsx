import { useState } from "react";
import { Captions, Crop, Layers, Type, Wand2 } from "lucide-react";
import type { Clip, Edit, Presets } from "../api";
import { CSS_FILTERS, FILTER_LABELS } from "../lib/util";
import { PresetSwatch } from "./OptionsForm";

type Tab = "captions" | "effects" | "layout" | "text";

interface Props {
  edit: Edit;
  onEdit: (e: Edit) => void;
  presets: Presets;
  clip: Clip;
  title: string;
  onTitle: (t: string) => void;
  hashtags: string;
  onHashtags: (h: string) => void;
  thumb: string;
  onApplyAll: () => void;
  speakerColors: string[] | null;
  speakerCount: number;
}

export function Inspector({
  edit, onEdit, presets, clip, title, onTitle, hashtags, onHashtags, thumb, onApplyAll, speakerColors, speakerCount,
}: Props) {
  const [tab, setTab] = useState<Tab>("captions");
  const cap = edit.captions;
  const setCap = (patch: Partial<Edit["captions"]>) => onEdit({ ...edit, captions: { ...cap, ...patch } });
  const preset = presets.captions.find((p) => p.id === cap.preset) ?? presets.captions[0];

  return (
    <div className="inspector">
      <div className="tabs">
        {([
          ["captions", "Captions", Captions],
          ["effects", "Effects", Wand2],
          ["layout", "Layout", Crop],
          ["text", "Title", Type],
        ] as [Tab, string, typeof Captions][]).map(([id, label, Icon]) => (
          <button key={id} className={tab === id ? "sel" : ""} onClick={() => setTab(id)}>
            <Icon size={15} />
            {label}
          </button>
        ))}
      </div>

      <div className="insp-body">
        {tab === "captions" && (
          <>
            <Toggle label="Show captions" on={cap.enabled} set={(v) => setCap({ enabled: v })} />
            <div className="swatches small">
              {presets.captions.map((p) => (
                <PresetSwatch key={p.id} p={p} selected={cap.preset === p.id} onClick={() => setCap({ preset: p.id, enabled: true })} />
              ))}
            </div>
            <Slider label="Position" min={0.15} max={0.9} step={0.01} value={cap.position} set={(v) => setCap({ position: v })} fmt={(v) => `${Math.round(v * 100)}%`} />
            <Slider label="Size" min={0.6} max={1.6} step={0.05} value={cap.size} set={(v) => setCap({ size: v })} fmt={(v) => `${Math.round(v * 100)}%`} />
            <Slider
              label="Words per caption"
              min={1}
              max={7}
              step={1}
              value={cap.words ?? preset.words}
              set={(v) => setCap({ words: v })}
              fmt={(v) => (cap.words === null ? `${v} (auto)` : String(v))}
            />
            <div className="field">
              <span className="field-label">Case</span>
              <div className="seg small">
                {([
                  [null, "Template"],
                  [true, "UPPER"],
                  [false, "Normal"],
                ] as [boolean | null, string][]).map(([v, l]) => (
                  <button key={l} className={cap.uppercase === v ? "sel" : ""} onClick={() => setCap({ uppercase: v })}>
                    {l}
                  </button>
                ))}
              </div>
            </div>
            <Toggle label="Highlight keywords" sub="Numbers, names and strong words get a second colour" on={cap.keywords} set={(v) => setCap({ keywords: v })} />
            <Toggle label="Auto emoji" sub="Adds a matching emoji above key words" on={cap.emoji} set={(v) => setCap({ emoji: v })} />
            {speakerCount >= 2 && (
              <>
                <Toggle
                  label="Colour by speaker"
                  sub="Each voice gets its own caption colour, picked to contrast with the highlight"
                  on={cap.speaker_colors}
                  set={(v) => setCap({ speaker_colors: v })}
                />
                {cap.speaker_colors && (
                  <div className="speaker-colors">
                    {Array.from({ length: speakerCount }, (_, k) => {
                      const auto = speakerColors?.[k] ?? "#ffffff";
                      const val = cap.speakers?.[String(k)] ?? auto;
                      const custom = !!cap.speakers?.[String(k)];
                      return (
                        <label key={k} className="speaker-color">
                          <input
                            type="color"
                            value={val.length === 7 ? val : "#ffffff"}
                            onChange={(e) => setCap({ speakers: { ...cap.speakers, [String(k)]: e.target.value } })}
                          />
                          <span>Speaker {k + 1}</span>
                          {custom && (
                            <button
                              className="link small"
                              onClick={(e) => {
                                e.preventDefault();
                                const next = { ...cap.speakers };
                                delete next[String(k)];
                                setCap({ speakers: next });
                              }}
                            >
                              auto
                            </button>
                          )}
                        </label>
                      );
                    })}
                  </div>
                )}
              </>
            )}
          </>
        )}

        {tab === "effects" && (
          <>
            <Toggle label="Auto zoom" sub="Punch-in on new sentences and key words" on={edit.zoom.enabled} set={(v) => onEdit({ ...edit, zoom: { ...edit.zoom, enabled: v } })} />
            {edit.zoom.enabled && (
              <Slider label="Zoom strength" min={0.04} max={0.3} step={0.01} value={edit.zoom.strength} set={(v) => onEdit({ ...edit, zoom: { ...edit.zoom, strength: v } })} fmt={(v) => `+${Math.round(v * 100)}%`} />
            )}
            <Toggle label="Progress bar" on={edit.progress.enabled} set={(v) => onEdit({ ...edit, progress: { ...edit.progress, enabled: v } })} />
            {edit.progress.enabled && (
              <div className="row gap8 field">
                <input type="color" value={edit.progress.color} onChange={(e) => onEdit({ ...edit, progress: { ...edit.progress, color: e.target.value } })} />
                <div className="seg small">
                  {(["top", "bottom"] as const).map((p) => (
                    <button key={p} className={edit.progress.position === p ? "sel" : ""} onClick={() => onEdit({ ...edit, progress: { ...edit.progress, position: p } })}>
                      {p}
                    </button>
                  ))}
                </div>
              </div>
            )}
            <div className="field">
              <span className="field-label">Colour</span>
              <div className="filters">
                {presets.filters.map((f) => (
                  <button key={f} className={`filter ${edit.filter === f ? "sel" : ""}`} onClick={() => onEdit({ ...edit, filter: f })}>
                    <div className="filter-art" style={{ backgroundImage: `url(${thumb})`, filter: CSS_FILTERS[f] }} />
                    <span>{FILTER_LABELS[f] ?? f}</span>
                  </button>
                ))}
              </div>
            </div>
            <Toggle label="Loudness normalize" sub="−14 LUFS, the level TikTok / Shorts / Reels use" on={edit.audio.normalize} set={(v) => onEdit({ ...edit, audio: { normalize: v } })} />
          </>
        )}

        {tab === "layout" && (
          <>
            <div className="field">
              <span className="field-label">Aspect ratio</span>
              <div className="aspects">
                {Object.entries(presets.aspects).map(([id, [w, h]]) => (
                  <button key={id} className={`aspect ${edit.aspect === id ? "sel" : ""}`} onClick={() => onEdit({ ...edit, aspect: id as Edit["aspect"] })}>
                    <span className="aspect-box" style={{ width: (w / Math.max(w, h)) * 34, height: (h / Math.max(w, h)) * 34 }} />
                    {id}
                  </button>
                ))}
              </div>
            </div>
            <div className="field">
              <span className="field-label">Framing</span>
              <div className="layout-list">
                {([
                  ["auto", "Auto", "Follows the speaker, splits two-person shots, blur-fit if nobody is on screen"],
                  ["track", "Speaker focus", "Cuts to whoever is talking, like switching podcast cameras"],
                  ["split", "Split screen", "Both people stacked when they share one camera shot"],
                  ["gaming", "Gaming / stream", "Face-cam in its own panel, gameplay in the other"],
                  ["center", "Center crop", "Static crop from the middle"],
                  ["fit", "Fit + blur", "Whole frame over a blurred background (gameplay, slides)"],
                ] as [Edit["layout"], string, string][]).map(([id, l, sub]) => (
                  <button key={id} className={`layout-opt ${edit.layout === id ? "sel" : ""}`} onClick={() => onEdit({ ...edit, layout: id })}>
                    <Layers size={15} />
                    <div>
                      <b>{l}</b>
                      <span>{sub}</span>
                    </div>
                  </button>
                ))}
              </div>
              <StreamControls edit={edit} onEdit={onEdit} />
              {clip.track?.coverage !== undefined && (
                <p className="hint">
                  Faces in {Math.round((clip.track.coverage ?? 0) * 100)}% of this clip
                  {clip.track.people ? `, ${clip.track.people} ${clip.track.people === 1 ? "person" : "people"}` : ""}
                  {clip.track.switches ? `, ${clip.track.switches} speaker cuts` : ""}.
                </p>
              )}
            </div>
          </>
        )}

        {tab === "text" && (
          <>
            <label className="field">
              <span className="field-label">Clip title</span>
              <input className="input" value={title} onChange={(e) => onTitle(e.target.value)} />
            </label>
            <Toggle label="Title on video" sub="A context banner or a short hook so viewers know what the clip is about" on={edit.title.enabled} set={(v) => onEdit({ ...edit, title: { ...edit.title, enabled: v } })} />
            {edit.title.enabled && (
              <>
                <div className="field">
                  <span className="field-label">Style</span>
                  <div className="seg small">
                    {([
                      ["auto", "Auto"],
                      ["banner", "Context banner"],
                      ["hook", "Hook at top"],
                    ] as const).map(([id, l]) => (
                      <button key={id} className={(edit.title.mode ?? "auto") === id ? "sel" : ""} onClick={() => onEdit({ ...edit, title: { ...edit.title, mode: id } })}>
                        {l}
                      </button>
                    ))}
                  </div>
                  <p className="hint">Auto puts a banner on the seam of gaming and split-screen clips for the whole clip, and a hook at the top of the others.</p>
                </div>
                <label className="field">
                  <span className="field-label">On-screen text</span>
                  <input className="input" placeholder={clip.banner || title} value={edit.title.text ?? ""} onChange={(e) => onEdit({ ...edit, title: { ...edit.title, text: e.target.value || null } })} />
                </label>
                {(edit.title.mode ?? "auto") === "hook" && (
                  <Slider label="Show for" min={1} max={10} step={0.5} value={edit.title.seconds} set={(v) => onEdit({ ...edit, title: { ...edit.title, seconds: v } })} fmt={(v) => `${v}s`} />
                )}
              </>
            )}
            <label className="field">
              <span className="field-label">Hashtags</span>
              <textarea className="input" rows={3} value={hashtags} onChange={(e) => onHashtags(e.target.value)} />
            </label>
          </>
        )}
      </div>
      <div className="insp-foot">
        <button className="btn ghost wide" onClick={onApplyAll} title="Copy this clip's style (captions, effects, layout) to every clip and re-render them">
          Apply style to all clips
        </button>
      </div>
    </div>
  );
}

function Toggle({ label, sub, on, set }: { label: string; sub?: string; on: boolean; set: (v: boolean) => void }) {
  return (
    <label className="switch-row">
      <div>
        <b>{label}</b>
        {sub && <span>{sub}</span>}
      </div>
      <span className={`switch ${on ? "on" : ""}`}>
        <input type="checkbox" checked={on} onChange={(e) => set(e.target.checked)} />
        <i />
      </span>
    </label>
  );
}

function Slider({ label, min, max, step, value, set, fmt }: {
  label: string; min: number; max: number; step: number; value: number; set: (v: number) => void; fmt?: (v: number) => string;
}) {
  return (
    <label className="field slider">
      <span className="field-label">
        {label} <em>{fmt ? fmt(value) : value}</em>
      </span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => set(parseFloat(e.target.value))} />
    </label>
  );
}

function StreamControls({ edit, onEdit }: { edit: Edit; onEdit: (e: Edit) => void }) {
  const st = edit.stream ?? { auto: true, cam: "top", cam_size: 0.48, cam_zoom: 1, game: "fill" };
  const set = (patch: Partial<Edit["stream"]>) => onEdit({ ...edit, stream: { ...st, ...patch } });
  const show = edit.layout === "gaming" || edit.layout === "auto";
  if (!show) return null;
  return (
    <div className="stream-box">
      <b>Gaming / stream</b>
      {edit.layout === "auto" && (
        <Toggle
          label="Use for face-cam scenes"
          sub="When the video is gameplay with a small webcam, Auto switches to this layout"
          on={st.auto}
          set={(v) => set({ auto: v })}
        />
      )}
      <div className="field">
        <span className="field-label">Face-cam position</span>
        <div className="seg small">
          {(["top", "bottom"] as const).map((v) => (
            <button key={v} className={st.cam === v ? "sel" : ""} onClick={() => set({ cam: v })}>
              {v === "top" ? "Top" : "Bottom"}
            </button>
          ))}
        </div>
      </div>
      <div className="field">
        <span className="field-label">Gameplay</span>
        <div className="seg small">
          {(["fill", "fit"] as const).map((v) => (
            <button key={v} className={st.game === v ? "sel" : ""} onClick={() => set({ game: v })}>
              {v === "fill" ? "Fill (crop centre)" : "Fit (whole screen)"}
            </button>
          ))}
        </div>
      </div>
      <Slider label="Face-cam size" min={0.25} max={0.55} step={0.01} value={st.cam_size} set={(v) => set({ cam_size: v })} fmt={(v) => `${Math.round(v * 100)}%`} />
      <Slider label="Face-cam zoom" min={0.6} max={1.8} step={0.05} value={st.cam_zoom ?? 1} set={(v) => set({ cam_zoom: v })} fmt={(v) => `${v.toFixed(2)}x`} />
    </div>
  );
}
