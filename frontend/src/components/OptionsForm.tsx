import type { CaptionPreset, Health, Options, Presets } from "../api";

const ASPECTS: [Options["aspect"], string][] = [
  ["9:16", "9:16"],
  ["1:1", "1:1"],
  ["4:5", "4:5"],
  ["16:9", "16:9"],
];
const LAYOUTS: [Options["layout"], string][] = [
  ["auto", "Auto"],
  ["track", "Speaker focus"],
  ["split", "Split screen"],
  ["gaming", "Gaming"],
  ["center", "Center"],
  ["fit", "Fit + blur"],
];

export function PresetSwatch({ p, selected, onClick }: { p: CaptionPreset; selected: boolean; onClick: () => void }) {
  const sample = p.upper ? "THE TRUTH" : "the truth";
  const [a, b] = sample.split(" ");
  return (
    <button className={`swatch ${selected ? "sel" : ""}`} onClick={onClick} type="button" title={p.label}>
      <div
        className="swatch-art"
        style={{
          fontFamily: `"${p.font}", "Arial Black", Impact, sans-serif`,
          fontWeight: p.bold ? 900 : 400,
          color: p.color,
          WebkitTextStroke: `${Math.max(2, p.outline_w * 0.5)}px ${p.outline}`,
          paintOrder: "stroke fill",
          textShadow: p.glow ? `0 0 10px ${p.active}` : undefined,
        }}
      >
        <span>{a}</span>
        <span
          style={
            p.anim === "box"
              ? { background: p.box ?? undefined, WebkitTextStroke: "0", padding: "0 4px", borderRadius: 3 }
              : { color: p.active ?? p.color }
          }
        >
          {b}
        </span>
      </div>
      <div className="swatch-name">{p.label}</div>
    </button>
  );
}

export function OptionsForm({
  value,
  onChange,
  presets,
  health,
  compact = false,
}: {
  value: Options;
  onChange: (o: Options) => void;
  presets: Presets | null;
  health: Health | null;
  compact?: boolean;
}) {
  const set = <K extends keyof Options>(k: K, v: Options[K]) => onChange({ ...value, [k]: v });
  const prov = health?.llm.providers ?? {};

  return (
    <div className={`options ${compact ? "compact" : ""}`}>
      <div className="opt-row">
        <div className="opt">
          <span className="field-label">Clips <em className="hint-inline">length picked by AI</em></span>
          <div className="stepper">
            <button type="button" onClick={() => set("clips", Math.max(1, value.clips - 1))}>−</button>
            <b>{value.clips}</b>
            <button type="button" onClick={() => set("clips", Math.min(20, value.clips + 1))}>+</button>
          </div>
        </div>
        <div className="opt">
          <span className="field-label">Quality bar</span>
          <div className="seg">
            {([
              ["any", "Any", "Keep the best clips, however strong"],
              ["good", "Good 65+", "Drop clips scoring under 65"],
              ["viral", "Viral only 75+", "Only clips scoring 75 or more; you may get fewer clips"],
            ] as [Options["quality"], string, string][]).map(([id, label, hint]) => (
              <button key={id} type="button" title={hint} className={(value.quality ?? "good") === id ? "sel" : ""} onClick={() => set("quality", id)}>
                {label}
              </button>
            ))}
          </div>
        </div>
        <div className="opt">
          <span className="field-label">Aspect</span>
          <div className="seg">
            {ASPECTS.map(([id, label]) => (
              <button key={id} type="button" className={value.aspect === id ? "sel" : ""} onClick={() => set("aspect", id)}>
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="opt">
        <span className="field-label">Caption template</span>
        <div className="swatches">
          {presets?.captions.map((p) => (
            <PresetSwatch key={p.id} p={p} selected={value.preset === p.id && value.captions} onClick={() => onChange({ ...value, preset: p.id, captions: true })} />
          ))}
          <button type="button" className={`swatch none ${!value.captions ? "sel" : ""}`} onClick={() => set("captions", false)}>
            <div className="swatch-art">No captions</div>
            <div className="swatch-name">Off</div>
          </button>
        </div>
      </div>

      <div className="opt-row">
        <div className="opt">
          <span className="field-label">AI clip finder</span>
          <div className="seg">
            {[
              ["ollama", "Ollama (local)"],
              ["openrouter", "OpenRouter"],
              ["none", "Rules only"],
            ].map(([id, label]) => (
              <button
                key={id}
                type="button"
                className={value.provider === id ? "sel" : ""}
                onClick={() => set("provider", id)}
                title={prov[id]?.detail}
              >
                {id !== "none" && <span className={`dot ${prov[id]?.ok ? "ok" : "warn"}`} />}
                {label}
              </button>
            ))}
          </div>
        </div>
        <div className="opt">
          <span className="field-label">Framing</span>
          <div className="seg">
            {LAYOUTS.map(([id, label]) => (
              <button key={id} type="button" className={value.layout === id ? "sel" : ""} onClick={() => set("layout", id)}>
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="opt-row">
        <label className="opt grow">
          <span className="field-label">Topic / niche (optional)</span>
          <input className="input" value={value.topic} onChange={(e) => set("topic", e.target.value)} placeholder="e.g. fitness, SaaS founders, true crime" />
        </label>
        <label className="opt">
          <span className="field-label">Whisper</span>
          <select className="input" value={value.whisper_model} onChange={(e) => set("whisper_model", e.target.value)}>
            {(health?.whisper_models ?? ["small", "medium", "large-v3-turbo"]).map((m) => (
              <option key={m}>{m}</option>
            ))}
          </select>
        </label>
        <label className="opt">
          <span className="field-label">Language</span>
          <input className="input narrow" value={value.language} onChange={(e) => set("language", e.target.value)} placeholder="auto" />
        </label>
      </div>

      <div className="toggles">
        {([
          ["zoom", "Auto zoom"],
          ["title", "Hook title"],
          ["emoji", "Emoji"],
          ["progress", "Progress bar"],
          ["silence", "Remove silences"],
          ["fillers", "Remove um / uh"],
        ] as [keyof Options, string][]).map(([k, label]) => (
          <label key={k} className={`toggle ${value[k] ? "on" : ""}`}>
            <input type="checkbox" checked={!!value[k]} onChange={(e) => set(k, e.target.checked as never)} />
            {label}
          </label>
        ))}
      </div>
    </div>
  );
}
