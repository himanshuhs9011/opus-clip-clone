import type { CSSProperties } from "react";
import type { CaptionPreset, Plan } from "../api";
import type { CaptionState } from "../lib/playback";

interface Props {
  plan: Plan;
  cap: CaptionState;
  t: number;       // output time
  k: number;       // preview px per output px
}

/** DOM version of the libass captions (opusclip/captions.py). Geometry is scaled from output pixels. */
export function CaptionOverlay({ plan, cap, t, k }: Props) {
  const e = plan.edit;
  const st = plan.preset;
  const fs = st.size * plan.scale * (e.captions.size || 1) * k;
  const W = plan.out.w * k;
  const H = plan.out.h * k;
  const tScale = plan.scale * k;

  const titleOn = e.title.enabled && plan.title && t < Math.min(e.title.seconds, plan.duration);
  const emoji = e.captions.enabled && e.captions.emoji ? plan.emoji.find((x) => t >= x.t && t < x.end) : undefined;
  const cue = e.captions.enabled && cap.cue >= 0 ? plan.cues[cap.cue] : null;
  const sp = cue?.[0]?.sp ?? -1;
  const base = (plan.speaker_colors && sp >= 0 && plan.speaker_colors[sp]) || st.color;

  return (
    <div className="overlay" style={{ width: W, height: H }}>
      {titleOn && (
        <div
          className="ov-title"
          style={{
            top: H * (plan.out.h > plan.out.w ? 0.11 : 0.07),
            fontSize: 54 * tScale,
            padding: `${10 * tScale}px ${18 * tScale}px`,
            maxWidth: W - 2 * 110 * tScale + 36 * tScale,
          }}
        >
          {plan.title}
        </div>
      )}
      {emoji && (
        <div
          key={`e${emoji.t}`}
          className="ov-emoji"
          style={{ top: H * e.captions.position - fs * 1.45, fontSize: fs * 1.2 }}
        >
          {emoji.emoji}
        </div>
      )}
      {cue && (
        <div
          key={`c${cap.cue}`}
          className={`ov-cue anim-${st.anim}`}
          style={{
            top: H * e.captions.position,
            width: W - 2 * 70 * plan.scale * k,
            ...baseText(st, fs, k, plan.scale),
            color: base,
            textTransform: plan.upper ? "uppercase" : "none",
            ["--bounce" as string]: `${46 * plan.scale * (e.captions.size || 1) * k}px`,
          }}
        >
          {cue.map((w, i) => (
            <Word key={w.id + i} preset={st} base={base} i={i} cap={cap} text={w.w} keyword={w.k && e.captions.keywords} fs={fs} />
          ))}
        </div>
      )}
    </div>
  );
}

function baseText(st: CaptionPreset, fs: number, k: number, scale: number): CSSProperties {
  const ow = Math.max(1, st.outline_w * scale * k);
  const sh = st.shadow * scale * k;
  const shadows = [sh > 0 ? `${sh}px ${sh}px 0 rgba(0,0,0,0.55)` : "", st.glow ? `0 0 ${st.glow * 4 * k * 3}px ${st.active ?? st.color}` : ""]
    .filter(Boolean)
    .join(", ");
  return {
    fontFamily: `"${st.font}", "Arial Black", Impact, sans-serif`,
    fontWeight: st.bold ? 900 : 400,
    fontSize: fs,
    color: st.color,
    WebkitTextStroke: `${ow * 2}px ${st.outline}`,
    paintOrder: "stroke fill",
    textShadow: shadows || undefined,
    lineHeight: 1.18,
  };
}

function Word({ preset: st, base, i, cap, text, keyword, fs }: {
  preset: CaptionPreset; base: string; i: number; cap: CaptionState; text: string; keyword: boolean; fs: number;
}) {
  const isActive = i === cap.active && !!st.active && st.anim !== "fade";
  const style: CSSProperties = {};
  if (st.anim === "karaoke") {
    const sung = st.active ?? st.color;
    if (i < cap.active) style.color = sung;
    else if (i === cap.active) {
      const p = Math.round(cap.progress * 100);
      style.backgroundImage = `linear-gradient(90deg, ${sung} ${p}%, ${base} ${p}%)`;
      style.WebkitBackgroundClip = "text";
      style.backgroundClip = "text";
      style.color = "transparent";
    }
  } else if (isActive) {
    style.color = st.active ?? st.color;
    if (st.anim === "pop") style.transform = "scale(1.1)";
    if (st.anim === "box" && st.box) {
      style.background = st.box;
      style.boxShadow = `0 0 0 ${fs * 0.12}px ${st.box}`;
      style.borderRadius = fs * 0.08;
      style.WebkitTextStroke = "0";
    }
  } else if (keyword && st.keyword) {
    style.color = st.keyword;
  }
  return (
    <>
      <span className="ov-word" style={style}>
        {text}
      </span>{" "}
    </>
  );
}
