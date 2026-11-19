import { useRef } from "react";
import type { Plan, Word } from "../api";
import { fmtTime } from "../lib/util";

interface Props {
  plan: Plan;
  words: Word[];           // source words around the clip (absolute times)
  view: [number, number];  // absolute window shown
  playhead: number;        // absolute source time
  onTrim: (start: number, end: number) => void;
  onSeek: (abs: number) => void;
}

/** Source-time strip: drag the handles to trim (snaps to word boundaries), click to seek. */
export function Timeline({ plan, words, view, playhead, onTrim, onSeek }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [v0, v1] = view;
  const span = Math.max(1, v1 - v0);
  const pct = (t: number) => `${((t - v0) / span) * 100}%`;
  const a = plan.a;
  const b = plan.b;

  const timeAt = (clientX: number) => {
    const r = ref.current!.getBoundingClientRect();
    return v0 + ((clientX - r.left) / r.width) * span;
  };

  const snap = (t: number, edge: "start" | "end") => {
    let best = t;
    let dist = 0.6;
    for (const w of words) {
      const x = edge === "start" ? w.s : w.e;
      const d = Math.abs(x - t);
      if (d < dist) {
        dist = d;
        best = x;
      }
    }
    return edge === "start" ? Math.max(0, best - 0.05) : best + 0.12;
  };

  const drag = (edge: "start" | "end") => (e: React.PointerEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const target = e.currentTarget as HTMLElement;
    target.setPointerCapture(e.pointerId);
    let s = a;
    let en = b;
    const move = (ev: PointerEvent) => {
      const t = timeAt(ev.clientX);
      if (edge === "start") s = Math.min(snap(t, "start"), en - 3);
      else en = Math.max(snap(t, "end"), s + 3);
      onTrim(Math.max(v0, s), Math.min(v1, en));
    };
    const up = () => {
      target.removeEventListener("pointermove", move);
      target.removeEventListener("pointerup", up);
    };
    target.addEventListener("pointermove", move);
    target.addEventListener("pointerup", up);
  };

  // cut regions = gaps between keep segments
  const cuts: [number, number][] = [];
  let cur = 0;
  for (const [s, e] of plan.segs) {
    if (s - cur > 0.05) cuts.push([a + cur, a + s]);
    cur = e;
  }
  if (b - a - cur > 0.05) cuts.push([a + cur, b]);

  const ticks: number[] = [];
  const step = span > 120 ? 30 : span > 60 ? 10 : 5;
  for (let t = Math.ceil(v0 / step) * step; t <= v1; t += step) ticks.push(t);

  return (
    <div className="timeline">
      <div className="tl-track" ref={ref} onPointerDown={(e) => onSeek(timeAt(e.clientX))}>
        {words.map((w, i) => (
          <span key={i} className="tl-word" style={{ left: pct(w.s), width: `${((w.e - w.s) / span) * 100}%` }} />
        ))}
        <div className="tl-dim" style={{ left: 0, width: pct(a) }} />
        <div className="tl-dim" style={{ left: pct(b), right: 0 }} />
        {cuts.map(([s, e], i) => (
          <div key={i} className="tl-cut" style={{ left: pct(s), width: `${((e - s) / span) * 100}%` }} />
        ))}
        <div className="tl-range" style={{ left: pct(a), width: `${((b - a) / span) * 100}%` }}>
          <div className="tl-handle left" onPointerDown={drag("start")} />
          <div className="tl-handle right" onPointerDown={drag("end")} />
        </div>
        <div className="tl-head" style={{ left: pct(playhead) }} />
      </div>
      <div className="tl-ticks">
        {ticks.map((t) => (
          <span key={t} style={{ left: pct(t) }}>{fmtTime(t)}</span>
        ))}
      </div>
      <div className="tl-info muted small">
        <span>Start {fmtTime(a, true)}</span>
        <span>
          Length <b>{plan.duration.toFixed(1)}s</b>
          {cuts.length > 0 && ` · ${cuts.length} cut${cuts.length > 1 ? "s" : ""} (−${(b - a - plan.duration).toFixed(1)}s)`}
        </span>
        <span>End {fmtTime(b, true)}</span>
      </div>
    </div>
  );
}
