// Timeline maths for the live preview. The plan comes from the backend
// (POST /api/projects/:id/clips/:cid/plan), so cuts, cue grouping and zoom
// keyframes are exactly what ffmpeg will render.
import type { Plan, PlanWord } from "../api";

/** Interpolate piecewise-linear keyframes [[t, v], ...]. */
export function interp(keys: [number, number][] | null | undefined, t: number, fallback = 0.5): number {
  if (!keys || !keys.length) return fallback;
  if (t <= keys[0][0]) return keys[0][1];
  for (let i = 0; i < keys.length - 1; i++) {
    const [t0, v0] = keys[i];
    const [t1, v1] = keys[i + 1];
    if (t <= t1) return v0 + ((v1 - v0) * (t - t0)) / Math.max(1e-6, t1 - t0);
  }
  return keys[keys.length - 1][1];
}

/** Clip-relative source time -> output time, or null inside a cut. */
export function toOutput(rel: number, segs: [number, number][]): number | null {
  let acc = 0;
  for (const [s, e] of segs) {
    if (rel < s - 1e-3) return null;
    if (rel <= e + 1e-3) return acc + Math.max(0, rel - s);
    acc += e - s;
  }
  return null;
}

/** Output time -> clip-relative source time. */
export function toSource(out: number, segs: [number, number][]): number {
  let acc = 0;
  for (const [s, e] of segs) {
    const len = e - s;
    if (out <= acc + len) return s + (out - acc);
    acc += len;
  }
  const last = segs[segs.length - 1];
  return last ? last[1] : 0;
}

/** If rel is inside a cut (or before the first segment), where should playback jump? */
export function nextPlayable(rel: number, segs: [number, number][]): number | null {
  for (const [s, e] of segs) {
    if (rel < s - 0.02) return s;
    if (rel <= e) return null;
  }
  return -1; // past the end
}

export interface CaptionState {
  cue: number;        // index into plan.cues, -1 = none
  active: number;     // active word in the cue
  progress: number;   // 0..1 progress through the active word (karaoke fill)
}

export function captionAt(plan: Plan, t: number): CaptionState {
  const cues = plan.cues;
  // binary search on cue start
  let lo = 0;
  let hi = cues.length - 1;
  let idx = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if ((cues[mid][0].as ?? cues[mid][0].s) <= t) {
      idx = mid;
      lo = mid + 1;
    } else hi = mid - 1;
  }
  if (idx < 0) return { cue: -1, active: -1, progress: 0 };
  const cue = cues[idx];
  const last = cue[cue.length - 1];
  if (t >= (last.ae ?? last.e)) return { cue: -1, active: -1, progress: 0 };
  let active = 0;
  for (let i = 0; i < cue.length; i++) if ((cue[i].as ?? cue[i].s) <= t) active = i;
  const w: PlanWord = cue[active];
  const span = Math.max(0.05, (cue[active + 1]?.s ?? w.e) - w.s);
  return { cue: idx, active, progress: Math.min(1, Math.max(0, (t - w.s) / span)) };
}

/** Crop window position (top-left, source pixels) for a clip-relative time. */
export function cropAt(plan: Plan, rel: number): { x: number; y: number } {
  const { w: cw, h: ch, axis } = plan.crop;
  const { w: sw, h: sh } = plan.source;
  const c = plan.layout === "track" && plan.keys ? interp(plan.keys, rel) : 0.5;
  if (axis === "x") return { x: Math.min(Math.max(c * sw - cw / 2, 0), sw - cw), y: 0 };
  if (axis === "y") return { x: 0, y: Math.min(Math.max(c * sh - ch / 2, 0), sh - ch) };
  return { x: 0, y: 0 };
}
