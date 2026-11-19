import { useEffect, useRef, useState } from "react";

export const fmtTime = (s: number, tenths = false) => {
  s = Math.max(0, s || 0);
  const m = Math.floor(s / 60);
  const sec = s - m * 60;
  return tenths
    ? `${m}:${sec.toFixed(1).padStart(4, "0")}`
    : `${m}:${String(Math.floor(sec)).padStart(2, "0")}`;
};

export const fmtDuration = (s: number) => {
  s = Math.round(s || 0);
  if (s >= 3600) return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
  if (s >= 60) return `${Math.floor(s / 60)}m ${s % 60}s`;
  return `${s}s`;
};

export const timeAgo = (t: number) => {
  const d = Date.now() / 1000 - t;
  if (d < 60) return "just now";
  if (d < 3600) return `${Math.floor(d / 60)}m ago`;
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`;
  return `${Math.floor(d / 86400)}d ago`;
};

export const scoreTone = (v: number) => (v >= 80 ? "hi" : v >= 65 ? "mid" : "lo");

export const gradeOf = (v: number) => {
  const cuts: [number, string][] = [
    [90, "A+"], [84, "A"], [78, "A-"], [73, "B+"], [68, "B"], [63, "B-"], [58, "C+"], [52, "C"], [45, "C-"],
  ];
  for (const [c, g] of cuts) if (v >= c) return g;
  return "D";
};

export const slug = (s: string) =>
  s.toLowerCase().replace(/[^\w]+/g, "-").replace(/^-|-$/g, "").slice(0, 40) || "clip";

export const clone = <T,>(x: T): T => JSON.parse(JSON.stringify(x));

export function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setV(value), ms);
    return () => clearTimeout(id);
  }, [value, ms]);
  return v;
}

/** setInterval that always calls the latest callback; pass null to pause. */
export function useInterval(cb: () => void, ms: number | null) {
  const ref = useRef(cb);
  ref.current = cb;
  useEffect(() => {
    if (ms === null) return;
    const id = setInterval(() => ref.current(), ms);
    return () => clearInterval(id);
  }, [ms]);
}

/** Tiny hash router: #/ , #/p/<id> , #/p/<id>/edit/<clip> */
export function useRoute(): string[] {
  const read = () => window.location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  const [parts, setParts] = useState(read);
  useEffect(() => {
    const on = () => setParts(read());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return parts;
}

export const go = (path: string) => {
  window.location.hash = path;
};

export const CSS_FILTERS: Record<string, string> = {
  none: "contrast(1.03) saturate(1.05)",
  vivid: "contrast(1.08) saturate(1.35)",
  warm: "sepia(0.18) saturate(1.15) contrast(1.02)",
  cool: "hue-rotate(-10deg) saturate(1.05) brightness(1.02)",
  cinematic: "contrast(1.12) saturate(0.85) sepia(0.06)",
  bw: "grayscale(1) contrast(1.12)",
  vintage: "sepia(0.35) contrast(0.95) saturate(0.9) brightness(1.03)",
};

export const FILTER_LABELS: Record<string, string> = {
  none: "Original",
  vivid: "Vivid",
  warm: "Warm",
  cool: "Cool",
  cinematic: "Cinematic",
  bw: "B&W",
  vintage: "Vintage",
};
