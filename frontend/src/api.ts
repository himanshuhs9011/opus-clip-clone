// Types mirror the Python models (opusclip/models.py, opusclip/edit.py).

export interface Score {
  total: number;
  hook: number;
  retention: number;
  emotion: number;
  value: number;
  quotability: number;
  reasons: string[];
  risks: string[];
  features: Record<string, number>;
  source: string;
  grade: string;
}

export interface Edit {
  start: number | null;
  end: number | null;
  aspect: "9:16" | "1:1" | "4:5" | "16:9";
  layout: "auto" | "track" | "split" | "gaming" | "center" | "fit";
  stream: {
    auto: boolean;
    template?: StreamTemplate;
    corner?: "tl" | "tr" | "bl" | "br";
    cam: "top" | "bottom";
    cam_size: number;
    cam_zoom: number;
    game: "fill" | "fit";
  };
  captions: {
    enabled: boolean;
    preset: string;
    position: number;
    size: number;
    uppercase: boolean | null;
    words: number | null;
    keywords: boolean;
    emoji: boolean;
    speaker_colors: boolean;
    speakers: Record<string, string>;
  };
  title: { enabled: boolean; text: string | null; seconds: number; mode?: "auto" | "banner" | "hook" };
  intro?: { enabled: boolean };
  progress: { enabled: boolean; color: string; position: "top" | "bottom" };
  zoom: { enabled: boolean; strength: number };
  cuts: { silence: boolean; fillers: boolean; removed: string[] };
  word_edits: Record<string, string>;
  filter: string;
  audio: { normalize: boolean };
}

export interface RenderState {
  // candidate: found by the Clip Agent, not made yet (the user picks which get rendered)
  status?: "candidate" | "queued" | "rendering" | "done" | "error";
  progress?: number;
  error?: string;
  seconds?: number;
  encoder?: string;
  duration?: number;
  message?: string;
  at?: number;
}

export interface Clip {
  id: string;
  start: number;
  end: number;
  duration: number;
  title: string;
  hook: string;
  summary: string;
  why: string;
  hashtags: string[];
  banner?: string;
  score: Score | null;
  rank: number;
  file: string | null;
  thumb: string | null;
  version: number;
  track: { coverage?: number; t0?: number; t1?: number; people?: number; switches?: number; speaker_map?: Record<string, number> };
  edit: Partial<Edit>;
  render: RenderState;
  error: string | null;
  kind?: "short" | "long";
  package?: ClipPackage;
}

/** Ready-to-post text written with the whole video in mind (opusclip/story.py). */
export interface ClipPackage {
  yt_title?: string;
  thumbnail_text?: string;
  description?: string;
  chapters?: [number, string][];
}

export interface Word {
  s: number;
  e: number;
  w: string;
  sp?: number;
}

export interface Sentence {
  id: number;
  start: number;
  end: number;
  text: string;
  words: Word[];
}

export interface Project {
  id: string;
  name: string;
  created: number;
  // picking: clips were found and wait for the user to pick which to make
  status: "queued" | "running" | "picking" | "done" | "error";
  stage: string;
  message: string;
  progress: number;
  error: string | null;
  duration: number;
  width: number;
  height: number;
  fps: number;
  language: string;
  preview: string | null;
  speakers: { count?: number; share?: number[] };
  clips: Clip[];
  settings: Record<string, string> & { quality?: { bar: number; level: string; kept: number; asked: number } };
  options: Options;
  timings: Record<string, number>;
  sentences?: Sentence[];
}

export interface ProjectSummary {
  id: string;
  name: string;
  status: string;
  stage: string;
  progress: number;
  created: number;
  duration: number;
  clips: number;
  best: number;
  thumb: string | null;
}

export interface Options {
  clips: number;
  length: "ai" | "short" | "medium" | "long";
  quality: "any" | "good" | "viral";
  topic: string;
  provider: string;
  llm_model: string | null;
  whisper_model: string;
  language: string;
  preset: string;
  captions: boolean;
  aspect: Edit["aspect"];
  layout: Edit["layout"];
  stream_template?: StreamTemplate;
  zoom: boolean;
  title: boolean;
  intro?: boolean;
  emoji: boolean;
  progress: boolean;
  silence: boolean;
  fillers: boolean;
  focus?: string;
  max_clip?: number | null;
  min_clip?: number | null;
  range?: AgentRange | null;
  pick?: boolean;
  format?: "short" | "long" | "both";
  long_len?: number | null;
}

// Clip Agent (opusclip/wizard.py)
export interface AgentRange {
  from: number | null;
  to: number | null;
  last: number | null;
}

export interface AgentRequest {
  text: string;
  url: string | null;
  focus: string;
  focus_by?: "ai";
  max_clip: number | null;
  min_clip: number | null;
  count: number | null;
  range: AgentRange | null;
  content: "auto" | "talk" | "gaming";
  stream_template: StreamTemplate | null;
  aspect: Edit["aspect"] | null;
  captions: boolean | null;
  preset: string | null;
  quality: Options["quality"] | null;
  make: "all" | number | null;
  format?: "short" | "long" | "both";
  long_len?: number | null;
}

export interface AgentPlan {
  summary: string;
  steps: { title: string; detail: string }[];
  questions: string[];
  ready: boolean;
}

export interface CaptionPreset {
  id: string;
  label: string;
  font: string;
  size: number;
  bold: boolean;
  upper: boolean;
  words: number;
  color: string;
  active: string | null;
  keyword: string | null;
  outline: string;
  outline_w: number;
  shadow: number;
  anim: "pop" | "bounce" | "box" | "karaoke" | "none" | "fade";
  box: string | null;
  glow: number;
}

export interface Presets {
  captions: CaptionPreset[];
  aspects: Record<string, [number, number]>;
  layouts: string[];
  filters: string[];
  lengths: Record<string, [number, number]>;
  default_edit: Edit;
  default_options: Options;
}

export interface PlanWord {
  s: number;
  e: number;
  w: string;
  k: boolean;
  id: string;
  sp?: number;
  as?: number;
  ae?: number;
}

export interface SplitHalf {
  id: number;
  h: number;
  x: [number, number][];
  y: [number, number][];
}

export interface SplitScene {
  t0: number;
  t1: number;
  top: SplitHalf;
  bottom: SplitHalf;
}

export type StreamTemplate = "stack" | "reaction" | "even" | "bubble" | "card" | "gameplay";

export interface StreamGeom {
  template?: StreamTemplate;
  inset?: [number, number, number, number] | null;
  corner?: "tl" | "tr" | "bl" | "br";
  cam_pos: "top" | "bottom";
  cam_h: number;
  game_h: number;
  game: "fill" | "fit";
  cam_crop: [number, number, number, number] | null;
  game_crop: [number, number, number, number];
}

export interface Plan {
  edit: Edit;
  a: number;
  b: number;
  duration: number;
  segs: [number, number][];
  words: PlanWord[];
  cues: PlanWord[][];
  zoom: [number, number][];
  emoji: { t: number; end: number; emoji: string }[];
  layout: "track" | "split" | "gaming" | "center" | "fit";
  keys: [number, number][] | null;
  stream: StreamGeom | null;
  stream_ranges: [number, number][];
  split: SplitScene[];
  fit_ranges?: [number, number][];
  split_crop: { w: number; h: number };
  voices: number[];
  speaker_colors: string[] | null;
  speaker_map: Record<string, number>;
  tracked: boolean;
  crop: { w: number; h: number; axis: "x" | "y" | "none" };
  out: { w: number; h: number };
  scale: number;
  preset: CaptionPreset;
  upper: boolean;
  per_cue: number;
  title: string;
  title_info?: { mode: "banner" | "hook"; enabled: boolean; text: string; y: number | null; seconds: number };
  banner?: string;
  video: string | null;
  video_offset?: number;
  partial?: boolean;
  source: { w: number; h: number; duration: number };
}

export interface Health {
  ok: boolean;
  ffmpeg: boolean;
  yt_dlp: boolean;
  encoder: string;
  whisper_model: string;
  whisper_models: string[];
  llm: {
    provider: string;
    providers: Record<string, { ok: boolean; detail: string; model: string; configured: boolean; name?: string }>;
    ollama_models: string[];
    openrouter_key_set: boolean;
  };
}

export interface AppSettings {
  llm_provider: string;
  ollama_host: string;
  ollama_model: string;
  ollama_models: string[];
  llm_base_url: string;
  llm_model: string;
  openrouter_key_set: boolean;
  openrouter_key_hint: string;
  whisper_model: string;
  whisper_models: string[];
  encoder: string;
}

export interface ProviderInfo {
  id: string;
  name: string;
  kind: "local" | "cloud";
  about: string;
  key_url: string;
  key_prefix: string;
  base_url: string;
  base_url_editable: boolean;
  key_set: boolean;
  key_hint: string;
  models: string[];
  suggested: string[];
  free_only: boolean;
  configured: boolean;
}

export interface PullState {
  model: string;
  status: string;
  completed: number;
  total: number;
  error: string;
  done: boolean;
}

export interface ProvidersState {
  active: string;
  providers: ProviderInfo[];
  ollama_models: string[];
  ollama_picks: { model: string; size: string; note: string }[];
  pull: PullState;
}

async function req<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try {
      const j = await r.json();
      msg = j.detail || msg;
    } catch {
      /* not json */
    }
    throw new Error(msg);
  }
  return r.json();
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => req<Health>("/api/health"),
  presets: () => req<Presets>("/api/presets"),
  settings: () => req<AppSettings>("/api/settings"),
  saveSettings: (s: Partial<AppSettings> & { llm_api_key?: string }) =>
    req<AppSettings>("/api/settings", json("PUT", s)),
  testLLM: (provider: string, model?: string, deep = false) =>
    req<{ ok: boolean; detail: string; label: string }>("/api/settings/test", json("POST", { provider, model, deep })),
  providers: () => req<ProvidersState>("/api/providers"),
  setActiveProvider: (id: string) => req<ProvidersState>("/api/providers/active", json("PUT", { id })),
  saveProvider: (id: string, patch: { key?: string; base_url?: string; models?: string[]; free_only?: boolean; activate?: boolean }) =>
    req<ProvidersState>(`/api/providers/${id}`, json("PUT", patch)),
  providerModels: (id: string) => req<{ models: { id: string; free: boolean | null }[] }>(`/api/providers/${id}/models`),
  testProvider: (id: string, model?: string) =>
    req<{ ok: boolean; detail: string }>(`/api/providers/${id}/test`, json("POST", { model })),
  pullModel: (model: string) => req<PullState>("/api/ollama/pull", json("POST", { model })),
  pullState: () => req<PullState>("/api/ollama/pull"),
  projects: () => req<{ projects: ProjectSummary[] }>("/api/projects").then((r) => r.projects),
  project: (id: string, transcript = true) => req<Project>(`/api/projects/${id}?transcript=${transcript}`),
  deleteProject: (id: string) => req<{ deleted: string }>(`/api/projects/${id}`, { method: "DELETE" }),
  rerun: (id: string, options: Partial<Options>) =>
    req<{ id: string }>(`/api/projects/${id}/rerun`, json("POST", { options })),
  create: (fd: FormData) => req<{ id: string }>("/api/projects", { method: "POST", body: fd }),
  updateClip: (pid: string, cid: string, body: { edit?: Edit; title?: string; hashtags?: string[]; render?: boolean }) =>
    req<Clip>(`/api/projects/${pid}/clips/${cid}`, json("PUT", body)),
  renderClip: (pid: string, cid: string) =>
    req<Clip>(`/api/projects/${pid}/clips/${cid}/render`, { method: "POST" }),
  plan: (pid: string, cid: string, edit: Edit | null, signal?: AbortSignal) =>
    req<Plan>(`/api/projects/${pid}/clips/${cid}/plan`, { ...json("POST", { edit }), signal }),
  agentPlan: (text: string, has_file: boolean) =>
    req<{ request: AgentRequest; plan: AgentPlan }>("/api/agent/plan", json("POST", { text, has_file })),
  agentFollowup: (pid: string, text: string) =>
    req<{ action: "make" | "refind"; ids?: string[]; request?: AgentRequest; plan?: AgentPlan }>(
      "/api/agent/followup", json("POST", { pid, text })),
  finish: (pid: string, ids: string[] | null) => req<{ id: string; status: string }>(`/api/projects/${pid}/finish`, json("POST", { ids })),
  applyAll: (pid: string, edit: Partial<Edit>, render = true) =>
    req<{ ok: boolean }>(`/api/projects/${pid}/apply-all`, json("POST", { edit, render })),
};

export const media = (pid: string, path: string | null, version?: number) =>
  path ? `/media/${pid}/${path}${version !== undefined ? `?v=${version}` : ""}` : "";
