import { forwardRef, useCallback, useEffect, useImperativeHandle, useLayoutEffect, useRef, useState } from "react";
import { Pause, Play, Repeat, Volume2, VolumeX } from "lucide-react";
import type { Plan } from "../api";
import { captionAt, cropAt, interp, nextPlayable, toOutput, toSource, type CaptionState } from "../lib/playback";
import { CSS_FILTERS, fmtTime } from "../lib/util";
import { CaptionOverlay } from "./CaptionOverlay";

export interface PreviewHandle {
  seekSource: (absTime: number) => void;
  seekOutput: (t: number) => void;
  toggle: () => void;
}

interface Props {
  plan: Plan;
  onTime?: (outT: number, absSource: number) => void;
  maxHeight?: number;
}

/**
 * Live, render-free preview: plays the source video and applies the crop path, jump cuts,
 * zoom, colour filter, captions, emoji, title and progress bar per frame.
 */
export const Preview = forwardRef<PreviewHandle, Props>(function Preview({ plan, onTime, maxHeight = 640 }, ref) {
  const stageRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const zoomRef = useRef<HTMLDivElement>(null);
  const bgRef = useRef<HTMLCanvasElement>(null);
  const barRef = useRef<HTMLDivElement>(null);
  const splitRef = useRef<HTMLDivElement>(null);
  const streamRef = useRef<HTMLDivElement>(null);
  const planRef = useRef(plan);
  planRef.current = plan;
  const onTimeRef = useRef(onTime);
  onTimeRef.current = onTime;

  const [size, setSize] = useState({ w: 360, h: 640 });
  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(false);
  const [loop, setLoop] = useState(true);
  const [outT, setOutT] = useState(0);
  const [cap, setCap] = useState<CaptionState>({ cue: -1, active: -1, progress: 0 });
  const [ready, setReady] = useState(false);
  const loopRef = useRef(loop);
  loopRef.current = loop;

  // ------------------------------------------------------------ stage size
  const ar = plan.out.w / plan.out.h;
  useLayoutEffect(() => {
    const el = stageRef.current?.parentElement;
    if (!el) return;
    const fit = () => {
      const availW = el.clientWidth;
      const availH = Math.min(maxHeight, window.innerHeight - 230);
      let h = Math.max(260, availH);
      let w = h * ar;
      if (w > availW) {
        w = availW;
        h = w / ar;
      }
      setSize({ w: Math.round(w), h: Math.round(h) });
    };
    fit();
    const ro = new ResizeObserver(fit);
    ro.observe(el);
    window.addEventListener("resize", fit);
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", fit);
    };
  }, [ar, maxHeight]);

  // ------------------------------------------------------------ per-frame update
  const capRef = useRef(cap);
  capRef.current = cap;
  const lastOut = useRef(-1);

  const frame = useCallback(() => {
    const v = videoRef.current;
    const p = planRef.current;
    if (!v || !p) return;
    // link projects play a downloaded piece: file time 0 = source time video_offset
    const off = p.video_offset ?? 0;
    const va = p.a - off;
    let rel = v.currentTime - va;
    const jump = nextPlayable(rel, p.segs);
    if (jump === -1) {
      if (loopRef.current) {
        v.currentTime = va + p.segs[0][0];
      } else {
        v.pause();
        v.currentTime = va + p.segs[p.segs.length - 1][1] - 0.04;
      }
      return;
    }
    if (jump !== null) {
      v.currentTime = va + jump;
      rel = jump;
    }
    const t = toOutput(rel, p.segs) ?? 0;

    // geometry
    const W = size.w;
    const H = size.h;
    // Auto shows scenes with nobody on screen (screen share, replays) whole over a blur
    const fitNow = p.layout === "fit" || (p.fit_ranges ?? []).some(([s, e]) => rel >= s && rel <= e);
    if (bgRef.current) bgRef.current.style.display = fitNow ? "block" : "none";
    if (!fitNow && p.crop.axis !== "none") {
      const k = W / p.crop.w;
      const { x, y } = cropAt(p, rel);
      v.style.width = `${p.source.w * k}px`;
      v.style.height = `${p.source.h * k}px`;
      v.style.transform = `translate(${-x * k}px, ${-y * k}px)`;
      v.style.top = "0";
    } else if (fitNow && p.crop.axis !== "none") {
      const s = Math.min(W / p.source.w, H / p.source.h);
      v.style.width = `${p.source.w * s}px`;
      v.style.height = `${p.source.h * s}px`;
      v.style.transform = `translate(${(W - p.source.w * s) / 2}px, ${(H - p.source.h * s) / 2}px)`;
      const bg = bgRef.current;
      if (bg && v.readyState >= 2) {
        const ctx = bg.getContext("2d");
        if (ctx) {
          const cs = Math.max(bg.width / p.source.w, bg.height / p.source.h);
          const dw = p.source.w * cs;
          const dh = p.source.h * cs;
          ctx.drawImage(v, (bg.width - dw) / 2, (bg.height - dh) / 2, dw, dh);
        }
      }
    } else {
      v.style.width = `${W}px`;
      v.style.height = `${H}px`;
      v.style.transform = "none";
    }
    // split screen: draw both people from the playing video onto two canvases
    const sc = p.layout === "split" ? p.split.find((s) => rel >= s.t0 && rel <= s.t1) : undefined;
    const sp = splitRef.current;
    if (sp) {
      sp.style.display = sc ? "block" : "none";
      if (sc && v.readyState >= 2 && v.videoWidth) {
        const vf = v.videoWidth / p.source.w;
        const { w: cw, h: ch } = p.split_crop;
        (["top", "bottom"] as const).forEach((half, i) => {
          const cv = sp.children[i] as HTMLCanvasElement;
          const ctx = cv.getContext("2d");
          if (!ctx) return;
          const cx = interp(sc[half].x, rel);
          const cy = interp(sc[half].y, rel);
          const x = Math.min(Math.max(cx * p.source.w - cw / 2, 0), p.source.w - cw);
          const y = Math.min(Math.max(cy * p.source.h - ch * 0.42, 0), p.source.h - ch);
          ctx.drawImage(v, x * vf, y * vf, cw * vf, ch * vf, 0, 0, cv.width, cv.height);
        });
      }
    }

    // gaming layout: face-cam panel + gameplay panel
    const gs = p.stream;
    const streamNow = !!gs && (p.layout === "gaming" || (p.stream_ranges ?? []).some(([s, e]) => rel >= s && rel <= e));
    const sr = streamRef.current;
    if (sr) {
      sr.style.display = streamNow ? "block" : "none";
      if (streamNow && gs && v.readyState >= 2 && v.videoWidth) {
        const vf = v.videoWidth / p.source.w;
        const [cam, game] = sr.children as unknown as HTMLCanvasElement[];
        const cctx = cam.getContext("2d");
        const [cx, cy, cw, ch] = gs.cam_crop;
        cctx?.drawImage(v, cx * vf, cy * vf, cw * vf, ch * vf, 0, 0, cam.width, cam.height);
        const gctx = game.getContext("2d");
        if (gctx) {
          if (gs.game === "fit") {
            const cover = Math.max(game.width / v.videoWidth, game.height / v.videoHeight);
            const fit = Math.min(game.width / v.videoWidth, game.height / v.videoHeight);
            gctx.filter = "blur(10px) brightness(0.85)";
            gctx.drawImage(v, (game.width - v.videoWidth * cover) / 2, (game.height - v.videoHeight * cover) / 2,
              v.videoWidth * cover, v.videoHeight * cover);
            gctx.filter = "none";
            gctx.drawImage(v, (game.width - v.videoWidth * fit) / 2, (game.height - v.videoHeight * fit) / 2,
              v.videoWidth * fit, v.videoHeight * fit);
          } else {
            const [gx, gy, gw, gh] = gs.game_crop;
            gctx.drawImage(v, gx * vf, gy * vf, gw * vf, gh * vf, 0, 0, game.width, game.height);
          }
        }
      }
    }

    const z = fitNow || streamNow ? 1 : interp(p.zoom, t, 1);
    if (zoomRef.current) zoomRef.current.style.transform = `scale(${z})`;
    if (barRef.current) barRef.current.style.width = `${Math.min(100, (t / Math.max(0.1, p.duration)) * 100)}%`;

    const c = captionAt(p, t);
    const prev = capRef.current;
    const karaoke = p.preset.anim === "karaoke";
    if (c.cue !== prev.cue || c.active !== prev.active || (karaoke && Math.abs(c.progress - prev.progress) > 0.04)) {
      setCap(c);
    }
    if (Math.abs(t - lastOut.current) > 0.08) {
      lastOut.current = t;
      setOutT(t);
      onTimeRef.current?.(t, v.currentTime + off);
    }
  }, [size]);

  useEffect(() => {
    let raf = 0;
    const tick = () => {
      frame();
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [frame]);

  // new clip range -> park at the start
  const rangeKey = `${plan.video}|${plan.a.toFixed(2)}`;
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const park = () => {
      const off = plan.video_offset ?? 0;
      if (v.currentTime < plan.a - off || v.currentTime > plan.b - off) v.currentTime = plan.a - off + (plan.segs[0]?.[0] ?? 0);
    };
    if (v.readyState >= 1) park();
    else v.addEventListener("loadedmetadata", park, { once: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rangeKey]);

  const seekOutput = useCallback((t: number) => {
    const v = videoRef.current;
    const p = planRef.current;
    if (!v) return;
    v.currentTime = p.a - (p.video_offset ?? 0) + toSource(Math.max(0, Math.min(p.duration - 0.05, t)), p.segs);
  }, []);

  const toggle = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) void v.play();
    else v.pause();
  }, []);

  useImperativeHandle(ref, () => ({
    seekSource: (abs: number) => {
      if (videoRef.current) videoRef.current.currentTime = Math.max(0, abs - (planRef.current.video_offset ?? 0));
    },
    seekOutput,
    toggle,
  }), [seekOutput, toggle]);

  const filter = CSS_FILTERS[plan.edit.filter] ?? CSS_FILTERS.none;
  const k = size.w / plan.out.w;

  return (
    <div className="preview">
      <div
        ref={stageRef}
        className="stage"
        style={{ width: size.w, height: size.h }}
        onClick={toggle}
      >
        {(plan.layout === "fit" || (plan.fit_ranges ?? []).length > 0) && plan.crop.axis !== "none" && (
          <canvas ref={bgRef} className="stage-bg" width={Math.max(8, Math.round(size.w / 8))} height={Math.max(8, Math.round(size.h / 8))} style={{ filter: `${filter} blur(6px) brightness(0.86)` }} />
        )}
        <div ref={zoomRef} className="stage-zoom" style={{ filter }}>
          {plan.video ? (
            <video
              ref={videoRef}
              className="stage-video"
              src={plan.video}
              preload="auto"
              playsInline
              muted={muted}
              onPlay={() => setPlaying(true)}
              onPause={() => setPlaying(false)}
              onLoadedData={() => setReady(true)}
            />
          ) : (
            <div className="stage-missing">Source video missing</div>
          )}
        </div>
        {plan.layout === "split" && (
          <div ref={splitRef} className="stage-split" style={{ filter, display: "none" }}>
            <canvas width={Math.round(size.w * 2)} height={Math.round(size.h)} />
            <canvas width={Math.round(size.w * 2)} height={Math.round(size.h)} />
          </div>
        )}
        {plan.stream && (
          <div ref={streamRef} className="stage-stream" style={{ filter, display: "none" }}>
            <canvas
              width={Math.round(size.w * 2)}
              height={Math.round((size.h * 2 * plan.stream.cam_h) / plan.out.h)}
              style={{ top: plan.stream.cam_pos === "top" ? 0 : `${(plan.stream.game_h / plan.out.h) * 100}%`, height: `${(plan.stream.cam_h / plan.out.h) * 100}%` }}
            />
            <canvas
              width={Math.round(size.w * 2)}
              height={Math.round((size.h * 2 * plan.stream.game_h) / plan.out.h)}
              style={{ top: plan.stream.cam_pos === "top" ? `${(plan.stream.cam_h / plan.out.h) * 100}%` : 0, height: `${(plan.stream.game_h / plan.out.h) * 100}%` }}
            />
          </div>
        )}
        <CaptionOverlay plan={plan} cap={cap} t={outT} k={k} />
        {plan.edit.progress.enabled && (
          <div
            className="stage-bar"
            style={{
              [plan.edit.progress.position === "bottom" ? "bottom" : "top"]: 0,
              height: Math.max(3, 12 * plan.scale * k),
            }}
          >
            <div ref={barRef} style={{ background: plan.edit.progress.color }} />
          </div>
        )}
        {!playing && ready && (
          <div className="stage-play">
            <Play size={30} fill="currentColor" />
          </div>
        )}
        {!plan.tracked && <div className="stage-note">Face path will be re-tracked on export</div>}
        {plan.partial && <div className="stage-note">Trimmed past the downloaded part: export fetches the rest</div>}
      </div>

      <div className="transport" style={{ width: size.w }}>
        <button className="icon-btn" onClick={toggle} title="Play / pause (space)">
          {playing ? <Pause size={18} /> : <Play size={18} />}
        </button>
        <input
          className="scrub"
          type="range"
          min={0}
          max={plan.duration}
          step={0.01}
          value={Math.min(outT, plan.duration)}
          onChange={(e) => seekOutput(parseFloat(e.target.value))}
        />
        <span className="mono small">
          {fmtTime(outT, true)} / {fmtTime(plan.duration, true)}
        </span>
        <button className={`icon-btn ${loop ? "on" : ""}`} onClick={() => setLoop(!loop)} title="Loop">
          <Repeat size={16} />
        </button>
        <button className="icon-btn" onClick={() => setMuted(!muted)} title="Mute">
          {muted ? <VolumeX size={16} /> : <Volume2 size={16} />}
        </button>
      </div>
    </div>
  );
});
