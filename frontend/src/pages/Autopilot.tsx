// Autopilot: watch channels, clip every new video, post the clips on a daily schedule.
// Backend: opusclip/autopilot.py + opusclip/publish.py, /api/autopilot/*.
import { useCallback, useEffect, useState } from "react";
import {
  AlertTriangle, Check, ExternalLink, FolderOpen, Loader2, Plus, Power, RefreshCw, Send, Trash2, X,
} from "lucide-react";
import { api, type AutoItem, type AutoState } from "../api";
import { go, timeAgo, useInterval } from "../lib/util";

const PLATFORM: Record<AutoItem["platform"], string> = {
  youtube: "YouTube", instagram: "Instagram Reels", facebook: "Facebook Reels", phone: "Phone folder",
};
const CAPS: [keyof AutoState["per_day"], string][] = [
  ["youtube_short", "YouTube Shorts"], ["youtube_long", "YouTube long videos"], ["instagram", "Instagram Reels"],
  ["facebook", "Facebook Reels"], ["phone", "Phone folder (Snapchat, Moj, Josh)"],
];

export function Autopilot() {
  const [st, setSt] = useState<AutoState | null>(null);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState("");
  const [filter, setFilter] = useState<AutoItem["status"] | "all">("all");

  const load = useCallback(() => api.autopilot().then((s) => { setSt(s); setErr(""); }).catch((e) => setErr(e.message)), []);
  useEffect(() => { load(); }, [load]);
  useInterval(load, 5000);

  const act = async (key: string, fn: () => Promise<AutoState | unknown>) => {
    setBusy(key);
    setErr("");
    try {
      const r = await fn();
      if (r && typeof r === "object" && "queue" in (r as object)) setSt(r as AutoState);
      else await load();
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy("");
    }
  };

  if (!st) return <div className="page center muted">{err ? <span className="err-text">{err}</span> : <Loader2 className="spin" />}</div>;

  const counts = st.queue.reduce<Record<string, number>>((m, it) => ({ ...m, [it.status]: (m[it.status] ?? 0) + 1 }), {});
  const items = [...st.queue].reverse().filter((it) => filter === "all" || it.status === filter).slice(0, 80);
  const working = st.jobs.filter((j) => j.status === "running" || j.status === "queued");
  const anyAccount = st.accounts.youtube.connected || st.accounts.meta.connected;

  return (
    <div className="page autopilot">
      <div className="ap-head">
        <div className="grow">
          <h1>Autopilot</h1>
          <p className="muted">Watches your channels, clips every new video and posts the clips every day, on its own, while ClipForge is open.</p>
        </div>
        <button className={`btn big ${st.enabled ? "primary" : "ghost"}`} disabled={busy === "power"}
          onClick={() => act("power", () => api.autopilotUpdate({ enabled: !st.enabled }))}>
          <Power size={18} /> {st.enabled ? "On" : "Off"}
        </button>
      </div>

      <div className="ap-status card">
        <span><b>{st.enabled ? "Running" : "Stopped"}</b></span>
        <span className="muted small">Last check {st.last_check ? timeAgo(st.last_check) : "never"} · every {st.check_minutes} min</span>
        <span className="muted small">Next posting {st.next_post || "—"}</span>
        <span className="muted small">Today: {CAPS.map(([k, label]) => `${label.split(" (")[0]} ${st.today[k] ?? 0}/${st.per_day[k]}`).join(" · ")}</span>
        <div className="row gap8">
          <button className="btn ghost small-btn" disabled={!!busy} onClick={() => act("check", api.autopilotCheck)}>
            <RefreshCw size={13} /> Check now
          </button>
          <button className="btn ghost small-btn" disabled={!!busy || !counts.waiting} onClick={() => act("post", api.autopilotPostNow)}>
            <Send size={13} /> Post now
          </button>
        </div>
      </div>
      {err && <div className="err-text"><AlertTriangle size={14} /> {err}</div>}
      <div className="ap-note small">
        <AlertTriangle size={14} /> Only clip channels you own or have permission for (for example a clipping program).
        Re-uploading other people's videos gets copyright strikes and Content ID claims.
      </div>

      <div className="ap-grid">
        <Sources st={st} busy={busy} act={act} />
        <Accounts st={st} busy={busy} act={act} />
        <Schedule st={st} busy={busy} act={act} />
      </div>

      {working.length > 0 && (
        <section className="card ap-card">
          <h2>Clipping now</h2>
          {working.map((j) => (
            <div key={j.pid} className="row gap8 small">
              <Loader2 size={13} className="spin" /> <b>{j.source_name}</b> {j.title}
              <button className="link small" onClick={() => go(`/p/${j.pid}`)}>Open project</button>
            </div>
          ))}
        </section>
      )}

      <section className="card ap-card">
        <div className="section-head">
          <h2>Posts</h2>
          <div className="seg small">
            {(["all", "review", "waiting", "posted", "failed", "skipped"] as const).map((f) => (
              <button key={f} className={filter === f ? "sel" : ""} onClick={() => setFilter(f)}>
                {f[0].toUpperCase() + f.slice(1)} {f === "all" ? st.queue.length : counts[f] ?? 0}
              </button>
            ))}
          </div>
        </div>
        {!anyAccount && <p className="muted small">Connect YouTube or Instagram / Facebook above. Until then clips only go to the phone folder.</p>}
        {!items.length && <p className="muted small">Nothing here yet. Add a channel, turn Autopilot on, and clips show up here after the next new video.</p>}
        <div className="ap-queue">
          {items.map((it) => (
            <div key={it.id} className={`ap-item st-${it.status}`}>
              <span className="ap-plat">{PLATFORM[it.platform]}{it.kind === "long" ? " · long" : ""}</span>
              <span className="grow ap-title" title={it.title}>{it.title}</span>
              <span className="muted small mono">{it.score}</span>
              <span className={`ap-badge st-${it.status}`}>{it.status}</span>
              {it.url && (it.url.startsWith("http")
                ? <a className="icon-btn" href={it.url} target="_blank" rel="noreferrer" title="Open"><ExternalLink size={14} /></a>
                : <span className="icon-btn" title={it.url}><FolderOpen size={14} /></span>)}
              {it.status === "review" && (
                <button className="icon-btn" title="Approve" onClick={() => act(it.id, () => api.autopilotItem(it.id, "approve"))}><Check size={14} /></button>
              )}
              {it.status === "failed" && (
                <button className="icon-btn" title={`Retry: ${it.error}`} onClick={() => act(it.id, () => api.autopilotItem(it.id, "retry"))}><RefreshCw size={14} /></button>
              )}
              {["review", "waiting", "failed"].includes(it.status) && (
                <button className="icon-btn" title="Skip" onClick={() => act(it.id, () => api.autopilotItem(it.id, "skip"))}><X size={14} /></button>
              )}
              <button className="icon-btn" title="Open the project" onClick={() => go(`/p/${it.pid}/edit/${it.cid}`)}><ExternalLink size={14} /></button>
              {it.error && it.status === "failed" && <div className="err-text small ap-err">{it.error}</div>}
            </div>
          ))}
        </div>
      </section>

      <section className="card ap-card">
        <h2>Activity</h2>
        <div className="ap-log">
          {[...st.log].reverse().map(([t, m], i) => (
            <div key={i}><span className="muted mono">{new Date(t * 1000).toLocaleString()}</span> {m}</div>
          ))}
          {!st.log.length && <span className="muted small">Nothing yet.</span>}
        </div>
      </section>
    </div>
  );
}

type Act = (key: string, fn: () => Promise<AutoState | unknown>) => Promise<void>;

function Sources({ st, busy, act }: { st: AutoState; busy: string; act: Act }) {
  const [url, setUrl] = useState("");
  const [format, setFormat] = useState<"short" | "long" | "both">("both");
  const [focus, setFocus] = useState("");
  const [backfill, setBackfill] = useState(1);
  return (
    <section className="card ap-card">
      <h2>Channels to watch</h2>
      {st.sources.map((s) => (
        <div key={s.id} className="ap-source">
          <div className="grow">
            <b>{s.name}</b>
            <div className="muted small ap-url">{s.url}</div>
            <div className="muted small">{s.format === "both" ? "Shorts + long videos" : s.format === "long" ? "Long videos" : "Shorts"}{s.focus ? ` · only: ${s.focus}` : ""}</div>
          </div>
          <button className="icon-btn danger" title="Stop watching" onClick={() => act(`rm${s.id}`, () => api.autopilotRemoveSource(s.id))}><Trash2 size={14} /></button>
        </div>
      ))}
      <div className="ap-form">
        <input placeholder="https://www.youtube.com/@channel  or a Kick / Twitch videos page" value={url} onChange={(e) => setUrl(e.target.value)} />
        <div className="seg small">
          {(["short", "long", "both"] as const).map((f) => (
            <button key={f} className={format === f ? "sel" : ""} onClick={() => setFormat(f)}>{f === "both" ? "Both" : f === "long" ? "Long videos" : "Shorts"}</button>
          ))}
        </div>
        <input placeholder="Only these moments (optional), e.g. funny moments" value={focus} onChange={(e) => setFocus(e.target.value)} />
        <label className="small muted row gap8">
          Also clip the latest
          <input type="number" min={0} max={5} value={backfill} onChange={(e) => setBackfill(Number(e.target.value))} className="ap-num" />
          video(s) now
        </label>
        <button className="btn primary" disabled={!url.trim() || busy === "add"}
          onClick={() => act("add", async () => { const r = await api.autopilotAddSource({ url, format, focus, backfill }); setUrl(""); setFocus(""); return r; })}>
          {busy === "add" ? <Loader2 size={15} className="spin" /> : <Plus size={15} />} Watch this channel
        </button>
      </div>
    </section>
  );
}

function Accounts({ st, busy, act }: { st: AutoState; busy: string; act: Act }) {
  const yt = st.accounts.youtube;
  const meta = st.accounts.meta;
  const [cid, setCid] = useState("");
  const [secret, setSecret] = useState("");
  const [appId, setAppId] = useState("");
  const [appSecret, setAppSecret] = useState("");
  const [token, setToken] = useState("");
  const [redirect, setRedirect] = useState("");

  return (
    <section className="card ap-card">
      <h2>Accounts</h2>
      <div className="ap-acc">
        <div className="row gap8"><b className="grow">YouTube</b>
          {yt.connected ? <span className="ok-text small"><Check size={13} /> {yt.channel}</span> : <span className="muted small">not connected</span>}
        </div>
        {!yt.connected && (
          <>
            <p className="muted small">In Google Cloud: create a project, enable <b>YouTube Data API v3</b>, then an OAuth client of type <b>Desktop app</b>. Paste its ID and secret, then sign in with the Google account that owns the channel.</p>
            <input placeholder="OAuth client ID" value={cid || yt.client_id} onChange={(e) => setCid(e.target.value)} />
            <input placeholder={yt.has_secret ? "Client secret (saved)" : "Client secret"} type="password" value={secret} onChange={(e) => setSecret(e.target.value)} />
            <button className="btn primary" disabled={busy === "yt"} onClick={() => act("yt", async () => {
              const r = await api.autopilotYoutube(cid || yt.client_id, secret);
              setRedirect(r.redirect_uri);
              window.open(r.auth_url, "_blank");
              return null;
            })}>Sign in with Google</button>
            {redirect && <p className="muted small">After signing in, this page updates by itself. Sign-in address: <code>{redirect}</code></p>}
          </>
        )}
        {yt.connected && <button className="link small" onClick={() => act("ytx", () => api.autopilotDisconnect("youtube"))}>Disconnect</button>}
        <p className="muted small">YouTube allows about 6 uploads a day on a new project, and keeps API uploads private until the project passes YouTube's API audit.</p>
      </div>

      <div className="ap-acc">
        <div className="row gap8"><b className="grow">Instagram + Facebook</b>
          {meta.connected ? <span className="ok-text small"><Check size={13} /> {meta.page_name}{meta.ig_username ? ` · @${meta.ig_username}` : ""}</span> : <span className="muted small">not connected</span>}
        </div>
        {!meta.connected ? (
          <>
            <p className="muted small">Needs an Instagram <b>Business or Creator</b> account linked to a Facebook Page, and a Meta app (developers.facebook.com). In Graph API Explorer pick your app and get a user token with <code>pages_show_list, pages_read_engagement, pages_manage_posts, instagram_basic, instagram_content_publish, business_management</code>.</p>
            <input placeholder="Meta app ID" value={appId || meta.app_id} onChange={(e) => setAppId(e.target.value)} />
            <input placeholder={meta.has_secret ? "App secret (saved)" : "App secret"} type="password" value={appSecret} onChange={(e) => setAppSecret(e.target.value)} />
            <input placeholder="User access token from Graph API Explorer" type="password" value={token} onChange={(e) => setToken(e.target.value)} />
            <button className="btn primary" disabled={busy === "meta" || !token} onClick={() => act("meta", () => api.autopilotMeta(appId || meta.app_id, appSecret, token))}>Connect</button>
          </>
        ) : (
          <>
            {meta.pages.length > 1 && (
              <select value={meta.page_id} onChange={(e) => act("page", () => api.autopilotMetaPage(e.target.value))}>
                {meta.pages.map((p) => <option key={p.id} value={p.id}>{p.name}{p.ig_username ? ` (@${p.ig_username})` : " (no Instagram)"}</option>)}
              </select>
            )}
            <button className="link small" onClick={() => act("metax", () => api.autopilotDisconnect("meta"))}>Disconnect</button>
          </>
        )}
      </div>

      <div className="ap-acc">
        <b>Snapchat Spotlight, Moj, Josh</b>
        <p className="muted small">These have no upload API. Clips for them are copied, with their caption, to <code>{st.phone_dir}</code>, one folder per day, ready to post from your phone. TikTok is left out (not available in India).</p>
      </div>
    </section>
  );
}

function Schedule({ st, act }: { st: AutoState; busy: string; act: Act }) {
  const [times, setTimes] = useState(st.post_times.join(", "));
  useEffect(() => setTimes(st.post_times.join(", ")), [st.post_times]);
  const save = (v: Partial<AutoState>) => act("sched", () => api.autopilotUpdate(v));
  return (
    <section className="card ap-card">
      <h2>Schedule</h2>
      <label className="ap-field">Posting times
        <input value={times} onChange={(e) => setTimes(e.target.value)}
          onBlur={() => save({ post_times: times.split(/[ ,]+/).filter(Boolean) })} placeholder="11:00, 15:00, 19:00" />
      </label>
      <div className="ap-caps">
        {CAPS.map(([k, label]) => (
          <label key={k} className="ap-field row gap8">
            <span className="grow">{label}</span>
            <input type="number" min={0} max={50} className="ap-num" value={st.per_day[k]}
              onChange={(e) => save({ per_day: { ...st.per_day, [k]: Number(e.target.value) } })} />
            <span className="muted small">a day</span>
          </label>
        ))}
      </div>
      <label className="ap-field row gap8"><span className="grow">Only post clips scoring</span>
        <input type="number" min={0} max={100} className="ap-num" value={st.min_score} onChange={(e) => save({ min_score: Number(e.target.value) })} />+
      </label>
      <label className="ap-field row gap8"><span className="grow">Check channels every</span>
        <input type="number" min={15} max={1440} className="ap-num" value={st.check_minutes} onChange={(e) => save({ check_minutes: Number(e.target.value) })} /> min
      </label>
      <label className="ap-field row gap8"><span className="grow">YouTube visibility</span>
        <select value={st.youtube_privacy} onChange={(e) => save({ youtube_privacy: e.target.value as AutoState["youtube_privacy"] })}>
          <option value="public">Public</option><option value="unlisted">Unlisted</option><option value="private">Private</option>
        </select>
      </label>
      <label className="ap-check"><input type="checkbox" checked={st.review} onChange={(e) => save({ review: e.target.checked })} />
        Let me approve each clip before it is posted</label>
      <label className="ap-check"><input type="checkbox" checked={st.credit} onChange={(e) => save({ credit: e.target.checked })} />
        Credit the source channel in captions</label>
    </section>
  );
}
