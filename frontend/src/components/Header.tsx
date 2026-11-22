import { Cpu, Settings, Sparkles } from "lucide-react";
import type { Health } from "../api";
import { go } from "../lib/util";

const PROVIDER_LABEL: Record<string, string> = {
  ollama: "Ollama",
  openrouter: "OpenRouter",
  openai: "OpenAI-compatible",
  none: "Rules only",
};

export function Header({ health, onSettings }: { health: Health | null; onSettings: () => void }) {
  const prov = health?.llm.provider ?? "";
  const st = health?.llm.providers[prov];
  const llmOk = prov === "none" || !!st?.ok;
  return (
    <header className="topbar">
      <a className="brand" href="#/" onClick={() => go("/")}>
        <span className="logo">
          <Sparkles size={16} />
        </span>
        ClipForge
      </a>
      <nav className="top-nav">
        <a href="#/">Projects</a>
        <a href="#/agent">Agent</a>
        <a href="#/autopilot">Autopilot</a>
      </nav>
      <div className="top-right">
        {health && (
          <>
            <span className="chip" title={st?.detail ?? ""}>
              <span className={`dot ${llmOk ? "ok" : "warn"}`} />
              {st?.name ?? PROVIDER_LABEL[prov] ?? prov}
              {prov !== "none" && st?.model ? <em>{st.model}</em> : null}
            </span>
            <span className="chip" title="Video encoder">
              <Cpu size={13} />
              {health.encoder}
            </span>
          </>
        )}
        <button className="icon-btn" onClick={onSettings} title="Settings">
          <Settings size={18} />
        </button>
      </div>
    </header>
  );
}
