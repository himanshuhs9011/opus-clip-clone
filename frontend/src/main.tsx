import { StrictMode, useCallback, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { api, type Health, type Presets } from "./api";
import { Header } from "./components/Header";
import { SettingsModal } from "./components/SettingsModal";
import { useRoute } from "./lib/util";
import { Editor } from "./pages/Editor";
import { Home } from "./pages/Home";
import { ProjectPage } from "./pages/Project";
import "./styles.css";

function App() {
  const route = useRoute();
  const [health, setHealth] = useState<Health | null>(null);
  const [presets, setPresets] = useState<Presets | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [offline, setOffline] = useState(false);

  const loadHealth = useCallback(() => {
    api.health().then((h) => {
      setHealth(h);
      setOffline(false);
    }).catch(() => setOffline(true));
  }, []);

  useEffect(() => {
    loadHealth();
    api.presets().then(setPresets).catch(() => setOffline(true));
  }, [loadHealth]);

  let page: React.ReactNode;
  if (route[0] === "p" && route[1] && route[2] === "edit" && route[3]) {
    page = presets ? <Editor key={route[1] + route[3]} pid={route[1]} cid={route[3]} presets={presets} /> : null;
  } else if (route[0] === "p" && route[1]) {
    page = <ProjectPage key={route[1]} id={route[1]} presets={presets} health={health} />;
  } else {
    page = <Home presets={presets} health={health} />;
  }

  return (
    <>
      <Header health={health} onSettings={() => setSettingsOpen(true)} />
      {offline && (
        <div className="offline">
          Can't reach the ClipForge server. Start it with <code>serve.bat</code> (or <code>python -m uvicorn app:app --port 8000</code>).
        </div>
      )}
      <main>{page}</main>
      {settingsOpen && <SettingsModal onClose={() => setSettingsOpen(false)} onSaved={loadHealth} />}
    </>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
