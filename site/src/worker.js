// ClipForge site: static pages from ./public, plus /download which streams the newest
// ClipForge-Setup.exe from the GitHub release so visitors download it straight from this site.
const RELEASE_FILE =
  "https://github.com/himanshuhs9011/opus-clip-clone/releases/latest/download/ClipForge-Setup.exe";

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/download" || url.pathname === "/ClipForge-Setup.exe") {
      const upstream = await fetch(RELEASE_FILE, {
        method: request.method === "HEAD" ? "HEAD" : "GET",
        redirect: "follow",
        cf: { cacheTtl: 3600, cacheEverything: true },
      });
      if (!upstream.ok) {
        return new Response("The installer is temporarily unavailable. Please try again in a minute.", {
          status: 502,
          headers: { "content-type": "text/plain; charset=utf-8" },
        });
      }
      const headers = new Headers({
        "content-type": "application/octet-stream",
        "content-disposition": 'attachment; filename="ClipForge-Setup.exe"',
        "cache-control": "public, max-age=3600",
      });
      const length = upstream.headers.get("content-length");
      if (length) headers.set("content-length", length);
      return new Response(upstream.body, { status: 200, headers });
    }
    return env.ASSETS.fetch(request);
  },
};
