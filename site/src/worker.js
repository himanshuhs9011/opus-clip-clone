// ClipForge site: static pages from ./public, plus /download which streams the newest
// ClipForge-Setup.exe from the GitHub release so visitors download it straight from this site.
// GitHub's "latest release" address, so a new release needs no change here. (Pinning a tag
// broke the button when that release was replaced: GitHub answered 404, visitors got 502.)
// The release asset must keep the name ClipForge-Setup.exe.
const RELEASE_FILE =
  "https://github.com/himanshuhs9011/opus-clip-clone/releases/latest/download/ClipForge-Setup.exe";
const CACHE_SECONDS = 600; // a new release reaches the site within 10 minutes

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/download" || url.pathname === "/ClipForge-Setup.exe") {
      const upstream = await fetch(RELEASE_FILE, {
        method: request.method === "HEAD" ? "HEAD" : "GET",
        redirect: "follow",
        cf: { cacheTtl: CACHE_SECONDS, cacheEverything: true },
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
        "cache-control": `public, max-age=${CACHE_SECONDS}`,
      });
      const length = upstream.headers.get("content-length");
      if (length) headers.set("content-length", length);
      return new Response(upstream.body, { status: 200, headers });
    }
    return env.ASSETS.fetch(request);
  },
};
