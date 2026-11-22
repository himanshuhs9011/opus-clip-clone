"""Posting finished clips through the platforms' official APIs.

* YouTube (Shorts and long videos): YouTube Data API v3, resumable upload. The account is
  connected once with Google sign-in (OAuth, a "Desktop app" client the user creates in Google
  Cloud). Only a refresh token is stored; the password never touches ClipForge.
  Limits: an upload costs ~1,600 of the default 10,000 daily quota units (about 6 a day), and
  until the Google Cloud project passes YouTube's API audit, uploaded videos stay private.
* Instagram Reels and Facebook Reels: Meta Graph API, resumable upload. Needs an Instagram
  Business / Creator account linked to a Facebook Page, and a Meta app. The user's token is
  swapped for a Page token that does not expire.
* Snapchat Spotlight, Moj and Josh have no public upload API: those clips go into a
  "post from your phone" folder with their captions instead.

Every call here is plain HTTP (httpx) so it can be tested with a mock transport.
"""
from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import urlencode

import httpx

GRAPH = "https://graph.facebook.com/v24.0"
YT_UPLOAD = "https://www.googleapis.com/upload/youtube/v3/videos"
YT_API = "https://www.googleapis.com/youtube/v3"
GOOGLE_TOKEN = "https://oauth2.googleapis.com/token"
GOOGLE_AUTH = "https://accounts.google.com/o/oauth2/v2/auth"
YT_SCOPES = "https://www.googleapis.com/auth/youtube.upload https://www.googleapis.com/auth/youtube.readonly"

_client: httpx.Client | None = None


class PublishError(RuntimeError):
    pass


def client() -> httpx.Client:
    """One shared HTTP client (tests swap it for one with a mock transport)."""
    global _client
    if _client is None:
        _client = httpx.Client(timeout=httpx.Timeout(60.0, read=600.0, write=600.0))
    return _client


def set_client(c: httpx.Client | None) -> None:
    global _client
    _client = c


def _check(r: httpx.Response, what: str) -> dict:
    try:
        data = r.json() if r.content else {}
    except ValueError:
        data = {}
    if r.status_code >= 300:
        err = data.get("error") if isinstance(data, dict) else None
        msg = (err.get("message") if isinstance(err, dict) else err) or r.text[:300]
        raise PublishError(f"{what}: HTTP {r.status_code} {msg}")
    return data if isinstance(data, dict) else {}


# --------------------------------------------------------------------------
# YouTube
# --------------------------------------------------------------------------
def youtube_auth_url(client_id: str, redirect_uri: str, state: str) -> str:
    return GOOGLE_AUTH + "?" + urlencode({
        "client_id": client_id, "redirect_uri": redirect_uri, "response_type": "code",
        "scope": YT_SCOPES, "access_type": "offline", "prompt": "consent", "state": state,
    })


def youtube_exchange(client_id: str, client_secret: str, code: str, redirect_uri: str) -> dict:
    data = _check(client().post(GOOGLE_TOKEN, data={
        "client_id": client_id, "client_secret": client_secret, "code": code,
        "redirect_uri": redirect_uri, "grant_type": "authorization_code"}), "Google sign-in")
    if not data.get("refresh_token"):
        raise PublishError("Google did not return a refresh token - remove ClipForge from your "
                           "Google account's third-party access and connect again")
    return data


def youtube_access_token(acc: dict) -> str:
    data = _check(client().post(GOOGLE_TOKEN, data={
        "client_id": acc["client_id"], "client_secret": acc["client_secret"],
        "refresh_token": acc["refresh_token"], "grant_type": "refresh_token"}), "Google token")
    return data["access_token"]


def youtube_channel(token: str) -> dict:
    data = _check(client().get(f"{YT_API}/channels", params={"part": "snippet", "mine": "true"},
                               headers={"Authorization": f"Bearer {token}"}), "YouTube channel")
    items = data.get("items") or []
    if not items:
        raise PublishError("this Google account has no YouTube channel")
    return {"id": items[0]["id"], "title": items[0]["snippet"]["title"]}


def youtube_upload(acc: dict, file: Path, title: str, description: str, tags: list[str],
                   privacy: str = "public", category: str = "20") -> dict:
    """Resumable upload; returns {"id", "url"}."""
    token = youtube_access_token(acc)
    size = file.stat().st_size
    meta = {"snippet": {"title": title[:100], "description": description[:4900],
                        "tags": [t.lstrip("#")[:30] for t in tags][:15], "categoryId": category},
            "status": {"privacyStatus": privacy, "selfDeclaredMadeForKids": False}}
    r = client().post(YT_UPLOAD, params={"uploadType": "resumable", "part": "snippet,status"},
                      json=meta, headers={"Authorization": f"Bearer {token}",
                                          "X-Upload-Content-Type": "video/mp4",
                                          "X-Upload-Content-Length": str(size)})
    _check(r, "YouTube upload start")
    session = r.headers.get("location")
    if not session:
        raise PublishError("YouTube did not return an upload address")
    with file.open("rb") as fh:
        r = client().put(session, content=fh, headers={"Content-Type": "video/mp4",
                                                         "Content-Length": str(size)})
    data = _check(r, "YouTube upload")
    vid = data.get("id")
    if not vid:
        raise PublishError("YouTube upload finished without a video id")
    return {"id": vid, "url": f"https://youtu.be/{vid}"}


# --------------------------------------------------------------------------
# Meta (Instagram + Facebook)
# --------------------------------------------------------------------------
def meta_connect(app_id: str, app_secret: str, user_token: str) -> dict:
    """Short-lived user token -> long-lived one -> the Pages it manages (Page tokens made from a
    long-lived user token do not expire) and each Page's linked Instagram account."""
    data = _check(client().get(f"{GRAPH}/oauth/access_token", params={
        "grant_type": "fb_exchange_token", "client_id": app_id, "client_secret": app_secret,
        "fb_exchange_token": user_token}), "Meta token exchange")
    long_user = data.get("access_token") or user_token
    pages = _check(client().get(f"{GRAPH}/me/accounts", params={
        "fields": "id,name,access_token,instagram_business_account{id,username}",
        "access_token": long_user}), "Facebook Pages").get("data") or []
    out = []
    for pg in pages:
        ig = pg.get("instagram_business_account") or {}
        out.append({"id": pg["id"], "name": pg.get("name", ""), "token": pg.get("access_token", ""),
                    "ig_id": ig.get("id", ""), "ig_username": ig.get("username", "")})
    if not out:
        raise PublishError("this Facebook account manages no Pages (Instagram posting needs a Page "
                           "linked to an Instagram Business or Creator account)")
    return {"pages": out}


def _rupload(url: str, token: str, file: Path, what: str) -> dict:
    size = file.stat().st_size
    with file.open("rb") as fh:
        r = client().post(url, content=fh, headers={"Authorization": f"OAuth {token}", "offset": "0",
                                                      "file_size": str(size)})
    return _check(r, what)


def instagram_reel(acc: dict, file: Path, caption: str, wait: float = 600.0) -> dict:
    """Instagram Reel through a resumable upload; returns {"id", "url"}."""
    ig, token = acc["ig_id"], acc["token"]
    if not ig:
        raise PublishError("the chosen Facebook Page has no linked Instagram Business account")
    box = _check(client().post(f"{GRAPH}/{ig}/media", data={
        "media_type": "REELS", "upload_type": "resumable", "caption": caption[:2200],
        "share_to_feed": "true", "access_token": token}), "Instagram upload start")
    cid = box.get("id")
    uri = box.get("uri") or f"https://rupload.facebook.com/ig-api-upload/v24.0/{cid}"
    _rupload(uri, token, file, "Instagram upload")
    t0 = time.time()
    while True:
        st = _check(client().get(f"{GRAPH}/{cid}", params={"fields": "status_code,status",
                                                           "access_token": token}), "Instagram status")
        code = st.get("status_code")
        if code == "FINISHED":
            break
        if code in ("ERROR", "EXPIRED"):
            raise PublishError(f"Instagram could not process the video: {st.get('status') or code}")
        if time.time() - t0 > wait:
            raise PublishError("Instagram is still processing the video - try again later")
        time.sleep(5)
    pub = _check(client().post(f"{GRAPH}/{ig}/media_publish", data={
        "creation_id": cid, "access_token": token}), "Instagram publish")
    mid = pub.get("id", "")
    link = ""
    try:
        link = _check(client().get(f"{GRAPH}/{mid}", params={"fields": "permalink", "access_token": token}),
                      "Instagram link").get("permalink", "")
    except PublishError:
        pass
    return {"id": mid, "url": link}


def facebook_reel(acc: dict, file: Path, description: str) -> dict:
    """Facebook Page Reel: start, upload, finish. Returns {"id", "url"}."""
    page, token = acc["page_id"], acc["token"]
    start = _check(client().post(f"{GRAPH}/{page}/video_reels", data={
        "upload_phase": "start", "access_token": token}), "Facebook upload start")
    vid = start.get("video_id")
    url = start.get("upload_url") or f"https://rupload.facebook.com/video-upload/v24.0/{vid}"
    _rupload(url, token, file, "Facebook upload")
    _check(client().post(f"{GRAPH}/{page}/video_reels", data={
        "upload_phase": "finish", "video_id": vid, "video_state": "PUBLISHED",
        "description": description[:2200], "access_token": token}), "Facebook publish")
    return {"id": vid, "url": f"https://www.facebook.com/reel/{vid}"}
