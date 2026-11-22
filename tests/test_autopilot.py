"""Autopilot + publishers against a fake platform (no network).   python -m pytest tests"""
from __future__ import annotations

import json

import httpx
import pytest

from opusclip import autopilot, publish
from opusclip.models import Clip, Score


class Fake:
    """Records every request and answers like YouTube / Graph API would."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.ig_polls = 0

    def __call__(self, req: httpx.Request) -> httpx.Response:
        url = str(req.url)
        self.calls.append((req.method, url.split("?")[0]))
        if "oauth2.googleapis.com/token" in url:
            return httpx.Response(200, json={"access_token": "ya29.x", "refresh_token": "1//r"})
        if "upload/youtube/v3/videos" in url and req.method == "POST":
            meta = json.loads(req.content)
            assert meta["snippet"]["title"] and meta["status"]["privacyStatus"]
            return httpx.Response(200, headers={"location": "https://upload.example/session1"})
        if url.startswith("https://upload.example/session1"):
            assert req.headers["content-length"] == str(len(req.content))
            return httpx.Response(200, json={"id": "VID123"})
        if url.endswith("/oauth/access_token") or "/oauth/access_token?" in url:
            return httpx.Response(200, json={"access_token": "LONG"})
        if "/me/accounts" in url:
            return httpx.Response(200, json={"data": [{"id": "PAGE1", "name": "My Page", "access_token": "PTOK",
                                                       "instagram_business_account": {"id": "IG1", "username": "me"}}]})
        if url.endswith("/IG1/media"):
            return httpx.Response(200, json={"id": "CONT1", "uri": "https://rupload.facebook.com/ig-api-upload/v24.0/CONT1"})
        if "rupload.facebook.com" in url:
            assert req.headers["authorization"].startswith("OAuth ") and req.headers["offset"] == "0"
            return httpx.Response(200, json={"success": True})
        if url.split("?")[0].endswith("/CONT1"):
            self.ig_polls += 1
            return httpx.Response(200, json={"status_code": "FINISHED"})
        if url.endswith("/IG1/media_publish"):
            return httpx.Response(200, json={"id": "MEDIA1"})
        if url.split("?")[0].endswith("/MEDIA1"):
            return httpx.Response(200, json={"permalink": "https://instagram.com/reel/abc"})
        if url.endswith("/PAGE1/video_reels"):
            body = dict(x.split("=", 1) for x in req.content.decode().split("&"))
            if body["upload_phase"] == "start":
                return httpx.Response(200, json={"video_id": "FBV1", "upload_url": "https://rupload.facebook.com/video-upload/v24.0/FBV1"})
            return httpx.Response(200, json={"success": True})
        return httpx.Response(404, json={"error": {"message": f"unexpected {req.method} {url}"}})


@pytest.fixture
def fake(tmp_path, monkeypatch):
    f = Fake()
    publish.set_client(httpx.Client(transport=httpx.MockTransport(f)))
    monkeypatch.setattr(autopilot, "PATH", tmp_path / "autopilot.json")
    monkeypatch.setattr(autopilot, "PHONE_DIR", tmp_path / "phone")
    monkeypatch.setattr(autopilot, "_state", None)
    yield f
    publish.set_client(None)


def video(tmp_path):
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"\x00" * 4096)
    return p


def test_youtube_upload(fake, tmp_path):
    acc = {"client_id": "c", "client_secret": "s", "refresh_token": "r"}
    res = publish.youtube_upload(acc, video(tmp_path), "Title", "Desc", ["#gta6"])
    assert res == {"id": "VID123", "url": "https://youtu.be/VID123"}
    assert ("PUT", "https://upload.example/session1") in fake.calls


def test_meta_connect_and_reels(fake, tmp_path):
    res = publish.meta_connect("app", "secret", "short")
    page = res["pages"][0]
    assert page["ig_id"] == "IG1" and page["token"] == "PTOK"
    ig = publish.instagram_reel({"ig_id": "IG1", "token": "PTOK"}, video(tmp_path), "caption")
    assert ig["url"] == "https://instagram.com/reel/abc" and fake.ig_polls == 1
    fb = publish.facebook_reel({"page_id": "PAGE1", "token": "PTOK"}, video(tmp_path), "desc")
    assert fb["id"] == "FBV1"


def test_publish_error_is_readable(fake, tmp_path):
    with pytest.raises(publish.PublishError, match="unexpected"):
        publish.instagram_reel({"ig_id": "NOPE", "token": "x"}, video(tmp_path), "c")


def test_captions_per_platform():
    c = Clip(id="clip_01", start=0, end=30, title="t", hashtags=["#gta6", "viral"],
             package={"yt_title": "Carry lands the plane", "description": "He clutches it."})
    yt = autopilot.caption_for(c, "youtube", "CarryisLive", "https://youtu.be/x")
    assert yt["title"] == "Carry lands the plane" and "#Shorts" in yt["text"] and "Source: CarryisLive" in yt["text"]
    ig = autopilot.caption_for(c, "instagram", "", "")
    assert ig["text"].startswith("Carry lands the plane") and "Source" not in ig["text"]
    long = Clip(id="clip_02", start=0, end=700, kind="long", package={
        "yt_title": "Episode", "chapters": [[0, "Start"], [120, "Fight"], [400, "Win"]]})
    assert "2:00 Fight" in autopilot.caption_for(long, "youtube", "", "")["text"]


def test_queue_caps_and_review(fake, tmp_path, monkeypatch):
    st = autopilot.load()
    st["accounts"]["youtube"] = {"refresh_token": "r", "client_id": "c", "client_secret": "s"}
    st["per_day"] = {"youtube_short": 2, "youtube_long": 1, "instagram": 0, "facebook": 0, "phone": 1}
    st["post_times"] = ["11:00"]
    st["min_score"] = 60

    class P:  # a finished project
        id = "abcd1234"
        settings = {"content": "gaming"}
        clips = [Clip(id=f"clip_0{i}", start=0, end=30, file=f"clips/c{i}.mp4",
                      score=Score(total=90 - i * 10, hook=0, retention=0, emotion=0, value=0, quotability=0))
                 for i in range(1, 5)]

    autopilot.enqueue(P, {"source_name": "Src", "url": "u"})
    q = st["queue"]
    # score 80, 70, 60 pass min 60 (50 does not); youtube + phone (no Meta account)
    assert len(q) == 6 and {it["platform"] for it in q} == {"youtube", "phone"}

    posted = []
    monkeypatch.setattr(autopilot, "post_item", lambda it: (posted.append(it["cid"]), it.update(status="posted")))
    assert autopilot.post_round() == 3               # 2 YouTube shorts + 1 phone, the caps
    assert autopilot.post_round() == 0               # caps used up for today
    assert posted[:2] == ["clip_01", "clip_02"]      # best first

    st["review"] = True
    autopilot.enqueue(P, {"source_name": "Src", "url": "u"})
    item = next(it for it in st["queue"] if it["status"] == "review")
    assert autopilot.set_item(item["id"], "approve")["status"] == "waiting"


def test_tabs():
    assert autopilot._tabs("https://www.youtube.com/@CarryisLive") == [
        "https://www.youtube.com/@CarryisLive/videos", "https://www.youtube.com/@CarryisLive/streams"]
    assert autopilot._tabs("https://www.youtube.com/@x/streams") == ["https://www.youtube.com/@x/streams"]
    assert autopilot._tabs("https://kick.com/someone/videos") == ["https://kick.com/someone/videos"]
