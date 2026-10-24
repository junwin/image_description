import json
import sys
from contextlib import contextmanager
from types import ModuleType
from unittest.mock import Mock
import urllib.error
import urllib.request

import pytest
from image_description.cli import instagram_reels_cli as reels
from image_description.cli import instagram_cli, instagram_video_ngrok


def setup_video(tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"0123456789")
    sidecar = tmp_path / "sample.json"
    sidecar.write_text(json.dumps({"original_title": "Somerset", "original_description": "Cows"}))
    return video, sidecar


def test_reel_api_sequence(monkeypatch):
    calls = []

    def fake_api(method, path, token, **kwargs):
        calls.append((method, path, kwargs))
        if path == "user/media":
            return {"id": "container"}
        if path == "container":
            return {"status_code": "FINISHED"}
        if path == "user/media_publish":
            return {"id": "reel123"}
        return {"permalink": "https://www.instagram.com/reel/abc/"}

    monkeypatch.setattr(instagram_cli, "_api", fake_api)
    result = reels.publish_reel("https://example.com/test.mp4", "Cows", "user", "token")
    assert result["id"] == "reel123"
    assert calls[0][2]["data"]["media_type"] == "REELS"
    assert calls[0][2]["data"]["video_url"] == "https://example.com/test.mp4"
    assert calls[0][2]["data"]["share_to_feed"] == "true"
    assert calls[2][2]["data"] == {"creation_id": "container"}


def test_reel_failure_no_publish(monkeypatch):
    calls = []

    def fake_api(method, path, token, **kwargs):
        calls.append(path)
        return {"id": "container"} if path == "user/media" else {"status_code": "ERROR", "status": "bad MP4"}

    monkeypatch.setattr(instagram_cli, "_api", fake_api)
    with pytest.raises(RuntimeError, match="ERROR"):
        reels.publish_reel("https://example.com/a.mp4", "caption", "user", "token")
    assert calls == ["user/media", "container"]


def test_reel_ngrok_records_sidecar_and_closes(tmp_path, monkeypatch):
    video, sidecar = setup_video(tmp_path)
    lifecycle = []

    @contextmanager
    def fake_host(path):
        assert path == video
        lifecycle.append("open")
        try:
            yield "https://test.ngrok-free.app/video.mp4"
        finally:
            lifecycle.append("close")

    monkeypatch.setattr(instagram_video_ngrok, "serve_video", fake_host)
    monkeypatch.setattr(instagram_cli, "load_credentials", lambda *a: ("user", "token"))

    def fake_publish(url, caption, user, token, share_to_feed=True):
        assert lifecycle == ["open"]
        assert caption == "Somerset\n\nCows"
        lifecycle.append("publish")
        return {"id": "reel123", "permalink": "https://instagram.com/reel/x/"}

    monkeypatch.setattr(reels, "publish_reel", fake_publish)
    reels.cmd_reel(str(video), use_ngrok=True)
    assert lifecycle == ["open", "publish", "close"]
    data = json.loads(sidecar.read_text())
    assert data["publish_history"][-1]["post_id"] == "reel123"


def test_dry_run_no_network(tmp_path, monkeypatch):
    video, sidecar = setup_video(tmp_path)
    monkeypatch.setattr(instagram_cli, "load_credentials", lambda *a: pytest.fail("credentials read"))
    reels.cmd_reel(str(video), use_ngrok=True, dry_run=True)
    assert "publish_history" not in json.loads(sidecar.read_text())


def test_video_server_range_and_isolation(tmp_path, monkeypatch):
    fake_ngrok = ModuleType("pyngrok")
    fake_ngrok.ngrok = Mock()
    fake_ngrok.ngrok.connect.return_value.public_url = "https://example.ngrok-free.app"
    monkeypatch.setitem(sys.modules, "pyngrok", fake_ngrok)
    video, _ = setup_video(tmp_path)
    with instagram_video_ngrok.serve_video(video) as url:
        port = fake_ngrok.ngrok.connect.call_args.kwargs["addr"]
        route = "/" + url.rsplit("/", 1)[-1]
        base = f"http://127.0.0.1:{port}"
        with urllib.request.urlopen(base + route) as response:
            assert response.read() == b"0123456789"
        req = urllib.request.Request(base + route, headers={"Range": "bytes=2-5"})
        with urllib.request.urlopen(req) as response:
            assert response.status == 206
            assert response.read() == b"2345"
            assert response.headers["Content-Range"] == "bytes 2-5/10"
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(base + "/sample.json")
        assert error.value.code == 404
    fake_ngrok.ngrok.disconnect.assert_called_once()
