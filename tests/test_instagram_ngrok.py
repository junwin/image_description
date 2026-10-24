import json
from contextlib import contextmanager
from unittest.mock import Mock
import urllib.error
import urllib.request

import pytest

from image_description.cli import instagram_cli, instagram_ngrok


def test_ngrok_mode_publishes_and_closes_before_recording(tmp_path, monkeypatch):
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"jpeg-data")
    sidecar = tmp_path / "photo.json"
    sidecar.write_text(json.dumps({"original_description": "One photograph"}))
    steps = []

    @contextmanager
    def fake_host(path):
        assert path == image
        steps.append("open")
        try:
            yield "https://example.ngrok-free.app/unique.jpg"
        finally:
            steps.append("close")

    def fake_publish(url, caption, uid, token):
        assert steps == ["open"]
        assert url == "https://example.ngrok-free.app/unique.jpg"
        assert caption == "One photograph"
        steps.append("publish")
        return {"id": "ig123", "permalink": "https://www.instagram.com/p/abc/"}

    monkeypatch.setattr(instagram_ngrok, "serve_image", fake_host)
    monkeypatch.setattr(instagram_cli, "load_credentials", lambda *args: ("user123", "token123"))
    monkeypatch.setattr(instagram_cli, "publish_photo", fake_publish)
    instagram_cli.cmd_post(str(image), use_ngrok=True)
    assert steps == ["open", "publish", "close"]
    record = json.loads(sidecar.read_text())["publish_history"][-1]
    assert record["platform"] == "instagram"
    assert record["post_id"] == "ig123"


def test_ngrok_failure_cleans_up_and_keeps_sidecar(tmp_path, monkeypatch):
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"jpeg-data")
    sidecar = tmp_path / "photo.json"
    sidecar.write_text("{}")
    steps = []

    @contextmanager
    def fake_host(path):
        try:
            yield "https://example.ngrok-free.app/image.jpg"
        finally:
            steps.append("closed")

    monkeypatch.setattr(instagram_ngrok, "serve_image", fake_host)
    monkeypatch.setattr(instagram_cli, "load_credentials", lambda *args: ("u", "t"))
    monkeypatch.setattr(instagram_cli, "publish_photo",
                        lambda *args: (_ for _ in ()).throw(RuntimeError("API failed")))
    with pytest.raises(RuntimeError, match="API failed"):
        instagram_cli.cmd_post(str(image), use_ngrok=True)
    assert steps == ["closed"]
    assert json.loads(sidecar.read_text()).get("publish_history") is None


def test_ngrok_and_url_mutually_exclusive(tmp_path):
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"jpeg-data")
    (tmp_path / "photo.json").write_text("{}")
    with pytest.raises(ValueError, match="Choose"):
        instagram_cli.cmd_post(str(image), image_url="https://example.com/a.jpg", use_ngrok=True)


def test_ngrok_dry_run_does_not_open_tunnel(tmp_path, monkeypatch):
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"jpeg-data")
    (tmp_path / "photo.json").write_text("{}")
    monkeypatch.setattr(instagram_ngrok, "serve_image", lambda *args: pytest.fail("tunnel opened"))
    instagram_cli.cmd_post(str(image), use_ngrok=True, dry_run=True)


def test_http_handler_only_serves_single_image(tmp_path, monkeypatch):
    # Mock tunnel only. Exercise the actual loopback HTTP server and URL path isolation.
    import sys
    from types import ModuleType
    fake_ngrok = ModuleType("pyngrok")
    fake_ngrok.ngrok = Mock()
    fake_ngrok.ngrok.connect.return_value.public_url = "https://example.ngrok-free.app"
    monkeypatch.setitem(sys.modules, "pyngrok", fake_ngrok)

    image = tmp_path / "photo.jpg"
    image.write_bytes(b"jpeg-data")
    (tmp_path / "secret.txt").write_text("not-public")
    # The proxy URL is fake, so inspect the bound port from the connect call.
    with instagram_ngrok.serve_image(image) as public_url:
        kwargs = fake_ngrok.ngrok.connect.call_args.kwargs
        port = kwargs["addr"]
        route = "/" + public_url.rsplit("/", 1)[-1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{route}") as resp:
            assert resp.read() == b"jpeg-data"
            assert resp.headers["Content-Type"] == "image/jpeg"
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/secret.txt")
        assert err.value.code == 404
    fake_ngrok.ngrok.disconnect.assert_called_once_with("https://example.ngrok-free.app")
