import json
from unittest.mock import Mock

import pytest

from image_description.cli import instagram_cli


def sample(tmp_path):
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"fake jpeg")
    sidecar = tmp_path / "photo.json"
    sidecar.write_text(json.dumps({"original_title": "Morning", "original_description": "Light"}))
    return image, sidecar


def test_dry_run_skips_credentials_and_requests(tmp_path, monkeypatch, capsys):
    image, sidecar = sample(tmp_path)
    monkeypatch.setattr(instagram_cli, "load_credentials", lambda *a: pytest.fail("credentials read"))
    monkeypatch.setattr(instagram_cli, "publish_photo", lambda *a: pytest.fail("published"))
    instagram_cli.cmd_post(str(image), dry_run=True)
    assert "Morning" in capsys.readouterr().out
    assert "publish_history" not in json.loads(sidecar.read_text())


def test_publish_container_and_sidecar(tmp_path, monkeypatch):
    image, sidecar = sample(tmp_path)
    monkeypatch.setattr(instagram_cli, "load_credentials", lambda *a: ("39297990789845861", "secret"))
    calls = []

    def fake_request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        response = Mock(ok=True)
        if url.endswith("/media"):
            response.json.return_value = {"id": "container123"}
        elif url.endswith("/container123"):
            response.json.return_value = {"status_code": "FINISHED"}
        elif url.endswith("/media_publish"):
            response.json.return_value = {"id": "media456"}
        else:
            response.json.return_value = {"id": "media456", "permalink": "https://www.instagram.com/p/example/"}
        return response

    monkeypatch.setattr(instagram_cli.requests, "request", fake_request)
    instagram_cli.cmd_post(str(image), image_url="https://example.com/photo.jpg")
    assert [c[0] for c in calls] == ["POST", "GET", "POST", "GET"]
    assert calls[0][2]["data"]["image_url"] == "https://example.com/photo.jpg"
    assert calls[0][2]["data"]["caption"] == "Morning\n\nLight"
    assert calls[0][2]["headers"]["Authorization"] == "Bearer secret"
    assert calls[2][2]["data"]["creation_id"] == "container123"
    record = json.loads(sidecar.read_text())["publish_history"][-1]
    assert record["platform"] == "instagram"
    assert record["post_id"] == "media456"
    assert record["url"] == "https://www.instagram.com/p/example/"


def test_failure_keeps_sidecar(tmp_path, monkeypatch):
    image, sidecar = sample(tmp_path)
    monkeypatch.setattr(instagram_cli, "load_credentials", lambda *a: ("123", "secret"))
    monkeypatch.setattr(instagram_cli, "publish_photo", lambda *a: (_ for _ in ()).throw(RuntimeError("failed")))
    with pytest.raises(RuntimeError, match="failed"):
        instagram_cli.cmd_post(str(image), image_url="https://example.com/photo.jpg")
    assert "publish_history" not in json.loads(sidecar.read_text())


def test_missing_url_no_credentials(tmp_path, monkeypatch):
    image, _ = sample(tmp_path)
    monkeypatch.setattr(instagram_cli, "load_credentials", lambda *a: pytest.fail("credentials read"))
    with pytest.raises(ValueError, match="image-url"):
        instagram_cli.cmd_post(str(image))


def test_credentials_validation(tmp_path):
    path = tmp_path / "creds.json"
    path.write_text(json.dumps({"instagram_user_id": "123"}))
    with pytest.raises(ValueError, match="access_token"):
        instagram_cli.load_credentials(path)


def test_non_https_rejected():
    with pytest.raises(ValueError, match="HTTPS"):
        instagram_cli.validate_image_url("http://example.com/photo.jpg")


def test_container_error_no_publish(monkeypatch):
    calls = []

    def fake_api(method, path, token, **kwargs):
        calls.append((method, path))
        if path.endswith("/media"):
            return {"id": "container"}
        return {"status_code": "ERROR"}

    monkeypatch.setattr(instagram_cli, "_api", fake_api)
    with pytest.raises(RuntimeError, match="ERROR"):
        instagram_cli.publish_photo("https://example.com/photo.jpg", "caption", "123", "secret")
    assert len(calls) == 2


def test_permalink_lookup_failure_still_records_publish(tmp_path, monkeypatch):
    image, sidecar = sample(tmp_path)
    monkeypatch.setattr(instagram_cli, "load_credentials", lambda *a: ("123", "secret"))
    calls = []

    def fake_api(method, path, token, **kwargs):
        calls.append(path)
        if path == "123/media":
            return {"id": "container"}
        if path == "container":
            return {"status_code": "FINISHED"}
        if path == "123/media_publish":
            return {"id": "published"}
        raise RuntimeError("permalink unavailable")

    monkeypatch.setattr(instagram_cli, "_api", fake_api)
    instagram_cli.cmd_post(str(image), image_url="https://example.com/photo.jpg")
    record = json.loads(sidecar.read_text())["publish_history"][-1]
    assert record["post_id"] == "published"
    assert record["url"] == ""
