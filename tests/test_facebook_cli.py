import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from image_description.cli import facebook_cli


def test_dry_run_does_not_read_credentials_or_publish(tmp_path, monkeypatch, capsys):
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"fake image")
    (tmp_path / "photo.json").write_text(
        json.dumps({"original_title": "Morning", "original_description": "Light", "hashtags": "#street"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(facebook_cli, "load_credentials", lambda *a: pytest.fail("credentials accessed"))
    monkeypatch.setattr(facebook_cli, "publish_photo", lambda *a: pytest.fail("unexpected publish"))
    facebook_cli.cmd_post(str(image), dry_run=True)
    assert "Morning" in capsys.readouterr().out


def test_upload_uses_page_token_and_records_history(tmp_path, monkeypatch):
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"jpeg-data")
    sidecar = tmp_path / "photo.json"
    sidecar.write_text(json.dumps({"original_description": "A photograph"}), encoding="utf-8")
    monkeypatch.setattr(facebook_cli, "load_credentials", lambda *a: ("136529803624992", "secret"))
    response = Mock()
    response.json.return_value = {"id": "photo123", "post_id": "136529803624992_456"}
    response.raise_for_status.return_value = None
    calls = []
    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return response
    monkeypatch.setattr(facebook_cli.requests, "post", fake_post)
    facebook_cli.cmd_post(str(image))
    assert len(calls) == 1
    assert calls[0][0].endswith("/136529803624992/photos")
    assert calls[0][1]["data"]["access_token"] == "secret"
    assert calls[0][1]["data"]["message"] == "A photograph"
    record = json.loads(sidecar.read_text(encoding="utf-8"))["publish_history"][-1]
    assert record["platform"] == "facebook"
    assert record["post_id"] == "136529803624992_456"


def test_failed_upload_does_not_record_history(tmp_path, monkeypatch):
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"jpeg-data")
    sidecar = tmp_path / "photo.json"
    sidecar.write_text(json.dumps({"original_description": "Caption"}), encoding="utf-8")
    monkeypatch.setattr(facebook_cli, "load_credentials", lambda *a: ("123", "secret"))
    def fail(*args, **kwargs):
        raise RuntimeError("API failed")
    monkeypatch.setattr(facebook_cli, "publish_photo", fail)
    with pytest.raises(RuntimeError):
        facebook_cli.cmd_post(str(image))
    assert json.loads(sidecar.read_text(encoding="utf-8")).get("publish_history") is None


def test_credentials_validation(tmp_path):
    path = tmp_path / "credentials.json"
    path.write_text(json.dumps({"page_id": "123"}))
    with pytest.raises(ValueError, match="page_access_token"):
        facebook_cli.load_credentials(path)
