import json
import os
import pytest
from pathlib import Path

from image_description.cli import social_post_builder_cli as spbc
from image_description.cli.social_post_builder_cli import main, Sidecar


def _write_sidecar(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def test_exit_when_model_returns_empty_for_requested_platform(tmp_path, capsys, monkeypatch):
    # Create a basic sidecar with keywords so default generator would produce hashtags
    sidecar = {
        "original_title": "Test",
        "original_description": "A test image",
        "title": "Test",
        "image_description": "",
        "enhanced_description": "",
        "keywords": ["test", "image"],
        "hashtags": "",
        "social_caption": "",
        "image_filename": "IMG_1001.jpg",
        "image_relative_path": "",
    }

    p = tmp_path / "IMG_1001.json"
    _write_sidecar(p, sidecar)

    # Monkeypatch the internal generator to simulate model returning no hashtags
    def fake_generate(sidecar_obj, platforms):
        # Return empty hashtags for 'mastodon' only
        return {
            "mastodon": {"text": "", "hashtags": []},
            "tumblr": {"text": "", "hashtags": ["#test"]},
            "bsky": {"text": "", "hashtags": ["#image"]},
        }

    monkeypatch.setattr(spbc, "_generate_social_for_sidecar", fake_generate)

    with pytest.raises(SystemExit) as exc:
        main([str(p), "--platforms", "mastodon", "tumblr", "bsky"])

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "model returned no hashtags" in captured.err.lower()
    assert "mastodon" in captured.err.lower()

    # Ensure no social derivative file was created
    social_path = Path(Sidecar.social_path_for(str(p)))
    assert not social_path.exists()


def test_success_when_all_platforms_have_hashtags(tmp_path, monkeypatch):
    sidecar = {
        "original_title": "Success",
        "original_description": "All good",
        "title": "Success",
        "image_description": "",
        "enhanced_description": "",
        "keywords": [],
        "hashtags": "#ok #done",
        "social_caption": "",
        "image_filename": "IMG_1002.jpg",
        "image_relative_path": "",
    }

    p = tmp_path / "IMG_1002.json"
    _write_sidecar(p, sidecar)

    # Fake generator that returns hashtags for all platforms
    def fake_generate_all(sidecar_obj, platforms):
        out = {}
        for pl in platforms:
            out[pl] = {"text": "x", "hashtags": ["#ok"]}
        return out

    monkeypatch.setattr(spbc, "_generate_social_for_sidecar", fake_generate_all)

    # Should not raise
    main([str(p), "--platforms", "mastodon", "tumblr"])

    social_path = Path(Sidecar.social_path_for(str(p)))
    assert social_path.exists()

    with open(social_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "core" in data and "social" in data
    # Ensure social has requested platforms
    assert "mastodon" in data["social"] and "tumblr" in data["social"]
    # Ensure core matches original sidecar values
    assert data["core"]["original_title"] == sidecar["original_title"]
