import json
import os
import pytest
from pathlib import Path

from image_description.cli.social_post_builder_cli import main, Sidecar


def _write_sidecar(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def test_exit_when_missing_hashtags(tmp_path, capsys):
    sidecar = {
        "original_title": "",
        "original_description": "A simple photo",
        "title": "",
        "visually_challenged_description": "",
        "enhanced_description": "",
        "keywords": [],
        "hashtags": "",
        "social_caption": "",
        "image_filename": "IMG_0001.jpg",
        "image_relative_path": "",
    }

    p = tmp_path / "IMG_0001.json"
    _write_sidecar(p, sidecar)

    with pytest.raises(SystemExit) as exc:
        main([str(p), "--platforms", "mastodon"])

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "no hashtags" in captured.err.lower()


def test_write_social_file_and_core_separate(tmp_path):
    sidecar = {
        "original_title": "Sunset",
        "original_description": "A sunset over the hills",
        "title": "Sunset",
        "visually_challenged_description": "",
        "enhanced_description": "A warm orange glow over rolling hills",
        "keywords": ["sunset", "hills", "goldenhour"],
        "hashtags": "",
        "social_caption": "",
        "image_filename": "IMG_0002.jpg",
        "image_relative_path": "",
    }

    p = tmp_path / "IMG_0002.json"
    _write_sidecar(p, sidecar)

    # Run CLI to generate social file
    main([str(p)])

    social_path = Path(Sidecar.social_path_for(str(p)))
    assert social_path.exists()

    with open(social_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "core" in data and "social" in data
    # Core should contain original fields
    for k, v in sidecar.items():
        assert data["core"].get(k) == v

    # Social should have entries for three platforms
    for platform in ["mastodon", "tumblr", "bsky"]:
        assert platform in data["social"]
        item = data["social"][platform]
        assert isinstance(item.get("hashtags"), list) and len(item["hashtags"]) > 0
        # Ensure hashtags start with '#'
        assert all(h.startswith("#") for h in item["hashtags"])