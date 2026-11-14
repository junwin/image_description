import json
from pathlib import Path

from PIL import Image

from image_description.cli import marie_prepare_cli, marie_video_cli


def test_prepare_preserves_source_and_creates_yaml(tmp_path):
    original = tmp_path / "incoming"
    original.mkdir()
    photo = original / "01-photo.png"
    Image.new("RGB", (80, 40), "red").save(photo)
    before = photo.read_bytes()
    output = tmp_path / "ready"
    marie_prepare_cli.main([
        str(original), "--output", str(output), "--format", "reel",
        "--no-describe", "--width", "108", "--height", "192",
    ])
    rendered = output / "001-01-photo.jpg"
    assert rendered.exists()
    with Image.open(rendered) as image:
        assert image.size == (108, 192)
    assert photo.read_bytes() == before
    sidecar = (output / "001-01-photo.yaml").read_text(encoding="utf-8")
    assert 'aspect_ratio: "108:192"' in sidecar
    assert "alt_text:" in sidecar


def test_prepare_uses_existing_description_pipeline(tmp_path, monkeypatch):
    original = tmp_path / "incoming"
    original.mkdir()
    Image.new("RGB", (50, 50)).save(original / "photo.jpg")
    output = tmp_path / "ready"
    seen = []
    def fake_process(path, **kwargs):
        seen.append(kwargs["preset"])
        Path(path).with_suffix(".json").write_text(json.dumps({
            "image_description": "A patterned scarf.",
            "enhanced_description": "A patterned scarf on a table.",
            "social_caption": "A little colour for your day.",
            "keywords": ["scarf"],
        }), encoding="utf-8")
    monkeypatch.setattr(marie_prepare_cli, "process_image", fake_process)
    marie_prepare_cli.main([str(original), "--output", str(output)])
    assert seen == ["marie-parie"]
    assert "A patterned scarf." in (output / "001-photo.yaml").read_text()


def test_video_command_has_crossfades_and_duration(tmp_path):
    photos = []
    for index in range(3):
        path = tmp_path / f"{index:03d}.jpg"
        Image.new("RGB", (108, 192)).save(path)
        photos.append(path)
    cmd = marie_video_cli.make_command(photos, tmp_path / "video.mp4", 9, 0.5, 30, "zoom")
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert graph.count("xfade=transition=fade") == 2
    assert cmd[cmd.index("-t") + 1] == "9"
    assert "libx264" in cmd
