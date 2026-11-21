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


def test_reel_720p_mixed_input_sizes_and_two_seconds_per_photo(tmp_path):
    import pytest
    photos = []
    for index, size in enumerate(((600, 900), (600, 400), (600, 600), (900, 600), (600, 900))):
        path = tmp_path / f"{index:03d}.jpg"
        Image.new("RGB", size).save(path)
        photos.append(path)
    duration = marie_video_cli.total_duration(len(photos), 2.0, 0.4)
    assert duration == pytest.approx(8.4)
    cmd = marie_video_cli.make_command(photos, tmp_path / "reel.mp4",
                                       duration, 0.4, 30, "zoom")
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert graph.count("xfade=transition=fade") == 4
    assert graph.count("pad=720:1280") == 5
    assert graph.count("zoompan=") == 5
    assert cmd[cmd.index("-t") + 1] == str(duration)
    assert "libx264" in cmd


def test_reel_without_fade_uses_concat(tmp_path):
    photo = tmp_path / "picture.jpg"
    Image.new("RGB", (600, 900)).save(photo)
    cmd = marie_video_cli.make_command([photo, photo], tmp_path / "reel.mp4",
                                       4.0, 0.0, 30, "none")
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "concat=n=2:v=1:a=0" in graph
    assert "xfade=" not in graph


def test_reel_ffmpeg_smoke_render(tmp_path):
    """Optional integration test: real MP4 with a crossfade, if FFmpeg exists."""
    import shutil
    import pytest
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg not installed")
    incoming = tmp_path / "input"
    incoming.mkdir()
    Image.new("RGB", (96, 64), "red").save(incoming / "001-red.jpg")
    Image.new("RGB", (64, 96), "blue").save(incoming / "002-blue.jpg")
    video = tmp_path / "reel.mp4"
    marie_video_cli.main([
        str(incoming), "--output", str(video), "--width", "72",
        "--height", "128", "--fps", "15", "--fade", "0.4",
        "--seconds-per-image", "2", "--movement", "zoom",
    ])
    assert video.exists()
    assert video.stat().st_size > 1000
