"""Fitting/inspection and motion quality regression tests for Marie-Parie."""
import shutil
import subprocess

import pytest
from PIL import Image

from image_description.cli.marie_inspect_cli import inspect_batch
from image_description.cli.marie_prepare_cli import render_photo
from image_description.cli.marie_video_cli import make_command


def test_square_photo_can_be_contained_on_gray_reel_canvas(tmp_path):
    original = tmp_path / "square.png"
    target = tmp_path / "fitted.jpg"
    Image.new("RGB", (300, 300), (220, 10, 30)).save(original)
    render_photo(original, target, (720, 1280), "contain", "#808080")
    with Image.open(target) as result:
        assert result.size == (720, 1280)
        assert all(abs(a - b) < 6 for a, b in zip(result.getpixel((0, 0)), (128, 128)))
        assert result.getpixel((360, 640))[0] > 200


def test_cover_fills_reel_frame_without_borders(tmp_path):
    original = tmp_path / "square.png"
    target = tmp_path / "cropped.jpg"
    Image.new("RGB", (300, 300), (220, 10, 30)).save(original)
    render_photo(original, target, (720, 1280), "cover")
    with Image.open(target) as result:
        assert result.size == (720, 1280)
        assert result.getpixel((0, 0))[0] > 200


def test_inspect_recommends_contain_for_square_and_cover_for_vertical(tmp_path):
    Image.new("RGB", (300, 300), "pink").save(tmp_path / "01-square.jpg")
    Image.new("RGB", (900, 1600), "blue").save(tmp_path / "02-portrait.jpg")
    report = inspect_batch(tmp_path, (720, 1280))
    assert report["count"] == 2
    assert report["images"][0]["cover_crop_percent"] == pytest.approx(43.8, abs=0.1)
    assert report["images"][0]["suggested_fit"] == "contain"
    assert report["images"][1]["cover_crop_percent"] == 0
    assert report["images"][1]["suggested_fit"] == "cover"
    assert report["suggested_batch_fit"] == "contain"


def test_video_super_sampling_is_applied_before_downscale(tmp_path):
    cmd = make_command([tmp_path / "a.jpg"], tmp_path / "out.mp4",
                       2, 0.4, 30, "zoom", (720, 1280), supersample=2)
    filters = cmd[cmd.index("-filter_complex") + 1]
    assert "scale=1440:2560:" in filters
    assert "s=1440x2560" in filters
    assert "scale=720:1280:flags=lanczos" in filters


def test_video_rejects_unsupported_super_sampling(tmp_path):
    with pytest.raises(ValueError, match="supersample"):
        make_command([tmp_path / "a.jpg"], tmp_path / "out.mp4",
                     2, 0.4, 30, "zoom", supersample=0)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_zoom_filter_runs_with_real_ffmpeg(tmp_path):
    image = tmp_path / "test.jpg"
    Image.new("RGB", (100, 180), "green").save(image)
    output = tmp_path / "preview.mp4"
    cmd = make_command([image], output, 0.5, 0.1, 10, "zoom",
                       (90, 160), supersample=2)
    subprocess.run(cmd, check=True, capture_output=True, timeout=45)
    assert output.is_file() and output.stat().st_size > 0
