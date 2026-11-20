import subprocess

from PIL import Image

from image_description.cli import marie_prepare_cli, marie_publish_cli


def test_web_converts_png_without_baked_padding(tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    Image.new("RGBA", (800, 400), (40, 80, 100, 255)).save(incoming / "01-jacket.png")
    output = tmp_path / "ready"
    marie_prepare_cli.main([str(incoming), "--output", str(output),
                            "--format", "web", "--no-describe"])
    photo = output / "001-01-jacket.jpg"
    with Image.open(photo) as image:
        assert image.format == "JPEG"
        assert image.size == (600, 300)
        # Natural mode preserves the source aspect ratio, without white padding.
        assert image.getpixel((0, 0))[0] < 100
    assert (incoming / "01-jacket.png").exists()


def test_publish_prepends_and_is_idempotent(tmp_path):
    repo = tmp_path / "site"
    gallery = repo / "assets" / "New Arrivals"
    gallery.mkdir(parents=True)
    (gallery / "gallery.txt").write_text("old-one.jpg\nold-two.png\n")
    subprocess.run(["git", "init", "-b", "staging", str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "Initial"], check=True, capture_output=True)
    incoming = tmp_path / "ready"
    incoming.mkdir()
    for name in ("001-new.jpg", "002-next.jpg"):
        Image.new("RGB", (20, 10)).save(incoming / name)
    marie_publish_cli.publish(incoming, repo, dry_run=True)
    assert (gallery / "gallery.txt").read_text() == "old-one.jpg\nold-two.png\n"
    marie_publish_cli.publish(incoming, repo, commit=True)
    expected = "001-new.jpg\n002-next.jpg\nold-one.jpg\nold-two.png\n"
    assert (gallery / "gallery.txt").read_text() == expected
    marie_publish_cli.publish(incoming, repo, commit=True)
    assert (gallery / "gallery.txt").read_text() == expected


def test_custom_aspect_ratio_and_crop_style(tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    Image.new("RGB", (900, 500), "blue").save(incoming / "sample.jpg")
    output = tmp_path / "output"
    marie_prepare_cli.main([str(incoming), "--output", str(output),
                            "--format", "web", "--aspect-ratio", "3:2",
                            "--width", "900", "--fit", "cover", "--no-describe"])
    with Image.open(output / "001-sample.jpg") as image:
        assert image.size == (900, 600)


def test_transparent_png_background_is_white(tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    Image.new("RGBA", (60, 90), (255, 0, 0, 0)).save(incoming / "transparent.png")
    output = tmp_path / "ready"
    marie_prepare_cli.main([str(incoming), "--output", str(output),
                            "--format", "web", "--no-describe"])
    with Image.open(output / "001-transparent.jpg") as image:
        assert image.size == (600, 900)
        assert all(channel > 245 for channel in image.getpixel((300, 450)))


def test_publish_dry_run_rejects_commit(tmp_path):
    import pytest
    with pytest.raises(ValueError, match="dry-run"):
        marie_publish_cli.publish(tmp_path, tmp_path, dry_run=True, commit=True)


def test_contain_keeps_exact_canvas_and_white_margins(tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    Image.new("RGB", (800, 400), "blue").save(incoming / "wide.jpg")
    output = tmp_path / "ready"
    marie_prepare_cli.main([str(incoming), "--output", str(output),
                            "--format", "web", "--fit", "contain", "--no-describe"])
    with Image.open(output / "001-wide.jpg") as image:
        assert image.size == (600, 900)
        assert all(v > 245 for v in image.getpixel((300, 0)))


def test_reel_defaults_to_fixed_portrait_canvas(tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    Image.new("RGB", (800, 400), "blue").save(incoming / "wide.jpg")
    output = tmp_path / "reel"
    marie_prepare_cli.main([str(incoming), "--output", str(output),
                            "--format", "reel", "--no-describe"])
    with Image.open(output / "001-wide.jpg") as image:
        assert image.size == (1080, 1920)
