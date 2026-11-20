import subprocess

from PIL import Image

from image_description.cli import marie_prepare_cli, marie_publish_cli


def test_web_converts_png_and_fits_to_target_ratio(tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    Image.new("RGBA", (800, 400), (40, 80, 100, 255)).save(incoming / "01-jacket.png")
    output = tmp_path / "ready"
    marie_prepare_cli.main([str(incoming), "--output", str(output),
                            "--format", "web", "--no-describe"])
    photo = output / "001-01-jacket.jpg"
    with Image.open(photo) as image:
        assert image.format == "JPEG"
        assert image.size == (600, 900)
        # Original 2:1 landscape image is contained in the 600:900 portrait frame.
        assert image.getpixel((0, 0))[0] > 245
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
