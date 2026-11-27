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

def test_marie_description_suggests_name_but_does_not_rename(tmp_path, monkeypatch):
    import json
    from image_description.cli import marie_describe_cli

    image = tmp_path / "001-8d67abc.jpg"
    Image.new("RGB", (600, 900), "pink").save(image)

    def fake_describe(path, **kwargs):
        assert kwargs["preset"] == "marie-parie"
        assert kwargs["suggest_filename"] is True
        image_path = __import__("pathlib").Path(path)
        image_path.with_suffix(".json").write_text(json.dumps({
            "image_description": "A pink floral scarf.",
            "enhanced_description": "Pink floral scarf on a table.",
            "social_caption": "A little colour!",
            "keywords": ["floral", "scarf"],
            "suggested_filename": "Pink Floral Scarf",
            "image_filename": image_path.name,
            "image_relative_path": image_path.name,
        }))

    monkeypatch.setattr(marie_describe_cli, "process_image", fake_describe)
    marie_describe_cli.main([str(tmp_path)])
    assert image.exists()
    assert not (tmp_path / "001-pink-floral-scarf.jpg").exists()
    metadata = json.loads(image.with_suffix(".json").read_text())
    assert metadata["suggested_filename"] == "001-pink-floral-scarf.jpg"
    assert metadata["image_filename"] == image.name
    assert 'suggested_filename: "001-pink-floral-scarf.jpg"' in image.with_suffix(".yaml").read_text()


def test_suggested_filename_does_not_duplicate_prefix():
    from image_description.cli.marie_describe_cli import proposed_filename
    assert proposed_filename("001-abc.jpg", "001-pink-floral-scarf.jpg") == "001-pink-floral-scarf.jpg"


def test_gallery_collection_folder_accepts_assets_prefix(tmp_path):
    import pytest
    from image_description.cli.marie_publish_cli import normalize_collection
    assert normalize_collection("Button_s Bougie") == "Button_s Bougie"
    assert normalize_collection("assets/Button_s Bougie") == "Button_s Bougie"
    assert normalize_collection("assets/New Arrivals") == "New Arrivals"
    for invalid in ("../outside", "assets/../outside", "", "assets/", "a/b"):
        with pytest.raises(ValueError):
            normalize_collection(invalid)


def test_publish_updates_only_selected_collection(tmp_path):
    repo = tmp_path / "site"
    arrivals = repo / "assets" / "New Arrivals"
    bougie = repo / "assets" / "Button_s Bougie"
    arrivals.mkdir(parents=True)
    bougie.mkdir(parents=True)
    (arrivals / "gallery.txt").write_text("arrivals-old.jpg\n")
    (bougie / "gallery.txt").write_text("bougie-old.jpg\n")
    subprocess.run(["git", "init", "-b", "staging", str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "Initial"], check=True, capture_output=True)
    prepared = tmp_path / "ready"
    prepared.mkdir()
    Image.new("RGB", (60, 90)).save(prepared / "001-new-item.jpg")
    marie_publish_cli.publish(prepared, repo, collection="assets/Button_s Bougie")
    assert (bougie / "001-new-item.jpg").exists()
    assert (bougie / "gallery.txt").read_text().splitlines() == [
        "001-new-item.jpg", "bougie-old.jpg"]
    assert (arrivals / "gallery.txt").read_text() == "arrivals-old.jpg\n"
    assert not (arrivals / "001-new-item.jpg").exists()


def test_local_preview_branch_can_prepare_but_not_commit_or_push(tmp_path):
    import pytest
    repo = tmp_path / "site"
    collection = repo / "assets" / "New Arrivals"
    collection.mkdir(parents=True)
    (collection / "gallery.txt").write_text("original.jpg\n")
    subprocess.run(["git", "init", "-b", "staging", str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "Initial"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "-b", "preview/marie-parie/arla-123"],
                   check=True, capture_output=True)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    Image.new("RGB", (40, 60), "pink").save(prepared / "001-new.jpg")
    marie_publish_cli.publish(prepared, repo, collection="New Arrivals")
    assert (collection / "001-new.jpg").is_file()
    assert (collection / "gallery.txt").read_text().startswith("001-new.jpg\n")
    with pytest.raises(ValueError, match="local preparation"):
        marie_publish_cli.publish(prepared, repo, commit=True)
    with pytest.raises(ValueError, match="local preparation"):
        marie_publish_cli.publish(prepared, repo, commit=True, push=True)
