"""Marie-Parie workspace conventions and batch command integration tests."""
import json
from pathlib import Path

import pytest
from PIL import Image

from image_description.marie_workspace import batch_paths
from image_description.cli import marie_batch_cli


BATCH = "batch-0123456789ab"


def test_batch_paths_group_all_artifacts_under_same_id(tmp_path):
    root = tmp_path / "marie-parie"
    paths = batch_paths(BATCH, root)
    assert paths.incoming == root / "incoming" / BATCH
    assert paths.web == root / "prepared" / "web" / BATCH
    assert paths.reel_images == root / "prepared" / "reel" / BATCH
    assert paths.reel_video == root / "output" / "reels" / f"{BATCH}.mp4"


@pytest.mark.parametrize("unsafe", ["../escape", "/tmp/foo", "batch-", "batch-123", "batch-0123456789AG"])
def test_batch_id_rejects_unexpected_paths(tmp_path, unsafe):
    with pytest.raises(ValueError, match="Batch must"):
        batch_paths(unsafe, tmp_path)


def test_paths_command_prints_expected_directories(tmp_path, capsys):
    marie_batch_cli.main(["--workspace", str(tmp_path), "--batch", BATCH, "paths"])
    report = json.loads(capsys.readouterr().out)
    assert report["incoming"] == str(tmp_path / "incoming" / BATCH)
    assert report["prepared_web"] == str(tmp_path / "prepared" / "web" / BATCH)


def test_batch_inspect_and_resize_without_touching_originals(tmp_path, capsys):
    source = tmp_path / "incoming" / BATCH
    source.mkdir(parents=True)
    original = source / "boutique.png"
    Image.new("RGB", (400, 400), "red").save(original)
    original_bytes = original.read_bytes()
    prefix = ["--workspace", str(tmp_path), "--batch", BATCH]
    marie_batch_cli.main(prefix + ["inspect"])
    report = json.loads(capsys.readouterr().out)
    assert report["count"] == 1
    assert report["images"][0]["suggested_fit"] == "contain"

    marie_batch_cli.main(prefix + ["resize", "--format", "reel",
                                   "--fit", "contain", "--background", "#888888"])
    out = tmp_path / "prepared" / "reel" / BATCH / "001-boutique.jpg"
    assert out.is_file()
    with Image.open(out) as rendered:
        assert rendered.size == (1080, 1920)
        pixel = rendered.getpixel((0, 0))
        assert all(abs(channel - 136) < 8 for channel in pixel)
    assert original.read_bytes() == original_bytes

    marie_batch_cli.main(prefix + ["resize", "--format", "web"])
    assert (tmp_path / "prepared" / "web" / BATCH / "001-boutique.jpg").exists()


def test_reel_requires_prepared_images_and_does_not_overwrite(tmp_path, monkeypatch):
    source = tmp_path / "incoming" / BATCH
    source.mkdir(parents=True)
    prefix = ["--workspace", str(tmp_path), "--batch", BATCH]
    with pytest.raises(SystemExit):
        marie_batch_cli.main(prefix + ["reel"])
    (tmp_path / "prepared" / "reel" / BATCH).mkdir(parents=True)
    destination = tmp_path / "output" / "reels" / f"{BATCH}.mp4"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"existing-output")
    with pytest.raises(SystemExit):
        marie_batch_cli.main(prefix + ["reel"])
    assert destination.read_bytes() == b"existing-output"
    called = []
    monkeypatch.setattr(marie_batch_cli, "video_main", lambda argv: called.append(argv))
    marie_batch_cli.main(prefix + ["reel", "--overwrite"])
    assert called
    assert str(destination) in called[0]


def test_web_describe_and_review_before_approved_rename(tmp_path, monkeypatch, capsys):
    source = tmp_path / "incoming" / BATCH
    source.mkdir(parents=True)
    Image.new("RGB", (200, 300), "pink").save(source / "jewellery.png")
    prefix = ["--workspace", str(tmp_path), "--batch", BATCH]
    marie_batch_cli.main(prefix + ["resize", "--format", "web"])
    web = tmp_path / "prepared" / "web" / BATCH

    from image_description.cli import marie_describe_cli
    def fake_process(path, **kwargs):
        assert kwargs["suggest_filename"] is True
        path = Path(path)
        path.with_suffix(".json").write_text(json.dumps({
            "image_filename": path.name,
            "image_relative_path": path.name,
            "suggested_filename": "silver-pendant-necklace",
            "image_description": "A pendant on a chain.",
            "enhanced_description": "Pendant jewellery on a plain background.",
            "keywords": ["pendant"],
        }))
    monkeypatch.setattr(marie_describe_cli, "process_image", fake_process)
    marie_batch_cli.main(prefix + ["describe"])
    original = web / "001-jewellery.jpg"
    assert original.is_file()
    assert (web / "001-jewellery.yaml").is_file()
    assert "A pendant on a chain." in (web / "001-jewellery.yaml").read_text()

    marie_batch_cli.main(prefix + ["review-names"])
    assert original.is_file()
    assert "001-silver-pendant-necklace.jpg" in capsys.readouterr().out

    marie_batch_cli.main(prefix + ["rename"])
    assert original.is_file()  # preview does not apply the name
    marie_batch_cli.main(prefix + ["rename", "--apply"])
    renamed = web / "001-silver-pendant-necklace.jpg"
    assert renamed.is_file()
    assert renamed.with_suffix(".yaml").is_file()
    assert renamed.with_suffix(".json").is_file()
    assert not original.exists()
    assert (source / "jewellery.png").exists()


def test_describe_needs_web_resize_first(tmp_path):
    source = tmp_path / "incoming" / BATCH
    source.mkdir(parents=True)
    with pytest.raises(SystemExit):
        marie_batch_cli.main(["--workspace", str(tmp_path), "--batch", BATCH, "describe"])
