"""Phase 1 review-and-rename regression tests (no AI or website access)."""
import json

import pytest
from PIL import Image

from image_description.cli import marie_describe_cli, marie_rename_cli


def make_prepared(folder, name, suggestion):
    image = folder / name
    Image.new("RGB", (40, 60), "pink").save(image)
    data = {
        "image_filename": name,
        "image_relative_path": name,
        "suggested_filename": suggestion,
        "image_description": "A dark floral top with a row of buttons.",
        "social_caption": "Flowers and buttons!",
        "keywords": ["floral top", "buttons"],
    }
    image.with_suffix(".json").write_text(json.dumps(data), encoding="utf-8")
    image.with_suffix(".yaml").write_text(
        'source: "original-camera-file.png"\n'
        + 'image: ' + json.dumps(name) + '\n'
        + 'suggested_filename: ' + json.dumps(suggestion) + '\n'
        + 'caption: "An edited caption, keep this!"\n',
        encoding="utf-8",
    )
    return image


def test_rename_preview_and_apply_updates_sidecars(tmp_path):
    original = make_prepared(tmp_path, "004-Blue Paisley Tunic Dress.jpg",
                             "004-dark-floral-top.jpg")
    before = original.read_bytes()
    marie_rename_cli.rename(tmp_path)
    assert original.exists()
    assert not (tmp_path / "004-dark-floral-top.jpg").exists()

    # Human review may correct the AI suggestion directly in the YAML.
    yaml_file = original.with_suffix(".yaml")
    updated = yaml_file.read_text().replace(
        "004-dark-floral-top.jpg", "004-dark-floral-button-tunic.jpg"
    )
    yaml_file.write_text(updated)
    marie_rename_cli.rename(tmp_path, apply=True)

    renamed = tmp_path / "004-dark-floral-button-tunic.jpg"
    assert renamed.read_bytes() == before
    assert not original.exists()
    assert not original.with_suffix(".json").exists()
    assert not original.with_suffix(".yaml").exists()
    data = json.loads(renamed.with_suffix(".json").read_text())
    assert data["image_filename"] == renamed.name
    assert data["image_relative_path"] == renamed.name
    assert data["suggested_filename"] == renamed.name
    yaml_text = renamed.with_suffix(".yaml").read_text()
    assert 'image: "004-dark-floral-button-tunic.jpg"' in yaml_text
    assert 'source: "original-camera-file.png"' in yaml_text
    assert 'caption: "An edited caption, keep this!"' in yaml_text
    assert marie_rename_cli.rename(tmp_path) == []


def test_rename_rejects_collision_before_mutation(tmp_path):
    first = make_prepared(tmp_path, "001-old.jpg", "001-floral-top.jpg")
    second = make_prepared(tmp_path, "002-old.jpg", "002-floral-top.jpg")
    (tmp_path / "001-floral-top.jpg").write_bytes(b"already here")
    with pytest.raises(ValueError, match="overwrite"):
        marie_rename_cli.rename(tmp_path, apply=True)
    assert first.exists()
    assert second.exists()


def test_rename_requires_correct_sequence(tmp_path):
    image = make_prepared(tmp_path, "004-old.jpg", "005-wrong-number.jpg")
    with pytest.raises(ValueError, match="sequence"):
        marie_rename_cli.rename(tmp_path)
    assert image.exists()


def test_rename_rejects_traversal(tmp_path):
    image = make_prepared(tmp_path, "004-old.jpg", "../somewhere.jpg")
    with pytest.raises(ValueError, match="Invalid"):
        marie_rename_cli.rename(tmp_path)
    assert image.exists()


def test_describe_repairs_existing_comma_packed_keywords(tmp_path, monkeypatch):
    photo = make_prepared(tmp_path, "004-tunic.jpg", "004-floral-button-tunic.jpg")
    data = json.loads(photo.with_suffix(".json").read_text())
    data["keywords"] = ["floral top, buttons, blue jeans", "buttons"]
    photo.with_suffix(".json").write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(marie_describe_cli, "process_image", lambda *args, **kwargs: None)
    marie_describe_cli.main([str(tmp_path)])
    updated = json.loads(photo.with_suffix(".json").read_text())
    assert updated["keywords"] == ["floral top", "buttons", "blue jeans"]
    yaml_text = photo.with_suffix(".yaml").read_text()
    assert '  - "floral top"' in yaml_text
    assert '  - "blue jeans"' in yaml_text
