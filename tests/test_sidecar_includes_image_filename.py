import json
from pathlib import Path

from image_description.sidecar import Sidecar


def test_sidecar_has_image_filename_key_when_missing_and_when_saved(tmp_path):
    # Original dict does NOT include image_filename
    original = {
        "original_title": "NoImage",
        "original_description": "No image filename provided",
        "title": "NoImage",
        "image_description": "",
        "enhanced_description": "",
        "keywords": [],
        "hashtags": "",
        "social_caption": "",
    }

    s = Sidecar.from_dict(original)

    # to_dict must always include the image_filename key (backwards compatible)
    d = s.to_dict()
    assert "image_filename" in d
    assert d["image_filename"] == ""

    # Saving should write the key to disk as well
    p = tmp_path / "IMG_0001.json"
    s.save(str(p))

    with open(p, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert "image_filename" in loaded
    assert loaded["image_filename"] == ""


def test_sidecar_backward_compat_reads_legacy_key(tmp_path):
    """Sidecar should read 'visually_challenged_description' from old JSON and
    expose it via the image_description field."""
    legacy = {
        "original_title": "Old",
        "original_description": "Old file",
        "title": "Old",
        "visually_challenged_description": "A legacy description",
        "enhanced_description": "",
        "keywords": [],
        "hashtags": "",
        "social_caption": "",
        "image_filename": "old.jpg",
    }

    p = tmp_path / "old_sidecar.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(legacy, f)

    s = Sidecar.load(str(p))
    assert s.image_description == "A legacy description"
    # legacy property also works
    assert s.visually_challenged_description == "A legacy description"


def test_sidecar_writes_new_key_only(tmp_path):
    """New sidecars should write 'image_description', not the legacy key."""
    s = Sidecar(
        original_title="New",
        original_description="Fresh",
        title="New",
        image_description="New description",
    )
    p = tmp_path / "new_sidecar.json"
    s.save(str(p))

    with open(p, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert "image_description" in loaded
    assert loaded["image_description"] == "New description"
    assert "visually_challenged_description" not in loaded
