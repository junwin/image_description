import json
from pathlib import Path

from image_description.sidecar import Sidecar


def test_sidecar_has_image_filename_key_when_missing_and_when_saved(tmp_path):
    # Original dict does NOT include image_filename
    original = {
        "original_title": "NoImage",
        "original_description": "No image filename provided",
        "title": "NoImage",
        "visually_challenged_description": "",
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
