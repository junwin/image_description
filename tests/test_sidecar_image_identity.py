"""Regression tests for the image identity contract (Option 1).

Contract:
- ``image_filename`` is the *base* filename of the image, e.g. "IMG_0001.jpg".
  It makes the image<->sidecar link explicit.
- ``image_relative_path`` is the image path relative to ``--image-root`` and may
  contain subdirectories, e.g. "2026/03/IMG_0001.jpg".
- When an image root is supplied, ``image_relative_path`` is the source of truth;
  ``image_filename`` is only a fallback for legacy/top-level sidecars.
"""

import json
from pathlib import Path

import pytest

from image_description.sidecar import Sidecar
from image_description.cli import social_post_builder_cli as spbc
from image_description.cli.social_post_builder_cli import main


def _write_sidecar(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _base_sidecar(**overrides) -> dict:
    data = {
        "original_title": "",
        "original_description": "A photo",
        "title": "",
        "image_description": "",
        "enhanced_description": "",
        "keywords": [],
        "hashtags": "#ok",
        "social_caption": "",
        "image_filename": "",
        "image_relative_path": "",
    }
    data.update(overrides)
    return data


# --- Sidecar.save -----------------------------------------------------------


def test_save_nested_image_relative_path_splits_identity(tmp_path):
    """image_relative_path keeps the subdirectory; image_filename is the basename."""
    root = tmp_path / "images"
    (root / "nested").mkdir(parents=True)

    s = Sidecar(image_relative_path="nested/IMG_0002.jpg")
    out = root / "nested" / "IMG_0002.json"
    s.save(str(out), image_root=str(root))

    assert s.image_filename == "IMG_0002.jpg"
    assert s.image_relative_path == "nested/IMG_0002.jpg"

    on_disk = _read_json(out)
    assert on_disk["image_filename"] == "IMG_0002.jpg"
    assert on_disk["image_relative_path"] == "nested/IMG_0002.jpg"


def test_save_legacy_top_level_sidecar_falls_back_to_filename(tmp_path):
    """A sidecar with only image_filename still resolves at top level."""
    root = tmp_path / "images"
    root.mkdir(parents=True)

    s = Sidecar(image_filename="IMG_0001.jpg", image_relative_path="")
    out = root / "IMG_0001.json"
    s.save(str(out), image_root=str(root))

    assert s.image_filename == "IMG_0001.jpg"
    assert s.image_relative_path == "IMG_0001.jpg"


def test_save_deep_nested_path_preserved(tmp_path):
    root = tmp_path / "images"
    (root / "2026" / "03").mkdir(parents=True)

    s = Sidecar(image_relative_path="2026/03/IMG_0003.jpg")
    s.save(str(root / "2026" / "03" / "IMG_0003.json"), image_root=str(root))

    assert s.image_filename == "IMG_0003.jpg"
    assert s.image_relative_path == "2026/03/IMG_0003.jpg"


def test_save_without_image_root_leaves_identity_untouched(tmp_path):
    """No image_root -> no validation/normalisation of the identity fields."""
    s = Sidecar(image_filename="IMG_0004.jpg", image_relative_path="")
    out = tmp_path / "IMG_0004.json"
    s.save(str(out))

    assert s.image_filename == "IMG_0004.jpg"
    assert s.image_relative_path == ""
    assert _read_json(out)["image_filename"] == "IMG_0004.jpg"


def test_save_rejects_path_escaping_image_root(tmp_path, capsys):
    root = tmp_path / "images"
    root.mkdir(parents=True)

    s = Sidecar(image_relative_path="../outside.jpg")
    with pytest.raises(SystemExit) as exc:
        s.save(str(root / "IMG_0005.json"), image_root=str(root))

    assert exc.value.code == 2
    assert "escapes the image root" in capsys.readouterr().err


def test_save_rejects_absolute_image_path(tmp_path, capsys):
    root = tmp_path / "images"
    root.mkdir(parents=True)

    s = Sidecar(image_relative_path="/etc/passwd")
    with pytest.raises(SystemExit) as exc:
        s.save(str(root / "IMG_0006.json"), image_root=str(root))

    assert exc.value.code == 2
    assert "absolute" in capsys.readouterr().err.lower()


def test_save_rejects_empty_image_identity(tmp_path, capsys):
    root = tmp_path / "images"
    root.mkdir(parents=True)

    s = Sidecar()  # no image_filename, no image_relative_path
    with pytest.raises(SystemExit) as exc:
        s.save(str(root / "IMG_0007.json"), image_root=str(root))

    assert exc.value.code == 2
    assert "neither" in capsys.readouterr().err.lower()


# --- social_post_builder_cli ------------------------------------------------


def _fake_generate_all(sidecar_obj, platforms):
    return {pl: {"text": "x", "hashtags": ["#ok"]} for pl in platforms}


def test_social_cli_records_nested_image_relative_path(tmp_path, monkeypatch):
    """Regression: nested image must not record a bare basename as rel path."""
    root = tmp_path / "images"
    nested = root / "2026" / "03"
    nested.mkdir(parents=True)

    sidecar_path = nested / "IMG_0010.json"
    _write_sidecar(
        sidecar_path,
        _base_sidecar(
            original_title="Nested",
            image_filename="IMG_0010.jpg",
            image_relative_path="2026/03/IMG_0010.jpg",
        ),
    )

    monkeypatch.setattr(spbc, "_generate_social_for_sidecar", _fake_generate_all)

    main(["2026/03/IMG_0010.json", "--image-root", str(root), "--platforms", "mastodon"])

    social_path = Path(Sidecar.social_path_for(str(sidecar_path)))
    assert social_path.exists()

    core = _read_json(social_path)["core"]
    assert core["image_filename"] == "IMG_0010.jpg"
    assert core["image_relative_path"] == "2026/03/IMG_0010.jpg"


def test_social_cli_falls_back_to_image_filename_for_legacy_sidecar(tmp_path, monkeypatch):
    """Legacy sidecar (no image_relative_path) still resolves via image_filename."""
    root = tmp_path / "images"
    root.mkdir(parents=True)

    sidecar_path = root / "IMG_0011.json"
    _write_sidecar(
        sidecar_path,
        _base_sidecar(image_filename="IMG_0011.jpg", image_relative_path=""),
    )

    monkeypatch.setattr(spbc, "_generate_social_for_sidecar", _fake_generate_all)

    main(["IMG_0011.json", "--image-root", str(root), "--platforms", "mastodon"])

    core = _read_json(Path(Sidecar.social_path_for(str(sidecar_path))))["core"]
    assert core["image_filename"] == "IMG_0011.jpg"
    assert core["image_relative_path"] == "IMG_0011.jpg"


def test_social_cli_fails_without_any_image_identity(tmp_path, capsys):
    root = tmp_path / "images"
    root.mkdir(parents=True)

    sidecar_path = root / "IMG_0012.json"
    _write_sidecar(sidecar_path, _base_sidecar())

    with pytest.raises(SystemExit) as exc:
        main(["IMG_0012.json", "--image-root", str(root), "--platforms", "mastodon"])

    assert exc.value.code == 2
    err = capsys.readouterr().err.lower()
    assert "neither" in err
    assert not Path(Sidecar.social_path_for(str(sidecar_path))).exists()
