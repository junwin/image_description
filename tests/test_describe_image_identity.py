import json

import pytest

from image_description.image import describe


def _stub_generation(monkeypatch):
    monkeypatch.setattr(
        describe,
        "show_image_iptc_meta",
        lambda _path: ("Title", "Description", ["keyword"], "Alt text"),
    )
    monkeypatch.setattr(describe, "get_exif_date", lambda _path: "2026-01-02T03:04:05")
    monkeypatch.setattr(
        describe,
        "generate_openai_description_and_keywords",
        lambda *args, **kwargs: ("Generated alt", "Enhanced", "Caption", ["new"]),
    )


def test_process_image_writes_identity_with_image_root(tmp_path, monkeypatch):
    _stub_generation(monkeypatch)
    root = tmp_path / "site"
    folder = root / "assets" / "Button_s Bougie"
    folder.mkdir(parents=True)
    image = folder / "product.jpg"
    image.write_bytes(b"image")

    describe.process_image(
        str(image),
        preset="marie-parie",
        image_root=str(root),
    )

    sidecar = json.loads((folder / "product.json").read_text())
    assert sidecar["image_filename"] == "product.jpg"
    assert sidecar["image_relative_path"] == "assets/Button_s Bougie/product.jpg"


def test_process_image_writes_nonempty_identity_without_image_root(tmp_path, monkeypatch):
    _stub_generation(monkeypatch)
    image = tmp_path / "product.jpg"
    image.write_bytes(b"image")

    describe.process_image(str(image), preset="marie-parie")

    sidecar = json.loads((tmp_path / "product.json").read_text())
    assert sidecar["image_filename"] == "product.jpg"
    assert sidecar["image_relative_path"] == "product.jpg"


def test_process_image_rejects_image_outside_root_before_model_call(tmp_path, monkeypatch):
    root = tmp_path / "site"
    root.mkdir()
    image = tmp_path / "outside.jpg"
    image.write_bytes(b"image")
    model_called = False

    def model(*args, **kwargs):
        nonlocal model_called
        model_called = True
        raise AssertionError("model must not be called")

    monkeypatch.setattr(describe, "generate_openai_description_and_keywords", model)

    with pytest.raises(ValueError, match="outside image root"):
        describe.process_image(
            str(image),
            preset="marie-parie",
            image_root=str(root),
        )

    assert model_called is False
    assert not (tmp_path / "outside.json").exists()
