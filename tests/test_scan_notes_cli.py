import sys
from pathlib import Path
import json
import yaml
import os

# Ensure package src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from image_description.notes import scan as scan_module
from image_description.cli import scan_notes_cli


def test_build_markdown_includes_fields(tmp_path):
    image = tmp_path / "IMG_0001.jpg"
    image.write_text("fake")
    data = {
        "image_description": "notebook page, black ink, ruled",
        "text": "This is a transcription.",
        "keywords": ["note", "todo"],
        "keywords_image": ["black ink", "handwriting"],
        "issues": [],
    }
    md = scan_module.build_markdown(str(image), data)
    assert md.startswith("---\n")
    # extract yaml between ---
    parts = md.split("---\n")
    assert len(parts) >= 3
    yaml_text = parts[1]
    parsed = yaml.safe_load(yaml_text)
    assert parsed["image_description"] == data["image_description"]
    assert parsed["file_name"] == image.name
    assert parsed["keywords"] == data["keywords"]
    assert parsed["keywords_image"] == data["keywords_image"]
    assert parsed["flags"] == []
    assert md.strip().endswith(data["text"]) or data["text"] in md


def test_process_scan_image_skips_existing_md(tmp_path, monkeypatch):
    # create a small valid image using PIL
    from PIL import Image
    img_path = tmp_path / "page.jpg"
    im = Image.new("RGB", (10, 10), color=(255, 255, 255))
    im.save(img_path)

    md_path = tmp_path / "page.md"
    md_path.write_text("existing")

    called = {"count": 0}

    def fake_call(*a, **k):
        called["count"] += 1
        return {}

    monkeypatch.setattr(scan_module, "_call_model", fake_call)

    created = scan_module.process_scan_image(str(img_path), overwrite=False)
    assert created is False
    assert called["count"] == 0


def test_cli_image_root_rejects_absolute_and_escape(tmp_path, monkeypatch):
    # Test absolute path rejected
    abs_image = tmp_path / "photo.jpg"
    abs_image.write_text("x")

    monkeypatch.chdir(str(tmp_path))

    # absolute path should cause SystemExit via resolve_image_and_relative
    monkeypatch.setenv("PYTHONWARNINGS", "ignore")
    sys_argv = ["scan-notes", "--image-root", str(tmp_path), str(abs_image)]
    monkeypatch.setattr(sys, "argv", sys_argv)
    try:
        scan_notes_cli.main()
    except SystemExit as e:
        assert e.code != 0
    else:
        raise AssertionError("Expected SystemExit for absolute path with --image-root")

    # escaping path should be rejected
    sys_argv = ["scan-notes", "--image-root", str(tmp_path), "../etc/passwd"]
    monkeypatch.setattr(sys, "argv", sys_argv)
    try:
        scan_notes_cli.main()
    except SystemExit as e:
        assert e.code != 0
    else:
        raise AssertionError("Expected SystemExit for escaping path with --image-root")


def test_non_image_file_skipped(tmp_path):
    txt = tmp_path / "not_image.txt"
    txt.write_text("hello")
    result = scan_module.process_scan_image(str(txt))
    assert result is False


def test_flags_empty_when_issues_empty():
    image = "/path/to/IMG_0001.jpg"
    data = {
        "image_description": "desc",
        "text": "t",
        "keywords": [],
        "keywords_image": [],
        "issues": [],
    }
    md = scan_module.build_markdown(image, data)
    parts = md.split("---\n")
    yaml_text = parts[1]
    parsed = yaml.safe_load(yaml_text)
    assert parsed["flags"] == []


def test_preprocess_encodes_rgb_jpeg(tmp_path):
    from PIL import Image
    import base64
    from image_description.image.openai_client import _encode_image_to_base64

    img_path = tmp_path / "page.jpg"
    im = Image.new("RGB", (20, 20), color=(120, 120, 120))
    im.save(img_path)

    b64 = _encode_image_to_base64(str(img_path), preprocess=True)
    data = base64.b64decode(b64)
    assert data[:2] == b"\xff\xd8"  # JPEG magic
    # decode to confirm it is a valid image
    decoded = Image.open(__import__("io").BytesIO(data))
    assert decoded.mode == "RGB"


def test_process_scan_image_forwards_settings(tmp_path, monkeypatch):
    from PIL import Image

    img_path = tmp_path / "page.jpg"
    im = Image.new("RGB", (10, 10), color=(255, 255, 255))
    im.save(img_path)

    captured = {}

    def fake_call(image_path, **kwargs):
        captured.update(kwargs)
        return {
            "image_description": "d",
            "text": "t",
            "keywords": [],
            "keywords_image": [],
            "issues": [],
        }

    monkeypatch.setattr(scan_module, "_call_model", fake_call)
    scan_module.process_scan_image(
        str(img_path),
        overwrite=True,
        model="gpt-4o",
        preprocess=True,
        provider="gemini",
        credential_path="/tmp/creds",
    )
    assert captured["model"] == "gpt-4o"
    assert captured["preprocess"] is True
    assert captured["provider"] == "gemini"
    assert captured["credential_path"] == "/tmp/creds"


def test_call_model_uses_galet_client(tmp_path, monkeypatch):
    """_call_model should go through galet's create_vision_response and parse JSON."""
    from PIL import Image

    img_path = tmp_path / "page.jpg"
    im = Image.new("RGB", (10, 10), color=(255, 255, 255))
    im.save(img_path)

    captured = {}

    def fake_create(prompt, image_b64, **kwargs):
        captured["prompt"] = prompt
        captured["image_b64"] = image_b64
        captured.update(kwargs)
        return json.dumps(
            {
                "image_description": "notebook page",
                "text": "transcribed body",
                "keywords": ["note"],
                "keywords_image": ["ink"],
                "issues": ["[illegible] at line 2"],
            }
        )

    monkeypatch.setattr(
        "image_description.image.galet_client.create_vision_response",
        fake_create,
    )

    data = scan_module._call_model(
        str(img_path),
        model="gpt-4o",
        preprocess=True,
        provider="openai",
        credential_path="/tmp/creds",
    )

    assert data["text"] == "transcribed body"
    assert data["issues"] == ["[illegible] at line 2"]
    assert captured["model"] == "gpt-4o"
    assert captured["provider"] == "openai"
    assert captured["credential_path"] == "/tmp/creds"
    assert captured["temperature"] == 0.2
    assert captured["image_b64"]
