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

    monkeypatch.setattr(scan_module, "_call_openai", fake_call)

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
