import os
import sys
import json
from pathlib import Path

import pytest
from PIL import Image

from src.image_description.notes.scan import build_markdown, process_scan_image
from src.image_description.cli import scan_notes_cli


def test_build_markdown_includes_fields(tmp_path):
    img = tmp_path / "IMG_0001.jpg"
    Image.new("RGB", (10, 10), color=(255, 255, 255)).save(img)

    data = {
        "image_description": "A handwritten note page.",
        "text": "This is the transcribed text.",
        "keywords": ["note", "handwriting"],
        "keywords_image": ["blue ink", "cursive"],
        "issues": [],
    }

    md = build_markdown(str(img), data)
    assert "image_description" in md
    assert "file_name" in md
    assert "keywords:" in md
    assert "keywords_image:" in md
    assert "flags:" in md
    assert "This is the transcribed text." in md


def test_process_scan_image_skips_existing_md(monkeypatch, tmp_path):
    img = tmp_path / "IMG_0002.jpg"
    Image.new("RGB", (10, 10), color=(255, 255, 255)).save(img)
    md = tmp_path / "IMG_0002.md"
    md.write_text("existing")

    called = {"model": False}

    def fake_call(*a, **k):
        called["model"] = True
        return {}

    # monkeypatch the internal _call_model to ensure it is NOT called
    import src.image_description.notes.scan as scan_mod

    monkeypatch.setattr(scan_mod, "_call_model", fake_call)

    created = process_scan_image(str(img), overwrite=False)
    assert created is False
    assert called["model"] is False


def test_image_root_rejects_absolute_and_escape(tmp_path):
    # absolute path should be rejected by resolve_image_and_relative via CLI
    abs_path = str(tmp_path / "sub")
    os.makedirs(abs_path, exist_ok=True)

    sys_argv = sys.argv.copy()
    try:
        sys.argv = ["prog", str(abs_path), "--image-root", str(tmp_path)]
        with pytest.raises(SystemExit):
            scan_notes_cli.main()
    finally:
        sys.argv = sys_argv


def test_non_image_file_skipped(tmp_path):
    txt = tmp_path / "notanimage.txt"
    txt.write_text("hello")

    created = process_scan_image(str(txt), overwrite=True)
    assert created is False


def test_flags_empty_when_no_issues(tmp_path):
    img = tmp_path / "IMG_0003.jpg"
    Image.new("RGB", (10, 10), color=(255, 255, 255)).save(img)

    data = {
        "image_description": "desc",
        "text": "t",
        "keywords": [],
        "keywords_image": [],
        "issues": [],
    }
    md = build_markdown(str(img), data)
    assert "flags: []" in md
