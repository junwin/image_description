import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import copy_primaries as cp


# ── format_date_folder ────────────────────────────────────────

def test_format_date_folder_exif_style():
    assert cp.format_date_folder("2021:12:25 13:28:15") == "2021Dec"

def test_format_date_folder_iso_with_t():
    assert cp.format_date_folder("2022-01-15T08:30:00") == "2022Jan"

def test_format_date_folder_iso_no_t():
    assert cp.format_date_folder("2022-01-15 08:30:00") == "2022Jan"

def test_format_date_folder_date_only():
    assert cp.format_date_folder("2023-06-01") == "2023Jun"

def test_format_date_folder_none():
    assert cp.format_date_folder(None) == "UnknownDate"

def test_format_date_folder_empty():
    assert cp.format_date_folder("") == "UnknownDate"

def test_format_date_folder_garbage():
    assert cp.format_date_folder("not-a-date") == "UnknownDate"

def test_format_date_folder_iso_with_microseconds():
    assert cp.format_date_folder("2024-03-10T11:22:33.123456") == "2024Mar"


# ── copy_primaries ────────────────────────────────────────────

def test_copy_primaries_basic(tmp_path):
    # Create a source file
    src = tmp_path / "source"
    src.mkdir()
    (src / "photo.jpg").write_text("image data")

    # Create the JSON
    report = {
        "photo.jpg": {
            "primary_path": str(src / "photo.jpg"),
            "date_taken": "2021:07:04 10:00:00",
        }
    }
    json_path = tmp_path / "report.json"
    json_path.write_text(json.dumps(report))

    # Target dir
    target = tmp_path / "target"

    cp.copy_primaries(str(json_path), str(target))

    expected = target / "2021Jul" / "photo.jpg"
    assert expected.exists()
    assert expected.read_text() == "image data"

def test_copy_primaries_falls_back_to_created(tmp_path):
    # Create source file
    src = tmp_path / "source"
    src.mkdir()
    (src / "pic.jpg").write_text("data")

    report = {
        "pic.jpg": {
            "primary_path": str(src / "pic.jpg"),
            "created": "2020-03-15T12:00:00",
        }
    }
    json_path = tmp_path / "report.json"
    json_path.write_text(json.dumps(report))

    target = tmp_path / "target"
    cp.copy_primaries(str(json_path), str(target))

    expected = target / "2020Mar" / "pic.jpg"
    assert expected.exists()

def test_copy_primaries_unknown_date(tmp_path):
    src = tmp_path / "source"
    src.mkdir()
    (src / "mystery.jpg").write_text("unknown date")

    report = {
        "mystery.jpg": {
            "primary_path": str(src / "mystery.jpg"),
        }
    }
    json_path = tmp_path / "report.json"
    json_path.write_text(json.dumps(report))

    target = tmp_path / "target"
    cp.copy_primaries(str(json_path), str(target))

    expected = target / "UnknownDate" / "mystery.jpg"
    assert expected.exists()

def test_copy_primaries_multiple_files(tmp_path):
    src = tmp_path / "source"
    src.mkdir()
    (src / "a.jpg").write_text("a")
    (src / "b.jpg").write_text("b")
    (src / "c.jpg").write_text("c")

    report = {
        "a.jpg": {"primary_path": str(src / "a.jpg"), "date_taken": "2021:01:15 00:00:00"},
        "b.jpg": {"primary_path": str(src / "b.jpg"), "date_taken": "2021:06:20 00:00:00"},
        "c.jpg": {"primary_path": str(src / "c.jpg"), "date_taken": "2022:12:01 00:00:00"},
    }
    json_path = tmp_path / "report.json"
    json_path.write_text(json.dumps(report))

    target = tmp_path / "target"
    cp.copy_primaries(str(json_path), str(target))

    assert (target / "2021Jan" / "a.jpg").exists()
    assert (target / "2021Jun" / "b.jpg").exists()
    assert (target / "2022Dec" / "c.jpg").exists()
