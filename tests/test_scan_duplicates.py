import sys
import json
from pathlib import Path

# Add scripts dir to path so we can import scan_duplicates
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import scan_duplicates as sd


# ── normpath_str ──────────────────────────────────────────────

def test_normpath_str_lowercases():
    assert sd.normpath_str(Path("/Foo/Bar/Baz.JPG")) == "/foo/bar/baz.jpg"

def test_normpath_str_backslashes():
    assert sd.normpath_str(Path("C:\\Users\\Test")) == "c:/users/test"


# ── should_exclude_path ───────────────────────────────────────

def test_should_exclude_matches_substring():
    assert sd.should_exclude_path(Path("/foo/realestate/bar.jpg"), ["realestate"]) is True

def test_should_exclude_case_insensitive():
    assert sd.should_exclude_path(Path("/FOO/REALESTATE/BAR.JPG"), ["realestate"]) is True

def test_should_exclude_no_match():
    assert sd.should_exclude_path(Path("/foo/bar/baz.jpg"), ["realestate"]) is False

def test_should_exclude_empty_list():
    assert sd.should_exclude_path(Path("/foo/bar/baz.jpg"), []) is False


# ── looks_like_thumb_or_icon ──────────────────────────────────

def test_looks_like_thumb_by_filename():
    assert sd.looks_like_thumb_or_icon(Path("/a/b/thumb_test.jpg")) is True
    assert sd.looks_like_thumb_or_icon(Path("/a/b/thumbnail_test.jpg")) is True
    assert sd.looks_like_thumb_or_icon(Path("/a/b/icon_test.jpg")) is True
    assert sd.looks_like_thumb_or_icon(Path("/a/b/preview_test.jpg")) is True

def test_looks_like_thumb_by_folder():
    assert sd.looks_like_thumb_or_icon(Path("/a/thumbnails/test.jpg")) is True

def test_looks_like_normal_file():
    assert sd.looks_like_thumb_or_icon(Path("/a/b/photo123.jpg")) is False
    assert sd.looks_like_thumb_or_icon(Path("/a/b/normal.jpg")) is False


# ── scan_files (integration with tmp_path) ────────────────────

def test_scan_no_files(tmp_path):
    result = sd.scan_files(tmp_path)
    assert result == {}

def test_scan_unique_files(tmp_path):
    (tmp_path / "photo1.jpg").write_text("a")
    (tmp_path / "photo2.jpg").write_text("b")
    result = sd.scan_files(tmp_path)
    assert len(result) == 2
    assert result["photo1.jpg"]["instances"] == []
    assert result["photo2.jpg"]["instances"] == []

def test_scan_duplicates(tmp_path):
    (tmp_path / "photo.jpg").write_text("primary")
    sub = tmp_path / "dup"
    sub.mkdir()
    (sub / "photo.jpg").write_text("duplicate")
    result = sd.scan_files(tmp_path)
    assert len(result) == 1
    record = result["photo.jpg"]
    assert record["instances"] != []
    assert len(record["instances"]) == 1

def test_scan_case_insensitive_duplicates(tmp_path):
    (tmp_path / "Photo.JPG").write_text("upper")
    (tmp_path / "photo.jpg").write_text("lower")
    result = sd.scan_files(tmp_path)
    # Both map to the same key "photo.jpg"
    assert len(result) == 1
    key = list(result.keys())[0]
    assert key == key.lower()
    assert len(result[key]["instances"]) == 1

def test_scan_extension_filter(tmp_path):
    (tmp_path / "a.jpg").write_text("a")
    (tmp_path / "b.png").write_text("b")
    (tmp_path / "c.jpg").write_text("c")
    result = sd.scan_files(tmp_path, extensions=[".jpg"])
    assert len(result) == 2
    assert "b.png" not in result

def test_scan_extension_filter_case_insensitive(tmp_path):
    (tmp_path / "a.JPG").write_text("a")
    result = sd.scan_files(tmp_path, extensions=[".jpg"])
    assert len(result) == 1

def test_scan_exclude_substrings(tmp_path):
    sub = tmp_path / "skipme"
    sub.mkdir()
    (sub / "a.jpg").write_text("skip")
    (tmp_path / "b.jpg").write_text("keep")
    result = sd.scan_files(tmp_path, exclude_substrings=["skipme"])
    assert len(result) == 1
    assert "b.jpg" in result

def test_scan_filters_small_named_files(tmp_path):
    """scan_files should skip files whose name/path contains thumb/icon keywords.
    We scan under a plain subdirectory so the tmp_path prefix (which contains
    the test function name) does not cause accidental matches."""
    plain = tmp_path / "photos"
    plain.mkdir()
    (plain / "thumbnail_photo.jpg").write_text("t")
    (plain / "normal.jpg").write_text("n")
    result = sd.scan_files(plain, skip_icons_thumbs=True)
    assert "thumbnail_photo.jpg" not in result
    assert "normal.jpg" in result

def test_scan_keeps_small_named_files_when_skip_off(tmp_path):
    plain = tmp_path / "photos"
    plain.mkdir()
    (plain / "icon_file.jpg").write_text("i")
    result = sd.scan_files(plain, skip_icons_thumbs=False)
    assert "icon_file.jpg" in result


# ── load_excludes_file ────────────────────────────────────────

def test_load_excludes_file(tmp_path):
    f = tmp_path / "excludes.txt"
    f.write_text("# comment\nskipme\n noskip \n")
    result = sd.load_excludes_file(f)
    assert "skipme" in result
    assert "noskip" in result
    assert not any(line.startswith("#") for line in result)

def test_load_excludes_file_none():
    assert sd.load_excludes_file(None) == []

def test_load_excludes_file_missing():
    assert sd.load_excludes_file(Path("/nonexistent/file.txt")) == []


# ── DEFAULT_EXCLUDES integrity ────────────────────────────────

def test_default_excludes_no_concatenation_bug():
    """Ensure no accidental string concatenation from missing commas."""
    # The bug: "source/repos" "20210713fpe" → "source/repos20210713fpe"
    assert "source/repos" in sd.DEFAULT_EXCLUDES
    assert "20210713fpe" in sd.DEFAULT_EXCLUDES
    bogus = [e for e in sd.DEFAULT_EXCLUDES if "source/repos2021" in e]
    assert bogus == [], f"Found concatenated string: {bogus}"
