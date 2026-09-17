import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import re

try:
    from PIL import Image
    from PIL.ExifTags import TAGS
except ImportError:
    Image = None


# ---------- helpers for exclusions ----------

# default path substrings to ignore (case-insensitive; we normalize paths with "/" separators)
DEFAULT_EXCLUDES = [
    "aok",
    "shilpa",
    "fpeshowmay",
    "2022nov/bev",
    "realestate",                    # matches RealEstate / Realestate / realestate
    "rawimages/2020/20200920re/jpeg",
    "rawimages/2020/20200127c",
    "rawimages/2019/c20191012",
    "photo/2017/2017-11-05",
    "junefpe/",
    "2021july/giselle",
    "cynthiabrick",
    "source/repos",
    "20210713fpe",
    "20210623fpe",
    "14009",
    "assydocs",
    "rawimages/2021/jpg",
    "rawimages/2020/20201009re/jpg",
    "pictures/2020_02_20  - fpe",
    "web/",          # often low-res
    "mls_web/",      # often low-res
]

THUMB_ICON_KEYWORDS = {"thumb", "thumbnail", "icon", "preview", "proxy"}

def normpath_str(p: Path) -> str:
    # Lowercase, POSIX separators for uniform substring checks
    return str(p).replace("\\", "/").lower()

def load_excludes_file(path: Path) -> list[str]:
    if not path:
        return []
    out = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s.replace("\\", "/").lower())
    except Exception:
        pass
    return out

def should_exclude_path(p: Path, exclude_substrings: list[str]) -> bool:
    np = normpath_str(p)
    return any(sub in np for sub in exclude_substrings)

def looks_like_thumb_or_icon(p: Path) -> bool:
    # filename or any parent containing typical thumb/icon words
    parts = [part.lower() for part in p.parts]
    stem_lower = p.stem.lower()
    name_checks = parts + [stem_lower]
    return any(any(k in part for k in THUMB_ICON_KEYWORDS) for part in name_checks)

def below_min_long_edge(p: Path, min_long_edge: int) -> bool:
    if min_long_edge <= 0 or not Image:
        return False
    try:
        with Image.open(p) as im:
            w, h = im.size
        return max(w, h) < min_long_edge
    except Exception:
        return False

def get_exif_date_taken(filepath):
    if not Image:
        return None
    try:
        with Image.open(filepath) as img:
            exif_data = img._getexif()
            if not exif_data:
                return None
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == 'DateTimeOriginal':
                    return value
    except Exception:
        pass
    return None


def get_file_metadata(filepath):
    stats = filepath.stat()
    created = datetime.fromtimestamp(stats.st_ctime).isoformat()
    modified = datetime.fromtimestamp(stats.st_mtime).isoformat()

    date_taken = None
    if filepath.suffix.lower() in [".jpg", ".jpeg", ".tiff", ".dng", ".rw2", ".rwl"] and Image:
        date_taken = get_exif_date_taken(str(filepath))

    return {
        "created": created,
        "modified": modified,
        "date_taken": date_taken or modified
    }


def get_sidecar_info(filepath: Path) -> dict:
    """Return sidecar presence for a file. Checks {stem}.json in same directory."""
    sidecar_path = filepath.with_suffix(".json")
    if sidecar_path.exists():
        return {"has_sidecar": True, "sidecar_path": str(sidecar_path)}
    return {"has_sidecar": False, "sidecar_path": None}


def scan_files(start_path, extensions=None, exclude_substrings=None, skip_icons_thumbs=True, min_long_edge=0):
    start_path = Path(start_path)
    exclude_substrings = exclude_substrings or []
    files_by_name = defaultdict(lambda: {
        "root_name": None,
        "filename": None,
        "date_taken": None,
        "created": None,
        "modified": None,
        "primary_path": None,
        "primary_has_sidecar": False,
        "primary_sidecar_path": None,
        "instances": []
    })

    for path in start_path.rglob("*"):
        if not path.is_file():
            continue

        # extension filter
        if extensions and path.suffix.lower() not in extensions:
            continue

        # path substring exclusions
        if should_exclude_path(path, exclude_substrings):
            continue

        # icon/thumbnail heuristics (by name/folders)
        if skip_icons_thumbs and looks_like_thumb_or_icon(path):
            continue

        # optional pixel-size filter for images
        if skip_icons_thumbs and below_min_long_edge(path, min_long_edge):
            continue

        root_name = path.stem
        filename = path.name
        metadata = get_file_metadata(path)
        sidecar = get_sidecar_info(path)

        file_key = filename.lower()
        record = files_by_name[file_key]

        if not record["primary_path"]:
            # First occurrence
            record.update({
                "root_name": root_name,
                "filename": filename,
                "date_taken": metadata["date_taken"],
                "created": metadata["created"],
                "modified": metadata["modified"],
                "primary_path": str(path),
                "primary_has_sidecar": sidecar["has_sidecar"],
                "primary_sidecar_path": sidecar["sidecar_path"],
            })
        else:
            # Duplicate instance — store as object with path + sidecar info
            record["instances"].append({
                "path": str(path),
                "has_sidecar": sidecar["has_sidecar"],
                "sidecar_path": sidecar["sidecar_path"],
            })

    return files_by_name

def main():
    parser = argparse.ArgumentParser(description="Scan folder for duplicate filenames (with exclusions).")
    parser.add_argument("start_path", help="Directory to scan, e.g., E:\\mybook\\picture2018")
    parser.add_argument(
        "--ext", nargs="+", default=[".rwl"],
        help="File extensions to include (e.g., .rwl .jpg .dng)"
    )
    parser.add_argument(
        "--excludes-file", type=Path, default=None,
        help="Path to a text file of substrings to exclude (one per line; comments start with #)."
    )
    parser.add_argument(
        "--exclude-substr", nargs="*", default=[], metavar="SUBSTR",
        help="Additional substrings to exclude (case-insensitive)."
    )
    parser.add_argument(
        "--no-skip-icons-thumbs", action="store_true",
        help="Disable automatic skipping of icons/thumbnails by name."
    )
    parser.add_argument(
        "--min-long-edge", type=int, default=0,
        help="If >0, skip images whose long edge is smaller than this many pixels (requires Pillow)."
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Path to write the JSON report. Default: file_report.json in current dir."
    )

    args = parser.parse_args()

    # Build the exclusion list
    excludes = list(DEFAULT_EXCLUDES)
    if args.excludes_file:
        excludes.extend(load_excludes_file(args.excludes_file))
    if args.exclude_substr:
        # normalize provided substrings
        excludes.extend([s.replace("\\", "/").lower() for s in args.exclude_substr])

    result = scan_files(
        args.start_path,
        extensions=[ext.lower() for ext in args.ext],
        exclude_substrings=excludes,
        skip_icons_thumbs=not args.no_skip_icons_thumbs,
        min_long_edge=args.min_long_edge
    )

    output_path = args.output or Path("file_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Scan complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
