#!/usr/bin/env python3
"""
CLI to promote top-engagement images to the website gallery.

Uses the `engagement_score` values written into sidecar JSON files by
ratings_cli, ranks them, and for the top N:

  1. copies each image into the website asset folder, and
  2. merges entries into the website gallery.json with alt text taken
     from the sidecar's image_description.

gallery.json entries use the format expected by the gallery page:

    [
      {
        "file": "/assets/images/R0002652.jpg",
        "alt": "A calm water surface reflects a cloudy sky..."
      }
    ]

Existing gallery.json entries are preserved by default: new entries are
appended (images already listed are skipped, not duplicated). Use
--replace to rebuild the list with only the top N.

Run it from the repo root:

    bash -lc "source .venv/bin/activate && \
        python -m src.image_description.cli.gallery_promote_cli <FOLDER> --top 10"

Arguments:
    FOLDER               Folder of sidecar JSON files (walked recursively,
                         same discovery as ratings_cli).

Options:
    --top N              Number of top-ranked images to promote (default 10)
    --assets DIR         Asset folder to copy images into
                         (default /home/junwin/src/repos/junwin.github.io/assets/images)
    --gallery DIR        Gallery folder containing gallery.json
                         (default /home/junwin/src/repos/junwin.github.io/gallery)
    --url-prefix PREFIX  URL path prefix used in gallery.json file entries
                         (default /assets/images)
    --replace            Rebuild gallery.json with only the promoted images
                         instead of merging into the existing list
    --dry-run            Show what would happen; copy and write nothing
    --json               Print machine-readable JSON output
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..sidecar import Sidecar
from .ratings_cli import iter_sidecar_paths

DEFAULT_ASSETS = "/home/junwin/src/repos/junwin.github.io/assets/images"
DEFAULT_GALLERY = "/home/junwin/src/repos/junwin.github.io/gallery"
DEFAULT_URL_PREFIX = "/assets/images"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


# ---------------------------------------------------------------------------
#  ranking
# ---------------------------------------------------------------------------

def rank_sidecars(folder: str) -> List[Dict[str, Any]]:
    """Return sidecars that have an engagement score, ranked by score desc.

    Sidecars without a score (never scanned by ratings_cli) are excluded,
    since ranking them would be meaningless.
    """
    rows: List[Dict[str, Any]] = []
    for path in iter_sidecar_paths(folder):
        sidecar = Sidecar.load(path)
        score = sidecar.extra.get("engagement_score")
        if not isinstance(score, (int, float)):
            continue
        rows.append(
            {
                "score": score,
                "title": sidecar.title or sidecar.original_title or os.path.basename(path),
                "sidecar_path": path,
                "sidecar": sidecar,
            }
        )
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows


# ---------------------------------------------------------------------------
#  image location + copy
# ---------------------------------------------------------------------------

def find_image(sidecar: Sidecar, sidecar_path: str, scan_root: str) -> Optional[Path]:
    """Locate the image file for a sidecar.

    Preference order:
      1. image_relative_path (relative to the scan root)
      2. image_filename (relative to the scan root)
      3. image_filename searched under the scan root
    Returns None if the image cannot be found.
    """
    root = Path(scan_root)
    candidates: List[Path] = []

    relative = sidecar.image_relative_path
    if isinstance(relative, str) and relative.strip():
        rel = relative.strip().replace("\\", "/")
        candidates.append(root / rel)
        candidates.append(Path(sidecar_path).parent / rel)
        candidates.append(Path(rel))  # already absolute or cwd-relative

    filename = sidecar.image_filename if isinstance(sidecar.image_filename, str) else ""
    if filename.strip():
        base = filename.strip().replace("\\", "/").split("/")[-1]
        candidates.append(root / base)
        candidates.append(Path(sidecar_path).parent / base)

    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
            return candidate.resolve()
    return None


def copy_image(image: Path, assets_dir: Path) -> str:
    """Copy an image into the assets dir. Returns the copied file's basename.

    An existing file with the same name is overwritten (same-named images
    are the same photograph, so a refresh keeps the site up to date).
    """
    destination = assets_dir / image.name
    shutil.copy2(str(image), str(destination))
    return destination.name


# ---------------------------------------------------------------------------
#  gallery.json
# ---------------------------------------------------------------------------

def load_gallery(gallery_json: Path) -> List[Dict[str, str]]:
    """Load the existing gallery.json, or return an empty list if missing."""
    if not gallery_json.is_file():
        return []
    with gallery_json.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, list):
        raise ValueError(f"Unexpected gallery.json content (not a list): {gallery_json}")
    return [entry if isinstance(entry, dict) else {} for entry in data]


def write_gallery_json(entries: List[Dict[str, str]], gallery_json: Path) -> None:
    """Atomically replace gallery.json with the given entries."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=str(gallery_json.parent),
            prefix=".gallery-", suffix=".tmp", delete=False,
        ) as stream:
            temporary = stream.name
            json.dump(entries, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary, str(gallery_json))
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def merge_gallery(
    existing: List[Dict[str, str]],
    new_entries: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Append new entries, skipping files already present in the gallery."""
    known_files = {
        (entry.get("file") or "").split("/")[-1] for entry in existing
    }
    merged = list(existing)
    added = 0
    for entry in new_entries:
        basename = (entry.get("file") or "").split("/")[-1]
        if basename in known_files:
            continue
        merged.append(entry)
        known_files.add(basename)
        added += 1
    return merged, added  # type: ignore[return-value]


# ---------------------------------------------------------------------------
#  promote
# ---------------------------------------------------------------------------

def promote(
    folder: str,
    top: int,
    assets: str,
    gallery: str,
    url_prefix: str,
    replace: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    """Promote the top N ranked images to the website gallery.

    Returns a summary dict used for both console and JSON output.
    """
    ranked = rank_sidecars(folder)
    if not ranked:
        raise ValueError(
            f"No sidecars with an engagement_score under {folder}. "
            "Run ratings_cli scan first."
        )

    selected = ranked[: top]
    assets_dir = Path(assets)
    gallery_dir = Path(gallery)
    gallery_json = gallery_dir / "gallery.json"

    new_entries: List[Dict[str, str]] = []
    copied: List[str] = []
    skipped: List[Dict[str, str]] = []

    for row in selected:
        sidecar = row["sidecar"]
        image = find_image(sidecar, row["sidecar_path"], folder)
        if image is None:
            skipped.append(
                {
                    "title": row["title"],
                    "reason": "image file not found "
                              "(check image_relative_path / image_filename)",
                }
            )
            continue

        alt = (sidecar.image_description or "").strip()
        if not alt:
            skipped.append({"title": row["title"], "reason": "no image_description for alt"})
            continue

        entry = {
            "file": f"{url_prefix.rstrip('/')}/{image.name}",
            "alt": alt,
        }
        new_entries.append(entry)

        if not dry_run:
            assets_dir.mkdir(parents=True, exist_ok=True)
            copy_image(image, assets_dir)
        copied.append(image.name)

    if not dry_run and new_entries:
        existing = [] if replace else load_gallery(gallery_json)
        if replace:
            merged, added = list(new_entries), len(new_entries)
        else:
            merged, added = merge_gallery(existing, new_entries)
        gallery_dir.mkdir(parents=True, exist_ok=True)
        write_gallery_json(merged, gallery_json)
    else:
        added = len(new_entries)

    return {
        "scanned_folder": folder,
        "ranked": len(ranked),
        "selected_top": len(selected),
        "copied": copied,
        "gallery_entries": new_entries,
        "added_to_gallery": added,
        "skipped": skipped,
        "gallery_json": str(gallery_json),
        "dry_run": dry_run,
        "replace": replace,
    }


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def cmd_promote(args: argparse.Namespace) -> None:
    summary = promote(
        folder=args.folder,
        top=args.top,
        assets=args.assets,
        gallery=args.gallery,
        url_prefix=args.url_prefix,
        replace=args.replace,
        dry_run=args.dry_run,
    )

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    print(
        f"Ranked {summary['ranked']} sidecar(s); promoting top "
        f"{summary['selected_top']} from {args.folder}"
        f"{' (dry run)' if summary['dry_run'] else ''}..."
    )
    for row in summary["gallery_entries"]:
        print(f"  + {row['file']}")
        print(f"      alt: {row['alt'][:80]}{'...' if len(row['alt']) > 80 else ''}")
    for row in summary["skipped"]:
        print(f"  ! {row['title']}: {row['reason']}")

    verb = "would add" if summary["dry_run"] else "added"
    print(
        f"\nDone. Copied {len(summary['copied'])} image(s) to assets; "
        f"{verb} {summary['added_to_gallery']} entr(y/ies) to "
        f"{summary['gallery_json']}"
        + (" (gallery rebuilt with only the promoted images)" if summary["replace"] else "")
        + "."
    )
    if summary["skipped"]:
        print(f"{len(summary['skipped'])} image(s) skipped - see above.", file=sys.stderr)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Promote the top N engagement-ranked images to the website "
            "gallery: copy images to the asset folder and merge entries "
            "into gallery.json with alt text from the sidecars."
        )
    )
    parser.add_argument("folder", help="Folder of sidecar JSON files (walked recursively)")
    parser.add_argument("--top", type=int, default=10, help="Number of top-ranked images to promote (default 10)")
    parser.add_argument("--assets", default=DEFAULT_ASSETS, help=f"Asset folder for images (default {DEFAULT_ASSETS})")
    parser.add_argument("--gallery", default=DEFAULT_GALLERY, help=f"Gallery folder containing gallery.json (default {DEFAULT_GALLERY})")
    parser.add_argument("--url-prefix", default=DEFAULT_URL_PREFIX, help=f"URL prefix for file entries (default {DEFAULT_URL_PREFIX})")
    parser.add_argument("--replace", action="store_true", help="Rebuild gallery.json with only the promoted images instead of merging")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen; write nothing")
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output")

    args = parser.parse_args(argv)
    try:
        cmd_promote(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
