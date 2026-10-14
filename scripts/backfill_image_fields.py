#!/usr/bin/env python3
"""
Backfill image_filename / image_relative_path in older sidecar JSON files.

Some older sidecars were written with empty image fields:

    "image_filename": "",
    "image_relative_path": "",

This script walks a folder of sidecar JSON files, finds sidecars with empty
image fields, locates the matching image by sidecar filename stem
(e.g. r00001234.json -> r00001234.jpg / .jpeg / .png / ...), and writes the
filename and relative path back into the sidecar.

Search order for the image:
  1. next to the sidecar file (same folder)
  2. anywhere under the scan root (folder argument)
  3. anywhere under --search-dir, if given (e.g. a photos archive folder)

The relative path stored is relative to the scan root, matching how
gallery_promote_cli resolves image_relative_path.

Run it from the repo root:

    bash -lc "source .venv/bin/activate && \
        python scripts/backfill_image_fields.py <FOLDER> [--search-dir DIR] [--dry-run]"

Options:
    FOLDER            Folder of sidecar JSON files (walked recursively)
    --search-dir DIR  Additional folder to search for images (optional)
    --dry-run         Show what would change; write nothing
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".gif"]


def build_image_index(roots: List[str]) -> Dict[str, Path]:
    """Index images by lowercase stem across the given root folders."""
    index: Dict[str, Path] = {}
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                path = Path(dirpath) / name
                if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    stem = path.stem.lower()
                    # Prefer the first occurrence (sidecar-local wins by
                    # ordering the scan root first).
                    if stem not in index:
                        index[stem] = path
    return index


def iter_sidecar_paths(folder: str) -> List[str]:
    """Walk `folder` recursively and return every sidecar JSON path."""
    paths: List[str] = []
    for root, _dirs, files in os.walk(folder):
        for name in sorted(files):
            if name.endswith(".json"):
                paths.append(os.path.join(root, name))
    return paths


def needs_backfill(data: Dict) -> bool:
    """True if either image field is missing or empty."""
    filename = (data.get("image_filename") or "").strip()
    relative = (data.get("image_relative_path") or "").strip()
    return not filename or not relative


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Backfill image_filename / image_relative_path in sidecar JSON files."
    )
    parser.add_argument("folder", help="Folder of sidecar JSON files (walked recursively)")
    parser.add_argument("--search-dir", default=None, help="Additional folder to search for images (optional)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change; write nothing")
    args = parser.parse_args(argv)

    sidecar_paths = iter_sidecar_paths(args.folder)
    if not sidecar_paths:
        print(f"No JSON files found under: {args.folder}")
        return

    # Index images: scan root first (sidecar-local matches win), then --search-dir.
    roots = [args.folder] + ([args.search_dir] if args.search_dir else [])
    index = build_image_index(roots)

    updated = 0
    missing = 0
    already_ok = 0
    errors = 0

    for path in sidecar_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  ! unreadable: {path} ({e})", file=sys.stderr)
            errors += 1
            continue

        if not isinstance(data, dict):
            continue

        if not needs_backfill(data):
            already_ok += 1
            continue

        stem = Path(path).stem.lower()
        image = index.get(stem)
        if image is None:
            print(f"  ! no image found for: {path}", file=sys.stderr)
            missing += 1
            continue

        # Relative to the scan root, matching gallery_promote_cli resolution.
        try:
            relative = str(image.relative_to(Path(args.folder)))
        except ValueError:
            # Image found outside the scan root: store an absolute path.
            relative = str(image)

        data["image_filename"] = image.name
        data["image_relative_path"] = relative

        if args.dry_run:
            print(f"  ~ would set {os.path.basename(path)}: {image.name} ({relative})")
        else:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
                f.write("\n")
            os.replace(tmp, path)
            print(f"  + updated {os.path.basename(path)}: {image.name}")
        updated += 1

    print(
        f"\nDone ({'dry run' if args.dry_run else 'written'}): "
        f"{updated} updated, {already_ok} already had image fields, "
        f"{missing} with no image found, {errors} unreadable."
    )
    if missing:
        print("Sidecars with no image found are listed above.", file=sys.stderr)


if __name__ == "__main__":
    main()
