#!/usr/bin/env python3
"""Organize scanned journal notes + images into an Obsidian vault folder.

Given --source-path, --target-path and --relative-folder NAME:

1. Copies *.jpg / *.jpeg / *.png from <source>/NAME to <target>/NAME/assets
   (overwrites existing files by default).
2. For each *.md in <source>/NAME:
   - reads the image file name from the YAML front matter (file_name:)
   - renames the 'keywords:' key to 'tags:' (Obsidian style)
   - sanitizes each tag so Obsidian accepts it (no spaces: 'Chicago flag'
     becomes 'Chicago-flag'; invalid characters are replaced with '-')
   - drops YAML keys that duplicate the embed / dilute search results:
     'image_description', 'keywords_image', 'flags'
   - inserts an Obsidian embed link ![[<file_name>]] right after the
     front matter block (no duplicate if already present)
   - writes the edited file to <target>/NAME (overwrites existing by default)
3. .md files without a YAML front matter block are skipped (warning printed).

Example:
    python scripts/organize_notes.py \
      --source-path /home/junwin/pishare/photography/notes_scan \
      --target-path /home/junwin/Documents/mynotes/journal \
      --relative-folder vol_5
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# Obsidian tags may only contain letters, digits, '_', '-' and '/'.
TAG_INVALID_RUN = re.compile(r"[^A-Za-z0-9_\-/]+")

# Top-level YAML keys to remove entirely (value + list items / continuations).
# These duplicate the embed content and dilute Obsidian search/embed results.
DROP_KEYS = ("image_description", "keywords_image", "flags")


def warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr)


def split_frontmatter(text: str):
    """Split markdown text into (frontmatter_lines, body_lines).

    Returns (None, None) when there is no leading YAML block.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None, None
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return None, None
    return lines[1:end], lines[end + 1:]


def get_yaml_scalar(fm_lines, key):
    for line in fm_lines:
        m = re.match(rf"^{re.escape(key)}\s*:\s*(.*)$", line)
        if m:
            return m.group(1).strip().strip("'\"")
    return None


def sanitize_tag(value: str) -> str:
    """Make a keyword usable as an Obsidian tag.

    Obsidian tags cannot contain spaces or most punctuation. We replace
    runs of invalid characters (including spaces) with '-', collapse
    repeated hyphens and strip leading/trailing hyphens. A leading '#'
    is removed. Falls back to 'untagged' if nothing valid remains.
    """
    v = value.strip()
    if v.startswith("#"):
        v = v[1:]
    v = TAG_INVALID_RUN.sub("-", v)
    v = re.sub(r"-{2,}", "-", v).strip("-")
    return v or "untagged"


def sanitize_tag_block(fm_lines, key_index):
    """Sanitize list items ('- value') directly under a tag-like key."""
    i = key_index + 1
    while i < len(fm_lines):
        line = fm_lines[i]
        if not line.startswith("-"):
            break
        item = line[1:].strip()
        if item:
            fm_lines[i] = f"- {sanitize_tag(item)}"
        i += 1
    return fm_lines


def rename_keywords_to_tags(fm_lines):
    """Rename the top-level 'keywords:' key to 'tags:' (Obsidian style).

    Leaves 'keywords_image:' alone. If 'tags:' already exists, does nothing
    (avoids duplicate keys) and warns. Tag values are sanitized so Obsidian
    recognizes them (no spaces).
    """
    has_tags = any(re.match(r"^tags\s*:", ln) for ln in fm_lines)
    has_keywords = any(re.match(r"^keywords\s*:", ln) for ln in fm_lines)
    if not has_keywords:
        return fm_lines
    if has_tags:
        warn("front matter already has 'tags:'; leaving 'keywords:' unchanged")
        return fm_lines

    # 1) sanitize the tag values in place
    for idx, ln in enumerate(fm_lines):
        if re.match(r"^keywords\s*:", ln):
            sanitize_tag_block(fm_lines, idx)

    # 2) rename the key
    out = []
    for ln in fm_lines:
        if re.match(r"^keywords\s*:", ln):
            ln = re.sub(r"^keywords(\s*:)", r"tags\1", ln, count=1)
        out.append(ln)
    return out


def remove_keys(fm_lines, keys):
    """Remove top-level YAML keys and everything belonging to them.

    Handles block scalars (continuation lines) and unindented list items
    ('- value'), as produced by the scanner. A new top-level key is detected
    by a line starting with 'name:' with no leading whitespace.
    """
    keys = set(keys)
    out = []
    current_key = None
    for line in fm_lines:
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:", line)
        if m:
            current_key = m.group(1)
            if current_key in keys:
                continue
            out.append(line)
        else:
            # list item / continuation line -> belongs to current key
            if current_key not in keys:
                out.append(line)
    return out


def insert_link(body_lines, file_name):
    """Insert ![[file_name]] right after the front matter, if not present."""
    link = f"![[{file_name}]]"
    if any(ln.strip() == link for ln in body_lines):
        return body_lines  # already present
    idx = 0
    while idx < len(body_lines) and body_lines[idx].strip() == "":
        idx += 1
    return body_lines[:idx] + [link, ""] + body_lines[idx:]


def process_md(src_md: Path, target_md: Path, copied_names, dry_run: bool, no_overwrite: bool):
    text = src_md.read_text(encoding="utf-8")
    fm, body = split_frontmatter(text)
    if fm is None:
        warn(f"no YAML front matter, skipping: {src_md.name}")
        return "skipped-no-yaml"

    file_name = get_yaml_scalar(fm, "file_name")
    if not file_name:
        warn(f"no 'file_name' in front matter, skipping: {src_md.name}")
        return "skipped-no-file-name"

    if target_md.exists() and no_overwrite:
        print(f"skipped (exists): {target_md}")
        return "skipped-exists"

    fm = rename_keywords_to_tags(fm)
    fm = remove_keys(fm, DROP_KEYS)
    body = insert_link(body, file_name)

    if file_name.lower() not in copied_names:
        warn(f"image not found for '{file_name}' (referenced by {src_md.name})")

    content = "---\n" + "\n".join(fm) + "\n---\n" + "\n".join(body) + "\n"

    if dry_run:
        print(f"[dry-run] would write: {target_md}")
        return "dry-run"

    target_md.parent.mkdir(parents=True, exist_ok=True)
    target_md.write_text(content, encoding="utf-8")
    print(f"wrote: {target_md}")
    return "written"


def main():
    parser = argparse.ArgumentParser(
        description="Copy scanned images to <target>/<folder>/assets and rewrite "
                    "matching .md notes into <target>/<folder> with Obsidian links."
    )
    parser.add_argument("--source-path", required=True,
                        help="Base source folder (e.g. /home/junwin/pishare/photography/notes_scan)")
    parser.add_argument("--target-path", required=True,
                        help="Base target folder (e.g. /home/junwin/Documents/mynotes/journal)")
    parser.add_argument("--relative-folder", required=True,
                        help="Folder name under both source and target (e.g. vol_5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without writing anything")
    parser.add_argument("--no-overwrite", action="store_true",
                        help="Skip images/.md files that already exist in the target")
    args = parser.parse_args()

    source = Path(args.source_path)
    target = Path(args.target_path)
    folder = args.relative_folder

    if not source.is_dir():
        print(f"error: source path does not exist: {source}", file=sys.stderr)
        sys.exit(2)

    src_dir = source / folder
    if not src_dir.is_dir():
        print(f"error: source folder not found: {src_dir}", file=sys.stderr)
        sys.exit(2)

    assets_dir = target / folder / "assets"

    # 1) copy images
    image_files = sorted(
        p for p in src_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    copied_names = set()
    for img in image_files:
        dest = assets_dir / img.name
        if dest.exists() and args.no_overwrite:
            print(f"skipped (exists): {dest}")
        elif args.dry_run:
            print(f"[dry-run] would copy: {img.name} -> {dest}")
        else:
            assets_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dest)
            print(f"copied: {img.name} -> {dest}")
        copied_names.add(img.name.lower())

    # 2) rewrite .md files
    md_files = sorted(
        p for p in src_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".md"
    )
    if not md_files:
        warn(f"no .md files found in {src_dir}")

    counts = {}
    for md in md_files:
        target_md = target / folder / md.name
        result = process_md(md, target_md, copied_names, args.dry_run, args.no_overwrite)
        counts[result] = counts.get(result, 0) + 1

    print(f"\nsummary: {len(image_files)} images, {len(md_files)} .md files")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")


if __name__ == "__main__":
    main()
