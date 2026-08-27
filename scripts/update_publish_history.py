#!/usr/bin/env python3
"""Update sidecar publish_history and move published work.

Step 2 of the publish-backfill exercise.

For each sidecar JSON under the output dir (excluding publish_data, published_work,
old, bak, __pycache__):

  * If it already has a non-empty publish_history -> move sidecar + image to
    published_work/ (preserving relative subdir).
  * If it has NO publish_history -> match against Mastodon AND Pixelfed post data,
    adding one publish_history entry per matched platform.

Matching: alt-text vs image_description (fallback enhanced_description/title).
Confident when normalized similarity >= 0.95 (true/false gap is ~0.97 vs ~0.79).

Usage:
  python -m scripts.update_publish_history --dry-run
  python -m scripts.update_publish_history            # apply
"""
import argparse
import difflib
import json
import re
import shutil
from pathlib import Path

OUTPUT_ROOT = Path("/home/junwin/pishare/photography/work/2026/output")
PUBLISH_DATA = OUTPUT_ROOT / "publish_data"
PUBLISHED_WORK = OUTPUT_ROOT / "published_work"

DATA_FILES = [
    (PUBLISH_DATA / "masto_data.json", "mastodon"),
    (PUBLISH_DATA / "pixelfed_data.json", "pixelfed"),
]

EXCLUDE_DIRS = {"publish_data", "published_work", "old", "bak", "__pycache__"}
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".gif", ".bmp")
CONFIDENT_RATIO = 0.95


def norm(s):
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def load_posts():
    """Return dict: platform -> list of media post dicts."""
    grouped = {}
    for path, platform in DATA_FILES:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        grouped[platform] = [p for p in data if p.get("has_media")]
    return grouped


def load_sidecars():
    results = []
    for p in OUTPUT_ROOT.rglob("*.json"):
        parts = set(p.relative_to(OUTPUT_ROOT).parts[:-1])
        if parts & EXCLUDE_DIRS:
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        results.append((p, data))
    return results


def find_image(sidecar_path, data):
    d = sidecar_path.parent
    fn = (data.get("image_filename") or "").strip()
    if fn and not fn.startswith("/"):
        cand = d / Path(fn).name
        if cand.exists() and cand.is_file():
            return cand
    for ext in IMAGE_EXTS:
        cand = d / (sidecar_path.stem + ext)
        if cand.exists() and cand.is_file():
            return cand
    return None


def _build_alt_index(posts):
    idx = {}
    for p in posts:
        for alt in p.get("alt_texts", []):
            key = norm(alt)
            if key:
                idx.setdefault(key, []).append(p)
    return idx


def match_one(data, posts):
    """Best match within one platform's posts. Returns (post, ratio) or (None, 0)."""
    if not posts:
        return None, 0.0
    desc = norm(data.get("image_description", ""))
    enh = norm(data.get("enhanced_description", ""))
    title = norm(data.get("title", ""))

    alt_to_posts = _build_alt_index(posts)

    # Exact
    for key in (desc, enh, title):
        if key and key in alt_to_posts and alt_to_posts[key]:
            return alt_to_posts[key][0], 1.0

    # Near
    best, best_ratio = None, 0.0
    for p in posts:
        for alt in p.get("alt_texts", []):
            r = difflib.SequenceMatcher(None, desc, norm(alt)).ratio()
            if r > best_ratio:
                best_ratio, best = r, p
    return best, best_ratio


def publish_entry(post):
    return {
        "platform": post.get("platform", "?"),
        "post_id": str(post.get("post_id", "")),
        "url": post.get("url", ""),
        "date": post.get("date", ""),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Report only, make no changes.")
    args = ap.parse_args()

    grouped = load_posts()
    sidecars = load_sidecars()

    to_move = []          # (path, image) already published
    matched = []          # (path, entries list) newly matched
    low = []              # (path, best_ratio) low-confidence, NOT applied
    unmatched = []        # path with no candidate

    for path, data in sidecars:
        ph = data.get("publish_history", [])
        has_ph = isinstance(ph, list) and len(ph) > 0
        if has_ph:
            to_move.append((path, find_image(path, data)))
            continue

        entries = []
        best_low = 0.0
        for platform, plist in grouped.items():
            post, ratio = match_one(data, plist)
            if post is not None and ratio >= CONFIDENT_RATIO:
                entries.append(publish_entry(post))
            elif ratio > best_low:
                best_low = ratio

        if entries:
            matched.append((path, entries))
        elif best_low > 0:
            low.append((path, best_low))
        else:
            unmatched.append(path)

    print(f"Posts loaded by platform: { {k: len(v) for k, v in grouped.items()} }")
    print(f"Sidecars scanned: {len(sidecars)}")
    print(f"  already published (will move): {len(to_move)}")
    print(f"  newly matched (will update+move): {len(matched)}")
    print(f"  low-confidence (NOT applied): {len(low)}")
    print(f"  unmatched (left in place): {len(unmatched)}")
    print()

    if low:
        print("=== Low-confidence candidates (NOT applied) ===")
        for path, ratio in low:
            print(f"  {ratio:.3f}  {path.relative_to(OUTPUT_ROOT)}")
        print()

    if args.dry_run:
        print("=== DRY RUN: no changes made ===")
        print("\nWill MOVE (already published):")
        for path, img in to_move:
            print(f"  {path.relative_to(OUTPUT_ROOT)}  [+ {img.name if img else 'NO IMAGE'}]")
        print("\nWill UPDATE + MOVE (newly matched):")
        for path, entries in matched:
            plats = ",".join(e["platform"] for e in entries)
            print(f"  {path.relative_to(OUTPUT_ROOT)}  <-  {plats}")
        return

    # --- Apply: update matched sidecars ---
    for path, entries in matched:
        data = json.loads(path.read_text(encoding="utf-8"))
        ph = data.setdefault("publish_history", [])
        if not isinstance(ph, list):
            ph = []
            data["publish_history"] = ph
        ph.extend(entries)
        path.write_text(json.dumps(data, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")

    # --- Apply: move published work ---
    move_list = list(to_move)
    for path, entries in matched:
        data = json.loads(path.read_text(encoding="utf-8"))
        move_list.append((path, find_image(path, data)))

    moved = 0
    for path, img in move_list:
        rel = path.relative_to(OUTPUT_ROOT)
        dest_dir = PUBLISHED_WORK / rel.parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(dest_dir / path.name))
        if img is not None and img.exists():
            shutil.move(str(img), str(dest_dir / img.name))
        moved += 1

    print(f"Updated {len(matched)} sidecars with publish_history.")
    print(f"Moved {moved} sidecars (+ images where found) to published_work/.")


if __name__ == "__main__":
    main()
