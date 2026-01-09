import argparse
import json
import os
import re
import shutil
from datetime import datetime, date
from typing import Any, Dict, List, Optional


def load_metadata(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def slugify(title: str) -> str:
    """Create a URL-friendly slug from a title."""
    title = title.strip().lower()
    # Replace non-alphanumeric with hyphens
    title = re.sub(r"[^a-z0-9]+", "-", title)
    # Remove leading/trailing hyphens
    title = title.strip("-")
    return title or "post"


def first_sentence(text: str) -> str:
    """Return the first sentence (rough heuristic)."""
    text = text.strip()
    if not text:
        return ""
    # Split on period, question mark, or exclamation mark
    for sep in [". ", "? ", "! "]:
        if sep in text:
            return text.split(sep, 1)[0].strip() + sep.strip()
    return text


def guess_image_web_path(json_path: str, explicit_image: Optional[str]) -> Optional[str]:
    """Return the image path to use in the blog front matter.

    If explicit_image is provided, use that as-is.
    Otherwise, try to guess by replacing .json with common image extensions
    and mapping to a plausible /assets/images/... path if the file exists.
    """
    if explicit_image:
        return explicit_image

    base, _ = os.path.splitext(json_path)
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        candidate = base + ext
        if os.path.exists(candidate):
            filename = os.path.basename(candidate)
            return f"/assets/images/{filename}"
    return None


def copy_image_to_repo(json_path: str, out_root: str, image_web_path: Optional[str]) -> None:
    """Copy the image file into the GitHub repo under the path implied by image_web_path.

    Assumes that if image_web_path starts with /assets/, then the target directory
    is <out_root>/assets/... and the source image is next to the JSON with the
    same base name and a common image extension.
    """
    if not image_web_path:
        return

    # Only handle /assets/... style paths for now
    if not image_web_path.startswith("/assets/"):
        return

    # Determine source image path (next to JSON, same basename)
    base, _ = os.path.splitext(json_path)
    src_path = None
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        candidate = base + ext
        if os.path.exists(candidate):
            src_path = candidate
            break

    if not src_path:
        print("Warning: could not find source image to copy for", json_path)
        return

    # Map /assets/... to <out_root>/assets/...
    rel_assets_path = image_web_path.lstrip("/")  # remove leading slash
    dest_path = os.path.join(out_root, rel_assets_path)
    dest_dir = os.path.dirname(dest_path)
    os.makedirs(dest_dir, exist_ok=True)

    try:
        shutil.copy2(src_path, dest_path)
        print(f"Copied image {src_path} -> {dest_path}")
    except Exception as e:
        print(f"Warning: failed to copy image {src_path} -> {dest_path}: {e}")


def build_front_matter(
    meta: Dict[str, Any],
    date_str: str,
    image_web_path: Optional[str],
    categories: List[str],
) -> str:
    """Build Jekyll YAML front matter for a blog post."""
    title = meta.get("title") or meta.get("original_title") or "Untitled"
    keywords = meta.get("keywords", [])
    if isinstance(keywords, list):
        tags = [str(k) for k in keywords]
    elif keywords:
        tags = [str(keywords)]
    else:
        tags = []

    enhanced_description = meta.get("enhanced_description", "")
    excerpt = first_sentence(enhanced_description)

    # Jekyll-style datetime with timezone; default to 10:00:00 -0500
    try:
        if " " in date_str:
            # Assume user provided full datetime; basic validation
            datetime.strptime(date_str.split(" ")[0], "%Y-%m-%d")
            date_out = date_str
        else:
            datetime.strptime(date_str, "%Y-%m-%d")
            date_out = f"{date_str} 10:00:00 -0500"
    except Exception:
        date_out = f"{date_str} 10:00:00 -0500"

    lines: List[str] = []
    lines.append("---")
    lines.append("layout: post")
    lines.append(f"title: \"{title}\"")
    lines.append(f"date: {date_out}")

    if categories:
        cats = ", ".join(categories)
        lines.append(f"categories: [{cats}]")

    if tags:
        tag_str = ", ".join(tags)
        lines.append(f"tags: [{tag_str}]")

    if image_web_path:
        lines.append(f"image: {image_web_path}")

    if excerpt:
        safe_excerpt = excerpt.replace("\"", "\\\"")
        lines.append(f"excerpt: \"{safe_excerpt}\"")

    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def build_body(meta: Dict[str, Any], image_web_path: Optional[str]) -> str:
    title = meta.get("title") or meta.get("original_title") or "Untitled"
    original_description = meta.get("original_description", "")
    enhanced_description = meta.get("enhanced_description", "")
    visually_challenged_description = meta.get("visually_challenged_description", "")
    keywords = meta.get("keywords", [])
    hashtags = meta.get("hashtags", "")

    lines: List[str] = []

    lines.append(f"# {title}")
    lines.append("")

    if image_web_path:
        lines.append(f"![{title}]({image_web_path})")
        lines.append("")

    if original_description:
        lines.append("## Original notes")
        lines.append(original_description)
        lines.append("")

    if enhanced_description:
        lines.append("## Enhanced description")
        lines.append(enhanced_description)
        lines.append("")

    if visually_challenged_description:
        lines.append("## Description for the visually challenged")
        lines.append(visually_challenged_description)
        lines.append("")

    if keywords:
        if isinstance(keywords, list):
            kw_str = ", ".join(str(k) for k in keywords)
        else:
            kw_str = str(keywords)
        lines.append("## Keywords")
        lines.append(kw_str)
        lines.append("")

    if hashtags:
        lines.append("## Hashtags")
        lines.append(str(hashtags))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Jekyll blog post Markdown file from an image metadata JSON "
            "and write it into a GitHub Pages repo (_posts), also copying the image."
        )
    )
    parser.add_argument("json_path", help="Path to the metadata JSON file.")
    parser.add_argument(
        "--date",
        help=(
            "Post date in YYYY-MM-DD or full 'YYYY-MM-DD HH:MM:SS -ZZZZ' format. "
            "If omitted, today's date is used."
        ),
    )
    parser.add_argument(
        "--out-root",
        required=True,
        help=(
            "Root of the GitHub Pages repo (e.g. /home/junwin/src/repos/junwin.github.io). "
            "The post will be written under _posts/."
        ),
    )
    parser.add_argument(
        "--image",
        help=(
            "Web path to the image for front matter and body (e.g. /assets/images/foo.jpg). "
            "If omitted, a simple guess is made based on the JSON filename."
        ),
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=[],
        help="Optional list of categories for the post.",
    )

    args = parser.parse_args()

    json_path = os.path.abspath(args.json_path)
    meta = load_metadata(json_path)

    # Determine date: use provided or today's date
    if args.date:
        try:
            if " " in args.date:
                datetime.strptime(args.date.split(" ")[0], "%Y-%m-%d")
                date_str = args.date
            else:
                datetime.strptime(args.date, "%Y-%m-%d")
                date_str = args.date
        except ValueError:
            raise SystemExit("--date must be in YYYY-MM-DD or 'YYYY-MM-DD HH:MM:SS -ZZZZ' format")
    else:
        today = date.today().strftime("%Y-%m-%d")
        date_str = today

    out_root = os.path.abspath(args.out_root)

    image_web_path = guess_image_web_path(json_path, args.image)

    # Copy the image into the GitHub repo if possible
    copy_image_to_repo(json_path, out_root, image_web_path)

    front_matter = build_front_matter(meta, date_str, image_web_path, args.categories)
    body = build_body(meta, image_web_path)

    # Use the JSON base filename as the slug, per your request
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    slug = base_name or "post"

    posts_dir = os.path.join(out_root, "_posts")
    os.makedirs(posts_dir, exist_ok=True)

    # Build filename: CURRENT_DATE-slug.md
    today_for_name = date.today().strftime("%Y-%m-%d")
    out_name = f"{today_for_name}-{slug}.md"
    out_path = os.path.join(posts_dir, out_name)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(front_matter)
        f.write(body)

    print(f"Wrote blog post to {out_path}")


if __name__ == "__main__":
    main()
