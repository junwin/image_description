import argparse
import os
import re
import shutil
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from ..sidecar import Sidecar


def slugify(title: str) -> str:
    title = title.strip().lower()
    title = re.sub(r"[^a-z0-9]+", "-", title)
    title = title.strip("-")
    return title or "post"


def first_sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    for sep in [". ", "? ", "! "]:
        if sep in text:
            return text.split(sep, 1)[0].strip() + sep.strip()
    return text


def guess_image_web_path(json_path: str, explicit_image: Optional[str]) -> Optional[str]:
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
    if not image_web_path:
        return
    if not image_web_path.startswith("/assets/"):
        return

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

    rel_assets_path = image_web_path.lstrip("/")
    dest_path = os.path.join(out_root, rel_assets_path)
    dest_dir = os.path.dirname(dest_path)
    os.makedirs(dest_dir, exist_ok=True)

    try:
        shutil.copy2(src_path, dest_path)
        print(f"Copied image {src_path} -> {dest_path}")
    except Exception as e:  # noqa: BLE001
        print(f"Warning: failed to copy image {src_path} -> {dest_path}: {e}")


def build_front_matter(
    meta: Dict[str, Any],
    date_str: str,
    image_web_path: Optional[str],
    categories: List[str],
) -> str:
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

    try:
        if " " in date_str:
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


def build_post_from_json(
    json_path: str,
    out_root: str,
    date_str: str,
    image_web_path: Optional[str],
    categories: List[str],
) -> str:
    """Build a Jekyll blog post Markdown file from an image metadata JSON file.

    This is the library-level function: it performs the core work and does not
    parse command-line arguments. It returns the path to the written Markdown
    file.
    """
    json_path = os.path.abspath(json_path)
    out_root = os.path.abspath(out_root)

    sidecar = Sidecar.load(json_path)
    meta = sidecar.to_dict()

    image_web_path = guess_image_web_path(json_path, image_web_path)

    copy_image_to_repo(json_path, out_root, image_web_path)

    front_matter = build_front_matter(meta, date_str, image_web_path, categories)
    body = build_body(meta, image_web_path)

    base_name = os.path.splitext(os.path.basename(json_path))[0]
    slug = base_name or "post"

    posts_dir = os.path.join(out_root, "_posts")
    os.makedirs(posts_dir, exist_ok=True)

    today_for_name = date.today().strftime("%Y-%m-%d")
    out_name = f"{today_for_name}-{slug}.md"
    out_path = os.path.join(posts_dir, out_name)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(front_matter)
        f.write(body)

    return out_path


def main(argv: Optional[List[str]] = None) -> None:
    """Argparse-based CLI entry point for building blog posts.

    This CLI is intentionally thin: it parses arguments and calls the
    library-level build_post_from_json() above.

    Parameters:
    - argv: optional list of arguments (for testing). If None, argparse reads
      from sys.argv.
    """
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

    args = parser.parse_args(argv)

    json_path = os.path.abspath(args.json_path)

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

    out_path = build_post_from_json(json_path, out_root, date_str, image_web_path, args.categories)

    # Mirror previous behavior: print the path to the written file
    print(f"Wrote blog post to {out_path}")


if __name__ == "__main__":
    main()


# Manual verification / minimal tests
# - Unit testing: call build_post_from_json() with a small Sidecar JSON file and a
#   temporary out_root, then assert the file exists and contains expected front
#   matter and body sections.
# - CLI smoke test (manual):
#     python -m image_description.post.blog_post_builder path/to/meta.json --out-root /tmp/site
#   should print the output path and create /tmp/site/_posts/YYYY-MM-DD-<slug>.md
# - Packaging note: to expose a console script entrypoint use:
#     blog-post-builder = image_description.post.blog_post_builder:main
#   or if you prefer a dedicated CLI module, add image_description/cli/blog_post_builder_cli.py
#   and point the entrypoint at image_description.cli.blog_post_builder_cli:main
