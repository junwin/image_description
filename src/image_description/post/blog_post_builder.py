import argparse
import os
import re
import shutil
from datetime import date, datetime, timezone
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
    """Guess a web-accessible image path for a sidecar JSON.

    If explicit_image is provided, it is returned unchanged. Otherwise, this
    looks for an image file next to the JSON sharing the same basename and,
    if found, returns a path under /assets/images/.
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
    """Copy the source image (next to json_path) into the out_root at
    image_web_path (must begin with /assets/...). If no source image is
    found or image_web_path is not under /assets/, the function returns and
    prints a warning.
    """
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


def _collect_all_hashtags(metas: List[Dict[str, Any]]) -> List[str]:
    """Collect deduplicated, ordered hashtags from all metas.

    Each meta's 'hashtags' field is a space-separated string like
    '#tag1 #tag2'.  Tags are stripped of the leading '#' and returned
    as a deduplicated list in first-seen order.
    """
    seen: set = set()
    tags: List[str] = []
    for meta in metas:
        raw = str(meta.get("hashtags", ""))
        for chunk in raw.split():
            tag = chunk.lstrip("#").strip()
            if tag and tag not in seen:
                seen.add(tag)
                tags.append(tag)
    return tags


def _clean_hashtags(raw: str) -> str:
    """Strip leading '#' from each tag and return a space-separated string."""
    return " ".join(chunk.lstrip("#") for chunk in raw.split())


def build_front_matter(
    metas: List[Dict[str, Any]],
    date_str: str,
    image_web_path: Optional[str],
    categories: List[str],
    title: Optional[str] = None,
) -> str:
    """Build the Jekyll front matter YAML block from one or more sidecar metas.

    - metas: list of dicts produced by Sidecar.to_dict()
    - date_str: date in YYYY-MM-DD or a full timestamp
    - image_web_path: optional web path to the lead image (e.g. /assets/images/foo.jpg)
    - categories: list of categories
    - title: optional override for the post title. When provided it is used as-is
      (not lowercased). When omitted, the title comes from the first sidecar and is
      lowercased.

    Tags are the deduplicated union of hashtags from ALL sidecars (stripped of '#').

    Timezone policy:
    - If date_str is YYYY-MM-DD, we emit that date with the *current* UTC time.
    - If date_str already includes a time/offset, we pass it through unchanged.
    - If date_str is invalid, we fall back to "now" in UTC.
    """
    if not metas:
        return "---\n---\n"

    first = metas[0]
    if title is not None:
        resolved_title = title.strip()
    else:
        resolved_title = (first.get("title") or first.get("original_title") or "Untitled").strip().lower()
    tags = _collect_all_hashtags(metas)

    enhanced_description = first.get("enhanced_description", "")
    excerpt = first_sentence(enhanced_description)

    try:
        if " " in date_str:
            # Validate the date portion; keep the rest as provided.
            datetime.strptime(date_str.split(" ")[0], "%Y-%m-%d")
            date_out = date_str
        else:
            # Date-only: keep the date, but use current UTC time-of-day.
            datetime.strptime(date_str, "%Y-%m-%d")
            now_utc = datetime.now(timezone.utc)
            date_out = f"{date_str} {now_utc.strftime('%H:%M:%S %z')}"
    except Exception:
        date_out = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %z")

    lines: List[str] = []
    lines.append("---")
    lines.append("layout: post")
    lines.append(f'title: "{resolved_title}"')
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


def build_body_single(meta: Dict[str, Any], image_web_path: Optional[str], subtitle: str) -> str:
    """Build the Markdown body for a single sidecar.

    Structure:
      ## {subtitle}        ← evocative hook
      ## {section_title}   ← per-image heading
      ![...](...)
      ### Original notes
      ### Image description
      ### Hashtags
    """
    section_title = (meta.get("title") or meta.get("original_title") or "Untitled").strip().lower()
    original_description = meta.get("original_description", "")
    image_description = meta.get("image_description", "")
    hashtags = meta.get("hashtags", "")

    lines: List[str] = []

    # Hook
    lines.append(f"## {subtitle}")
    lines.append("")

    # Per-image section
    lines.append(f"## {section_title}")
    lines.append("")

    if image_web_path:
        lines.append(f"![{section_title}]({image_web_path})")
        lines.append("")

    if original_description:
        lines.append("### Original notes")
        lines.append(original_description)
        lines.append("")

    if image_description:
        lines.append("### Image description")
        lines.append(image_description)
        lines.append("")

    if hashtags:
        clean_tags = _clean_hashtags(str(hashtags))
        lines.append("### Hashtags")
        lines.append(clean_tags)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_body_multiple(
    metas: List[Dict[str, Any]],
    image_web_paths: List[Optional[str]],
    subtitle: str,
) -> str:
    """Build the Markdown body for multiple sidecars combined into one post.

    Structure:
      ## {subtitle}           ← evocative hook
      ## {section_title_1}    ← per-image heading
      ![...](...)
      ### Original notes
      ### Image description
      ### Hashtags
      ...repeated for each sidecar...
    """
    if not metas:
        return ""

    lines: List[str] = []

    # Hook
    lines.append(f"## {subtitle}")
    lines.append("")

    for idx, meta in enumerate(metas):
        section_title = (
            meta.get("title")
            or meta.get("original_title")
            or os.path.splitext(os.path.basename(meta.get("__source", "")))[0]
        )
        if not section_title:
            section_title = f"Image {idx + 1}"
        section_title = section_title.strip().lower()

        lines.append(f"## {section_title}")
        lines.append("")

        image_web_path = image_web_paths[idx] if idx < len(image_web_paths) else None
        if image_web_path:
            lines.append(f"![{section_title}]({image_web_path})")
            lines.append("")

        original_description = meta.get("original_description", "")
        image_description = meta.get("image_description", "")
        hashtags = meta.get("hashtags", "")

        if original_description:
            lines.append("### Original notes")
            lines.append(original_description)
            lines.append("")

        if image_description:
            lines.append("### Image description")
            lines.append(image_description)
            lines.append("")

        if hashtags:
            clean_tags = _clean_hashtags(str(hashtags))
            lines.append("### Hashtags")
            lines.append(clean_tags)
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_post_from_json_paths(
    json_paths: List[str],
    out_root: str,
    date_str: str,
    explicit_image: Optional[str],
    categories: List[str],
    subtitle: Optional[str] = None,
    title: Optional[str] = None,
) -> str:
    """Build a Jekyll blog post Markdown file from one or more image metadata JSON files.

    Behavior:
    - If json_paths contains a single file, single-sidecar post.
    - If multiple files, they are combined into one post.
    - Tags in frontmatter are the deduplicated union of all per-image hashtags.
    - Subtitle defaults to the first sidecar's original_description if not provided.
    - Title overrides the post title in front matter. The slug is also derived from
      the title when provided. When omitted, the title comes from the first sidecar.

    Returns the path to the written Markdown file.
    """
    json_paths = [os.path.abspath(p) for p in json_paths]
    out_root = os.path.abspath(out_root)

    # Load sidecars
    metas: List[Dict[str, Any]] = []
    image_web_paths: List[Optional[str]] = []
    for p in json_paths:
        side = Sidecar.load(p)
        d = side.to_dict()
        d["__source"] = p
        metas.append(d)

    # Default subtitle
    if subtitle is None:
        subtitle = metas[0].get("original_description", "") if metas else ""

    # If single sidecar, keep compatibility and allow explicit_image override
    if len(metas) == 1:
        json_path = json_paths[0]

        image_web_path = guess_image_web_path(json_path, explicit_image)
        image_web_paths = [image_web_path]

        copy_image_to_repo(json_path, out_root, image_web_path)

        front_matter = build_front_matter(metas, date_str, image_web_path, categories, title=title)
        body = build_body_single(metas[0], image_web_path, subtitle)

        base_name = os.path.splitext(os.path.basename(json_path))[0]
        slug = slugify(title) if title else base_name or "post"

    else:
        if explicit_image:
            raise SystemExit("--image may not be used when providing multiple JSON sidecars")

        for p in json_paths:
            img = guess_image_web_path(p, None)
            image_web_paths.append(img)
            copy_image_to_repo(p, out_root, img)

        front_image = image_web_paths[0] if image_web_paths else None
        front_matter = build_front_matter(metas, date_str, front_image, categories, title=title)
        body = build_body_multiple(metas, image_web_paths, subtitle)

        if title:
            slug = slugify(title)
        else:
            base_name = os.path.splitext(os.path.basename(json_paths[0]))[0]
            slug = base_name or "post"

    # Write output
    posts_dir = os.path.join(out_root, "_posts")
    os.makedirs(posts_dir, exist_ok=True)

    today_for_name = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_name = f"{today_for_name}-{slug}.md"
    out_path = os.path.join(posts_dir, out_name)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(front_matter)
        f.write(body)

    return out_path


def collect_json_paths(inputs: List[str]) -> List[str]:
    """Resolve CLI inputs (files or directories) into a sorted list of JSON file paths.

    - If a path is a directory, all files ending in .json in that directory are
      included (non-recursive)
    - If a path is a file, it is included.
    - Paths are returned in a deterministic sorted order.
    """
    found: List[str] = []
    for p in inputs:
        p = os.path.abspath(p)
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                if name.lower().endswith(".json"):
                    found.append(os.path.join(p, name))
        elif os.path.isfile(p):
            found.append(p)
        else:
            raise SystemExit(f"Path not found: {p}")

    # Remove duplicates while preserving order
    seen = set()
    out = []
    for p in found:
        if p not in seen:
            seen.add(p)
            out.append(p)
    if not out:
        raise SystemExit("No JSON sidecar files found in the provided paths")
    return out


def main(argv: Optional[List[str]] = None) -> None:
    """Argparse-based CLI entry point for building blog posts from one or more
    image metadata JSON sidecar files.

    Usage:
      python -m image_description.post.blog_post_builder <json-or-dir> [<json-or-dir> ...] \\
          --out-root /path/to/site [--date YYYY-MM-DD] [--image /assets/images/foo.jpg] \\
          [--categories cat1 cat2] [--subtitle "evocative hook"] [--title "Post Title"]

    Notes:
    - You may provide one or more JSON files or directories containing JSON files.
    - When providing multiple JSON sidecars they are combined into a single
      post. In that case --image is invalid (use per-sidecar images next to
      each JSON instead).
    - --subtitle sets the evocative hook that appears as the first ## heading
      in the body. Defaults to the first sidecar's original_description.
    - --title overrides the post title. Defaults to the first sidecar's title.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build a Jekyll blog post Markdown file from one or more image metadata "
            "JSON sidecars and write it into a GitHub Pages repo (_posts), also "
            "copying any adjacent images into /assets."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more JSON sidecar files or directories containing JSON sidecars",
    )
    parser.add_argument(
        "--out-root",
        required=True,
        help="Path to the root of the GitHub Pages/Jekyll repo (contains _posts)",
    )
    parser.add_argument(
        "--date",
        default=date.today().strftime("%Y-%m-%d"),
        help="Post date (YYYY-MM-DD) or full timestamp. If YYYY-MM-DD, UTC is used.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Optional web path to image (single JSON only), e.g. /assets/images/foo.jpg",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=[],
        help="Optional list of categories",
    )
    parser.add_argument(
        "--subtitle",
        default=None,
        help="Evocative hook (## heading) that leads the post body. Defaults to original_description.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Post title override. Defaults to the first sidecar's title (lowercased).",
    )

    args = parser.parse_args(argv)

    json_paths = collect_json_paths(args.inputs)
    out_path = build_post_from_json_paths(
        json_paths=json_paths,
        out_root=args.out_root,
        date_str=args.date,
        explicit_image=args.image,
        categories=args.categories,
        subtitle=args.subtitle,
        title=args.title,
    )
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
