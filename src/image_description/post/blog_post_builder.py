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
