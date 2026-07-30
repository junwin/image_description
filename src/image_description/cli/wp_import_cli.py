#!/usr/bin/env python3
"""
CLI for importing a WordPress.com post into a Jekyll _posts/ markdown file.

Workflow:
  1. Fetch post by slug from WordPress.com (structured data)
  2. Convert HTML content to Markdown
  3. Download images to assets/images/, rewrite URLs
  4. Build Jekyll frontmatter (layout, title, date, categories, tags, image, excerpt)
  5. Write _posts/{date}-{slug}.md

Usage:
    bash -lc "source .venv/bin/activate && python -m src.image_description.cli.wp_import_cli <slug>"
"""

import argparse
import html as html_mod
import json
import os
import re
import sys
import urllib.request
import urllib.error
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

# Import internal helpers from wpcom_cli
from src.image_description.cli.wpcom_cli import (
    _get_post_by_slug,
    _get_token,
    _get_site,
    _html_to_markdown,
)

DEFAULT_JEKYLL_DIR = "/home/junwin/src/repos/junwin.github.io"

_IMG_SRC_RE = re.compile(r'<img[^>]+src=["\']([^"\']+)["\']', re.IGNORECASE)
_IMG_MD_RE = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
_HTML_TAG_RE = re.compile(r'<[^>]+>')


def _clean_plain_text(html_text: str) -> str:
    """Strip HTML tags and decode entities, yielding plain text."""
    text = _HTML_TAG_RE.sub('', html_text)
    text = html_mod.unescape(text)
    text = text.replace('\u00a0', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _normalize_title(title: str) -> str:
    """Collapse whitespace and decode HTML entities in title."""
    title = html_mod.unescape(title)
    return re.sub(r'\s+', ' ', title).strip()


def _extract_image_urls(html: str) -> List[str]:
    """Extract image URLs from HTML content."""
    return _IMG_SRC_RE.findall(html)


def _download_image(url: str, dest_dir: str) -> Optional[str]:
    """Download an image to dest_dir. Returns the local filename, or None on failure."""
    parsed = urlparse(url)
    fname = os.path.basename(parsed.path)
    if not fname:
        fname = "image.jpg"

    dest_path = os.path.join(dest_dir, fname)

    if os.path.exists(dest_path):
        print(f"  (skip, already exists) {fname}")
        return fname

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "wp-import/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        with open(dest_path, "wb") as f:
            f.write(data)
        print(f"  downloaded {fname} ({len(data)} bytes)")
        return fname
    except Exception as e:
        sys.stderr.write(f"  WARNING: failed to download {url}: {e}\n")
        return None


def _rewrite_markdown_images(md: str, url_to_local: Dict[str, str]) -> str:
    """Replace remote image URLs with local /assets/images/ paths in markdown."""
    def _replace(m):
        alt = m.group(1)
        url = m.group(2)
        if url in url_to_local:
            local = url_to_local[url]
            return f"![{alt}](/assets/images/{local})"
        return m.group(0)

    return _IMG_MD_RE.sub(_replace, md)


def _strip_broken_leading_image(md: str, featured_fname: Optional[str]) -> str:
    """Remove the featured image when it's embedded at the top of WP content.

    WP often embeds the featured image as the first element with junk alt text
    like 'picture of garbage'. html2text turns this into:

        ![](/assets/images/fname.jpg)junk text

    (all on one line, no space between the image markdown and the alt text).

    We strip the whole line if it starts with the featured image and the rest
    is just short junk text.
    """
    if not featured_fname:
        return md

    escaped = re.escape(featured_fname)
    # Match: ![](/assets/images/FNAME) followed by junk (same line)
    # The alt text from WP is typically short, lowercase, no punctuation
    pattern = re.compile(
        r'^!\[[^\]]*\]\([^)]*' + escaped + r'\)[a-z\s]{0,50}\s*\n?',
        re.MULTILINE,
    )
    md = pattern.sub('', md, count=1)

    return md.strip()


def _build_jekyll_post(
    title: str,
    date_str: str,
    slug: str,
    categories: List[str],
    tags: List[str],
    image: Optional[str],
    excerpt: str,
    markdown_body: str,
) -> str:
    """Build a complete Jekyll .md post string."""
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        date_formatted = dt.strftime("%Y-%m-%d %H:%M:%S %z")
    except (ValueError, AttributeError):
        date_formatted = date_str

    cats_yaml = json.dumps(categories) if categories else "[]"
    tags_yaml = json.dumps(tags)
    img_line = f"\nimage: /assets/images/{image}" if image else ""
    excerpt_clean = excerpt.replace('"', '\\"')

    frontmatter = f"""---
layout: post
title: "{title}"
date: {date_formatted}
categories: {cats_yaml}
tags: {tags_yaml}{img_line}
excerpt: "{excerpt_clean}"
---

"""

    body = markdown_body.strip()
    if image:
        body = f"![{title}](/assets/images/{image})\n\n{body}"

    return frontmatter + body + "\n"


def cmd_import(
    slug: str,
    jekyll_dir: str,
    dry_run: bool = False,
    overwrite: bool = False,
    site: Optional[str] = None,
    code: Optional[str] = None,
) -> None:
    """Import a WordPress.com post by slug into Jekyll _posts/."""

    # 1. Get auth & fetch post
    token = _get_token(require_fresh=False, code=code)
    resolved_site = _get_site(site)
    post = _get_post_by_slug(site=resolved_site, token=token, slug=slug)

    # 2. Extract metadata
    title = _normalize_title(post.get("title", "(untitled)"))
    date_raw = post.get("date", "")
    wp_slug = post.get("slug", slug)
    excerpt_raw = _clean_plain_text(post.get("excerpt", "") or "")
    content_html = post.get("content", "")

    categories_raw = post.get("categories", {}) or {}
    tags_raw = post.get("tags", {}) or {}

    categories = [
        c["name"] for c in (categories_raw.values() if isinstance(categories_raw, dict) else categories_raw)
        if isinstance(c, dict) and c.get("name")
    ]
    tags = [
        t["name"] for t in (tags_raw.values() if isinstance(tags_raw, dict) else tags_raw)
        if isinstance(t, dict) and t.get("name")
    ]

    categories = [c for c in categories if c.lower() != "uncategorized"]
    if not categories:
        categories = ["reflections"]

    featured_image_url = post.get("featured_image", "") or ""
    featured_image = None

    # 3. Convert HTML → Markdown
    markdown_body = _html_to_markdown(content_html)

    # 4. Find & download images
    img_urls = _extract_image_urls(content_html)
    assets_dir = os.path.join(jekyll_dir, "assets", "images")
    os.makedirs(assets_dir, exist_ok=True)

    url_to_local: Dict[str, str] = {}
    if img_urls:
        print(f"Found {len(img_urls)} image(s):")
        for url in img_urls:
            fname = _download_image(url, assets_dir)
            if fname:
                url_to_local[url] = fname
                if not featured_image:
                    featured_image = fname

    if featured_image_url and not featured_image:
        fname = _download_image(featured_image_url, assets_dir)
        if fname:
            featured_image = fname

    # 5. Rewrite image URLs in markdown
    if url_to_local:
        markdown_body = _rewrite_markdown_images(markdown_body, url_to_local)

    # Strip redundant embedded featured image (junk alt text from WP)
    markdown_body = _strip_broken_leading_image(markdown_body, featured_image)

    # 6. Build the Jekyll post
    jekyll_post = _build_jekyll_post(
        title=title,
        date_str=date_raw,
        slug=wp_slug,
        categories=categories,
        tags=tags,
        image=featured_image,
        excerpt=excerpt_raw,
        markdown_body=markdown_body,
    )

    # 7. Write to _posts/
    try:
        dt = datetime.fromisoformat(date_raw.replace("Z", "+00:00"))
        date_prefix = dt.strftime("%Y-%m-%d")
    except (ValueError, AttributeError):
        date_prefix = date_raw[:10] if len(date_raw) >= 10 else "0000-00-00"

    filename = f"{date_prefix}-{wp_slug}.md"
    dest_path = os.path.join(jekyll_dir, "_posts", filename)

    if dry_run:
        print(f"\n--- DRY RUN: would write {dest_path} ---")
        print(jekyll_post[:2000])
        if len(jekyll_post) > 2000:
            print(f"\n[... {len(jekyll_post) - 2000} more chars ...]")
        print("--- END DRY RUN ---")
        return

    if os.path.exists(dest_path) and not overwrite:
        sys.stderr.write(f"File already exists: {dest_path}\n")
        sys.stderr.write("Use --overwrite to replace.\n")
        raise SystemExit(1)

    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(jekyll_post)

    print(f"\nImported: {dest_path}")
    print(f"  title:      {title}")
    print(f"  date:       {date_prefix}")
    print(f"  categories: {categories}")
    print(f"  tags:       {len(tags)} tag(s)")
    if featured_image:
        print(f"  image:      {featured_image}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Import a WordPress.com post into a Jekyll _posts/ markdown file."
    )
    parser.add_argument(
        "slug",
        help="WordPress.com post slug (e.g. 'breathing-in-i-know-i-am-breathing-in')",
    )
    parser.add_argument(
        "--jekyll-dir",
        default=DEFAULT_JEKYLL_DIR,
        help=f"Path to Jekyll repo (default: {DEFAULT_JEKYLL_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without writing files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing _posts/ file if present.",
    )
    parser.add_argument(
        "--site",
        default=None,
        help="WordPress.com site domain.",
    )
    parser.add_argument(
        "--code",
        default=None,
        help="OAuth authorization code.",
    )

    args = parser.parse_args(argv)

    cmd_import(
        slug=args.slug,
        jekyll_dir=args.jekyll_dir,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        site=args.site,
        code=args.code,
    )


if __name__ == "__main__":
    main()
