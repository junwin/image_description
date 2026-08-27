#!/usr/bin/env python3
"""
CLI for publishing to Medium via the REST API (v1).

Medium uses self-issued integration tokens, not OAuth. Generate one at:
    Settings → Security and Apps → Integration tokens

When called from an agent, the command will be in the form:
    bash -lc "source .venv/bin/activate && python -m src.image_description.cli.medium_cli <subcommand> <args>"

Subcommands:
    auth --token <TOKEN>     Save integration token (verified against /v1/me)
    me                       Show authenticated user info
    post <PATH>              Publish a markdown post to Medium
    list                     List recent posts (via RSS feed)
"""

import argparse
import json
import os
import re
import sys
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

import yaml

CRED_PATH = "/home/junwin/credential/medium.json"
MEDIUM_API_BASE = "https://api.medium.com/v1"


# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------

def _load_creds() -> Dict[str, Any]:
    """Load the Medium credential file."""
    if not os.path.exists(CRED_PATH):
        sys.stderr.write(f"Credential file not found: {CRED_PATH}\n")
        sys.stderr.write(
            "Generate an integration token at https://medium.com/me/settings\n"
            "Then run: medium_cli auth --token <TOKEN>\n"
        )
        raise SystemExit(1)
    with open(CRED_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_creds_optional() -> Optional[Dict[str, Any]]:
    """Load creds if they exist, otherwise return None."""
    if not os.path.exists(CRED_PATH):
        return None
    with open(CRED_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_creds(creds: Dict[str, Any]) -> None:
    """Save the credential file atomically."""
    tmp = CRED_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(creds, f, indent=4, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, CRED_PATH)


def _get_token() -> str:
    """Get the stored integration token."""
    creds = _load_creds()
    token = creds.get("integration_token", "").strip()
    if not token:
        sys.stderr.write(
            "No integration token stored. Run 'auth --token <TOKEN>' first.\n"
        )
        raise SystemExit(1)
    return token


def _api_request(
    method: str,
    path: str,
    body: Optional[bytes] = None,
) -> Dict[str, Any]:
    """Make an authenticated request to the Medium API."""
    token = _get_token()
    url = f"{MEDIUM_API_BASE}{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Accept-Charset": "utf-8",
    }
    if body:
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=body, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        sys.stderr.write(f"Medium API error: HTTP {e.code}\n{body_text}\n")
        raise SystemExit(1)


# ---------------------------------------------------------------------------
#  markdown parsing
# ---------------------------------------------------------------------------

_IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_HASHTAG_SECTION_RE = re.compile(r"### Hashtags\n(.+?)(?=\n\n|\Z)", re.DOTALL)


def _strip_hashtag_sections(body: str) -> str:
    """Remove ### Hashtags sections from the markdown body."""
    return _HASHTAG_SECTION_RE.sub("", body)


def _rewrite_image_url(url: str, image_base: str) -> str:
    """Rewrite relative /assets/… URLs to absolute."""
    if url.startswith("/assets/"):
        return image_base + url
    return url


def _parse_jekyll_post(md_path: str, image_base: str = "https://junwin.github.io") -> Dict[str, Any]:
    """Parse a Jekyll markdown post into Medium API payload.

    Returns dict with keys:
        title, content (markdown), tags, canonicalUrl, date
    """
    with open(md_path, "r", encoding="utf-8") as f:
        raw = f.read()

    parts = raw.split("---", 2)
    if len(parts) < 3:
        sys.stderr.write("Invalid Jekyll post: no front matter found.\n")
        raise SystemExit(1)

    front_matter = yaml.safe_load(parts[1])
    if not isinstance(front_matter, dict):
        sys.stderr.write("Invalid Jekyll post: front matter is not a mapping.\n")
        raise SystemExit(1)

    body = parts[2].strip()

    # Strip hashtag sections (redundant on Medium)
    body = _strip_hashtag_sections(body)

    # Rewrite image URLs
    body = _IMG_RE.sub(
        lambda m: f"![{m.group(1)}]({_rewrite_image_url(m.group(2), image_base)})",
        body,
    )

    # Tags: max 5 for Medium
    yaml_tags: List[str] = front_matter.get("tags") or []
    if not isinstance(yaml_tags, list):
        yaml_tags = [str(yaml_tags)]
    tags = [str(t).strip() for t in yaml_tags if str(t).strip()][:5]

    # Canonical URL — if this is cross-posted from johnunwin.com
    canonical = front_matter.get("canonical_url", "")

    # Date from front matter
    date_str = str(front_matter.get("date", ""))

    return {
        "title": str(front_matter.get("title", "")),
        "content": body,
        "tags": tags,
        "canonical_url": str(canonical) if canonical else "",
        "date": date_str,
    }


# ---------------------------------------------------------------------------
#  RSS feed parsing for list subcommand
# ---------------------------------------------------------------------------

def _fetch_rss(username: str) -> List[Dict[str, str]]:
    """Fetch and parse the Medium RSS feed for a user.

    Returns list of dicts with keys: title, link, pubDate, category
    """
    feed_url = f"https://medium.com/feed/@{username}"
    try:
        req = urllib.request.Request(feed_url, headers={"User-Agent": "medium_cli/1.0"})
        with urllib.request.urlopen(req) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        sys.stderr.write(f"Failed to fetch RSS feed: HTTP {e.code}\n")
        sys.stderr.write(
            "Make sure your Medium username is correct and your profile is public.\n"
        )
        raise SystemExit(1)

    root = ET.fromstring(raw)
    posts: List[Dict[str, str]] = []

    for item in root.findall(".//item"):
        title_el = item.find("title")
        link_el = item.find("link")
        pubdate_el = item.find("pubDate")

        categories = [c.text for c in item.findall("category") if c.text]

        posts.append({
            "title": title_el.text if title_el is not None and title_el.text else "(untitled)",
            "link": link_el.text if link_el is not None and link_el.text else "",
            "pubDate": pubdate_el.text if pubdate_el is not None and pubdate_el.text else "",
            "categories": ", ".join(categories),
        })

    return posts


# ---------------------------------------------------------------------------
#  subcommand implementations
# ---------------------------------------------------------------------------

def cmd_auth(token: str) -> None:
    """Save and verify a Medium integration token."""
    if not token:
        sys.stderr.write("--token is required.\n")
        raise SystemExit(1)

    # Verify the token by calling /v1/me
    url = f"{MEDIUM_API_BASE}/me"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    req = urllib.request.Request(url, headers=headers, method="GET")

    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        sys.stderr.write(f"Token verification failed: HTTP {e.code}\n{body}\n")
        raise SystemExit(1)

    user_id = data.get("data", {}).get("id", "")
    username = data.get("data", {}).get("username", "")

    creds = {
        "integration_token": token,
        "user_id": user_id,
        "username": username,
    }
    _save_creds(creds)
    print(f"Token verified. Logged in as @{username} (user ID: {user_id})")


def cmd_me() -> None:
    """Show authenticated user info."""
    data = _api_request("GET", "/me")
    user = data.get("data", {})

    print(f"ID:       {user.get('id', '?')}")
    print(f"Username: {user.get('username', '?')}")
    print(f"Name:     {user.get('name', '?')}")
    print(f"URL:      {user.get('url', '?')}")
    print(f"Image:    {user.get('imageUrl', '?')}")


def cmd_list(limit: int = 20) -> None:
    """List recent Medium posts via RSS feed."""
    creds = _load_creds()
    username = creds.get("username", "").strip()
    if not username:
        sys.stderr.write(
            "No username in credential file. Run 'medium_cli me' first, "
            "or re-authenticate with 'auth --token <TOKEN>'.\n"
        )
        raise SystemExit(1)

    print(f"Fetching posts for @{username} via RSS...\n")
    posts = _fetch_rss(username)

    if not posts:
        print("No posts found.")
        return

    shown = posts[:limit]
    for i, p in enumerate(shown):
        pubdate = p["pubDate"][:16] if p["pubDate"] else "?"
        print(f"{'#' + str(i+1):<4} {pubdate}  {p['title']}")
        if p["categories"]:
            print(f"     Tags: {p['categories']}")
        print(f"     {p['link']}")
        print()

    print(f"{len(shown)} of {len(posts)} post(s) shown.")


def cmd_post(
    md_path: str,
    draft: bool,
    dry_run: bool,
    title_override: Optional[str] = None,
    tags_override: Optional[str] = None,
    canonical_url: Optional[str] = None,
) -> None:
    """Post a Jekyll markdown file to Medium.

    1. Parse the Jekyll .md file (front matter + markdown body).
    2. Strip hashtag sections, rewrite image URLs.
    3. POST to Medium API as markdown.

    Dry-run mode does NOT require credentials — it only parses and previews.
    """
    # Parse the post (no credentials needed)
    post = _parse_jekyll_post(md_path)

    # Apply overrides
    title = title_override or post["title"]
    if tags_override:
        tags = [t.strip() for t in tags_override.split(",") if t.strip()][:5]
    else:
        tags = post["tags"]

    publish_status = "draft" if draft else "public"

    if dry_run:
        creds = _load_creds_optional()
        user_id = creds.get("user_id", "(not configured)") if creds else "(not configured)"
        print("=" * 60)
        print(f"User ID:       {user_id}")
        print(f"Title:         {title}")
        print(f"Status:        {publish_status}")
        print(f"Tags:          {', '.join(tags)}")
        if canonical_url:
            print(f"Canonical URL: {canonical_url}")
        elif post["canonical_url"]:
            print(f"Canonical URL: {post['canonical_url']}")
        print(f"Source:        {md_path}")
        print(f"Date:          {post['date']}")
        print("-" * 60)
        content_preview = post["content"][:1500]
        print(content_preview)
        if len(post["content"]) > 1500:
            print(f"\n[... {len(post['content']) - 1500} more chars ...]")
        print("=" * 60)
        print("DRY RUN — nothing posted.")
        return

    # For real posts, we need credentials
    creds = _load_creds()
    user_id = creds.get("user_id", "").strip()
    if not user_id:
        sys.stderr.write("No user_id in credential file. Run 'me' or re-authenticate.\n")
        raise SystemExit(1)

    # Build payload
    payload: Dict[str, Any] = {
        "title": title,
        "contentFormat": "markdown",
        "content": post["content"],
        "publishStatus": publish_status,
    }
    if tags:
        payload["tags"] = tags
    canonical = canonical_url or post["canonical_url"]
    if canonical:
        payload["canonicalUrl"] = canonical

    body = json.dumps(payload).encode("utf-8")

    result = _api_request("POST", f"/users/{user_id}/posts", body=body)

    post_data = result.get("data", result)
    post_url = post_data.get("url", "")
    post_id = post_data.get("id", "?")

    print(f"Posted! ID={post_id}  status={publish_status}")
    if post_url:
        print(f"URL: {post_url}")


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Publish markdown posts to Medium via REST API."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # auth --token <TOKEN>
    p_auth = sub.add_parser("auth", help="Save and verify Medium integration token")
    p_auth.add_argument(
        "--token",
        required=True,
        help="Medium integration token (from https://medium.com/me/settings)",
    )

    # me
    sub.add_parser("me", help="Show authenticated user info")

    # list [--limit <N>]
    p_list = sub.add_parser("list", help="List recent Medium posts (via RSS)")
    p_list.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of posts to list (default: 20).",
    )

    # post <PATH> [--title <TITLE>] [--tags <TAGS>] [--draft|--publish]
    #            [--dry-run] [--canonical-url <URL>]
    p_post = sub.add_parser("post", help="Publish a markdown post to Medium")
    p_post.add_argument("path", help="Path to Jekyll markdown post (.md)")
    p_post.add_argument("--title", default=None, help="Override post title")
    p_post.add_argument(
        "--tags",
        default=None,
        help="Comma-separated tags (max 5). Overrides front-matter tags.",
    )
    p_post.add_argument(
        "--draft",
        action="store_true",
        default=True,
        help="Post as draft (default).",
    )
    p_post.add_argument(
        "--publish",
        action="store_true",
        help="Publish immediately instead of draft.",
    )
    p_post.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without posting.",
    )
    p_post.add_argument(
        "--canonical-url",
        default=None,
        help="Canonical URL for cross-posting (e.g. your blog URL).",
    )

    args = parser.parse_args(argv)

    if args.command == "auth":
        cmd_auth(args.token)
    elif args.command == "me":
        cmd_me()
    elif args.command == "list":
        cmd_list(limit=args.limit)
    elif args.command == "post":
        is_draft = not args.publish
        cmd_post(
            md_path=args.path,
            draft=is_draft,
            dry_run=args.dry_run,
            title_override=args.title,
            tags_override=args.tags,
            canonical_url=args.canonical_url,
        )
    else:
        parser.print_help()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
