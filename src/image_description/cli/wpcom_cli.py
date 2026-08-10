#!/usr/bin/env python3
"""
CLI for posting Jekyll markdown posts to WordPress.com via OAuth.

When called from an agent, the command will be in the form:
    bash -lc "source .venv/bin/activate && python -m src.image_description.cli.wpcom_cli <subcommand> <args>"

Subcommands:
    auth-url              Print the WordPress.com OAuth authorize URL
    auth-code <CODE>      Exchange an auth code for an access token
    list                  List published and draft posts
    get <ID|slug|URL>     Fetch a single post by ID, slug, or full URL
    post <PATH>           Post a Jekyll markdown file to WordPress.com
"""

import argparse
import json
import os
import re
import sys
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Tuple

import yaml
import markdown

CRED_PATH = "/home/junwin/credential/wordpress_com.json"
WPCOM_AUTHORIZE_URL = "https://public-api.wordpress.com/oauth2/authorize"
WPCOM_TOKEN_URL = "https://public-api.wordpress.com/oauth2/token"
WPCOM_POST_URL = "https://public-api.wordpress.com/rest/v1.1/sites/{site}/posts/new"
WPCOM_LIST_URL = "https://public-api.wordpress.com/rest/v1.1/sites/{site}/posts"
WPCOM_GET_URL = "https://public-api.wordpress.com/rest/v1.1/sites/{site}/posts/{post_id}"
WPCOM_GET_SLUG_URL = "https://public-api.wordpress.com/rest/v1.1/sites/{site}/posts/slug:{slug}"
IMAGE_BASE_URL = "https://junwin.github.io"


# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------

def _load_creds() -> Dict[str, Any]:
    """Load the WordPress.com credential file."""
    if not os.path.exists(CRED_PATH):
        sys.stderr.write(f"Credential file not found: {CRED_PATH}\n")
        raise SystemExit(1)
    with open(CRED_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_creds(creds: Dict[str, Any]) -> None:
    """Save the credential file atomically (write to temp then rename)."""
    tmp = CRED_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(creds, f, indent=4, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, CRED_PATH)


def _exchange_code_for_token(code: str) -> str:
    """
    Exchange an OAuth authorization code for an access token.
    Returns the access_token string.
    Also saves it into the credential file.
    """
    creds = _load_creds()
    client_id = creds.get("client_id", "").strip()
    client_secret = creds.get("client_secret", "").strip()
    redirect_uri = creds.get("redirect_uri", "http://localhost").strip()

    if not client_id or not client_secret:
        sys.stderr.write("Missing client_id or client_secret in credential file.\n")
        raise SystemExit(1)

    body_data = urllib.parse.urlencode({
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }).encode("utf-8")

    req = urllib.request.Request(
        WPCOM_TOKEN_URL,
        data=body_data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        sys.stderr.write(f"Token exchange failed: HTTP {e.code}\n{body}\n")
        raise SystemExit(1)

    access_token = data.get("access_token")
    if not access_token:
        sys.stderr.write(
            "Token exchange succeeded but no access_token in response.\n"
            f"{json.dumps(data, indent=2)}\n"
        )
        raise SystemExit(1)

    # Persist
    creds["access_token"] = access_token
    _save_creds(creds)

    return access_token


def _get_token(require_fresh: bool, code: Optional[str]) -> str:
    """
    Get a valid access token.
    If code is provided, exchange it (and save).
    Otherwise return the stored token.
    """
    if code:
        return _exchange_code_for_token(code)

    creds = _load_creds()
    token = creds.get("access_token", "").strip()
    if not token:
        sys.stderr.write(
            "No stored access token. Run 'auth-url' to get an authorize URL, "
            "then provide the code via 'auth-code <CODE>' or 'post --code <CODE>'.\n"
        )
        raise SystemExit(1)
    return token


def _get_site(site_override: Optional[str]) -> str:
    """Resolve site: CLI arg first, then credential file."""
    if site_override:
        return site_override.strip()
    creds = _load_creds()
    site = creds.get("site", "").strip()
    if not site:
        sys.stderr.write(
            "No site specified. Provide --site or add 'site' to credential file.\n"
        )
        raise SystemExit(1)
    return site


# ---------------------------------------------------------------------------
#  markdown → HTML conversion
# ---------------------------------------------------------------------------

_IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

# Matches "### Hashtags\n" followed by the tag line until a blank line or end.
_HASHTAG_SECTION_RE = re.compile(r"### Hashtags\n(.+?)(?=\n\n|\Z)", re.DOTALL)


def _rewrite_image_url(url: str) -> str:
    """Rewrite relative /assets/… URLs to absolute IMAGE_BASE_URL/…"""
    if url.startswith("/assets/"):
        return IMAGE_BASE_URL + url
    return url


def _strip_hashtag_sections(body: str) -> str:
    """Remove ### Hashtags sections from the markdown body (they are redundant)."""
    return _HASHTAG_SECTION_RE.sub("", body)


def _convert_markdown_body(body: str, yaml_tags: List[str]) -> str:
    """Strip hashtag sections, rewrite image URLs, convert markdown to HTML.

    Appends tags from the YAML front matter at the very end as a paragraph
    with smaller, non-bold text (no # prefix).
    """
    body = _strip_hashtag_sections(body)
    body = _IMG_RE.sub(lambda m: f"![{m.group(1)}]({_rewrite_image_url(m.group(2))})", body)
    html = markdown.markdown(body, extensions=["extra"])
    if yaml_tags:
        tag_html = (
            '<p class="post-tags">'
            '<span style="font-size: smaller; font-weight: normal;">'
            + ", ".join(yaml_tags) +
            "</span></p>"
        )
        html += "\n" + tag_html
    return html


# ---------------------------------------------------------------------------
#  HTML → markdown conversion (for get subcommand)
# ---------------------------------------------------------------------------

def _html_to_markdown(html: str) -> str:
    """Convert HTML to markdown using html2text if available."""
    try:
        import html2text
    except ImportError:
        sys.stderr.write(
            "The html2text library is required for --format markdown. "
            "Install it with: pip install html2text\n"
        )
        raise SystemExit(1)
    h = html2text.HTML2Text()
    h.body_width = 0          # don't wrap lines
    h.ignore_links = False
    h.ignore_images = False
    h.ignore_emphasis = False
    h.protect_links = True
    return h.handle(html).strip()


# ---------------------------------------------------------------------------
#  Jekyll post parsing
# ---------------------------------------------------------------------------

def _parse_jekyll_post(md_path: str) -> Dict[str, Any]:
    """
    Parse a Jekyll markdown post.

    Returns dict with keys:
        title, date, categories, tags, excerpt, html_content
    """
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    parts = content.split("---", 2)
    if len(parts) < 3:
        sys.stderr.write("Invalid Jekyll post: no front matter found.\n")
        raise SystemExit(1)

    front_matter = yaml.safe_load(parts[1])
    if not isinstance(front_matter, dict):
        sys.stderr.write("Invalid Jekyll post: front matter is not a mapping.\n")
        raise SystemExit(1)

    body = parts[2].strip()

    # YAML tags as a plain list (for display at end of post)
    yaml_tags: List[str] = front_matter.get("tags") or []

    html_content = _convert_markdown_body(body, yaml_tags)

    # Flatten list fields to comma-separated strings for WordPress API
    def _join(val):
        if isinstance(val, list):
            return ", ".join(str(v) for v in val)
        if val is None:
            return ""
        return str(val)

    return {
        "title": front_matter.get("title", ""),
        "date": str(front_matter.get("date", "")),
        "categories": _join(front_matter.get("categories", "")),
        "tags": _join(yaml_tags),
        "excerpt": str(front_matter.get("excerpt", "")),
        "html_content": html_content,
    }


# ---------------------------------------------------------------------------
#  WordPress API calls
# ---------------------------------------------------------------------------

def _post_to_wordpress(
    site: str,
    token: str,
    title: str,
    html_content: str,
    status: str,
    excerpt: str,
    categories: str,
    tags: str,
) -> Dict[str, Any]:
    """POST a new post to WordPress.com. Returns the decoded JSON response."""

    payload = json.dumps({
        "title": title,
        "content": html_content,
        "status": status,
        "excerpt": excerpt,
        "categories": categories,
        "tags": tags,
    }).encode("utf-8")

    url = WPCOM_POST_URL.format(site=site)
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        sys.stderr.write(f"WordPress API error: HTTP {e.code}\n{body}\n")
        raise SystemExit(1)


def _list_posts(
    site: str,
    token: str,
    status: Optional[str] = None,
    number: int = 50,
) -> List[Dict[str, Any]]:
    """Fetch posts from WordPress.com. Returns list of post dicts."""
    params = {"number": str(number)}
    if status:
        params["status"] = status

    url = WPCOM_LIST_URL.format(site=site) + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}"},
        method="GET",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("posts", [])
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        sys.stderr.write(f"WordPress API error: HTTP {e.code}\n{body}\n")
        raise SystemExit(1)


def _get_post(
    site: str,
    token: str,
    post_id: str,
) -> Dict[str, Any]:
    """Fetch a single post by ID from WordPress.com."""
    url = WPCOM_GET_URL.format(site=site, post_id=post_id)
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}"},
        method="GET",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        sys.stderr.write(f"WordPress API error: HTTP {e.code}\n{body}\n")
        raise SystemExit(1)


def _get_post_by_slug(
    site: str,
    token: str,
    slug: str,
) -> Dict[str, Any]:
    """Fetch a single post by slug from WordPress.com."""
    url = WPCOM_GET_SLUG_URL.format(site=site, slug=slug)
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}"},
        method="GET",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        sys.stderr.write(f"WordPress API error: HTTP {e.code}\n{body}\n")
        raise SystemExit(1)


# ---------------------------------------------------------------------------
#  slug / URL parsing for get subcommand
# ---------------------------------------------------------------------------

_SLUG_FROM_URL_RE = re.compile(r"/([^/]+)/?$")


def _is_post_id(value: str) -> bool:
    """Return True if value looks like a numeric post ID."""
    return value.isdigit()


def _extract_slug_from_url(url: str) -> Optional[str]:
    """Extract the slug from a WordPress.com post URL.

    Examples:
        https://johnunwinphotography.blog/2025/05/27/some-post/
        → 'some-post'

        https://johnunwinphotography.blog/2025/05/27/some-post
        → 'some-post'
    """
    # Remove trailing slash then grab last segment
    url = url.rstrip("/")
    m = _SLUG_FROM_URL_RE.search(url)
    if m:
        slug = m.group(1)
        # Exclude date-looking segments like 2025/05/27
        if slug and not re.match(r"^\d{4}$", slug) and not re.match(r"^\d{2}$", slug):
            return slug
    return None


# ---------------------------------------------------------------------------
#  subcommand implementations
# ---------------------------------------------------------------------------

def cmd_auth_url() -> None:
    """Print the OAuth authorize URL so the user can open it in a browser."""
    creds = _load_creds()
    client_id = creds.get("client_id", "").strip()
    redirect_uri = creds.get("redirect_uri", "http://localhost").strip()

    if not client_id:
        sys.stderr.write("Missing client_id in credential file.\n")
        raise SystemExit(1)

    url = (
        f"{WPCOM_AUTHORIZE_URL}"
        f"?client_id={client_id}"
        f"&redirect_uri={redirect_uri}"
        f"&response_type=code"
        f"&scope=global"
    )
    print(url)


def cmd_auth_code(code: str) -> None:
    """Exchange an authorization code for an access token and store it."""
    token = _exchange_code_for_token(code)
    print(f"Access token saved. (first 8 chars: {token[:8]}...)")
    print("You can now use 'post <PATH>' without --code.")


def cmd_list(
    status: Optional[str] = None,
    number: int = 50,
    site: Optional[str] = None,
    code: Optional[str] = None,
) -> None:
    """List WordPress.com posts."""
    token = _get_token(require_fresh=False, code=code)
    resolved_site = _get_site(site)

    posts = _list_posts(site=resolved_site, token=token, status=status, number=number)

    if not posts:
        status_label = status or "any"
        print(f"No {status_label} posts found.")
        return

    # Header
    print(f"{'ID':<8} {'Status':<10} {'Title':<50} URL")
    print("-" * 120)
    for p in posts:
        pid = p.get("ID", "?")
        pstatus = p.get("status", "?")
        title = (p.get("title") or "(untitled)")[:48]
        url = p.get("URL", "-")
        print(f"{pid!s:<8} {pstatus:<10} {title:<50} {url}")

    print(f"\n{len(posts)} post(s) shown.")


def cmd_get(
    identifier: str,
    format: str = "html",
    site: Optional[str] = None,
    code: Optional[str] = None,
) -> None:
    """Fetch a single post by ID, slug, or URL and print its details."""
    token = _get_token(require_fresh=False, code=code)
    resolved_site = _get_site(site)

    # Determine what the identifier is: ID, slug, or URL
    if _is_post_id(identifier):
        post = _get_post(site=resolved_site, token=token, post_id=identifier)
    elif identifier.startswith("http"):
        slug = _extract_slug_from_url(identifier)
        if not slug:
            sys.stderr.write(
                f"Could not extract slug from URL: {identifier}\n"
            )
            raise SystemExit(1)
        post = _get_post_by_slug(site=resolved_site, token=token, slug=slug)
    else:
        # Assume it's a slug
        post = _get_post_by_slug(site=resolved_site, token=token, slug=identifier)

    # Extract fields
    post_id = post.get("ID", "?")
    title = post.get("title", "(untitled)")
    date = post.get("date", "?")
    status = post.get("status", "?")
    url = post.get("URL", "?")
    excerpt = post.get("excerpt", "") or ""
    content_html = post.get("content", "")

    # Categories and tags come back as dicts with "name" keys
    categories_raw = post.get("categories", {}) or {}
    tags_raw = post.get("tags", {}) or {}
    categories = ", ".join(
        c["name"] for c in (categories_raw.values() if isinstance(categories_raw, dict) else categories_raw)
        if isinstance(c, dict) and c.get("name")
    )
    tags = ", ".join(
        t["name"] for t in (tags_raw.values() if isinstance(tags_raw, dict) else tags_raw)
        if isinstance(t, dict) and t.get("name")
    )

    # Output
    print(f"ID:         {post_id}")
    print(f"Title:      {title}")
    print(f"Date:       {date}")
    print(f"Status:     {status}")
    print(f"URL:        {url}")
    print(f"Categories: {categories}")
    print(f"Tags:       {tags}")
    print(f"Excerpt:    {excerpt[:200]}{'...' if len(excerpt) > 200 else ''}")
    print("-" * 60)

    if format == "markdown":
        print(_html_to_markdown(content_html))
    else:
        print(content_html)


def cmd_post(
    md_path: str,
    draft: bool,
    dry_run: bool,
    code: Optional[str] = None,
    site: Optional[str] = None,
) -> None:
    """
    Post a Jekyll markdown file to WordPress.com.

    1. Get/refresh an access token.
    2. Parse the Jekyll .md file (front matter + markdown body).
    3. Append YAML front-matter tags as smaller, non-bold text at end.
    4. Rewrite relative /assets/… image URLs to absolute.
    5. Convert markdown to HTML.
    6. POST to WordPress.com REST API (draft by default).
    """
    token = _get_token(require_fresh=False, code=code)
    resolved_site = _get_site(site)
    status = "draft" if draft else "publish"

    # Parse & convert
    post = _parse_jekyll_post(md_path)

    if dry_run:
        print("=" * 60)
        print(f"Site:       {resolved_site}")
        print(f"Status:     {status}")
        print(f"Title:      {post['title']}")
        print(f"Date:       {post['date']}")
        print(f"Categories: {post['categories']}")
        print(f"Tags:       {post['tags'][:200]}{'...' if len(post['tags']) > 200 else ''}")
        print(f"Excerpt:    {post['excerpt'][:200]}{'...' if len(post['excerpt']) > 200 else ''}")
        print("-" * 60)
        print(post["html_content"][:2000])
        if len(post["html_content"]) > 2000:
            print(f"\n[... {len(post['html_content']) - 2000} more chars ...]")
        print("=" * 60)
        print("DRY RUN — nothing posted.")
        return

    # Post
    result = _post_to_wordpress(
        site=resolved_site,
        token=token,
        title=post["title"],
        html_content=post["html_content"],
        status=status,
        excerpt=post["excerpt"],
        categories=post["categories"],
        tags=post["tags"],
    )

    post_url = result.get("URL", "unknown")
    post_id = result.get("ID", "?")
    print(f"Posted! ID={post_id}  status={status}")
    print(f"URL: {post_url}")


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Post Jekyll markdown posts to WordPress.com via OAuth."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # auth-url
    sub.add_parser("auth-url", help="Print the OAuth authorize URL")

    # auth-code <CODE>
    p_code = sub.add_parser("auth-code", help="Exchange auth code for access token")
    p_code.add_argument("code", help="OAuth authorization code from the redirect URL")

    # list [--status <STATUS>] [--number <N>] [--site <SITE>]
    p_list = sub.add_parser("list", help="List WordPress.com posts")
    p_list.add_argument(
        "--status",
        default=None,
        choices=["publish", "draft", "trash", "future", "private"],
        help="Filter by post status (default: all).",
    )
    p_list.add_argument(
        "--number",
        type=int,
        default=50,
        help="Number of posts to fetch (default: 50).",
    )
    p_list.add_argument(
        "--site",
        default=None,
        help="WordPress.com site domain.",
    )
    p_list.add_argument(
        "--code",
        default=None,
        help="OAuth authorization code.",
    )

    # get <ID|slug|URL> [--format html|markdown] [--site <SITE>] [--code <CODE>]
    p_get = sub.add_parser(
        "get",
        help="Fetch a single post by ID, slug, or full URL",
    )
    p_get.add_argument(
        "identifier",
        help="Post ID, slug, or full URL (e.g. 123, some-post, "
             "https://johnunwinphotography.blog/2025/05/27/some-post/)",
    )
    p_get.add_argument(
        "--format",
        default="html",
        choices=["html", "markdown"],
        help="Output format for post content (default: html).",
    )
    p_get.add_argument(
        "--site",
        default=None,
        help="WordPress.com site domain.",
    )
    p_get.add_argument(
        "--code",
        default=None,
        help="OAuth authorization code.",
    )

    # post <PATH> [--draft|--publish] [--dry-run] [--code <CODE>] [--site <SITE>]
    p_post = sub.add_parser("post", help="Post a markdown file to WordPress.com")
    p_post.add_argument("path", help="Path to Jekyll markdown post (.md)")
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
        help="Convert and print but do not actually post.",
    )
    p_post.add_argument(
        "--code",
        default=None,
        help="OAuth authorization code (optional; exchanges for token then posts).",
    )
    p_post.add_argument(
        "--site",
        default=None,
        help="WordPress.com site domain (e.g. junwin.wordpress.com). "
             "If omitted, reads 'site' from the credential file.",
    )

    args = parser.parse_args(argv)

    if args.command == "auth-url":
        cmd_auth_url()
    elif args.command == "auth-code":
        cmd_auth_code(args.code)
    elif args.command == "list":
        cmd_list(
            status=args.status,
            number=args.number,
            site=args.site,
            code=args.code,
        )
    elif args.command == "get":
        cmd_get(
            identifier=args.identifier,
            format=args.format,
            site=args.site,
            code=args.code,
        )
    elif args.command == "post":
        is_draft = not args.publish
        cmd_post(
            md_path=args.path,
            draft=is_draft,
            dry_run=args.dry_run,
            code=args.code,
            site=args.site,
        )
    else:
        parser.print_help()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
