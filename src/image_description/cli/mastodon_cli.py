#!/usr/bin/env python3
"""
CLI for posting images to Mastodon via OAuth.

When called from an agent, the command will be in the form:
    bash -lc "source .venv/bin/activate && python -m src.image_description.cli.mastodon_cli <subcommand> <args>"

Subcommands:
    auth-url              Print the Mastodon OAuth authorize URL
    auth-code <CODE>      Exchange an auth code for an access token
    post <PATH>           Post an image (with sidecar JSON) to Mastodon
    list                  List recent posts
    delete <ID>           Delete a post
"""

import argparse
import json
import mimetypes
import os
import re
import sys
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional
from pathlib import Path

CRED_PATH = "/home/junwin/credential/mastodon.json"


# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------

def _load_creds() -> Dict[str, Any]:
    """Load the Mastodon credential file."""
    if not os.path.exists(CRED_PATH):
        sys.stderr.write(f"Credential file not found: {CRED_PATH}\n")
        raise SystemExit(1)
    with open(CRED_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_creds(creds: Dict[str, Any]) -> None:
    """Save the credential file atomically."""
    tmp = CRED_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(creds, f, indent=4, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, CRED_PATH)


def _get_instance(creds: Dict[str, Any]) -> str:
    instance = creds.get("instance", "").strip()
    if not instance:
        sys.stderr.write("Missing 'instance' in credential file.\n")
        raise SystemExit(1)
    return instance


def _exchange_code_for_token(code: str) -> str:
    """Exchange an OAuth authorization code for an access token. Saves to cred file."""
    creds = _load_creds()
    instance = _get_instance(creds)
    client_id = creds.get("client_id", "").strip()
    client_secret = creds.get("client_secret", "").strip()
    redirect_uri = "urn:ietf:wg:oauth:2.0:oob"

    if not client_id or not client_secret:
        sys.stderr.write("Missing client_id or client_secret in credential file.\n")
        raise SystemExit(1)

    token_url = f"https://{instance}/oauth/token"
    body_data = urllib.parse.urlencode({
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
        "scope": "read write",
    }).encode("utf-8")

    req = urllib.request.Request(
        token_url,
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

    creds["access_token"] = access_token
    _save_creds(creds)
    return access_token


def _get_token(code: Optional[str] = None) -> str:
    """Get a valid access token. If code provided, exchange it first."""
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


def _api_request(
    instance: str,
    token: str,
    method: str,
    path: str,
    body: Optional[bytes] = None,
    content_type: str = "application/json",
) -> Dict[str, Any]:
    """Make an authenticated request to the Mastodon API."""
    url = f"https://{instance}{path}"
    headers = {"Authorization": f"Bearer {token}"}
    if body:
        headers["Content-Type"] = content_type

    req = urllib.request.Request(url, data=body, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        sys.stderr.write(f"Mastodon API error: HTTP {e.code}\n{body_text}\n")
        raise SystemExit(1)


# ---------------------------------------------------------------------------
#  sidecar helper
# ---------------------------------------------------------------------------

def _find_sidecar(image_path: str) -> Optional[str]:
    """Find the sidecar JSON for an image. Checks .json and .social.json variants.

    For photo.jpg, checks:
      - photo.json
      - photo.social.json
    Returns the path if found, or None.
    """
    base = os.path.splitext(image_path)[0]
    for suffix in (".json", ".social.json"):
        candidate = base + suffix
        if os.path.exists(candidate):
            return candidate
    return None


def _load_sidecar(sidecar_path: str) -> Dict[str, Any]:
    """Load a sidecar JSON file."""
    with open(sidecar_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
#  media upload
# ---------------------------------------------------------------------------

def _upload_media(
    instance: str,
    token: str,
    image_path: str,
    description: str,
) -> str:
    """Upload an image to Mastodon. Returns the media ID."""
    if not os.path.exists(image_path):
        sys.stderr.write(f"Image not found: {image_path}\n")
        raise SystemExit(1)

    filename = os.path.basename(image_path)
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"

    # Read file bytes
    with open(image_path, "rb") as f:
        file_bytes = f.read()

    # Build multipart form data
    boundary = "----FormBoundaryMastodonCLI"
    body_parts: List[bytes] = []

    # File field
    body_parts.append(f"--{boundary}\r\n".encode("utf-8"))
    body_parts.append(
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        .encode("utf-8")
    )
    body_parts.append(f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"))
    body_parts.append(file_bytes)
    body_parts.append(b"\r\n")

    # Description field
    if description:
        body_parts.append(f"--{boundary}\r\n".encode("utf-8"))
        body_parts.append(
            'Content-Disposition: form-data; name="description"\r\n\r\n'
            .encode("utf-8")
        )
        body_parts.append(description.encode("utf-8"))
        body_parts.append(b"\r\n")

    body_parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(body_parts)

    content_type = f"multipart/form-data; boundary={boundary}"

    result = _api_request(
        instance=instance,
        token=token,
        method="POST",
        path="/api/v2/media",
        body=body,
        content_type=content_type,
    )

    media_id = result.get("id")
    if not media_id:
        sys.stderr.write(
            "Media upload succeeded but no id in response.\n"
            f"{json.dumps(result, indent=2)}\n"
        )
        raise SystemExit(1)
    return str(media_id)


# ---------------------------------------------------------------------------
#  post status
# ---------------------------------------------------------------------------

def _post_status(
    instance: str,
    token: str,
    status_text: str,
    media_ids: List[str],
    visibility: str = "public",
) -> Dict[str, Any]:
    """Post a status (optionally with media) to Mastodon."""
    payload: Dict[str, Any] = {
        "status": status_text,
        "visibility": visibility,
    }
    if media_ids:
        payload["media_ids"] = media_ids

    body = json.dumps(payload).encode("utf-8")
    return _api_request(
        instance=instance,
        token=token,
        method="POST",
        path="/api/v1/statuses",
        body=body,
    )


# ---------------------------------------------------------------------------
#  hashtag processing
# ---------------------------------------------------------------------------

# Tags to remove from sidecar hashtags
_REMOVE_TAGS = {"#places", "#genre"}

# Tags to always prepend (deduplicated if already present)
_ALWAYS_PREPEND = ["#photography", "#photo"]


def _process_hashtags(raw_hashtags: str) -> str:
    """Process hashtags from the sidecar.

    1. Remove #places and #genre.
    2. Prepend #photography and #photo (deduped).
    """
    # Split into individual tags, stripping whitespace and empty strings
    tags = [t.strip() for t in raw_hashtags.split() if t.strip()]

    # 1. Remove unwanted tags
    tags = [t for t in tags if t not in _REMOVE_TAGS]

    # 2. Prepend always-tags, deduping
    result: List[str] = []
    for prepend_tag in _ALWAYS_PREPEND:
        if prepend_tag in tags:
            tags.remove(prepend_tag)
        result.append(prepend_tag)
    result.extend(tags)

    return " ".join(result)


# ---------------------------------------------------------------------------
#  build status text
# ---------------------------------------------------------------------------

def _build_status_text(
    sidecar: Dict[str, Any],
    extra_text: Optional[str],
    visibility: str,
) -> str:
    """Build the status text from sidecar data and optional extra text.

    Precedence for title: original_title > title.
    Precedence for body: original_description > social_caption > enhanced_description.
    """
    parts: List[str] = []

    # Title: prefer original_title, fall back to title
    title = (sidecar.get("original_title") or sidecar.get("title") or "").strip()
    if title:
        parts.append(title)

    # Body: prefer original_description, then social_caption, then enhanced_description
    body_text = (
        sidecar.get("original_description")
        or sidecar.get("social_caption")
        or sidecar.get("enhanced_description")
        or ""
    ).strip()
    if body_text:
        parts.append(body_text)

    # Extra user text
    if extra_text:
        parts.append(extra_text.strip())

    # Hashtags (processed: remove places/genre, prepend photography/photo)
    hashtags = (sidecar.get("hashtags") or "").strip()
    if hashtags:
        parts.append(_process_hashtags(hashtags))

    status_text = "\n\n".join(parts)

    # Mastodon has a 500-char limit
    if len(status_text) > 500:
        sys.stderr.write(
            f"Warning: status text is {len(status_text)} chars (limit: 500). "
            "It will be truncated.\n"
        )
        status_text = status_text[:497] + "..."

    return status_text


def _build_alt_text(sidecar: Dict[str, Any]) -> str:
    """Build alt text from the sidecar. Prefers image_description."""
    desc = (sidecar.get("image_description") or "").strip()
    if not desc:
        # Fallback to title
        desc = (sidecar.get("title") or "").strip()
    # Mastodon alt text limit is 1500 chars
    if len(desc) > 1500:
        desc = desc[:1497] + "..."
    return desc


# ---------------------------------------------------------------------------
#  display helpers for list
# ---------------------------------------------------------------------------

def _extract_media_filename(url: str) -> str:
    """Extract the filename from a media URL."""
    # URL like: https://cdn.mastodon.social/.../original/photo.jpg
    path = urllib.parse.urlparse(url).path
    return os.path.basename(path)


def _plain_text(html: str) -> str:
    """Strip HTML tags, return plain text."""
    return re.sub(r"<[^>]+>", "", html).strip()


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


# ---------------------------------------------------------------------------
#  subcommand implementations
# ---------------------------------------------------------------------------

def cmd_auth_url() -> None:
    """Print the OAuth authorize URL."""
    creds = _load_creds()
    instance = _get_instance(creds)
    client_id = creds.get("client_id", "").strip()

    if not client_id:
        sys.stderr.write("Missing client_id in credential file.\n")
        raise SystemExit(1)

    url = (
        f"https://{instance}/oauth/authorize"
        f"?client_id={client_id}"
        f"&redirect_uri=urn:ietf:wg:oauth:2.0:oob"
        f"&response_type=code"
        f"&scope=read+write"
    )
    print(url)


def cmd_auth_code(code: str) -> None:
    """Exchange an authorization code for an access token."""
    token = _exchange_code_for_token(code)
    print(f"Access token saved. (first 8 chars: {token[:8]}...)")


def cmd_list(
    code: Optional[str] = None,
    limit: int = 20,
) -> None:
    """List recent Mastodon posts with full details."""
    token = _get_token(code=code)
    creds = _load_creds()
    instance = _get_instance(creds)

    # Get account ID
    account = _api_request(instance, token, "GET", "/api/v1/accounts/verify_credentials")
    account_id = account.get("id")
    if not account_id:
        sys.stderr.write("Could not get account ID.\n")
        raise SystemExit(1)

    # Get statuses
    path = f"/api/v1/accounts/{account_id}/statuses?limit={limit}"
    statuses = _api_request(instance, token, "GET", path)

    if not statuses:
        print("No posts found.")
        return

    for i, s in enumerate(statuses):
        sid = s.get("id", "?")
        created = s.get("created_at", "?")
        url = s.get("url", "")
        content_html = s.get("content", "") or ""
        content_plain = _truncate(_plain_text(content_html), 80)

        # Likes
        favourites_count = s.get("favourites_count", 0)
        reblogs_count = s.get("reblogs_count", 0)
        replies_count = s.get("replies_count", 0)

        # Tags
        tags = s.get("tags", [])
        tag_names = [t.get("name", "") for t in tags]

        # Media attachments
        media_attachments = s.get("media_attachments", [])

        # --- Print post block ---
        print(f"ID:        {sid}")
        print(f"Created:   {created}")
        print(f"Likes:     {favourites_count}  |  Boosts: {reblogs_count}  |  Replies: {replies_count}")

        if tag_names:
            print(f"Tags:      {' '.join('#' + t for t in tag_names)}")

        if media_attachments:
            for m in media_attachments:
                media_url = m.get("url", "")
                filename = _extract_media_filename(media_url) if media_url else "?"
                alt = m.get("description", "") or "(no alt text)"
                print(f"Media:     {filename}")
                print(f"Alt text:  {_truncate(alt, 120)}")

        print(f"Content:   {content_plain}")
        if url:
            print(f"URL:       {url}")
        print("-" * 60)
        print()

    print(f"{len(statuses)} post(s) shown.")


def cmd_delete(
    post_id: str,
    code: Optional[str] = None,
) -> None:
    """Delete a Mastodon post by ID."""
    token = _get_token(code=code)
    creds = _load_creds()
    instance = _get_instance(creds)

    _api_request(instance, token, "DELETE", f"/api/v1/statuses/{post_id}")
    print(f"Deleted post {post_id}.")


def cmd_post(
    image_path: str,
    text: Optional[str] = None,
    visibility: str = "public",
    dry_run: bool = False,
    code: Optional[str] = None,
) -> None:
    """Post an image to Mastodon.

    1. Find and load the sidecar JSON for the image.
    2. Build alt text and status text from the sidecar.
    3. Upload the image to Mastodon.
    4. Post a status with the image attached.
    """
    token = _get_token(code=code)
    creds = _load_creds()
    instance = _get_instance(creds)

    if not os.path.exists(image_path):
        sys.stderr.write(f"Image not found: {image_path}\n")
        raise SystemExit(1)

    # Find sidecar
    sidecar_path = _find_sidecar(image_path)
    if not sidecar_path:
        sys.stderr.write(
            f"No sidecar JSON found for {image_path}. "
            "Expected {os.path.splitext(image_path)[0]}.json or .social.json\n"
        )
        raise SystemExit(1)

    sidecar = _load_sidecar(sidecar_path)

    alt_text = _build_alt_text(sidecar)
    status_text = _build_status_text(sidecar, text, visibility)

    if dry_run:
        print("=" * 60)
        print(f"Instance:   {instance}")
        print(f"Visibility: {visibility}")
        print(f"Image:      {image_path}")
        print(f"Sidecar:    {sidecar_path}")
        print(f"Alt text:   {alt_text[:200]}{'...' if len(alt_text) > 200 else ''}")
        print("-" * 60)
        print("Status text:")
        print(status_text)
        print("=" * 60)
        print("DRY RUN — nothing posted.")
        return

    # Upload media
    print(f"Uploading: {os.path.basename(image_path)} ...")
    media_id = _upload_media(instance, token, image_path, alt_text)
    print(f"Media ID: {media_id}")

    # Post status
    result = _post_status(instance, token, status_text, [media_id], visibility)
    post_id = result.get("id", "?")
    post_url = result.get("url", "")
    print(f"Posted! ID={post_id}")
    if post_url:
        print(f"URL: {post_url}")


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Post images to Mastodon via OAuth."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # auth-url
    sub.add_parser("auth-url", help="Print the OAuth authorize URL")

    # auth-code <CODE>
    p_code = sub.add_parser("auth-code", help="Exchange auth code for access token")
    p_code.add_argument("code", help="OAuth authorization code")

    # list [--code <CODE>] [--limit <N>]
    p_list = sub.add_parser("list", help="List recent Mastodon posts")
    p_list.add_argument("--code", default=None, help="OAuth authorization code")
    p_list.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of posts to list (default: 20).",
    )

    # delete <ID> [--code <CODE>]
    p_del = sub.add_parser("delete", help="Delete a Mastodon post by ID")
    p_del.add_argument("post_id", help="ID of the post to delete")
    p_del.add_argument("--code", default=None, help="OAuth authorization code")

    # post <PATH> [--text <TEXT>] [--visibility <VIS>] [--dry-run] [--code <CODE>]
    p_post = sub.add_parser("post", help="Post an image to Mastodon")
    p_post.add_argument("path", help="Path to the image file")
    p_post.add_argument(
        "--text",
        default=None,
        help="Additional text to include in the status.",
    )
    p_post.add_argument(
        "--visibility",
        default="public",
        choices=["public", "unlisted", "private", "direct"],
        help="Post visibility (default: public).",
    )
    p_post.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without posting.",
    )
    p_post.add_argument(
        "--code",
        default=None,
        help="OAuth authorization code (optional).",
    )

    args = parser.parse_args(argv)

    if args.command == "auth-url":
        cmd_auth_url()
    elif args.command == "auth-code":
        cmd_auth_code(args.code)
    elif args.command == "list":
        cmd_list(code=args.code, limit=args.limit)
    elif args.command == "delete":
        cmd_delete(post_id=args.post_id, code=args.code)
    elif args.command == "post":
        cmd_post(
            image_path=args.path,
            text=args.text,
            visibility=args.visibility,
            dry_run=args.dry_run,
            code=args.code,
        )
    else:
        parser.print_help()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
