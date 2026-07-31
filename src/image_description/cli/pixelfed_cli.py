"""
Pixelfed publishing CLI.

Uses Pixelfed's Mastodon-compatible REST API (v1).

Subcommands:
    auth-url          Print OAuth authorize URL
    auth-code <CODE>  Exchange authorization code for access token
    post <IMAGE>      Upload image + post status using sidecar JSON
    list              List your recent posts
    get <ID>          Get full detail for a single post
    delete <ID>       Delete a post by ID

Credential file: /home/junwin/credential/pixelfed.json
Hashtag rules: remove #places, #genre; prepend #photography, #photo
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import requests

from ..sidecar import Sidecar
from ..social_utils import build_social_text, build_alt_text

CRED_PATH = "/home/junwin/credential/pixelfed.json"
INSTANCE = "pixelfed.social"
API_BASE = f"https://{INSTANCE}"
SCOPES = "read write"

# Redirect URI for OAuth (desktop app / loopback)
REDIRECT_URI = "urn:ietf:wg:oauth:2.0:oob"

# --- Credential helpers ---

def _load_creds() -> dict:
    if not os.path.exists(CRED_PATH):
        return {}
    with open(CRED_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_creds(creds: dict) -> None:
    os.makedirs(os.path.dirname(CRED_PATH), exist_ok=True)
    with open(CRED_PATH, "w", encoding="utf-8") as f:
        json.dump(creds, f, indent=4)


def _ensure_app() -> dict:
    """Ensure an OAuth app is registered; returns creds dict with client_id/client_secret."""
    creds = _load_creds()
    if creds.get("client_id") and creds.get("client_secret"):
        return creds

    # Register a new app
    resp = requests.post(f"{API_BASE}/api/v1/apps", data={
        "client_name": "Image Description CLI",
        "redirect_uris": REDIRECT_URI,
        "scopes": SCOPES,
        "website": "https://junwin.github.io",
    })
    resp.raise_for_status()
    data = resp.json()
    creds["instance"] = INSTANCE
    creds["client_id"] = data["client_id"]
    creds["client_secret"] = data["client_secret"]
    _save_creds(creds)
    return creds


def _get_token() -> str:
    creds = _load_creds()
    token = creds.get("access_token")
    if not token:
        print("No access token found. Run 'auth-url' then 'auth-code <CODE>' first.", file=sys.stderr)
        sys.exit(1)
    return token


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# --- Sidecar loading ---

def _find_sidecar(image_path: str) -> Tuple[Optional[Sidecar], Optional[str]]:
    """Find sidecar JSON for an image: .json or .social.json sibling.

    Returns (Sidecar, path) tuple, or (None, None) if not found.
    """
    base = Path(image_path)
    # Try .json first
    json_path = base.with_suffix(".json")
    if json_path.exists():
        return Sidecar.load(str(json_path)), str(json_path)
    # Try .social.json
    social_path = base.with_suffix(".social.json")
    if social_path.exists():
        return Sidecar.load(str(social_path)), str(social_path)
    # Try stem-based (for files like image.jpg → image.json)
    stem_json = base.parent / f"{base.stem}.json"
    if stem_json.exists():
        return Sidecar.load(str(stem_json)), str(stem_json)
    return None, None


# --- Helpers for display ---

def _strip_html(text: str) -> str:
    """Strip basic HTML tags from status content."""
    import re
    return re.sub(r"<[^>]+>", "", text)


def _counts_str(post: dict) -> str:
    """Build a compact counts string: ❤N ↩N 💬N"""
    fav = post.get("favourites_count", 0)
    reblog = post.get("reblogs_count", 0)
    replies = post.get("replies_count", 0)
    return f"❤{fav} ↩{reblog} 💬{replies}"


def _print_post_detail(post: dict) -> None:
    """Print full detail for a single post."""
    pid = post.get("id", "")
    created = post.get("created_at", "")[:19]
    url = post.get("url", "")
    visibility = post.get("visibility", "")
    sensitive = post.get("sensitive", False)
    spoiler = post.get("spoiler_text", "")
    content = _strip_html(post.get("content", ""))

    print(f"ID:         {pid}")
    print(f"Date:       {created}")
    print(f"URL:        {url}")
    print(f"Visibility: {visibility}")
    print(f"Counts:     {_counts_str(post)}")
    if sensitive:
        print(f"Sensitive:  True")
    if spoiler:
        print(f"CW:         {spoiler}")
    print()

    # Content
    print("Content:")
    print(content or "(empty)")
    print()

    # Media attachments
    media = post.get("media_attachments", [])
    if media:
        print(f"Media ({len(media)}):")
        for i, m in enumerate(media):
            print(f"  [{i+1}] ID: {m.get('id')}  Type: {m.get('type', 'unknown')}")
            print(f"      URL: {m.get('url', '')}")
            desc = m.get("description")
            if desc:
                print(f"      Alt: {desc}")
        print()

    # Tags
    tags = post.get("tags", [])
    if tags:
        print("Tags:")
        for t in tags:
            print(f"  {t.get('name', '')}")
        print()

    # Application
    app = post.get("application", {})
    if app:
        print(f"App:   {app.get('name', 'N/A')}")
    print()


# --- Subcommands ---

def cmd_auth_url() -> None:
    """Print the OAuth authorize URL."""
    creds = _ensure_app()
    url = (
        f"{API_BASE}/oauth/authorize"
        f"?client_id={creds['client_id']}"
        f"&redirect_uri={REDIRECT_URI}"
        f"&response_type=code"
        f"&scope={SCOPES.replace(' ', '+')}"
    )
    print("Open this URL in your browser and authorize:")
    print(url)


def cmd_auth_code(code: str) -> None:
    """Exchange authorization code for access token."""
    creds = _ensure_app()
    resp = requests.post(f"{API_BASE}/oauth/token", data={
        "client_id": creds["client_id"],
        "client_secret": creds["client_secret"],
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code",
        "code": code,
        "scope": SCOPES,
    })
    if resp.status_code != 200:
        print(f"Error exchanging code: {resp.status_code} {resp.text}", file=sys.stderr)
        sys.exit(1)
    data = resp.json()
    creds["access_token"] = data["access_token"]
    _save_creds(creds)
    print("Access token saved.")


def cmd_post(image_path: str, dry_run: bool = False, caption_field: str = "social_caption") -> None:
    """Upload an image and post to Pixelfed.

    After a successful post, writes publish info (platform, post_id, url, date)
    back to the sidecar JSON.
    """
    token = _get_token()

    # Resolve image
    img = Path(image_path).resolve()
    if not img.exists():
        print(f"Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    # Load sidecar
    sidecar, sidecar_path = _find_sidecar(str(img))
    if sidecar is None:
        print(f"No sidecar JSON found for: {image_path}", file=sys.stderr)
        sys.exit(1)

    # Convert Sidecar to dict for shared utils
    sidecar_data = sidecar.to_dict()

    status = build_social_text(sidecar_data, caption_field=caption_field)
    alt_text = build_alt_text(sidecar_data)

    if dry_run:
        print("=== DRY RUN ===")
        print(f"Image: {img}")
        print(f"Caption field: {caption_field}")
        print(f"Alt text: {alt_text}")
        print(f"Status:\n{status}")
        print("=== End dry run ===")
        return

    headers = {"Authorization": f"Bearer {token}"}

    # Step 1: Upload media
    with open(img, "rb") as f:
        files = {"file": (img.name, f, "image/jpeg")}
        data = {"description": alt_text}
        media_resp = requests.post(
            f"{API_BASE}/api/v1/media",
            headers={"Authorization": f"Bearer {token}"},
            files=files,
            data=data,
        )
    if media_resp.status_code != 200:
        print(f"Media upload failed: {media_resp.status_code} {media_resp.text}", file=sys.stderr)
        sys.exit(1)
    media_id = media_resp.json()["id"]
    print(f"Media uploaded, ID: {media_id}")

    # Step 2: Publish status
    status_resp = requests.post(
        f"{API_BASE}/api/v1/statuses",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        data={"status": status, "media_ids[]": [media_id]},
    )
    if status_resp.status_code == 200:
        result = status_resp.json()
        post_id = str(result["id"])
        post_url = result.get("url", "")
        print(f"Posted! ID: {post_id}, URL: {post_url}")

        # Write publish info back to sidecar
        if sidecar_path:
            try:
                sidecar.add_publish_event("pixelfed", post_id, post_url)
                sidecar.save(sidecar_path)
                print(f"Sidecar updated: {sidecar_path}")
            except Exception as e:
                print(f"Warning: could not update sidecar: {e}", file=sys.stderr)
    else:
        print(f"Post failed: {status_resp.status_code} {status_resp.text}", file=sys.stderr)
        sys.exit(1)


def cmd_list(limit: int = 20) -> None:
    """List recent posts."""
    token = _get_token()
    resp = requests.get(
        f"{API_BASE}/api/v1/accounts/verify_credentials",
        headers=_headers(token),
    )
    if resp.status_code != 200:
        print(f"Failed to get account: {resp.status_code} {resp.text}", file=sys.stderr)
        sys.exit(1)
    account_id = resp.json()["id"]

    resp = requests.get(
        f"{API_BASE}/api/v1/accounts/{account_id}/statuses",
        headers=_headers(token),
        params={"limit": limit},
    )
    if resp.status_code != 200:
        print(f"Failed to list posts: {resp.status_code} {resp.text}", file=sys.stderr)
        sys.exit(1)

    posts = resp.json()
    if not posts:
        print("No posts found.")
        return

    for p in posts:
        created = p.get("created_at", "")[:19]
        pid = p["id"]
        content = _strip_html(p.get("content", ""))[:80].replace("\n", " ")
        url = p.get("url", "")
        counts = _counts_str(p)
        print(f"ID: {pid}  {counts}  Date: {created}")
        print(f"  {content}...")
        print(f"  {url}")
        print()

    print(f"{len(posts)} post(s) shown.")


def cmd_get(post_id: str) -> None:
    """Fetch full detail for a single post."""
    token = _get_token()
    resp = requests.get(
        f"{API_BASE}/api/v1/statuses/{post_id}",
        headers=_headers(token),
    )
    if resp.status_code != 200:
        print(f"Failed to get post {post_id}: {resp.status_code} {resp.text}", file=sys.stderr)
        sys.exit(1)

    post = resp.json()
    _print_post_detail(post)


def cmd_delete(post_id: str) -> None:
    """Delete a post by ID."""
    token = _get_token()
    resp = requests.delete(
        f"{API_BASE}/api/v1/statuses/{post_id}",
        headers=_headers(token),
    )
    if resp.status_code == 200:
        print(f"Deleted post {post_id}.")
    else:
        print(f"Delete failed: {resp.status_code} {resp.text}", file=sys.stderr)
        sys.exit(1)


# --- Main ---

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Publish to Pixelfed using sidecar JSON metadata.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # auth-url
    sub.add_parser("auth-url", help="Print OAuth authorize URL")

    # auth-code
    p_auth_code = sub.add_parser("auth-code", help="Exchange code for access token")
    p_auth_code.add_argument("code", help="Authorization code from browser redirect")

    # post
    p_post = sub.add_parser("post", help="Upload image and post status")
    p_post.add_argument("image", help="Path to image file")
    p_post.add_argument("--dry-run", action="store_true", help="Preview without posting")
    p_post.add_argument(
        "--caption-field",
        default="social_caption",
        help="Sidecar field to use for the post caption (default: social_caption).",
    )

    # list
    p_list = sub.add_parser("list", help="List recent posts")
    p_list.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of posts to list (default: 20).",
    )

    # get
    p_get = sub.add_parser("get", help="Get full detail for a single post")
    p_get.add_argument("post_id", help="Post ID to fetch")

    # delete
    p_delete = sub.add_parser("delete", help="Delete a post by ID")
    p_delete.add_argument("post_id", help="Post ID to delete")

    args = parser.parse_args(argv)

    if args.command == "auth-url":
        cmd_auth_url()
    elif args.command == "auth-code":
        cmd_auth_code(args.code)
    elif args.command == "post":
        cmd_post(args.image, dry_run=args.dry_run, caption_field=args.caption_field)
    elif args.command == "list":
        cmd_list(limit=args.limit)
    elif args.command == "get":
        cmd_get(args.post_id)
    elif args.command == "delete":
        cmd_delete(args.post_id)


if __name__ == "__main__":
    main()
