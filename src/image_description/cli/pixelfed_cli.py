"""
Pixelfed publishing CLI.

Uses Pixelfed's Mastodon-compatible REST API (v1).

Subcommands:
    auth-url          Print OAuth authorize URL
    auth-code <CODE>  Exchange authorization code for access token
    post <IMAGE>      Upload image + post status using sidecar JSON
    list              List your recent posts
    delete <ID>       Delete a post by ID

Credential file: /home/junwin/credential/pixelfed.json
Hashtag rules: remove #places, #genre; prepend #photography, #photo
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import requests

from ..sidecar import Sidecar

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


# --- Hashtag processing ---

REMOVE_TAGS = {"#places", "#genre"}
PREPEND_TAGS = ["#photography", "#photo"]


def _process_hashtags(raw: str) -> str:
    """Apply hashtag rules: remove unwanted, deduplicate, prepend canonical tags."""
    tags = [t.strip() for t in raw.split() if t.strip()]
    # Remove unwanted
    tags = [t for t in tags if t.lower() not in {r.lower() for r in REMOVE_TAGS}]
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for t in tags:
        lower = t.lower()
        if lower not in seen:
            seen.add(lower)
            deduped.append(t)
    # Prepend canonical tags (skip if already present)
    result = []
    for pt in PREPEND_TAGS:
        if pt.lower() not in seen:
            result.append(pt)
            seen.add(pt.lower())
    result.extend(deduped)
    return " ".join(result)


# --- Sidecar loading ---

def _find_sidecar(image_path: str) -> Optional[Sidecar]:
    """Find sidecar JSON for an image: .json or .social.json sibling."""
    base = Path(image_path)
    # Try .json first
    json_path = base.with_suffix(".json")
    if json_path.exists():
        return Sidecar.load(str(json_path))
    # Try .social.json
    social_path = base.with_suffix(".social.json")
    if social_path.exists():
        return Sidecar.load(str(social_path))
    # Try stem-based (for files like image.jpg → image.json)
    stem_json = base.parent / f"{base.stem}.json"
    if stem_json.exists():
        return Sidecar.load(str(stem_json))
    return None


def _build_status(sidecar: Sidecar) -> str:
    """Build the status text from sidecar fields."""
    # Title line
    title = sidecar.title or sidecar.original_title or ""
    # Caption: prefer social_caption, then enhanced_description, then original_description
    social_caption = sidecar.extra.get("social_caption", "")
    description = social_caption or sidecar.enhanced_description or sidecar.original_description or ""

    # Process hashtags
    hashtags = _process_hashtags(sidecar.hashtags)

    lines = []
    if title:
        lines.append(title)
    if description and description != title:
        lines.append("")
        lines.append(description)
    if hashtags:
        lines.append("")
        lines.append(hashtags)

    return "\n".join(lines).strip()


def _get_alt_text(sidecar: Sidecar) -> str:
    """Get alt text from sidecar (image_description or visually_challenged_description)."""
    image_desc = sidecar.extra.get("image_description", "")
    if image_desc:
        return image_desc
    return sidecar.visually_challenged_description or ""


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


def cmd_post(image_path: str, dry_run: bool = False) -> None:
    """Upload an image and post to Pixelfed."""
    token = _get_token()

    # Resolve image
    img = Path(image_path).resolve()
    if not img.exists():
        print(f"Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    # Load sidecar
    sidecar = _find_sidecar(str(img))
    if sidecar is None:
        print(f"No sidecar JSON found for: {image_path}", file=sys.stderr)
        sys.exit(1)

    status = _build_status(sidecar)
    alt_text = _get_alt_text(sidecar)

    if dry_run:
        print("=== DRY RUN ===")
        print(f"Image: {img}")
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
        print(f"Posted! ID: {result['id']}, URL: {result.get('url', 'N/A')}")
    else:
        print(f"Post failed: {status_resp.status_code} {status_resp.text}", file=sys.stderr)
        sys.exit(1)


def cmd_list() -> None:
    """List recent posts."""
    token = _get_token()
    resp = requests.get(
        f"{API_BASE}/api/v1/accounts/verify_credentials",
        headers={"Authorization": f"Bearer {token}"},
    )
    if resp.status_code != 200:
        print(f"Failed to get account: {resp.status_code} {resp.text}", file=sys.stderr)
        sys.exit(1)
    account_id = resp.json()["id"]

    resp = requests.get(
        f"{API_BASE}/api/v1/accounts/{account_id}/statuses",
        headers={"Authorization": f"Bearer {token}"},
        params={"limit": 20},
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
        content = p.get("content", "")[:80].replace("\n", " ")
        url = p.get("url", "")
        print(f"ID: {pid}  Date: {created}  URL: {url}")
        print(f"  {content}...")
        print()


def cmd_delete(post_id: str) -> None:
    """Delete a post by ID."""
    token = _get_token()
    resp = requests.delete(
        f"{API_BASE}/api/v1/statuses/{post_id}",
        headers={"Authorization": f"Bearer {token}"},
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

    # list
    sub.add_parser("list", help="List recent posts")

    # delete
    p_delete = sub.add_parser("delete", help="Delete a post by ID")
    p_delete.add_argument("post_id", help="Post ID to delete")

    args = parser.parse_args(argv)

    if args.command == "auth-url":
        cmd_auth_url()
    elif args.command == "auth-code":
        cmd_auth_code(args.code)
    elif args.command == "post":
        cmd_post(args.image, dry_run=args.dry_run)
    elif args.command == "list":
        cmd_list()
    elif args.command == "delete":
        cmd_delete(args.post_id)


if __name__ == "__main__":
    main()
