#!/usr/bin/env python3
"""
CLI for posting images to Bluesky via AT Protocol (app password auth).

When called from an agent, the command will be in the form:
    bash -lc "source .venv/bin/activate && python -m src.image_description.cli.bluesky_cli <subcommand> <args>"

Subcommands:
    auth                  Save Bluesky handle + app password to credential file
    post <PATH>           Post an image (with sidecar JSON) to Bluesky
    list                  List recent posts
    delete <URI>          Delete a post by AT URI

Auth: Bluesky uses app passwords, not OAuth. Generate one at:
    Settings → Privacy & Security → App Passwords
"""

import argparse
import json
import mimetypes
import os
import sys
from typing import Any, Dict, List, Optional
from pathlib import Path

from atproto import Client

from ..sidecar import Sidecar
from ..social_utils import build_social_text, build_alt_text

CRED_PATH = "/home/junwin/credential/bluesky.json"

# Bluesky post text limit in graphemes
BSKY_TEXT_LIMIT = 300


# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------

def _load_creds() -> Dict[str, Any]:
    """Load the Bluesky credential file."""
    if not os.path.exists(CRED_PATH):
        sys.stderr.write(f"Credential file not found: {CRED_PATH}\n")
        sys.stderr.write(
            "Create it with: bluesky_cli auth --handle your.handle.bsky.social "
            "--app-password xxxx-xxxx-xxxx-xxxx\n"
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


def _get_client() -> Client:
    """Create and log in a Bluesky client from stored credentials."""
    creds = _load_creds()
    handle = creds.get("handle", "").strip()
    app_password = creds.get("app_password", "").strip()

    if not handle or not app_password:
        sys.stderr.write(
            "Missing 'handle' or 'app_password' in credential file.\n"
        )
        raise SystemExit(1)

    client = Client()
    try:
        client.login(handle, app_password)
    except Exception as e:
        sys.stderr.write(f"Bluesky login failed: {e}\n")
        raise SystemExit(1)
    return client


# ---------------------------------------------------------------------------
#  sidecar helper
# ---------------------------------------------------------------------------

def _find_sidecar(image_path: str) -> Optional[str]:
    """Find the sidecar JSON for an image. Checks .json and .social.json variants."""
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
#  display helpers for list
# ---------------------------------------------------------------------------

def _truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def _extract_rkey(uri: str) -> str:
    """Extract the record key (rkey) from an at:// URI."""
    # at://did:plc:xxx/app.bsky.feed.post/rkey
    parts = uri.split("/")
    return parts[-1] if parts else uri


# ---------------------------------------------------------------------------
#  subcommand implementations
# ---------------------------------------------------------------------------

def cmd_auth(handle: str, app_password: str) -> None:
    """Save Bluesky credentials. Verifies with a test login first."""
    if not handle or not app_password:
        sys.stderr.write("Both --handle and --app-password are required.\n")
        raise SystemExit(1)

    # Verify credentials by attempting a login
    client = Client()
    try:
        client.login(handle, app_password)
    except Exception as e:
        sys.stderr.write(f"Login verification failed: {e}\n")
        raise SystemExit(1)

    creds = {
        "handle": handle,
        "app_password": app_password,
    }
    _save_creds(creds)
    print(f"Credentials saved. Logged in as @{handle}")


def cmd_list(limit: int = 20) -> None:
    """List recent Bluesky posts."""
    client = _get_client()
    creds = _load_creds()
    handle = creds.get("handle", "").strip()

    try:
        response = client.get_author_feed(actor=handle, limit=limit)
    except Exception as e:
        sys.stderr.write(f"Failed to get author feed: {e}\n")
        raise SystemExit(1)

    feed = response.feed
    if not feed:
        print("No posts found.")
        return

    for i, item in enumerate(feed):
        post = item.post
        uri = post.uri
        record = post.record

        created = record.created_at or "?"
        text = record.text or ""

        # Stats
        like_count = post.like_count or 0
        repost_count = post.repost_count or 0
        reply_count = post.reply_count or 0

        # Images
        embed = record.embed
        has_image = False
        if embed is not None:
            embed_images = getattr(embed, "images", None)
            if embed_images:
                has_image = True

        # Build post URL
        rkey = _extract_rkey(uri)
        post_url = f"https://bsky.app/profile/{handle}/post/{rkey}"

        # --- Print post block ---
        print(f"URI:       {uri}")
        print(f"RKEY:      {rkey}")
        print(f"Created:   {created}")
        print(f"Likes:     {like_count}  |  Reposts: {repost_count}  |  Replies: {reply_count}")
        print(f"Has image: {'yes' if has_image else 'no'}")
        print(f"Text:      {_truncate(text, 120)}")
        print(f"URL:       {post_url}")
        print("-" * 60)
        print()

    print(f"{len(feed)} post(s) shown.")


def cmd_delete(uri: str) -> None:
    """Delete a Bluesky post by its AT URI."""
    client = _get_client()

    try:
        client.delete_post(uri)
    except Exception as e:
        sys.stderr.write(f"Failed to delete post: {e}\n")
        raise SystemExit(1)

    print(f"Deleted post: {uri}")


def cmd_post(
    image_path: str,
    text: Optional[str] = None,
    dry_run: bool = False,
    caption_field: str = "social_caption",
) -> None:
    """Post an image to Bluesky.

    1. Find and load the sidecar JSON for the image.
    2. Build alt text and status text from the sidecar.
    3. Upload the image as a blob.
    4. Create a post with the image embedded.
    5. Write publish info back to the sidecar.
    """
    if not os.path.exists(image_path):
        sys.stderr.write(f"Image not found: {image_path}\n")
        raise SystemExit(1)

    # Find sidecar
    sidecar_path = _find_sidecar(image_path)
    if not sidecar_path:
        sys.stderr.write(
            f"No sidecar JSON found for {image_path}. "
            f"Expected {os.path.splitext(image_path)[0]}.json or .social.json\n"
        )
        raise SystemExit(1)

    sidecar = _load_sidecar(sidecar_path)

    alt_text = build_alt_text(sidecar)
    status_text = build_social_text(
        sidecar,
        caption_field=caption_field,
        extra_text=text,
        char_limit=BSKY_TEXT_LIMIT,
    )

    if dry_run:
        creds = _load_creds_optional()
        handle = creds.get("handle", "").strip() if creds else "(not configured)"
        print("=" * 60)
        print(f"Handle:        @{handle}")
        print(f"Caption field: {caption_field}")
        print(f"Image:         {image_path}")
        print(f"Sidecar:       {sidecar_path}")
        print(f"Alt text:      {alt_text[:200]}{'...' if len(alt_text) > 200 else ''}")
        print("-" * 60)
        print("Status text:")
        print(status_text)
        print("=" * 60)
        print("DRY RUN — nothing posted.")
        return

    client = _get_client()
    creds = _load_creds()
    handle = creds.get("handle", "").strip()

    # Read image bytes
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Upload and post
    print(f"Uploading: {os.path.basename(image_path)} ...")
    try:
        result = client.send_image(
            text=status_text,
            image=image_bytes,
            image_alt=alt_text,
        )
    except Exception as e:
        sys.stderr.write(f"Failed to post to Bluesky: {e}\n")
        raise SystemExit(1)

    post_uri = result.uri
    rkey = _extract_rkey(post_uri)
    post_url = f"https://bsky.app/profile/{handle}/post/{rkey}"
    print(f"Posted! URI={post_uri}")
    print(f"URL: {post_url}")

    # Write publish info back to sidecar
    try:
        sidecar_obj = Sidecar.load(sidecar_path)
        sidecar_obj.add_publish_event("bluesky", post_uri, post_url)
        sidecar_obj.save(sidecar_path)
        print(f"Sidecar updated: {sidecar_path}")
    except Exception as e:
        sys.stderr.write(f"Warning: could not update sidecar: {e}\n")


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Post images to Bluesky via AT Protocol."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # auth --handle <HANDLE> --app-password <PASSWORD>
    p_auth = sub.add_parser("auth", help="Save Bluesky credentials")
    p_auth.add_argument("--handle", required=True, help="Bluesky handle (e.g. user.bsky.social)")
    p_auth.add_argument("--app-password", required=True, help="Bluesky app password")

    # list [--limit <N>]
    p_list = sub.add_parser("list", help="List recent Bluesky posts")
    p_list.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of posts to list (default: 20).",
    )

    # delete <URI>
    p_del = sub.add_parser("delete", help="Delete a Bluesky post by AT URI")
    p_del.add_argument("uri", help="AT URI of the post to delete (e.g. at://did:plc:xxx/app.bsky.feed.post/rkey)")

    # post <PATH> [--text <TEXT>] [--dry-run] [--caption-field <FIELD>]
    p_post = sub.add_parser("post", help="Post an image to Bluesky")
    p_post.add_argument("path", help="Path to the image file")
    p_post.add_argument(
        "--text",
        default=None,
        help="Additional text to include in the post.",
    )
    p_post.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without posting.",
    )
    p_post.add_argument(
        "--caption-field",
        default="social_caption",
        help="Sidecar field to use for the post caption (default: social_caption).",
    )

    args = parser.parse_args(argv)

    if args.command == "auth":
        cmd_auth(args.handle, args.app_password)
    elif args.command == "list":
        cmd_list(limit=args.limit)
    elif args.command == "delete":
        cmd_delete(uri=args.uri)
    elif args.command == "post":
        cmd_post(
            image_path=args.path,
            text=args.text,
            dry_run=args.dry_run,
            caption_field=args.caption_field,
        )
    else:
        parser.print_help()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
