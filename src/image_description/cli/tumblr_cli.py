#!/usr/bin/env python3
"""
CLI for posting images to Tumblr via OAuth 1.0a.

When called from an agent, the command will be in the form:
    bash -lc "source .venv/bin/activate && python -m src.image_description.cli.tumblr_cli <subcommand> <args>"

Subcommands:
    auth-url              Print the Tumblr OAuth authorize URL
    auth-code <VERIFIER>  Exchange OAuth verifier for access token
    post <PATH>           Post an image (with sidecar JSON) to Tumblr
    list                  List recent posts
    delete <ID>           Delete a post by ID

Auth: Tumblr uses OAuth 1.0a. You need:
    - consumer_key / consumer_secret (register an app at https://www.tumblr.com/oauth/apps)
      Use http://localhost:8080 as the Default callback URL.
    - Then run auth-url, authorize in browser, run auth-code with the verifier.
"""

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
from typing import Any, Dict, List, Optional
from pathlib import Path

import requests
from requests_oauthlib import OAuth1Session

from ..sidecar import Sidecar
from ..social_utils import build_social_text, build_alt_text, process_hashtags

CRED_PATH = "/home/junwin/credential/tumblr.json"
TUMBLR_API_BASE = "https://api.tumblr.com/v2"
TUMBLR_REQUEST_TOKEN_URL = "https://www.tumblr.com/oauth/request_token"
TUMBLR_AUTHORIZE_URL = "https://www.tumblr.com/oauth/authorize"
TUMBLR_ACCESS_TOKEN_URL = "https://www.tumblr.com/oauth/access_token"

# Default OAuth callback URL — must match what you registered at https://www.tumblr.com/oauth/apps
DEFAULT_CALLBACK_URL = "http://localhost:8080"

# ---------------------------------------------------------------------------
#  credential helpers
# ---------------------------------------------------------------------------

def _load_creds() -> Dict[str, Any]:
    """Load the Tumblr credential file."""
    if not os.path.exists(CRED_PATH):
        sys.stderr.write(f"Credential file not found: {CRED_PATH}\n")
        sys.stderr.write(
            "Create it as JSON with keys: consumer_key, consumer_secret, blog_identifier.\n"
            "Register an app at https://www.tumblr.com/oauth/apps to get consumer keys.\n"
            "Use http://localhost:8080 as the Default callback URL.\n"
            "Then run: tumblr_cli auth-url\n"
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


def _get_callback_url(creds: Dict[str, Any]) -> str:
    """Get the OAuth callback URL from creds or use the default."""
    return creds.get("callback_url", DEFAULT_CALLBACK_URL).strip()


def _get_oauth_session(token: Optional[str] = None, token_secret: Optional[str] = None) -> OAuth1Session:
    """Create an OAuth1Session from stored consumer credentials.

    If token/token_secret are given, use them as resource owner key/secret.
    Otherwise, leave them blank (for request token fetch).
    """
    creds = _load_creds()
    consumer_key = creds.get("consumer_key", "").strip()
    consumer_secret = creds.get("consumer_secret", "").strip()

    if not consumer_key or not consumer_secret:
        sys.stderr.write("Missing consumer_key or consumer_secret in credential file.\n")
        raise SystemExit(1)

    return OAuth1Session(
        consumer_key,
        client_secret=consumer_secret,
        resource_owner_key=token or None,
        resource_owner_secret=token_secret or None,
        callback_uri=_get_callback_url(creds),
    )


def _get_blog_identifier() -> str:
    """Get blog identifier from creds."""
    creds = _load_creds()
    blog = creds.get("blog_identifier", "").strip()
    if not blog:
        sys.stderr.write("Missing blog_identifier in credential file.\n")
        raise SystemExit(1)
    return blog


# ---------------------------------------------------------------------------
#  sidecar helpers
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
#  hashtag processing for Tumblr (bare tags, no # prefix)
# ---------------------------------------------------------------------------

def _tags_for_tumblr(hashtags_string: str) -> str:
    """Convert space-separated hashtags to Tumblr comma-separated bare tags.

    process_hashtags() returns space-separated #tags.
    Tumblr expects comma-separated tags without # prefix.
    """
    if not hashtags_string:
        return ""
    tags = hashtags_string.split()
    # Strip leading '#' from each tag
    bare = [t.lstrip("#") for t in tags if t.strip()]
    return ",".join(bare)


# ---------------------------------------------------------------------------
#  display helpers for list
# ---------------------------------------------------------------------------

def _strip_html(text: str) -> str:
    """Strip HTML tags, return plain text."""
    return re.sub(r"<[^>]+>", "", text).strip()


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


# ---------------------------------------------------------------------------
#  subcommand: auth-url
# ---------------------------------------------------------------------------

def cmd_auth_url() -> None:
    """Fetch a request token and print the Tumblr OAuth authorize URL.

    Stores the request token + secret in the cred file so auth-code can use them.
    """
    # Create session with just consumer keys (no access token yet)
    oauth = _get_oauth_session()

    try:
        fetch_response = oauth.fetch_request_token(TUMBLR_REQUEST_TOKEN_URL)
    except Exception as e:
        sys.stderr.write(f"Failed to fetch request token: {e}\n")
        raise SystemExit(1)

    request_token = fetch_response.get("oauth_token", "")
    request_token_secret = fetch_response.get("oauth_token_secret", "")

    if not request_token:
        sys.stderr.write("No oauth_token in request token response.\n")
        raise SystemExit(1)

    # Store request token in creds temporarily
    creds = _load_creds()
    creds["request_token"] = request_token
    creds["request_token_secret"] = request_token_secret
    _save_creds(creds)

    auth_url = f"{TUMBLR_AUTHORIZE_URL}?oauth_token={request_token}"
    print("Open this URL in your browser and authorize:")
    print(auth_url)
    print()
    print("After authorizing, you will be redirected to a page that won't load.")
    print("Copy the 'oauth_verifier' value from the browser address bar, then run:")
    print("  tumblr_cli auth-code <OAUTH_VERIFIER>")


# ---------------------------------------------------------------------------
#  subcommand: auth-code
# ---------------------------------------------------------------------------

def cmd_auth_code(verifier: str) -> None:
    """Exchange the OAuth verifier for an access token. Saves to cred file."""
    creds = _load_creds()
    request_token = creds.pop("request_token", "")
    request_token_secret = creds.pop("request_token_secret", "")

    if not request_token:
        sys.stderr.write("No stored request token. Run 'auth-url' first.\n")
        raise SystemExit(1)

    oauth = _get_oauth_session(token=request_token, token_secret=request_token_secret)

    try:
        tokens = oauth.fetch_access_token(
            TUMBLR_ACCESS_TOKEN_URL,
            verifier=verifier,
        )
    except Exception as e:
        # Restore request token in creds on failure
        creds["request_token"] = request_token
        creds["request_token_secret"] = request_token_secret
        _save_creds(creds)
        sys.stderr.write(f"Failed to exchange verifier for access token: {e}\n")
        raise SystemExit(1)

    access_token = tokens.get("oauth_token", "")
    access_token_secret = tokens.get("oauth_token_secret", "")

    if not access_token:
        sys.stderr.write("No oauth_token in access token response.\n")
        raise SystemExit(1)

    creds["oauth_token"] = access_token
    creds["oauth_token_secret"] = access_token_secret
    _save_creds(creds)
    print(f"Access token saved. (first 8 chars: {access_token[:8]}...)")


# ---------------------------------------------------------------------------
#  subcommand: post
# ---------------------------------------------------------------------------

def cmd_post(
    image_path: str,
    text: Optional[str] = None,
    dry_run: bool = False,
    caption_field: str = "social_caption",
) -> None:
    """Post an image to Tumblr using the legacy photo API (type=photo + data64).

    1. Find and load the sidecar JSON for the image.
    2. Build alt text and caption text from the sidecar.
    3. POST to Tumblr API as a photo post.
    4. Write publish info back to the sidecar.
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
    caption_text = build_social_text(
        sidecar,
        caption_field=caption_field,
        extra_text=text,
        char_limit=None,  # Tumblr has no practical character limit
    )

    # Process hashtags for Tumblr (bare tags, comma-separated)
    raw_hashtags = (sidecar.get("hashtags") or "").strip()
    tumblr_tags = ""
    if raw_hashtags:
        processed = process_hashtags(raw_hashtags)
        tumblr_tags = _tags_for_tumblr(processed)

    # Determine MIME type
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"

    # Dry run: preview without credentials
    if dry_run:
        creds = _load_creds_optional()
        blog = creds.get("blog_identifier", "").strip() if creds else "(not configured)"
        print("=" * 60)
        print(f"Blog:          {blog}")
        print(f"Caption field: {caption_field}")
        print(f"Image:         {image_path}")
        print(f"Sidecar:       {sidecar_path}")
        print(f"MIME type:     {mime_type}")
        print(f"Alt text:      {alt_text[:200]}{'...' if len(alt_text) > 200 else ''}")
        print(f"Tags:          {tumblr_tags}")
        print("-" * 60)
        print("Caption text:")
        print(caption_text)
        print("=" * 60)
        print("DRY RUN — nothing posted.")
        return

    # Real post: credentials required
    creds = _load_creds()
    access_token = creds.get("oauth_token", "").strip()
    access_token_secret = creds.get("oauth_token_secret", "").strip()

    if not access_token:
        sys.stderr.write(
            "No stored access token. Run 'auth-url' then 'auth-code <VERIFIER>' first.\n"
        )
        raise SystemExit(1)

    blog = _get_blog_identifier()

    # Read and base64-encode the image
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # Create OAuth session for the API call
    oauth = _get_oauth_session(token=access_token, token_secret=access_token_secret)

    # Use legacy photo API (singular /post, type=photo + data64)
    # NPF endpoint (/posts) returns 400 for this app; old API works.
    post_url = f"{TUMBLR_API_BASE}/blog/{blog}.tumblr.com/post"
    payload: Dict[str, Any] = {
        "type": "photo",
        "data64": image_b64,
        "caption": caption_text,
        "state": "published",
    }
    if tumblr_tags:
        payload["tags"] = tumblr_tags

    print(f"Uploading: {os.path.basename(image_path)} ...")
    try:
        resp = oauth.post(post_url, data=payload)
    except Exception as e:
        sys.stderr.write(f"Tumblr API request failed: {e}\n")
        raise SystemExit(1)

    if resp.status_code not in (200, 201):
        sys.stderr.write(
            f"Tumblr API error: HTTP {resp.status_code}\n{resp.text}\n"
        )
        raise SystemExit(1)

    data = resp.json()
    response_data = data.get("response", {})
    post_id = response_data.get("id")
    if not post_id:
        sys.stderr.write(
            "Post succeeded but no id in response.\n"
            f"{json.dumps(data, indent=2)}\n"
        )
        raise SystemExit(1)

    # Build post URL
    post_url_str = f"https://{blog}.tumblr.com/post/{post_id}"
    print(f"Posted! ID={post_id}")
    print(f"URL: {post_url_str}")

    # Write publish info back to sidecar
    try:
        sidecar_obj = Sidecar.load(sidecar_path)
        sidecar_obj.add_publish_event("tumblr", str(post_id), post_url_str)
        sidecar_obj.save(sidecar_path)
        print(f"Sidecar updated: {sidecar_path}")
    except Exception as e:
        sys.stderr.write(f"Warning: could not update sidecar: {e}\n")


# ---------------------------------------------------------------------------
#  subcommand: list
# ---------------------------------------------------------------------------

def cmd_list(limit: int = 20) -> None:
    """List recent Tumblr posts."""
    creds = _load_creds()
    access_token = creds.get("oauth_token", "").strip()
    access_token_secret = creds.get("oauth_token_secret", "").strip()

    if not access_token:
        sys.stderr.write("No stored access token. Run auth-url/auth-code first.\n")
        raise SystemExit(1)

    blog = _get_blog_identifier()
    oauth = _get_oauth_session(token=access_token, token_secret=access_token_secret)

    list_url = f"{TUMBLR_API_BASE}/blog/{blog}/posts"
    params = {"limit": limit}

    try:
        resp = oauth.get(list_url, params=params)
    except Exception as e:
        sys.stderr.write(f"Tumblr API request failed: {e}\n")
        raise SystemExit(1)

    if resp.status_code != 200:
        sys.stderr.write(f"Tumblr API error: HTTP {resp.status_code}\n{resp.text}\n")
        raise SystemExit(1)

    data = resp.json()
    posts = data.get("response", {}).get("posts", [])

    if not posts:
        print("No posts found.")
        return

    for p in posts:
        pid = p.get("id", "?")
        post_type = p.get("type", "?")
        created = p.get("date", "?")

        # Summary text
        summary = p.get("summary", "") or ""
        caption = p.get("caption", "") or ""
        text_content = summary or _strip_html(caption)

        # Tags
        tags = p.get("tags", [])
        tags_str = ", ".join(tags) if tags else ""

        # URL
        short_url = p.get("short_url", "") or f"https://{blog}/post/{pid}"

        # Notes
        note_count = p.get("note_count", 0)

        print(f"ID:        {pid}")
        print(f"Type:      {post_type}")
        print(f"Date:      {created}")
        print(f"Notes:     {note_count}")
        if tags_str:
            print(f"Tags:      {tags_str}")
        print(f"Content:   {_truncate(text_content, 120)}")
        print(f"URL:       {short_url}")
        print("-" * 60)
        print()

    print(f"{len(posts)} post(s) shown.")


# ---------------------------------------------------------------------------
#  subcommand: delete
# ---------------------------------------------------------------------------

def cmd_delete(post_id: str) -> None:
    """Delete a Tumblr post by ID."""
    creds = _load_creds()
    access_token = creds.get("oauth_token", "").strip()
    access_token_secret = creds.get("oauth_token_secret", "").strip()

    if not access_token:
        sys.stderr.write("No stored access token. Run auth-url/auth-code first.\n")
        raise SystemExit(1)

    blog = _get_blog_identifier()
    oauth = _get_oauth_session(token=access_token, token_secret=access_token_secret)

    delete_url = f"{TUMBLR_API_BASE}/blog/{blog}/post/delete"
    payload = {"id": post_id}

    try:
        resp = oauth.post(delete_url, data=payload)
    except Exception as e:
        sys.stderr.write(f"Tumblr API request failed: {e}\n")
        raise SystemExit(1)

    if resp.status_code == 200:
        print(f"Deleted post {post_id}.")
    else:
        sys.stderr.write(f"Delete failed: HTTP {resp.status_code}\n{resp.text}\n")
        raise SystemExit(1)


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Post images to Tumblr via OAuth 1.0a."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # auth-url
    sub.add_parser("auth-url", help="Print the OAuth authorize URL")

    # auth-code <VERIFIER>
    p_code = sub.add_parser("auth-code", help="Exchange OAuth verifier for access token")
    p_code.add_argument("verifier", help="OAuth verifier from browser redirect")

    # post <PATH> [--text <TEXT>] [--dry-run] [--caption-field <FIELD>]
    p_post = sub.add_parser("post", help="Post an image to Tumblr")
    p_post.add_argument("path", help="Path to the image file")
    p_post.add_argument(
        "--text",
        default=None,
        help="Additional text to include in the caption.",
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

    # list [--limit <N>]
    p_list = sub.add_parser("list", help="List recent Tumblr posts")
    p_list.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of posts to list (default: 20).",
    )

    # delete <ID>
    p_del = sub.add_parser("delete", help="Delete a Tumblr post by ID")
    p_del.add_argument("post_id", help="ID of the post to delete")

    args = parser.parse_args(argv)

    if args.command == "auth-url":
        cmd_auth_url()
    elif args.command == "auth-code":
        cmd_auth_code(args.verifier)
    elif args.command == "post":
        cmd_post(
            image_path=args.path,
            text=args.text,
            dry_run=args.dry_run,
            caption_field=args.caption_field,
        )
    elif args.command == "list":
        cmd_list(limit=args.limit)
    elif args.command == "delete":
        cmd_delete(post_id=args.post_id)
    else:
        parser.print_help()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
