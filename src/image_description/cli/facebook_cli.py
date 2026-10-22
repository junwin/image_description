"""Publish a local JPEG to a Facebook Page using its saved Page token.

Uses the same caption and publish-history conventions as the other social CLIs.
No Facebook credentials are ever stored in the repository.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import requests

from ..sidecar import Sidecar
from ..social_utils import build_social_text

API_BASE = "https://graph.facebook.com/v26.0"
DEFAULT_CREDENTIAL_PATH = Path.home() / ".config" / "facebook" / "page_credentials.json"


def load_credentials(path=None):
    path = Path(path or os.environ.get("FACEBOOK_CREDENTIAL_PATH") or DEFAULT_CREDENTIAL_PATH).expanduser()
    data = json.loads(path.read_text(encoding="utf-8"))
    page_id = str(data.get("page_id") or "").strip()
    token = str(data.get("page_access_token") or "").strip()
    if not page_id or not token:
        raise ValueError(f"Missing page_id or page_access_token in {path}")
    return page_id, token


def find_sidecar(image_path):
    image = Path(image_path)
    for path in (image.with_suffix(".json"), image.with_suffix(".social.json")):
        if path.is_file():
            return path
    raise FileNotFoundError(f"No JSON sidecar found for {image}")


def publish_photo(image_path, caption, page_id, page_token):
    """Return the Page photo/post identifiers returned by Facebook."""
    image = Path(image_path)
    if image.suffix.lower() not in (".jpg", ".jpeg"):
        raise ValueError("Facebook photo publisher currently accepts JPEG files only")
    if not image.is_file():
        raise FileNotFoundError(image)
    with image.open("rb") as handle:
        response = requests.post(
            f"{API_BASE}/{page_id}/photos",
            data={"access_token": page_token, "message": caption, "published": "true"},
            files={"source": (image.name, handle, "image/jpeg")},
            timeout=60,
        )
    response.raise_for_status()
    result = response.json()
    if not result.get("id"):
        raise RuntimeError("Facebook returned no photo ID")
    return result


def cmd_post(image_path, dry_run=False, caption_field=None, text=None, credentials=None):
    image = Path(image_path).expanduser().resolve()
    if not image.is_file():
        raise FileNotFoundError(image)
    if image.suffix.lower() not in (".jpg", ".jpeg"):
        raise ValueError("Only JPEG images are currently supported")

    sidecar_path = find_sidecar(image)
    sidecar = Sidecar.load(str(sidecar_path))
    caption = build_social_text(sidecar.to_dict(), caption_field=caption_field, extra_text=text)

    if dry_run:
        print(f"DRY RUN - nothing published\nImage: {image}\nSidecar: {sidecar_path}\nCaption:\n{caption}")
        return

    page_id, token = load_credentials(credentials)
    result = publish_photo(image, caption, page_id, token)
    photo_id = str(result["id"])
    post_id = str(result.get("post_id") or photo_id)
    url = f"https://www.facebook.com/{post_id}" if result.get("post_id") else ""
    sidecar.add_publish_event("facebook", post_id, url)
    sidecar.save(str(sidecar_path))
    print(f"Published Facebook photo {photo_id}")
    if result.get("post_id"):
        print(f"Post ID: {post_id}\nURL: {url}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Publish JPEGs to a Facebook Page using sidecar metadata")
    sub = parser.add_subparsers(dest="command", required=True)
    post = sub.add_parser("post", help="Publish an image to the configured Facebook Page")
    post.add_argument("image")
    post.add_argument("--dry-run", action="store_true", help="Preview without credentials, upload, or sidecar changes")
    post.add_argument("--caption-field", help="Alternative sidecar caption field")
    post.add_argument("--text", help="Additional caption text")
    post.add_argument("--credentials", help="Override credential JSON file path")
    args = parser.parse_args(argv)
    try:
        if args.command == "post":
            cmd_post(args.image, args.dry_run, args.caption_field, args.text, args.credentials)
    except (OSError, ValueError, KeyError, RuntimeError, requests.RequestException) as exc:
        print(f"Facebook publish failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
