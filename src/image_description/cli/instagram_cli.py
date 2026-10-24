"""Publish public JPEG URLs to Instagram using separate Instagram Login credentials.

Matches the Facebook CLI's sidecar caption and publish-history conventions.
Instagram fetches images over HTTPS; it does not accept local JPEG uploads.
"""
import argparse
from contextlib import nullcontext
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from ..sidecar import Sidecar
from ..social_utils import build_social_text
from .facebook_cli import find_sidecar

API_BASE = "https://graph.instagram.com/v26.0"
DEFAULT_CREDENTIAL_PATH = Path.home() / ".config" / "instagram" / "jdunwin_credentials.json"


def load_credentials(path=None):
    path = Path(path or os.environ.get("INSTAGRAM_CREDENTIAL_PATH") or DEFAULT_CREDENTIAL_PATH).expanduser()
    data = json.loads(path.read_text(encoding="utf-8"))
    user_id = str(data.get("instagram_user_id") or "").strip()
    token = str(data.get("access_token") or "").strip()
    if not user_id or not token:
        raise ValueError(f"Missing instagram_user_id or access_token in {path}")
    return user_id, token


def validate_image_url(image_url):
    parsed = urlparse(image_url)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("--image-url must be a public HTTPS URL without embedded credentials")
    return image_url


def _api(method, path, token, **kwargs):
    response = requests.request(
        method, f"{API_BASE}/{path}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30, **kwargs,
    )
    if not response.ok:
        # Never include the token or a potentially credential-bearing request URL in errors.
        try:
            error = response.json().get("error", {})
        except ValueError:
            error = {}
        raise RuntimeError(
            f"Instagram API HTTP {response.status_code}: "
            f"{error.get('message', 'request failed')} (code {error.get('code', 'unknown')})"
        )
    return response.json()


def publish_photo(image_url, caption, user_id, token, poll_attempts=10, poll_interval=10):
    """Create, wait for, then publish a media container; return media ID and permalink."""
    validate_image_url(image_url)
    container = _api("POST", f"{user_id}/media", token,
                     data={"image_url": image_url, "caption": caption})
    container_id = str(container.get("id") or "")
    if not container_id:
        raise RuntimeError("Instagram returned no container ID")

    for attempt in range(poll_attempts):
        status = _api("GET", container_id, token, params={"fields": "status_code"}).get("status_code")
        if status == "FINISHED":
            break
        if status in ("ERROR", "EXPIRED"):
            raise RuntimeError(f"Instagram container {container_id} status: {status}")
        if attempt < poll_attempts - 1:
            time.sleep(poll_interval)
    else:
        raise RuntimeError(f"Instagram container {container_id} not ready; nothing published")

    published = _api("POST", f"{user_id}/media_publish", token,
                     data={"creation_id": container_id})
    media_id = str(published.get("id") or "")
    if not media_id:
        raise RuntimeError("Instagram publish returned no media ID; check Instagram before retrying")
    # Publishing is already complete. A permalink lookup failure must not encourage re-posting.
    try:
        media = _api("GET", media_id, token, params={"fields": "id,permalink"})
        permalink = str(media.get("permalink") or "")
    except (RuntimeError, ValueError, requests.RequestException):
        permalink = ""
    return {"id": media_id, "permalink": permalink}


def cmd_post(image_path, image_url=None, dry_run=False, caption_field=None, text=None, credentials=None, use_ngrok=False):
    image = Path(image_path).expanduser().resolve()
    if not image.is_file():
        raise FileNotFoundError(image)
    if image.suffix.lower() not in (".jpg", ".jpeg"):
        raise ValueError("Only JPEG images are currently supported")

    sidecar_path = find_sidecar(image)
    sidecar = Sidecar.load(str(sidecar_path))
    caption = build_social_text(sidecar.to_dict(), caption_field=caption_field, extra_text=text)

    if image_url and use_ngrok:
        raise ValueError("Choose --ngrok or --image-url, not both")
    if image_url:
        validate_image_url(image_url)
    if dry_run:
        print(f"DRY RUN - nothing published\nImage: {image}\nImage URL: {image_url or '(not supplied)'}\nSidecar: {sidecar_path}\nCaption:\n{caption}")
        return
    if not image_url and not use_ngrok:
        raise ValueError("--image-url or --ngrok is required for Instagram publishing")

    user_id, token = load_credentials(credentials)
    if use_ngrok:
        from .instagram_ngrok import serve_image
        image_source = serve_image(image)
    else:
        image_source = nullcontext(image_url)
    with image_source as hosted_url:
        print(f"Instagram image URL: {hosted_url}")
        result = publish_photo(hosted_url, caption, user_id, token)
    media_id = result["id"]
    permalink = result["permalink"]
    sidecar.add_publish_event("instagram", media_id, permalink)
    sidecar.save(str(sidecar_path))
    print(f"Published Instagram media {media_id}")
    if permalink:
        print(f"URL: {permalink}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Publish a JPEG via public HTTPS URL to Instagram")
    sub = parser.add_subparsers(dest="command", required=True)
    post = sub.add_parser("post", help="Publish an image to the configured Instagram account")
    post.add_argument("image", help="Local JPEG used to find its JSON sidecar")
    post.add_argument("--image-url", help="Publicly accessible HTTPS JPEG URL")
    post.add_argument("--ngrok", action="store_true", help="Temporarily serve this JPEG over an ngrok HTTPS tunnel")
    post.add_argument("--dry-run", action="store_true", help="Preview without credentials, API calls or sidecar edits")
    post.add_argument("--caption-field", help="Alternative sidecar caption field")
    post.add_argument("--text", help="Additional caption text")
    post.add_argument("--credentials", help="Override credential JSON file path")
    args = parser.parse_args(argv)
    try:
        if args.command == "post":
            cmd_post(args.image, args.image_url, args.dry_run, args.caption_field, args.text, args.credentials, args.ngrok)
    except (OSError, ValueError, KeyError, RuntimeError, requests.RequestException) as exc:
        print(f"Instagram publish failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
