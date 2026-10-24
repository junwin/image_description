"""Standalone Instagram Reel publisher; not part of the image processing pipeline."""
import argparse
from contextlib import nullcontext
import sys
import time
from pathlib import Path

import requests

from . import instagram_cli
from ..sidecar import Sidecar
from ..social_utils import build_social_text


def publish_reel(video_url, caption, user_id, token, share_to_feed=True,
                 poll_attempts=60, poll_interval=10):
    instagram_cli.validate_image_url(video_url)
    container = instagram_cli._api(
        "POST", f"{user_id}/media", token,
        data={"media_type": "REELS", "video_url": video_url,
              "caption": caption, "share_to_feed": str(share_to_feed).lower()},
    )
    container_id = str(container.get("id") or "")
    if not container_id:
        raise RuntimeError("Instagram returned no Reel container ID")
    for attempt in range(poll_attempts):
        result = instagram_cli._api(
            "GET", container_id, token, params={"fields": "status_code,status"}
        )
        status = result.get("status_code")
        if status == "FINISHED":
            break
        if status in ("ERROR", "EXPIRED"):
            raise RuntimeError(f"Instagram Reel processing failed: {result}")
        if attempt < poll_attempts - 1:
            time.sleep(poll_interval)
    else:
        raise RuntimeError(f"Reel container {container_id} did not finish; not published")

    published = instagram_cli._api(
        "POST", f"{user_id}/media_publish", token,
        data={"creation_id": container_id},
    )
    media_id = str(published.get("id") or "")
    if not media_id:
        raise RuntimeError("No Reel media ID returned; check Instagram before retrying")
    try:
        details = instagram_cli._api(
            "GET", media_id, token, params={"fields": "id,permalink"}
        )
        url = str(details.get("permalink") or "")
    except (RuntimeError, ValueError, requests.RequestException):
        url = ""
    return {"id": media_id, "permalink": url}


def cmd_reel(video_path, video_url=None, use_ngrok=False, dry_run=False,
             caption_field=None, text=None, credentials=None, share_to_feed=True):
    video = Path(video_path).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    if video.suffix.lower() != ".mp4":
        raise ValueError("Reels currently support MP4 files only")
    sidecar_path = instagram_cli.find_sidecar(video)
    sidecar = Sidecar.load(str(sidecar_path))
    caption = build_social_text(
        sidecar.to_dict(), caption_field=caption_field, extra_text=text
    )
    if video_url and use_ngrok:
        raise ValueError("Choose --ngrok or --video-url, not both")
    if video_url:
        instagram_cli.validate_image_url(video_url)
    if dry_run:
        print(f"DRY RUN - nothing published\nVideo: {video}\n"
              f"Sidecar: {sidecar_path}\nCaption:\n{caption}")
        return
    if not video_url and not use_ngrok:
        raise ValueError("--video-url or --ngrok is required")
    user_id, token = instagram_cli.load_credentials(credentials)
    if use_ngrok:
        from .instagram_video_ngrok import serve_video
        source = serve_video(video)
    else:
        source = nullcontext(video_url)
    with source as url:
        result = publish_reel(url, caption, user_id, token, share_to_feed)
    sidecar.add_publish_event("instagram", result["id"], result["permalink"])
    sidecar.save(str(sidecar_path))
    print(f"Published Instagram Reel {result['id']}")
    if result["permalink"]:
        print(f"URL: {result['permalink']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Publish an MP4 Instagram Reel")
    parser.add_argument("video", help="MP4 file, with adjacent JSON sidecar")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video-url", help="Public direct HTTPS MP4 URL")
    source.add_argument("--ngrok", action="store_true", help="Temporarily host the MP4")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--caption-field")
    parser.add_argument("--text")
    parser.add_argument("--credentials")
    parser.add_argument("--no-share-to-feed", action="store_true")
    args = parser.parse_args(argv)
    try:
        cmd_reel(args.video, args.video_url, args.ngrok, args.dry_run,
                 args.caption_field, args.text, args.credentials,
                 not args.no_share_to_feed)
    except (OSError, ValueError, KeyError, RuntimeError, requests.RequestException) as exc:
        print(f"Instagram Reel publish failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
