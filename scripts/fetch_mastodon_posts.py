#!/usr/bin/env python3
"""Fetch the last N Mastodon posts and dump structured JSON.

Fields per post: post_id, url, date, hashtags (list), alt_text (full, per media).
Also includes a short content snippet for context and flags non-image posts.

Usage:
    python -m scripts.fetch_mastodon_posts --limit 20 --out /path/to/out.json
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from image_description.cli.mastodon_cli import _load_creds, _get_instance, _get_token, _api_request


def _plain_text(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html or "").strip()


def _unescape(s: str) -> str:
    import html as _html
    return _html.unescape(s or "")


def fetch(limit: int = 20):
    token = _get_token()
    creds = _load_creds()
    instance = _get_instance(creds)

    account = _api_request(instance, token, "GET", "/api/v1/accounts/verify_credentials")
    account_id = account.get("id")
    if not account_id:
        raise SystemExit("Could not get account ID.")

    path = f"/api/v1/accounts/{account_id}/statuses?limit={limit}"
    statuses = _api_request(instance, token, "GET", path)

    posts = []
    for s in statuses:
        media = s.get("media_attachments", [])
        hashtags = [t.get("name", "") for t in s.get("tags", [])]
        alt_texts = [_unescape(m.get("description", "")) for m in media]
        # filter out empty alt texts
        alt_texts = [a for a in alt_texts if a]

        posts.append({
            "post_id": s.get("id"),
            "url": s.get("url", ""),
            "date": s.get("created_at", ""),
            "hashtags": hashtags,
            "alt_text": alt_texts[0] if alt_texts else "",
            "has_media": bool(media),
            "content": _unescape(_plain_text(s.get("content", ""))),
        })

    return posts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--out", default=None, help="Output JSON path. Prints to stdout if omitted.")
    args = p.parse_args()

    posts = fetch(args.limit)
    payload = json.dumps(posts, indent=2, ensure_ascii=False)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(payload + "\n")
        print(f"Wrote {len(posts)} posts to {args.out}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
