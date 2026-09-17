#!/usr/bin/env python3
"""Fetch up to N Mastodon posts (paginated) and export JSON + readable markdown.

Fields per post: post_id, url, date (created_at), content (plain text),
hashtags (list), alt_texts (list, one per media), has_media.

Usage:
    python -m scripts.fetch_mastodon_export --limit 200 \
        --out-json /home/junwin/pishare/photography/work/2026/output/publish_data/masto_data.json \
        --out-md   /home/junwin/pishare/photography/work/2026/output/publish_data/masto_data.md
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from image_description.cli.mastodon_cli import (
    _load_creds,
    _get_instance,
    _get_token,
    _api_request,
)


def _plain_text(html):
    return re.sub(r"<[^>]+>", "", html or "").strip()


def _unescape(s):
    import html as _html
    return _html.unescape(s or "")


def fetch(limit=200):
    token = _get_token()
    creds = _load_creds()
    instance = _get_instance(creds)

    account = _api_request(instance, token, "GET", "/api/v1/accounts/verify_credentials")
    account_id = account.get("id")
    if not account_id:
        raise SystemExit("Could not get account ID.")

    posts = []
    max_id = None
    page_size = 40  # Mastodon default max per page
    while len(posts) < limit:
        path = f"/api/v1/accounts/{account_id}/statuses?limit={page_size}"
        if max_id is not None:
            path += f"&max_id={max_id}"
        statuses = _api_request(instance, token, "GET", path)
        if not statuses:
            break
        for s in statuses:
            media = s.get("media_attachments", [])
            hashtags = [t.get("name", "") for t in s.get("tags", [])]
            alt_texts = [_unescape(m.get("description", "")) for m in media]
            alt_texts = [a for a in alt_texts if a]

            posts.append({
                "post_id": s.get("id"),
                "url": s.get("url", ""),
                "date": s.get("created_at", ""),
                "content": _unescape(_plain_text(s.get("content", ""))),
                "hashtags": hashtags,
                "alt_texts": alt_texts,
                "has_media": bool(media),
            })
        # Paginate: next page is older than the oldest status we just got.
        oldest = statuses[-1].get("id")
        if not oldest or oldest == max_id:
            break
        max_id = int(oldest) - 1

    return posts[:limit]


def _md(posts):
    lines = ["# Mastodon Posts", ""]
    lines.append(f"Total: {len(posts)} posts")
    lines.append("")
    for i, p in enumerate(posts, 1):
        lines.append(f"## {i}. {p['date'][:10]} — {p['post_id']}")
        lines.append(f"- **post_id**: {p['post_id']}")
        lines.append(f"- **url**: {p['url']}")
        lines.append(f"- **date**: {p['date']}")
        tags = " ".join("#" + t for t in p["hashtags"]) if p["hashtags"] else "(none)"
        lines.append(f"- **hashtags**: {tags}")
        alt = " || ".join(p["alt_texts"]) if p["alt_texts"] else "(no alt text)"
        lines.append(f"- **alt_text**: {alt}")
        content = p["content"].replace("\n", " / ") if p["content"] else "(none)"
        lines.append(f"- **description**: {content}")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()

    posts = fetch(args.limit)
    print(f"Fetched {len(posts)} posts.", file=sys.stderr)

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(posts, f, indent=2, ensure_ascii=False)
        print(f"Wrote JSON: {args.out_json}", file=sys.stderr)

    if args.out_md:
        os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write(_md(posts) + "\n")
        print(f"Wrote MD: {args.out_md}", file=sys.stderr)

    if not args.out_json and not args.out_md:
        print(json.dumps(posts, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
