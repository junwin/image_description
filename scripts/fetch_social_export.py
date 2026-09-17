#!/usr/bin/env python3
"""Fetch posts from a Mastodon-compatible API (Mastodon or Pixelfed).

Pixelfed exposes a Mastodon-compatible REST API, so the same fetch logic works
for both. Fields per post: platform, post_id, url, date, content, hashtags,
alt_texts, has_media.

Usage:
    python -m scripts.fetch_social_export --platform pixelfed --limit 200 \
        --out-json <path.json> --out-md <path.md>
    python -m scripts.fetch_social_export --platform mastodon --limit 200 \
        --out-json <path.json> --out-md <path.md>
"""
import argparse
import html as _html
import json
import os
import re
import sys
import urllib.error
import urllib.request

CRED_PATHS = {
    "mastodon": "/home/junwin/credential/mastodon.json",
    "pixelfed": "/home/junwin/credential/pixelfed.json",
}
DEFAULT_INSTANCE = {
    "mastodon": "mastodon.social",
    "pixelfed": "pixelfed.social",
}


def _load_creds(platform):
    path = CRED_PATHS[platform]
    if not os.path.exists(path):
        raise SystemExit(f"Credential file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _api_request(instance, token, method, path):
    url = f"https://{instance}{path}"
    headers = {"Authorization": f"Bearer {token}"}
    req = urllib.request.Request(url, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise SystemExit(f"API error HTTP {e.code} for {url}\n{body[:2000]}")


def _plain_text(html_text):
    return re.sub(r"<[^>]+>", "", html_text or "").strip()


def _unescape(s):
    return _html.unescape(s or "")


def fetch(platform, limit=200):
    creds = _load_creds(platform)
    instance = creds.get("instance", "").strip() or DEFAULT_INSTANCE[platform]
    token = creds.get("access_token", "").strip()
    if not token:
        raise SystemExit(f"No access_token for {platform}.")

    account = _api_request(instance, token, "GET", "/api/v1/accounts/verify_credentials")
    account_id = account.get("id")
    if not account_id:
        raise SystemExit("Could not get account ID.")

    posts = []
    max_id = None
    page_size = 40
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
                "platform": platform,
                "post_id": s.get("id"),
                "url": s.get("url", ""),
                "date": s.get("created_at", ""),
                "content": _unescape(_plain_text(s.get("content", ""))),
                "hashtags": hashtags,
                "alt_texts": alt_texts,
                "has_media": bool(media),
            })
        oldest = statuses[-1].get("id")
        if not oldest or oldest == max_id:
            break
        max_id = int(oldest) - 1

    return posts[:limit]


def _md(posts, platform):
    lines = [f"# {platform.title()} Posts", ""]
    lines.append(f"Total: {len(posts)} posts")
    lines.append("")
    for i, p in enumerate(posts, 1):
        lines.append(f"## {i}. {p['date'][:10]} — {p['post_id']}")
        lines.append(f"- **platform**: {p['platform']}")
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
    ap.add_argument("--platform", required=True, choices=["mastodon", "pixelfed"])
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()

    posts = fetch(args.platform, args.limit)
    print(f"Fetched {len(posts)} posts from {args.platform}.", file=sys.stderr)

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(posts, f, indent=2, ensure_ascii=False)
        print(f"Wrote JSON: {args.out_json}", file=sys.stderr)

    if args.out_md:
        os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write(_md(posts, args.platform) + "\n")
        print(f"Wrote MD: {args.out_md}", file=sys.stderr)

    if not args.out_json and not args.out_md:
        print(json.dumps(posts, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
