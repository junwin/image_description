"""Read comments and replies on recent Instagram media using Instagram Login."""

import json

from . import instagram_cli

MEDIA_FIELDS = "id,caption,permalink,timestamp,media_type"
COMMENT_FIELDS = "id,text,username,timestamp"


def _iter_edge(path, token, fields, limit=None):
    """Read cursor pages on the original edge; never follow token-bearing next URLs."""
    params = {"fields": fields, "limit": min(limit, 50) if limit else 50}
    seen_cursors = set()
    seen_ids = set()
    count = 0
    while True:
        response = instagram_cli._api("GET", path, token, params=dict(params))
        records = response.get("data")
        if not isinstance(records, list):
            raise RuntimeError("Instagram returned an invalid list response")
        for record in records:
            if not isinstance(record, dict) or not record.get("id"):
                raise RuntimeError("Instagram returned a record without an ID")
            record_id = str(record["id"])
            if record_id in seen_ids:
                continue
            seen_ids.add(record_id)
            yield record
            count += 1
            if limit is not None and count >= limit:
                return
        paging = response.get("paging") or {}
        if not paging.get("next"):
            return
        after = (paging.get("cursors") or {}).get("after")
        if not isinstance(after, str) or not after or after in seen_cursors:
            raise RuntimeError("Instagram pagination did not provide a new cursor; results incomplete")
        seen_cursors.add(after)
        params["after"] = after


def fetch_comments(user_id, token, posts=10, include_replies=True):
    """Fetch the most recent media returned by Instagram, then all visible comments."""
    if type(posts) is not int or posts <= 0:
        raise ValueError("--posts must be a positive integer")
    media = list(_iter_edge(f"{user_id}/media", token, MEDIA_FIELDS, limit=posts))
    report = []
    for post in media:
        item = dict(post)
        comments = list(_iter_edge(f"{post['id']}/comments", token, COMMENT_FIELDS))
        item["comments"] = []
        for comment in comments:
            entry = dict(comment)
            if include_replies:
                entry["replies"] = list(_iter_edge(
                    f"{comment['id']}/replies", token, COMMENT_FIELDS,
                ))
            item["comments"].append(entry)
        report.append(item)
    return {"instagram_user_id": user_id, "posts": report}


def _print_comment(comment, indent="  "):
    author = comment.get("username") or "unknown author"
    timestamp = comment.get("timestamp") or "unknown date"
    print(f"{indent}{author} [{timestamp}] (ID: {comment['id']})")
    for line in str(comment.get("text") or "").splitlines():
        print(f"{indent}  {line}")


def cmd_comments(posts=10, credentials=None, json_output=False, include_replies=True):
    # Validate before opening credentials or contacting Instagram.
    if type(posts) is not int or posts <= 0:
        raise ValueError("--posts must be a positive integer")
    user_id, token = instagram_cli.load_credentials(credentials)
    try:
        report = fetch_comments(user_id, token, posts=posts, include_replies=include_replies)
    except RuntimeError as exc:
        message = str(exc)
        if "HTTP 403" in message or "(code 10)" in message or "(code 200)" in message:
            raise RuntimeError(
                f"{message}. Check that this Instagram Login token was granted "
                "instagram_business_basic and instagram_business_manage_comments. "
                "An app permission request alone does not confirm the token's grants."
            ) from None
        raise
    # Print only after every requested edge has succeeded, so errors cannot look
    # like a successful empty/partial report. No sidecars or publish history change.
    if json_output:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report
    if not report["posts"]:
        print("No Instagram posts returned.")
    for index, post in enumerate(report["posts"], 1):
        print(f"\nPost {index}: {post.get('timestamp') or 'unknown date'} "
              f"[{post.get('media_type') or 'media'}] (ID: {post['id']})")
        if post.get("permalink"):
            print(post["permalink"])
        if post.get("caption"):
            print(post["caption"])
        if not post["comments"]:
            print("  No comments returned.")
        for comment in post["comments"]:
            _print_comment(comment)
            for reply in comment.get("replies", []):
                _print_comment(reply, indent="    Reply: ")
    return report
