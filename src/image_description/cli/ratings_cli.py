#!/usr/bin/env python3
"""
CLI for assessing engagement of published posts (issue #20: Ratings script).

Walks a folder of sidecar JSON files, reads each sidecar's publish_history,
fetches current engagement data (likes / reposts / comments) from every
platform the image was published to, and writes the result back into the
sidecar next to the publish_history entry.

Only the LAST scan is kept for each platform (the engagement object is
overwritten in place).

An overall engagement score is computed across all platforms and stored as
`engagement_score` on the sidecar (the Sidecar dataclass preserves unknown
fields via `extra`, so nothing is lost on save).

Run it from the repo root:

    bash -lc "source .venv/bin/activate && \
        python -m src.image_description.cli.ratings_cli <subcommand> <args>"

Subcommands:
    scan <FOLDER>    Fetch engagement for every sidecar with publish_history
                     and write the data back into the sidecar files.
    report <FOLDER>  Rank sidecars by engagement score (no API calls; uses
                     the data already stored by `scan`).

Options:
    --platforms p1,p2   Only query these platforms (default: all)
    --refresh           Refetch even if cached data is less than a day old
    --file-delay N      Seconds to pause between sidecars after live API
                        calls (default 10; 0 disables)
    --dry-run           Fetch but do not write anything back
    --json              Print machine-readable JSON output

Score formula (per platform, then summed):
    score = likes * 1 + reposts * 2 + comments * 3
(Tumblr exposes a single note_count; it is recorded as `notes` and counted
like likes. Adjust the weights below if you want a different ranking.)

Caching (be nice to the platforms):
    Each publish_history entry's engagement object stores a scanned_at
    timestamp. If that scan is less than SCAN_CACHE_DAYS old (default 1
    day), the cached counts are reused and the platform is NOT contacted
    again. Use --refresh to force a refetch.

Error handling:
    - A missing/deleted post (404) or a per-platform API error is recorded
      as an error for that platform; the scan continues.
    - Rate-limit (429) responses are retried with exponential backoff, and a
      small delay is inserted between API calls to stay well within limits.
    - After a sidecar that made live API calls, the scan pauses
      FILE_DELAY seconds (default 10) before the next file, so a full
      folder scan trickles gently instead of hammering the platforms.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from ..sidecar import Sidecar

# Score weights: likes, reposts/boosts/reblogs, comments/replies.
WEIGHT_LIKES = 1
WEIGHT_REPOSTS = 2
WEIGHT_COMMENTS = 3

# Politeness / rate-limit settings.
REQUEST_DELAY = 0.5     # seconds to sleep between platform API calls
FILE_DELAY = 10         # seconds to pause between sidecar files (live fetches)
MAX_RETRIES = 3         # retries for rate-limit (429) responses
RETRY_BASE_DELAY = 5    # seconds; doubles each retry (5, 10, 20)
SCAN_CACHE_DAYS = 1     # reuse cached engagement younger than this


# ---------------------------------------------------------------------------
#  engagement schema
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().isoformat()


def make_engagement(
    likes: int = 0,
    reposts: int = 0,
    comments: int = 0,
    notes: Optional[int] = None,
    score: int = 0,
) -> Dict[str, Any]:
    """Build the engagement object stored in a publish_history entry."""
    data: Dict[str, Any] = {
        "likes": likes,
        "reposts": reposts,
        "comments": comments,
        "score": score,
        "scanned_at": _now_iso(),
    }
    if notes is not None:
        data["notes"] = notes
    return data


def engagement_is_fresh(engagement: Any) -> bool:
    """True if a stored engagement object was scanned less than a day ago.

    Anything missing, malformed, or with an unparsable timestamp is
    treated as stale (so it is refetched).
    """
    if not isinstance(engagement, dict):
        return False
    scanned_at = engagement.get("scanned_at")
    if not isinstance(scanned_at, str) or not scanned_at.strip():
        return False
    try:
        scanned = datetime.fromisoformat(scanned_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    if scanned.tzinfo is None:
        scanned = scanned.replace(tzinfo=timezone.utc)
    return _now() - scanned < timedelta(days=SCAN_CACHE_DAYS)


# ---------------------------------------------------------------------------
#  platform fetchers
#
# Each fetcher reuses the credential files and API helpers already used by
# the publishing CLIs (no new credentials, no new API code).
#
# All fetchers return (likes, reposts, comments) or raise.
# ---------------------------------------------------------------------------

def fetch_mastodon(post_id: str) -> Tuple[int, int, int]:
    """Mastodon: GET /api/v1/statuses/{id} (favourites / reblogs / replies)."""
    from . import mastodon_cli

    creds = mastodon_cli._load_creds()
    instance = mastodon_cli._get_instance(creds)
    token = mastodon_cli._get_token()
    status = mastodon_cli._api_request(
        instance, token, "GET", f"/api/v1/statuses/{post_id}"
    )
    return (
        int(status.get("favourites_count", 0)),
        int(status.get("reblogs_count", 0)),
        int(status.get("replies_count", 0)),
    )


def fetch_pixelfed(post_id: str) -> Tuple[int, int, int]:
    """Pixelfed: Mastodon-compatible API on pixelfed.social."""
    from . import pixelfed_cli

    token = pixelfed_cli._get_token()
    resp = pixelfed_cli.requests.get(
        f"{pixelfed_cli.API_BASE}/api/v1/statuses/{post_id}",
        headers=pixelfed_cli._headers(token),
        timeout=30,
    )
    resp.raise_for_status()
    status = resp.json()
    return (
        int(status.get("favourites_count", 0)),
        int(status.get("reblogs_count", 0)),
        int(status.get("replies_count", 0)),
    )


def fetch_tumblr(post_id: str) -> Tuple[int, int, int, int]:
    """Tumblr: GET /v2/blog/{blog}/posts?id={post_id}. Returns (likes, reposts, comments, notes).

    Fetching a single post uses the `id` query parameter on the blog posts
    endpoint (the /posts/{x} path segment is a post-type filter, not an id).
    This endpoint only exposes a single note_count; individual like / reblog
    / reply breakdowns are not available here, so notes is the aggregate
    and is also returned separately for scoring.
    """
    from . import tumblr_cli

    creds = tumblr_cli._load_creds()
    token = creds.get("oauth_token", "").strip()
    secret = creds.get("oauth_token_secret", "").strip()
    if not token:
        raise RuntimeError("No stored Tumblr access token (run tumblr_cli auth first).")
    blog = tumblr_cli._get_blog_identifier()
    oauth = tumblr_cli._get_oauth_session(token=token, token_secret=secret)

    resp = oauth.get(
        f"{tumblr_cli.TUMBLR_API_BASE}/blog/{blog}/posts",
        params={"id": post_id},
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Tumblr API error: HTTP {resp.status_code} {resp.text[:200]}")
    posts = resp.json().get("response", {}).get("posts", [])
    if not posts:
        raise RuntimeError(f"Tumblr post {post_id} not found on {blog}.")
    note_count = int(posts[0].get("note_count", 0))
    # No breakdown available from this endpoint: report notes as likes bucket.
    return (note_count, 0, 0, note_count)


def fetch_bluesky(post_id: str) -> Tuple[int, int, int]:
    """Bluesky: atproto get_posts. post_id is an at:// URI."""
    from . import bluesky_cli

    client = bluesky_cli._get_client()
    response = client.get_posts([post_id])
    if not response.posts:
        raise RuntimeError(f"Bluesky post not found: {post_id}")
    post = response.posts[0]
    return (
        int(post.like_count or 0),
        int(post.repost_count or 0),
        int(post.reply_count or 0),
    )


# Registry: platform name -> fetcher. Add new platforms here.
FETCHERS = {
    "mastodon": fetch_mastodon,
    "pixelfed": fetch_pixelfed,
    "tumblr": fetch_tumblr,
    "bluesky": fetch_bluesky,
}


def fetch_engagement(platform: str, post_id: str) -> Tuple[Dict[str, Any], int]:
    """Fetch engagement for one published post.

    Returns (engagement_dict, platform_score).
    Raises for unsupported platforms or API errors.
    """
    fetcher = FETCHERS.get(platform)
    if fetcher is None:
        raise RuntimeError(f"Unsupported platform: {platform}")

    if platform == "tumblr":
        likes, reposts, comments, notes = fetcher(post_id)
        score = (
            WEIGHT_LIKES * likes
            + WEIGHT_REPOSTS * reposts
            + WEIGHT_COMMENTS * comments
        )
        return make_engagement(likes, reposts, comments, notes=notes, score=score), score

    likes, reposts, comments = fetcher(post_id)
    score = (
        WEIGHT_LIKES * likes
        + WEIGHT_REPOSTS * reposts
        + WEIGHT_COMMENTS * comments
    )
    return make_engagement(likes, reposts, comments, score=score), score


# ---------------------------------------------------------------------------
#  fetch wrapper: rate-limit retries + error normalization
# ---------------------------------------------------------------------------

def _is_rate_limit(msg: str) -> bool:
    """Heuristic: does an error message indicate a rate-limit response?"""
    m = msg.lower()
    return "429" in m or "rate limit" in m or "too many requests" in m


def _fetch_with_retries(platform: str, post_id: str) -> Tuple[Dict[str, Any], int]:
    """Fetch engagement, retrying with backoff on rate-limit responses.

    API errors such as 404 (deleted/missing post) raise immediately;
    rate-limit responses are retried up to MAX_RETRIES times. SystemExit
    raised by the CLI helpers is normalized into RuntimeError so a single
    failing post can be recorded instead of aborting the scan.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            return fetch_engagement(platform, post_id)
        except SystemExit as e:
            raise RuntimeError(str(e) or f"{platform}: CLI error (exit)") from e
        except Exception as e:
            if not _is_rate_limit(str(e)):
                raise
            last_exc = e
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            print(
                f"  rate limited on {platform}, retrying in {delay}s "
                f"(attempt {attempt + 1}/{MAX_RETRIES})...",
                file=sys.stderr,
            )
            time.sleep(delay)
    raise RuntimeError(
        f"{platform}: still rate limited after {MAX_RETRIES} retries ({last_exc})"
    )


# ---------------------------------------------------------------------------
#  sidecar scanning
# ---------------------------------------------------------------------------

def iter_sidecar_paths(folder: str) -> List[str]:
    """Walk `folder` recursively and return every sidecar JSON path.

    A sidecar is any .json file that contains a publish_history list.
    Invalid JSON files are reported and skipped.
    """
    paths: List[str] = []
    for root, _dirs, files in os.walk(folder):
        for name in sorted(files):
            if not name.endswith(".json"):
                continue
            path = os.path.join(root, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"  skipping unreadable file: {path} ({e})", file=sys.stderr)
                continue
            if isinstance(data, dict) and isinstance(data.get("publish_history"), list):
                paths.append(path)
    return paths


def scan_sidecar(
    sidecar_path: str,
    platforms: Optional[List[str]] = None,
    dry_run: bool = False,
    refresh: bool = False,
) -> Dict[str, Any]:
    """Scan one sidecar: fetch engagement for each platform in publish_history.

    Cached engagement scanned less than SCAN_CACHE_DAYS ago is reused and
    the platform is not contacted again, unless refresh is True.

    Returns a summary dict: {"path", "title", "score", "platforms",
    "errors", "live_fetches"}.
    """
    sidecar = Sidecar.load(sidecar_path)
    total_score = 0
    live_fetches = 0
    errors: List[str] = []
    platform_rows: List[Dict[str, Any]] = []

    for entry in sidecar.publish_history:
        platform = str(entry.get("platform", "")).strip().lower()
        post_id = str(entry.get("post_id", "")).strip()
        if not platform or not post_id:
            continue
        if platforms is not None and platform not in platforms:
            platform_rows.append({"platform": platform, "skipped": "filtered"})
            continue
        if platform not in FETCHERS:
            platform_rows.append({"platform": platform, "skipped": "unsupported"})
            continue

        cached = entry.get("engagement")
        if not refresh and engagement_is_fresh(cached):
            # Be nice: reuse data scanned less than a day ago.
            score = int(cached.get("score", 0))
            total_score += score
            platform_rows.append(
                {
                    "platform": platform,
                    "likes": cached.get("likes", 0),
                    "reposts": cached.get("reposts", 0),
                    "comments": cached.get("comments", 0),
                    "score": score,
                    "cached": True,
                }
            )
            continue

        try:
            engagement, score = _fetch_with_retries(platform, post_id)
        except Exception as e:
            # A missing/deleted post (404) or persistent API error is
            # recorded against this platform; the scan continues.
            errors.append(f"{platform}/{post_id}: {e}")
            platform_rows.append({"platform": platform, "error": str(e)})
            continue

        time.sleep(REQUEST_DELAY)  # be polite to the platform APIs
        live_fetches += 1

        entry["engagement"] = engagement  # last scan wins (issue requirement)
        total_score += score
        platform_rows.append(
            {
                "platform": platform,
                "likes": engagement["likes"],
                "reposts": engagement["reposts"],
                "comments": engagement["comments"],
                "score": score,
            }
        )

    # Stored at the top level of the sidecar JSON (preserved via Sidecar.extra).
    sidecar.extra["engagement_score"] = total_score
    if not dry_run:
        sidecar.save(sidecar_path)

    return {
        "path": sidecar_path,
        "title": sidecar.title or sidecar.original_title or os.path.basename(sidecar_path),
        "score": total_score,
        "platforms": platform_rows,
        "errors": errors,
        "live_fetches": live_fetches,
    }


# ---------------------------------------------------------------------------
#  subcommands
# ---------------------------------------------------------------------------

def cmd_scan(
    folder: str,
    platforms: Optional[List[str]],
    dry_run: bool,
    as_json: bool,
    refresh: bool,
    file_delay: float,
) -> None:
    paths = iter_sidecar_paths(folder)
    if not paths:
        print(f"No sidecar files with publish_history found under: {folder}")
        return
    print(f"Scanning {len(paths)} sidecar(s) under {folder}"
          f"{' (dry run)' if dry_run else ''}...\n")

    results: List[Dict[str, Any]] = []
    failures = 0
    for index, path in enumerate(paths):
        summary = scan_sidecar(
            path, platforms=platforms, dry_run=dry_run, refresh=refresh
        )
        results.append(summary)
        if not as_json:
            title = summary["title"]
            print(f"{summary['score']:>6}  {title}")
            for row in summary["platforms"]:
                if "error" in row:
                    print(f"         ! {row['platform']}: {row['error']}")
                    failures += 1
                elif "skipped" in row:
                    print(f"         - {row['platform']}: skipped ({row['skipped']})")
                else:
                    marker = " (cached)" if row.get("cached") else ""
                    print(
                        f"         {row['platform']}: "
                        f"likes={row['likes']} reposts={row['reposts']} "
                        f"comments={row['comments']} (score {row['score']}){marker}"
                    )
        elif summary["errors"]:
            failures += len(summary["errors"])

        # Be nice: pause between files that hit the APIs. Fully-cached
        # or skipped files do not need the pause.
        if file_delay > 0 and summary["live_fetches"] > 0 and index < len(paths) - 1:
            print(f"         (pausing {file_delay:g}s before next file...)",
                  file=sys.stderr)
            time.sleep(file_delay)

    if as_json:
        print(json.dumps(results, indent=2, ensure_ascii=False))

    written = "would write" if dry_run else "wrote"
    print(f"\nDone. {written} engagement data for {len(paths)} sidecar(s).")
    if failures:
        print(f"{failures} platform fetch(es) failed - see messages above.", file=sys.stderr)


def cmd_report(folder: str, as_json: bool, limit: int = 0) -> None:
    """Rank sidecars by their stored engagement_score (no API calls)."""
    rows: List[Dict[str, Any]] = []
    for path in iter_sidecar_paths(folder):
        sidecar = Sidecar.load(path)
        score = sidecar.extra.get("engagement_score", 0)
        n_platforms = len(sidecar.publish_history)
        rows.append(
            {
                "score": score,
                "title": sidecar.title or sidecar.original_title or os.path.basename(path),
                "platforms": n_platforms,
                "path": path,
            }
        )
    rows.sort(key=lambda r: r["score"], reverse=True)
    if limit:
        rows = rows[:limit]

    if as_json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return

    print(f"{'Score':>7}  {'Plat':>4}  Title")
    print("-" * 70)
    for row in rows:
        print(f"{row['score']:>7}  {row['platforms']:>4}  {row['title']}")


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------

def _parse_platforms(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [p.strip().lower() for p in value.split(",") if p.strip()]


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fetch and rank engagement of published posts (issue #20)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="Fetch engagement and update sidecars")
    p_scan.add_argument("folder", help="Folder to walk for sidecar JSON files")
    p_scan.add_argument(
        "--platforms", default=None,
        help="Comma-separated platform filter (default: all known platforms)",
    )
    p_scan.add_argument(
        "--refresh", action="store_true",
        help="Refetch even if cached data is less than a day old",
    )
    p_scan.add_argument(
        "--file-delay", type=float, default=FILE_DELAY,
        help=f"Seconds to pause between sidecars after live API calls "
             f"(default {FILE_DELAY}; 0 disables)",
    )
    p_scan.add_argument(
        "--dry-run", action="store_true",
        help="Fetch but do not write anything back",
    )
    p_scan.add_argument("--json", action="store_true", help="JSON output")

    p_report = sub.add_parser("report", help="Rank sidecars by engagement score")
    p_report.add_argument("folder", help="Folder to walk for sidecar JSON files")
    p_report.add_argument("--limit", type=int, default=0, help="Show top N only")
    p_report.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args(argv)

    if args.command == "scan":
        cmd_scan(
            folder=args.folder,
            platforms=_parse_platforms(args.platforms),
            dry_run=args.dry_run,
            as_json=args.json,
            refresh=args.refresh,
            file_delay=args.file_delay,
        )
    elif args.command == "report":
        cmd_report(folder=args.folder, as_json=args.json, limit=args.limit)
    else:
        parser.print_help()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
