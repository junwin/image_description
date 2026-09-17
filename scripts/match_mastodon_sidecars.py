#!/usr/bin/env python3
"""Match Mastodon posts (hashtags + alt text) to image sidecar JSON files.

Strategy:
  1. Parse the Mastodon export markdown into image posts.
  2. Recursively load every sidecar JSON under the output dir (skips */old/*).
  3. For each Mastodon post, score every sidecar on:
       - hashtag Jaccard similarity
       - alt-text vs description text similarity (difflib SequenceMatcher)
  4. Report the best match (and runner-up) with a confidence flag.

Usage:
  python match_mastodon_sidecars.py
  python match_mastodon_sidecars.py --mastodon /path/to/posts.md \
      --sidecars /path/to/output --report /path/to/report.md
"""

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

DEFAULT_MASTODON = "/home/junwin/Documents/mynotes/tempfiles/mastodon_posts.md"
DEFAULT_SIDECARS = "/home/junwin/pishare/photography/work/2026/output"
DEFAULT_REPORT = "/home/junwin/Documents/mynotes/tempfiles/mastodon_sidecar_matches.md"


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def hashtags_to_set(tags: str) -> set:
    """'#photo #bnw' -> {'photo','bnw'}."""
    if not tags:
        return set()
    return {t.lstrip("#").lower() for t in tags.split() if t.strip()}


def parse_mastodon(path: Path) -> list[dict]:
    """Return list of image posts: {num, date, title, hashtags(set), alt}."""
    text = path.read_text(encoding="utf-8")
    posts = []
    # Split on '## ' section headers.
    sections = re.split(r"(?m)^## ", text)
    for sec in sections:
        if not sec.strip():
            continue
        lines = sec.splitlines()
        header = lines[0].strip() if lines else ""
        m = re.match(r"(\d+)\.\s+(\S+)\s*—\s*(.*)", header)
        if not m:
            continue
        num, date, title = m.group(1), m.group(2), m.group(3).strip()
        hashtags = set()
        alt = ""
        for line in lines[1:]:
            if line.startswith("- Hashtags:"):
                hashtags = hashtags_to_set(line[len("- Hashtags:"):])
            elif line.startswith("- Alt:"):
                alt = normalize_text(line[len("- Alt:"):])
        if alt or hashtags:  # image post
            posts.append({
                "num": num, "date": date, "title": title,
                "hashtags": hashtags, "alt": alt,
            })
    return posts


def load_sidecars(root: Path) -> list[dict]:
    sidecars = []
    for p in root.rglob("*.json"):
        if "/old/" in str(p) or "\\old\\" in str(p):
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        tags = hashtags_to_set(data.get("hashtags", ""))
        # Fallback: first keyword string is often a comma-separated tag list.
        if not tags:
            kw = data.get("keywords", [])
            if kw and isinstance(kw[0], str):
                tags = {t.strip().lstrip("#").lower()
                        for t in re.split(r"[,\s]+", kw[0]) if t.strip()}
        texts = [
            normalize_text(data.get("image_description", "")),
            normalize_text(data.get("enhanced_description", "")),
            normalize_text(data.get("social_caption", "")),
        ]
        sidecars.append({
            "file": str(p.relative_to(root)),
            "hashtags": tags,
            "texts": [t for t in texts if t],
            "raw": data,
        })
    return sidecars


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def text_sim(alt: str, texts: list[str]) -> float:
    if not alt or not texts:
        return 0.0
    best = 0.0
    for t in texts:
        best = max(best, difflib.SequenceMatcher(None, alt, t).ratio())
    return best


def match_post(post: dict, sidecars: list[dict]) -> list[tuple[float, dict]]:
    scored = []
    for sc in sidecars:
        h = jaccard(post["hashtags"], sc["hashtags"])
        t = text_sim(post["alt"], sc["texts"])
        # hashtags are more distinctive; alt text confirms identity.
        combined = 0.6 * h + 0.4 * t
        scored.append((combined, sc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def confidence(score: float) -> str:
    if score >= 0.6:
        return "HIGH"
    if score >= 0.35:
        return "MEDIUM"
    return "LOW"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mastodon", default=DEFAULT_MASTODON)
    ap.add_argument("--sidecars", default=DEFAULT_SIDECARS)
    ap.add_argument("--report", default=DEFAULT_REPORT)
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()

    posts = parse_mastodon(Path(args.mastodon))
    sidecars = load_sidecars(Path(args.sidecars))
    print(f"Loaded {len(posts)} image posts, {len(sidecars)} sidecars.", file=sys.stderr)

    lines = ["# Mastodon ↔ Sidecar Matches", ""]
    for post in posts:
        ranked = match_post(post, sidecars)
        best_score, best = ranked[0]
        conf = confidence(best_score)
        print(f"\n[{post['num']}] {post['date']} — {post['title']}")
        print(f"    BEST ({conf}, score={best_score:.2f}): {best['file']}")
        for score, sc in ranked[1:args.top]:
            print(f"      alt {score:.2f}: {sc['file']}")

        lines.append(f"## {post['num']}. {post['date']} — {post['title']}")
        lines.append(f"- Confidence: **{conf}** (score {best_score:.2f})")
        lines.append(f"- Sidecar: `{best['file']}`")
        lines.append(f"- Mastodon tags: {' '.join(sorted(post['hashtags']))}")
        lines.append(f"- Sidecar tags: {' '.join(sorted(best['hashtags']))}")
        lines.append(f"- Alt: {post['alt']}")
        lines.append(f"- Sidecar desc: {best['texts'][0] if best['texts'] else ''}")
        lines.append("")

    Path(args.report).write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote report: {args.report}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
