#!/usr/bin/env python3
"""Export Mastodon posts JSON to a readable markdown file with all fields."""
import json
import sys

def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "/home/junwin/Documents/mynotes/tempfiles/mastodon_posts.json"
    dst = sys.argv[2] if len(sys.argv) > 2 else "/home/junwin/Documents/mynotes/tempfiles/mastodon_posts_full.md"

    data = json.load(open(src))
    lines = ["# Mastodon — Last 20 Posts (post_id, url, date, hashtags, alt text)", ""]
    for i, p in enumerate(data, 1):
        d = p["date"][:10]
        lines.append("## %d. %s" % (i, d))
        lines.append("- **post_id**: %s" % p["post_id"])
        lines.append("- **url**: %s" % p["url"])
        lines.append("- **date**: %s" % p["date"])
        tags = " ".join("#" + t for t in p["hashtags"]) if p["hashtags"] else "(none)"
        lines.append("- **hashtags**: %s" % tags)
        alt = p["alt_text"] if p["alt_text"] else "(no image)"
        lines.append("- **alt_text**: %s" % alt)
        lines.append("")
    open(dst, "w").write("\n".join(lines))
    print("wrote", dst)

if __name__ == "__main__":
    main()
