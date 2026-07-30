#!/usr/bin/env python3
"""Delete WordPress posts by ID."""
import json, urllib.request, urllib.error, sys

c = json.load(open("/home/junwin/credential/wordpress_com.json"))
site = c["site"]
token = c["access_token"]
base = f"https://public-api.wordpress.com/rest/v1.1/sites/{site}/posts"

# Accept post IDs from command-line args
pids = sys.argv[1:]
if not pids:
    print("Usage: delete_wp_posts.py <PID1> [PID2 ...]")
    sys.exit(1)

for pid in pids:
    url = f"{base}/{pid}/delete"
    req = urllib.request.Request(
        url, method="POST", headers={"Authorization": f"Bearer {token}"}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print(f"Deleted post {pid}: ID={data.get('ID', '?')} status={data.get('status', '?')}")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"Failed to delete {pid}: HTTP {e.code}")
        print(body[:300])
        sys.exit(1)
