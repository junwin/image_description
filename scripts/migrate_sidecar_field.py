#!/usr/bin/env python3
"""One-shot migration: rename 'visually_challenged_description' to 'image_description'
in all sidecar JSON files under the photography directory.

Usage:
    python scripts/migrate_sidecar_field.py [--dry-run]
"""

import argparse
import json
import os
import sys

PHOTOGRAPHY_ROOT = "/home/junwin/pishare/photography"


def migrate_file(path: str) -> bool:
    """Read, rename key, write. Returns True if changed."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "visually_challenged_description" not in data:
        return False

    # Move the value to the new key
    data["image_description"] = data.pop("visually_challenged_description")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print changes but don't write")
    args = parser.parse_args()

    changed = 0
    skipped = 0
    errors = 0

    for root, dirs, files in os.walk(PHOTOGRAPHY_ROOT):
        for name in files:
            if not name.endswith(".json"):
                continue
            if name.endswith(".social.json"):
                continue

            full = os.path.join(root, name)
            try:
                if args.dry_run:
                    with open(full, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if "visually_challenged_description" in data:
                        print(f"Would migrate: {full}")
                        changed += 1
                    else:
                        skipped += 1
                else:
                    if migrate_file(full):
                        print(f"Migrated: {full}")
                        changed += 1
                    else:
                        skipped += 1
            except Exception as e:
                print(f"Error: {full}: {e}", file=sys.stderr)
                errors += 1

    print(f"\nDone. Changed: {changed}, Skipped: {skipped}, Errors: {errors}")


if __name__ == "__main__":
    main()
