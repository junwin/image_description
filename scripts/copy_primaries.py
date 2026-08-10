import os
import json
import shutil
from pathlib import Path
from datetime import datetime


def ensure_folder(path):
    Path(path).mkdir(parents=True, exist_ok=True)


from datetime import datetime

def format_date_folder(date_str: str) -> str:
    if not date_str:
        return "UnknownDate"

    # Try known patterns first
    formats = [
        "%Y:%m:%d %H:%M:%S",        # EXIF style (2021:01:02 13:28:15)
        "%Y-%m-%d %H:%M:%S",        # ISO without T
        "%Y-%m-%dT%H:%M:%S",        # ISO with T
        "%Y-%m-%dT%H:%M:%S.%f",     # ISO with microseconds
        "%Y:%m:%d",                 # EXIF date only
        "%Y-%m-%d"                  # ISO date only
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%Y%b")  # e.g., 2021Dec
        except ValueError:
            continue

    # Final fallback: try fromisoformat (handles many variants, including microseconds)
    try:
        dt = datetime.fromisoformat(date_str)
        return dt.strftime("%Y%b")
    except Exception:
        return "UnknownDate"


def copy_primaries(json_path, target_base):
    with open(json_path, "r", encoding="utf-8") as f:
        file_data = json.load(f)

    total = 0
    for record in file_data.values():
        primary_path = record.get("primary_path")
        date_str = record.get("date_taken") or record.get("created") or record.get("modified")
        folder_name = format_date_folder(date_str)

        destination_folder = Path(target_base) / folder_name
        ensure_folder(destination_folder)

        dest_file = destination_folder / Path(primary_path).name

        try:
            shutil.copy2(primary_path, dest_file)
            print(f"✔ Copied: {primary_path} → {dest_file}")
            total += 1
        except Exception as e:
            print(f"⚠️ Failed to copy {primary_path}: {e}")

    print(f"\n✅ Finished copying {total} files.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Copy primary files to target folders by date.")
    parser.add_argument("json", help="Path to JSON file from scan_duplicates.py")
    parser.add_argument("target", help="Target base folder to copy into (e.g. E:\\photography\\rawimages\\leica)")
    args = parser.parse_args()

    copy_primaries(args.json, args.target)
