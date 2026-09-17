import subprocess
from typing import List, Optional, Tuple


def run_exiftool(args: List[str]) -> Tuple[int, str, str]:
    """Run exiftool with given args, return (returncode, stdout, stderr)."""
    cmd = ["exiftool"] + args
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    return proc.returncode, out, err


def get_exif_date(file_path: str) -> str:
    """Extract EXIF capture date from an image file.

    Tries DateTimeOriginal first, falls back to CreateDate.
    Returns empty string if neither is found or on any error.
    """
    code, out, err = run_exiftool(
        [
            "-EXIF:DateTimeOriginal",
            "-EXIF:CreateDate",
            file_path,
        ]
    )
    if code != 0:
        print(f"exiftool error reading EXIF dates from {file_path}: {err}")
        return ""

    date_original = ""
    create_date = ""

    for line in out.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if "Date/Time Original" in key or "DateTimeOriginal" in key:
            date_original = value
        elif "Create Date" in key or "CreateDate" in key:
            create_date = value

    return date_original or create_date or ""


def show_image_iptc_meta(file_path: str) -> Tuple[str, str, List[str], str]:
    """Return (title, description, keywords, alt_text) from IPTC/XMP using exiftool."""
    title = ""
    description = ""
    keywords: List[str] = []
    alt_text = ""

    code, out, err = run_exiftool(
        [
            "-IPTC:ObjectName",
            "-IPTC:Caption-Abstract",
            "-IPTC:Keywords",
            "-XMP:AltTextAccessibility",
            file_path,
        ]
    )
    if code != 0:
        print(f"exiftool error reading IPTC from {file_path}: {err}")
        return title, description, keywords, alt_text

    for line in out.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key.endswith("Object Name") or key.endswith("ObjectName"):
            title = value
        elif key.endswith("Caption-Abstract"):
            description = value
        elif key.endswith("Keywords"):
            # exiftool may output multiple lines for multiple keywords
            keywords.append(value)
        elif key.endswith("Alt Text Accessibility") or key.endswith("AltTextAccessibility"):
            alt_text = value

    return title, description, keywords, alt_text


def write_iptc_meta(
    file_path: str,
    title: Optional[str] = None,
    description: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    alt_text: Optional[str] = None,
) -> None:
    args: List[str] = []
    if title is not None:
        args.append(f"-IPTC:ObjectName={title}")
    if description is not None:
        args.append(f"-IPTC:Caption-Abstract={description}")
    if alt_text is not None:
        args.append(f"-XMP:AltTextAccessibility={alt_text}")
    if keywords is not None:
        # Clear existing keywords then add new ones
        args.append("-IPTC:Keywords=")
        for kw in keywords:
            args.append(f"-IPTC:Keywords+={kw}")

    args.append(file_path)

    code, _out, err = run_exiftool(args)
    if code != 0:
        print(f"exiftool error writing IPTC to {file_path}: {err}")
    else:
        print(f"Updated IPTC metadata for {file_path}")
