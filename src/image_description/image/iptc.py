import subprocess
from typing import List, Optional, Tuple


def run_exiftool(args: List[str]) -> Tuple[int, str, str]:
    """Run exiftool with given args, return (returncode, stdout, stderr)."""
    cmd = ["exiftool"] + args
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    return proc.returncode, out, err


def show_image_iptc_meta(file_path: str) -> Tuple[str, str, List[str]]:
    """Return (title, description, keywords) from IPTC using exiftool."""
    title = ""
    description = ""
    keywords: List[str] = []

    code, out, err = run_exiftool(
        [
            "-IPTC:ObjectName",
            "-IPTC:Caption-Abstract",
            "-IPTC:Keywords",
            file_path,
        ]
    )
    if code != 0:
        print(f"exiftool error reading IPTC from {file_path}: {err}")
        return title, description, keywords

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

    return title, description, keywords


def write_iptc_meta(
    file_path: str,
    title: Optional[str] = None,
    description: Optional[str] = None,
    keywords: Optional[List[str]] = None,
) -> None:
    args: List[str] = []
    if title is not None:
        args.append(f"-IPTC:ObjectName={title}")
    if description is not None:
        args.append(f"-IPTC:Caption-Abstract={description}")
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
