import os
import argparse
import json
import base64
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from openai import OpenAI


# -----------------------------------------------------------------------------
# Config + credential loading (aligned with lucy project)
# -----------------------------------------------------------------------------


class ConfigManager:
    def __init__(self, config_path: str):
        self._config_path = config_path
        self._config: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self._config_path):
            with open(self._config_path, "r", encoding="utf-8") as f:
                self._config = json.load(f)
        else:
            self._config = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)


_config = ConfigManager("config.json")
credential_path = os.getenv(
    "CREDENTIAL_PATH", _config.get("credential_path", os.path.expanduser("~/credential"))
)

with open(os.path.join(credential_path, "oaicred.json"), "r", encoding="utf-8") as f:
    config_data = json.load(f)

openai_api_key = os.getenv("OPENAI_API_KEY", config_data.get("openai_api_key"))
client = OpenAI(api_key=openai_api_key)


# -----------------------------------------------------------------------------
# Constants and prompt presets
# -----------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
MAX_FILE_SIZE_MB = 20

PROMPT_PRESETS: Dict[str, str] = {
    "orwell_basic": (
        "You are a careful, precise writer. "
        "Describe the image in clear, concrete language. "
        "Avoid jargon and unnecessary words. "
        "Return JSON with the following keys: "
        "'visually_challenged_description' (one paragraph), "
        "'enhanced_description' (one paragraph), and "
        "'keywords' (a list of 8-15 short keywords)."
    ),
    "orwell_ways_of_seeing": (
        "Act as a thoughtful artist and writer. "
        "Consider John Berger's separation of (a) what the image is and (b) what it is trying to say. "
        "Lean toward what the image is trying to say, but stay grounded in what is visible. "
        "Please adhere strictly to the following style guidelines: "
        "1. Follow George Orwell's rules: use short words, cut unnecessary words, and avoid jargon. "
        "2. Use a minimalist and evocative style. Be precise, not flowery. "
        "3. Adopt a reflective, understated tone. Avoid any boastfulness. "
        "4. Use a two-sentence structure if possible: first a direct description, then a reflective observation. "
        "Keep the final output concise. "
        "Return JSON with the following keys: "
        "'visually_challenged_description' (one paragraph), "
        "'enhanced_description' (one paragraph), and "
        "'keywords' (a list of 8-15 short keywords)."
    ),
}


def build_prompt(
    preset_name: str,
    title: str,
    description: str,
    existing_keywords: List[str],
) -> str:
    base_prompt = PROMPT_PRESETS.get(preset_name, PROMPT_PRESETS["orwell_ways_of_seeing"])
    kw_str = ", ".join(existing_keywords) if existing_keywords else "(none)"

    return (
        f"{base_prompt}\n\n"
        f"Here is some metadata I already have – this typically deals with what the image is.\n\n"
        f"Title: {title}\n"
        f"Existing description: {description}\n"
        f"Existing keywords: {kw_str}\n\n"
        f"Please respond ONLY with a single JSON object matching the requested keys."
    )


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def is_image_file(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext in SUPPORTED_EXTENSIONS


def file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def encode_image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


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

    code, out, err = run_exiftool([
        "-IPTC:ObjectName",
        "-IPTC:Caption-Abstract",
        "-IPTC:Keywords",
        file_path,
    ])
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

    code, out, err = run_exiftool(args)
    if code != 0:
        print(f"exiftool error writing IPTC to {file_path}: {err}")
    else:
        print(f"Updated IPTC metadata for {file_path}")


# -----------------------------------------------------------------------------
# OpenAI interaction
# -----------------------------------------------------------------------------


def generate_openai_description_and_keywords(
    image_path: str,
    title: str,
    description: str,
    existing_keywords: List[str],
    preset: str = "orwell_ways_of_seeing",
) -> Tuple[str, str, List[str]]:
    """Call OpenAI vision model and return (vc_desc, enhanced_desc, keywords)."""

    prompt = build_prompt(preset, title, description, existing_keywords)
    image_b64 = encode_image_to_base64(image_path)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                        },
                    },
                ],
            }
        ],
        temperature=0.4,
    )

    content = response.choices[0].message.content or ""

    # Try to extract JSON from the response
    text = content.strip()
    if text.startswith("```"):
        # Strip code fences if present
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: try to find a JSON object in the text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                raise RuntimeError(f"Model response was not valid JSON: {text}")
        else:
            raise RuntimeError(f"Model response was not valid JSON: {text}")

    vc_desc = data.get("visually_challenged_description", "").strip()
    enhanced_desc = data.get("enhanced_description", "").strip()
    new_keywords = data.get("keywords", [])
    if not isinstance(new_keywords, list):
        new_keywords = [str(new_keywords)]

    return vc_desc, enhanced_desc, [str(k).strip() for k in new_keywords if str(k).strip()]


# -----------------------------------------------------------------------------
# Core processing
# -----------------------------------------------------------------------------


def merge_keywords(existing: List[str], new: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for kw in existing + new:
        norm = kw.strip()
        if not norm:
            continue
        lower = norm.lower()
        if lower not in seen:
            seen.add(lower)
            merged.append(norm)
    return merged


def build_hashtags(keywords: List[str]) -> str:
    tags = ["#" + kw.replace(" ", "").lower() for kw in keywords]
    return " ".join(tags)


def process_image(image_path: str, preset: str = "orwell_ways_of_seeing") -> None:
    if not is_image_file(image_path):
        print(f"Skipping non-image file: {image_path}")
        return

    size_mb = file_size_mb(image_path)
    if size_mb > MAX_FILE_SIZE_MB:
        print(f"Skipping {image_path}: file size {size_mb:.2f} MB exceeds limit of {MAX_FILE_SIZE_MB} MB")
        return

    print(f"Processing {image_path} ({size_mb:.2f} MB)")

    existing_title, existing_description, existing_keywords = show_image_iptc_meta(image_path)

    # Call OpenAI to get descriptions and new keywords
    vc_desc, enhanced_desc, new_keywords = generate_openai_description_and_keywords(
        image_path,
        existing_title,
        existing_description,
        existing_keywords,
        preset=preset,
    )

    merged_keywords = merge_keywords(existing_keywords, new_keywords)
    hashtags = build_hashtags(merged_keywords)

    # JSON sidecar path
    base, _ = os.path.splitext(image_path)
    json_file_path = base + ".json"

    metadata: Dict[str, Any] = {
        # Original metadata from the image (what you wrote in your editor)
        "original_title": existing_title,
        "original_description": existing_description,

        # Current working title (you can later change this to a shorter one if you like)
        "title": existing_title,

        # AI-generated fields
        "visually_challenged_description": vc_desc,
        "enhanced_description": enhanced_desc,
        "keywords": merged_keywords,
        "hashtags": hashtags,
    }

    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"Wrote metadata to {json_file_path}")


def process_directory(directory: str, preset: str = "orwell_ways_of_seeing") -> None:
    for root, _, files in os.walk(directory):
        for name in files:
            path = os.path.join(root, name)
            if is_image_file(path):
                try:
                    process_image(path, preset=preset)
                except Exception as e:  # noqa: BLE001
                    print(f"Error processing {path}: {e}")


def embed_metadata(directory: str) -> None:
    """Read JSON sidecars and write selected fields back into image IPTC."""
    for root, _, files in os.walk(directory):
        for name in files:
            path = os.path.join(root, name)
            if not is_image_file(path):
                continue

            base, _ = os.path.splitext(path)
            json_file_path = base + ".json"
            if not os.path.exists(json_file_path):
                continue

            try:
                with open(json_file_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception as e:  # noqa: BLE001
                print(f"Error reading {json_file_path}: {e}")
                continue

            title = meta.get("title") or meta.get("original_title")
            description = meta.get("enhanced_description") or meta.get("original_description")
            keywords = meta.get("keywords") or []
            if not isinstance(keywords, list):
                keywords = [str(keywords)]

            write_iptc_meta(path, title=title, description=description, keywords=keywords)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Describe images and manage metadata using OpenAI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # describe command
    describe_parser = subparsers.add_parser(
        "describe",
        help="Generate JSON sidecar files with descriptions and keywords.",
    )
    describe_parser.add_argument("path", help="Image file or directory to process.")
    describe_parser.add_argument(
        "--preset",
        choices=list(PROMPT_PRESETS.keys()),
        default="orwell_ways_of_seeing",
        help="Prompt preset to use.",
    )

    # embed command
    embed_parser = subparsers.add_parser(
        "embed",
        help="Embed metadata from JSON sidecars back into image IPTC.",
    )
    embed_parser.add_argument("directory", help="Directory containing images and JSON sidecars.")

    args = parser.parse_args()

    if args.command == "describe":
        if os.path.isdir(args.path):
            process_directory(args.path, preset=args.preset)
        else:
            process_image(args.path, preset=args.preset)
    elif args.command == "embed":
        embed_metadata(args.directory)


if __name__ == "__main__":
    main()
