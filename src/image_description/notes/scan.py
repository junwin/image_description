import json
import os
import sys
from typing import Dict, List, Optional

import yaml
from PIL import UnidentifiedImageError, Image

from ..paths import resolve_image_and_relative, iter_images
from .prompts import SCAN_PROMPT

DEFAULT_SCAN_MODEL = "gemini-3.6-flash"


def build_markdown(image_path: str, data: Dict) -> str:
    """Build Obsidian Markdown content from model data.

    YAML frontmatter with the exact fields: image_description, file_name,
    keywords, keywords_image, flags. Body is the transcription text.
    """
    front = {
        "image_description": data.get("image_description", ""),
        "file_name": os.path.basename(image_path),
        "keywords": data.get("keywords", []),
        "keywords_image": data.get("keywords_image", []),
        "flags": data.get("issues", []) or [],
    }

    # Use safe_dump with desired options
    yaml_text = yaml.safe_dump(front, default_flow_style=False, sort_keys=False, allow_unicode=True)

    body = data.get("text", "")

    return f"---\n{yaml_text}---\n\n{body}\n"


def _call_model(
    image_path: str,
    max_side: Optional[int] = None,
    model: str = DEFAULT_SCAN_MODEL,
    preprocess: bool = False,
    provider: Optional[str] = None,
    credential_path: Optional[str] = None,
) -> Dict:
    # Lazy import to avoid hard dependency during import-time (helps tests).
    from ..image.galet_client import create_vision_response, parse_json_response
    from ..image.openai_client import _encode_image_to_base64

    image_b64 = _encode_image_to_base64(image_path, max_side=max_side, preprocess=preprocess)

    text = create_vision_response(
        SCAN_PROMPT,
        image_b64,
        model=model,
        provider=provider,
        credential_path=credential_path,
        temperature=0.2,
    )

    data = parse_json_response(text)

    # Validate required keys
    required = ["image_description", "text", "keywords", "keywords_image", "issues"]
    for k in required:
        if k not in data:
            print(f"Model response missing key: {k}", file=sys.stderr)
            raise RuntimeError(f"Model response missing key: {k}")

    return data


def process_scan_image(
    image_path: str,
    overwrite: bool = False,
    max_side: Optional[int] = None,
    model: str = DEFAULT_SCAN_MODEL,
    preprocess: bool = False,
    provider: Optional[str] = None,
    credential_path: Optional[str] = None,
) -> bool:
    """Process a single image. Returns True if a markdown file was created.

    Skips non-image files and existing .md files when overwrite is False.
    """
    md_path = os.path.splitext(image_path)[0] + ".md"

    if not os.path.isfile(image_path):
        print(f"Skipping non-file: {image_path}")
        return False

    # Quick check: can PIL open it?
    try:
        Image.open(image_path)
    except UnidentifiedImageError:
        print(f"Skipping non-image file: {image_path}")
        return False
    except Exception as e:
        print(f"Error opening image {image_path}: {e}", file=sys.stderr)
        return False

    if os.path.exists(md_path) and not overwrite:
        print(f"Skipping existing markdown {md_path}")
        return False

    # Call model
    try:
        data = _call_model(
            image_path,
            max_side=max_side,
            model=model,
            preprocess=preprocess,
            provider=provider,
            credential_path=credential_path,
        )
    except Exception as e:
        print(f"Error calling model for {image_path}: {e}", file=sys.stderr)
        raise

    md = build_markdown(image_path, data)

    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)
    except Exception as e:
        print(f"Error writing markdown {md_path}: {e}", file=sys.stderr)
        raise

    print(f"Wrote {md_path}")
    return True


def process_scan_directory(
    directory: str,
    overwrite: bool = False,
    max_side: Optional[int] = None,
    model: str = DEFAULT_SCAN_MODEL,
    preprocess: bool = False,
    provider: Optional[str] = None,
    credential_path: Optional[str] = None,
) -> None:
    images = list(iter_images(directory))
    for img in images:
        try:
            process_scan_image(
                img,
                overwrite=overwrite,
                max_side=max_side,
                model=model,
                preprocess=preprocess,
                provider=provider,
                credential_path=credential_path,
            )
        except Exception as e:
            print(f"Error processing {img}: {e}", file=sys.stderr)
            # keep going
