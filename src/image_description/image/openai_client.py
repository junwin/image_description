import base64
import io
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from PIL import Image

from .prompts import build_prompt


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


def _encode_image_to_base64(path: str, max_side: Optional[int] = None) -> str:
    """Read an image file and return a base64-encoded JPEG string.

    If max_side is provided, the image is downscaled in-memory so that its
    longest side does not exceed max_side pixels. The original file is never
    modified.

    Args:
        path: Absolute path to the image file.
        max_side: Optional maximum dimension for the longest side, in pixels.

    Returns:
        Base64-encoded string of the (possibly resized) JPEG image.
    """
    img = Image.open(path)

    # Ensure we're in RGB mode (handles RGBA, P, etc.)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    if max_side is not None:
        w, h = img.size
        longest = max(w, h)
        if longest > max_side:
            ratio = max_side / longest
            new_size = (int(w * ratio), int(h * ratio))
            img = img.resize(new_size, Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _load_openai_client() -> OpenAI:
    """Create an OpenAI client.

    Precedence:
    1) OPENAI_API_KEY environment variable
    2) credential file (oaicred.json) if present

    Rationale:
    - Env vars are the most common OSS pattern and work well with CI, containers,
      and secret managers.
    - The credential file path is kept for backwards compatibility.
    """

    # Preferred: environment variable
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        return OpenAI(api_key=openai_api_key)

    # Back-compat: credential file
    _config = ConfigManager("config.json")
    credential_path = os.getenv(
        "CREDENTIAL_PATH", _config.get("credential_path", os.path.expanduser("~/credential"))
    )

    cred_file = os.path.join(credential_path, "oaicred.json")
    if os.path.exists(cred_file):
        with open(cred_file, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        openai_api_key = config_data.get("openai_api_key")

    if not openai_api_key:
        raise RuntimeError(
            "OpenAI API key not found. Set OPENAI_API_KEY or provide oaicred.json in CREDENTIAL_PATH."
        )

    return OpenAI(api_key=openai_api_key)


def generate_openai_description_and_keywords(
    image_path: str,
    title: str,
    description: str,
    existing_keywords: List[str],
    preset: str,
    max_side: Optional[int] = None,
) -> Tuple[str, str, str, List[str]]:
    """Call OpenAI vision model and return (img_desc, enhanced_desc, social_caption, keywords).

    Args:
        image_path: Absolute path to the image file.
        title: Existing IPTC title (or empty string).
        description: Existing IPTC description (or empty string).
        existing_keywords: Existing IPTC keywords.
        preset: Prompt preset name.
        max_side: If set, downscale the image in-memory before sending so its
                  longest side does not exceed this many pixels. The original
                  file is never modified.

    Returns:
        Tuple of (image_description, enhanced_description, social_caption, keywords).
    """

    client = _load_openai_client()

    prompt = build_prompt(preset, title, description, existing_keywords)
    image_b64 = _encode_image_to_base64(image_path, max_side=max_side)

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

    # Read image_description: prefer new key, fall back to legacy key from model response
    img_desc = str(data.get("image_description", "") or "").strip()
    if not img_desc:
        img_desc = str(data.get("visually_challenged_description", "") or "").strip()

    enhanced_desc = str(data.get("enhanced_description", "") or "").strip()
    social_caption = str(data.get("social_caption", "") or "").strip()

    new_keywords = data.get("keywords", [])
    if not isinstance(new_keywords, list):
        new_keywords = [str(new_keywords)]

    return (
        img_desc,
        enhanced_desc,
        social_caption,
        [str(k).strip() for k in new_keywords if str(k).strip()],
    )
