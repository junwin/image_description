import base64
import io
from typing import List, Optional, Tuple

from PIL import Image, ImageFilter, ImageOps

from .prompts import build_prompt


def _preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """Improve handwriting legibility before sending to the vision model.

    Steps: grayscale -> autocontrast -> median denoise -> unsharp mask.
    Returns an RGB image suitable for JPEG encoding.
    """
    if img.mode != "L":
        img = img.convert("L")
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    return img.convert("RGB")


def _encode_image_to_base64(path: str, max_side: Optional[int] = None, preprocess: bool = False) -> str:
    """Read an image file and return a base64-encoded JPEG string.

    If max_side is provided, the image is downscaled in-memory so that its
    longest side does not exceed max_side pixels. The original file is never
    modified.

    If preprocess is True, the image is enhanced for OCR (grayscale,
    contrast stretch, denoise, sharpen) before encoding.

    Args:
        path: Absolute path to the image file.
        max_side: Optional maximum dimension for the longest side, in pixels.
        preprocess: If True, apply OCR-oriented preprocessing.

    Returns:
        Base64-encoded string of the (possibly resized/preprocessed) JPEG image.
    """
    img = Image.open(path)

    # Ensure we're in RGB mode (handles RGBA, P, etc.)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    if preprocess:
        img = _preprocess_for_ocr(img)

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


def generate_openai_description_and_keywords(
    image_path: str,
    title: str,
    description: str,
    existing_keywords: List[str],
    preset: str,
    max_side: Optional[int] = None,
    model: str = "gpt-4o-mini",
    provider: Optional[str] = None,
    credential_path: Optional[str] = None,
) -> Tuple[str, str, str, List[str]]:
    """Call a vision model via galet and return (img_desc, enhanced_desc, social_caption, keywords).

    Args:
        image_path: Absolute path to the image file.
        title: Existing IPTC title (or empty string).
        description: Existing IPTC description (or empty string).
        existing_keywords: Existing IPTC keywords.
        preset: Prompt preset name.
        max_side: If set, downscale the image in-memory before sending so its
                  longest side does not exceed this many pixels. The original
                  file is never modified.
        model: Model name (default gpt-4o-mini).
        provider: Explicit galet provider (openai/gemini/...); None -> routing.
        credential_path: Directory with galet credential files; None -> galet defaults.

    Returns:
        Tuple of (image_description, enhanced_description, social_caption, keywords).
    """
    from .galet_client import create_vision_response, parse_json_response

    prompt = build_prompt(preset, title, description, existing_keywords)
    image_b64 = _encode_image_to_base64(image_path, max_side=max_side)

    text = create_vision_response(
        prompt,
        image_b64,
        model=model,
        provider=provider,
        credential_path=credential_path,
        temperature=0.4,
    )

    data = parse_json_response(text)

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
