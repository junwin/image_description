"""Shared utilities for social media publishing CLIs.

All four CLIs (mastodon, pixelfed, bluesky, tumblr) use these functions to
build post text from sidecar data. This ensures consistent behavior across
platforms.

- process_hashtags: remove #places/#genre, prepend #photography/#photo
- build_social_text: build status text. By default the post body comes from
  the photographer's own words (original_description), falling back to the
  AI image_description when they are missing. An explicit caption_field
  (e.g. "social_caption") can be passed to use that sidecar field instead.
- build_alt_text: always uses image_description
"""

from typing import Any, Dict, List, Optional


_REMOVE_TAGS = {"#places", "#genre"}
_PREPEND_TAGS = ["#photography", "#photo"]


def process_hashtags(raw_hashtags: str) -> str:
    """Process hashtags from a sidecar.

    1. Remove #places and #genre.
    2. Prepend #photography and #photo (deduped).

    Returns a space-separated string of hashtags.
    """
    tags = [t.strip() for t in raw_hashtags.split() if t.strip()]
    tags = [t for t in tags if t not in _REMOVE_TAGS]

    result: List[str] = []
    for prepend_tag in _PREPEND_TAGS:
        if prepend_tag in tags:
            tags.remove(prepend_tag)
        result.append(prepend_tag)
    result.extend(tags)

    return " ".join(result)


def build_social_text(
    sidecar_data: Dict[str, Any],
    caption_field: Optional[str] = None,
    extra_text: Optional[str] = None,
    char_limit: Optional[int] = None,
) -> str:
    """Build social media post text from sidecar data.

    Title: original_title (preferred) or title.
    Caption: by default original_description (the photographer's own words),
        falling back to image_description when missing. If caption_field is
        given (e.g. "social_caption"), that sidecar field is used instead.
    Hashtags: processed via process_hashtags().

    Returns the assembled post text, truncated to char_limit if set.
    """
    parts: List[str] = []

    # Title
    title = (sidecar_data.get("original_title") or sidecar_data.get("title") or "").strip()
    if title:
        parts.append(title)

    # Caption: own words by default; explicit field overrides.
    if caption_field:
        caption = (sidecar_data.get(caption_field) or "").strip()
    else:
        caption = (
            sidecar_data.get("original_description")
            or sidecar_data.get("image_description")
            or ""
        ).strip()
    if caption:
        parts.append(caption)

    # Extra user text
    if extra_text:
        parts.append(extra_text.strip())

    # Hashtags
    hashtags = (sidecar_data.get("hashtags") or "").strip()
    if hashtags:
        parts.append(process_hashtags(hashtags))

    text = "\n\n".join(parts)

    if char_limit and len(text) > char_limit:
        import sys
        sys.stderr.write(
            f"Warning: status text is {len(text)} chars "
            f"(limit: {char_limit}). It will be truncated.\n"
        )
        text = text[:char_limit - 3] + "..."

    return text


def build_alt_text(sidecar_data: Dict[str, Any]) -> str:
    """Build ALT text. Always uses image_description. Falls back to title."""
    desc = (sidecar_data.get("image_description") or "").strip()
    if not desc:
        desc = (sidecar_data.get("title") or "").strip()
    return desc
