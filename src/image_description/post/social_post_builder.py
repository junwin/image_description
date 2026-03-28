"""Social post builder helpers.

This module contains the social generation logic previously embedded in the CLI.
It is designed to be importable and testable independently of CLI parsing.
"""
from typing import Any, Dict, List


def parse_hashtags_from_string(s: str) -> List[str]:
    """Parse a comma/whitespace separated hashtag string into normalized tags.

    Returned tags will start with '#'. An empty input yields an empty list.
    """
    if not s:
        return []
    # Split on whitespace or commas
    parts = [p.strip() for p in s.replace(",", " ").split()]
    parts = [p for p in parts if p]
    # Ensure tags start with '#'
    normalized = [p if p.startswith("#") else f"#{p}" for p in parts]
    return normalized


def generate_social_for_sidecar(sidecar: Any, platforms: List[str]) -> Dict[str, Any]:
    """Generate social post text and hashtags for requested platforms.

    The implementation purposely does not call an external model. Instead it
    constructs short platform texts from existing sidecar fields and derives
    hashtags from either sidecar.hashtags or sidecar.keywords. The factual core
    remains separate from generated/social derivative.
    """

    # Derive a base caption: prefer social_caption, fall back to enhanced_description,
    # then original_description.
    caption = (
        (getattr(sidecar, "social_caption", "") or "").strip()
        or (getattr(sidecar, "enhanced_description", "") or "").strip()
        or (getattr(sidecar, "original_description", "") or "").strip()
        or ""
    )

    # Derive hashtags: if sidecar.hashtags provided (string), parse that.
    hashtags = parse_hashtags_from_string(getattr(sidecar, "hashtags", "") or "")

    # If no explicit hashtags, fall back to keywords
    if not hashtags and getattr(sidecar, "keywords", None):
        # choose up to 6 keywords
        chosen = getattr(sidecar, "keywords")[:6]
        hashtags = [f"#{kw.lstrip('#')}" for kw in chosen if kw]

    results: Dict[str, Any] = {}
    for platform in platforms:
        # Build a short platform-specific text. Keep conservative and short.
        if platform == "mastodon":
            text = (
                getattr(sidecar, "social_caption", "")
                or getattr(sidecar, "title", "")
                or getattr(sidecar, "original_title", "")
                or ""
            )
        elif platform == "tumblr":
            # Tumblr prefers slightly longer captions
            text = ((getattr(sidecar, "social_caption", "") or "") + "\n\n" + (getattr(sidecar, "enhanced_description", "") or "")).strip()
        elif platform == "bsky":
            # BlueSky prefers brief posts
            text = ((getattr(sidecar, "social_caption", "") or "")[:240]).strip()
        else:
            text = caption

        results[platform] = {"text": text, "hashtags": list(hashtags)}

    return results
