import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from ..sidecar import Sidecar


def _parse_hashtags_from_string(s: str) -> List[str]:
    if not s:
        return []
    # Split on whitespace or commas
    parts = [p.strip() for p in s.replace(",", " ").split()]
    parts = [p for p in parts if p]
    # Ensure tags start with '#'
    normalized = [p if p.startswith("#") else f"#{p}" for p in parts]
    return normalized


def _generate_social_for_sidecar(sidecar: Sidecar, platforms: List[str]) -> Dict[str, Any]:
    """Generate social post text and hashtags for requested platforms.

    Implementation note:
    - This implementation does not call an external model. Instead it
      constructs short platform texts from existing sidecar fields and derives
      hashtags from either sidecar.hashtags or sidecar.keywords. The goal is to
      keep the factual core separate from the generated/social derivative.
    """

    # Derive a base caption: prefer social_caption, fall back to enhanced_description,
    # then original_description.
    caption = sidecar.social_caption.strip() or sidecar.enhanced_description.strip() or sidecar.original_description.strip() or ""

    # Derive hashtags: if sidecar.hashtags provided (string), parse that.
    hashtags = _parse_hashtags_from_string(sidecar.hashtags)

    # If no explicit hashtags, fall back to keywords
    if not hashtags and sidecar.keywords:
        # choose up to 6 keywords
        chosen = sidecar.keywords[:6]
        hashtags = [f"#{kw.lstrip('#')}" for kw in chosen if kw]

    results: Dict[str, Any] = {}
    for platform in platforms:
        # Build a short platform-specific text. Keep conservative and short.
        if platform == "mastodon":
            text = caption or sidecar.title or sidecar.original_title or ""
        elif platform == "tumblr":
            # Tumblr prefers slightly longer captions
            text = (caption + "\n\n" + (sidecar.enhanced_description or "")).strip()
        elif platform == "bsky":
            # BlueSky prefers brief posts
            text = (caption[:240]).strip()
        else:
            text = caption

        results[platform] = {"text": text, "hashtags": list(hashtags)}

    return results


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate platform-specific social post derivatives (*.social.json) from "
            "an image sidecar JSON. The factual core remains in the original sidecar "
            "and generated content is written to a sibling .social.json file."
        )
    )
    parser.add_argument("json_path", nargs=1, help="Path to the metadata JSON sidecar file.")
    parser.add_argument(
        "--platforms",
        nargs="+",
        choices=["mastodon", "tumblr", "bsky"],
        default=["mastodon", "tumblr", "bsky"],
        help="One or more target platforms to generate content for (default: all).",
    )
    parser.add_argument(
        "--overwrite-sidecar",
        action="store_true",
        help="Overwrite existing .social.json derivative files. By default existing derivatives are preserved.",
    )

    args = parser.parse_args(argv)

    in_path = args.json_path[0]
    if not os.path.exists(in_path):
        sys.stderr.write(f"Error: sidecar JSON not found: {in_path}\n")
        raise SystemExit(2)

    social_path = Sidecar.social_path_for(in_path)
    if os.path.exists(social_path) and not args.overwrite_sidecar:
        # Mirror previous tools: print a warning and exit non-zero to signal no-op
        sys.stderr.write(f"Warning: social derivative already exists at {social_path}; use --overwrite-sidecar to replace\n")
        raise SystemExit(2)

    try:
        sidecar = Sidecar.load(in_path)
    except Exception as e:
        sys.stderr.write(f"Error loading sidecar: {e}\n")
        raise SystemExit(2)

    # Generate social content
    social_data = _generate_social_for_sidecar(sidecar, args.platforms)

    # Validation: hashtags must be present (non-empty list) for each requested platform
    missing_hashtags = [p for p, d in social_data.items() if not d.get("hashtags")]
    if missing_hashtags:
        sys.stderr.write(
            f"Warning: model returned no hashtags for platforms: {', '.join(missing_hashtags)}\n"
        )
        # Per requirements: print a warning and exit non-zero
        raise SystemExit(2)

    # Build derivative JSON with clear separation: 'core' factual sidecar and 'social' derivative
    derivative: Dict[str, Any] = {"core": sidecar.to_dict(), "social": social_data}

    try:
        os.makedirs(os.path.dirname(os.path.abspath(social_path)) or ".", exist_ok=True)
        with open(social_path, "w", encoding="utf-8") as f:
            json.dump(derivative, f, indent=2, ensure_ascii=False)
    except Exception as e:
        sys.stderr.write(f"Error writing social derivative: {e}\n")
        raise SystemExit(2)

    print(f"Wrote social derivative to {social_path}")


if __name__ == "__main__":
    main()
