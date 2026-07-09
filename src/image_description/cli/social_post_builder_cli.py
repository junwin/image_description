#!/usr/bin/env python3
"""
CLI for generating platform-specific social post derivatives from image sidecar JSON.

When called from an agent, the command will be in the form:
    bash -lc "source .venv/bin/activate && python -m src.image_description.cli.social_post_builder_cli <args>"

Arguments:
    json_path: Path to the metadata JSON sidecar file or a directory when used with --image-root (absolute path)
    --platforms: One or more target platforms to generate content for (default: all)
    --overwrite-sidecar: Overwrite existing .social.json derivative files
    --image-root: When set, the provided json_path must be relative to image_root

The factual core remains in the original sidecar and generated content is written
to a sibling .social.json file. If the model returns an empty list ([]) for hashtags/tags
for any requested platform, the CLI prints a warning to stderr and exits non-zero.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from ..sidecar import Sidecar, resolve_image_against_root, list_files_non_recursive


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


def _process_single_sidecar(sidecar_path: str, platforms: List[str], overwrite: bool, image_root: Optional[str]) -> Optional[str]:
    """
    Process a single sidecar JSON path. Returns None on success or an error message on failure.
    """
    if not os.path.exists(sidecar_path):
        return f"sidecar JSON not found: {sidecar_path}"

    social_path = Sidecar.social_path_for(sidecar_path)
    if os.path.exists(social_path) and not overwrite:
        return f"social derivative already exists at {social_path}; use --overwrite-sidecar to replace"

    try:
        sidecar = Sidecar.load(sidecar_path)
    except Exception as e:
        return f"Error loading sidecar: {e}"

    # When image_root is provided, ensure we can compute image_relative_path for the core.
    if image_root is not None:
        if not sidecar.image_filename:
            return f"Error: --image-root was provided but sidecar.image_filename is empty for {sidecar_path}"
        try:
            # resolve_image_against_root will validate and return (abs, rel)
            _, rel = resolve_image_against_root(image_root, sidecar.image_filename)
            # set in-memory; do not overwrite on disk
            sidecar.image_relative_path = rel
        except SystemExit:
            return f"Error: sidecar.image_filename resolves outside image_root for {sidecar_path}"

    # Generate social content
    social_data = _generate_social_for_sidecar(sidecar, platforms)

    # Validation: hashtags must be present (non-empty list) for each requested platform
    missing_hashtags = [p for p, d in social_data.items() if not d.get("hashtags")]
    if missing_hashtags:
        return f"model returned no hashtags for platforms: {', '.join(missing_hashtags)} (sidecar: {sidecar_path})"

    # Build derivative JSON with clear separation: 'core' factual sidecar and 'social' derivative
    derivative: Dict[str, Any] = {"core": sidecar.to_dict(), "social": social_data}

    try:
        os.makedirs(os.path.dirname(os.path.abspath(social_path)) or ".", exist_ok=True)
        with open(social_path, "w", encoding="utf-8") as f:
            json.dump(derivative, f, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"Error writing social derivative: {e}"

    print(f"Wrote social derivative to {social_path}")
    return None


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate platform-specific social post derivatives (*.social.json) from "
            "an image sidecar JSON. The factual core remains in the original sidecar "
            "and generated content is written to a sibling .social.json file."
        )
    )
    parser.add_argument("json_path", nargs=1, help="Path to the metadata JSON sidecar file or a directory when used with --image-root (absolute path).")
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
    parser.add_argument(
        "--image-root",
        dest="image_root",
        help=(
            "When set, the provided json_path must be relative to image_root. "
            "If json_path resolves to a directory, process files in that directory non-recursively."
        ),
    )

    args = parser.parse_args(argv)

    in_path = args.json_path[0]
    image_root = args.image_root

    sidecar_paths: List[str] = []

    if image_root:
        # Resolve provided path against image_root. This enforces the relative-path rule
        try:
            resolved_abs, rel = resolve_image_against_root(image_root, in_path)
        except SystemExit:
            # resolve_image_against_root writes its own stderr message before exiting
            raise

        if os.path.isdir(resolved_abs):
            # List files non-recursively under the provided directory
            entries = list_files_non_recursive(os.path.realpath(image_root), rel)
            # We are only interested in sidecar JSON files (exclude .social.json)
            for e in entries:
                if e.endswith(".json") and not e.endswith(".social.json"):
                    sidecar_paths.append(os.path.join(os.path.realpath(image_root), e))
        else:
            sidecar_paths.append(resolved_abs)
    else:
        # No image_root: treat in_path as a direct filesystem path
        if os.path.isdir(in_path):
            # Non-recursive: list files in the directory and pick .json files (exclude .social.json)
            for name in os.listdir(in_path):
                full = os.path.join(in_path, name)
                if os.path.isfile(full) and name.endswith(".json") and not name.endswith(".social.json"):
                    sidecar_paths.append(full)
        else:
            sidecar_paths.append(in_path)

    failures: List[str] = []
    for sp in sidecar_paths:
        err = _process_single_sidecar(sp, args.platforms, args.overwrite_sidecar, image_root)
        if err:
            sys.stderr.write(f"Warning: {err}\n")
            failures.append(err)

    if failures:
        sys.stderr.write(f"Completed with {len(failures)} failure(s).\n")
        raise SystemExit(2)

    # Success
    return


if __name__ == "__main__":
    main()