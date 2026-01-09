#!/usr/bin/env python3
"""Wrapper script to call the package CLI implementation.
This preserves the behavior of the original top-level post_builder.py script by
delegating to image_description.post.post_builder.main().
"""

import os
import sys

# Make package importable when running the script from the repository tree
# (adds ../src to sys.path if the package isn't already importable).
try:
    from image_description.post import post_builder
except Exception:
    here = os.path.dirname(__file__)
    candidate = os.path.normpath(os.path.join(here, "..", "src"))
    if candidate not in sys.path:
        sys.path.insert(0, candidate)
    from image_description.post import post_builder


def main() -> None:
    # If the package module has main(), call it. Otherwise, fall back to
    # build_from_json-based invocation to preserve behavior.
    if hasattr(post_builder, "main") and callable(post_builder.main):
        return post_builder.main()

    # Fallback (shouldn't be needed because we added main to the package):
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Build Markdown or YAML from image metadata JSON.",
    )
    parser.add_argument("json_path", help="Path to the metadata JSON file.")
    parser.add_argument(
        "--format",
        choices=["md", "markdown", "yaml", "yml"],
        default="md",
        help="Output format (default: md).",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--image-path",
        help=(
            "Explicit image path to include. If omitted, tries to guess from "
            "JSON filename."
        ),
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Do not include the social-post prompt in the output.",
    )

    args = parser.parse_args()

    output = post_builder.build_from_json(
        args.json_path, args.format, args.image_path, include_prompt=not args.no_prompt
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
    else:
        sys.stdout.write(output)


if __name__ == "__main__":
    main()
