#!/usr/bin/env python3
"""
CLI wrapper for post_builder.build_from_json.

When called from an agent, the command will be in the form:
    bash -lc "source .venv/bin/activate && python -m src.image_description.cli.post_builder_cli <args>"

Arguments:
    json_file: Path to JSON sidecar file (absolute path)
    --format/-f: Output format (md/markdown/yaml/yml)
    --output/-o: Output file path (optional, defaults to stdout)
    --no-prompt: Do not include prompt text in output
    --prompt-only: Output only the prompt text (mutually exclusive with --no-prompt)
"""

import argparse
import json
import sys
from pathlib import Path

from ..post.post_builder import build_from_json


def main():
    parser = argparse.ArgumentParser(
        description="Generate social media posts from JSON sidecar files."
    )
    parser.add_argument(
        "json_file",
        help="Path to JSON sidecar file (absolute path)",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["md", "markdown", "yaml", "yml"],
        default="md",
        help="Output format (default: md)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file path (optional, defaults to stdout)",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Do not include prompt text in output",
    )
    parser.add_argument(
        "--prompt-only",
        action="store_true",
        help="Output only the prompt text (mutually exclusive with --no-prompt)",
    )

    args = parser.parse_args()

    # Validate mutually exclusive options
    if args.no_prompt and args.prompt_only:
        print(
            "Error: --no-prompt and --prompt-only are mutually exclusive",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load JSON
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {json_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # Handle prompt-only mode
    if args.prompt_only:
        prompt = data.get("prompt", "")
        if not prompt:
            print(f"Warning: No prompt found in {json_path}", file=sys.stderr)
        output_text = prompt
    else:
        # Build post
        try:
            output_text = build_from_json(
                str(json_path),  # Pass file path, not loaded data
                fmt=args.format,  # Use correct parameter name "fmt"
                include_prompt=not args.no_prompt,
            )
        except Exception as e:
            print(f"Error building post: {e}", file=sys.stderr)
            sys.exit(1)

    # Write output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_text)
        print(f"Output written to {output_path}")
    else:
        print(output_text)


if __name__ == "__main__":
    main()