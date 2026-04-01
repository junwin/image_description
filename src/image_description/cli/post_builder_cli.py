"""CLI wrapper for post_builder.build_from_json.

Added --prompt-only which outputs only the social-post prompt text (no
headings or surrounding code fences). This CLI intentionally avoids
changing the default behaviour.

Manual test:
- python -m src.image_description.cli.post_builder_cli path/to/sidecar.json --prompt-only
  should print only the prompt text.

Note: changes limited to this file per scope constraints; no unit tests were
added here. If you would like automated tests added, expand the scope.
"""

import argparse

from ..post.post_builder import build_from_json, build_prompt
from ..sidecar import Sidecar


def main() -> None:
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

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--no-prompt",
        action="store_true",
        help="Do not include the social-post prompt in the output.",
    )
    group.add_argument(
        "--prompt-only",
        action="store_true",
        help="Output only the social-post prompt text (no headings or other sections).",
    )

    args = parser.parse_args()

    # If prompt-only is requested simply load the sidecar and print the prompt
    if args.prompt_only:
        # Load sidecar to access metadata
        sidecar = Sidecar.load(args.json_path)
        meta = sidecar.to_dict()
        prompt_text = build_prompt(meta)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(prompt_text)
        else:
            print(prompt_text)
        return

    output = build_from_json(
        args.json_path,
        fmt=args.format,
        image_path=args.image_path,
        include_prompt=not args.no_prompt,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
