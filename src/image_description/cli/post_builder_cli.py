import argparse

from ..post.post_builder import build_from_json


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
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Do not include the social-post prompt in the output.",
    )

    args = parser.parse_args()

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
