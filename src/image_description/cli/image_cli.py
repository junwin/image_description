import argparse
import os

from ..image.describe import embed_metadata, process_directory, process_image
from ..image.prompts import PROMPT_PRESETS


def main() -> None:
    parser = argparse.ArgumentParser(description="Describe images and manage metadata using OpenAI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    describe_parser = subparsers.add_parser(
        "describe",
        help="Generate JSON sidecar files with descriptions and keywords.",
    )
    describe_parser.add_argument("path", help="Image file or directory to process.")
    describe_parser.add_argument(
        "--preset",
        choices=list(PROMPT_PRESETS.keys()),
        default="orwell_ways_of_seeing",
        help="Prompt preset to use.",
    )

    embed_parser = subparsers.add_parser(
        "embed",
        help="Embed metadata from JSON sidecars back into image IPTC.",
    )
    embed_parser.add_argument("directory", help="Directory containing images and JSON sidecars.")

    args = parser.parse_args()

    if args.command == "describe":
        if os.path.isdir(args.path):
            process_directory(args.path, preset=args.preset)
        else:
            process_image(args.path, preset=args.preset)
    elif args.command == "embed":
        embed_metadata(args.directory)


if __name__ == "__main__":
    main()
