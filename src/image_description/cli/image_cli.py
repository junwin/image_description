import argparse
import os

from ..image.describe import embed_metadata, process_directory, process_image
from ..image.prompts import PROMPT_PRESETS
from ..paths import iter_images, sidecar_path_for_image


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
    describe_parser.add_argument(
        "--overwrite-sidecar",
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help=(
            "If set, existing JSON sidecar files will be overwritten. "
            "By default existing sidecars are skipped to avoid unnecessary OpenAI calls."
        ),
    )

    embed_parser = subparsers.add_parser(
        "embed",
        help="Embed metadata from JSON sidecars back into image IPTC.",
    )
    embed_parser.add_argument("directory", help="Directory containing images and JSON sidecars.")

    args = parser.parse_args()

    if args.command == "describe":
        if os.path.isdir(args.path):
            if args.overwrite:
                # Preserve existing behavior when overwriting is requested
                process_directory(args.path, preset=args.preset)
            else:
                # Iterate images and skip any that already have sidecars to avoid
                # unnecessary OpenAI calls.
                for path in iter_images(args.path):
                    try:
                        json_file_path = sidecar_path_for_image(path)
                        if os.path.exists(json_file_path):
                            print(f"Skipping {path}: sidecar {json_file_path} already exists")
                            continue
                        process_image(path, preset=args.preset)
                    except Exception as e:  # noqa: BLE001
                        print(f"Error processing {path}: {e}")
        else:
            # Single file: check for existing sidecar early and skip if not overwriting
            json_file_path = sidecar_path_for_image(args.path)
            if os.path.exists(json_file_path) and not args.overwrite:
                print(f"Skipping {args.path}: sidecar {json_file_path} already exists")
            else:
                process_image(args.path, preset=args.preset)
    elif args.command == "embed":
        embed_metadata(args.directory)


if __name__ == "__main__":
    main()
