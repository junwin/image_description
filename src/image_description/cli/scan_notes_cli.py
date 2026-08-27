import argparse
import os
import sys

from ..paths import resolve_image_and_relative, iter_images
from ..notes.scan import process_scan_directory, process_scan_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan photographed notes and produce Obsidian markdown files.")
    parser.add_argument("path", help="Image file or directory to process.")
    parser.add_argument(
        "--image-root",
        dest="image_root",
        help=(
            "Optional image root directory. When provided, the positional `path` must be relative. "
            "The effective image path is image_root / path and must resolve inside image_root. "
            "If the resolved path is a directory, files in that directory are processed non-recursively."
        ),
    )
    parser.add_argument(
        "--overwrite-md",
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="If set, existing .md files will be overwritten. By default existing .md are skipped.",
    )
    parser.add_argument(
        "--max-side",
        dest="max_side",
        type=int,
        default=None,
        help=(
            "If set, downscale images in-memory before sending to the OpenAI API so their longest side does not exceed this many pixels."
        ),
    )

    args = parser.parse_args()

    if args.image_root:
        try:
            abs_path, rel = resolve_image_and_relative(args.image_root, args.path)
        except SystemExit:
            # resolve_image_and_relative prints to stderr and exits with non-zero; propagate
            raise

        if os.path.isdir(abs_path):
            process_scan_directory(abs_path, overwrite=args.overwrite, max_side=args.max_side)
        else:
            try:
                process_scan_image(abs_path, overwrite=args.overwrite, max_side=args.max_side)
            except SystemExit:
                raise
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(2)
    else:
        # no image_root: use provided path directly
        if os.path.isdir(args.path):
            process_scan_directory(args.path, overwrite=args.overwrite, max_side=args.max_side)
        else:
            try:
                process_scan_image(args.path, overwrite=args.overwrite, max_side=args.max_side)
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(2)


if __name__ == "__main__":
    main()
