import argparse
import os
import sys

from ..image.describe import embed_metadata, process_directory, process_image
from ..image.iptc import get_exif_date
from ..image.prompts import PROMPT_PRESETS
from ..paths import resolve_image_and_relative, iter_images, sidecar_path_for_image
from ..sidecar import Sidecar


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
    describe_parser.add_argument(
        "--max-side",
        dest="max_side",
        type=int,
        default=None,
        help=(
            "If set, downscale images in-memory before sending to the OpenAI API "
            "so their longest side does not exceed this many pixels. "
            "Original files on disk are never modified."
        ),
    )
    describe_parser.add_argument(
        "--image-root",
        dest="image_root",
        help=(
            "Optional image root directory. When provided, the positional `path` must be relative. "
            "The effective image path is image_root / path and must resolve inside image_root. "
            "If the resolved path is a directory, files in that directory are processed non-recursively."
        ),
    )

    embed_parser = subparsers.add_parser(
        "embed",
        help="Embed metadata from JSON sidecars back into image IPTC.",
    )
    embed_parser.add_argument("directory", help="Directory containing images and JSON sidecars.")

    backfill_parser = subparsers.add_parser(
        "backfill-dates",
        help="Backfill capture_datetime into existing sidecar JSONs from EXIF data.",
    )
    backfill_parser.add_argument("path", help="Image file or directory to process.")
    backfill_parser.add_argument(
        "--image-root",
        dest="image_root",
        help=(
            "Optional image root directory. When provided, the positional `path` must be relative. "
            "The effective image path is image_root / path and must resolve inside image_root. "
            "If the resolved path is a directory, files in that directory are processed non-recursively."
        ),
    )

    args = parser.parse_args()

    if args.command == "describe":
        # If image_root is set, validate and resolve the provided path early.
        if args.image_root:
            # resolve_image_and_relative will print errors to stderr and exit non-zero on invalid inputs.
            abs_path, rel = resolve_image_and_relative(args.image_root, args.path)

            if os.path.isdir(abs_path):
                # Gather images in top-level of this directory (non-recursive)
                images = list(iter_images(abs_path))
                existed_before = {p: os.path.exists(sidecar_path_for_image(p)) for p in images}

                # Process directory (this will create sidecars where appropriate)
                process_directory(abs_path, preset=args.preset, overwrite=args.overwrite,
                                  max_side=args.max_side)

                # Post-process sidecars to include image_relative_path when allowed.
                for image_path in images:
                    json_path = sidecar_path_for_image(image_path)
                    if not os.path.exists(json_path):
                        # Nothing to update
                        continue
                    if existed_before.get(image_path) and not args.overwrite:
                        # Respect default no-overwrite policy
                        continue

                    try:
                        sidecar = Sidecar.load(json_path)
                    except Exception as e:
                        print(f"Error reading sidecar {json_path}: {e}", file=sys.stderr)
                        sys.exit(2)

                    # image_relative should be the path relative to image_root
                    image_rel = os.path.relpath(image_path, start=args.image_root)
                    # Store relative path (may include subdirectories)
                    sidecar.image_filename = image_rel
                    try:
                        sidecar.save(json_path, image_root=args.image_root)
                    except SystemExit:
                        # sidecar.save prints its own error to stderr and exits; re-raise
                        raise
                    except Exception as e:
                        print(f"Error writing updated sidecar {json_path}: {e}", file=sys.stderr)
                        sys.exit(2)

            else:
                # Single image path case
                abs_path = abs_path
                json_path = sidecar_path_for_image(abs_path)
                existed_before = os.path.exists(json_path)

                process_image(abs_path, preset=args.preset, overwrite=args.overwrite,
                              max_side=args.max_side)

                if not os.path.exists(json_path):
                    # process_image may have skipped this file (non-image, too large, etc.)
                    print(f"No sidecar produced for {abs_path}", file=sys.stderr)
                    return

                if existed_before and not args.overwrite:
                    # Respect default behavior: do not overwrite existing sidecars
                    return

                try:
                    sidecar = Sidecar.load(json_path)
                except Exception as e:
                    print(f"Error reading sidecar {json_path}: {e}", file=sys.stderr)
                    sys.exit(2)

                image_rel = os.path.relpath(abs_path, start=args.image_root)
                sidecar.image_filename = image_rel
                try:
                    sidecar.save(json_path, image_root=args.image_root)
                except SystemExit:
                    raise
                except Exception as e:
                    print(f"Error writing updated sidecar {json_path}: {e}", file=sys.stderr)
                    sys.exit(2)

        else:
            # No image_root: preserve existing behavior
            if os.path.isdir(args.path):
                process_directory(args.path, preset=args.preset, overwrite=args.overwrite,
                                  max_side=args.max_side)
            else:
                process_image(args.path, preset=args.preset, overwrite=args.overwrite,
                              max_side=args.max_side)

    elif args.command == "embed":
        embed_metadata(args.directory)

    elif args.command == "backfill-dates":
        _backfill_dates(args)


def _backfill_dates(args) -> None:
    """Backfill capture_datetime into existing sidecar JSONs from EXIF.

    Only sets capture_datetime on sidecars where it's currently empty.
    Does not modify any other fields.
    """
    # Resolve the path, optionally against image_root.
    if args.image_root:
        abs_path, _rel = resolve_image_and_relative(args.image_root, args.path)
    else:
        abs_path = os.path.abspath(args.path)

    # Collect image paths.
    if os.path.isdir(abs_path):
        image_paths = list(iter_images(abs_path))
    elif os.path.isfile(abs_path):
        image_paths = [abs_path]
    else:
        print(f"Path not found: {abs_path}", file=sys.stderr)
        sys.exit(1)

    updated = 0
    skipped_missing_sidecar = 0
    skipped_already_set = 0
    skipped_no_exif = 0

    for image_path in image_paths:
        json_path = sidecar_path_for_image(image_path)
        if not os.path.exists(json_path):
            print(f"Skipping {os.path.basename(image_path)}: no sidecar found")
            skipped_missing_sidecar += 1
            continue

        try:
            sidecar = Sidecar.load(json_path)
        except Exception as e:
            print(f"Error reading {json_path}: {e}", file=sys.stderr)
            continue

        # Only backfill if currently empty
        if sidecar.capture_datetime:
            print(f"Skipping {os.path.basename(image_path)}: capture_datetime already set ({sidecar.capture_datetime})")
            skipped_already_set += 1
            continue

        exif_date = get_exif_date(image_path)
        if not exif_date:
            print(f"Skipping {os.path.basename(image_path)}: no EXIF date found")
            skipped_no_exif += 1
            continue

        sidecar.capture_datetime = exif_date
        try:
            sidecar.save(json_path)
        except Exception as e:
            print(f"Error saving {json_path}: {e}", file=sys.stderr)
            continue

        print(f"Updated {os.path.basename(image_path)}: capture_datetime = {exif_date}")
        updated += 1

    print(
        f"\nBackfill complete: {updated} updated, "
        f"{skipped_already_set} already had dates, "
        f"{skipped_no_exif} had no EXIF date, "
        f"{skipped_missing_sidecar} had no sidecar."
    )


if __name__ == "__main__":
    main()
