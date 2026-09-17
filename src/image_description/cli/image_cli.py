import argparse
import os
import sys

from ..image.describe import embed_metadata, process_directory, process_image
from ..image.iptc import get_exif_date
from ..image.prompts import PROMPT_PRESETS
from ..paths import is_image_file, resolve_image_and_relative, iter_images, sidecar_path_for_image
from ..sidecar import Sidecar


def main() -> None:
    parser = argparse.ArgumentParser(description="Describe images and manage metadata using vision models (via galet).")
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
            "By default existing sidecars are skipped to avoid unnecessary API calls."
        ),
    )
    describe_parser.add_argument(
        "--max-side",
        dest="max_side",
        type=int,
        default=None,
        help=(
            "If set, downscale images in-memory before sending to the API "
            "so their longest side does not exceed this many pixels. "
            "Original files on disk are never modified."
        ),
    )
    describe_parser.add_argument(
        "--model",
        dest="model",
        default="gpt-4o-mini",
        help="Vision model used for descriptions (default: gpt-4o-mini).",
    )
    describe_parser.add_argument(
        "--provider",
        dest="provider",
        default=None,
        choices=["openai", "gemini", "deepseek", "mistral", "ollama"],
        help=(
            "Model provider. When omitted, the provider is inferred from the model name "
            "(gemini-* -> gemini, etc.) and falls back to openai."
        ),
    )
    describe_parser.add_argument(
        "--credential-path",
        dest="credential_path",
        default=None,
        help=(
            "Directory holding galet credential files (oaicred.json, gemini_cred.json, ...). "
            "Defaults to GALET_CREDENTIAL_PATH, then ~/credential, otherwise provider env vars."
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

    list_ready_parser = subparsers.add_parser(
        "list-ready",
        help="List images whose sidecar has can_publish: true (ready to publish).",
    )
    list_ready_parser.add_argument("path", help="Image file or directory to scan.")
    list_ready_parser.add_argument(
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
                                  max_side=args.max_side, model=args.model,
                                  provider=args.provider, credential_path=args.credential_path)

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

                    # image_filename is the bare base filename; the path relative
                    # to image_root (which may include subdirectories) goes into
                    # image_relative_path.
                    sidecar.image_filename = os.path.basename(image_path)
                    sidecar.image_relative_path = os.path.relpath(image_path, start=args.image_root)
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
                              max_side=args.max_side, model=args.model,
                              provider=args.provider, credential_path=args.credential_path)

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

                sidecar.image_filename = os.path.basename(abs_path)
                sidecar.image_relative_path = os.path.relpath(abs_path, start=args.image_root)
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
                                  max_side=args.max_side, model=args.model,
                                  provider=args.provider, credential_path=args.credential_path)
            else:
                process_image(args.path, preset=args.preset, overwrite=args.overwrite,
                              max_side=args.max_side, model=args.model,
                              provider=args.provider, credential_path=args.credential_path)

    elif args.command == "embed":
        embed_metadata(args.directory)

    elif args.command == "backfill-dates":
        _backfill_dates(args)

    elif args.command == "list-ready":
        _list_ready(args)


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


def _list_ready(args) -> None:
    """List images whose sidecar has can_publish: true.

    Scans a single image file, or a directory non-recursively. For each
    image, looks for its core sidecar (image.json), falling back to the
    sibling social sidecar (image.social.json). Prints the image path for
    every sidecar flagged can_publish: true.

    Output paths:
      - with --image-root: relative to the image root (reusable with --image-root)
      - without: absolute paths

    stdout is machine-readable (one matching image path per line);
    informational messages go to stderr.
    """
    if args.image_root:
        abs_path, _rel = resolve_image_and_relative(args.image_root, args.path)
        root_abs = os.path.realpath(args.image_root)
    else:
        abs_path = os.path.abspath(args.path)
        root_abs = None

    if not os.path.exists(abs_path):
        print(f"Path not found: {abs_path}", file=sys.stderr)
        sys.exit(1)

    if os.path.isdir(abs_path):
        image_paths = list(iter_images(abs_path))
    elif is_image_file(abs_path):
        image_paths = [abs_path]
    else:
        print(f"Not an image file: {abs_path}", file=sys.stderr)
        sys.exit(1)

    ready = []
    for image_path in image_paths:
        core = sidecar_path_for_image(image_path)
        base, _ext = os.path.splitext(image_path)
        candidates = [core, base + ".social.json"]

        sidecar_file = next((c for c in candidates if os.path.exists(c)), None)
        if sidecar_file is None:
            print(f"No sidecar for {image_path}", file=sys.stderr)
            continue

        try:
            sidecar = Sidecar.load(sidecar_file)
        except Exception as e:
            print(f"Error reading {sidecar_file}: {e}", file=sys.stderr)
            continue

        if sidecar.can_publish:
            if root_abs is not None:
                ready.append(os.path.relpath(os.path.realpath(image_path), start=root_abs))
            else:
                ready.append(os.path.abspath(image_path))

    for p in ready:
        print(p)

    if not ready:
        print("No images ready to publish (can_publish: true).", file=sys.stderr)


if __name__ == "__main__":
    main()
