import argparse
import os
import sys

from ..notes.scan import process_scan_directory, process_scan_image
from ..paths import resolve_image_and_relative


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan photographed notes and produce Obsidian markdown files using Google Gemini (vision, via galet)."
    )
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
        "--overwrite-md", "--overwrite",
        dest="overwrite",
        action="store_true",
        help="If set, existing .md files will be overwritten. By default existing markdown files are skipped.",
    )
    parser.add_argument(
        "--max-side",
        dest="max_side",
        type=int,
        default=None,
        help=(
            "If set, downscale images in-memory before sending to the API so their longest side does not exceed this many pixels."
        ),
    )
    parser.add_argument(
        "--model",
        dest="model",
        default="gemini-3.6-flash",
        help=(
            "Vision model used for transcription (default: gemini-3.6-flash). "
            "Google Gemini handles handwriting best; it is the default. "
            "OpenAI can be selected explicitly, e.g. --model gpt-4o --provider openai."
        ),
    )
    parser.add_argument(
        "--provider",
        dest="provider",
        default=None,
        choices=["openai", "gemini", "deepseek", "mistral", "ollama"],
        help=(
            "Model provider. When omitted, the provider is inferred from the model name "
            "(gemini-* -> gemini, etc.) and falls back to openai."
        ),
    )
    parser.add_argument(
        "--credential-path",
        dest="credential_path",
        default=None,
        help=(
            "Directory holding galet credential files (oaicred.json, gemini_cred.json, ...). "
            "Defaults to GALET_CREDENTIAL_PATH, then ~/credential, otherwise provider env vars."
        ),
    )
    parser.add_argument(
        "--preprocess",
        dest="preprocess",
        action="store_true",
        help=(
            "Preprocess the image (grayscale, contrast stretch, denoise, sharpen) "
            "before sending to the model. Helps with low-contrast or noisy scans."
        ),
    )

    args = parser.parse_args()

    # Resolve path against image_root if provided; resolve_image_and_relative handles validation
    if args.image_root:
        abs_path, rel = resolve_image_and_relative(args.image_root, args.path)
    else:
        abs_path = os.path.abspath(args.path)

    if not os.path.exists(abs_path):
        print(f"Path not found: {abs_path}", file=sys.stderr)
        sys.exit(1)

    # Directory case
    if os.path.isdir(abs_path):
        try:
            process_scan_directory(
                abs_path,
                overwrite=args.overwrite,
                max_side=args.max_side,
                model=args.model,
                preprocess=args.preprocess,
                provider=args.provider,
                credential_path=args.credential_path,
            )
        except SystemExit:
            raise
        except Exception as e:
            print(f"Error processing directory {abs_path}: {e}", file=sys.stderr)
            sys.exit(2)
        return

    # Single image
    try:
        created = process_scan_image(
            abs_path,
            overwrite=args.overwrite,
            max_side=args.max_side,
            model=args.model,
            preprocess=args.preprocess,
            provider=args.provider,
            credential_path=args.credential_path,
        )
    except SystemExit:
        raise
    except Exception as e:
        print(f"Error processing image {abs_path}: {e}", file=sys.stderr)
        sys.exit(2)

    if not created:
        # process_scan_image prints its own skip messages
        return


if __name__ == "__main__":
    main()
