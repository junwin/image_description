import argparse
import os
from datetime import date, datetime
from typing import List, Optional

from ..post.blog_post_builder import (
    build_post_from_json_paths,
    collect_json_paths,
)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entrypoint for building a Jekyll blog post from one or more image sidecar JSON files.

    This CLI supports a single JSON (backward compatible) or multiple JSONs /
    directories containing JSON files. When multiple sidecars are provided they
    are combined into a single post: the title from the first sidecar is used
    as the post title, a single H1 is written at the top of the body, and each
    sidecar is rendered under its own H2 section. Internal subsections use
    H3 headings.

    Note on images:
    - For a single JSON you may pass --image to explicitly set the web path
      used in the front matter and body (e.g. /assets/images/foo.jpg). If
      omitted the tool will look for an image file next to the JSON and copy
      it into the site's /assets/images/ directory.
    - When providing multiple JSONs, --image is invalid. Each sidecar will use
      the image found next to its JSON file and the CLI will copy those images
      into the target site's assets directory.

    Parameters:
    - argv: optional list of arguments for testing; if None, argparse reads from sys.argv.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build a Jekyll blog post Markdown file from one or more image metadata "
            "JSON sidecars and write it into a GitHub Pages repo (_posts), also "
            "copying image files into the site's /assets/images/ directory."
        )
    )
    parser.add_argument(
        "json_path",
        nargs="+",
        help=(
            "One or more metadata JSON files or directories containing JSON files. "
            "Directories are scanned non-recursively for .json files."
        ),
    )
    parser.add_argument(
        "--date",
        help=(
            "Post date in YYYY-MM-DD or full 'YYYY-MM-DD HH:MM:SS -ZZZZ' format. "
            "If omitted, today's date is used."
        ),
    )
    parser.add_argument(
        "--out-root",
        required=True,
        help=(
            "Root of the GitHub Pages repo (e.g. /home/junwin/src/repos/junwin.github.io). "
            "The post will be written under _posts/."
        ),
    )
    parser.add_argument(
        "--image",
        help=(
            "Web path to the image for front matter and body (e.g. /assets/images/foo.jpg). "
            "Valid only when providing a single JSON sidecar; when providing multiple "
            "sidecars this option is disallowed and the tool will use images found "
            "next to each JSON."
        ),
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=[],
        help="Optional list of categories for the post.",
    )

    args = parser.parse_args(argv)

    # Resolve json paths (support files and directories)
    json_inputs = args.json_path or []
    json_paths = collect_json_paths(json_inputs)

    # Determine date: use provided or today's date
    if args.date:
        try:
            if " " in args.date:
                datetime.strptime(args.date.split(" ")[0], "%Y-%m-%d")
                date_str = args.date
            else:
                datetime.strptime(args.date, "%Y-%m-%d")
                date_str = args.date
        except ValueError:
            raise SystemExit("--date must be in YYYY-MM-DD or 'YYYY-MM-DD HH:MM:SS -ZZZZ' format")
    else:
        today = date.today().strftime("%Y-%m-%d")
        date_str = today

    out_root = os.path.abspath(args.out_root)

    # If multiple JSONs provided, --image is invalid
    if len(json_paths) > 1 and args.image:
        raise SystemExit("--image may not be used when providing multiple JSON sidecars")

    out_path = build_post_from_json_paths(json_paths, out_root, date_str, args.image, args.categories)

    # Mirror previous behavior: print the path to the written file
    print(f"Wrote blog post to {out_path}")


if __name__ == "__main__":
    main()
