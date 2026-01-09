import argparse
import os
from datetime import date, datetime
from typing import List, Optional

from ..post.blog_post_builder import build_post_from_json, guess_image_web_path


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entrypoint for building a Jekyll blog post from an image sidecar JSON.

    This module is intentionally thin: it parses command-line arguments and
    delegates the work to image_description.post.blog_post_builder.build_post_from_json().

    Parameters:
    - argv: optional list of arguments for testing; if None, argparse reads
      from sys.argv.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build a Jekyll blog post Markdown file from an image metadata JSON "
            "and write it into a GitHub Pages repo (_posts), also copying the image."
        )
    )
    parser.add_argument("json_path", help="Path to the metadata JSON file.")
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
            "If omitted, a simple guess is made based on the JSON filename."
        ),
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=[],
        help="Optional list of categories for the post.",
    )

    args = parser.parse_args(argv)

    json_path = os.path.abspath(args.json_path)

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

    image_web_path = guess_image_web_path(json_path, args.image)

    out_path = build_post_from_json(json_path, out_root, date_str, image_web_path, args.categories)

    # Mirror previous behavior: print the path to the written file
    print(f"Wrote blog post to {out_path}")


if __name__ == "__main__":
    main()


# Manual verification / minimal tests
# - Unit testing: call build_post_from_json() from tests with a small Sidecar JSON
#   fixture and a temporary out_root, then assert the file exists and contains
#   expected front matter and body sections.
# - CLI smoke test (manual):
#     python -m image_description.cli.blog_post_builder_cli path/to/meta.json --out-root /tmp/site
#   should print the output path and create /tmp/site/_posts/YYYY-MM-DD-<slug>.md
# - Packaging note: to expose a console script entrypoint add to your packaging
#   configuration (setup.cfg/pyproject.toml):
#       [options.entry_points]
#       console_scripts =
#           blog-post-builder = image_description.cli.blog_post_builder_cli:main
