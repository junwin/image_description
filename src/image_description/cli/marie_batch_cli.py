"""Marie-Parie UAT batch workflow, with predictable folder names.

Lucy attachments export originals into Pictures/marie-parie/incoming/batch-<id>.
Use the batch ID for every subsequent command. Raw originals are never altered.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from image_description.marie_workspace import batch_paths
from image_description.cli.marie_prepare_cli import main as prepare_main
from image_description.cli.marie_inspect_cli import inspect_batch
from image_description.cli.marie_video_cli import main as video_main


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path,
                        help="Workspace root (default: ~/Pictures/marie-parie)")
    parser.add_argument("--batch", required=True,
                        help="Batch name returned by Lucy attachments export")
    command = parser.add_subparsers(dest="action", required=True)
    command.add_parser("paths", help="Print source, prepared and output paths as JSON")
    command.add_parser("inspect", help="Assess crop vs contain, without AI")

    resize = command.add_parser("resize", help="Prepare web or Reel images")
    resize.add_argument("--format", choices=["web", "reel"], required=True)
    resize.add_argument("--fit", choices=["auto", "natural", "contain", "contain-blur", "cover"], default="auto")
    resize.add_argument("--background", default="white",
                        help="Canvas fill for solid contain (ignored by contain-blur)")
    resize.add_argument("--overwrite", action="store_true",
                        help="Regenerate previously prepared images")

    reel = command.add_parser("reel", help="Render a Reel from prepared/reel batch images")
    reel.add_argument("--seconds-per-image", type=float, default=2.0)
    reel.add_argument("--seconds", type=float, help="Optional total video duration")
    reel.add_argument("--fade", type=float, default=0.4)
    reel.add_argument("--fps", type=int, default=30)
    reel.add_argument("--movement", choices=["zoom", "none"], default="zoom")
    reel.add_argument("--supersample", type=int, choices=[1, 2, 3], default=2)
    reel.add_argument("--overwrite", action="store_true",
                      help="Replace existing Reel output")

    args = parser.parse_args(argv)
    try:
        paths = batch_paths(args.batch, args.workspace)
    except ValueError as error:
        parser.error(str(error))

    if args.action == "paths":
        print(json.dumps(paths.as_dict(), indent=2))
        return

    if not paths.incoming.is_dir():
        parser.error(f"Original batch not found: {paths.incoming}")

    if args.action == "inspect":
        try:
            print(json.dumps(inspect_batch(paths.incoming), indent=2))
        except (ValueError, OSError) as error:
            parser.error(str(error))
        return

    if args.action == "resize":
        destination = paths.web if args.format == "web" else paths.reel_images
        arguments = [
            str(paths.incoming), "--output", str(destination), "--format",
            args.format, "--fit", args.fit, "--background", args.background,
            "--no-describe",
        ]
        if args.overwrite:
            arguments.append("--overwrite")
        prepare_main(arguments)
        return

    if args.action == "reel":
        if not paths.reel_images.is_dir():
            parser.error(f"Prepare Reel images first: {paths.reel_images}")
        if paths.reel_video.exists() and not args.overwrite:
            parser.error("Finished Reel already exists; pass --overwrite to replace")
        arguments = [
            str(paths.reel_images), "--output", str(paths.reel_video),
            "--seconds-per-image", str(args.seconds_per_image),
            "--fade", str(args.fade), "--fps", str(args.fps),
            "--movement", args.movement, "--supersample", str(args.supersample),
        ]
        if args.seconds is not None:
            arguments.extend(["--seconds", str(args.seconds)])
        video_main(arguments)


if __name__ == "__main__":
    main()
