"""Inspect a batch of original photos and suggest non-destructive Reel fitting.

No AI requests or output images are generated. Inspect uses EXIF-corrected
dimensions and estimates how much of each image cover mode would discard.
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps

from image_description.cli.marie_prepare_cli import images_in, SIZES


def inspect_photo(path, target):
    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw)
        width, height = image.size
    target_ratio = target[0] / target[1]
    source_ratio = width / height
    # Cover crops one dimension; report the source-area fraction discarded.
    crop_fraction = max(0.0, 1.0 - min(source_ratio / target_ratio,
                                        target_ratio / source_ratio))
    # Keep recommendations conservative; protect the full image on uncertain crops.
    recommendation = "cover" if crop_fraction <= 0.12 else "contain"
    return {
        "file": path.name,
        "width": width, "height": height,
        "orientation": "square" if width == height else (
            "landscape" if width > height else "portrait"),
        "aspect_ratio": round(source_ratio, 4),
        "cover_crop_percent": round(crop_fraction * 100, 1),
        "suggested_fit": recommendation,
        "reason": ("Minor crop expected" if recommendation == "cover"
                   else "Substantial crop; keep entire photograph unless approved"),
    }


def inspect_batch(directory, target=SIZES["reel"]):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError("Input must be an existing directory")
    if min(target) <= 0:
        raise ValueError("Target dimensions must be positive")
    sources = images_in(directory)
    if not sources:
        raise ValueError("No JPG, PNG or WebP images found")
    items = [inspect_photo(path, target) for path in sources]
    return {
        "target": {"width": target[0], "height": target[1]},
        "count": len(items),
        "images": items,
        "suggested_batch_fit": (
            "contain" if any(item["suggested_fit"] == "contain" for item in items)
            else "cover"
        ),
        "note": ("Suggestions are based only on geometry, not subject position. "
                 "Review crops visually before approving cover. Use --fit contain "
                 "--background gray to preserve every pixel, or --fit cover to fill."),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--format", choices=SIZES, default="reel")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON for Gwen")
    args = parser.parse_args(argv)
    dimensions = SIZES[args.format]
    if args.width is not None or args.height is not None:
        if args.width is None or args.height is None:
            parser.error("--width and --height must be supplied together")
        dimensions = (args.width, args.height)
    try:
        report = inspect_batch(args.input, dimensions)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Target: {dimensions[0]}x{dimensions[1]}")
        for item in report["images"]:
            print(f"{item['file']}: {item['width']}x{item['height']}, "
                  f"cover crops ~{item['cover_crop_percent']}%, "
                  f"suggest {item['suggested_fit']}")
        print(f"Suggested batch fit: {report['suggested_batch_fit']}")
        print(report["note"])


if __name__ == "__main__":
    main()
