"""Prepare Marie-Parie photos and YAML metadata without changing source images."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps
from image_description.image.describe import process_image


SIZES = {"web": (600, 900), "reel": (1080, 1920)}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def images_in(directory):
    return sorted((p for p in directory.iterdir()
                   if p.is_file() and p.suffix.lower() in EXTENSIONS),
                  key=lambda p: p.name.lower())


def render_photo(source, destination, dimensions, fit="contain"):
    """Auto-orient, then either preserve the entire photo or fill by cropping."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as original:
        oriented = ImageOps.exif_transpose(original)
        # JPEG cannot retain transparency: flatten RGBA/P images against white.
        if "A" in oriented.getbands() or "transparency" in oriented.info:
            rgba = oriented.convert("RGBA")
            image = Image.new("RGB", rgba.size, (255, 255, 255))
            image.paste(rgba, mask=rgba.getchannel("A"))
        else:
            image = oriented.convert("RGB")
        if fit == "cover":
            result = ImageOps.fit(image, dimensions, method=Image.Resampling.LANCZOS)
        else:
            # Plain neutral background avoids cropping or visual distractions.
            bg = Image.new("RGB", dimensions, (255, 255, 255))
            foreground = ImageOps.contain(image, dimensions, method=Image.Resampling.LANCZOS)
            left = (dimensions[0] - foreground.width) // 2
            top = (dimensions[1] - foreground.height) // 2
            bg.paste(foreground, (left, top))
            result = bg
        result.save(destination, "JPEG", quality=92, optimize=True)


def yaml_scalar(value):
    # JSON double-quoted strings are valid YAML 1.2 scalars.
    return json.dumps(str(value or ""), ensure_ascii=False)


def write_yaml(destination, source, image, ratio, metadata):
    lines = [
        "source: " + yaml_scalar(source.name),
        "image: " + yaml_scalar(image.name),
        "aspect_ratio: " + yaml_scalar(ratio),
        "preset: marie-parie",
        "alt_text: " + yaml_scalar(metadata.get("image_description")),
        "description: " + yaml_scalar(metadata.get("enhanced_description")),
        "caption: " + yaml_scalar(metadata.get("social_caption")),
        "hashtags: " + yaml_scalar(metadata.get("hashtags")),
        "keywords:",
    ]
    lines.extend("  - " + yaml_scalar(v) for v in metadata.get("keywords", []))
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Folder of original photographs")
    parser.add_argument("--output", type=Path, required=True, help="Output folder (must differ from input)")
    parser.add_argument("--format", choices=SIZES, default="web")
    parser.add_argument("--fit", choices=["contain", "cover"], default="contain")
    parser.add_argument("--aspect-ratio", help="Target width:height, e.g. 3:2 or 600:413; keeps preset width unless --width is supplied")
    parser.add_argument("--width", type=int, help="Override preset width")
    parser.add_argument("--height", type=int, help="Override preset height")
    parser.add_argument("--no-describe", action="store_true", help="Prepare photos without AI calls")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing outputs and metadata")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--provider", choices=["openai", "gemini", "deepseek", "mistral", "ollama"])
    parser.add_argument("--credential-path")
    args = parser.parse_args(argv)

    folder = args.input.resolve()
    output = args.output.resolve()
    if not folder.is_dir():
        parser.error("input must be an existing directory")
    if output == folder or output in folder.parents:
        parser.error("output must be separate from input")
    dimensions = SIZES[args.format]
    if args.aspect_ratio:
        if args.height is not None:
            parser.error("--height cannot be combined with --aspect-ratio")
        try:
            ratio_w, ratio_h = (int(part.strip()) for part in args.aspect_ratio.split(":"))
            if ratio_w <= 0 or ratio_h <= 0:
                raise ValueError()
        except (ValueError, TypeError):
            parser.error("--aspect-ratio must be positive integers WIDTH:HEIGHT, e.g. 3:2")
        width = args.width if args.width is not None else dimensions[0]
        dimensions = (width, round(width * ratio_h / ratio_w))
    elif args.width is not None or args.height is not None:
        if args.width is None or args.height is None:
            parser.error("--width and --height must be supplied together unless using --aspect-ratio")
        dimensions = (args.width, args.height)
    if min(dimensions) < 1:
        parser.error("dimensions must be positive")
    if args.format == "reel" and (dimensions[0] % 2 or dimensions[1] % 2):
        parser.error("reel dimensions must be even for H.264 compatibility")

    sources = images_in(folder)
    if not sources:
        parser.error("no JPG, PNG or WebP images found")
    output.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(sources, start=1):
        # Numbered exports preserve the order for the video stage.
        image = output / f"{index:03d}-{source.stem}.jpg"
        sidecar = image.with_suffix(".json")
        yaml_path = image.with_suffix(".yaml")
        if image.exists() and not args.overwrite:
            print(f"Using existing {image.name}")
        else:
            render_photo(source, image, dimensions, args.fit)
            print(f"Prepared {image.name}")
        if not args.no_describe:
            process_image(str(image), preset="marie-parie", overwrite=args.overwrite,
                          model=args.model, provider=args.provider,
                          credential_path=args.credential_path)
        metadata = {}
        if sidecar.exists():
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if not yaml_path.exists() or args.overwrite:
            with Image.open(image) as exported:
                actual_width, actual_height = exported.size
            write_yaml(yaml_path, source, image, f"{actual_width}:{actual_height}", metadata)
    print(f"Completed {len(sources)} images in {output}")


if __name__ == "__main__":
    main()
