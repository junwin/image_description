"""Prepare Marie-Parie photos and YAML metadata without changing source images."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps

from image_description.image.describe import process_image

SIZES = {"web": (1200, 1200), "reel": (1080, 1920)}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def images_in(directory):
    return sorted((p for p in directory.iterdir()
                   if p.is_file() and p.suffix.lower() in EXTENSIONS),
                  key=lambda p: p.name.lower())


def render_photo(source, destination, dimensions, fit="contain"):
    """Auto-orient, then either preserve the entire photo or fill by cropping."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        if fit == "cover":
            result = ImageOps.fit(image, dimensions, method=Image.Resampling.LANCZOS)
        else:
            # Blurred photographic background avoids stretching or cutting off products.
            bg = ImageOps.fit(image, dimensions, method=Image.Resampling.LANCZOS)
            bg = bg.filter(ImageFilter.GaussianBlur(radius=35))
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
    width, height = SIZES[args.format]
    width = args.width or width
    height = args.height or height
    if width < 1 or height < 1:
        parser.error("width and height must be positive")
    if args.format == "reel" and (width % 2 or height % 2):
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
            render_photo(source, image, (width, height), args.fit)
            print(f"Prepared {image.name}")
        if not args.no_describe:
            process_image(str(image), preset="marie-parie", overwrite=args.overwrite,
                          model=args.model, provider=args.provider,
                          credential_path=args.credential_path)
        metadata = {}
        if sidecar.exists():
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if not yaml_path.exists() or args.overwrite:
            write_yaml(yaml_path, source, image, f"{width}:{height}", metadata)
    print(f"Completed {len(sources)} images in {output}")


if __name__ == "__main__":
    main()
