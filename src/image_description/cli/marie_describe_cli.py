"""Generate Marie-Parie JSON and YAML descriptions for previously prepared images."""
import argparse
import json
from pathlib import Path

from PIL import Image

from image_description.image.describe import process_image
from image_description.cli.marie_prepare_cli import images_in, write_yaml


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Folder of resized images")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--provider", choices=["openai", "gemini", "deepseek", "mistral", "ollama"])
    parser.add_argument("--credential-path")
    args = parser.parse_args(argv)
    if not args.input.is_dir():
        parser.error("input must be an existing directory")
    sources = images_in(args.input)
    if not sources:
        parser.error("no images found")

    for image in sources:
        sidecar = image.with_suffix(".json")
        yaml_path = image.with_suffix(".yaml")
        process_image(str(image), preset="marie-parie", overwrite=args.overwrite,
                      model=args.model, provider=args.provider,
                      credential_path=args.credential_path)
        if not sidecar.is_file():
            print(f"Skipped YAML for {image.name}: no JSON sidecar")
            continue
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        previous = yaml_path.read_text(encoding="utf-8") if yaml_path.exists() else ""
        original = image.name
        for line in previous.splitlines():
            if line.startswith("source: "):
                try:
                    original = json.loads(line[len("source: "):])
                except json.JSONDecodeError:
                    pass
                break
        with Image.open(image) as photo:
            width, height = photo.size
        write_yaml(yaml_path, Path(original), image, f"{width}:{height}", metadata)
        print(f"Described {image.name}")


if __name__ == "__main__":
    main()
