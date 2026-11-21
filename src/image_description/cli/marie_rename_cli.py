"""Review and apply AI-suggested filenames to prepared Marie-Parie images.

No AI calls. By default prints a preview. --apply renames the prepared JPEG
and sidecars after checking the entire batch for missing files and collisions.
The original incoming images are never touched.
"""
import argparse
import json
import re
from pathlib import Path

from image_description.cli.marie_prepare_cli import images_in


SAFE_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*[.]jpg$")


def _target_filename(source, metadata):
    raw = metadata.get("suggested_filename", "")
    if not isinstance(raw, str) or not raw.strip():
        return None
    name = raw.strip()
    if not SAFE_NAME.fullmatch(name):
        raise ValueError(f"Invalid suggested_filename for {source.name}: {name!r}")
    match = re.match(r"^(\d{3}-)", source.name)
    if match and not name.startswith(match.group(1)):
        raise ValueError(f"Suggestion must preserve sequence {match.group(1)}: {name}")
    return name


def build_plan(folder):
    """Return prevalidated rename operations; never alter files."""
    directory = Path(folder).resolve()
    if not directory.is_dir():
        raise ValueError(f"Missing prepared image folder: {directory}")
    all_images = images_in(directory)
    changes = []
    reserved = set()
    for image in all_images:
        if image.suffix.lower() != ".jpg":
            raise ValueError(f"Prepared folder must contain JPEGs: {image.name}")
        sidecar = image.with_suffix(".json")
        yaml_path = image.with_suffix(".yaml")
        if not sidecar.is_file() or not yaml_path.is_file():
            raise ValueError(f"Missing JSON/YAML sidecar for {image.name}")
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError(f"JSON metadata is not an object: {sidecar}")
        if metadata.get("image_filename") not in (None, "", image.name):
            raise ValueError(f"JSON image identity does not match: {image.name}")
        if metadata.get("image_relative_path") not in (None, "", image.name):
            raise ValueError(f"JSON relative path does not match: {image.name}")
        yaml_text = yaml_path.read_text(encoding="utf-8")
        lines = yaml_text.splitlines()
        name_lines = [line.split(": ", 1)[1] for line in lines
                      if line.startswith("suggested_filename: ")]
        if len(name_lines) != 1:
            raise ValueError(f"Expected one suggested_filename field: {yaml_path}")
        try:
            reviewed_name = json.loads(name_lines[0])
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid suggested_filename YAML scalar: {yaml_path}") from error
        # The YAML file is the review surface; edited names override the AI.
        name = _target_filename(image, {"suggested_filename": reviewed_name})
        if not name or name == image.name:
            continue
        destination = directory / name
        if name.casefold() in reserved:
            raise ValueError(f"Duplicate proposed name: {name}")
        reserved.add(name.casefold())
        if image.with_name(image.stem + ".social.json").exists():
            raise ValueError(f"Social sidecar exists; rename separately: {image.name}")
        for suffix in (".jpg", ".json", ".yaml"):
            target = destination.with_suffix(suffix)
            if target.exists():
                raise ValueError(f"Refusing to overwrite existing file: {target.name}")
        # Replace only filename fields: preserve manually edited captions etc.
        if not any(line.startswith("image: ") for line in lines):
            raise ValueError(f"Missing image field: {yaml_path}")
        if not any(line.startswith("suggested_filename: ") for line in lines):
            raise ValueError(f"Missing suggested_filename field: {yaml_path}")
        new_lines = []
        for line in lines:
            if line.startswith("image: "):
                line = "image: " + json.dumps(name, ensure_ascii=False)
            elif line.startswith("suggested_filename: "):
                line = "suggested_filename: " + json.dumps(name, ensure_ascii=False)
            new_lines.append(line)
        new_yaml = "\n".join(new_lines) + "\n"
        metadata["image_filename"] = name
        metadata["image_relative_path"] = name
        metadata["suggested_filename"] = name
        changes.append((image, destination, metadata, new_yaml))
    return changes


def rename(folder, apply=False):
    changes = build_plan(folder)
    if not changes:
        print("No proposed renames.")
        return []
    for source, target, _, _ in changes:
        print(f"{source.name}  ->  {target.name}")
    if not apply:
        print(f"Preview only: {len(changes)} rename(s). Use --apply after review.")
        return changes

    # Stage updated sidecars first; preflight above prevents name collisions.
    for source, target, metadata, yaml_text in changes:
        new_json = target.with_suffix(".json")
        new_yaml = target.with_suffix(".yaml")
        new_json.write_text(json.dumps(metadata, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
        new_yaml.write_text(yaml_text, encoding="utf-8")
        source.rename(target)
        source.with_suffix(".json").unlink()
        source.with_suffix(".yaml").unlink()
        print(f"Renamed {target.name}")
    return changes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Prepared web image folder")
    parser.add_argument("--apply", action="store_true", help="Rename after preflight; default previews only")
    args = parser.parse_args(argv)
    try:
        rename(args.input, apply=args.apply)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
