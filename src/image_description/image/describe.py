import os
from typing import List

from ..paths import is_image_file, iter_images, sidecar_path_for_image
from ..sidecar import Sidecar
from .iptc import show_image_iptc_meta, write_iptc_meta
from .openai_client import generate_openai_description_and_keywords


MAX_FILE_SIZE_MB = 20


def file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def merge_keywords(existing: List[str], new: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for kw in existing + new:
        norm = kw.strip()
        if not norm:
            continue
        lower = norm.lower()
        if lower not in seen:
            seen.add(lower)
            merged.append(norm)
    return merged


def build_hashtags(keywords: List[str]) -> str:
    """Build a space-separated hashtag string from keywords.

    Notes:
    - Keywords sometimes come back as a single comma-separated string
      (e.g. "photography, photo, Nature"). We split those into individual tags.
    - We normalize by lowercasing and removing spaces.
    - We de-duplicate case-insensitively.
    """

    seen = set()
    tags: List[str] = []

    for kw in keywords:
        if kw is None:
            continue

        # Split comma-separated keyword strings into individual tokens
        parts = [p.strip() for p in str(kw).split(",")]
        for part in parts:
            if not part:
                continue

            normalized = part.replace(" ", "").lower()
            if not normalized:
                continue

            if normalized not in seen:
                seen.add(normalized)
                tags.append("#" + normalized)

    return " ".join(tags)


def process_image(image_path: str, preset: str) -> None:
    if not is_image_file(image_path):
        print(f"Skipping non-image file: {image_path}")
        return

    size_mb = file_size_mb(image_path)
    if size_mb > MAX_FILE_SIZE_MB:
        print(
            f"Skipping {image_path}: file size {size_mb:.2f} MB exceeds limit of {MAX_FILE_SIZE_MB} MB"
        )
        return

    print(f"Processing {image_path} ({size_mb:.2f} MB)")

    existing_title, existing_description, existing_keywords = show_image_iptc_meta(image_path)

    vc_desc, enhanced_desc, new_keywords = generate_openai_description_and_keywords(
        image_path,
        existing_title,
        existing_description,
        existing_keywords,
        preset=preset,
    )

    merged_keywords = merge_keywords(existing_keywords, new_keywords)
    hashtags = build_hashtags(merged_keywords)

    sidecar = Sidecar(
        original_title=existing_title,
        original_description=existing_description,
        title=existing_title,
        visually_challenged_description=vc_desc,
        enhanced_description=enhanced_desc,
        keywords=merged_keywords,
        hashtags=hashtags,
    )

    json_file_path = sidecar_path_for_image(image_path)
    sidecar.save(json_file_path)
    print(f"Wrote metadata to {json_file_path}")


def process_directory(directory: str, preset: str) -> None:
    for path in iter_images(directory):
        try:
            process_image(path, preset=preset)
        except Exception as e:  # noqa: BLE001
            print(f"Error processing {path}: {e}")


def embed_metadata(directory: str) -> None:
    """Read JSON sidecars and write selected fields back into image IPTC."""
    for path in iter_images(directory):
        json_file_path = sidecar_path_for_image(path)
        if not os.path.exists(json_file_path):
            continue

        try:
            sidecar = Sidecar.load(json_file_path)
        except Exception as e:  # noqa: BLE001
            print(f"Error reading {json_file_path}: {e}")
            continue

        title = sidecar.title or sidecar.original_title
        description = sidecar.enhanced_description or sidecar.original_description
        keywords = sidecar.keywords or []

        write_iptc_meta(path, title=title, description=description, keywords=keywords)
