import os
from typing import Iterator, Optional


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def is_image_file(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext in SUPPORTED_EXTENSIONS


def sidecar_path_for_image(image_path: str) -> str:
    base, _ = os.path.splitext(image_path)
    return base + ".json"


def guess_image_path(json_path: str) -> Optional[str]:
    base, _ = os.path.splitext(json_path)
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        candidate = base + ext
        if os.path.exists(candidate):
            return candidate
    return None


def iter_images(directory: str) -> Iterator[str]:
    for root, _, files in os.walk(directory):
        for name in files:
            path = os.path.join(root, name)
            if is_image_file(path):
                yield path
