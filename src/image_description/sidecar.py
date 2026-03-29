import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Sidecar:
    """Represents the JSON sidecar contract.

    Keep this permissive and forward-compatible:
    - unknown fields are preserved in `extra`
    - missing fields default to empty values
    """

    original_title: str = ""
    original_description: str = ""
    title: str = ""
    visually_challenged_description: str = ""
    enhanced_description: str = ""
    keywords: List[str] = field(default_factory=list)
    hashtags: str = ""
    social_caption: str = ""

    # Added per design doc: image identity information.
    # `image_filename` is required to make the image-sidecar link explicit.
    # `image_relative_path` is optional and may be empty; kept for future use.
    image_filename: str = ""
    image_relative_path: str = ""

    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Sidecar":
        known_keys = {
            "original_title",
            "original_description",
            "title",
            "visually_challenged_description",
            "enhanced_description",
            "keywords",
            "hashtags",
            "social_caption",
            "image_filename",
            "image_relative_path",
        }

        keywords = data.get("keywords", [])
        if not isinstance(keywords, list):
            keywords = [str(keywords)]
        keywords = [str(k).strip() for k in keywords if str(k).strip()]

        extra = {k: v for k, v in data.items() if k not in known_keys}

        return cls(
            original_title=str(data.get("original_title", "") or ""),
            original_description=str(data.get("original_description", "") or ""),
            title=str(data.get("title", "") or ""),
            visually_challenged_description=str(
                data.get("visually_challenged_description", "") or ""
            ),
            enhanced_description=str(data.get("enhanced_description", "") or ""),
            keywords=keywords,
            hashtags=str(data.get("hashtags", "") or ""),
            social_caption=str(data.get("social_caption", "") or ""),
            image_filename=str(data.get("image_filename", "") or ""),
            image_relative_path=str(data.get("image_relative_path", "") or ""),
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "original_title": self.original_title,
            "original_description": self.original_description,
            "title": self.title,
            "visually_challenged_description": self.visually_challenged_description,
            "enhanced_description": self.enhanced_description,
            "keywords": list(self.keywords),
            "hashtags": self.hashtags,
            "social_caption": self.social_caption,
            "image_filename": self.image_filename,
            "image_relative_path": self.image_relative_path,
        }
        data.update(self.extra)
        return data

    @classmethod
    def load(cls, json_path: str) -> "Sidecar":
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Sidecar JSON must be an object: {json_path}")
        return cls.from_dict(data)

    def save(self, json_path: str, image_root: Optional[str] = None) -> None:
        """
        Save the sidecar to json_path. If image_root is provided, compute and set
        image_relative_path from image_root and this sidecar's image_filename.

        Validation behaviour when image_root is provided:
        - image_filename must be set and must NOT be an absolute path.
        - the resolved path (join(image_root, image_filename) normalized) must be
          located within image_root (no escaping via ..).

        On validation errors, print a message to stderr and exit non-zero.
        """
        # If an image_root was supplied, validate and compute image_relative_path.
        if image_root is not None:
            if not self.image_filename:
                sys.stderr.write("Error: --image-root was provided but sidecar.image_filename is empty.\n")
                sys.exit(2)

            # Reject absolute image filenames when image_root is used.
            if os.path.isabs(self.image_filename):
                sys.stderr.write("Error: absolute image paths are not allowed when --image-root is set.\n")
                sys.exit(2)

            # Normalize the image_root to an absolute canonical path (follow symlinks).
            root_abs = os.path.realpath(image_root)
            # Join and normalize the target image path.
            target = os.path.normpath(os.path.join(root_abs, self.image_filename))
            target = os.path.realpath(target)

            # Ensure the resolved target path is within the image_root.
            try:
                common = os.path.commonpath([root_abs, target])
            except ValueError:
                # In case paths are on different drives (Windows) or similar issues.
                sys.stderr.write("Error: invalid image_root or image_filename; cannot compute common path.\n")
                sys.exit(2)

            if common != root_abs:
                sys.stderr.write("Error: resolved image path escapes the image root (possible '..' in path).\n")
                sys.exit(2)

            # Compute relative path (may contain subdirectories)
            rel = os.path.relpath(target, start=root_abs)
            # Store the relative path using OS-native separators.
            self.image_relative_path = rel

        # Ensure destination directory exists
        os.makedirs(os.path.dirname(os.path.abspath(json_path)) or ".", exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)

    @staticmethod
    def social_path_for(sidecar_path: str) -> str:
        """Return the suggested sibling social JSON path for a given sidecar path.

        For: /path/to/IMG_1234.json -> /path/to/IMG_1234.social.json
        """
        root, ext = os.path.splitext(sidecar_path)
        return f"{root}.social.json"


# Helper for CLI use: resolve and validate an image path given an image_root option.
# This function is used by CLIs to apply the --image-root semantics.
def resolve_image_against_root(image_root: Optional[str], image_path: str) -> Tuple[str, str]:
    """
    Resolve image_path against image_root and return a tuple (resolved_abs_path, image_relative_path).

    Behaviour:
    - If image_root is None: resolved_abs_path = os.path.abspath(image_path); image_relative_path = os.path.basename(image_path)
    - If image_root is provided:
      - reject absolute image_path (print to stderr and exit non-zero)
      - join image_root + image_path, normalize and ensure the final path is within image_root
      - return (resolved_abs, relpath)

    On validation errors, prints to stderr and exits with code 2.
    """
    if image_root is None:
        resolved = os.path.abspath(image_path)
        rel = os.path.basename(image_path)
        return resolved, rel

    if os.path.isabs(image_path):
        sys.stderr.write("Error: absolute image paths are not allowed when --image-root is set.\n")
        sys.exit(2)

    root_abs = os.path.realpath(image_root)
    target = os.path.normpath(os.path.join(root_abs, image_path))
    target = os.path.realpath(target)

    try:
        common = os.path.commonpath([root_abs, target])
    except ValueError:
        sys.stderr.write("Error: invalid image_root or image_path; cannot compute common path.\n")
        sys.exit(2)

    if common != root_abs:
        sys.stderr.write("Error: resolved image path escapes the image root (possible '..' in path).\n")
        sys.exit(2)

    rel = os.path.relpath(target, start=root_abs)
    return target, rel


# Utility: list non-recursive image files in a directory resolved against image_root.
# CLIs can call this when the positional path resolves to a directory.
def list_files_non_recursive(root: str, rel_dir: str) -> List[str]:
    """
    Return a list of filenames (not absolute paths) contained directly in root/rel_dir.

    - root is an absolute path to the image_root
    - rel_dir is a path relative to root
    The returned list contains filenames relative to root (i.e. joined rel_dir + name).
    """
    base = os.path.realpath(os.path.join(root, rel_dir))
    # Ensure base is within root
    try:
        common = os.path.commonpath([os.path.realpath(root), base])
    except ValueError:
        return []
    if common != os.path.realpath(root):
        return []
    if not os.path.isdir(base):
        return []
    entries = []
    for name in os.listdir(base):
        full = os.path.join(base, name)
        if os.path.isfile(full):
            # Return path relative to root
            entries.append(os.path.relpath(full, start=root))
    return entries
