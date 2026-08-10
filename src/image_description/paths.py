import os
import sys
import tempfile
from typing import Iterator, Optional, Tuple


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def is_image_file(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext.lower() in {e.lower() for e in SUPPORTED_EXTENSIONS}


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
    """
    Iterate image files in a single directory (non-recursive).
    """
    if not os.path.isdir(directory):
        return
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if os.path.isfile(path) and is_image_file(path):
            yield path


def resolve_image_and_relative(image_root: Optional[str], image_path: str) -> Tuple[str, Optional[str]]:
    """
    Resolve an image path with an optional image_root.

    Rules when image_root is provided:
    - image_path must NOT be absolute (reject and exit non-zero).
    - Join image_root + image_path, normalize and resolve symlinks.
    - Ensure the resolved path is inside image_root (no ".." escape or symlink escape). If it escapes, print an error and exit non-zero.

    Returns: (absolute_image_path, image_relative_path_or_None)
    """
    if image_root is None:
        abs_path = os.path.abspath(image_path)
        return abs_path, None

    # When image_root is set, we expect a relative image_path.
    if os.path.isabs(image_path):
        print(f"Error: absolute image path not allowed when --image-root is used: {image_path}", file=sys.stderr)
        sys.exit(2)

    root_abs = os.path.abspath(image_root)
    # Use realpath to avoid symlink-based escapes
    root_real = os.path.realpath(root_abs)

    # Join and normalize the provided relative path
    joined = os.path.normpath(os.path.join(root_abs, image_path))
    joined_real = os.path.realpath(joined)

    try:
        # Compare commonpath of real paths to prevent escapes via '..' or symlinks
        common = os.path.commonpath([root_real, joined_real])
    except ValueError:
        # Different mounts/drives (unlikely on Unix), treat as escape
        print(f"Error: resolved path is not within the image root: {joined_real}", file=sys.stderr)
        sys.exit(2)

    if common != root_real:
        print(f"Error: resolved path escapes image root. root={root_real}, resolved={joined_real}", file=sys.stderr)
        sys.exit(2)

    rel = os.path.relpath(joined_real, root_real)
    return joined_real, rel


def sidecar_core_for_image(image_path: str, image_root: Optional[str] = None) -> dict:
    """
    Produce the minimal core fields for a sidecar related to image identity.

    Returns a dict with at least:
      - image_path: absolute path to the image
      - image_filename: base filename
      - image_relative_path: relative path under image_root (if image_root provided)
    """
    abs_path, rel = resolve_image_and_relative(image_root, image_path)
    return {
        "image_path": abs_path,
        "image_filename": os.path.basename(abs_path),
        "image_relative_path": rel,
    }


# Small validation tests that can be run by executing this module directly.
def _run_self_tests() -> None:
    import shutil

    print("Running paths.py self-tests...")

    tmp = tempfile.mkdtemp(prefix="paths_test_")
    try:
        root = os.path.join(tmp, "root")
        os.makedirs(root)

        # create a top-level image and a nested image
        img1 = os.path.join(root, "img1.JPG")
        with open(img1, "wb") as f:
            f.write(b"\x00")

        nested = os.path.join(root, "nested")
        os.makedirs(nested)
        img2 = os.path.join(nested, "img2.jpg")
        with open(img2, "wb") as f:
            f.write(b"\x00")

        # iter_images should only return the top-level image (non-recursive)
        images = list(iter_images(root))
        assert img1 in images and img2 not in images, f"iter_images returned unexpected list: {images}"

        # resolve_image_and_relative: success case
        abs_path, rel = resolve_image_and_relative(root, "img1.JPG")
        assert os.path.abspath(img1) == abs_path
        assert rel == "img1.JPG"

        # resolve_image_and_relative: reject absolute image path when root is set
        try:
            resolve_image_and_relative(root, os.path.abspath(img1))
        except SystemExit as e:
            assert e.code != 0
        else:
            raise AssertionError("Expected SystemExit for absolute image path when image_root is set")

        # resolve_image_and_relative: reject path that escapes root
        try:
            resolve_image_and_relative(root, "../etc/passwd")
        except SystemExit as e:
            assert e.code != 0
        else:
            raise AssertionError("Expected SystemExit for escaping path when image_root is set")

        # resolve_image_and_relative: reject symlink that points outside the root
        outside = os.path.join(tmp, "outside")
        os.makedirs(outside)
        outside_file = os.path.join(outside, "evil.jpg")
        with open(outside_file, "wb") as f:
            f.write(b"\x00")

        # create a symlink inside root that points to the outside file
        symlink_path = os.path.join(root, "link.jpg")
        try:
            os.symlink(outside_file, symlink_path)
        except (AttributeError, OSError):
            # symlink may not be supported on the platform running tests; skip this part
            print("Skipping symlink test; symlinks not supported in this environment.")
        else:
            try:
                resolve_image_and_relative(root, "link.jpg")
            except SystemExit as e:
                assert e.code != 0
            else:
                raise AssertionError("Expected SystemExit for symlink that resolves outside image_root")

        # resolve_image_and_relative: directory path should resolve and return rel
        abs_dir, rel_dir = resolve_image_and_relative(root, "nested")
        assert os.path.abspath(nested) == abs_dir
        assert rel_dir == os.path.join("nested")

        print("All self-tests passed.")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    _run_self_tests()
