"""Build an ordered web gallery from existing image metadata sidecars."""
import json
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

from .paths import is_image_file, sidecar_path_for_image


def build_gallery(folder: str, image_root: Optional[str] = None,
                  images: Optional[Sequence[str]] = None) -> List[dict]:
    """Read a folder's images, or selected basenames in caller-supplied order.

    With image_root, folder must be relative to that root. Without it, folder
    is a filesystem path and doubles as the metadata root for identity checks.
    Output entries contain only file (a folder-relative basename) and alt.
    No source image or sidecar is changed.
    """
    if image_root is None:
        base = Path(folder).resolve()
        root = base
    else:
        root = Path(image_root).resolve()
        if Path(folder).is_absolute():
            raise ValueError('Folder must be relative when --image-root is set')
        base = (root / folder).resolve()
        try:
            base.relative_to(root)
        except ValueError:
            raise ValueError('Folder escapes the image root')
    if not base.is_dir():
        raise ValueError('Image folder does not exist: {}'.format(base))

    if images is None:
        names = sorted((p.name for p in base.iterdir()
                        if p.is_file() and is_image_file(p.name)),
                       key=lambda name: (name.casefold(), name))
    else:
        names = list(images)
    if not names:
        raise ValueError('No images selected; existing gallery has not been changed')
    if len(set(names)) != len(names):
        raise ValueError('Selection contains duplicate filenames')

    records = []
    used_sidecars = set()
    for name in names:
        if not isinstance(name, str) or not name or '/' in name or '\\' in name:
            raise ValueError('Selected images must be filenames within the folder')
        image = (base / name).resolve()
        try:
            image.relative_to(base)
        except ValueError:
            raise ValueError('Image escapes the selected folder: {}'.format(name))
        if not image.is_file() or not is_image_file(name):
            raise ValueError('Missing or unsupported image: {}'.format(name))
        sidecar = Path(sidecar_path_for_image(str(base / name)))
        if sidecar in used_sidecars:
            raise ValueError('Selected images share a sidecar: {}'.format(sidecar.name))
        used_sidecars.add(sidecar)
        if not sidecar.is_file():
            raise ValueError('Missing sidecar for {}: {}'.format(name, sidecar.name))
        try:
            sidecar.resolve().relative_to(base)
        except ValueError:
            raise ValueError('Sidecar escapes the selected folder: {}'.format(sidecar.name))
        with sidecar.open(encoding='utf-8') as stream:
            record = json.load(stream)
        if not isinstance(record, dict):
            raise ValueError('Sidecar must be a JSON object: {}'.format(sidecar.name))
        alt = record.get('image_description') or record.get('visually_challenged_description')
        if not isinstance(alt, str) or not alt.strip():
            raise ValueError('Missing image_description (alt text) for {}'.format(name))

        # Check explicit identity, including older sidecars with a path in
        # image_filename. Path values in the gallery use URL-friendly slashes.
        identity = record.get('image_filename')
        if identity and (not isinstance(identity, str) or
                         identity.replace('\\', '/').split('/')[-1] != name):
            raise ValueError('Sidecar image_filename does not match {}'.format(name))
        relative = (base / name).relative_to(root).as_posix()
        recorded_path = record.get('image_relative_path')
        if recorded_path and (not isinstance(recorded_path, str) or
                              recorded_path.replace('\\', '/') != relative):
            raise ValueError('Sidecar image_relative_path does not match {} under the image root'.format(name))
        # Legacy sidecars without identity cannot safely distinguish same-stem
        # files such as photo.jpg and photo.png.
        if not identity and not recorded_path:
            siblings = [p for p in base.iterdir() if p.is_file() and
                        is_image_file(p.name) and p.stem == Path(name).stem]
            if len(siblings) > 1:
                raise ValueError('Ambiguous sidecar identity for {}'.format(name))
        records.append({'file': name, 'alt': alt.strip()})
    return records


def write_gallery(records: List[dict], output: str) -> None:
    """Replace the gallery atomically, after all records have been validated."""
    destination = Path(output)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                         dir=str(destination.parent),
                                         prefix='.gallery-', suffix='.tmp',
                                         delete=False) as stream:
            temporary = stream.name
            json.dump(records, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
        os.replace(temporary, str(destination))
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
