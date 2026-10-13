"""Export complete image sidecars to a web gallery JSON array."""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from ..gallery import build_gallery, write_gallery
from ..paths import is_image_file


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=(
        'Build gallery.json from image sidecars, preserving full records and '
        'adding file and alt aliases. Does not generate metadata or publish posts.'))
    parser.add_argument('folder', help='Folder to scan (non-recursive). Relative to --image-root when supplied.')
    parser.add_argument('--image-root', help='Website home folder; gallery file paths are relative to it.')
    parser.add_argument('--images', nargs='+', help='Selected filenames within the folder, in display order. Default: all images alphabetically.')
    args = parser.parse_args(argv)
    try:
        records = build_gallery(args.folder, image_root=args.image_root, images=args.images)
        folder = Path(args.image_root) / args.folder if args.image_root else Path(args.folder)
        output = folder / 'gallery.json'
        # Avoid replacing an image's core sidecar when that image is named gallery.
        if any(path.stem == 'gallery' and is_image_file(path.name)
               for path in folder.iterdir() if path.is_file()):
            raise ValueError('The image basename gallery is reserved for gallery.json; rename that image and its sidecar first')
        if output.is_symlink():
            raise ValueError('Refusing to replace a symlink at gallery.json')
        write_gallery(records, str(output))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print('Error: {}'.format(exc), file=sys.stderr)
        raise SystemExit(2)
    print('Wrote {} images to {}'.format(len(records), output))


if __name__ == '__main__':
    main()
