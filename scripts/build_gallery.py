"""Run the gallery builder directly from a source checkout."""
import sys
from pathlib import Path

# Keep direct invocation usable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from image_description.cli.gallery_builder_cli import main


if __name__ == '__main__':
    main()
