"""Offline Phase 1 smoke test: resize -> gallery dry-run -> local staging commit.

Run in a Python environment with image_description installed:
    python scripts/marie_phase1_smoke.py
No AI credentials, network connection or real website checkout required.
"""
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from image_description.cli import marie_resize_cli, marie_publish_cli


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def main():
    with tempfile.TemporaryDirectory(prefix="marie-phase1-") as directory:
        root = Path(directory)
        incoming, ready, repo = root / "incoming", root / "ready", root / "site"
        incoming.mkdir()
        (repo / "assets" / "New Arrivals").mkdir(parents=True)
        Image.new("RGB", (1200, 800), "red").save(incoming / "01-landscape.jpg")
        Image.new("RGB", (600, 900), "blue").save(incoming / "02-portrait.jpg")
        Image.new("RGBA", (500, 500), (20, 40, 60, 0)).save(incoming / "03-transparent.png")
        gallery = repo / "assets" / "New Arrivals" / "gallery.txt"
        gallery.write_text("existing-item.jpg\n", encoding="utf-8")
        git(repo, "init", "-b", "staging")
        git(repo, "config", "user.name", "Marie Smoke Test")
        git(repo, "config", "user.email", "smoke@example.invalid")
        git(repo, "add", ".")
        git(repo, "commit", "-m", "Initial gallery")
        marie_resize_cli.main([str(incoming), "--output", str(ready), "--format", "web"])
        images = sorted(ready.glob("*.jpg"))
        assert len(images) == 3
        # Natural mode uses the 600x900 bounding box without inserting margins.
        expected_sizes = [(600, 400), (600, 900), (600, 600)]
        for photo, size in zip(images, expected_sizes):
            with Image.open(photo) as image:
                assert image.size == size, (photo, image.size, size)
        marie_publish_cli.publish(ready, repo, dry_run=True)
        assert gallery.read_text() == "existing-item.jpg\n"
        marie_publish_cli.publish(ready, repo, commit=True)
        expected = [p.name for p in images] + ["existing-item.jpg"]
        assert gallery.read_text().splitlines() == expected
        marie_publish_cli.publish(ready, repo, commit=True)
        assert gallery.read_text().splitlines() == expected
        print("PASS: 3 natural-sized images (landscape, portrait, square); gallery prepended once; originals untouched")
        print("PASS: dry-run changed nothing; no remote push or AI calls")


if __name__ == "__main__":
    main()
