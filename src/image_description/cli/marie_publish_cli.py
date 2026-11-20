"""Copy prepared Marie-Parie web images to a local staging checkout.

The command never changes the original images, never touches the production branch,
and requires an explicit --commit and --push for Git operations.
"""
import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args],
                          check=True, capture_output=True, text=True).stdout.strip()


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish(source_dir, repo, collection="New Arrivals", dry_run=False,
            commit=False, push=False):
    source_dir, repo = Path(source_dir).resolve(), Path(repo).resolve()
    if push and not commit:
        raise ValueError("--push requires --commit")
    if not source_dir.is_dir():
        raise ValueError("Prepared image folder not found")
    if not (repo / ".git").exists():
        raise ValueError("Website checkout must be a local Git repository")
    if git(repo, "branch", "--show-current") != "staging":
        raise ValueError("Website checkout must be on staging (never main)")
    if git(repo, "status", "--porcelain"):
        raise ValueError("Website checkout has uncommitted changes; use a clean checkout")
    target = repo / "assets" / collection
    gallery = target / "gallery.txt"
    if not gallery.is_file():
        raise ValueError(f"Gallery not found: {gallery}")
    sources = sorted((p for p in source_dir.iterdir()
                      if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                     key=lambda p: p.name.casefold())
    if not sources:
        raise ValueError("No prepared JPG/PNG images found")
    names = [p.name for p in sources]
    if len(set(n.casefold() for n in names)) != len(names):
        raise ValueError("Case-insensitive duplicate filenames in prepared images")
    current = [line.strip() for line in gallery.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    for source in sources:
        destination = target / source.name
        if destination.exists() and fingerprint(source) != fingerprint(destination):
            raise ValueError(f"Refusing to overwrite different image: {destination.name}")
    # New images in filename order first; older entries retain their relative order.
    seen = set()
    ordered = []
    for name in names + current:
        if name.casefold() not in seen:
            ordered.append(name)
            seen.add(name.casefold())
    print(f"Collection: {collection} | images: {len(sources)} | gallery entries: {len(ordered)}")
    for name in names:
        print(f"  {name}")
    if dry_run:
        print("Dry run: no files changed")
        return ordered
    for source in sources:
        destination = target / source.name
        if not destination.exists():
            shutil.copy2(source, destination)
    new_gallery = "\n".join(ordered) + "\n"
    if gallery.read_text(encoding="utf-8") != new_gallery:
        gallery.write_text(new_gallery, encoding="utf-8")
    if commit:
        paths = [str((target / name).relative_to(repo)) for name in names]
        paths.append(str(gallery.relative_to(repo)))
        git(repo, "add", "--", *paths)
        staged = subprocess.run(["git", "-C", str(repo), "diff", "--cached", "--quiet"], check=False)
        if staged.returncode == 1:
            git(repo, "commit", "-m", f"Add {len(names)} Marie-Parie {collection} images")
        elif staged.returncode != 0:
            raise ValueError("Unable to check staged changes")
        if push:
            git(repo, "push", "origin", "staging")
    return ordered


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Folder containing prepared web JPGs")
    parser.add_argument("--repo", type=Path, required=True, help="Local marie-parie-web checkout")
    parser.add_argument("--collection", default="New Arrivals")
    parser.add_argument("--dry-run", action="store_true", help="Preview without copying")
    parser.add_argument("--commit", action="store_true", help="Commit changes on staging")
    parser.add_argument("--push", action="store_true", help="Push staging (requires --commit)")
    args = parser.parse_args(argv)
    try:
        publish(args.input, args.repo, args.collection, args.dry_run, args.commit, args.push)
    except (ValueError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
