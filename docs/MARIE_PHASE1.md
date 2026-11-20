# Marie-Parie Phase 1 trial (supervised, local)

Run these commands on the machine with the Python repos and a **local**
`marie-parie-web` checkout. All processing is local until an explicit Git
push. Do not run the first trial against the real staging repository.

## Setup

```bash
cd ~/src/repos/image_description
git switch feature/marie-parie-media-workflow
python3 -m venv .venv
source .venv/bin/activate
pip install -e ../galet -e ".[dev]"  # one pip invocation for both editable projects
python scripts/marie_phase1_smoke.py
pytest -q tests/test_marie_media.py tests/test_marie_publish.py
```

The smoke test makes three synthetic pictures and a temporary local Git
repository. It checks 600 × 400 JPEG conversion, a no-write dry run, insertion
at the top of `gallery.txt`, a local commit, and idempotent replay. It does
**not** contact the AI provider, GitHub or Cloudflare.

## Real images

Place several representative photographs in `incoming/` (include landscape,
portrait and square; JPG and PNG). Use a **fresh** output folder when changing
ratio, size, or fit, otherwise supply `--overwrite`:

```bash
marie-resize incoming --output ready/web --format web --fit contain
marie-resize incoming --output ready/reel --format reel --fit contain
```

Default web is **600 × 400 (3:2 landscape)**, Reel is **1080 × 1920
(9:16 portrait)**; both output JPEGs. `--aspect-ratio 4:5`,
`--width 800 --aspect-ratio 3:2`, or `--width 600 --height 400`
can override the target. `--fit cover` crops, while `--fit contain`
preserves the image against a white background.

Review exported images first. Then:

```bash
marie-describe ready/web
marie-publish ready/web --repo ../marie-parie-web --dry-run
```

For an actual staging update, ensure the website checkout is on a
**clean** `staging` branch and up to date with `origin/staging`.
Review `gallery.txt` and source images, then run:

```bash
marie-publish ready/web --repo ../marie-parie-web --commit
git -C ../marie-parie-web show --stat --oneline HEAD
git -C ../marie-parie-web push origin staging
```

The publisher copies JPG/PNG files to `assets/New Arrivals/` and places
their filenames first in `gallery.txt`, preserving the rest of the list.
Only `git push` triggers remote deployment. The repository's `main`
branch is never touched by this workflow.

Optional Reel video, assuming FFmpeg is installed:

```bash
marie-video ready/reel --output ready/reel.mp4 --seconds 20
```

The video remains local until published separately.

## Known MVP constraints

- Filenames remain numbered source identifiers. AI-generated filenames are
  deferred until there is an approval/renaming step.
- `marie-publish` requires a clean local `staging` checkout. It does not
  fetch, merge, or resolve remote changes.
- Reprocessing the same file with different settings needs `--overwrite`
  or a fresh output folder.
- Review AI descriptions and visual crops; automated tests do not validate
  descriptive accuracy.
- For additional staff or unattended publishing, introduce authenticated
  MCP tools and explicit authorization as Phase 2.
