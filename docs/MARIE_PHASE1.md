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
repository. It checks natural (unpadded) JPEG conversion, a no-write dry run, insertion
at the top of `gallery.txt`, a local commit, and idempotent replay. It does
**not** contact the AI provider, GitHub or Cloudflare.

## Real images

Place several representative photographs in `incoming/` (include landscape,
portrait and square; JPG and PNG). Use a **fresh** output folder when changing
ratio, size, or fit, otherwise supply `--overwrite`:

```bash
marie-resize incoming --output ready/web --format web
marie-resize incoming --output ready/reel --format reel --fit contain
```

Default web has **600 × 900 maximum bounds (2:3 portrait)** with
`--fit natural`: image proportions are preserved, and the website supplies
any side whitespace using its own CSS. Thus landscape images may be 600×400,
while 2:3 portrait images will be 600×900. **No white margins are added.**
The Reel preset is **1080 × 1920 (9:16 portrait)** with fixed-size
`--fit contain` by default. Both modes export JPEGs.
Use `--fit cover` to crop, or `--fit contain` to intentionally pad to a fixed
canvas. Dimensions can be overridden using `--aspect-ratio 4:5`,
`--width 800 --aspect-ratio 2:3`, or `--width 600 --height 900`.

Review exported images first. Then:

```bash
marie-describe ready/web
# Check captions and correct suggested_filename in each YAML as needed:
marie-rename ready/web              # review changes only
marie-rename ready/web --apply      # explicitly apply reviewed filenames
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

## AI naming and annotations

`marie-describe ready/web` uses the Marie-Parie Galet preset to create factual
alt text, merchandise-focused descriptions, captions and an AI suggested name
for the main item. Keywords are separate YAML entries. The command also
normalizes old JSON sidecars with comma-packed keywords **without another AI
call** (unless `--overwrite` is specified).

Before publishing, inspect each `.yaml` file. To approve or correct a suggested
name, edit its `suggested_filename:` value. Keep the numeric sequence prefix
and use lowercase hyphenated words ending in `.jpg`, for example
`004-dark-floral-button-tunic.jpg`.

```bash
marie-rename /home/junwin/Pictures/ready-mp/web
# Preview checks every file and makes NO changes.

marie-rename /home/junwin/Pictures/ready-mp/web --apply
# Renames the prepared JPEG, JSON and YAML together; never touches incoming.
```

The renamer refuses missing sidecars, name collisions, unsafe paths or a changed
sequence prefix. It updates `image_filename`, `image_relative_path` and
`suggested_filename` in JSON, and `image` and `suggested_filename` in YAML
while preserving the original source name and manual caption edits. Run it
**before** `marie-publish` so the website gallery receives final filenames.
Applying is intentionally explicit; a dry run is the default.

If older JSON already exists without a name suggestion, run
`marie-describe ready/web --overwrite` to request a new suggestion (and incur
new AI calls). Once an image has been renamed, do not rerun `marie-resize`
into the same output folder; use a fresh directory for another batch.

## System prerequisites

AI descriptions need `exiftool` in the PATH. On Linux Mint:

```bash
sudo apt install libimage-exiftool-perl
```

Reels additionally need `ffmpeg`.
