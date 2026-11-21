# Command inventory (develop baseline, November 2026)

The project uses a `src/` package layout. Install `pip install -e .` to get the
registered commands. Other modules can be run with `python -m image_description.cli.MODULE`.
The descriptions below are an orientation to the existing modules; check `--help`
for current options. Scripts under `scripts/` are mostly maintenance utilities.

## Main commands and CLI modules

| Command/module | Purpose |
|---|---|
| `image` / `image_cli.py` | Describe photos using a vision model, create JSON metadata sidecars; embed IPTC; backfill EXIF dates; list publish-ready images. |
| `post-builder` / `post_builder_cli.py` | Export metadata from JSON sidecars to YAML or Markdown. |
| `blog-post-builder` / `blog_post_builder_cli.py` | Create a Jekyll blog post from metadata. |
| `gallery-builder` / `gallery_builder_cli.py` | Build gallery data from image metadata. |
| `gallery_promote_cli.py` | Promote/arrange gallery images. |
| `social_post_builder_cli.py` | Compose social posts using structured sidecar data and platform prompts. |
| `facebook` / `facebook_cli.py` | Publish a photograph to a Facebook Page. |
| `instagram` / `instagram_cli.py` | Publish a photo and read Instagram comments. |
| `instagram-reel` / `instagram_reels_cli.py` | Upload an existing MP4 to Instagram as a Reel (does not create video). |
| `instagram_comments.py` | Instagram comments retrieval helper. |
| `instagram_ngrok.py` | Temporary HTTPS hosting helper for Instagram photos. |
| `instagram_video_ngrok.py` | Temporary HTTPS hosting helper for Reel video, including range requests. |
| `mastodon_cli.py` | Publish/manage Mastodon posts. |
| `pixelfed_cli.py` | Publish/manage Pixelfed posts. |
| `bluesky_cli.py` | Publish/manage Bluesky posts. |
| `tumblr_cli.py` | Publish/manage Tumblr posts. |
| `medium_cli.py` | Publish/manage Medium content. |
| `wpcom_cli.py` | Work with WordPress.com posts. |
| `wp_import_cli.py` | Import WordPress content. |
| `ratings_cli.py` | Scan social engagement and calculate scores. |
| `scan-notes` / `scan_notes_cli.py` | Turn scans of notes into Obsidian Markdown. |

## Maintenance and migration scripts

| Script | Purpose |
|---|---|
| `backfill_image_fields.py` | Backfill missing image fields in existing sidecars. |
| `build_gallery.py` | Wrapper for gallery builder. |
| `copy_primaries.py` | Copy selected primary photographs. |
| `delete_wp_posts.py` | Delete WordPress posts; destructive, inspect first. |
| `export_mastodon_full.py` | Export Mastodon content. |
| `fetch_mastodon_export.py` | Retrieve Mastodon export data. |
| `fetch_mastodon_posts.py` | Fetch Mastodon posts. |
| `fetch_social_export.py` | Retrieve social-platform export data. |
| `match_mastodon_sidecars.py` | Match fetched Mastodon posts back to local sidecars. |
| `migrate_sidecar_field.py` | Migrate a field across sidecars. |
| `organize_notes.py` | Organize scanned-note files. |
| `post_builder.py` | Legacy/wrapper post-building script. |
| `publish_all.sh` | Batch publishing orchestration; check before running. |
| `scan_duplicates.py` | Find duplicate images or sidecars. |
| `update_publish_history.py` | Update sidecar publication history. |
| `_test_max_side.py` | Manual test of image downscaling. |
| `_test_small.py` (root) | Small ad hoc test. |

## Marie-Parie media workflow (new, separate commands)

Install the package and FFmpeg (for videos):

```bash
pip install -e ../galet -e ".[dev]"
sudo apt install ffmpeg  # Debian/Ubuntu/Linux Mint
```

1. Put N original JPG/PNG/WebP photographs in `incoming/`. Inputs are processed
   in filename order; prefix names 01, 02 etc. to specify sequence.
2. Resize the images independently. This creates numbered JPEGs and YAML stubs;
   `marie-describe` optionally generates JSON descriptions and fills in the YAML.

```bash
marie-resize incoming --output ready/reel --format reel
marie-resize incoming --output ready/web --format web
marie-describe ready/web  # optional AI descriptions and YAML sidecars
marie-rename ready/web    # preview approved names; add --apply to rename files and sidecars
```

The web preset is a **600×900 bounding box (2:3 portrait)** with **natural** fit:
the source aspect ratio is preserved, with no added margins or cropping. For
example, a 1200×800 landscape source becomes 600×400, while a 600×900 portrait
remains 600×900. The website gallery centers each image using CSS. The Reel
preset is **1080×1920 (9:16)** with **contain** fit by default, because
slideshow images must have identical dimensions.

`--fit auto` (default) selects natural for web or contain for Reel.
`--fit natural` preserves source proportions within the bounding box.
`--fit contain` adds white margins to create an exact-sized frame.
`--fit cover` crops the image to fill an exact-sized frame.
Use `--aspect-ratio 4:5` to change the bounding ratio, or provide
`--width` and `--height` together to specify exact bounds.
All exports are JPEGs, including PNG inputs, with EXIF orientation applied.
Source images remain unchanged. Use `--overwrite` to regenerate existing
images and associated metadata.

3. Make a silent MP4 using gentle alternating zoom movement and crossfades:

```bash
marie-video ready/reel --output ready/marie-parie.mp4 --seconds-per-image 2 --fade 0.4
marie-video ready/reel --output ready/static.mp4 --seconds-per-image 2 --movement none
```

The default is **2 seconds per photo** with overlapping **0.4-second
crossfades**, subtle zoom/pan and a **720×1280 (9:16) H.264** output. The
duration is `N × seconds-per-image − (N−1) × fade`, so five photos produce
about 8.4 seconds. Use `--seconds` only to specify a fixed total duration.
Mixed input dimensions and proportions are fitted automatically to the
video frame on white; there is no soundtrack. Review before sharing. The existing `instagram-reel` command can publish this MP4
separately after you prepare a matching JSON video sidecar and provide an
accessible video URL or ngrok tunnel.

**Review before posting:** AI captions may describe a product inaccurately.
Check each alt text and caption, especially names, materials, and availability.
Neither command posts to social media. No image or video creation modifies the
original incoming photographs.

## Publish prepared web photos to the staging gallery

The separate `marie-publish` command copies the numbered JPG exports into
`assets/New Arrivals/` in a **clean local checkout of `marie-parie-web` on
branch `staging`**. It adds their exact filenames at the *top* of
`assets/New Arrivals/gallery.txt`, preserving existing order and avoiding
repeated entries. It refuses to overwrite different existing files.

```bash
marie-resize incoming --output ready/web --format web
marie-publish ready/web --repo ../marie-parie-web --dry-run
# Copy and commit to staging after reviewing the dry-run:
marie-publish ready/web --repo ../marie-parie-web --commit
# After reviewing the local commit, publish the staging branch:
git -C ../marie-parie-web push origin staging
```

The command never modifies production (`main`), and never commits or pushes
unless explicitly requested. `--push` requires `--commit`. Follow the
website repository's feature-branch policy for code changes; this utility is
for routine staged content delivery only. The default gallery uses literal
filenames, not descriptions. AI-generated filenames can be added later after
human review. Reel creation remains a separate `marie-video` step.

### Configurable output ratio and fit

Web images default to `--fit natural`, with a **600×900 bounding box**. These
do **not** all become 600×900; their original aspect ratios are preserved.

```bash
marie-resize incoming --output ready/web --format web                    # natural, no padding
marie-resize incoming --output ready/web-padded --format web --fit contain
marie-resize incoming --output ready/web-crop --format web --fit cover
marie-resize incoming --output ready/other --format web --aspect-ratio 4:5 --width 800
marie-resize incoming --output ready/reel --format reel                  # 1080x1920 contain
```

Use a fresh output folder or `--overwrite` when changing dimensions or fit.
For a Reel use contain/cover, not natural, to keep identical frames.

## Separation of responsibilities

- `marie-resize` is image-only, requires no AI credentials and preserves the originals. Default web output: 600×900 maximum bounds and natural proportions, configurable with `--aspect-ratio`, `--width`, `--height`, and `--fit natural|contain|cover`. It writes an initially empty YAML summary.
- `marie-describe ready/web` adds AI-generated JSON metadata, keyword lists, and reviewable YAML with filename suggestions. Use `--overwrite` to regenerate descriptions.
- `marie-rename ready/web` previews safe, manually reviewed names from YAML. `--apply` renames each prepared JPEG, JSON and YAML together; run before publishing.
- `marie-publish ready/web --repo ../marie-parie-web` updates the local staging gallery, separately from description and resizing.
- `marie-video ready/reel --output ready/reel.mp4` produces an optional slideshow.
- `marie-prepare` is retained as a compatibility command for the combined operation.

No command above deploys to `main` automatically.

### Review names before publishing

Check and optionally edit `suggested_filename` in each YAML sidecar. Then:

```bash
marie-rename ready/web
marie-rename ready/web --apply
marie-publish ready/web --repo ../marie-parie-web --dry-run
```

The rename command never contacts the AI, edits the original incoming folder,
or writes to the website repo. It rejects collisions, missing sidecars and
invalid filenames; `--apply` must be supplied explicitly. See
`docs/MARIE_PHASE1.md` for the full process.

### Publish into a selected website collection

Provide the folder within `assets/` using `--collection`. For example:

```bash
marie-publish ready/web --repo ../marie-parie-web --collection "Button_s Bougie" --dry-run
marie-publish ready/web --repo ../marie-parie-web --collection "assets/Button_s Bougie" --commit
```

Both collection spellings identify the existing `assets/Button_s Bougie/`
folder. Only JPEG/PNG files are copied, and the selected `gallery.txt` gets
new filenames at its top. Other collections are unchanged. The repository
must be a clean local `staging` checkout, and pushing remains explicit.
