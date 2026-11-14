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
pip install -e .
sudo apt install ffmpeg  # Debian/Ubuntu/Linux Mint
```

1. Put N original JPG/PNG/WebP photographs in `incoming/`. Inputs are processed
   in filename order; prefix names 01, 02 etc. to specify sequence.
2. Prepare either portrait Reel assets or square web assets. Each creates a
   numbered JPG export, an existing-format JSON sidecar, and a simple YAML
   summary with alt text, description, caption and keywords.

```bash
marie-prepare incoming --output ready/reel --format reel
marie-prepare incoming --output ready/web --format web
# Fast/offline image-only mode; use on a separate, fresh output directory:
marie-prepare incoming --output ready/draft --format reel --no-describe
```

Preset dimensions are Reel **1080x1920 (9:16)** and web **1200x1200 (1:1)**.
Use `--width` and `--height` for other webpage sizes. Default
`--fit contain` preserves each complete photograph against a blurred
background; `--fit cover` fills the frame by cropping (verify that products
are not cut off). EXIF orientation is applied. Source images are never
changed. Generated JSON and YAML sidecars live in the output folder; the
existing image-description pipeline and `marie-parie` prompt are reused.
`--overwrite` refreshes existing exports and descriptions. Without it,
previously generated images and JSON sidecars are reused.

3. Make a silent MP4 using gentle alternating zoom movement and crossfades:

```bash
marie-video ready/reel --output ready/marie-parie.mp4 --seconds 15 --fade 0.6
marie-video ready/reel --output ready/static.mp4 --seconds 20 --movement none
```

The duration is the **entire MP4**, not seconds per photograph. The video is
H.264/yuv420p, with no soundtrack; suitable to review before sharing or
publishing. The existing `instagram-reel` command can publish this MP4
separately after you prepare a matching JSON video sidecar and provide an
accessible video URL or ngrok tunnel.

**Review before posting:** AI captions may describe a product inaccurately.
Check each alt text and caption, especially names, materials, and availability.
Neither command posts to social media. No image or video creation modifies the
original incoming photographs.
