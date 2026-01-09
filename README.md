# Image Description Tools

This project helps you generate rich, consistent metadata for your photographs using OpenAI vision models, and then export that metadata into formats you can reuse (JSON sidecars, Markdown/YAML exports, and Jekyll blog posts).

## CLI usage (recommended)

To see the available options for the CLIs use the module invocation (recommended when running from the repo):

- `python -m src.image_description.cli.image_cli --help`
- `python -m src.image_description.cli.post_builder_cli --help`
- `python -m src.image_description.cli.blog_post_builder_cli --help`

Target-state console scripts (when the package is installed) would be:

- `image`
- `post-builder`
- `blog-post-builder`

These console scripts map to the same CLI entry points as the module invocations above.

---

There are **three** main tools:

- `image_cli` – scans images, calls the OpenAI API, and writes JSON sidecar files; can also embed metadata back into the image files via `exiftool`.
- `post_builder_cli` – reads one JSON sidecar and exports the metadata as **Markdown** or **YAML** (optionally including a “prompt for social post”).
- `blog_post_builder_cli` – reads one JSON sidecar and writes a **Jekyll** post into a GitHub Pages repo (`_posts/`), optionally copying the image into the repo under `/assets/...`.

The design is:

- **General, reusable metadata first** (titles, descriptions, keywords, hashtags).
- **Tailored prompts later**, when you actually want to create a social post or blog entry.

---

## 1. Prerequisites

### 1.1 Python and virtual environment

You’ll need Python 3.9+ installed.

From the project directory:

```bash
cd ~/src/repos/image_description
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The `requirements.txt` includes:

- `openai` – for calling the OpenAI API
- `Pillow` – for basic image handling
- `requests` – for HTTP calls (used by some helpers)

### 1.2 OpenAI credentials

The project expects your OpenAI API key to be stored in a small JSON file, and the path to that file is configured via `config.json`.

1. Create a credentials directory, for example:

   ```bash
   mkdir -p /home/junwin/credential
   ```

2. Create `oaicred.json` inside that directory:

   ```json
   {
     "openai_api_key": "sk-...your-key-here..."
   }
   ```

3. In the project root (`~/src/repos/image_description`), create a `config.json`:

   ```json
   {
     "credential_path": "/home/junwin/credential"
   }
   ```

On Windows you can either:

- Use a different `credential_path` in `config.json` (e.g. `"C:/Users/you/credential"`), or
- Set the `CREDENTIAL_PATH` environment variable to override the path.

The code will look for `oaicred.json` in `credential_path` and read `openai_api_key` from there.

You can also override the key via the `OPENAI_API_KEY` environment variable if you prefer.

### 1.3 exiftool (required for IPTC read/write)

The image tool reads and writes **IPTC metadata** (title/description/keywords) by calling the command-line tool **ExifTool**.

Install it and make sure it’s on your `PATH`:

- **Ubuntu/Debian**:

  ```bash
  sudo apt-get update
  sudo apt-get install exiftool
  ```

- **macOS** (Homebrew):

  ```bash
  brew install exiftool
  ```

- **Windows**:

  - Download from https://exiftool.org/
  - Put `exiftool(-k).exe` somewhere stable (e.g. `C:\Tools\exiftool\exiftool.exe`).
  - Add that folder to your `PATH`.

Verify it works:

```bash
exiftool -ver
```

> Note: If you only use `describe` (JSON sidecars) you can get pretty far without ExifTool, but `embed` requires it.

---

## 2. `image_cli` – generate and embed metadata

`image_cli` is the main tool that talks to the OpenAI API and manages JSON sidecar files.

It has two subcommands:

- `describe` – generate JSON metadata sidecars for images.
- `embed` – write metadata from JSON back into the image files.

### 2.1 Command-line interface

```bash
python -m src.image_description.cli.image_cli describe PATH [--preset PRESET]
python -m src.image_description.cli.image_cli embed DIRECTORY
```

- `PATH` can be a single image file or a directory of images.
- `DIRECTORY` must be a directory containing images and their JSON sidecars.
- `PRESET` selects which prompt preset to use (see below).

### 2.2 Basic usage: describe a directory of images

From the project root:

```bash
cd ~/src/repos/image_description
source .venv/bin/activate

python -m src.image_description.cli.image_cli describe /path/to/your/images
```

For each supported image file (e.g. `.jpg`, `.jpeg`, `.png`) in that directory, the script will:

1. Read existing IPTC metadata (title, description, keywords) if present.
2. Call the OpenAI vision model with a prompt preset.
3. Receive structured JSON from the model.
4. Merge existing and new keywords.
5. Write a JSON sidecar file next to the image, e.g.:

```
2A9A8326.jpg   -> 2A9A8326.json
```

The JSON structure includes:

```json
{
  "original_title": "...",              // from the image, if present
  "original_description": "...",        // from the image, if present
  "title": "...",                       // working title (starts as original_title)
  "visually_challenged_description": "...",
  "enhanced_description": "...",
  "keywords": ["...", "..."],
  "hashtags": "#tag1 #tag2 ..."
}
```

### 2.3 Prompt presets

Named prompt presets control how the model describes the image. Two presets are currently defined:

- `orwell_basic` – simple, clear description and keywords.
- `orwell_ways_of_seeing` – inspired by John Berger’s "Ways of Seeing", with a focus on what the image is trying to say.

Choose a preset with `--preset`:

```bash
python -m src.image_description.cli.image_cli describe /path/to/images --preset orwell_basic
python -m src.image_description.cli.image_cli describe /path/to/images --preset orwell_ways_of_seeing
```

If you omit `--preset`, the default is `orwell_ways_of_seeing`.

### 2.4 Describe a single image

```bash
python -m src.image_description.cli.image_cli describe /path/to/images/2A9A8326.jpg
```

This will create `/path/to/images/2A9A8326.json`.

### 2.5 Embed metadata back into images

Once you’re happy with the JSON sidecars, you can write the metadata back into the image files using `embed`:

```bash
python -m src.image_description.cli.image_cli embed /path/to/images
```

For each image with a matching JSON file, the script will:

- Set the IPTC title from `title` (or `original_title` if `title` is missing).
- Set the IPTC description from `enhanced_description` (or `original_description` if missing).
- Set IPTC keywords from the `keywords` list.

This uses `exiftool` under the hood.

> Tip: keep backups of your original images (or work on copies) when experimenting with metadata embedding.

---

## 3. `post_builder_cli` – export metadata as YAML or Markdown

`post_builder_cli` reads one of the JSON sidecar files and produces a clean YAML or Markdown file containing the metadata.

### 3.1 YAML output

```bash
python -m src.image_description.cli.post_builder_cli \
  /path/to/images/2A9A8326.json \
  --format yaml
```

This will print YAML to stdout (or you can use `--output` to write a file).

### 3.2 Markdown output

```bash
python -m src.image_description.cli.post_builder_cli /path/to/images/2A9A8326.json
```

This prints Markdown to stdout (or use `--output` to write a file).

### 3.3 Explicit image path (optional)

```bash
python -m src.image_description.cli.post_builder_cli \
  /path/to/images/2A9A8326.json \
  --format yaml \
  --image-path /images/2021/2021Jan/2A9A8326.jpg
```

You can also omit the social-post prompt:

```bash
python -m src.image_description.cli.post_builder_cli /path/to/images/2A9A8326.json --no-prompt
```

---

## 4. `blog_post_builder_cli` – write a Jekyll post into a repo

`blog_post_builder_cli` takes a JSON sidecar and writes a Jekyll-compatible Markdown post into a GitHub Pages repo.

### 4.1 Command-line interface

```bash
python -m src.image_description.cli.blog_post_builder_cli JSON_PATH --out-root /path/to/jekyll/repo [options]
```

### 4.2 Example

```bash
python -m src.image_description.cli.blog_post_builder_cli \
  /home/junwin/nucshare/2025/Brisbane/Output/web/R0004509.json \
  --date 2025-12-24 \
  --out-root /home/junwin/src/repos/junwin.github.io \
  --image /assets/images/junwin/2025/R0004509.jpg \
  --categories photography
```

Important detail (matches the current code):

- The **output filename** always uses **today’s date**, not `--date`:
- `<out-root>/_posts/<TODAY>-<json_basename>.md`

---

## 5. VS Code integration

Use the `.vscode/launch.json` configurations to run the tools via `module` (equivalent to `python -m ...`).

You can edit the paths in `"args"` to point at whatever directory you are currently working on.

There are also presets for `post_builder_cli` that operate on the currently open file:

- **Python: Build post from JSON (YAML) [module]**
- **Python: Build post from JSON (Markdown) [module]**
