# Image Description Tools

This project helps you generate rich, consistent metadata for your photographs using OpenAI vision models, and then export that metadata into formats you can reuse (JSON sidecars, Markdown/YAML exports, and Jekyll blog posts).

CLI usage (recommended)

- To see the available options for the CLIs use the module invocation (recommended when running from the repo):

  - python -m src.image_description.cli.image_cli --help
  - python -m src.image_description.cli.post_builder_cli --help
  - python -m src.image_description.cli.blog_post_builder_cli --help

- Target-state console scripts (when the package is installed) would be:

  - image
  - post-builder
  - blog-post-builder

  These console scripts map to the same CLI entry points as the module invocations above.

- Developer wrapper scripts in the repo (for quick local execution) are the top-level Python files:

  - image.py
  - post_builder.py
  - blog_post_builder.py

  Relationship and guidance:
  - The wrapper scripts (image.py, post_builder.py, blog_post_builder.py) are convenience entry points for running the tools directly from the project directory during development. They call into the same library code as the CLI entry points but are not the installed console scripts.
  - For local development, you can run the wrapper scripts directly (e.g. `python3 image.py describe ...`). For a more reproducible invocation (and to exercise the package entry points), prefer the module form (python -m src.image_description.cli.<cli_module> ...).
  - When the package is installed (pip install -e . or a real distribution), the package will provide console scripts named `image`, `post-builder`, and `blog-post-builder` which are equivalent to the module invocations.

There are **three** main scripts:

- `image.py` – scans images, calls the OpenAI API, and writes JSON sidecar files; can also embed metadata back into the image files via `exiftool`.
- `post_builder.py` – reads one JSON sidecar and exports the metadata as **Markdown** or **YAML** (optionally including a “prompt for social post”).
- `blog_post_builder.py` – reads one JSON sidecar and writes a **Jekyll** post into a GitHub Pages repo (`_posts/`), optionally copying the image into the repo under `/assets/...`.

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

`image.py` reads and writes **IPTC metadata** (title/description/keywords) by calling the command-line tool **ExifTool**.

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

> Note: If you only use `image.py describe` (JSON sidecars) you can get pretty far without ExifTool, but `image.py embed` requires it.

---

## 2. `image.py` – generate and embed metadata

`image.py` is the main script that talks to the OpenAI API and manages JSON sidecar files.

It has two subcommands:

- `describe` – generate JSON metadata sidecars for images.
- `embed` – write metadata from JSON back into the image files.

### 2.1 Command-line interface

The general form is:

```bash
python3 image.py describe PATH [--preset PRESET]
python3 image.py embed DIRECTORY
```

- `PATH` can be a single image file or a directory of images.
- `DIRECTORY` must be a directory containing images and their JSON sidecars.
- `PRESET` selects which prompt preset to use (see below).

### 2.2 Basic usage: describe a directory of images

From the project root:

```bash
cd ~/src/repos/image_description
source .venv/bin/activate

python3 image.py describe /path/to/your/images
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

`image.py` uses named prompt presets to control how the model describes the image. Two presets are currently defined:

- `orwell_basic` – simple, clear description and keywords.
- `orwell_ways_of_seeing` – inspired by John Berger’s "Ways of Seeing", with a focus on what the image is trying to say.

You can choose a preset with `--preset`:

```bash
python3 image.py describe /path/to/images --preset orwell_basic
python3 image.py describe /path/to/images --preset orwell_ways_of_seeing
```

If you omit `--preset`, the default is `orwell_ways_of_seeing`.

### 2.4 Describe a single image

```bash
python3 image.py describe /path/to/images/2A9A8326.jpg
```

This will create `/path/to/images/2A9A8326.json`.

### 2.5 Embed metadata back into images

Once you’re happy with the JSON sidecars, you can write the metadata back into the image files using `embed`:

```bash
python3 image.py embed /path/to/images
```

For each image with a matching JSON file, the script will:

- Set the IPTC title from `title` (or `original_title` if `title` is missing).
- Set the IPTC description from `enhanced_description` (or `original_description` if missing).
- Set IPTC keywords from the `keywords` list.

This uses `exiftool` under the hood.

> Tip: keep backups of your original images (or work on copies) when experimenting with metadata embedding.

---

## 3. `post_builder.py` – export metadata as YAML or Markdown

`post_builder.py` reads one of the JSON sidecar files and produces a clean YAML or Markdown file containing the metadata.

This is useful for:

- Feeding metadata into a static site generator.
- Keeping a text archive of your images.
- Preparing material for later, more tailored prompts.

### 3.1 YAML output

To create a YAML file next to the JSON:

```bash
python3 post_builder.py \
  /path/to/images/2A9A8326.json \
  --format yaml
```

This will print YAML to stdout (or you can use `--output` to write a file).

The YAML includes fields like:

```yaml
title: Three silos in snow
original_title: Three silos in snow
original_description: "Three grain silos in a snowy field near the farm..."
visually_challenged_description: "The image shows three large, cylindrical metal silos..."
enhanced_description: "This monochromatic image captures the stark beauty..."
keywords:
  - landscape
  - snow
  - nature
hashtags: "#landscape #snow #nature"
image: /path/to/images/2A9A8326.jpg   # if the script can guess the image path
prompt_for_social_post: |
  Act as a thoughtful artist and writer...
```

### 3.2 Markdown output

To create Markdown:

```bash
python3 post_builder.py /path/to/images/2A9A8326.json
```

This prints Markdown to stdout (or use `--output` to write a file).

If the script can guess the image path (by replacing `.json` with `.jpg`/`.jpeg`/`.png`), it will include a simple image reference near the top.

### 3.3 Explicit image path (optional)

If your image lives somewhere else or has a different name, you can pass it explicitly:

```bash
python3 post_builder.py \
  /path/to/images/2A9A8326.json \
  --format yaml \
  --image-path /images/2021/2021Jan/2A9A8326.jpg
```

You can also omit the social-post prompt:

```bash
python3 post_builder.py /path/to/images/2A9A8326.json --no-prompt
```

---

## 4. `blog_post_builder.py` – write a Jekyll post into a repo

`blog_post_builder.py` takes a JSON sidecar and writes a Jekyll-compatible Markdown post into a GitHub Pages repo.

It can also copy the image into the repo if the image path you use is under `/assets/...`.

### 4.1 Command-line interface

```bash
python3 blog_post_builder.py JSON_PATH --out-root /path/to/jekyll/repo [options]
```

Options:

- `--out-root` (required): root of the Jekyll/GitHub Pages repo. The post is written to `<out-root>/_posts/`.
- `--date`: date used in the post front matter. Accepts `YYYY-MM-DD` or `YYYY-MM-DD HH:MM:SS -ZZZZ`.
  - If omitted, it uses today’s date.
  - If you pass only `YYYY-MM-DD`, the script writes `10:00:00 -0500` as the time/zone.
- `--image`: web path used in front matter and body (example: `/assets/images/foo.jpg`).
  - If omitted, it guesses `/assets/images/<filename>` if it finds an image next to the JSON.
  - If it starts with `/assets/`, the script will copy the source image (next to the JSON, same basename) into `<out-root>/assets/...`.
- `--categories`: optional list of categories.

### 4.2 Example

```bash
python3 blog_post_builder.py \
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

The project includes a `.vscode/launch.json` with helpful run configurations.

For `image.py` there are four presets:

- **Python: Image Describe (Linux)**

  ```jsonc
  {
    "name": "Python: Image Describe (Linux)",
    "type": "python",
    "request": "launch",
    "program": "${workspaceFolder}/image.py",
    "args": [
      "describe",
      "/home/junwin/nucshare/2025/Brisbane/Output/web",
      "--preset",
      "orwell_ways_of_seeing"
    ],
    "console": "integratedTerminal",
    "justMyCode": true,
    "env": {
      "CREDENTIAL_PATH": "/home/junwin/credential"
    }
  }
  ```

- **Python: Image Embed (Linux)**

  ```jsonc
  {
    "name": "Python: Image Embed (Linux)",
    "type": "python",
    "request": "launch",
    "program": "${workspaceFolder}/image.py",
    "args": [
      "embed",
      "/home/junwin/nucshare/2025/Brisbane/Output/web"
    ],
    "console": "integratedTerminal",
    "justMyCode": true,
    "env": {
      "CREDENTIAL_PATH": "/home/junwin/credential"
    }
  }
  ```

- **Python: Image Describe (Windows)**

  ```jsonc
  {
    "name": "Python: Image Describe (Windows)",
    "type": "python",
    "request": "launch",
    "program": "${workspaceFolder}/image.py",
    "args": [
      "describe",
      "E:/photography/work/2024/2024Aug_wales/Output",
      "--preset",
      "orwell_ways_of_seeing"
    ],
    "console": "integratedTerminal",
    "justMyCode": true,
    "env": {
      "CREDENTIAL_PATH": "C:/Users/junwin/credential"
    }
  }
  ```

- **Python: Image Embed (Windows)**

  ```jsonc
  {
    "name": "Python: Image Embed (Windows)",
    "type": "python",
    "request": "launch",
    "program": "${workspaceFolder}/image.py",
    "args": [
      "embed",
      "E:/photography/work/2024/2024Aug_wales/Output"
    ],
    "console": "integratedTerminal",
    "justMyCode": true,
    "env": {
      "CREDENTIAL_PATH": "C:/Users/junwin/credential"
    }
  }
  ```

You can edit the paths in `"args"` to point at whatever directory you are currently working on.

There are also two presets for `post_builder.py` that operate on the currently open file:

- **Python: Build post from JSON (YAML)**
- **Python: Build post from JSON (Markdown)**

---

## 6. Typical workflow

A common end-to-end workflow might look like this:

1. **Prepare credentials and environment**
   - Set up `oaicred.json` and `config.json`.
   - Install dependencies with `pip install -r requirements.txt`.
   - Ensure `exiftool` is installed.

2. **Generate metadata JSON**

   ```bash
   python3 image.py describe /home/junwin/nucshare/2021/2021Jan/Output/web
   ```

3. **Review and tweak JSON**
   - Open the generated `.json` files.
   - Optionally adjust `title`, `enhanced_description`, or `keywords` by hand.

4. **Export YAML or Markdown**

   ```bash
   python3 post_builder.py /home/junwin/nucshare/2021/2021Jan/Output/web/2A9A8326.json --format yaml
   # or
   python3 post_builder.py /home/junwin/nucshare/2021/2021Jan/Output/web/2A9A8326.json
   ```

5. **(Optional) Embed metadata back into images**

   ```bash
   python3 image.py embed /home/junwin/nucshare/2021/2021Jan/Output/web
   ```

6. **Build a Jekyll blog post**

   ```bash
python3 blog_post_builder.py \
     /home/junwin/nucshare/2025/Brisbane/Output/web/R0004509.json \
     --date 2025-12-24 \
     --out-root /home/junwin/src/repos/junwin.github.io \
     --image /assets/images/junwin/2025/R0004509.jpg \
     --categories photography
   ```

---

## 7. Notes and future ideas

- The current prompts are designed to be general and reusable. You can add new presets in `image.py` or separate scripts for specific platforms.
- `post_builder.py` intentionally exports **metadata + optional prompt**, so you can keep your archive clean and apply prompts on demand.
- `blog_post_builder.py` is intentionally simple: it turns one JSON sidecar into one Jekyll post and (optionally) copies the image into your site repo.
