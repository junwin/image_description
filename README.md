# Image Description Tools

This project helps you generate rich, consistent metadata for your photographs using OpenAI vision models, and then export that metadata as JSON, YAML, or Markdown for use in editors, blogs, or archives.

There are two main scripts:

- `image.py` – scans images, calls the OpenAI API, and writes JSON sidecar files; can also embed metadata back into the image files via `exiftool`.
- `post_builder.py` – reads one of those JSON files and produces a clean YAML or Markdown file with just the metadata.

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

### 1.3 exiftool

To read and write IPTC metadata, the scripts use `exiftool`.

Install it and make sure it’s on your `PATH`:

- **Ubuntu/Debian**:

  ```bash
  sudo apt-get install exiftool
  ```

- **macOS** (Homebrew):

  ```bash
  brew install exiftool
  ```

- **Windows**:

  - Download from https://exiftool.org/
  - Add the `exiftool.exe` location to your `PATH`.

Check it works:

```bash
exiftool -ver
```

---

## 2. `image.py` – generate and embed metadata

`image.py` is the main script that talks to the OpenAI API and manages JSON sidecar files.

It has two subcommands:

- `describe` – generate JSON metadata sidecars for images.
- `embed` – write metadata from JSON back into the image files.

### 2.1 Basic usage: describe a directory of images

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

### 2.2 Prompt presets

`image.py` uses named prompt presets to control how the model describes the image. Two presets are currently defined:

- `orwell_basic` – simple, clear description and keywords.
- `orwell_ways_of_seeing` – inspired by John Berger’s "Ways of Seeing", with a focus on what the image is trying to say.

You can choose a preset with `--preset`:

```bash
python3 image.py describe /path/to/images --preset orwell_basic
python3 image.py describe /path/to/images --preset orwell_ways_of_seeing
```

If you omit `--preset`, a default is used (currently `orwell_ways_of_seeing`).

### 2.3 Describe a single image

```bash
python3 image.py describe /path/to/images/2A9A8326.jpg
```

This will create `/path/to/images/2A9A8326.json`.

### 2.4 Embed metadata back into images

Once you’re happy with the JSON sidecars, you can write the metadata back into the image files using `embed`:

```bash
python3 image.py embed /path/to/images
```

For each image with a matching JSON file, the script will:

- Set the IPTC title from `title` (or `original_title` if `title` is missing).
- Set the IPTC description from `enhanced_description` (or `original_description` if missing).
- Set IPTC keywords from the `keywords` list.

This uses `exiftool` under the hood.

> **Tip:** It’s a good idea to keep backups of your original images, or work on copies, when experimenting with metadata embedding.

---

## 3. `post_builder.py` – export metadata as YAML or Markdown

`post_builder.py` reads one of the JSON sidecar files and produces a clean YAML or Markdown file containing just the metadata.

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

This will create:

```text
/path/to/images/2A9A8326.yaml
```

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
  - monochrome
  - winter
  - silos
  - gray
  - contrast
  - industrial
hashtags: "#landscape #snow #nature #monochrome #winter #silos #gray #contrast #industrial"
image: /path/to/images/2A9A8326.jpg   # if the script can guess the image path
```

### 3.2 Markdown output

To create a Markdown file next to the JSON:

```bash
python3 post_builder.py /path/to/images/2A9A8326.json
```

This will create:

```text
/path/to/images/2A9A8326.md
```

The Markdown includes sections for:

- Title
- Original notes (title + description)
- Enhanced description
- Description for the visually challenged
- Keywords
- Hashtags

If the script can guess the image path (by replacing `.json` with `.jpg`/`.jpeg`/`.png`), it will also include a simple image reference at the top.

### 3.3 Explicit image path (optional)

If your image lives somewhere else or has a different name, you can pass it explicitly:

```bash
python3 post_builder.py \
  /path/to/images/2A9A8326.json \
  --format yaml \
  --image-path /images/2021/2021Jan/2A9A8326.jpg
```

---

## 4. VS Code integration

The project includes a `.vscode/launch.json` with helpful run configurations:

- Run `image.py` on a Linux image directory.
- Run `image.py` on a Windows image directory.
- Run `post_builder.py` on the currently open JSON file to produce YAML or Markdown next to it.

To use them:

1. Open `~/src/repos/image_description` in VS Code.
2. Open the Run and Debug panel.
3. Choose the configuration you want (e.g. "Python: Image (Linux)" or "Python: Build post from JSON (YAML)").
4. Start debugging.

---

## 5. Typical workflow

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

6. **(Later) Tailor prompts for social posts or blogs**
   - Use the JSON/YAML/Markdown as input to a separate script or tool that applies your preferred writing prompt (e.g. for Mastodon, Tumblr, Bluesky).

---

## 6. Notes and future ideas

- The current prompts are designed to be general and reusable. You can add new presets in `image.py` or separate scripts for specific platforms.
- `post_builder.py` intentionally outputs **only metadata**, not prompts, so you can keep your archive clean and apply prompts on demand.
- If you want to support more image formats or metadata fields, both scripts are structured to make that relatively easy.
