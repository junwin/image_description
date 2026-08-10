from typing import Any, Dict, List, Optional, Sequence, Union
import os
import shutil
from pathlib import Path

from ..paths import guess_image_path
from ..sidecar import Sidecar


PROMPT_TEMPLATE = """Act as a thoughtful artist and writer. Prepare a Mastodon, tumblr and bsky post for a new photograph I've taken.
please include suggested hashtags (5 or 6) and text for image description.

Consider John Berger's separation of a) what the image is  b) what is it trying to say  - I would like to swing the balance to what the image is trying to say.

Here is some metadata I already have  - this typically deals with what the image is.

Title: {title}
Original description: {original_description}
Enhanced description: {enhanced_description}
Image description: {image_description}
Keywords: {keywords}
Existing hashtags: {hashtags}

Please adhere strictly to the following style guidelines:
1. Follow George Orwell's rules: use short words, cut unnecessary words, and avoid jargon.
2. Use a minimalist and evocative style. Be precise, not flowery.
3. Adopt a reflective, understated tone. Avoid any boastfulness.
4. Use a two-sentence structure if possible: first a direct description, then a reflective observation.

Keep the final output concise.
"""


def build_prompt(meta: Dict[str, Any]) -> str:
    """Construct the social-post prompt text from metadata."""
    title = meta.get("title") or meta.get("original_title") or ""
    original_description = meta.get("original_description", "")
    enhanced_description = meta.get("enhanced_description", "")
    image_description = meta.get("image_description", "")
    keywords = meta.get("keywords", [])
    if isinstance(keywords, list):
        keywords_str = ", ".join(keywords)
    else:
        keywords_str = str(keywords)
    hashtags = meta.get("hashtags", "")

    return PROMPT_TEMPLATE.format(
        title=title,
        original_description=original_description,
        enhanced_description=enhanced_description,
        image_description=image_description,
        keywords=keywords_str,
        hashtags=hashtags,
    )


def prompt_from_sidecar_path(json_path: str) -> str:
    """Load a JSON sidecar and return the raw prompt text for social post.

    This helper provides a clean way for CLIs to request only the prompt
    text (without surrounding headings or code fences).
    """
    sidecar = Sidecar.load(json_path)
    meta = sidecar.to_dict()
    return build_prompt(meta)


def _yaml_escape(s: str) -> str:
    if any(c in s for c in [":", "-", "#", "{", "}", "[", "]", ",", "&", "*", "?", "|", ">", "%", "@", "`", '"', "'"]):
        return '"' + s.replace('"', '\\"') + '"'
    if "\\n" in s:
        return "|-\\n  " + s.replace("\\n", "\\n  ")
    return s


def _markdown_for_meta(
    meta: Dict[str, Any],
    image_path: Optional[str],
    include_prompt: bool = True,
    title_heading_level: int = 1,
) -> str:
    """Render a single sidecar meta object to markdown.

    title_heading_level controls the heading level used for the title line:
    - 1 (default): writes a top-level H1 (used for single-file output)
    - 2: writes an H2 (used when producing a multi-sidecar document where the
         overall document already has an H1)
    """
    title = meta.get("title") or meta.get("original_title") or "Untitled"
    original_title = meta.get("original_title", "")
    original_description = meta.get("original_description", "")
    enhanced_description = meta.get("enhanced_description", "")
    image_description = meta.get("image_description", "")
    keywords = meta.get("keywords", [])
    hashtags = meta.get("hashtags", "")

    lines: List[str] = []

    # Title
    lines.append(f"{('#' * title_heading_level)} {title}")
    lines.append("")

    # Image
    if image_path:
        lines.append(f"![{title}]({image_path})")
        lines.append("")

    # Use subheadings (one level deeper than the title)
    sub_h = title_heading_level + 1

    if original_title:
        lines.append(f"{('#' * sub_h)} Original title:")
        lines.append(f"{original_title}")
        lines.append("")

    if original_description:
        lines.append(f"{('#' * sub_h)} Original notes")
        lines.append(original_description)
        lines.append("")

    if enhanced_description:
        lines.append(f"{('#' * sub_h)} Enhanced description")
        lines.append(enhanced_description)
        lines.append("")

    if image_description:
        lines.append(f"{('#' * sub_h)} Image description")
        lines.append(image_description)
        lines.append("")

    if keywords:
        if isinstance(keywords, list):
            kw_str = ", ".join(keywords)
        else:
            kw_str = str(keywords)
        lines.append(f"{('#' * sub_h)} Keywords")
        lines.append(kw_str)
        lines.append("")

    if hashtags:
        lines.append(f"{('#' * sub_h)} Hashtags")
        lines.append(hashtags)
        lines.append("")

    if include_prompt:
        lines.append(f"{('#' * sub_h)} Prompt for social post")
        lines.append("```text")
        lines.append(build_prompt(meta))
        lines.append("```")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _yaml_for_meta(meta: Dict[str, Any], image_path: Optional[str], include_prompt: bool = True) -> str:
    """Render a single sidecar meta object to YAML (simple emitter).

    This intentionally keeps the previous minimal YAML style while avoiding
    additional dependencies.
    """
    lines: List[str] = []

    title = meta.get("title") or meta.get("original_title") or "Untitled"
    lines.append(f"title: {_yaml_escape(str(title))}")

    for key in [
        "original_title",
        "original_description",
        "image_description",
        "enhanced_description",
        "hashtags",
    ]:
        if key in meta and meta[key]:
            value = str(meta[key])
            if "\n" in value:
                lines.append(f"{key}: |")
                for line in value.splitlines():
                    lines.append(f"  {line}")
            else:
                lines.append(f"{key}: {_yaml_escape(value)}")

    keywords = meta.get("keywords", [])
    if isinstance(keywords, list) and keywords:
        lines.append("keywords:")
        for kw in keywords:
            lines.append(f"  - {_yaml_escape(str(kw))}")
    elif keywords:
        lines.append(f"keywords: {_yaml_escape(str(keywords))}")

    if image_path:
        lines.append(f"image: {_yaml_escape(image_path)}")

    if include_prompt:
        prompt = build_prompt(meta)
        lines.append("prompt_for_social_post: |")
        for line in prompt.splitlines():
            lines.append(f"  {line}")

    return "\n".join(lines).rstrip() + "\n"


def build_from_json(
    json_paths: Union[str, Sequence[str]],
    fmt: str = "md",
    image_path: Optional[str] = None,
    include_prompt: bool = True,
    *,
    # new optional behaviour for multi-sidecar handling
    copy_images: bool = False,
    assets_dir: Optional[str] = None,
    asset_url_prefix: Optional[str] = None,
    document_title: Optional[str] = None,
) -> str:
    """Build output from one or more image metadata JSON sidecars.

    Backward compatible: callers that pass a single json path (as a string)
    will receive the same output as before.

    New features:
    - Accept a sequence of JSON paths to produce a single combined document.
    - Optionally copy matching image files into an assets directory and update
      the image links to point to the copied files (useful when assembling post
      content for a website).

    Parameters
    - json_paths: single path or iterable of paths to JSON sidecar files.
    - fmt: 'md'|'markdown' or 'yaml'|'yml'
    - image_path: explicit image path to include. INVALID when multiple jsons
      are provided (raises ValueError).
    - copy_images: if True, copy discovered images into assets_dir (required).
    - assets_dir: target directory to copy images into (created if missing).
    - asset_url_prefix: optional URL/path prefix to use when referencing copied
      images in the output. If omitted, the literal filesystem path of the
      copied file is used.
    - document_title: when emitting multi-sidecar markdown, use this as the
      overarching H1. If omitted, uses the first sidecar title or
      'Untitled Collection'.
    """

    # Normalize inputs
    single_input = isinstance(json_paths, (str, bytes))
    if single_input:
        paths = [json_paths]  # type: ignore[list-item]
    else:
        paths = list(json_paths)  # type: ignore[arg-type]

    if len(paths) == 0:
        raise ValueError("No JSON paths provided")

    fmt = (fmt or "md").lower()

    # Validate incompatible combinations
    if len(paths) > 1 and image_path:
        raise ValueError("--image-path cannot be used with multiple JSON sidecars")
    if copy_images and not assets_dir:
        raise ValueError("copy_images=True requires an assets_dir to be provided")

    outputs: List[str] = []

    # Prepare assets dir if requested
    if copy_images and assets_dir:
        assets_path = Path(assets_dir)
        assets_path.mkdir(parents=True, exist_ok=True)
    else:
        assets_path = None

    for idx, jp in enumerate(paths):
        sidecar = Sidecar.load(jp)
        meta = sidecar.to_dict()

        # Determine image source
        this_image_path = None
        if image_path:
            this_image_path = image_path
        else:
            try:
                this_image_path = guess_image_path(jp)
            except Exception:
                this_image_path = None

        # Copy image if requested
        if assets_path and this_image_path:
            src = Path(this_image_path)
            if not src.exists():
                # If guessed path is relative to JSON's dir, resolve it
                alt = Path(jp).with_name(src.name)
                if alt.exists():
                    src = alt
            if not src.exists():
                # Can't find source image; raise clear error
                raise FileNotFoundError(f"Could not locate image for {jp}: {this_image_path}")

            dest = assets_path / src.name
            shutil.copy2(str(src), str(dest))

            # Compute link target
            if asset_url_prefix:
                # Ensure prefix ends with a slash
                prefix = asset_url_prefix
                if not prefix.endswith("/"):
                    prefix = prefix + "/"
                link_path = prefix + src.name
            else:
                # Use filesystem path to copied file
                link_path = str(dest)

            this_image_path = link_path

        # Render
        if fmt in ("md", "markdown"):
            # For multi-file output we produce a single document with one H1 at
            # the top and H2 per sidecar. For single-file callers preserve the
            # original H1 behaviour.
            if len(paths) == 1:
                outputs.append(_markdown_for_meta(meta, this_image_path, include_prompt, title_heading_level=1))
            else:
                # Each sidecar section uses H2 as the title (H1 is produced once)
                outputs.append(_markdown_for_meta(meta, this_image_path, include_prompt, title_heading_level=2))
        else:
            outputs.append(_yaml_for_meta(meta, this_image_path, include_prompt))

    # Combine outputs
    if fmt in ("md", "markdown"):
        if len(outputs) == 1:
            return outputs[0]
        # Multi-sidecar markdown: one H1 at the top
        if document_title:
            doc_title = document_title
        else:
            # Use the first sidecar title or sensible default
            first_meta = Sidecar.load(paths[0]).to_dict()
            doc_title = first_meta.get("title") or first_meta.get("original_title") or "Untitled Collection"

        combined: List[str] = []
        combined.append(f"# {doc_title}")
        combined.append("")
        for section in outputs:
            combined.append(section)
            combined.append("")
        return "\n".join(combined).rstrip() + "\n"

    # YAML: join documents with a YAML document separator for multi-sidecar
    if len(outputs) == 1:
        return outputs[0]
    return "\n---\n".join(outputs).rstrip() + "\n"


# CLI compatibility: add a main() so this package module can be used as a script
def main() -> None:
    """Command-line interface for building Markdown or YAML from JSON sidecars.

    New options support creating combined documents from multiple JSON files
    and copying image files into a target assets directory. Backwards
    compatible with older invocations that pass a single JSON path.
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Build Markdown or YAML from one or more image metadata JSON sidecars.",
    )
    parser.add_argument(
        "json_path",
        nargs="+",
        help="One or more paths to metadata JSON file(s).",
    )
    parser.add_argument(
        "--format",
        choices=["md", "markdown", "yaml", "yml"],
        default="md",
        help="Output format (default: md).",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--image-path",
        help=(
            "Explicit image path to include. If omitted, tries to guess from "
            "JSON filename. NOTE: cannot be used when passing multiple JSON files."
        ),
    )

    # Mutually exclusive control for prompt-related output
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--no-prompt",
        action="store_true",
        help="Do not include the social-post prompt in the output.",
    )
    prompt_group.add_argument(
        "--prompt-only",
        action="store_true",
        help="Output only the prompt for social post (plain text). No headings or other sections.",
    )

    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy discovered image files into --assets-dir. Requires --assets-dir.",
    )
    parser.add_argument(
        "--assets-dir",
        help="Target directory to copy image files into (created if missing).",
    )
    parser.add_argument(
        "--asset-url-prefix",
        help=(
            "Optional URL or path prefix to reference copied images (e.g. '/assets/images'). "
            "If omitted the filesystem path to the copied file will be used in the output."
        ),
    )
    parser.add_argument(
        "--title",
        help="Optional document title to use when combining multiple sidecars into one markdown document.",
    )

    args = parser.parse_args()

    # If prompt-only was requested, bypass the normal full document renderer and
    # print only the raw prompt(s). For multiple sidecars we'll print each
    # prompt separated by a blank line.
    try:
        if args.prompt_only:
            # args.json_path is a list (nargs='+')
            prompts: List[str] = []
            for jp in args.json_path:
                prompts.append(prompt_from_sidecar_path(jp))
            output = "\n\n".join(prompts).rstrip() + "\n"
        else:
            output = build_from_json(
                args.json_path if len(args.json_path) > 1 else args.json_path[0],
                fmt=args.format,
                image_path=args.image_path,
                include_prompt=not args.no_prompt,
                copy_images=args.copy_images,
                assets_dir=args.assets_dir,
                asset_url_prefix=args.asset_url_prefix,
                document_title=args.title,
            )
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(2)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
    else:
        sys.stdout.write(output)


if __name__ == "__main__":
    main()


# Manual test notes:
# If no test framework is present, verify manually with:
# python -m src.image_description.post.post_builder \
#   path/to/sidecar.json --prompt-only
# This should print only the raw prompt text (no markdown headings, fences, etc.).
# For the standard behavior try:
# python -m src.image_description.post.post_builder path/to/sidecar.json
# which should produce the full markdown including the Prompt for social post
# section wrapped in a ```text fence.
