from typing import Any, Dict, List, Optional

from ..paths import guess_image_path
from ..sidecar import Sidecar


PROMPT_TEMPLATE = """Act as a thoughtful artist and writer. Prepare a Mastodon, tumblr and bsky post for a new photograph I've taken.
please include suggested hashtags (5 or 6) and text for the visually challenged.

Consider John Berger's separation of a) what the image is  b) what is it trying to say  - I would like to swing the balance to what the image is trying to say.

Here is some metadata I already have  - this typically deals with what the image is.

Title: {title}
Original description: {original_description}
Enhanced description: {enhanced_description}
Visually challenged description: {visually_challenged_description}
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
    title = meta.get("title") or meta.get("original_title") or ""
    original_description = meta.get("original_description", "")
    enhanced_description = meta.get("enhanced_description", "")
    visually_challenged_description = meta.get("visually_challenged_description", "")
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
        visually_challenged_description=visually_challenged_description,
        keywords=keywords_str,
        hashtags=hashtags,
    )


def to_markdown(
    meta: Dict[str, Any],
    image_path: Optional[str],
    include_prompt: bool = True,
) -> str:
    title = meta.get("title") or meta.get("original_title") or "Untitled"
    original_title = meta.get("original_title", "")
    original_description = meta.get("original_description", "")
    enhanced_description = meta.get("enhanced_description", "")
    visually_challenged_description = meta.get("visually_challenged_description", "")
    keywords = meta.get("keywords", [])
    hashtags = meta.get("hashtags", "")

    lines: List[str] = []

    lines.append(f"# {title}")
    lines.append("")

    if image_path:
        lines.append(f"![{title}]({image_path})")
        lines.append("")

    lines.append("## Original notes")
    if original_title:
        lines.append(f"**Original title:** {original_title}")
        lines.append("")
    if original_description:
        lines.append(original_description)
        lines.append("")

    if enhanced_description:
        lines.append("## Enhanced description")
        lines.append(enhanced_description)
        lines.append("")

    if visually_challenged_description:
        lines.append("## Description for the visually challenged")
        lines.append(visually_challenged_description)
        lines.append("")

    if keywords:
        if isinstance(keywords, list):
            kw_str = ", ".join(keywords)
        else:
            kw_str = str(keywords)
        lines.append("## Keywords")
        lines.append(kw_str)
        lines.append("")

    if hashtags:
        lines.append("## Hashtags")
        lines.append(hashtags)
        lines.append("")

    if include_prompt:
        lines.append("## Prompt for social post")
        lines.append("```text")
        lines.append(build_prompt(meta))
        lines.append("```")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def to_yaml(
    meta: Dict[str, Any],
    image_path: Optional[str],
    include_prompt: bool = True,
) -> str:
    """Minimal YAML emitter to avoid extra dependencies."""

    def yaml_escape(s: str) -> str:
        if any(
            c in s
            for c in [":", "-", "#", "{", "}", "[", "]", ",", "&", "*", "?", "|", ">", "%", "@", "`", "\"", "'"]
        ):
            return "\"" + s.replace("\"", "\\\"") + "\""
        if "\n" in s:
            return "|-\n  " + s.replace("\n", "\n  ")
        return s

    lines: List[str] = []

    title = meta.get("title") or meta.get("original_title") or "Untitled"
    lines.append(f"title: {yaml_escape(str(title))}")

    for key in [
        "original_title",
        "original_description",
        "visually_challenged_description",
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
                lines.append(f"{key}: {yaml_escape(value)}")

    keywords = meta.get("keywords", [])
    if isinstance(keywords, list) and keywords:
        lines.append("keywords:")
        for kw in keywords:
            lines.append(f"  - {yaml_escape(str(kw))}")
    elif keywords:
        lines.append(f"keywords: {yaml_escape(str(keywords))}")

    if image_path:
        lines.append(f"image: {yaml_escape(image_path)}")

    if include_prompt:
        prompt = build_prompt(meta)
        lines.append("prompt_for_social_post: |")
        for line in prompt.splitlines():
            lines.append(f"  {line}")

    return "\n".join(lines).rstrip() + "\n"


def build_from_json(
    json_path: str,
    fmt: str,
    image_path: Optional[str],
    include_prompt: bool,
) -> str:
    sidecar = Sidecar.load(json_path)
    meta = sidecar.to_dict()

    if not image_path:
        image_path = guess_image_path(json_path)

    fmt = fmt.lower()
    if fmt in ("md", "markdown"):
        return to_markdown(meta, image_path, include_prompt=include_prompt)
    return to_yaml(meta, image_path, include_prompt=include_prompt)


# CLI compatibility: add a main() so this package module can be used as a script
def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Build Markdown or YAML from image metadata JSON.",
    )
    parser.add_argument("json_path", help="Path to the metadata JSON file.")
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
            "JSON filename."
        ),
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Do not include the social-post prompt in the output.",
    )

    args = parser.parse_args()

    output = build_from_json(
        args.json_path, args.format, args.image_path, include_prompt=not args.no_prompt
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
    else:
        # Print to stdout
        sys.stdout.write(output)


if __name__ == "__main__":
    main()
