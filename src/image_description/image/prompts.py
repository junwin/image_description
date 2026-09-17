from typing import Dict, List


PROMPT_PRESETS: Dict[str, str] = {
    "orwell_basic": (
        "You are a careful, precise writer. "
        "Describe the image in clear, concrete language. "
        "Avoid jargon and unnecessary words. "
        "Return JSON with the following keys: "
        "'image_description' (one paragraph), "
        "'enhanced_description' (one paragraph), "
        "'hashtags' (a single string of 5-8 hashtags), "
        "'social_caption' (1-2 sentences), "
        "and 'keywords' (a list of 8-15 short keywords)."
    ),
    "orwell_ways_of_seeing": (
        "Act as a careful, precise writer. "
        "Return two kinds of text: (A) strict alt text and (B) a short caption for social media. "
        "\n\n"
        "A) image_description (IMAGE DESCRIPTION / ALT TEXT)\n"
        "- Describe only what is literally visible in the image.\n"
        "- Do not guess identity, relationships, emotions, intent, time, place, or story.\n"
        "- If something is uncertain, say so (e.g., 'a person', 'possibly a sign').\n"
        "- Mention key objects, setting, actions, composition (framing, foreground/background), and any readable text.\n"
        "- Keep it concise: 1-3 sentences, ~40-80 words.\n"
        "\n"
        "B) social_caption (SOCIAL POST CAPTION)\n"
        "- 1-2 sentences.\n"
        "- May be lightly reflective, but must stay grounded in what is visible.\n"
        "- Follow Orwell's rules: short words, cut unnecessary words, avoid jargon.\n"
        "- Understated tone; no boastfulness; no salesy language.\n"
        "\n\n"
        "Also provide hashtags and keywords:\n"
        "- hashtags: a single string of 5-8 relevant hashtags (include the #).\n"
        "- keywords: a list of 8-15 short keywords.\n\n"
        "Return JSON with the following keys: "
        "'image_description', 'enhanced_description', 'social_caption', 'hashtags', and 'keywords'. "
        "For enhanced_description: one short paragraph that expands the literal description with a little more detail, "
        "but still avoid guessing or inventing context."
    ),
}


def build_prompt(
    preset_name: str,
    title: str,
    description: str,
    existing_keywords: List[str],
) -> str:
    base_prompt = PROMPT_PRESETS.get(preset_name, PROMPT_PRESETS["orwell_ways_of_seeing"])
    kw_str = ", ".join(existing_keywords) if existing_keywords else "(none)"

    return (
        f"{base_prompt}\n\n"
        f"Here is some metadata I already have \u2013 treat it as hints, but do not copy claims that are not visible.\n\n"
        f"Title: {title}\n"
        f"Existing description: {description}\n"
        f"Existing keywords: {kw_str}\n\n"
        f"Please respond ONLY with a single JSON object matching the requested keys."
    )
