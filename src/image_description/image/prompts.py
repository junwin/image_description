from typing import Dict, List


PROMPT_PRESETS: Dict[str, str] = {
    "orwell_basic": (
        "You are a careful, precise writer. "
        "Describe the image in clear, concrete language. "
        "Avoid jargon and unnecessary words. "
        "Return JSON with the following keys: "
        "'visually_challenged_description' (one paragraph), "
        "'enhanced_description' (one paragraph), and "
        "'keywords' (a list of 8-15 short keywords)."
    ),
    "orwell_ways_of_seeing": (
        "Act as a thoughtful artist and writer. "
        "Consider John Berger's separation of (a) what the image is and (b) what it is trying to say. "
        "Lean toward what the image is trying to say, but stay grounded in what is visible. "
        "Please adhere strictly to the following style guidelines: "
        "1. Follow George Orwell's rules: use short words, cut unnecessary words, and avoid jargon. "
        "2. Use a minimalist and evocative style. Be precise, not flowery. "
        "3. Adopt a reflective, understated tone. Avoid any boastfulness. "
        "4. Use a two-sentence structure if possible: first a direct description, then a reflective observation. "
        "Keep the final output concise. "
        "Return JSON with the following keys: "
        "'visually_challenged_description' (one paragraph), "
        "'enhanced_description' (one paragraph), and "
        "'keywords' (a list of 8-15 short keywords)."
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
        f"Here is some metadata I already have – this typically deals with what the image is.\n\n"
        f"Title: {title}\n"
        f"Existing description: {description}\n"
        f"Existing keywords: {kw_str}\n\n"
        f"Please respond ONLY with a single JSON object matching the requested keys."
    )
