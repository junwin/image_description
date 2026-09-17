"""Provider-agnostic vision calls via galet.

Replaces direct native-API (OpenAI chat.completions) calls in the image and
notes pipelines. galet (repos/galet) normalizes OpenAI, Gemini, DeepSeek,
Mistral, and Ollama behind one RouterApi interface.

Credentials: galet resolves keys from the provider env var (OPENAI_API_KEY,
GEMINI_API_KEY, ...) or from a credential file in the directory named by
``credential_path`` / ``GALET_CREDENTIAL_PATH`` (e.g. oaicred.json with key
``openai_api_key``, gemini_cred.json with key ``gemini_api_key``).

When no path is given anywhere, we fall back to ``~/credential`` if it exists
(this mirrors the previous native-client behaviour).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Optional

from galet.router_api import RouterApi
from galet.settings import Settings


def _resolve_credential_path(credential_path: Optional[str]) -> Optional[str]:
    """Resolve the galet credential directory.

    Order: explicit arg -> GALET_CREDENTIAL_PATH env -> ~/credential (if present).
    """
    if credential_path:
        return credential_path
    env_path = os.environ.get("GALET_CREDENTIAL_PATH")
    if env_path:
        return env_path
    default = os.path.expanduser("~/credential")
    if os.path.isdir(default):
        return default
    return None


def create_vision_response(
    prompt: str,
    image_b64: str,
    *,
    model: str,
    provider: Optional[str] = None,
    credential_path: Optional[str] = None,
    temperature: float = 0.2,
) -> str:
    """Send a prompt + inline image to a vision model via galet.

    Args:
        prompt: Text prompt to send alongside the image.
        image_b64: Base64-encoded JPEG image data (without the data: prefix).
        model: Model name (e.g. ``gpt-4o``, ``gemini-3.6-flash``).
        provider: Explicit provider name (openai/gemini/...). When None the
            model-name prefix is used, falling back to openai.
        credential_path: Directory holding galet credential files. When None,
            galet uses GALET_CREDENTIAL_PATH, then ~/credential, then env vars.
        temperature: Sampling temperature.

    Returns:
        Raw output text from the model.
    """
    settings = Settings(credential_path=_resolve_credential_path(credential_path))
    router = RouterApi(settings=settings)

    response = router.create_response(
        model=model,
        provider=provider,
        temperature=temperature,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "source": {"data": image_b64, "mime_type": "image/jpeg"},
                    },
                ],
            }
        ],
    )

    return response.output_text or ""


# Characters allowed after a backslash in a JSON string escape.
_VALID_JSON_ESCAPES = set('"\\/bfnrtu')


def _repair_invalid_escapes(text: str) -> str:
    """Drop backslashes that precede a character which is not a valid JSON escape.

    Vision models occasionally emit escapes such as ``\\!`` or ``\\ `` instead of
    a plain character. json.loads rejects these; this repair keeps the character
    and removes the offending backslash. Only ever called after a normal parse
    has already failed.
    """
    out = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\" and i + 1 < n and text[i + 1] not in _VALID_JSON_ESCAPES:
            out.append(text[i + 1])
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def parse_json_response(text: str) -> Dict[str, Any]:
    """Extract a JSON object from model output text.

    Handles optional ```json code fences and stray prose around the object.
    Falls back to repairing common invalid backslash escapes before giving up.
    Prints the offending text to stderr and raises RuntimeError on failure.
    """
    content = text.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = content[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
            # Repair common LLM mistake: invalid backslash escapes (e.g. \!, \ )
            try:
                return json.loads(_repair_invalid_escapes(candidate))
            except json.JSONDecodeError:
                pass
        print(f"Model response was not valid JSON: {content}", file=sys.stderr)
        raise RuntimeError("Model response was not valid JSON")
