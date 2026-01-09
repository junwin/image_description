import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Sidecar:
    """Represents the JSON sidecar contract.

    Keep this permissive and forward-compatible:
    - unknown fields are preserved in `extra`
    - missing fields default to empty values
    """

    original_title: str = ""
    original_description: str = ""
    title: str = ""
    visually_challenged_description: str = ""
    enhanced_description: str = ""
    keywords: List[str] = field(default_factory=list)
    hashtags: str = ""

    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Sidecar":
        known_keys = {
            "original_title",
            "original_description",
            "title",
            "visually_challenged_description",
            "enhanced_description",
            "keywords",
            "hashtags",
        }

        keywords = data.get("keywords", [])
        if not isinstance(keywords, list):
            keywords = [str(keywords)]
        keywords = [str(k).strip() for k in keywords if str(k).strip()]

        extra = {k: v for k, v in data.items() if k not in known_keys}

        return cls(
            original_title=str(data.get("original_title", "") or ""),
            original_description=str(data.get("original_description", "") or ""),
            title=str(data.get("title", "") or ""),
            visually_challenged_description=str(
                data.get("visually_challenged_description", "") or ""
            ),
            enhanced_description=str(data.get("enhanced_description", "") or ""),
            keywords=keywords,
            hashtags=str(data.get("hashtags", "") or ""),
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "original_title": self.original_title,
            "original_description": self.original_description,
            "title": self.title,
            "visually_challenged_description": self.visually_challenged_description,
            "enhanced_description": self.enhanced_description,
            "keywords": list(self.keywords),
            "hashtags": self.hashtags,
        }
        data.update(self.extra)
        return data

    @classmethod
    def load(cls, json_path: str) -> "Sidecar":
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Sidecar JSON must be an object: {json_path}")
        return cls.from_dict(data)

    def save(self, json_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(json_path)) or ".", exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)
