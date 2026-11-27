"""Conventional Marie-Parie batch paths (no history migration or implicit writes).

Lucy attachments exports directly into workspace/incoming/batch-XXXXXXXXXXXX.
Each exported batch name remains the key for its prepared media and Reel.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

BATCH_PATTERN = re.compile(r"batch-[0-9a-f]{12}\Z")


@dataclass(frozen=True)
class MarieBatchPaths:
    workspace: Path
    batch: str

    @property
    def incoming(self) -> Path:
        return self.workspace / "incoming" / self.batch

    @property
    def web(self) -> Path:
        return self.workspace / "prepared" / "web" / self.batch

    @property
    def reel_images(self) -> Path:
        return self.workspace / "prepared" / "reel" / self.batch

    @property
    def reel_video(self) -> Path:
        return self.workspace / "output" / "reels" / f"{self.batch}.mp4"

    def as_dict(self) -> dict[str, str]:
        return {
            "workspace": str(self.workspace),
            "batch": self.batch,
            "incoming": str(self.incoming),
            "prepared_web": str(self.web),
            "prepared_reel": str(self.reel_images),
            "reel_video": str(self.reel_video),
        }


def batch_paths(batch: str, workspace: str | Path | None = None) -> MarieBatchPaths:
    if not isinstance(batch, str) or not BATCH_PATTERN.fullmatch(batch):
        raise ValueError("Batch must be the 'batch-' plus 12 lowercase hex digits returned by Lucy attachments export")
    root = (Path(workspace).expanduser() if workspace is not None
            else Path.home() / "Pictures" / "marie-parie")
    root = root.resolve()
    return MarieBatchPaths(workspace=root, batch=batch)
