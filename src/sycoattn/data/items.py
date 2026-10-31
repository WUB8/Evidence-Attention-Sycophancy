"""A QA item: an evidence passage, a yes/no question, and the evidence-supported answer."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Item:
    id: str
    evidence: str
    question: str
    answer: bool                      # True = "yes"
    source: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def answer_str(self) -> str:
        return "yes" if self.answer else "no"


def save_items(items: list[Item], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for it in items:
            f.write(json.dumps(asdict(it)) + "\n")


def read_items(path: str | Path) -> list[Item]:
    with Path(path).open() as f:
        return [Item(**json.loads(line)) for line in f if line.strip()]


def load_items(source: str, max_items: int | None = None, seed: int = 0, **kwargs) -> list[Item]:
    if source == "multiwoz":
        from .multiwoz import build_multiwoz_items
        return build_multiwoz_items(max_items=max_items, seed=seed, **kwargs)
    if source == "boolq":
        from .boolq import build_boolq_items
        return build_boolq_items(max_items=max_items, seed=seed)
    raise ValueError(f"unknown data source: {source}")
